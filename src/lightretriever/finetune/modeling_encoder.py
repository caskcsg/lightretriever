#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Full Implementation of Single-Vector Dual-Tower Dense Retriever.

@Time    :   2024/08/29
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import copy
import json
import yaml
from typing import Optional
from itertools import chain
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers import (
    PreTrainedModel,
    AutoModel,
    BatchEncoding,
    HfArgumentParser,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from peft.utils import CONFIG_NAME as PEFT_CONFIG_NAME

from .dense_pooling import pooling
from .dense_projector import DenseLinearProjector
from ..utils.data_utils import load_tokenizer, resize_emb
from ..utils.monkey_patch import apply_bidirectional_attention
from ..utils.nested_input import apply_seqlen_cumulate
from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    logs: Optional[dict[str, any]] = None


class EncoderModel(nn.Module):
    """ Full featured model for retriever training & encoding. """
    TRANSFORMER_CLS = AutoModel
    is_gradient_checkpointing = False

    def __init__(
            self,
            lm_q: PreTrainedModel | PeftModel,
            lm_p: PreTrainedModel | PeftModel,
            model_args: ModelArguments,
            train_args: Optional[TrainingArguments] = None,
            data_args: Optional[DataArguments] = None,
            den_pooler_q: Optional[DenseLinearProjector] = None,
            den_pooler_p: Optional[DenseLinearProjector] = None,
        ):
        super(EncoderModel, self).__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.den_pooler_q = den_pooler_q
        self.den_pooler_p = den_pooler_p

        if self.model_args.cumulative_seq:
            apply_seqlen_cumulate(self.lm_p)
            if self.model_args.untie_encoder:
                apply_seqlen_cumulate(self.lm_q)
        
        if self.model_args.enable_bidirectional_attention:
            apply_bidirectional_attention(self.lm_p)
            if self.model_args.untie_encoder:
                apply_bidirectional_attention(self.lm_q)
        
        if self.model_args.liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            # Patch the model with liger kernels. Use the default kernel configurations.
            if isinstance(self.lm_p, PreTrainedModel) and isinstance(self.lm_q, PreTrainedModel):
                _apply_liger_kernel_to_instance(model=self.lm_p)
                if self.model_args.untie_encoder:
                    _apply_liger_kernel_to_instance(model=self.lm_q)
            elif isinstance(self.lm_p, PeftModel) and isinstance(self.lm_q, PeftModel):
                _apply_liger_kernel_to_instance(model=self.lm_p.get_base_model())
                if self.model_args.untie_encoder:
                    _apply_liger_kernel_to_instance(model=self.lm_q.get_base_model())
            else:
                logger.warning(
                    "The model is not an instance of PreTrainedModel/PeftModel. No liger kernels will be applied."
                )

        # Training related
        # Need not to execute this block when inferencing
        if train_args is not None:
            self.config = lm_p.config       # DS initialization will use model.config
            
            try:
                from flash_attn.losses.cross_entropy import CrossEntropyLoss
                self.cross_entropy = CrossEntropyLoss(reduction=self.train_args.loss_reduction, inplace_backward=True)
                        
            except ImportError:
                logger.info(
                    "Optimized flash-attention CrossEntropyLoss not found (run `pip install git+https://github.com/Dao-AILab/flash-attention.git#egg=xentropy_cuda_lib&subdirectory=csrc/xentropy`)"
                )
                self.cross_entropy = nn.CrossEntropyLoss(reduction=self.train_args.loss_reduction)
            
            if dist.is_initialized():
                self.process_rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            
            # Compatiable with FSDP Auto Wrap
            _no_split_modules_lm_p: list[str] = getattr(lm_p, "_no_split_modules", []) or []
            _no_split_modules_lm_q: list[str] = getattr(lm_q, "_no_split_modules", []) or []
            self._no_split_modules = list(set(chain(_no_split_modules_lm_p, _no_split_modules_lm_q)))

    def forward(
            self, 
            query: dict[str, Tensor] = None, 
            passage: dict[str, Tensor] = None,
            ce_scores: Tensor = None,
            only_hn: bool = False,
            **kwargs,
        ):
        """
        Model forward.

        Args:
            query (dict[str, Tensor] | BatchEncoding): Inputs with shape [batch_size, query_seq_len].
            passage (dict[str, Tensor] | BatchEncoding): Inputs with shape [train_n_passages * batch_size, passage_seq_len].
            ce_scores (Tensor): Re-ranker scores for q-p pairs, this is used for distilation. Shape [batch_size, train_n_passages].
            only_hn (Tensor[bool]): Whether to only use hard negatives, and disable in-batch / cross-batch negatives. Shape [batch_size].
        
        Note:
            Dynamic outputs depands on query / passage inputs:
            - `Training`: query, passage are all not None. Model.training == True.
            - `Evaluating`: query, passage are all not None. Model.training == False.
            - `Inferencing`: query is None / passage is None.
        """
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # For inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps,
                loss=0.,
                scores=0.,
                logs=dict()
            )

        # For training
        if self.training:
            return self.compute_loss(q_reps=q_reps, p_reps=p_reps, ce_scores=ce_scores, only_hn=only_hn, **kwargs)
        # For eval
        else:
            q_reps_eval = q_reps.unsqueeze(1)      # B 1 D
            p_reps_eval = p_reps.view(q_reps.shape[0], -1, q_reps.shape[-1]) # B N D
            scores = self.compute_similarity(q_reps_eval, p_reps_eval).squeeze(1)      # B N
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps,
                loss=0.,
                scores=scores,
                logs=dict()
            )
    
    def compute_loss(
            self, 
            q_reps: Tensor, 
            p_reps: Tensor, 
            ce_scores: Tensor = None,
            only_hn: Tensor = None,
            **kwargs,
        ):
        """ Compute Contrastive Loss and/or Distilation Loss
            GradCache separates the forward & contrastive loss computation, thus
            this function is separated from model.forward. And this function is 
            shared with model.forward & GradCache.compute_loss

            Args:
                q_reps (Tensor): Query representations of current batch. Shape [batch_size, rep_dims].
                p_reps (Tensor): Passage representations of current batch. Shape [batch_size, rep_dims].
                ce_scores (Tensor): Re-ranker scores for q-p pairs, this is used for distilation. Shape [batch_size, train_n_passages].
                only_hn (Tensor[bool]): Whether to only use hard negatives, and disable in-batch / cross-batch negatives. Shape [batch_size].
            
            Notes:
                Contrastive Learning relies on negatives sampling, we can use multiple negatives to optimize the encoder.
                1) Only hard negatives: Explicitly passing `only_hn[i]=True`
                2) Hard + in-batch negatives: Passing `only_hn[:]=False` and setting `negatives_x_device=False`
                3) Hard + in-batch + cross-batch negatives: Passing `only_hn[:]=False` and setting `negatives_x_device=True`
        """
        q_bs, p_bs = q_reps.shape[0], p_reps.shape[0]
        n_psg = p_bs // q_bs
        temperature = kwargs.get("temperature", None) or self.train_args.temperature    # Support override

        scores: Tensor = None
        student_scores: Optional[Tensor] = None
        loss: Tensor = 0.
        logs: dict[str, float] = dict()   # Customized Logs

        if self.train_args.clloss_coef > 0:
            # Gather embeddings
            if self.train_args.negatives_x_device:
                # Gather for Cross Batch Negatives. Loss scale issue exists because of
                # mean reduction of CrossEntropy over batch size dimension (both query
                # and passage sides) and unsupport of diffentiable all gather.
                q_reps_full = self._dist_gather_tensor(q_reps)
                p_reps_full = self._dist_gather_tensor(p_reps)
            else:
                q_reps_full = q_reps
                p_reps_full = p_reps

            # Similarity computation
            scores = self.compute_similarity(q_reps_full, p_reps_full) / temperature

            # Mask in/cross-batch negatives, only use hard negatives
            if only_hn is not None:
                if self.train_args.negatives_x_device:
                    only_hn = self._dist_gather_tensor(only_hn)
                assert only_hn.dim() == 1
                
                if torch.any(only_hn):
                    scores_mask = torch.zeros_like(scores, dtype=torch.bool)
                    # For q_reps_full[idx], only preserve the region [idx*n_psg: (idx+1)*n_psg], mask all other region
                    for idx, only_hn_flag in enumerate(only_hn):
                        if only_hn_flag:
                            scores_mask[idx][:idx*n_psg] = True
                            scores_mask[idx][(idx+1)*n_psg:] = True
                    scores.masked_fill_(scores_mask, torch.finfo(scores.dtype).min)      # mask with -inf

            # Labels of Cross Entropy
            # [0, 1, 2, ...] * train_n_passages
            target = torch.arange(scores.shape[0], device=scores.device, dtype=torch.long) * n_psg
            
            # Cross Entropy Loss
            clloss: Tensor = self.cross_entropy(scores, target) * self.train_args.clloss_coef
            loss += clloss
            logs['clloss'] = clloss.item() if self.train_args.loss_reduction == 'mean' else clloss.mean().item()
        
        if self.train_args.distillation and ce_scores is not None:
            q_reps_student = q_reps.unsqueeze(1)      # B 1 D
            p_reps_student = p_reps.view(q_bs, n_psg, q_reps.shape[-1]) # B N D

            student_scores = self.compute_similarity(q_reps_student, p_reps_student).squeeze(1) / self.train_args.distill_temperature        # B 1 N -> B N
            teacher_scores = ce_scores.view(student_scores.shape[0], student_scores.shape[1]) / self.train_args.distill_temperature
            
            # As a normal practice, Temperature-scaled KL Loss requires `*T^2` to calculate the proper scale
            # of the gradients. However, we aren't doing it here to leave this tuning to the users.
            klloss = self.klloss(student_scores, teacher_scores) * self.train_args.distill_coef
            loss += klloss
            logs['klloss'] = klloss.item() if self.train_args.loss_reduction == 'mean' else klloss.mean().item()
        
        # Record domain loss if using homogenous batching
        if domain_name_list := kwargs.get('domain_name', None):
            assert isinstance(domain_name_list, list)
            if len(set(domain_name_list)) == 1:  # homogenous batching
                domain_name = domain_name_list[0]
                logs[f'channel/{domain_name}'] = loss.item() if self.train_args.loss_reduction == 'mean' else loss.mean().item()
        
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            logs=logs,
        )
    
    def _apply_gradient_checkpointing(self, model: PreTrainedModel, gradient_checkpointing_kwargs: dict = None):
        ds_config = gradient_checkpointing_kwargs.pop("ds_config", None)
        
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        
        # Replace deepspeed activate checkpoint if available
        if ds_config is not None and "activation_checkpointing" in ds_config:
            try:
                import deepspeed
                deepspeed_is_initialized = deepspeed.comm.comm.is_initialized()
            except:
                deepspeed_is_initialized = False
            if deepspeed_is_initialized:
                logger.info(f"Setting DeepSpeed Activation Checkpointing..")
                deepspeed.checkpointing.configure(mpu_=None, deepspeed_config=ds_config)
                model._set_gradient_checkpointing(gradient_checkpointing_func=deepspeed.checkpointing.checkpoint)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict = None):
        self.is_gradient_checkpointing = True
        self._apply_gradient_checkpointing(self.lm_q.base_model, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if self.model_args.untie_encoder:
            self._apply_gradient_checkpointing(self.lm_p.base_model, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def encode_passage(self, psg: Optional[BatchEncoding], normalize: Optional[bool] = None, **kwargs):
        """
        Encoding passage.

        Args:
            psg (Optional[BatchEncoding]): Inputs with input_ids, attention_mask (optional). 
                                           Shape [batch_size, seq_len].
            normalize (Optional[bool]): Overriding whether to use l2 normalization for embedding.
                                        - `None`: Listen to self.model_args.normalize.
                                        - `True/False`: Activate normalization / Deactivate normalization.
        
        Returns:
            Embedding: Shape [batch_size, rep_dim].
        """
        if psg is None:
            return None
        
        # Format Inputs
        forward_kwargs = {
            'input_ids': psg['input_ids'],
            'attention_mask': psg['attention_mask'],
        }
        for optional_input_field in ['token_type_ids', 'position_ids']:
            if optional_input_field in psg:
                forward_kwargs[optional_input_field] = psg[optional_input_field]
        forward_kwargs['return_dict'] = True
        forward_kwargs['use_cache'] = False    # Do not return `past_key_values`
        forward_kwargs['output_hidden_states'] = True if self.model_args.pooling_strategy_psg in ["avg_first_last", "avg_top2"] else False

        # Forward
        psg_out: BaseModelOutput = self.lm_p(**forward_kwargs)

        # Rep Pooling
        p_reps = pooling(
            last_hidden=psg_out.last_hidden_state,
            hidden_states=psg_out.hidden_states,
            attention_mask=psg['attention_mask'],
            pooling_strategy=self.model_args.pooling_strategy_psg,
        )
        if self.den_pooler_p is not None:
            p_reps = self.den_pooler_p(p_reps)  # D * d
        if self.model_args.dense_shrink_dim:
            p_reps = p_reps[..., :self.model_args.dense_shrink_dim]

        # Below conditions activates the normalization
        # 1) Functional input `normalize==True`
        # 2) Functional input `normalize is None`, looking for `self.model_args.normalize`
        if normalize or (normalize is None and self.model_args.normalize):
            p_reps = F.normalize(p_reps, p=2, dim=1)
        return p_reps

    def encode_query(self, qry: Optional[BatchEncoding], normalize: Optional[bool] = None, **kwargs):
        """
        Encoding query.

        Args:
            qry (Optional[BatchEncoding]): Inputs with input_ids, attention_mask (optional). 
                                           Shape [batch_size, seq_len].
            normalize (Optional[bool]): Overriding whether to use l2 normalization for embedding.
                                        - `None`: Listen to self.model_args.normalize.
                                        - `True/False`: Activate normalization / Deactivate normalization.
        
        Returns:
            Embedding: Shape [batch_size, rep_dim].
        """
        if qry is None:
            return None

        # Format Inputs
        forward_kwargs = {
            'input_ids': qry['input_ids'],
            'attention_mask': qry['attention_mask'],
        }
        for optional_input_field in ['token_type_ids', 'position_ids']:
            if optional_input_field in qry:
                forward_kwargs[optional_input_field] = qry[optional_input_field]
        forward_kwargs['return_dict'] = True
        forward_kwargs['use_cache'] = False    # Do not return `past_key_values`
        forward_kwargs['output_hidden_states'] = True if self.model_args.pooling_strategy_qry in ["avg_first_last", "avg_top2"] else False
        
        # Forward
        qry_out: BaseModelOutput = self.lm_q(**forward_kwargs)

        # Rep Pooling
        q_reps = pooling(
            last_hidden=qry_out.last_hidden_state,
            hidden_states=qry_out.hidden_states,
            attention_mask=qry['attention_mask'],
            pooling_strategy=self.model_args.pooling_strategy_qry,
        )
        if self.den_pooler_q is not None:
            q_reps = self.den_pooler_q(q_reps)  # D * d
        if self.model_args.dense_shrink_dim:
            q_reps = q_reps[..., :self.model_args.dense_shrink_dim]
        if normalize or (normalize is None and self.model_args.normalize):
            q_reps = F.normalize(q_reps, p=2, dim=1)
        return q_reps

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        Input:  1) q_reps: [batch_size, hidden_dim]
                    p_reps: [batch_size * train_n_passages, hidden_dim]
                    return:  [batch_size, train_n_passages]
                2) q_reps: [batch_size, 1, hidden_dim]
                    p_reps: [batch_size, train_n_passages, hidden_dim]
                    return:  [batch_size, 1, train_n_passages]
        Return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def _dist_gather_tensor(self, t: Optional[Tensor]):
        """ All gather a Tensor with the same shape """
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def klloss(self, student_scores: Tensor, teacher_scores: Tensor) -> Tensor:
        """ Calculate klloss for distilation from teacher to student """
        klloss = F.kl_div(F.log_softmax(student_scores, dim=-1), 
                        F.softmax(teacher_scores, dim=-1), 
                        reduction='batchmean')
        return klloss  # choose 'sum' or 'mean' depending on loss scale
    
    @staticmethod
    def load_tokenizer(
            model_name_or_path: str, 
            model_args: ModelArguments,
        ):
        """ Helper function to load tokenizer for EncoderModel """
        tokenizer = load_tokenizer(
            model_name_or_path, 
            cache_dir=model_args.cache_dir, 
            use_fast=model_args.use_fast_tokenizer,
            edit_tokenizer_normalizers=model_args.edit_tokenizer_normalizers,
            lowercase=model_args.lowercase,
            edit_tokenizer_post_processor=model_args.edit_tokenizer_post_processor,
            add_bos_num=model_args.add_bos_num,
            add_eos_num=model_args.add_eos_num,
            add_pooling_token_num=model_args.add_pooling_token_num,
            add_pad_token=model_args.add_pad_token,
            pad_token=model_args.pad_token,
            add_sep_token=model_args.add_sep_token,
            sep_token=model_args.sep_token,
        )
        return tokenizer
    
    @classmethod
    def _load_model(
            cls,
            model_name_or_path: str,
            model_args: ModelArguments,
            merge_peft_weights: bool = False,
            **hf_kwargs,
        ) -> PreTrainedModel:
        """ Helper functions to load model """

        # Load tokenizer for resizing embedding layer if necessary
        tokenizer = cls.load_tokenizer(model_name_or_path, model_args=model_args)
        
        # Load LoRA Model
        if os.path.exists(os.path.join(model_name_or_path, PEFT_CONFIG_NAME)):
            logger.info(f"Peft config is found at {model_name_or_path}.")
            # Load Base HF Model & Peft Adapters
            config = LoraConfig.from_pretrained(model_name_or_path)
            base_model = cls.TRANSFORMER_CLS.from_pretrained(config.base_model_name_or_path, **hf_kwargs)
            # Resize base_model embedding layer if necessary
            resize_emb(base_model, tokenizer, pad_to_multiple_of=model_args.pad_to_multiple_of)
            hf_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=config, is_trainable=True, torch_device=hf_kwargs.get("device_map", None))
            if merge_peft_weights:
                hf_model = hf_model.merge_and_unload()  # Merge to single HF Model
        
        # Load HF Model
        else:
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            # Resize hf_model embedding layer if necessary
            resize_emb(hf_model, tokenizer, pad_to_multiple_of=model_args.pad_to_multiple_of)
        
        return hf_model
    
    @classmethod
    def _load_pooler(
            cls,
            model_name_or_path: str,
        ) -> Optional[nn.Module]:
        """ Helper functions to Load PyTorch Pooler if exists """
        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = DenseLinearProjector(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None
        
        return pooler
    
    @staticmethod
    def _build_pooler(
            projection_in_dim: int, 
            projection_out_dim: int, 
            model_name_or_path: str = ""
        ):
        """ Init a new Pooler """
        pooler = DenseLinearProjector(
            input_dim=projection_in_dim,
            output_dim=projection_out_dim,
        )
        pooler.load(model_name_or_path)
        return pooler
    
    @staticmethod
    def _build_lora_model(
            base_model: PreTrainedModel,
            base_model_name_or_path: str,
            train_args: TrainingArguments,
        ):
        """ Helper functions to init a LoRA for `base_model` """
        peft_config = LoraConfig(
            base_model_name_or_path=base_model_name_or_path,
            task_type=TaskType.FEATURE_EXTRACTION,
            r=train_args.lora_r,
            lora_alpha=train_args.lora_alpha,
            lora_dropout=train_args.lora_dropout,
            target_modules='all-linear',
            inference_mode=False
        )
        peft_model = get_peft_model(base_model, peft_config)
        return peft_model

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            data_args: DataArguments,
            **hf_kwargs,
        ):
        """
        Building a model from local checkpoint / online for training.
        """
        # Load model and pooler
        # Load chechpoint from local dir
        if os.path.isdir(model_args.model_name_or_path_qry):
            if model_args.untie_encoder:                
                logger.info(f'Loading query model weight from {model_args.model_name_or_path_qry}')
                lm_q = cls._load_model(model_args.model_name_or_path_qry, model_args=model_args, merge_peft_weights=True, **hf_kwargs)
                den_pooler_q = cls._load_pooler(model_args.model_name_or_path_qry)

                logger.info(f'Loading passage model weight from {model_args.model_name_or_path_psg}')
                lm_p = cls._load_model(model_args.model_name_or_path_psg, model_args=model_args, merge_peft_weights=True, **hf_kwargs)
                den_pooler_p = cls._load_pooler(model_args.model_name_or_path_psg)
            else:
                lm_q = cls._load_model(model_args.model_name_or_path, model_args=model_args, merge_peft_weights=True, **hf_kwargs)
                lm_p = lm_q

                den_pooler_q = cls._load_pooler(model_args.model_name_or_path)
                den_pooler_p = den_pooler_q
        # Load checkpoint online
        else:
            lm_q = cls._load_model(model_args.model_name_or_path, model_args=model_args, merge_peft_weights=True, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

            # TODO: Pooler cannot be loaded online
            den_pooler_q, den_pooler_p = None, None
        
        # Learnable MLP that project from in_dim to out_dim
        if model_args.add_pooler:
            # Build Pooler from scratch if they are not loaded yet
            if (den_pooler_q is None) or (den_pooler_p is None):
                if model_args.projection_in_dim_qry is None:
                    model_args.projection_in_dim_qry = lm_q.config.hidden_size  # Auto infer MLP Fan-in
                den_pooler_q = cls._build_pooler(
                    projection_in_dim=model_args.projection_in_dim_qry, 
                    projection_out_dim=model_args.projection_out_dim_qry,
                    model_name_or_path=model_args.model_name_or_path_qry,
                )
                if "torch_dtype" in hf_kwargs:
                    den_pooler_q = den_pooler_q.to(dtype=hf_kwargs["torch_dtype"])

                if model_args.untie_encoder:
                    if model_args.projection_in_dim_psg is None:
                        model_args.projection_in_dim_psg = lm_p.config.hidden_size  # Auto infer MLP Fan-in
                    den_pooler_p = cls._build_pooler(
                        projection_in_dim=model_args.projection_in_dim_psg, 
                        projection_out_dim=model_args.projection_out_dim_psg,
                        model_name_or_path=model_args.model_name_or_path_psg,
                    )
                    if "torch_dtype" in hf_kwargs:
                        den_pooler_p = den_pooler_p.to(dtype=hf_kwargs["torch_dtype"])
                else:
                    if model_args.projection_in_dim_psg is None:
                        model_args.projection_in_dim_psg = model_args.projection_in_dim_qry
                    den_pooler_p = den_pooler_q
        else:
            assert (den_pooler_q is None) and (den_pooler_p is None), "Poolers are already loaded, but your setting is fine-tuning without them. Please further check your configuration."

        # Enable input embedding require gradient
        if train_args.gradient_checkpointing:
            lm_q.enable_input_require_grads()
            lm_p.enable_input_require_grads()
        
        # LoRA
        if train_args.lora:
            lm_q = cls._build_lora_model(lm_q, model_args.model_name_or_path_qry, train_args)
            if model_args.untie_encoder:
                lm_p = cls._build_lora_model(lm_p, model_args.model_name_or_path_psg, train_args)
            else:
                lm_p = lm_q

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            model_args=model_args,
            train_args=train_args,
            data_args=data_args,
            den_pooler_q=den_pooler_q,
            den_pooler_p=den_pooler_p,
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path: str,
            model_args: ModelArguments = None,
            train_args: TrainingArguments = None,
            data_args: DataArguments = None,
            **hf_kwargs,
        ):
        """
        Load retriever (saved by .save() function) for inferencing.

        Args:
            model_name_or_path: Folder path to saved retriever.
            model_args: All hyper-parameters for inferencing.
            train_args: Optional. Do not needed for inferencing.
        """
        merge_peft_weights = train_args is None     # True: Inference/Initial Training(Build); False: Resume training
        # Resume model_args if not set
        if model_args is None:
            _local_model_args_path = os.path.join(model_name_or_path, 'model_args.yaml')
            if os.path.exists(_local_model_args_path):
                logger.info(f"Reading config file from {_local_model_args_path}")
                parsed_tuple = HfArgumentParser(ModelArguments).parse_yaml_file(_local_model_args_path)
                model_args: ModelArguments = parsed_tuple[0]
                model_args.model_name_or_path = model_name_or_path
            else:
                raise FileNotFoundError(f"Please pass a model_args to initialize the model.")

        # Load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                model_args.untie_encoder = True
                model_args.model_name_or_path_qry = _qry_model_path
                model_args.model_name_or_path_psg = _psg_model_path

                logger.info(f'Found separate weight for query/passage encoders')
                logger.info(f'Loading query model weight from {_qry_model_path}')
                lm_q = cls._load_model(_qry_model_path, model_args=model_args, merge_peft_weights=merge_peft_weights, **hf_kwargs)

                den_pooler_q = cls._load_pooler(_qry_model_path)
                if den_pooler_q is not None:
                    den_pooler_q = den_pooler_q.to(device=lm_q.device, dtype=lm_q.dtype)
                
                logger.info(f'Loading passage model weight from {_psg_model_path}')
                lm_p = cls._load_model(_psg_model_path, model_args=model_args, merge_peft_weights=merge_peft_weights, **hf_kwargs)

                den_pooler_p = cls._load_pooler(_psg_model_path)
                if den_pooler_p is not None:
                    den_pooler_p = den_pooler_p.to(device=lm_p.device, dtype=lm_p.dtype)
            else:
                model_args.untie_encoder = False
                model_args.model_name_or_path_qry = model_name_or_path
                model_args.model_name_or_path_psg = model_name_or_path

                logger.info(f'Loading tied model weight from {model_name_or_path}')
                lm_q = cls._load_model(model_name_or_path, model_args=model_args, merge_peft_weights=merge_peft_weights, **hf_kwargs)
                lm_p = lm_q

                den_pooler_q = cls._load_pooler(model_name_or_path)
                if den_pooler_q is not None:
                    den_pooler_q = den_pooler_q.to(device=lm_q.device, dtype=lm_q.dtype)
                den_pooler_p = den_pooler_q
        else:
            model_args.untie_encoder = False
            model_args.model_name_or_path_qry = model_name_or_path
            model_args.model_name_or_path_psg = model_name_or_path
            
            logger.info(f'Loading tied model weight from {model_name_or_path}')
            lm_q = cls._load_model(model_name_or_path, model_args=model_args, merge_peft_weights=merge_peft_weights, **hf_kwargs)
            lm_p = lm_q

            # TODO: Poolers cannot be loaded online
            den_pooler_q = cls._load_pooler(_qry_model_path)
            if den_pooler_q is not None:
                den_pooler_q = den_pooler_q.to(device=lm_q.device, dtype=lm_q.dtype)
            den_pooler_p = den_pooler_q
        
        if (den_pooler_q is not None) or (den_pooler_p is not None):
            model_args.add_pooler = True

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            model_args=model_args,
            train_args=train_args,
            data_args=data_args,
            den_pooler_q=den_pooler_q,
            den_pooler_p=den_pooler_p,
        ).to(lm_q.device)
        return model
    
    @staticmethod
    def _get_prefix_dict(state_dict: dict[str, Tensor], prefix: str):
        """ Get dict items with prefixed keys, meanwhile remove the prefix """
        if state_dict is None:
            return None
        
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    def save(self, output_dir: str, state_dict: Optional[dict[str, any]]=None, **hf_kwargs):
        """ Save HF format checkpoint & Poolers 
            Args:
                output_dir (str): Path to save checkpoints & poolers.
                state_dict (dict[str, Tensor]): Optional model state dict, provided by Accelerator's all-gather.
        """
        # Dump model_args
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'model_args.yaml'), 'w') as f:
            yaml.dump(self.model_args.__dict__, f, indent=2)
        
        # Save ckpt
        state_dict_lm_q, state_dict_lm_p, state_dict_pooler_q, state_dict_pooler_p = None, None, None, None
        if self.model_args.untie_encoder:
            if state_dict is not None:
                # Unwrap state dict keys
                state_dict_lm_q = self._get_prefix_dict(state_dict, 'lm_q.')
                state_dict_lm_p = self._get_prefix_dict(state_dict, 'lm_p.')
            
            _qry_model_path = os.path.join(output_dir, 'query_model')
            _psg_model_path = os.path.join(output_dir, 'passage_model')
            os.makedirs(_qry_model_path, exist_ok=True)
            os.makedirs(_psg_model_path, exist_ok=True)
            self.lm_q.save_pretrained(_qry_model_path, state_dict=state_dict_lm_q, **hf_kwargs)
            self.lm_p.save_pretrained(_psg_model_path, state_dict=state_dict_lm_p, **hf_kwargs)

            if self.den_pooler_q:
                if state_dict is not None:
                    state_dict_pooler_q = self._get_prefix_dict(state_dict, 'den_pooler_q.')
                self.den_pooler_q.save_pooler(_qry_model_path, state_dict=state_dict_pooler_q)
            
            if self.den_pooler_p:
                if state_dict is not None:
                    state_dict_pooler_p = self._get_prefix_dict(state_dict, 'den_pooler_p.')
                self.den_pooler_p.save_pooler(_psg_model_path, state_dict=state_dict_pooler_p)
        else:
            if state_dict is not None:
                state_dict_lm_q = self._get_prefix_dict(state_dict, 'lm_q.')
            self.lm_q.save_pretrained(output_dir, state_dict=state_dict_lm_q, **hf_kwargs)

            if self.den_pooler_q:
                if state_dict is not None:
                    state_dict_pooler_q = self._get_prefix_dict(state_dict, 'den_pooler_q.')
                self.den_pooler_q.save_pooler(output_dir, state_dict=state_dict_pooler_q)
