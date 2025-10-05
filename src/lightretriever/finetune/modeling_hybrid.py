#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Hybrid (Dense + Sparse) Model Implementation.

@Time    :   2024/08/29
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import time
from typing import Callable, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from transformers import (
    AutoModelForMaskedLM, 
    AutoModelForCausalLM, 
    BatchEncoding, 
    AutoTokenizer, 
    PreTrainedModel,
    BertForMaskedLM,
    XLMRobertaForMaskedLM,
    XLMRobertaForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM, 
    MistralForCausalLM, 
    PhiForCausalLM, 
    Qwen2ForCausalLM,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead
from peft import PeftModel

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from ..utils.nested_input import apply_seqlen_cumulate
from ..utils.monkey_patch import apply_bidirectional_attention
from .dense_pooling import pooling, mean_eos_pooling
from .dense_projector import DenseLinearProjector
from .modeling_encoder import EncoderModel, EncoderOutput
from .sparse_pooling import (
    get_sparse_attention_mask, 
    top_p_sampling, 
    top_k_sampling, 
    get_unique_token_ids,
    get_scores_with_indices,
    aggregate,
)
from .sparse_projector import SparseLinearProjector, SparseDownProjector
from .emb_bag_mixin import EmbeddingBagMixin
from .sparse_converter_mixin import SparseConverterMixin

import logging
logger = logging.getLogger(__name__)

def get_base_model(model: PreTrainedModel):
    """ Get Base Model from MLM/CLM Model Classes """
    if isinstance(model, PeftModel):
        model_unwraped = model.get_base_model()
    else:
        model_unwraped = model
    
    if isinstance(model_unwraped, BertForMaskedLM):
        return model_unwraped.bert
    elif isinstance(model_unwraped, (XLMRobertaForMaskedLM, XLMRobertaForCausalLM)):
        return model_unwraped.roberta
    elif isinstance(model_unwraped, GPTNeoXForCausalLM):
        return model_unwraped.gpt_neox
    elif "ForCausalLM" in model_unwraped.__class__.__name__ :
        return model_unwraped.model     # All Llama-alike models
    else:
        raise NotImplementedError(f"Unrecognized type of model_unwraped {type(model_unwraped)}.")

def get_lm_head(model: PreTrainedModel):
    """ Get LM Head from MLM/CLM Model Classes, used for sparse training """
    if isinstance(model, PeftModel):
        model_unwraped = model.get_base_model()
    else:
        model_unwraped = model
    
    if isinstance(model_unwraped, BertForMaskedLM):
        return model_unwraped.cls
    elif isinstance(model_unwraped, (XLMRobertaForMaskedLM, XLMRobertaForCausalLM)):
        return model_unwraped.lm_head
    elif isinstance(model_unwraped, GPTNeoXForCausalLM):
        return model_unwraped.embed_out
    elif "ForCausalLM" in model_unwraped.__class__.__name__:
        return model_unwraped.lm_head
    else:
        raise NotImplementedError(f"Unrecognized type of model_unwraped {type(model_unwraped)}.")


class HybridModel(EncoderModel, EmbeddingBagMixin, SparseConverterMixin):
    def __init__(
        self,
        lm_q: PreTrainedModel | PeftModel,
        lm_p: PreTrainedModel | PeftModel,
        model_args: ModelArguments,
        train_args: Optional[TrainingArguments] = None,
        data_args: Optional[DataArguments] = None,
        den_pooler_q: Optional[DenseLinearProjector] = None,
        den_pooler_p: Optional[DenseLinearProjector] = None,
        spr_pooler_q: Optional[SparseLinearProjector | SparseDownProjector] = None,
        spr_pooler_p: Optional[SparseLinearProjector | SparseDownProjector] = None,
    ):
        # Init parent
        EncoderModel.__init__(self, lm_q=lm_q, lm_p=lm_p, model_args=model_args, train_args=train_args, data_args=data_args, den_pooler_q=den_pooler_q, den_pooler_p=den_pooler_p)

        # Re-wrap forward monkey patches
        if self.model_args.cumulative_seq:
            apply_seqlen_cumulate(self.lm_p_base_unwrap)
            if self.model_args.untie_encoder:
                apply_seqlen_cumulate(self.lm_q_base_unwrap)
        
        if self.model_args.enable_bidirectional_attention:
            apply_bidirectional_attention(self.lm_p_base_unwrap)
            if self.model_args.untie_encoder:
                apply_bidirectional_attention(self.lm_q_base_unwrap)

        # Vocab related
        tokenizer = self.load_tokenizer(model_args.model_name_or_path, model_args=model_args)
        self.vocab_dict = {v: k.strip("'") for k, v in tokenizer.get_vocab().items()}    # idx -> token
        self.sep_token_id = tokenizer.sep_token_id
        self.eos_token_id = tokenizer.eos_token_id
        ## Init mixins
        SparseConverterMixin.__init__(self, self.vocab_dict)
        EmbeddingBagMixin.__init__(self)

        # Sparse Projector project the representations from `hidden_dim` to `vocab_dim`
        self.spr_pooler_q = spr_pooler_q
        self.spr_pooler_p = spr_pooler_p

        # Scaling factor for all regulation losses (Only used during Training)
        self.reg_scaling_factor: float = 1.0
    
    @property
    def lm_q_base_unwrap(self):
        return get_base_model(self.lm_q)
    
    @property
    def lm_p_base_unwrap(self):
        return get_base_model(self.lm_p)
    
    def get_sparse_emb(
        self, 
        logits: Tensor,
        is_query: bool = False,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        unique_token_ids: Optional[list[list[int]]] = None,
    ):
        """ Sparsify method implementations. 
            Note: In-place operations can save memory, but only for inferencing, do not support bwd
        
            Args:
                logits (Tensor): `Pooled` or `Aggregated` logits. Shape `[batch_size, vocab_size]`.
                is_query (bool): A `boolean flag` used for top-p / top-k sampling with `seperate ratios`.
                input_ids (Tensor): Shape `[batch_size, seq_len]`. Input ids used by `adaptive top-k` to count how many
                                    unique tokens per line of input. Thus the expected number of `top-k` is `number of
                                    unique tokens` * `query/passage expansion ratio`
                attention_mask (Tensor): Shape `[batch_size, seq_len]`. Attention mask is used when pooling from original 
                                    input ids is activated.
                unique_token_ids (list[list[int]]): List of Pre-tokenized Unique token ids. this is obtained from 
                                    `ICUWordPreTokenized`, which can be used for `unique token pooling`.
        """
        assert logits.ndim <= 2, f"logits' dim is {logits.ndim}, please pass logits with correct shape."
        if logits.ndim <= 1:
            logits = logits.view(1, logits.shape[-1])
        
        # ** Sparsify **
        # (Optional) Logits sampling: ICUWordPreTokenized token ids
        if self.model_args.sparse_pool_from_unique_token_ids:
            # Only pooling unique tokens
            assert unique_token_ids is not None
            logits = get_scores_with_indices(logits, indices=unique_token_ids)
        
        # (Optional) Logits sampling: Original input ids
        if (is_query and self.model_args.sparse_pool_from_original_input_ids_qry) or \
            ((not is_query) and self.model_args.sparse_pool_from_original_input_ids_psg):
            mask = get_sparse_attention_mask(input_ids, attention_mask, sep_token_id=self.sep_token_id, remove_prompt=self.model_args.add_sep_token) # [TODO] Fix remove_prompt for BERT-like models, which depends on [SEP] at last but not prompt seperator
            unique_input_token_ids = get_unique_token_ids(input_ids, mask)
            logits = get_scores_with_indices(logits, indices=unique_input_token_ids)
        
        # [Suggested] ReLU, Log Saturation
        if self.model_args.sparse_use_relu:
            logits = torch.relu(logits) if self.training else torch.relu_(logits)
        
        if self.model_args.sparse_use_log_saturation:
            logits = torch.log1p(logits) if self.training else torch.log1p_(logits)

        ## (Optional) Top-p
        logits = top_p_sampling(
            logits, 
            top_p=self.model_args.sparse_top_p_qry if is_query else self.model_args.sparse_top_p_psg, 
            min_tokens_to_keep=self.model_args.sparse_min_tokens_to_keep
        )
        
        ## (Optional) Top-k
        logits = top_k_sampling(
            logits, 
            top_k=self.model_args.sparse_top_k_qry if is_query else self.model_args.sparse_top_k_psg, 
            min_tokens_to_keep=self.model_args.sparse_min_tokens_to_keep
        )

        return logits
    
    def encode_passage(
        self, 
        psg: Optional[BatchEncoding | dict[str, any]], 
        normalize: Optional[bool] = None, 
        encode_dense: Optional[bool] = None, 
        encode_sparse: Optional[bool] = None, 
        **kwargs
    ):
        """
        Encoding passage.

        Args:
            psg (Optional[BatchEncoding | dict[str, any]]): Inputs with input_ids, attention_mask (optional). 
                                           Shape [batch_size, seq_len].
            normalize (Optional[bool]): Overriding whether to use l2 normalization for embedding.
                                        - `None`: Listen to self.model_args.normalize.
                                        - `True/False`: Activate normalization / Deactivate normalization.
            encode_dense (Optional[bool]): Overriding whether to encode dense representations.
            encode_sparse (Optional[bool]): Overriding whether to encode sparse representations.
        
        Returns: 
            Dict of
            - dense_reps (Optional[Tensor], fp16/bf16): Shape [batch_size, rep_dim].
            - sparse_reps (Optional[Tensor], always fp32): Shape [batch_size, vocab_dim].
        """
        # Safety check
        if psg is None:
            return None
        
        normalize = normalize or (normalize is None and self.model_args.normalize)
        encode_dense = encode_dense or (encode_dense is None and self.model_args.hybrid_use_dense_vector)
        encode_sparse = encode_sparse or (encode_sparse is None and self.model_args.hybrid_use_sparse_vector)

        # To support LightRetriever Asymmetric Retrieval
        ## query emb vec <-> passage dense vec
        ## query token id vec <-> passage sparse vec
        if self.model_args.hybrid_use_emb_vector:
            encode_dense = True

        if self.model_args.hybrid_use_token_id_vector:
            encode_sparse = True

        # Format Inputs
        forward_kwargs = {
            'input_ids': psg['input_ids'],
            'attention_mask': psg.get('attention_mask', None),
        }
        for optional_input_field in ['token_type_ids', 'position_ids']:
            if optional_input_field in psg:
                forward_kwargs[optional_input_field] = psg[optional_input_field]
        forward_kwargs['return_dict'] = True
        forward_kwargs['use_cache'] = False    # Do not return `past_key_values`
        forward_kwargs['output_hidden_states'] = True if self.model_args.pooling_strategy_psg in ["avg_first_last", "avg_top2"] else False

        # Forward
        psg_out: BaseModelOutput = self.lm_p_base_unwrap(**forward_kwargs)

        # Rep Pooling
        p_reps_dict: dict[str, Tensor] = dict()
        ## Get dense Emb
        if encode_dense:
            p_reps = pooling(
                last_hidden=psg_out.last_hidden_state,
                hidden_states=psg_out.hidden_states,
                attention_mask=psg.get('attention_mask', None),
                pooling_strategy=self.model_args.pooling_strategy_psg,
            )
            if self.den_pooler_p is not None:
                p_reps = self.den_pooler_p(p_reps)  # D * d
            if self.model_args.dense_shrink_dim:
                p_reps = p_reps[..., :self.model_args.dense_shrink_dim]
            if normalize:
                p_reps = F.normalize(p_reps, p=2, dim=-1)
            p_reps_dict["dense_reps"] = p_reps
        
        ## Get Sparse Emb
        if encode_sparse:
            # Sparse Pooling
            sparse_hidden = psg_out.last_hidden_state   # [batch_size, seq_len, hidden_dim]
            spr_proj = self.spr_pooler_p or get_lm_head(self.lm_p) 
            if self.model_args.sparse_pooling_strategy is not None:
                # Pooling: Pool then proj
                sparse_hidden = pooling(
                    sparse_hidden, 
                    attention_mask=psg.get('attention_mask', None), 
                    pooling_strategy=self.model_args.sparse_pooling_strategy
                )   # [batch_size, hidden_dim]

                _spr_proj_kwargs = {}
                if isinstance(spr_proj, SparseDownProjector):
                    _spr_proj_kwargs["input_ids"] = psg["input_ids"]
                
                logits = spr_proj.forward(sparse_hidden, **_spr_proj_kwargs)
            else:
                # Aggregation: Proj then aggr
                sparse_attention_mask = get_sparse_attention_mask(
                    psg['input_ids'], psg['attention_mask'], sep_token_id=self.sep_token_id, 
                    remove_prompt=self.model_args.add_sep_token
                ) # [TODO] Fix remove_prompt for BERT-like models, which depends on [SEP] at last but not prompt seperator
                logits = aggregate(
                    sparse_hidden, 
                    lm_head=spr_proj, 
                    sparse_attention_mask=sparse_attention_mask,
                    sparse_use_max_aggregation=self.model_args.sparse_use_max_aggregation,
                )
            
            # Sparsify
            aggregated_psg_out = self.get_sparse_emb(
                logits, is_query=False, input_ids=psg['input_ids'], attention_mask=psg.get('attention_mask', None),
                unique_token_ids=psg['unique_token_ids']
            )
            
            # TODO: Not sure if we need normalization for sparse embedding!
            # https://github.com/naver/splade/issues/64
            # https://github.com/naver/splade/issues/34
            # if normalize or (normalize is None and self.model_args.normalize):
            #     aggregated_psg_out = F.normalize(aggregated_psg_out, p=2, dim=-1)

            p_reps_dict["sparse_reps"] = aggregated_psg_out

        return p_reps_dict

    def encode_query(
        self, 
        qry: Optional[BatchEncoding | dict[str, any]], 
        normalize: Optional[bool] = None, 
        encode_dense: Optional[bool] = None, 
        encode_sparse: Optional[bool] = None, 
        encode_emb_reps: Optional[bool] = None, 
        encode_token_id_reps: Optional[bool] = None, 
        **kwargs
    ):
        """
        Encoding query.

        Args:
            qry (Optional[BatchEncoding] | dict[str, any]): Inputs with input_ids, attention_mask (optional). 
                                           Shape [batch_size, seq_len].
            normalize (Optional[bool]): Overriding whether to use l2 normalization for embedding.
                                        - `None`: Listen to self.model_args.normalize.
                                        - `True/False`: Activate normalization / Deactivate normalization.
            encode_dense (Optional[bool]): Overriding whether to encode dense representations.
            encode_sparse (Optional[bool]): Overriding whether to encode sparse representations.
            encode_emb_reps (Optional[bool]): Overriding whether to encode Embedding Layer representations.
            encode_token_id_reps (Optional[bool]): Overriding whether to encode token id representations.
        
        Returns: 
            Dict of
            - dense_reps (Optional[Tensor]): Shape [batch_size, rep_dim].
            - sparse_reps (Optional[Tensor]): Shape [batch_size, vocab_dim].
            - emb_reps (Optional[Tensor]): Shape [batch_size, rep_dim].
            - token_id_reps (Optional[Tensor | str]): Shape [batch_size, vocab_dim]. Tensor when `self.training == True`; Str when `self.training == False`
        """
        # Safety check
        if qry is None:
            return None
        
        normalize = normalize or (normalize is None and self.model_args.normalize)
        encode_dense = encode_dense or (encode_dense is None and self.model_args.hybrid_use_dense_vector)
        encode_sparse = encode_sparse or (encode_sparse is None and self.model_args.hybrid_use_sparse_vector)
        encode_emb_reps = encode_emb_reps or (encode_emb_reps is None and self.model_args.hybrid_use_emb_vector)
        encode_token_id_reps = encode_token_id_reps or (encode_token_id_reps is None and self.model_args.hybrid_use_token_id_vector)

        # No model forward is needed when `encode_emb_reps` / `encode_token_id_reps`
        if use_model := encode_dense or encode_sparse:
            # Format Inputs
            forward_kwargs = {
                'input_ids': qry['input_ids'],
                'attention_mask': qry.get('attention_mask', None),
            }
            for optional_input_field in ['token_type_ids', 'position_ids']:
                if optional_input_field in qry:
                    forward_kwargs[optional_input_field] = qry[optional_input_field]
            forward_kwargs['return_dict'] = True
            forward_kwargs['use_cache'] = False    # Do not return `past_key_values`
            forward_kwargs['output_hidden_states'] = True if self.model_args.pooling_strategy_qry in ["avg_first_last", "avg_top2"] else False

            # Forward
            qry_out: BaseModelOutput = self.lm_q_base_unwrap(**forward_kwargs)

        # Rep Pooling
        q_reps_dict: dict[str, Tensor] = dict()
        ## Get dense Emb
        if encode_dense:
            q_reps = pooling(
                last_hidden=qry_out.last_hidden_state,
                hidden_states=qry_out.hidden_states,
                attention_mask=qry.get('attention_mask', None),
                pooling_strategy=self.model_args.pooling_strategy_qry,
            )
            if self.den_pooler_q is not None:
                q_reps = self.den_pooler_q(q_reps)  # D * d
            if self.model_args.dense_shrink_dim:
                q_reps = q_reps[..., :self.model_args.dense_shrink_dim]
            if normalize:
                q_reps = F.normalize(q_reps, p=2, dim=-1)
            
            q_reps_dict["dense_reps"] = q_reps
        
        ## Get Sparse Emb
        if encode_sparse:
            # Sparse Pooling
            sparse_hidden = qry_out.last_hidden_state   # [batch_size, seq_len, hidden_dim]
            spr_proj = self.spr_pooler_q or get_lm_head(self.lm_q) 
            if self.model_args.sparse_pooling_strategy is not None:
                # Pooling: Pool then proj
                sparse_hidden = pooling(
                    sparse_hidden, 
                    attention_mask=qry.get('attention_mask', None), 
                    pooling_strategy=self.model_args.sparse_pooling_strategy
                )   # [batch_size, hidden_dim]

                _spr_proj_kwargs = {}
                if isinstance(spr_proj, SparseDownProjector):
                    _spr_proj_kwargs["input_ids"] = qry["input_ids"]
                
                logits = spr_proj.forward(sparse_hidden, **_spr_proj_kwargs)
            else:
                # Aggregation: Proj then aggr
                sparse_attention_mask = get_sparse_attention_mask(qry['input_ids'], qry['attention_mask'], sep_token_id=self.sep_token_id, remove_prompt=self.model_args.add_sep_token) # [TODO] Fix remove_prompt for BERT-like models, which depends on [SEP] at last but not prompt seperator
                logits = aggregate(
                    sparse_hidden, 
                    lm_head=spr_proj, 
                    sparse_attention_mask=sparse_attention_mask,
                    sparse_use_max_aggregation=self.model_args.sparse_use_max_aggregation,
                )
            
            # Sparsify
            aggregated_qry_out = self.get_sparse_emb(
                logits, is_query=True, input_ids=qry['input_ids'], attention_mask=qry.get('attention_mask', None),
                unique_token_ids=qry.get('unique_token_ids', None), 
            )
            
            # TODO: Not sure if we need normalization for sparse embedding!
            # if normalize:
            #     aggregated_qry_out = F.normalize(aggregated_qry_out, p=2, dim=-1)

            q_reps_dict["sparse_reps"] = aggregated_qry_out
        
        ## Get Embedding Bag reps
        if encode_emb_reps:
            if self.model_args.noncontextual_query_embedding:
                # Train & Encode `dense query reps` using `Non-contextual query embedding`, 
                # which is a mean-pooled (prompted)-query eos token embedding. 
                # Query tokens are indivisually encoded (with optional prompt). 
                # Then query sentence embedding is aggregated by mean pooling the 
                # coresponding eos token ids.
                
                if self.training:
                    # Training with concated inputs for efficiency
                    # Note: Flash attention / Cumulated forward is not supported because of customized 4-D attention_mask
                    non_ctx_qry_out: BaseModelOutput = self.lm_q_base_unwrap(
                        input_ids=qry['nonctx_tok_emb_input_ids'],
                        attention_mask=qry['nonctx_tok_emb_attention_mask'],
                        position_ids=qry['nonctx_tok_emb_position_ids'],
                        return_dict=True, 
                        use_cache=False,    # Do not return `past_key_values`
                        output_hidden_states=True if self.model_args.pooling_strategy_qry in ["avg_first_last", "avg_top2"] else False,
                        **kwargs
                    )
                    emb_reps = mean_eos_pooling(
                        non_ctx_qry_out.last_hidden_state,
                        input_ids=qry['nonctx_tok_emb_input_ids'],
                        attention_mask=qry['nonctx_tok_emb_attention_mask_2d'],
                        eos_id=self.eos_token_id,
                    )
                else:
                    # Inference with EmbeddingBag
                    assert self.emb_bag is not None, "Please load a EmbeddingBag to encode a EmbeddingBag layer for inference." 
                    emb_reps = self.emb_bag.forward(input=qry['nonctx_tok_emb_input_ids'], offsets=qry['nonctx_tok_emb_offsets'])
            
            else:
                # Ablation: LM's input Embedding layer as EmbeddingBag. Performance of this setting
                #           is bad, should not directly use LM's input Embedding.
                emb_layer = self.lm_q.get_input_embeddings()
                inputs_embeds: Tensor = emb_layer(qry['input_ids'])
                emb_reps = pooling(
                    last_hidden=inputs_embeds,
                    attention_mask=qry.get('attention_mask', None),
                    pooling_strategy='mean',
                )
            
            if self.model_args.dense_shrink_dim:
                emb_reps = emb_reps[..., :self.model_args.dense_shrink_dim]
            if normalize:
                emb_reps = F.normalize(emb_reps, p=2, dim=-1)
            q_reps_dict["emb_reps"] = emb_reps
        
        ## Get Bag-of-word Token id reps
        if encode_token_id_reps:
            if self.training:
                q_reps_dict["token_id_reps"] = qry['token_id_reps_pt']
            else:
                q_reps_dict["token_id_reps"] = qry['token_id_reps_str']

        return q_reps_dict
    
    def _call_compute_loss(
        self,
        # Inputs
        q_reps: Tensor, 
        p_reps: Tensor,
        ce_scores: Optional[Tensor],
        only_hn: Optional[Tensor],
        # Outputs Related
        loss: Tensor,
        scores: dict[str, Tensor],
        logs: defaultdict[str, float],
        score_name: str,
        log_prefix: str = "",
        log_suffix: str = "",
        **kwargs,
    ):
        """ Helper function to compute loss and postprocess related scores/logs.

        This function:
            1. Compute `Contrastive Loss (CL)` for `q_reps` & `p_reps`. 
            2. `KL Loss` will also be computed if `ce_scores` is given. 
            3. Postprocess `final loss`, related `scores/logs`
        
        Args:
            q_reps (Tensor): Query representations.
            p_reps (Tensor): Passage representations.
            ce_scores (Optional[Tensor]): Cross-encoder scores (for KL loss).
            only_hn (Optional[Tensor]): Hard negatives only mask.
            loss (Tensor): Output placeholder for loss.
            scores (dict[str, Tensor]): Output placeholder for scores.
            logs (defaultdict[str, float]): Output placeholder for logs.
            score_name (str): Name of the score.
            log_prefix (str, optional): Prefix for log keys.
            log_suffix (str, optional): Suffix for log keys.

        Returns:
            A tuple containing:
                - lm_out (EncoderOutput): Output of `compute_loss`.
                - loss (Tensor): Final loss.
                - scores (dict[str, Tensor]): Scores map with `score_name -> current gathered scores`.
                - logs (defaultdict[str, float]): Logs map with `{prefix}name{suffix} -> current float log value`.
        """
        # Compute Loss
        lm_out: EncoderOutput = super(HybridModel, self).compute_loss(
            q_reps, p_reps, ce_scores=ce_scores, only_hn=only_hn, **kwargs)
        
        # Add to final loss
        loss += lm_out.loss

        # Add gathered scores to dict
        scores[score_name] = lm_out.scores

        # Parse & Add logs 
        if "clloss" in lm_out.logs:
            logs[f"{log_prefix}clloss{log_suffix}"] = lm_out.logs.pop("clloss")
        if "klloss" in lm_out.logs:
            logs[f"{log_prefix}klloss{log_suffix}"] = lm_out.logs.pop("klloss")
        if "soft_celoss" in lm_out.logs:
            logs[f"{log_prefix}soft_celoss{log_suffix}"] = lm_out.logs.pop("soft_celoss")
        for k, v in lm_out.logs.items():  # Remaining logs
            logs[k] += v
        
        return lm_out, loss, scores, logs
    
    def _call_kl_loss(
        self,
        # Inputs
        student_scores: Tensor, 
        teacher_scores: Tensor,
        kl_coef: float,
        # Outputs Related
        loss: Tensor,
        logs: defaultdict[str, float],
        log_prefix: str = "",
        log_suffix: str = "",
        **kwargs,
    ):
        """ Helper function to compute customized KL loss and postprocess related logs. 

        Args:
            student_scores (Tensor): Student scores.
            teacher_scores (Tensor): Teacher scores.
            kl_coef (float): KL loss scaling factor.
            loss (Tensor): Output placeholder for loss.
            logs (defaultdict[str, float]): Output placeholder for logs.
            log_prefix (str, optional): Prefix for log keys.
            log_suffix (str, optional): Suffix for log keys.

        Returns:
            A Tuple of 
                - loss (Tensor): Final loss.
                - logs (defaultdict[str, float]): Logs map with `{prefix}name{suffix} -> current float log value`.
        """
        # Compute Loss
        klloss = self.klloss(student_scores, teacher_scores) * kl_coef

        # Parse & Add logs 
        logs[f"{log_prefix}klloss{log_suffix}"] = klloss.item() if self.train_args.loss_reduction == 'mean' else klloss.mean().item()
        
        loss += klloss

        return loss, logs
    
    def shrink(self, tensor: Tensor, dim: Optional[int]):
        """ Shrink tensor's last dimension to `dim`. Then renormalize if necessary. 
            This is a helper function for MRL training / inferencing.
        """
        # Do not shrink dim if not set
        if not dim:
            return tensor

        # Shrink the last dim
        tensor_dim = tensor.shape[-1]
        assert dim <= tensor_dim, \
            f"Dimension {dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
        
        tensor = tensor[..., :dim]
        if self.model_args.normalize:
            tensor = F.normalize(tensor, p=2, dim=-1)
        
        return tensor

    def compute_loss(
        self, 
        q_reps: Tensor | dict[str, Tensor], 
        p_reps: Tensor | dict[str, Tensor], 
        ce_scores: Optional[Tensor] = None,
        only_hn: Optional[Tensor] = None,
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
        assert isinstance(q_reps, dict) and isinstance(p_reps, dict)

        # Unpack
        q_dense_reps = q_reps.get("dense_reps", None)
        q_sparse_reps = q_reps.get("sparse_reps", None)
        q_emb_reps = q_reps.get("emb_reps", None)
        q_token_id_reps = q_reps.get("token_id_reps", None)

        p_dense_reps = p_reps.get("dense_reps", None)
        p_sparse_reps = p_reps.get("sparse_reps", None)

        loss = 0.
        logs = defaultdict(float)
        scores = dict()

        # Dense
        if use_dense := (q_dense_reps is not None) and (p_dense_reps is not None):
            # MRL Loss Computation
            for mrl_dim in self.train_args.matryoshka_dims:
                dense_lm_out, loss, scores, logs = self._call_compute_loss(
                    self.shrink(q_dense_reps, mrl_dim), self.shrink(p_dense_reps, mrl_dim), 
                    ce_scores, only_hn, loss, scores, logs,
                    score_name="dense", log_prefix="mrl_loss/den_", log_suffix=f"-dim{mrl_dim}", **kwargs
                )
        
        # Sparse
        if use_sparse := (q_sparse_reps is not None) and (p_sparse_reps is not None):
            sparse_lm_out, loss, scores, logs = self._call_compute_loss(
                q_sparse_reps, p_sparse_reps, ce_scores, only_hn, loss, scores, logs,
                score_name="sparse", log_prefix="spr_", 
                temperature=self.train_args.sparse_temperature, # Override sparse temperature
                **kwargs
            )
        
        # Asymmetric Dense (Embedding Bag Reps)
        if use_emb_reps := (q_emb_reps is not None) and (p_dense_reps is not None):
            for mrl_dim in self.train_args.matryoshka_dims:
                imb_dense_lm_out, loss, scores, logs = self._call_compute_loss(
                    self.shrink(q_emb_reps, mrl_dim), self.shrink(p_dense_reps, mrl_dim),
                    ce_scores, only_hn, loss, scores, logs,
                    score_name="imb_dense", log_prefix="mrl_loss/imbden_", log_suffix=f"-dim{mrl_dim}", **kwargs
                )
                
                if self.train_args.emb_den_reps_distillation:
                    # Distillation: q_emb_reps -> q_dense_reps
                    assert use_dense, "Please enable `use_dense` to distill q_emb_reps -> q_dense_reps."
                    loss, logs = self._call_kl_loss(
                        self.shrink(q_emb_reps, mrl_dim), self.shrink(q_dense_reps, mrl_dim).detach(),
                        self.train_args.emb_reps_distill_coef, loss, logs,
                        log_prefix="mrl_loss/emb_den_reps_", log_suffix=f"-dim{mrl_dim}", **kwargs,
                    )
                
                if self.train_args.emb_den_scores_distillation:
                    # Distillation: q_emb_reps * p_dense_reps -> q_dense_reps * p_dense_reps
                    assert use_dense, "Please enable `use_dense` to distill q_emb_reps * p_dense_reps -> q_dense_reps * p_dense_reps."
                    loss, logs = self._call_kl_loss(
                        imb_dense_lm_out.scores, dense_lm_out.scores.detach(), 
                        self.train_args.emb_reps_distill_coef, loss, logs,
                        log_prefix="mrl_loss/emb_den_scores_", log_suffix=f"-dim{mrl_dim}", **kwargs,
                    )
        
        # Asymmetric Sparse (Token id reps)
        if use_token_id_reps := (q_token_id_reps is not None) and (p_sparse_reps is not None):
            imb_sparse_lm_out, loss, scores, logs = self._call_compute_loss(
                q_token_id_reps, p_sparse_reps, ce_scores, only_hn, loss, scores, logs,
                score_name="imb_sparse", log_prefix="imbspr_", 
                temperature=self.train_args.sparse_temperature, # Override sparse temperature
                **kwargs
            )
            
            if self.train_args.tok_den_scores_distillation:
                # Distillation: q_token_id_reps * p_sparse_reps -> q_dense_reps * p_dense_reps
                assert use_dense, "Please enable `use_dense` to distill q_token_id_reps * p_sparse_reps -> q_dense_reps * p_dense_reps."

                # Sparse reps uses dot product
                # Dense reps uses cosine similarity
                # Thus only dense reps needs to be scaled by temperature
                loss, logs = self._call_kl_loss(
                    imb_sparse_lm_out.scores, dense_lm_out.scores.detach(), self.train_args.tok_reps_distill_coef, loss, logs,
                    log_prefix="tok_den_scores_", **kwargs,
                )
        
        # All regulators, gates, sparsifier losses
        apply_query_regulations = use_sparse
        apply_passage_regulations = use_sparse or use_token_id_reps
        if apply_query_regulations or apply_passage_regulations:
            # ** Regulator **
            if self.train_args.add_flops:
                if apply_query_regulations:
                    q_flops_loss = self.flops(q_sparse_reps) * self.train_args.q_norm_loss_factor * self.reg_scaling_factor
                    loss += q_flops_loss
                    logs['q_flops_loss'] = q_flops_loss.item()
                if apply_passage_regulations:
                    p_flops_loss = self.flops(p_sparse_reps) * self.train_args.p_norm_loss_factor * self.reg_scaling_factor
                    loss += p_flops_loss
                    logs['p_flops_loss'] = p_flops_loss.item()
            
            if self.train_args.add_vector_norm:
                if apply_query_regulations:
                    q_norm_loss = self.norm_loss(q_sparse_reps, ord=self.train_args.norm_ord) * self.train_args.q_norm_loss_factor * self.reg_scaling_factor
                    loss += q_norm_loss
                    logs['q_norm_loss'] = q_norm_loss.item()
                if apply_passage_regulations:
                    p_norm_loss = self.norm_loss(p_sparse_reps, ord=self.train_args.norm_ord) * self.train_args.p_norm_loss_factor * self.reg_scaling_factor
                    loss += p_norm_loss
                    logs['p_norm_loss'] = p_norm_loss.item()
                logs['norm_ord'] = self.train_args.norm_ord
            
            # ** Statistics for Sparse Reps Training **
            with torch.no_grad():
                if apply_query_regulations:
                    bs, vocab_size = q_sparse_reps.shape[0], q_sparse_reps.shape[-1]
                else:
                    bs, vocab_size = q_token_id_reps.shape[0], q_token_id_reps.shape[-1]
                n_psg = p_sparse_reps.shape[0] // bs
                pos_idx = torch.arange(bs, dtype=torch.int64, device=p_sparse_reps.device) * n_psg

                # Record sparsity
                if apply_query_regulations:
                    q_l0: Tensor = torch.linalg.vector_norm(torch.abs(q_sparse_reps), ord=0, dim=-1)
                    q_value_max, q_value_min, q_value_mean, q_value_median, q_scaled_non_zero_cnt = self.rowwise_nonzero_stats(q_sparse_reps)
                else:
                    q_l0: Tensor = torch.linalg.vector_norm(torch.abs(q_token_id_reps), ord=0, dim=-1)
                    q_value_max, q_value_min, q_value_mean, q_value_median, q_scaled_non_zero_cnt = self.rowwise_nonzero_stats(q_token_id_reps)
                
                logs['spr_stats/q_l0'] = q_l0.mean().item()
                logs['spr_stats/q_value_max'] = q_value_max.mean().item()
                logs['spr_stats/q_value_min'] = q_value_min.mean().item()
                logs['spr_stats/q_value_mean'] = q_value_mean.mean().item()
                logs['spr_stats/q_value_median'] = q_value_median.mean().item()
                logs['spr_stats/q_scaled_non_zero_cnt'] = q_scaled_non_zero_cnt.mean().item()

                p_l0: Tensor = torch.linalg.vector_norm(torch.abs(p_sparse_reps), ord=0, dim=-1)
                p_value_max, p_value_min, p_value_mean, p_value_median, p_scaled_non_zero_cnt = self.rowwise_nonzero_stats(p_sparse_reps)

                logs['spr_stats/p_l0'] = p_l0.mean().item()
                logs['spr_stats/p_value_max'] = p_value_max.mean().item()
                logs['spr_stats/p_value_min'] = p_value_min.mean().item()
                logs['spr_stats/p_value_mean'] = p_value_mean.mean().item()
                logs['spr_stats/p_value_median'] = p_value_median.mean().item()
                logs['spr_stats/p_scaled_non_zero_cnt'] = p_scaled_non_zero_cnt.mean().item()

                # Calculate expansion ratio on train set
                if q_unique_token_ids := kwargs.get('q_unique_token_ids', None):
                    q_num_input_token: list[int] = list(map(len, q_unique_token_ids))
                    q_num_input_token: Tensor = torch.tensor(q_num_input_token, dtype=torch.int, device=q_l0.device)
                    logs['spr_stats/q_expan_ratio'] = (q_l0 / q_num_input_token).mean().item()
                
                if p_unique_token_ids := kwargs.get('p_unique_token_ids', None):
                    p_num_input_token: list[int] = list(map(len, p_unique_token_ids))
                    p_num_input_token: Tensor = torch.tensor(p_num_input_token, dtype=torch.int, device=p_l0.device)
                    logs['spr_stats/p_expan_ratio'] = (p_l0 / p_num_input_token).mean().item()
                
                if q_unique_token_ids is not None and p_unique_token_ids is not None:
                    # Record domain related sparse alignment metrics if using homogenous batching
                    if domain_name_list := kwargs.get('domain_name', None):
                        assert isinstance(domain_name_list, list)
                        if len(set(domain_name_list)) == 1:  # homogenous batching
                            domain_name = domain_name_list[0]

                            # Count how many pairs of qry & psg pos that does not overlap at all
                            pos_idx = torch.arange(bs, dtype=torch.int64, device=p_sparse_reps.device) * n_psg
                            p_sparse_reps_pos = p_sparse_reps[pos_idx, ...]  # Get positive psg [q_bs, vocab_size]
                            if apply_query_regulations:
                                num_overlap = torch.logical_and(q_sparse_reps, p_sparse_reps_pos).int().sum(-1)     # [q_bs]
                            else:
                                num_overlap = torch.logical_and(q_token_id_reps, p_sparse_reps_pos).int().sum(-1)     # [q_bs]
                            non_overlap_cnt = bs - torch.count_nonzero(num_overlap)
                            logs[f'q_p_pos_non_overlap_cnt/{domain_name}'] = non_overlap_cnt.item()
                            
                            # Count how many pairs of qry & psg neg that does not overlap at all
                            p_sparse_reps_neg = p_sparse_reps.view(bs, n_psg, vocab_size)[..., 1:, :]   # [q_bs, n_psg-1, vocab_size]
                            if apply_query_regulations:
                                num_overlap = torch.logical_and(q_sparse_reps.unsqueeze(1), p_sparse_reps_neg).view(-1, vocab_size).int().sum(-1)     # [q_bs]
                            else:
                                num_overlap = torch.logical_and(q_token_id_reps.unsqueeze(1), p_sparse_reps_neg).view(-1, vocab_size).int().sum(-1)     # [q_bs]
                            non_overlap_cnt = bs * (n_psg-1) - torch.count_nonzero(num_overlap)
                            logs[f'q_p_neg_non_overlap_cnt/{domain_name}'] = non_overlap_cnt.item()
        
        lm_out = EncoderOutput(q_reps=q_reps, p_reps=p_reps, loss=loss, scores=scores, logs=dict(logs))
        return lm_out
    
    @staticmethod
    def rowwise_nonzero_stats(x: Tensor, scale_factor=100):
        assert x.dim() == 2, "Input tensor must be 2-dimensional"

        mask = x != 0
        x_nonzero = [row[mask[i]] for i, row in enumerate(x)]  # Get non-zero elements of each row

        # Calculate stats
        max_vals = torch.tensor([row.max() if len(row) > 0 else 0 for row in x_nonzero], dtype=torch.float32, device=x.device)
        min_vals = torch.tensor([row.min() if len(row) > 0 else 0 for row in x_nonzero], dtype=torch.float32, device=x.device)
        mean_vals = torch.tensor([row.float().mean() if len(row) > 0 else 0 for row in x_nonzero], dtype=torch.float32, device=x.device)
        median_vals = torch.tensor([row.float().median() if len(row) > 0 else 0 for row in x_nonzero], dtype=torch.float32, device=x.device)

        # Calculate `x * scale_factor` then round as int, sum to get nonzero scaled counts
        x_scaled = torch.floor(x * scale_factor)
        nonzero_scaled_counts = (x_scaled != 0).sum(dim=1)

        return max_vals, min_vals, mean_vals, median_vals, nonzero_scaled_counts.float()
    
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
        # Transformers cls
        if model_args.hybrid_model_architecture == 'gpt':
            cls.TRANSFORMER_CLS = AutoModelForCausalLM
        elif model_args.hybrid_model_architecture == 'bert':
            cls.TRANSFORMER_CLS = AutoModelForMaskedLM
        else:
            raise NotImplementedError(f"Unsupported hybrid model architecture {model_args.hybrid_model_architecture}")
        
        # Non-contextual Query Embedding
        if train_args is not None and \
            model_args.noncontextual_query_embedding and \
            train_args.attn_implementation == "flash_attention_2":
            # Training with noncontextual_query_embedding needs SDPA for lm_q, FA2 for lm_p
            from ..utils.monkey_patch import hacking_fa2_forward_w_4d_attn_mask
            hacking_fa2_forward_w_4d_attn_mask()
        
        model = super(HybridModel, cls).build(model_args, train_args, data_args, **hf_kwargs)
        
        if model_args.use_sparse_linear_projector:
            vocab_size, hidden_dim = model.lm_p.get_input_embeddings().weight.shape

            init_lm_head = get_lm_head(model.lm_p)
            init_weight = init_lm_head.weight if isinstance(init_lm_head, nn.Linear) else None
            model.spr_pooler_p = SparseLinearProjector.build(hidden_dim, vocab_size, model_dir=model_args.model_name_or_path_psg, init_weight=init_weight)

            if model_args.untie_encoder:
                init_lm_head = get_lm_head(model.lm_q)
                init_weight = init_lm_head.weight if isinstance(init_lm_head, nn.Linear) else None
                model.spr_pooler_q = SparseLinearProjector.build(hidden_dim, vocab_size, model_dir=model_args.model_name_or_path_qry, init_weight=init_weight)
            else:
                model.spr_pooler_q = model.spr_pooler_p
        
        if model_args.use_sparse_down_projector:
            vocab_size, hidden_dim = model.lm_p.get_input_embeddings().weight.shape

            init_lm_head = get_lm_head(model.lm_p)
            init_weight = init_lm_head.weight if isinstance(init_lm_head, nn.Linear) else None
            model.spr_pooler_p = SparseDownProjector.build(vocab_size=vocab_size, hidden_dim=hidden_dim, model_dir=model_args.model_name_or_path_psg, init_weight=init_weight)

            if model_args.untie_encoder:
                init_lm_head = get_lm_head(model.lm_q)
                init_weight = init_lm_head.weight if isinstance(init_lm_head, nn.Linear) else None
                model.spr_pooler_q = SparseDownProjector.build(vocab_size=vocab_size, hidden_dim=hidden_dim, model_dir=model_args.model_name_or_path_qry, init_weight=init_weight)
            else:
                model.spr_pooler_q = model.spr_pooler_p
        
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
        # Resume model_args if not set
        if model_args is None:
            model_args = cls._load_model_args(model_name_or_path)
        
        # Transformers cls
        if model_args.hybrid_model_architecture == 'gpt':
            cls.TRANSFORMER_CLS = AutoModelForCausalLM
        elif model_args.hybrid_model_architecture == 'bert':
            cls.TRANSFORMER_CLS = AutoModelForMaskedLM
        else:
            raise NotImplementedError(f"Unsupported hybrid model architecture {model_args.hybrid_model_architecture}")

        model = super(HybridModel, cls).load(model_name_or_path, model_args, train_args, data_args, **hf_kwargs)
        
        if model_args.use_sparse_linear_projector:
            if model_args.untie_encoder:
                model.spr_pooler_p = SparseLinearProjector.load(os.path.join(model_name_or_path, 'passage_model')).to(model.lm_p.device)
                model.spr_pooler_q = SparseLinearProjector.load(os.path.join(model_name_or_path, 'query_model')).to(model.lm_q.device)
            else:
                model.spr_pooler_p = SparseLinearProjector.load(model_name_or_path).to(model.lm_p.device)
                model.spr_pooler_q = model.spr_pooler_p
        
        if model_args.use_sparse_down_projector:
            vocab_size, hidden_dim = model.lm_p.get_input_embeddings().weight.shape
            if model_args.untie_encoder:
                model.spr_pooler_p = SparseDownProjector.load(os.path.join(model_name_or_path, 'passage_model'), vocab_size=vocab_size).to(model.lm_p.device)
                model.spr_pooler_q = SparseDownProjector.load(os.path.join(model_name_or_path, 'query_model'), vocab_size=vocab_size).to(model.lm_q.device)
            else:
                model.spr_pooler_p = SparseDownProjector.load(model_name_or_path, vocab_size=vocab_size).to(model.lm_p.device)
                model.spr_pooler_q = model.spr_pooler_p
        
        return model
    
    def save(self, output_dir: str, state_dict: dict[str, any]=None, **hf_kwargs):
        super(HybridModel, self).save(output_dir, state_dict, **hf_kwargs)

        if self.spr_pooler_p is not None:
            if self.model_args.untie_encoder:
                self.spr_pooler_p.save_pooler(os.path.join(output_dir, 'passage_model'), state_dict=self._get_prefix_dict(state_dict, 'spr_pooler_p.'))
                self.spr_pooler_q.save_pooler(os.path.join(output_dir, 'query_model'), state_dict=self._get_prefix_dict(state_dict, 'spr_pooler_q.'))
            else:
                self.spr_pooler_p.save_pooler(output_dir, state_dict=self._get_prefix_dict(state_dict, 'spr_pooler_p.'))
    
    @staticmethod
    def flops(inputs: Tensor):
        """ Additional loss to control the sparsity of inputs
            http://arxiv.org/abs/2004.05665
            https://github.com/naver/splade/blob/main/splade/losses/regularization.py#L24

            Args:
                inputs (Tensor): Shape of [batch_size, sparse_dim]

            Notes:
                FLOPs is batch-related, requires carefully tuning.
        """
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)
    
    @staticmethod
    def norm_loss(inputs: Tensor, ord: int=0):
        normed_inputs = torch.linalg.vector_norm(torch.abs(inputs), ord=ord, dim=-1)
        return torch.mean(normed_inputs)

