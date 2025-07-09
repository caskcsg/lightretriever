#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
AutoModelForSequenceClassification Wrapper Class for Reranker Training.

@Time    :   2024/06/19
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    BatchEncoding,
    PretrainedConfig,
    BertForSequenceClassification,
    XLMRobertaForSequenceClassification,
    GPTNeoXForSequenceClassification,
    BertForMaskedLM,
    XLMRobertaForCausalLM,
    GPTNeoXForCausalLM,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from peft.utils import CONFIG_NAME as PEFT_CONFIG_NAME

from ..finetune.modeling_encoder import pooling
from ..utils.data_utils import load_tokenizer, resize_emb
from ..utils.monkey_patch import apply_bidirectional_attention
from ..utils.nested_input import apply_seqlen_cumulate
from .arguments import ModelArguments, DataArguments, RerankerTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)

def get_base_model(model: PreTrainedModel):
    """ Get Base Model from MLM/CLM Model Classes """
    if isinstance(model, PeftModel):
        model_unwraped = model.get_base_model()
    else:
        model_unwraped = model
    
    if isinstance(model_unwraped, (BertForSequenceClassification, BertForMaskedLM)):
        return model_unwraped.bert
    elif isinstance(model_unwraped, (XLMRobertaForSequenceClassification, XLMRobertaForCausalLM)):
        return model_unwraped.roberta
    elif isinstance(model_unwraped, (GPTNeoXForSequenceClassification, GPTNeoXForCausalLM)):
        return model_unwraped.gpt_neox
    elif ("ForSequenceClassification" in model_unwraped.__class__.__name__) or ("ForCausalLM" in model_unwraped.__class__.__name__):
        return model_unwraped.model
    else:
        raise NotImplementedError(f"Unrecognized type of model_unwraped {type(model_unwraped)}.")

def get_lm_head(model: PreTrainedModel):
    """ Get LM Head from MLM/CLM Model Classes """
    if isinstance(model, PeftModel):
        model_unwraped = model.get_base_model()
    else:
        model_unwraped = model
    
    if isinstance(model_unwraped, BertForMaskedLM):
        return model_unwraped.cls
    elif isinstance(model_unwraped, XLMRobertaForCausalLM):
        return model_unwraped.lm_head
    elif isinstance(model_unwraped, GPTNeoXForCausalLM):
        return model_unwraped.embed_out
    elif "ForCausalLM" in model_unwraped.__class__.__name__:
        return model_unwraped.lm_head
    else:
        raise NotImplementedError(f"Unrecognized type of model_unwraped {type(model_unwraped)}.")

@dataclass
class SequenceClassifierOutputWithLogs(SequenceClassifierOutput):
    logs: Optional[Dict[str, any]] = None

class CrossEncoder(nn.Module):
    """
    A simple warpper for `AutoModelForSequenceClassification` for reranker training
    """
    TRANSFORMER_CLS = AutoModelForSequenceClassification
    is_gradient_checkpointing = False
    PEFT_TASK_TYPE = TaskType.SEQ_CLS

    def __init__(
        self, 
        lm: Union[PreTrainedModel, PeftModel],
        model_args: ModelArguments, 
        data_args: Optional[DataArguments],
        training_args: Optional[TrainingArguments]
    ):
        super().__init__()
        self.lm = lm
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if self.model_args.cumulative_seq:
            apply_seqlen_cumulate(get_base_model(self.lm))
        
        if self.model_args.enable_bidirectional_attention:
            apply_bidirectional_attention(self.lm)
        
        if self.model_args.liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            # Patch the model with liger kernels. Use the default kernel configurations.
            model_unwarped = get_base_model(self.lm)
            if isinstance(model_unwarped, PreTrainedModel):
                _apply_liger_kernel_to_instance(model=model_unwarped)
            else:
                logger.warning(
                    "The model is not an instance of PreTrainedModel/PeftModel. No liger kernels will be applied."
                )

        if training_args is not None and self.training:
            self.register_buffer(
                'target_label',
                torch.zeros(self.training_args.per_device_train_batch_size, dtype=torch.long),
                persistent=False,
            )
            
            try:
                from flash_attn.losses.cross_entropy import CrossEntropyLoss
                self.cross_entropy = CrossEntropyLoss(reduction="mean", inplace_backward=True)
                        
            except ImportError:
                logger.info(
                    "Optimized flash-attention CrossEntropyLoss not found (run `pip install git+https://github.com/Dao-AILab/flash-attention.git#egg=xentropy_cuda_lib&subdirectory=csrc/xentropy`)"
                )
                self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        
            # Compatiable with FSDP Auto Wrap
            self._no_split_modules: List[str] = getattr(lm, "_no_split_modules", []) or []
    
    def _call_lm(self, batch: BatchEncoding | dict[str, any], **kwargs):
        lm_out: SequenceClassifierOutput = self.lm(**batch, use_cache=False, return_dict=True)
        if self.model_args.sigmoid_normalize:
            lm_out.logits = F.sigmoid(lm_out.logits)
        return lm_out
    
    def forward(self, batch: BatchEncoding | dict[str, any], **kwargs):
        lm_out = self._call_lm(batch, **kwargs)

        if self.training:   # Training with Listwise Ranking Loss
            logits = lm_out.logits / self.training_args.temperature
            scores = logits.view(
                self.training_args.per_device_train_batch_size,
                self.data_args.train_n_passages
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutputWithLogs(
                loss=loss,
                logits=logits,
                hidden_states=lm_out.hidden_states,
                attentions=lm_out.attentions
            )
        else:   # Evaluating or Inferencing
            return SequenceClassifierOutputWithLogs(**lm_out)
    
    @staticmethod
    def _apply_gradient_checkpointing(model: PreTrainedModel, gradient_checkpointing_kwargs: dict = None):
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
        self._apply_gradient_checkpointing(self.lm.base_model, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    @staticmethod
    def load_tokenizer(
            model_name_or_path: str, 
            model_args: ModelArguments,
        ):
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
    def load_model(
        cls, 
        model_name_or_path: str,
        model_args: ModelArguments,
        data_args: Optional[DataArguments] = None,
        training_args: Optional[TrainingArguments] = None,
        **hf_kwargs
    ):
        # Load tokenizer for resizing embedding layer if necessary
        tokenizer = cls.load_tokenizer(model_name_or_path, model_args=model_args)
        if os.path.exists(os.path.join(model_name_or_path, PEFT_CONFIG_NAME)):
            logger.info(f"Peft config is found at {model_name_or_path}.")
            # Load Base HF Model & Peft Adapters
            config = LoraConfig.from_pretrained(model_name_or_path)
            base_model = cls.TRANSFORMER_CLS.from_pretrained(config.base_model_name_or_path, **hf_kwargs)
            # Resize base_model embedding layer if necessary
            resize_emb(base_model, tokenizer, pad_to_multiple_of=model_args.pad_to_multiple_of)
            hf_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=config, is_trainable=True, torch_device=hf_kwargs.get("device_map", None))
            hf_model: PreTrainedModel = hf_model.merge_and_unload()  # Merge to single HF Model
        else:
            hf_model: PreTrainedModel = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            # Resize hf_model embedding layer if necessary
            resize_emb(hf_model, tokenizer, pad_to_multiple_of=model_args.pad_to_multiple_of)
        
        if hf_model.config.pad_token_id != tokenizer.pad_token_id:
            hf_model.config.pad_token_id = tokenizer.pad_token_id

        if training_args is not None and training_args.lora:
            hf_model = cls._build_lora_model(hf_model, model_args, training_args)
        
        model = cls(hf_model, model_args, data_args, training_args)
        return model

    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str,
        model_args: ModelArguments,
        data_args: Optional[DataArguments] = None,
        training_args: Optional[TrainingArguments] = None,
        **hf_kwargs
    ):
        hf_kwargs["num_labels"] = 1
        return cls.load_model(model_name_or_path, model_args, data_args, training_args, **hf_kwargs)

    def save_pretrained(self, output_dir: str, state_dict: Dict[str, any]=None, **kwargs):
        if state_dict is not None:
            prefix = 'lm.'
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        
        self.lm.save_pretrained(output_dir, state_dict=state_dict, **kwargs)
    
    @classmethod
    def _build_lora_model(
        cls,
        base_model: PreTrainedModel, 
        model_args: ModelArguments, 
        training_args: TrainingArguments
    ):
        peft_config = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=cls.PEFT_TASK_TYPE,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules='all-linear',
            inference_mode=False
        )

        hf_model = get_peft_model(base_model, peft_config)
        return hf_model


class CrossEncoderLogits(CrossEncoder):
    """
    A simple warpper for `AutoModelForCausalLM` with logit probality pooling
    """
    TRANSFORMER_CLS = AutoModelForCausalLM
    PEFT_TASK_TYPE = TaskType.FEATURE_EXTRACTION

    def __init__(
        self, 
        lm: Union[PreTrainedModel, PeftModel],
        model_args: ModelArguments, 
        data_args: Optional[DataArguments],
        training_args: Optional[TrainingArguments]
    ):
        super().__init__(lm, model_args, data_args, training_args)
        tokenizer = self.load_tokenizer(model_args.model_name_or_path, model_args)
        self.identifier_token_id = tokenizer.encode("yes", add_special_tokens=False)[-1]
    
    def _call_lm(self, batch: BatchEncoding | dict[str, any], **kwargs):
        base_lm_out: BaseModelOutput = get_base_model(self.lm)(**batch, use_cache=False, return_dict=True)
        last_hidden_states = pooling(
            base_lm_out.last_hidden_state, # bs, seq_len, hidden_size
            attention_mask=batch["attention_mask"],
            pooling_strategy='lasttoken',
        )   # bs, hidden_size
        last_logits: torch.Tensor = get_lm_head(self.lm)(last_hidden_states)    # bs, vocab_size
        lm_out = SequenceClassifierOutput(logits=last_logits[..., self.identifier_token_id])    # bs
        if self.model_args.sigmoid_normalize:
            lm_out.logits = F.sigmoid(lm_out.logits)
        return lm_out

    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str,
        model_args: ModelArguments,
        data_args: Optional[DataArguments] = None,
        training_args: Optional[TrainingArguments] = None,
        **hf_kwargs
    ):
        return cls.load_model(model_name_or_path, model_args, data_args, training_args, **hf_kwargs)
