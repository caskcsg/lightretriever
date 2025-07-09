#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Base training arguments shared by retriever & reranker fine-tuning.

@Time    :   2024/12/31
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import json
from typing import Optional
from dataclasses import dataclass, field, asdict

from transformers import TrainingArguments

import logging
logger = logging.getLogger(__name__)

@dataclass
class DomainConfig:
    epoch: dict[str, float] | None = field(
        default=None, metadata={"help": "Domain name -> reference epoch."}
    )
    domain_weights: dict[str, float] | None = field(
        default=None, metadata={"help": "Domain name -> domain weights."}
    )
    n_groups: int | None = field(
        default=None, metadata={"help": "Number of domains / groups."}
    )
    domain_ids: dict[str, int] | None = field(
        default=None, metadata={"help": "Domain name -> domain id."}
    )
    size: dict[str, int] | None = field(
        default=None, metadata={"help": "Domain name -> Number of item/lines in this domain."}
    )
    category_list: dict[str, list[str]] | None = field(
        default=None, metadata={"help": "Category name -> List of domain names. This is optionally used to group the domain from the same category"}
    )
    ref_length: int | None = field(
        default=None, metadata={"help": "Reference length of dataset for one epoch."}
    )

    # Set some alias
    @property
    def domain_to_idx(self):
        return self.domain_ids
    
    @property
    def domain_size(self):
        return self.size
    
    def __post_init__(self):
        if (self.domain_weights is not None) and (self.n_groups is None):
            self.n_groups = len(self.domain_weights)
    
    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class BaseDataArguments:
    """
    Base Arguments for what data we are going to input our model for training and eval.
    """
    # Negative hyperparameters
    train_n_passages: int = field(default=8)
    eval_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    # Interleavable datasets
    domain_config_path: Optional[str] = field(
        default=None, metadata={"help": "Path to json format domain config."}
    )
    preprocessed_dir: Optional[str] = field(
        default=None, metadata={"help": "Root folder path of all processed domains."}
    )
    add_domain_id: bool = field(
        default=True, metadata={"help": "Add domain index."}
    )
    add_domain_name: bool = field(
        default=True, metadata={"help": "Add domain name. This helps to add instruct to query."}
    )
    add_prompt: bool = field(
        default=False, metadata={"help": "Add prompt for training. Please explicitly setting this to true if you want to add prompt to the query side."}
    )
    add_prompt_prob: float = field(
        default=1.0, metadata={"help": "Probality to add prompt to query, range (0, 1]."}
    )
    prompt_type: str = field(
        default="e5", metadata={"help": "Choosing among 'e5', 'instructor', 'bge', 'default_reranker', 'e5_reranker', 'instructor_reranker'."}
    )
    append_prompt_sep: bool = field(
        default=False, metadata={"help": "Add a sep token [SEP] after the prompt."}
    )
    encoding_task_name: Optional[str] = field(
        default=None, metadata={"help": "Task name used during the inference."}
    )
    stopping_strategy: str = field(
        default="all_exhausted", metadata={"help": "Set to 'first_exhausted' for less sampling "
                                "or 'all_exhausted' for oversampling."
                                "See `datasets.interleave_datasets`"}
    )

    homogenous_batch: bool = field(
        default=False, metadata={"help": "Yeilds a homogenous batch from one dataset at each iteration."}
    )

    domain_config: DomainConfig | None = field(
        default=None, metadata={"help": "Domain config init from `domain_config_path`."}
    )

    def __post_init__(self):
        # Load domain weights from local file if posible
        if (self.domain_config_path is not None) and (self.domain_config is None):
            with open(self.domain_config_path, 'r') as f:
                domain_config: dict = json.load(f)
                self.domain_config = DomainConfig(**domain_config)
        
        # Warn when not using `add_domain_name`
        if not self.add_domain_name:
            logger.warning("`add_domain_name` is disabled, training with clusterining or classification tasks will not disable their coresponding in-batch / cross-batch negatives. If you are training on these tasks, please set this argument to `True`.")
        
        # Check add prompt settings
        if (not self.add_prompt) and (self.add_prompt_prob > 0):
            logger.warning(f"Setting add_prompt_prob to -1. If you want to finetune with query prompt, please enable `--add_prompt`.")
            self.add_prompt_prob = -1.0
    
    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class BaseModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Model Args
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
                    "This will override both `model_name_or_path_qry` and `model_name_or_path_psg`."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model archeticture used in training."
                    "Choose among ['EncoderModel']."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=8,
        metadata={"help": "Pad the embedding dimension to `pad_to_multiple_of` when resizing the embedding."},
    )

    # Tokenizer Args: Lowercase / Special Tokens
    edit_tokenizer_normalizers: bool = field(
        default=False,
        metadata={
            "help": "Allow editing normalizers. Enable this to make `lowercase` work."
        },
    )
    lowercase: bool = field(
        default=False,
        metadata={
            "help": "Whether to lower case all inputs during training or inferencing."
        },
    )
    edit_tokenizer_post_processor: bool = field(
        default=False,
        metadata={
            "help": "Allow editing post-processor. Enable this to make all `TemplateProcessing` work."
        },
    )
    add_bos_num: int = field(
        default=-1,
        metadata={
            "help": "How many <|bos|> do we want to prepend at the begin of text."
        }
    )
    add_eos_num: int = field(
        default=-1,
        metadata={
            "help": "How many <|eos|> do we want to append at the end of text."
        }
    )
    add_pooling_token_num: int = field(
        default=-1,
        metadata={
            "help": "How many <|pooling_token_x|> do we want to append at the end of text."
        }
    )
    add_pad_token: bool = field(
        default=True,
        metadata={
            "help": "Whether to add a pad token. Default: `True`."
        },
    )
    pad_token: str = field(
        default="<|pad|>",
        metadata={
            "help": "The pad token to add. Default `<|pad|>`, but this will potentially enlarge the vocab size, "
                    "and therefore need to resize the embedding layer and save them properly. Choosing a preserved "
                    "token from tokenizer's vocab will avoid enlarging the size of embedding layer. Recommendations: \n"
                    "1. `Qwen`: `<|im_end|>` \n"
                    "2. `Llama3`: `<|reserved_special_token_0|>` \n"
                    "3. `Mistral0.1`: `<unk>` \n"
                    "4. `Mistral0.3`: `[control_8]` \n"
                    "5. `Gemma`: `<|pad|>` \n"
        },
    )
    add_sep_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to add a sep token. Default: `False`."
        },
    )
    sep_token: str = field(
        default="<|sep|>",
        metadata={
            "help": "The sep token to add. Default `<|sep|>`, but this will potentially enlarge the vocab size, "
                    "and therefore need to resize the embedding layer and save them properly. Choosing a preserved "
                    "token from tokenizer's vocab will avoid enlarging the size of embedding layer. Recommendations: \n"
                    "1. `Qwen`: `<|im_start|>` \n"
                    "2. `Llama`: `<|reserved_special_token_1|>` \n"
                    "3. `Mistral0.1`: `<s>` \n"
                    "4. `Mistral0.3`: `[/INST]` \n"
                    "5. `Gemma`: `<bos>` \n"
        },
    )

    # Optimization Args
    ## Sequence Packing
    cumulative_seq: bool = field(
        default=False, 
        metadata={
            "help": "Whether to use automatic cumulative sequences. Cumulative sequence removes all pad tokens from "
                    "original inputs, and stride all other tokens within seq_len dimension. This is very useful to "
                    "decrease memory usages and speed up training. Flash attention is mandatory during the model forward."
        }
    )
    ## Bidirectional Attention for GPT
    enable_bidirectional_attention: bool = field(
        default=False, 
        metadata={
            "help": "Whether to enable bi-directional attention for Auto-Regression models."
        }
    )
    ## Liger Kernel
    liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable the Liger Kernel for model training."},
    )

    def __post_init__(self):
        if self.model_name_or_path is not None:
            # Set some pre-defined <|pad|> / <|sep|> tokens according to model_name
            if self.pad_token == "<|pad|>":     # is default
                # Change default value
                if "qwen" in self.model_name_or_path.lower():
                    self.pad_token = "<|im_end|>"
                elif "llama" in self.model_name_or_path.lower():
                    self.pad_token = "<|reserved_special_token_0|>"
                elif "mistral-7b-v0.1" in self.model_name_or_path.lower():
                    self.pad_token = "<unk>"
                elif "mistral-7b-v0.3" in self.model_name_or_path.lower():
                    self.pad_token = "[control_8]"
                elif "gemma" in self.model_name_or_path.lower():
                    self.pad_token = "<|pad|>"
            
            if self.sep_token == "<|sep|>":     # is default
                # Change default value
                if "qwen" in self.model_name_or_path.lower():
                    self.sep_token = "<|im_start|>"
                elif "llama" in self.model_name_or_path.lower():
                    self.sep_token = "<|reserved_special_token_1|>"
                elif "mistral-7b-v0.1" in self.model_name_or_path.lower():
                    self.sep_token = "<s>"
                elif "mistral-7b-v0.3" in self.model_name_or_path.lower():
                    self.sep_token = "[/INST]"
                elif "gemma" in self.model_name_or_path.lower():
                    self.sep_token = "<bos>"
    
    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class BaseTrainingArguments(TrainingArguments):
    # Cross-Entropy
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature scale for bi-encoder clloss and reranker celoss."}
    )

    # Trainer Related
    dataloader_drop_last: bool = field(
        default=True, 
        metadata={
            "help": "Drop the last incomplete batch if it is not divisible by the batch size."
                    "This is mendatory for embedding training."
        }
    )
    min_lr_ratio: float = field(default=0.0)
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "Choose among `flash_attention_2`, `sdpa`, `eager`."}
    )
    logging_path: Optional[str] = field(
        default=None, metadata={"help": "Path for redirecting Transformers logs to local file."}
    )

    # Peft Config
    lora: bool = field(default=False, metadata={"help": "Use LoRA in Fine-tuning."})
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension (the \"rank\")."})
    lora_alpha: int = field(default=32, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout probability for Lora layers."})

    def __post_init__(self):
        super().__post_init__()

        if self.resume_from_checkpoint is not None:
            if self.resume_from_checkpoint.lower() in ["false", 'f']:
                self.resume_from_checkpoint = None
            elif self.resume_from_checkpoint.lower() in ["true", 't']:
                self.resume_from_checkpoint = True
