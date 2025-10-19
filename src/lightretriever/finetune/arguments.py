#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training arguments.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
from typing import Optional
from dataclasses import dataclass, field

from ..arguments import (
    BaseDataArguments, 
    BaseModelArguments, 
    BaseTrainingArguments, 
    DomainConfig
)

@dataclass
class DataArguments(BaseDataArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # Max sequence lengths
    q_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    # Single Dataset Path
    # TODO: `query_collection`, `passage_collection` & `corpus_path` are rarely used because of `Interleavable datasets`
    query_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    passage_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    corpus_path: Optional[str] = field(
        default=None, metadata={"help": "Path to train triples / encode corpus data"}
    )
    dev_path: Optional[str] = field(
        default=None, metadata={"help": "Path to development triples, the same format as training negative triples."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # Encoding Args
    # TODO: These args are going to be deprecated because of EvalArguments for RPC Encode.
    qrel_path: Optional[str] = field(
        default=None, metadata={"help": "Path to qrels for filtering out queries to encode."}
    )
    encoded_save_prefix: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)


@dataclass
class ModelArguments(BaseModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Retriever Args
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    pooling_strategy: str = field(
        default=None,
        metadata={
            "help": "Pooling strategy. Choose between mean/cls/lasttoken."
        },
    )
    score_function: str = field(
        default="cos_sim",
        metadata={
            "help": "Pooling strategy. Choose between dot/cos_sim."
        },
    )
    normalize: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to l2 normalize the representation. This feature is controlled by `score_function`."
                    "`score_function==dot`: `normalize=False`."
                    "`score_function==cos_sim`: `normalize=True`."
        },
    )
    
    # Dense dimension shrink
    dense_shrink_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Shrink the dimension of dense reps to `dim` by calling `dense_reps[..., :dense_shrink_dim]`. "
                    "This affect both training and inferencing. Commonly, if we want to train with MRL, we should "
                    "set `train_args.matryoshka_dims` for HybridModel, do not use this arg. Then if we want "
                    " to inference with a dim, we can use this `model_args.dense_shrink_dim`."
        },
    )

    # MLP Pooler
    add_pooler: bool = field(
        default=False,
        metadata={"help": "Add a MLP layer on top of pooled embedding."}
    )
    projection_out_dim: int = field(
        default=None,
        metadata={"help": "MLP Fan-out."}
    )
    
    # Indivisual settings
    model_name_or_path_qry: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
                    "Will be overriden if `model_name_or_path` is set."
        },
    )
    model_name_or_path_psg: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
                    "Will be overriden if `model_name_or_path` is set."
        },
    )
    pooling_strategy_qry: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pooling strategy of query model. Choose between mean/cls/lasttoken. Will be overriden by `pooling_strategy`"
        },
    )
    pooling_strategy_psg: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pooling strategy of passage model. Choose between mean/cls/lasttoken. Will be overriden by `pooling_strategy`"
        },
    )
    projection_out_dim_qry: Optional[int] = field(
        default=None,
        metadata={"help": "MLP Fan-out. Will be overrided if `projection_out_dim` is set."}
    )
    projection_out_dim_psg: Optional[int] = field(
        default=None,
        metadata={"help": "MLP Fan-out. Will be overrided if `projection_out_dim` is set."}
    )

    # *** Hybrid Model ***
    # ==> Transformers CLS
    hybrid_model_architecture: str = field(
        default='gpt',
        metadata={
            "help": "Model architecture used by Hybrid Model. `gpt` will use AutoModelForCausalLM; `bert` will use AutoModelForMaskedLM."
        },
    )

    # ==> Vector Types
    hybrid_use_dense_vector: bool = field(
        default=False,
        metadata={"help": "Train & Encode using dense vector."}
    )
    hybrid_use_sparse_vector: bool = field(
        default=False,
        metadata={"help": "Train & Encode using sparse vector."}
    )
    hybrid_use_emb_vector: bool = field(
        default=False,
        metadata={"help": "Train & Encode using averaged Embedding layer output on query side. Note that when this is enabled, passage encoder will also encode dense vector even if `hybrid_use_dense_vector` is False."}
    )
    hybrid_use_token_id_vector: bool = field(
        default=False,
        metadata={"help": "Train & Encode using token_id vector on query side. Note that when this is enabled, passage encoder will also encode sparse vector even if `hybrid_use_sparse_vector` is False."}
    )

    # ==> Emb vec
    noncontextual_query_embedding: bool = field(
        default=False,
        metadata={"help": "Train & Encode `dense query reps` using `Non-contextual query embedding`, which is a mean-pooled (prompted)-query eos token embedding. Query tokens are indivisually encoded (with optional prompt). Then query sentence embedding is aggregated by mean pooling the coresponding eos token ids."}
    )
    noncontextual_prompt_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Optional string to prepend in front of each prompts of Non-contextual query instructions."}
    )

    # ==> Token id rep
    token_id_vector_type: str = field(
        default="sum",
        metadata={
            "help": "Aggregation method of tokenized query input ids. Choose among ['bow', 'sum']. "
                    "1. `bow`: Directly use set(input_ids) as query's token id vector. `tok -> 1` "
                    "2. `sum`: Use number of each token as query's token id vector. `tok -> # of this tok` "
        }
    )

    # ==> Pooling then projection / Projection then Aggregation
    sparse_pooling_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pooling strategy for `xth-to-last token pooling`. Choice ['lasttoken', 'second_to_last'] "
                    "Default: `None`. This will not perform pooling, but use aggregation instead."
        }
    )
    sparse_use_max_aggregation: bool = field(
        default=True,
        metadata={
            "help": "Whether to use efficient max function for pooling sparse vectors."
                    "If False, will use an **inefficient** mean pooling."
                    "Effective only when `sparse_pooling_strategy == None`."
        }
    )

    # ==> Sparsify methods
    use_icu_word_pretokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use ICUWordPreTokenizer to pretokenize the texts for getting `unique tokens`. "
                    "If `False`, `unique tokens` will be obtained from tokenizing `whitespace + text`. "
                    "Note: 1. ICUWordPreTokenizer removes all whitespaces by default. "
                    "      2. Almost every GPT BPE tokenizer merge the space before a certain word to a single token. "
                    "      To avoid lexical mismatch of the first token, we are tokenizing `whitespace + text` to "
                    "      get `unique tokens`."
        }
    )
    sparse_pool_from_unique_token_ids: bool = field(
        default=False,
        metadata={"help": "Whether to sparsify logits with unique token ids obtained from ICUWordPreTokenizer."}
    )
    sparse_pool_from_original_input_ids_qry: bool = field(
        default=False,
        metadata={"help": "Whether to pool the sparse query embedding only from token appear in the input ids."}
    )
    sparse_pool_from_original_input_ids_psg: bool = field(
        default=False,
        metadata={"help": "Whether to pool the sparse passage embedding only from token appear in the input ids."}
    )
    sparse_min_tokens_to_keep: int = field(
        default=8,
        metadata={"help": "Min tokens to keep for top-p / top-k sampling."}
    )
    sparse_remove_stopwords: bool = field(
        default=False,
        metadata={"help": "Whether to remove stopwords with ICUWordPreTokenizer."}
    )

    # ==> Postprocess methods
    sparse_use_relu: bool = field(
        default=False,
        metadata={"help": "Whether to use relu for pooling sparse vectors"}
    )
    sparse_use_log_saturation: bool = field(
        default=False,
        metadata={"help": "Whether to use log saturation for pooling sparse vectors"}
    )

    sparse_top_p_qry: float = field(
        default=1.0,
        metadata={"help": "Top-p of nucleus sampling, range [0, 1]."}
    )
    sparse_top_p_psg: float = field(
        default=1.0,
        metadata={"help": "Top-p of nucleus sampling, range [0, 1]."}
    )
    sparse_top_k_qry: int = field(
        default=0,
        metadata={
            "help": "Top-k of nucleus sampling, 0 to disable."
                    "Please hard limit <1k of query topk. Avoid error: https://github.com/castorini/anserini/issues/745"
        }
    )
    sparse_top_k_psg: int = field(
        default=0,
        metadata={"help": "Top-k of nucleus sampling, 0 to disable."}
    )

    # ==> SparseLinearProjector
    use_sparse_linear_projector: bool = field(
        default=False,
        metadata={
            "help": "SparseLinearProjector project a hidden state to vocab dimension with only a linear layer."
        }
    )

    # ==> SparseDownProjector
    use_sparse_down_projector: bool = field(
        default=False,
        metadata={
            "help": "A Down Projector which map `hidden_states` to `a float` number "
                    " This is used for reproducing BGE-m3 sparse retrieval."
        }
    )


    def __post_init__(self):
        super().__post_init__()
        if self.score_function == "dot":
            self.normalize = False
        elif self.score_function == "cos_sim":
            self.normalize = True
        else:
            raise ValueError(f"The score function is {self.score_function}. This is not supported yet.")
        
        if self.model_name_or_path:
            self.model_name_or_path_qry = self.model_name_or_path
            self.model_name_or_path_psg = self.model_name_or_path
            if self.untie_encoder:
                _qry_model_path = os.path.join(self.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(self.model_name_or_path, 'passage_model')
                if os.path.exists(_qry_model_path) and os.path.exists(_psg_model_path):
                    self.model_name_or_path_qry = _qry_model_path
                    self.model_name_or_path_psg = _psg_model_path
        
        if self.pooling_strategy:
            self.pooling_strategy_qry = self.pooling_strategy
            self.pooling_strategy_psg = self.pooling_strategy
        
        if self.projection_out_dim:
            self.projection_out_dim_qry = self.projection_out_dim
            self.projection_out_dim_psg = self.projection_out_dim

@dataclass
class RetrieverTrainingArguments(BaseTrainingArguments):
    # Model Implementation Related
    clloss_coef: float = field(
        default=1.0,
        metadata={"help": "Scale factor for clloss."}
    )
    distillation: bool = field(
        default=False,
        metadata={"help": "KL loss between Retriever query-passage scores and CrossEncoder scores."}
    )
    distill_coef: float = field(
        default=1.0,
        metadata={"help": "Scale factor for distill loss."}
    )
    distill_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for distill loss."}
    )
    loss_reduction: str = field(
        default='mean', metadata={"help": "Loss reduction of CrossEntropy Loss. Choose among `mean`, `none`."}
    )
    negatives_x_device: bool = field(
        default=False, 
        metadata={
            "help": "Share in-batch negatives across global ranks."
        }
    )

    ## Hybrid Model
    sparse_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature scale for sparse clloss."}
    )
    add_flops: bool = field(default=False)
    add_vector_norm: bool = field(default=False)
    norm_ord: int = field(default=1)
    q_norm_loss_factor: float = field(default=1.0)
    p_norm_loss_factor: float = field(default=1.0)

    # ** Regulation Factor Scheduler **
    use_reg_scheduler: bool = field(
        default=False,
        metadata={
            "help": "Add a regulation scheduler to quadratic increase regulator until time T."
        }
    )
    reg_t: int = field(
        default=2000,
        metadata={
            "help": "Increase regulator factor from 0 to `reg_t`."
        }
    )
    reg_max: int = field(
        default=4000,
        metadata={
            "help": "Decrease regulator factor from `reg_t` to `reg_max`. Only valid for `linear_decay`."
        }
    )
    min_reg_ratio: float = field(
        default=0.,
        metadata={
            "help": "Min ratio of decayed regulator factor."
        }
    )
    reg_type: str = field(
        default="quadratic",
        metadata={
            "help": "Choose among ['quadratic', 'linear_decay']."
        }
    )

    # ** Imbalanced Dense Embedding **
    emb_den_reps_distillation: bool = field(
        default=False,
        metadata={"help": "KL loss between Query Emb Reps & Query Den Reps."}
    )
    emb_den_scores_distillation: bool = field(
        default=False,
        metadata={"help": "KL loss between `Query Emb`-`Psg` Scores & `Query Den`-`Psg` Scores."}
    )
    emb_reps_distill_coef: float = field(
        default=1.0,
        metadata={"help": "Scale factor for Query Emb Reps distill loss."}
    )

    # ** Imbalanced Token id Embedding **
    tok_den_scores_distillation: bool = field(
        default=False,
        metadata={"help": "KL loss between `Query Token id`-`Psg` Scores & `Query Den`-`Psg` Scores."}
    )
    tok_reps_distill_coef: float = field(
        default=1.0,
        metadata={"help": "Scale factor for Query Token id Reps distill loss."}
    )

    # ** Matryoshka Representation Learning during Training Stage **
    matryoshka_dims: list[Optional[int]] = field(
        default_factory=lambda: [None],
        metadata={
            "help": "Dimensions of Matryoshka Representation Learning."
                    "Default is `[None]`, which means do not use Matryoshka Representation for training."
        }
    )
    # matryoshka_weights: Optional[list[float]] = field(
    #     default=None,
    #     metadata={"help": "Loss weights of Matryoshka Representation Learning."}
    # )

    # *** GradCache ***
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=32)
    gc_p_chunk_size: int = field(default=4)
    gc_dynamic_chunking: bool = field(default=False, metadata={"help": "Dynamic adjust the chunk size of GradCache."})
    gc_anchor_chunk_size: int = field(
        default=4,
        metadata={
            "help": "Anchor chunk size for dynamic adjust the chunk size of GradCache."
        }
    )
    gc_anchor_seqlen: int = field(
        default=512,
        metadata={
            "help": "Anchor max sequence length for dynamic adjust the chunk size of GradCache."
                    "The actual max sequence length will be adjusted by the following methods dynamiclly:"
                    "current chunk size = gc_anchor_chunk_size * gc_anchor_seqlen^2 / current_seq_len^2"
        }
    )
    no_sync_except_last: bool = field(
        default=False, 
        metadata={
            "help": "Whether to disable grad sync for GradCache accumulation backwards."
                    "This helps reduces the communication overhead of accumulated backwards."
                    "But it can induce more GPU memory usage for FSDP."
                    "Also, Deepspeed is not compatiable with this behavior."
        }
    )

