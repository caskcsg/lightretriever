#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Inferencing arguments.

@Time    :   2025/01/02
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import logging
from typing import Optional, Union
from dataclasses import dataclass, field

from ..finetune.arguments import ModelArguments as RetrieverModelArguments

import torch
logger = logging.getLogger(__name__)

@dataclass
class InferenceArguments(RetrieverModelArguments):
    """
    Inference arguments.
    """
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for evaluation."}
    )
    model_type: Optional[str] = field(
        default="EncoderModel",
        metadata={
            "help": "The model archeticture used in training."
                    "Choose among ['EncoderModel', 'ImbalancedEncoderModel', 'RerankerModel']."
        },
    )
    inference_arch: str = field(
        default="PytorchRPCExactSearchModel",
        metadata={
            "help": "The model archeticture used in training. "
                    "Choose among [ "
                    "    'PytorchRPCExactSearchModel' (RPC + Hybrid), "
        },
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for encoding."}
    )
    append_prompt_sep: bool = field(
        default=False, metadata={"help": "Add a sep token [SEP] after the prompt."}
    )

    # Data args
    q_max_len: int = field(
        default=128, metadata={"help": "Query maxlen."}
    )
    p_max_len: int = field(
        default=512, metadata={"help": "Passage maxlen."}
    )
    max_length: int = field(
        default=1024, metadata={"help": "Reranker maxlen."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    padding: Union[bool, str] = field(
        default=True,
    )

    # Model args
    bf16: bool = field(
        default=False, metadata={"help": "Bfloat16."}
    )
    fp16: bool = field(
        default=False, metadata={"help": "Float16."}
    )
    seed: int = field(
        default=42, metadata={"help": "Seed."}
    )
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "Choose among `flash_attention_2`, `sdpa`, `eager`."}
    )

    # RPC Related
    local_rank: int = field(
        default=-1, metadata={"help": "Local rank initilized by `LOCAL_RANK`."}
    )
    rank: int = field(
        default=-1, metadata={"help": "Global rank initilized by `RANK`."}
    )
    world_size: int = field(
        default=0, metadata={"help": "World size initilized by `WORLD_SIZE`."}
    )
    master_addr: str = field(
        default="127.0.0.1", metadata={"help": "Master address."}
    )
    master_port: int = field(
        default=12345, metadata={"help": "Master port."}
    )
    debug: bool = field(
        default=False, metadata={"help": "Debug encoding function"}
    )

    # Anserini
    anserini_lang: Optional[str] = field(
        default=None, metadata={"help": "Analyzer language (ISO 3166 two-letter code)."}
    )
    anserini_vector_type: str = field(
        default="JsonVectorCollection", metadata={
            "help": "Choose amond `JsonVectorCollection`, `JsonCollection`."
                    " JsonVectorCollection (Impact Search only): corpus_emb is a list of dict {tok_id: freq} "
                    " JsonCollection (Both): corpus_emb is a list of str "
                    " tok_id1 tok_id1 tok_id2 ... "
        }
    )
    anserini_pretokenized: bool = field(
        default=True, metadata={"help": "Index/Search pre-tokenized collections without any additional stemming, stopword processing"}
    )
    anserini_impact_search: bool = field(
        default=True, metadata={"help": "Whether to perform impact search."}
    )
    anserini_bm25_k1: float = field(
        default=0.9, metadata={"help": "BM25 k1."}
    )
    anserini_bm25_b: float = field(
        default=0.4, metadata={"help": "BM25 b."}
    )

    # EmbeddingBag
    eval_batch_size_embedding_bag: int = field(
        default=5000, metadata={"help": "Batch size for encoding."}
    )

    def __post_init__(self):
        super().__post_init__()

        if self.pad_to_max_length:
            self.padding = "max_length"
        
        # Init dist args
        if local_rank := os.getenv("LOCAL_RANK", None):
            self.local_rank = int(local_rank)
        if rank := os.getenv("RANK", None):
            self.rank = int(rank)
        if world_size := os.getenv("WORLD_SIZE", None):
            self.world_size = int(world_size)
        if master_addr := os.getenv("MASTER_ADDR", None):
            self.master_addr = str(master_addr)
        if master_port := os.getenv("MASTER_PORT", None):
            self.master_port = str(master_port)

        # Parse dtype
        self.dtype = None
        if self.bf16:
            self.dtype = torch.bfloat16
        elif self.fp16:
            self.dtype = torch.float16
