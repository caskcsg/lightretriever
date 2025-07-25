#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training arguments.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''
from typing import Optional
from dataclasses import dataclass, field

from ..arguments import (
    BaseDataArguments, 
    BaseModelArguments, 
    BaseTrainingArguments as RerankerTrainingArguments, 
    DomainConfig
)

@dataclass
class DataArguments(BaseDataArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # Max sequence lengths
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
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
    # TODO: `dataset_name`, `corpus_dir` & `corpus_path` are rarely used because of `Interleavable datasets`
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    corpus_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to train triples directory"}
    )
    corpus_path: Optional[str] = field(
        default=None, metadata={"help": "Path to train triples data"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    # Encoding Args: For Encode of Reranking task
    tsv_ranks_path: Optional[str] = field(
        default=None,  metadata={"help": "Path to .ranks.tsv generated by dual-encoder, format [qid, pid, score]"}
    )
    query_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    passage_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    reranking_depth:  int = field(
        default=1000,
        metadata={
            "help": "Reranking depth, will only re-score `top-depth` per qid using CrossEncoder"
        },
    )
    rerank_save_path: str = field(default=None, metadata={"help": "Output tsv path to save the reranked results"})


@dataclass
class ModelArguments(BaseModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    cross_encoder_type: str = field(
        default="CrossEncoder",
        metadata={
            "help": "`CrossEncoder` uses AutoModelForSequenceClassification by down-project the last hidden states to a number as reranker score; `CrossEncoderLogits` first up-project the last hidden states to vocab, then extract the identifier token id as reranker score."
        },
    )
    sigmoid_normalize: bool = field(
        default=False,
        metadata={
            "help": "Whether to normalize the output to range (0, 1) with `nn.Sigmoid()`."
        },
    )
