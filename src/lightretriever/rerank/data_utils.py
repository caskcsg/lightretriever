#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training datasets.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''
import random
from itertools import chain
from dataclasses import dataclass
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

import datasets
from transformers import BatchEncoding, DataCollatorWithPadding

from .arguments import DataArguments
from ..utils.data_utils import build_corpus_idx_to_row, read_corpus, process_tsv_file
from ..trainer import ContrastiveTrainer

import logging
logger = logging.getLogger(__name__)

@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize train dataset.
    """
    separator = " " # WhiteSpace separator between passage title & text
    
    def _get_query(self, item: Dict[str, str]) -> str:
        """ Format a Query for Reranker Training
            Args:
                item (Dict[str, str]): A dict of triples, format {
                    "query": str,
                    "positive_passages": List of {"title": str, "text": str}
                    "negative_passages": List of {"title": str, "text": str}
                }
        """
        query = item["query"]
        if "query_prompt" in item:
            query = item["query_prompt"] + query
        
        return query
    
    def _get_passages(self, item: Dict[str, Dict[str, str]]) -> List[str]:
        """ Format a Passage for Reranker Training
            Args:
                item (Dict[str, str]): A dict of triples, format {
                    "query": str,
                    "positive_passages": List of {"title": str, "text": str}
                    "negative_passages": List of {"title": str, "text": str}
                }
        """
        assert isinstance(item["positive_passages"], list) and isinstance(item["negative_passages"], list)
        assert len(item["positive_passages"]) == 1, f"Reranker Training needs 1 positive passage, but found {len(item['positive_passages'])}"

        all_psgs: List[str] = []
        for psg in chain(item["positive_passages"], item["negative_passages"]):
            if "title" in psg and psg["title"]:
                text = psg["title"] + self.separator + psg["text"]
            else:
                text = psg["text"]
            
            if "passage_prompt" in item:
                text = item["passage_prompt"] + text
            
            all_psgs.append(text)
        
        return all_psgs

    def __call__(self, features: List[dict]):
        text_pairs: List[List[str]] = list()
        for query, passages in zip(map(self._get_query, features), map(self._get_passages, features), strict=True):
            for passage in passages:
                text_pairs.append([query, passage])

        # Tokenize
        encoded: BatchEncoding = self.tokenizer(
            text_pairs,
            max_length=self.max_length,
            truncation='longest_first',
            padding=self.padding,
            add_special_tokens=True,
            return_tensors=self.return_tensors,
        )
        return {"batch": encoded}


@dataclass
class IterableTrainCollator(TrainCollator):
    """
    Copied from: llmemb/finetune/data_utils.py

    IterableTrainCollator for sample batch examples, processing & tokenize train dataset.

    Note:
    1. `query_prompt`: will be read from dataset["instruction"].
    2. `passage_prompt`: always be `'\nPassage: '`.
    """
    train_n_passages: int = 2
    seed: int = 42
    positive_passage_no_shuffle: bool = False
    negative_passage_no_shuffle: bool = False
    add_prompt_prob: float = -1.
    prompt_type: str = 'reranker_noinst'
    append_prompt_sep: bool = False

    def __post_init__(self):
        # super(IterableTrainCollator, self).__post_init__()
        self.rng = random.Random(self.seed)
    
    def __call__(self, group: List[dict]):
        return super(IterableTrainCollator, self).__call__(list(map(self.get_item, group)))
    
    def get_item(self, group: dict):
        # TODO: 1) self.seed should be added to idx; 2) No epoch involved
        # seed: int = int(group.get('_train_dataset_idx', self.seed))
        # rng = random.Random(seed)

        rng = self.rng

        # Sample One Positive
        group_positives = group['positive_passages']
        if self.positive_passage_no_shuffle:
            pos_psg: Dict[str, any] = group_positives[0]
        else:
            pos_psg: Dict[str, any] = rng.choice(group_positives)
        
        # Sample Negatives
        group_negatives = group['negative_passages']
        negative_size = self.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = rng.choices(group_negatives, k=negative_size)
        else:
            if self.train_n_passages == 1:
                negs = []
            elif self.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                negs = rng.sample(group_negatives, k=negative_size)
        
        rets = {
            "query": group["query"],
            "positive_passages": [pos_psg],
            "negative_passages": negs,
            "domain_name": group["domain_name"],
        }

        if "domain_ids" in group:
            rets["domain_ids"] = group["domain_ids"]
        
        if self.add_prompt_prob > 0 and self.add_prompt_prob <= 1:
            if self.add_prompt_prob >= 1.0 or rng.random() <= self.add_prompt_prob:    # Speed up when add_prompt_prob >= 1.0
                rets["query_prompt"] = group["instruction"]
                if self.append_prompt_sep:
                    rets["query_prompt"] += self.tokenizer.sep_token + " "      # `{prompt}{sep_token} {text}`
                rets["passage_prompt"] = "\nPassage: "

        return rets


# Single Dataset Implementation
# TODO: `TrainDataset` is rarely used because of `Interleavable datasets`
class TrainDataset(Dataset):
    """ Wrapper for Sampling Positive / Negative Passages """
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            trainer: ContrastiveTrainer = None,
    ):
        self.train_data = dataset
        self.trainer = trainer
        self.data_args = data_args

    def __len__(self):
        return len(self.train_data) 

    def __getitem__(self, item) -> Dict[str, any]:
        group = self.train_data[item]
        _hashed_seed = hash(item + self.trainer.args.seed)

        epoch = int(self.trainer.state.epoch)

        qry = group['query']

        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        return {
            "query": qry,
            "positive_passages": [pos_psg],
            "negative_passages": negs,
        }


# TODO: Below classes are going to be deprecated in favor of RPC impl of RerankerModel.
class EncodeDataset(Dataset):
    """ A dataset receives tsv (query_id, passage_id, any) as inputs """
    def __init__(
            self,
            data_args: DataArguments,
            tsv_ranks_path: str,       # Path to .ranks.tsv generated by dual-encoder, format [qid, pid, score]
            query_collection: str,        # Path to query corpus
            passage_collection: str,      # Path to passage corpus
            depth: int=1000,           # Reranking depth, will only re-score `top-depth` per qid using CrossEncoder
    ):
        self.data_args = data_args
        # Load query corpus
        self.query_dataset: datasets.Dataset = read_corpus(query_collection)
        self.idx_to_query: Dict[str, int] = build_corpus_idx_to_row(self.query_dataset)
        # Load passage corpus
        self.passage_dataset: datasets.Dataset = read_corpus(passage_collection)
        self.idx_to_passage: Dict[str, int] = build_corpus_idx_to_row(self.passage_dataset)
        # Load query_id, passage_id generated by dual-encoder 
        self.qp_pairs: List[Tuple[str, str]] = process_tsv_file(tsv_ranks_path, depth=depth)    # (qid, pid) pairs

    def __len__(self):
        return len(self.qp_pairs) 

    def __getitem__(self, index) -> Dict[str, any]:
        qid, pid = self.qp_pairs[index]
        item = {
            "query_id": qid,
            "query": self.query_dataset[self.idx_to_query[qid]],
            "passage_id": pid,
            "passage": self.passage_dataset[self.idx_to_passage[pid]],
        }
        return item
    
    def shard_(self, num_shards: int, index: int):
        """ In-place shard operation """
        div = len(self) // num_shards
        mod = len(self) % num_shards
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        self.qp_pairs = self.qp_pairs[start: end]


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize encode dataset.
    """
    separator: str = " "        # WhiteSpace
    
    def _get_text(self, item: Dict[str, str]):
        if "title" in item:
            return item["title"] + self.separator + item["text"]
        else:
            return item["text"]

    def __call__(self, features: List[dict]):
        query_ids: List[str] = list()
        passage_ids: List[str] = list()
        texts: List[List[str]] = list()

        for item in features:
            query_ids.append(item["query_id"])
            passage_ids.append(item["passage_id"])
            texts.append([self._get_text(item["query"]), self._get_text(item["passage"])])
        
        encoded: BatchEncoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation='longest_first',
            padding=self.padding,
            add_special_tokens=True,
            return_tensors=self.return_tensors,
        )
        return query_ids, passage_ids, encoded

