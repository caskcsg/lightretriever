#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training datasets.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''

import random
import numpy as np
from itertools import chain
from functools import partial
from dataclasses import dataclass
from typing import Optional
from collections import Counter

import datasets
from transformers import PreTrainedTokenizerBase, BatchEncoding, DataCollatorWithPadding

import torch
from torch import Tensor
from torch.utils.data import Dataset

from sparse_emb_util import ICUWordPreTokenizer

from .arguments import DataArguments
from .nonctx_emb_utils import tokenize_nonctx_qry_tok_emb
from ..trainer import ContrastiveTrainer
from ..utils.data_utils import read_corpus, build_corpus_idx_to_row, get_icu_word_pretokenizer

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize train dataset.
    """
    q_max_len: int = 512
    p_max_len: int = 512
    # separator: str = getattr(self.tokenizer, "sep_token", ' ')  # [SEP]
    separator: str = " "        # WhiteSpace


    # LightRetriever's Dense: Non-contextual query embedding
    noncontextual_query_embedding: bool = False
    noncontextual_prompt_prefix: Optional[str] = None   # Optional string to prepend in front of each prompts.

    # LightRetriever's Sparse: Term-based sparse reps
    token_id_vector_type: str = "sum"
    use_icu_word_pretokenizer: bool = False
    sparse_remove_stopwords: bool = False
    word_tokenizer: Optional[ICUWordPreTokenizer] = None   # ICU Word Tokenizer
    emb_size: Optional[int] = None      # Used to create bow / bce label
    
    use_nested_tensor: bool = False 
    gpt_is_casual: bool = True
    
    def _get_query(self, item: dict[str, str], prepend_prompt=True, prepend_whitespace=False) -> str:
        """ Format a Query for Embedding Training
            Args:
                item (dict[str, str]): A dict of triples, format {
                    "query": str,
                    "positive_passages": list of {"title": str, "text": str}
                    "negative_passages": list of {"title": str, "text": str}
                }
                prepend_prompt (bool): Override flag. Default to activate.
        """
        query = item["query"]
        if prepend_whitespace:
            query = " " + query
        if prepend_prompt and "query_prompt" in item:
            query = item["query_prompt"] + query
        
        return query
    
    def _get_passages(self, item: dict[str, dict[str, str]], prepend_prompt=True, prepend_whitespace=False) -> list[str]:
        """ Format a Passage for Embedding Training
            Args:
                item (dict[str, str]): A dict of triples, format {
                    "query": str,
                    "positive_passages": list of {"title": str, "text": str}
                    "negative_passages": list of {"title": str, "text": str}
                }
                prepend_prompt (bool): Override flag. Default to activate.
        """
        assert isinstance(item["positive_passages"], list) and isinstance(item["negative_passages"], list)
        assert len(item["positive_passages"]) == 1, f"Contrastive learning needs 1 positive passage, but found {len(item['positive_passages'])}"

        all_psgs: list[str] = []
        for psg in chain(item["positive_passages"], item["negative_passages"]):
            if "title" in psg and psg["title"]:
                text = psg["title"] + self.separator + psg["text"]
            else:
                text = psg["text"]
            
            if prepend_whitespace:
                text = " " + text
            if prepend_prompt and "passage_prompt" in item:
                text = item["passage_prompt"] + text
            
            all_psgs.append(text)
        
        return all_psgs
    
    def to_nested_tensor(self,input_ids: Tensor, attention_mask: Tensor):
        valid_lengths = attention_mask.sum(dim=1).tolist()
        trimmed_sequences = [input_ids[i, :valid_lengths[i]] for i in range(input_ids.size(0))]
        nested_tensor = torch.nested.nested_tensor(trimmed_sequences, layout=torch.jagged)
        return nested_tensor
    
    def _get_token_id_reps(self, word_token_ids: list[list[int]], unique_token_ids: Optional[list[list[int]]]=None):
        """ Get token id reps for Parameter-free Encoder 
            Args:
                word_token_ids (list[list[int]]): Tokenized token ids.
                unique_token_ids (list[list[int]]): Optional deduplicated token ids.
            
            Returns:
                Tuple of
                - token_id_str_reps (list[str])
                - token_id_json_reps (list[dict[str, int]])
                - token_id_pt_reps (Tensor)
        """
        batch_size = len(word_token_ids)
        if unique_token_ids is None:
            unique_token_ids = [list(set(item)) for item in word_token_ids]
        
        token_id_str_reps: list[str] = []
        token_id_json_reps: list[dict[str, int]] = []
        token_id_pt_reps: Tensor = torch.zeros([batch_size, self.emb_size], dtype=torch.float32)

        if self.token_id_vector_type == "bow":
            # `bow`: Directly use set(input_ids) as query's token id vector. `tok -> 1` 
            for i in range(batch_size):
                curr_unique_ids_list = unique_token_ids[i]

                # Str reps: concat str token_ids
                str_rep_curr = " ".join(str(token_id) for token_id in curr_unique_ids_list)
                token_id_str_reps.append(str_rep_curr)

                # Json reps: {"Token_id": 1, ...}
                json_rep_curr = {str(token_id): 1 for token_id in curr_unique_ids_list}
                token_id_json_reps.append(json_rep_curr)

                # Pytorch reps: token_id -> 1.0
                token_id_pt_reps[i, curr_unique_ids_list] = 1.0
        
        elif self.token_id_vector_type == "sum":
            # `sum`: Use number of each token as query's token id vector. `tok -> # of this tok`
            for i in range(batch_size):
                curr_token_ids_list = word_token_ids[i]
                # Str reps: concat str token_ids
                str_rep_curr = " ".join(str(token_id) for token_id in curr_token_ids_list)
                token_id_str_reps.append(str_rep_curr)

                # Json reps: {"Token_id": # of it, ...}
                json_rep_curr = {str(k): v for k, v in Counter(curr_token_ids_list).items()}
                token_id_json_reps.append(json_rep_curr)

                # Pytorch reps: token_id -> # of it
                curr_token_ids_list_pt_counts = torch.tensor(curr_token_ids_list).bincount()
                token_id_pt_reps[i, :curr_token_ids_list_pt_counts.shape[0]] = curr_token_ids_list_pt_counts

        else:
            raise NotImplementedError()
        
        return token_id_str_reps, token_id_json_reps, token_id_pt_reps

    def __call__(self, features: list[dict]):
        batch_size = len(features)
        # Tokenize `Query`
        q_texts: list[str] = list(map(self._get_query, features))
        q_tokenized: BatchEncoding = self.tokenizer(
            q_texts,
            max_length=self.q_max_len,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=self.return_tensors,
        )
        
        if self.noncontextual_query_embedding:
            # Generate (Prompted) token chunks
            q_nonctx_tok_emb_tokenized = tokenize_nonctx_qry_tok_emb(
                queries=[i["query"] for i in features],
                tokenizer=self.tokenizer,
                max_len=self.q_max_len,
                prompts=[i["query_prompt"] for i in features] if "query_prompt" in features[0] else None,
                noncontextual_prompt_prefix=self.noncontextual_prompt_prefix,
                is_casual=self.gpt_is_casual,
            )
            q_tokenized["nonctx_tok_emb_input_ids"] = q_nonctx_tok_emb_tokenized["input_ids"]
            q_tokenized["nonctx_tok_emb_attention_mask"] = q_nonctx_tok_emb_tokenized["attention_mask"]
            q_tokenized["nonctx_tok_emb_position_ids"] = q_nonctx_tok_emb_tokenized["position_ids"]
            q_tokenized["nonctx_tok_emb_attention_mask_2d"] = q_nonctx_tok_emb_tokenized["attention_mask_2d"]

        # Process `Passage` & `Negatives`
        p_texts: list[str] = sum(map(self._get_passages, features), [])
        
        # Tokenize Passage
        p_tokenized: BatchEncoding = self.tokenizer(
            p_texts,
            max_length=self.p_max_len,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=self.return_tensors,
        )

        processed = {
            "query": dict(q_tokenized),
            "passage": dict(p_tokenized),
        }
        
        # Sample CE Scores for Distillation
        # Note: Some datasets (e.g. synthetic.parquet) may not contain ce scores
        #       We should set its `ce_score` to `np.NINF` / `np.NAN` if it's not available.
        #       And do not process ce_scores if it's `np.NINF` / `np.NAN`.
        if "ce_score" in features[0]["positive_passages"][0] and \
            features[0]["positive_passages"][0]["ce_score"] is not None and \
           (not np.isneginf(float(features[0]["positive_passages"][0]["ce_score"]))) and \
            (not np.isnan(float(features[0]["positive_passages"][0]["ce_score"]))):
            ce_scores: list[float] = []
            for item in features:
                ce_scores.append(float(item["positive_passages"][0]["ce_score"]))
                for _neg in item["negative_passages"]:
                    ce_scores.append(float(_neg["ce_score"]))
            processed["ce_scores"] = torch.tensor(ce_scores, dtype=torch.float32)

        if "domain_ids" in features[0]:
            processed["domain_ids"] = torch.tensor([features[i]["domain_ids"] for i in range(len(features))], dtype=torch.int64)
        
        if "domain_name" in features[0]:
            processed["domain_name"] = [features[i]["domain_name"] for i in range(len(features))]    

            # Support masking in-batch / cross-batch negatives when encounter special tasks
            task_prefixs_for_only_hn = ["clustering", "classification"]
            only_hn: list[bool] = []
            for item in features:
                # Only use hard negatives, do not use in-batch / cross-batch negatives
                if any(_prefix in item["domain_name"] for _prefix in task_prefixs_for_only_hn):
                    only_hn.append(True)
                else:
                    only_hn.append(False)
            processed["only_hn"] = torch.tensor(only_hn, dtype=torch.bool)
        

        # ** Sparse Pooling **
        # Get token ids for sparse pooling: 
        # 1) ICU Word PreTokenize -> HF Tokenize -> Set Dedup
        # 2) Direct using HF Tokenize -> Set Dedup (Our method)
        q_text_neat: list[str] = list(map(partial(self._get_query, prepend_prompt=False, prepend_whitespace=True), features))
        p_text_neat: list[str] = sum(map(partial(self._get_passages, prepend_prompt=False, prepend_whitespace=True), features), [])

        if self.use_icu_word_pretokenizer:
            if self.word_tokenizer is None:
                self.word_tokenizer = get_icu_word_pretokenizer()
            
            ## Pretokenize via ICU Tokenizer
            q_pretokenized = self.word_tokenizer(q_text_neat, remove_stopwords=self.sparse_remove_stopwords)
            p_pretokenized = self.word_tokenizer(p_text_neat, remove_stopwords=self.sparse_remove_stopwords)
            ## Then tokenize with HFTokenizer
            q_word_token_ids: list[list[int]] = self.tokenizer(
                q_pretokenized, is_split_into_words=True, add_special_tokens=False)["input_ids"]
            p_word_token_ids: list[list[int]] = self.tokenizer(
                p_pretokenized, is_split_into_words=True, add_special_tokens=False)["input_ids"]
        else:
            ## Direct Tokenize using HFTokenzier
            q_word_token_ids: list[list[int]] = self.tokenizer(
                q_text_neat, 
                max_length=self.q_max_len,
                truncation='only_first',
                add_special_tokens=False
            )["input_ids"]
            p_word_token_ids: list[list[int]] = self.tokenizer(
                p_text_neat, 
                max_length=self.p_max_len,
                truncation='only_first',
                add_special_tokens=False
            )["input_ids"]
        
        # Obtain unique ids
        q_unique_token_ids: list[list[int]] = [list(set(item)) for item in q_word_token_ids]
        p_unique_token_ids: list[list[int]] = [list(set(item)) for item in p_word_token_ids]
        ## Add them to model inputs, for sparse pooling (list chunking supported by GradCache)
        processed["query"]["unique_token_ids"] = q_unique_token_ids
        processed["passage"]["unique_token_ids"] = p_unique_token_ids
        ## Add them to kwargs for some statistics
        processed["q_unique_token_ids"] = q_unique_token_ids
        processed["p_unique_token_ids"] = p_unique_token_ids

        # ** Parameter-free Query Encoder **
        q_token_id_str_reps, q_token_id_json_reps, q_token_id_pt_reps = self._get_token_id_reps(q_word_token_ids, q_unique_token_ids)
        processed["query"]["token_id_reps_str"] = q_token_id_str_reps
        processed["query"]["token_id_reps_json"] = q_token_id_json_reps
        processed["query"]["token_id_reps_pt"] = q_token_id_pt_reps

        # ** Sparse Training **
        n_psg = len(p_unique_token_ids) // batch_size
        # Add a BCE/BoW label for Sparse Training
        q_unique_bce_label = torch.zeros([batch_size, self.emb_size], dtype=torch.float32)
        q_p_pos_unique_bce_label = torch.zeros([batch_size, self.emb_size], dtype=torch.float32)
        q_unique_bow_label = torch.zeros([batch_size, self.emb_size], dtype=torch.float32)
        q_p_pos_unique_bow_label = torch.zeros([batch_size, self.emb_size], dtype=torch.float32)
        for i in range(batch_size):
            ## BCE
            q_unique_bce_label[i, q_unique_token_ids[i]] = 1.0

            q_p_pos_unique_token_ids_curr = list(set(q_unique_token_ids[i]) | set(p_unique_token_ids[i*n_psg]))
            q_p_pos_unique_bce_label[i, q_p_pos_unique_token_ids_curr] = 1.0

            ## BoW
            q_unique_bow_label[i, q_unique_token_ids[i]] = 1.0 / len(q_unique_token_ids[i]) if len(q_unique_token_ids[i]) > 0 else 0
            q_p_pos_unique_bow_label[i, q_p_pos_unique_token_ids_curr] = 1.0 / len(q_p_pos_unique_token_ids_curr) if len(q_p_pos_unique_token_ids_curr) > 0 else 0
        
        processed["q_unique_bce_label"] = q_unique_bce_label
        processed["q_p_pos_unique_bce_label"] = q_p_pos_unique_bce_label
        processed["q_unique_bow_label"] = q_unique_bow_label
        processed["q_p_pos_unique_bow_label"] = q_p_pos_unique_bow_label
        
        return processed


@dataclass
class IterableTrainCollator(TrainCollator):
    """
    IterableTrainCollator for sample batch examples, processing & tokenize train dataset.

    Note:
    1. `query_prompt`: will be read from dataset["instruction"].
    2. `passage_prompt`: always NOT add passage prompt.
    """
    train_n_passages: int = 2
    seed: int = 42
    positive_passage_no_shuffle: bool = False
    negative_passage_no_shuffle: bool = False
    add_prompt_prob: float = -1.
    prompt_type: str = 'e5'
    append_prompt_sep: bool = False

    def __post_init__(self):
        self.rng = random.Random(self.seed)
    
    def __call__(self, group: list[dict]):
        return super(IterableTrainCollator, self).__call__(list(map(self.get_item, group)))
    
    def get_item(self, group: dict):
        # TODO: 1) self.seed should be added to idx; 2) No epoch involved
        # seed: int = int(group.get('_train_dataset_idx', self.seed))
        # rng = random.Random(seed)

        rng = self.rng

        # Sample One Positive
        group_positives = group['positive_passages']
        if self.positive_passage_no_shuffle:
            pos_psg: dict[str, any] = group_positives[0]
        else:
            pos_psg: dict[str, any] = rng.choice(group_positives)
        
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

        return rets


# Single Dataset Implementation
# TODO: `TrainDataset` is rarely used because of `Interleavable datasets`
class TrainDataset(Dataset):
    """ Wrapper for Sampling Positive / Negative Passages """
    def __init__(
            self,
            data_args: DataArguments,
            dataset: str,                                  # String Path to Training Triples with Negatives
            query_collection: Optional[str] = None,        # String Path to query corpus
            passage_collection: Optional[str] = None,      # String Path to passage corpus
            trainer: ContrastiveTrainer = None,
            train_n_passages: int = 8,
            positive_passage_no_shuffle: bool = False,
            negative_passage_no_shuffle: bool = False,
    ):
        self.train_data = read_corpus(dataset)
        self.trainer = trainer
        self.data_args = data_args
        self.train_n_passages = train_n_passages
        self.positive_passage_no_shuffle = positive_passage_no_shuffle
        self.negative_passage_no_shuffle = negative_passage_no_shuffle

        self.read_text_from_collections = (query_collection is not None) and (passage_collection is not None)
        if query_collection is not None:
            # Load query corpus
            self.query_dataset: datasets.Dataset = read_corpus(query_collection)
            self.idx_to_query: dict[str, int] = build_corpus_idx_to_row(self.query_dataset)
        
        if passage_collection is not None:
            # Load passage corpus
            self.passage_dataset: datasets.Dataset = read_corpus(passage_collection)
            self.idx_to_passage: dict[str, int] = build_corpus_idx_to_row(self.passage_dataset)
    
    def get_query(self, _id: str) -> dict:
        return self.query_dataset[self.idx_to_query[_id]]
    
    def get_passage(self, _id: str) -> dict:
        return self.passage_dataset[self.idx_to_passage[_id]]
    
    def __len__(self):
        return len(self.train_data) 

    def __getitem__(self, index: int) -> dict[str, any]:
        group = self.train_data[index]
        _hashed_seed = hash(index + self.trainer.args.seed)

        epoch = int(self.trainer.state.epoch)

        # Read Query
        if self.read_text_from_collections:
            qry: str = self.get_query(group['_id'])['text']
        else:
            qry: str = group['text']

        # Sample One Positive
        group_positives = group['positive_passages']
        if self.positive_passage_no_shuffle:
            pos_psg: dict[str, any] = group_positives[0]
        else:
            pos_psg: dict[str, any] = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        if self.read_text_from_collections:
            pos_psg.update(self.get_passage(pos_psg['docid']))

        # Sample Negatives
        group_negatives = group['negative_passages']
        negative_size = self.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.train_n_passages == 1:
            negs = []
        elif self.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]
        
        if self.read_text_from_collections:
            negs_w_texts = list()
            for item in negs:
                item.update(self.get_passage(item['docid']))
                negs_w_texts.append(item)
            negs = negs_w_texts

        return {
            "query": qry,
            "positive_passages": [pos_psg],
            "negative_passages": negs,
        }

