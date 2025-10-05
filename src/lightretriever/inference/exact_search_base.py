#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Base class for Exact Search Model.

ExactSearchModel encodes embeddings with 
1. PyTorch Single Process.
2. PyTorch RPC remote calls.

@Time    :   2025/01/02
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import time
import math
import logging
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
import torch.utils.data
from torch import Tensor

import datasets
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils import PreTrainedTokenizerBase, BatchEncoding, PaddingStrategy

from sparse_emb_util import ICUWordPreTokenizer

from .arguments import InferenceArguments
from .utils import move_to_device, device_context, empty_cache
from ..finetune.modeling_encoder import EncoderModel
from ..finetune.modeling_hybrid import HybridModel
from ..finetune.nonctx_emb_utils import tokenize_nonctx_qry_emb_bag, construct_embedding_bag_parallel
from ..utils.data_utils import get_icu_word_pretokenizer

logger = logging.getLogger(__name__)

@dataclass
class ExactSearchModelBase:
    """
    A Wrapper for EncoderModel.
    
    Exact Search (ES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """
    args: InferenceArguments
    model: Optional[EncoderModel | HybridModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None

    query_prompt: Optional[str] = None
    corpus_prompt: Optional[str] = None
    encoding_kwargs: dict[str, any] = field(default_factory=dict)
    
    def parse_texts(
        self,
        texts: list[str] | list[dict[str, str]] | datasets.Dataset,
        prompt: Optional[str] = None
    ):
        """
        Format given texts to HF dataset format.
        Args:
            texts (`list[str] | list[dict[str, str]] | datasets.Dataset`): list of sentences to encode, 
                which can be formated as: 
                1) [str, ...]; 
                2) [{"title": str (`optional`), "text": str, "prompt": prompt_str (`optional`)}, ...]; 
                3) HF dataset with "title", "text", "prompt" (`optional`) columns. 
        """
        if isinstance(texts, list):
            assert len(texts) > 0, "Empty lists."
            if isinstance(texts[0], str):
                collected = datasets.Dataset.from_list([{"text": t} for t in texts])
            elif isinstance(texts[0], dict):
                collected = datasets.Dataset.from_list(texts)            
            else:
                raise NotImplementedError(f"Unrecognized texts[0] type {type(texts[0])}")
        elif isinstance(texts, datasets.Dataset):
            collected = texts
        else:
            raise NotImplementedError(f"Unrecognized type {type(texts)}")
        
        # `prompt` only works when there's no existing prompt.
        if ("prompt" not in collected.column_names) and prompt:
            if prompt and self.args.append_prompt_sep:
                prompt += self.tokenizer.sep_token + " "    # `{prompt}{sep_token} {text}`
            
            collected = collected.add_column("prompt", [prompt] * len(collected))
        
        return collected

    def encode_queries(
        self, 
        queries: list[str] | list[dict[str, str]] | datasets.Dataset, 
        batch_size: int, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True, 
        **kwargs
    ) -> Tensor:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            queries (`list[str] | list[dict[str, str]] | datasets.Dataset`): list of sentences to encode, 
                which can be formated as: 
                1) [str, ...]; 
                2) [{"text": str, "prompt": prompt_str (`optional`)}, ...]; 
                3) HF dataset with "text", "prompt" (`optional`) columns. 
            batch_size (`int`): Batch size for the encoding

        Returns:
            `np.ndarray` or `Tensor`: Embeddings for the given sentences.
        """
        return self._encode(
            self.parse_texts(queries, prompt=self.query_prompt), 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            convert_to_tensor=convert_to_tensor, 
            encode_is_query=True, 
            **kwargs
        )

    def encode_corpus(
        self, 
        corpus: list[str] | list[dict[str, str]] | datasets.Dataset, 
        batch_size: int, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True, 
        **kwargs
    ) -> Tensor:
        """
        Returns embeddings for the given sentences.
        Args:
            corpus (`list[str] | list[dict[str, str]] | datasets.Dataset`): list of sentences to encode, 
                which can be formated as: 
                1) [str, ...]; 
                2) [{"title": str (`optional`), "text": str, "prompt": prompt_str (`optional`)}, ...]; 
                3) HF dataset with "title", "text", "prompt" (`optional`) columns. 
            batch_size (`int`): Batch size for the encoding

        Returns:
            `np.ndarray` or `Tensor`: Embeddings for the given sentences
        """
        return self.encode(
            corpus, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            convert_to_tensor=convert_to_tensor, 
            **kwargs
        )
    
    def encode(
        self, 
        sentences: list[str] | list[dict[str, str]] | datasets.Dataset, 
        batch_size: int, 
        show_progress_bar: bool = True, 
        convert_to_tensor: bool = True, 
        **kwargs
    ):
        """
        Returns a embeddings for the given sentences.
        Args:
            sentences (`list[str] | list[dict[str, str]] | datasets.Dataset`): List of sentences to encode
                which can be formated as: 
                1) [str, ...]; 
                2) [{"title": str (`optional`), "text": str, "prompt": prompt_str (`optional`)}, ...]; 
                3) HF dataset with "title", "text", "prompt" (`optional`) columns. 
            batch_size (`int`): Batch size for the encoding

        Returns:
            `np.ndarray` or `Tensor`: Embeddings for the given sentences
        """
        return self._encode(
            self.parse_texts(sentences, prompt=self.corpus_prompt), 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            convert_to_tensor=convert_to_tensor, 
            encode_is_query=False, 
            **kwargs
        )
    
    def _encode(
        self, 
        dataset: datasets.Dataset, 
        batch_size: int, 
        show_progress_bar: bool, 
        convert_to_tensor: bool, 
        encode_is_query: bool,
        num_workers: int=16,
        # kwargs are not used here for now, if we want to change model's args, 
        # like `score_function`, please directly modify the model
        # TODO: Support `encoding_kwargs` recently added by MTEB
        **kwargs
    ) -> Union[
        Tensor, np.ndarray,             # Single emb type - dense
        list[str | dict[str, int]],     # Single emb type - sparse
        dict[str, Tensor | np.ndarray | list[str | dict[str, int]]],    # Multi emb types
    ]:
        raise NotImplementedError("Abstract class. Override this function for detailed implementation.")
    


def call_batch_encode(model: EncoderModel | HybridModel, batch: dict | BatchEncoding, encode_is_query: bool, encoding_kwargs: dict):
    """ Call model encode with tokenized batch inputs, do necessary post-processings: 
        1. Dense Embedding: Move to cpu.
        2. Sparse Embedding: Convert to String (query) / dict (passage) format.
    """
    device = model.lm_p.device
    batch = move_to_device(batch, device=device)
    with torch.no_grad(), torch.autocast(device.type), device_context(device):
        if encode_is_query:
            embeddings: Union[Tensor, dict[str, Tensor]] = model.encode_query(qry=batch, **encoding_kwargs)
        else:
            embeddings: Union[Tensor, dict[str, Tensor]] = model.encode_passage(psg=batch, **encoding_kwargs)
    
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.cpu()
    elif isinstance(embeddings, dict):
        for k in embeddings.keys():
            if embeddings[k] is None:
                continue
            
            if k == "sparse_reps":
                # # For Tantivy + BM25
                # embeddings[k] = model.convert_sparse_reps_to_pseudo_text(
                #         embeddings[k],
                #         quantization_factor=100,   # Choose depends on vector max/min
                #         convert_id_to_token=False,  # False: Use id / True: Use Token
                #     )

                # For Anserini
                if encode_is_query:
                    embeddings[k] = model.convert_sparse_reps_to_pseudo_text(
                        embeddings[k],
                        quantization_factor=100,   # Choose depends on vector max/min
                        convert_id_to_token=False,  # False: Use id / True: Use Token
                    )
                else:
                    anserini_vector_type = encoding_kwargs.get("anserini_vector_type", "JsonVectorCollection")
                    if anserini_vector_type == "JsonVectorCollection":
                        embeddings[k] = model.convert_sparse_reps_to_json(
                            embeddings[k],
                            quantization_factor=100,   # Choose depends on vector max/min
                            convert_id_to_token=False,  # False: Use id / True: Use Token
                        )
                    elif anserini_vector_type == "JsonCollection":
                        embeddings[k] = model.convert_sparse_reps_to_pseudo_text(
                            embeddings[k],
                            quantization_factor=100,   # Choose depends on vector max/min
                            convert_id_to_token=False,  # False: Use id / True: Use Token
                        )
                    else:
                        raise TypeError(f"Unsupported type {anserini_vector_type}")
            elif k == "token_id_reps":
                # Nothing to convert here, just check its type
                assert isinstance(embeddings[k][0], (str, dict)), f"The token_id_reps should be list[str] | list[dict[str, int]]."
            else:
                embeddings[k] = embeddings[k].cpu()
    else:
        raise NotImplementedError()

    return embeddings


# === DataCollator ===
@dataclass
class EncodeCollator:
    tokenizer: PreTrainedTokenizerBase
    encode_is_query: bool
    q_max_len: int = 512
    p_max_len: int = 512
    pad_to_max_length: bool = False
    padding: bool = True
    truncation: str = 'only_first'
    return_tensors: str = "pt"
    
    # LightRetriever's Dense: Non-contextual query embedding
    noncontextual_query_embedding: bool = False
    noncontextual_prompt_prefix: Optional[str] = None   # Optional string to prepend in front of each prompts.

    # LightRetriever's Sparse: Term-based sparse reps
    token_id_vector_type: str = "sum"
    use_icu_word_pretokenizer: bool = False
    sparse_remove_stopwords: bool = False
    word_tokenizer: Optional[ICUWordPreTokenizer] = None
    emb_size: Optional[int] = None      # Used to create bow / bce label
    
    def _get_text(
        self, 
        item: dict[str, str], 
        prepend_prompt=False, 
        prepend_whitespace=False
    ):
        """ Format a Text for Embedding Encoding
            Args:
                item (dict[str, str]): A dict of prompt + query, format {"prompt": str, "text": str}
                prepend_prompt (bool): Override flag. Default to activate.
        """
        if "title" in item and item["title"]:
            text = item["title"] + " " + item["text"]
        else:
            text = item["text"]
        if prepend_whitespace:
            text = " " + text
        if prepend_prompt and "prompt" in item:
            text = item["prompt"] + text
        
        return text
    
    def format_texts(
        self,
        texts: list[dict[str, str]],
        prepend_prompt=False, 
        prepend_whitespace=False
    ):
        """ Format List of Texts for Embedding Encoding
            Args:
                texts: list[dict[str, str]]: A list of dict of prompt + query, format [{"prompt": str, "text": str}]
                prepend_prompt (bool): Override flag. Default to activate.
        """
        return [self._get_text(
            item, 
            prepend_prompt=prepend_prompt, 
            prepend_whitespace=prepend_whitespace
        ) for item in texts]
    
    def __call__(
        self, 
        texts: list[dict[str, str]]
    ) -> dict[str, Tensor | list]:
        batch_size = len(texts)
        max_length = self.q_max_len if self.encode_is_query else self.p_max_len

        texts_merged = self.format_texts(texts, prepend_prompt=True)
        encoded = dict(self.tokenizer(
            texts_merged,
            max_length=max_length,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            # return_token_type_ids=False,
            return_tensors=self.return_tensors,
        ))

        if self.noncontextual_query_embedding and self.encode_is_query:
            # [TODO: Concated Impl] Generate (Prompted) token chunks
            # q_nonctx_tok_emb_tokenized = tokenize_nonctx_qry_tok_emb(
            #     queries=self.format_texts(texts),
            #     tokenizer=self.tokenizer,
            #     max_len=self.max_length,
            #     prompts=[i["prompt"] for i in texts] if "prompt" in texts[0] else None,
            #     noncontextual_prompt_prefix=self.noncontextual_prompt_prefix,
            #     is_casual=True,
            # )
            # encoded["nonctx_tok_emb_input_ids"] = q_nonctx_tok_emb_tokenized["input_ids"]
            # encoded["nonctx_tok_emb_attention_mask"] = q_nonctx_tok_emb_tokenized["attention_mask"]
            # encoded["nonctx_tok_emb_position_ids"] = q_nonctx_tok_emb_tokenized["position_ids"]
            # encoded["nonctx_tok_emb_attention_mask_2d"] = q_nonctx_tok_emb_tokenized["attention_mask_2d"]

            # EmbeddingBag Inputs
            q_nonctx_tok_emb_tokenized = tokenize_nonctx_qry_emb_bag(
                queries=self.format_texts(texts),
                tokenizer=self.tokenizer,
                max_len=max_length,
            )
            encoded["nonctx_tok_emb_input_ids"] = q_nonctx_tok_emb_tokenized["input_ids"]
            encoded["nonctx_tok_emb_offsets"] = q_nonctx_tok_emb_tokenized["offsets"]

        # ** Sparse Pooling **
        if self.use_icu_word_pretokenizer:
            if self.word_tokenizer is None:
                self.word_tokenizer = get_icu_word_pretokenizer()
            
            word_lists = self.word_tokenizer(self.format_texts(texts), remove_stopwords=self.sparse_remove_stopwords)
            token_ids: list[list[int]] = self.tokenizer(
                word_lists, 
                is_split_into_words=True, 
                add_special_tokens=False
            )['input_ids']
        else:
            texts_neat = self.format_texts(texts, prepend_whitespace=True)
            token_ids: list[list[int]] = self.tokenizer(
                texts_neat, 
                max_length=max_length,
                truncation=self.truncation,
                add_special_tokens=False
            )['input_ids']
        unique_token_ids = [list(set(item)) for item in token_ids]
        encoded['unique_token_ids'] = unique_token_ids

        # ** Parameter-free Query Encoder **
        token_id_reps_str: list[str] = []
        token_id_reps_json: list[dict[str, int]] = []
        # token_id_reps_pt = torch.zeros([batch_size, self.emb_size], dtype=torch.float32)

        if self.token_id_vector_type == "bow":
            # `bow`: Directly use set(input_ids) as query's token id vector. `tok -> 1` 
            for i in range(batch_size):
                curr_unique_ids_list = unique_token_ids[i]

                # Str reps: concat str token_ids
                str_rep_curr = " ".join(str(token_id) for token_id in curr_unique_ids_list)
                token_id_reps_str.append(str_rep_curr)

                # Json reps: {"Token_id": 1, ...}
                json_rep_curr = {str(token_id): 1 for token_id in curr_unique_ids_list}
                token_id_reps_json.append(json_rep_curr)

                # # Pytorch reps: token_id -> 1.0
                # token_id_reps_pt[i, curr_unique_ids_list] = 1.0
        
        elif self.token_id_vector_type == "sum":
            # `sum`: Use number of each token as query's token id vector. `tok -> # of this tok`
            for i in range(batch_size):
                curr_token_ids_list = token_ids[i]
                # Str reps: concat str token_ids
                str_rep_curr = " ".join(str(token_id) for token_id in curr_token_ids_list)
                token_id_reps_str.append(str_rep_curr)

                # Json reps: {"Token_id": # of it, ...}
                json_rep_curr = {str(k): v for k, v in Counter(curr_token_ids_list).items()}
                token_id_reps_json.append(json_rep_curr)

                # # Pytorch reps: token_id -> # of it
                # curr_token_ids_list_pt_counts = torch.tensor(curr_token_ids_list).bincount()
                # token_id_reps_pt[i, :curr_token_ids_list_pt_counts.shape[0]] = curr_token_ids_list_pt_counts

        else:
            raise NotImplementedError()
        
        encoded["token_id_reps_str"] = token_id_reps_str
        encoded["token_id_reps_json"] = token_id_reps_json
        # encoded["token_id_reps_pt"] = token_id_reps_pt

        return encoded
