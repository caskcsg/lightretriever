#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
copied from eval/modeling_utils.py

ExactSearchModel encodes embeddings with PyTorch RPC remote calls.

@Time    :   2025/01/02
@Author  :   Ma (Ma787639046@outlook.com)
'''
import math
import os
import time
import atexit
import logging
import numpy as np
import queue
from threading import Thread
from queue import Queue
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
import torch.utils.data
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch import Tensor

import datasets
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils import PreTrainedTokenizerBase, BatchEncoding, PaddingStrategy

from sparse_emb_util import ICUWordPreTokenizer

from .arguments import InferenceArguments
from .utils import move_to_cuda
from ..finetune.modeling_encoder import EncoderModel
from ..finetune.modeling_hybrid import HybridModel
from ..finetune.nonctx_emb_utils import tokenize_nonctx_qry_emb_bag, construct_embedding_bag_parallel
from ..utils.data_utils import STOPWORD_SETS

logger = logging.getLogger(__name__)

_MODEL_CLS: dict[str, EncoderModel | HybridModel] = {
    "EncoderModel": EncoderModel,
    "HybridModel": HybridModel,
}

@dataclass
class ExactSearchModel:
    """
    A Wrapper for EncoderModel.
    
    Exact Search (ES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """
    args: InferenceArguments

    # Only used by the RPC main process
    query_prompt: Optional[str] = None
    corpus_prompt: Optional[str] = None
    encoding_kwargs: dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.args.world_size > 0 and self.args.local_rank >= 0, \
            "Please use rpc encoding with torchrun"

        # Process input/output data, send encoding message only in main rank
        if self.args.local_rank == 0:
            self.input_queue = mp.Queue()
            self.output_queue = mp.Queue()
            self.lower_boundary_of_queue_size: int = 32
            self.up_boundary_of_queue_size: int = 64
            assert self.up_boundary_of_queue_size > self.lower_boundary_of_queue_size

            # Feeder threader
            # 1) Fetch a data from input queue
            # 2) Execute RPC call with rank=device_id
            # 3) Put returns to output queue
            self.threads: list[Thread] = []
            for device_id in range(self.args.world_size):
                p = Thread(
                    target=self._start_mt,
                    args=(device_id, self.args, self.input_queue, self.output_queue),
                    daemon=True,   # Auto destruct all feeder threads
                )
                p.start()
                self.threads.append(p)

        self.tokenizer = EncoderModel.load_tokenizer(self.args.model_name_or_path, self.args)
        self.model = _MODEL_CLS[self.args.model_type].load(
            model_name_or_path=self.args.model_name_or_path,
            model_args=self.args,
            # HF Argument
            attn_implementation=self.args.attn_implementation,
            torch_dtype=self.args.dtype,
            device_map=self.args.local_rank,
            # device_map=0,   # `CUDA_VISIBLE_DEVICES` is set when int(local_rank) > 0
            # device_map="auto" if os.getenv("CUDA_VISIBLE_DEVICES") else target_device,
        ).eval()
        
        global MODEL_REGISTRY
        MODEL_REGISTRY = {"worker": self}

        atexit.register(self.__del__)

        # Inject Anserini specific arguments
        self.encoding_kwargs["anserini_vector_type"] = self.args.anserini_vector_type
    
    def __del__(self):
        """
        Stops all processes started with start_multi_process_pool
        """
        if self.args.local_rank == 0:
            if self.threads is not None:
                # Sending close signal chunk_id = None to all processes
                for _ in range(len(self.threads)):
                    self.input_queue.put([None])
                
                for p in self.threads:
                    p.join(timeout=10)
            
            if self.input_queue is not None:
                self.input_queue.close()

            if self.output_queue is not None:
                self.output_queue.close()
    
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
    ) -> Union[Tensor, np.ndarray]:
        # Support of EmbeddingBag
        if noncontextual_query_embedding := self.args.noncontextual_query_embedding and encode_is_query:
            # Construct EmbeddingBag on Rank0
            emb_bag_prompt = self.query_prompt or dataset[0].get("prompt", None)    # str / None
            if self.args.noncontextual_prompt_prefix and emb_bag_prompt:
                emb_bag_prompt = self.args.noncontextual_prompt_prefix + emb_bag_prompt
            elif self.args.noncontextual_prompt_prefix and (not emb_bag_prompt):
                emb_bag_prompt = self.args.noncontextual_prompt_prefix
            
            emb_bag_exists = self.model.emb_bag is not None
            emb_bag_need_recompute = self.model.emb_bag_prompt != emb_bag_prompt
            if (not emb_bag_exists) or (emb_bag_exists and emb_bag_need_recompute):
                # Need to compute a new EmbeddingBag
                emb_bag = construct_embedding_bag_parallel(
                    hidden_size=self.model.lm_q_base_unwrap.config.hidden_size,
                    tokenizer=self.tokenizer,
                    encode_func=_rpc_lm_q_encode_last_pooling_parallel,
                    input_queue=self.input_queue,
                    output_queue=self.output_queue,
                    prompt=emb_bag_prompt, 
                    batch_size=self.args.eval_batch_size_embedding_bag
                )
                self.empty_cache()

                for rank in range(self.args.world_size):
                    rpc.rpc_sync(
                        rank, 
                        _rpc_set_embedding_bag, 
                        kwargs={
                            "emb_bag": emb_bag,
                            "emb_bag_prompt": emb_bag_prompt,
                        }
                    )
        
        if self.args.debug:
            num_workers = 0
        
        # Reduce num_workers if batches are not too large
        num_workers = min(num_workers, max(1, math.ceil(len(dataset) / batch_size) // 4))

        data_iter = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            # Python Multi-Processing duplicates in-memory python dataset because of Reference Counts.
            # See: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
            #
            # Posiable Fix: Use HF dataset
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=EncodeCollator(
                self.tokenizer, padding=self.args.padding, 
                max_length=self.args.q_max_len if encode_is_query else self.args.p_max_len,
                emb_size=self.model.lm_p.get_input_embeddings().weight.shape[0],
                sparse_remove_stopwords=self.args.sparse_remove_stopwords,
                use_icu_word_pretokenizer=self.args.use_icu_word_pretokenizer,
                noncontextual_query_embedding=noncontextual_query_embedding,
                noncontextual_prompt_prefix=self.args.noncontextual_prompt_prefix,
                token_id_vector_type=self.args.token_id_vector_type,
            ),
        )

        input_cnt, output_cnt = 0, 0
        pbar = tqdm(
            total=len(data_iter), mininterval=10, disable=(not show_progress_bar), 
            desc=f"Encoding {'query' if encode_is_query else 'corpus'} [n_gpu={self.args.world_size}, bs={batch_size}]"
        )

        encoded_embeds: dict[str, dict[int, Tensor | str | list]] = dict()    # emb_name -> chunk_id -> emb

        def put_emb_to_dict(output_chunk_id: int, emb: Tensor | np.ndarray | list[str] | list[dict] | dict[str, any]):
            """ Helper function to put `emb` to `encoded_embeds`.
                Note:
                    Dense reps (Tensor | np.ndarray)
                    Sparse reps (list[str] | list[dict])
                    Hybrid reps (dict[str, Tensor | np.ndarray | list[str] | list[dict]])
            """
            if isinstance(emb, (Tensor, np.ndarray)):
                emb = {'dense_reps': emb}
            elif isinstance(emb, list):
                emb = {'sparse_reps': emb}
            
            assert isinstance(emb, dict), "Unrecognized type {type(emb)} for emb. Please encode embeddings as Tensor / np.ndarray / list / dict[str, any]"
            for k, v in emb.items():
                if k not in encoded_embeds:
                    encoded_embeds[k] = dict()
                encoded_embeds[k][output_chunk_id] = v

        for chunk_id, batch in enumerate(data_iter):
            if self.args.debug:
                emb = _rpc_encode(batch, encode_is_query, self.encoding_kwargs)
                put_emb_to_dict(chunk_id, emb)
                pbar.update()
            else:
                self.input_queue.put([chunk_id, _rpc_encode, batch, encode_is_query, self.encoding_kwargs])
                input_cnt += 1

                # If queue size exceeds the up_boundary, we will fetch some data before sending
                # more chunk to the input queue. This avoids the overwhelm of Process's queue
                if input_cnt - output_cnt > self.up_boundary_of_queue_size:
                    while input_cnt - output_cnt > self.lower_boundary_of_queue_size:
                        output_chunk_id, emb = self.output_queue.get()
                        put_emb_to_dict(output_chunk_id, emb)
                        output_cnt += 1
                        pbar.update()
        
        for _ in range(input_cnt - output_cnt):
            output_chunk_id, emb = self.output_queue.get()
            put_emb_to_dict(output_chunk_id, emb)
            output_cnt += 1
            pbar.update()

        pbar.close()

        # Compatiable issue: If emb is natually dict format, we should also return them in dict format
        return_dict_format_results = True if isinstance(emb, dict) else False
        # Concat Tensor
        logger.info("Collecting encoded embeddings...")
        for k in encoded_embeds.keys():
            if isinstance(encoded_embeds[k][0], Tensor):
                encoded_embeds[k] = torch.cat([encoded_embeds[k][i] for i in range(len(encoded_embeds[k]))], dim=0)
                if not convert_to_tensor:
                    encoded_embeds[k] = encoded_embeds[k].numpy()
            elif isinstance(encoded_embeds[k][0], np.ndarray):
                encoded_embeds[k] = np.concatenate([encoded_embeds[k][i] for i in range(len(encoded_embeds[k]))], axis=0)
            elif isinstance(encoded_embeds[k][0], list):
                _collected_list_emb: list[any] = []
                for i in range(len(encoded_embeds[k])):
                    _collected_list_emb.extend(encoded_embeds[k][i])
                encoded_embeds[k] = _collected_list_emb
            else:
                raise TypeError(f"Unable to collect embeddings with type {type(encoded_embeds[k][0])}.")
        
        # Clear up cache
        self.empty_cache()

        if return_dict_format_results:
            return encoded_embeds
        else:
            emb_names: list[str] = list(encoded_embeds.keys())
            assert len(emb_names) == 1, "Not single representations, please encode multi-representations with dict[str, any] format"
            return encoded_embeds[emb_names[0]]
    
    @classmethod
    def _start_mt(cls, target_device: int, args: InferenceArguments, input_queue: Queue, results_queue: Queue):
        """ Start method of multi-processing """
        while True:
            try:
                chunk_id, *args = input_queue.get()
                if chunk_id is None:  # Use chunk_id == None as close signal
                    logging.warning(f"[{target_device}] Exit Signal received, terminating..")
                    break
                
                assert len(args) >= 1, "Please pass a function pointer which is pickle serializable."
                # Unpack function pointer
                func, *args = args
                assert isinstance(func, Callable)

                rpc_call_success: bool = False
                retry_cnt: int = 3
                retry_interval: float = 2.0
                while (rpc_call_success is False) and (retry_cnt > 0):
                    try:
                        rets = rpc.rpc_sync(target_device, func, args=args)
                        rpc_call_success = True
                    except RuntimeError as e:
                        logger.error(e)
                        logger.error(f"[target_device: {target_device}] Retry counts: {retry_cnt}. Retrying after {retry_interval}s...")
                        retry_cnt -= 1
                        time.sleep(retry_interval)
                        rpc.rpc_sync(target_device, _cuda_empty_cache)

                results_queue.put((chunk_id, rets))
            except queue.Empty:
                break
    
    def empty_cache(self):
        for rank in range(self.args.world_size):
            rpc.rpc_sync(rank, _cuda_empty_cache)
            # rpc.rpc_sync(rank, torch.cuda.empty_cache)

def call_batch_encode(model: EncoderModel | HybridModel, batch: dict | BatchEncoding, encode_is_query: bool, encoding_kwargs: dict):
    """ Call model encode with tokenized batch inputs, do necessary post-processings: 
        1. Dense Embedding: Move to cpu.
        2. Sparse Embedding: Convert to String (query) / dict (passage) format.
    """
    batch = move_to_cuda(batch, device=model.lm_p.device)
    with torch.no_grad(), torch.autocast(device_type="cuda"), torch.cuda.device(model.lm_p.device):
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

def _rpc_encode(batch: dict | BatchEncoding, encode_is_query: bool, encoding_kwargs: dict):
    """
    Encode warpper function used by each `worker`
    This function is executed on workers.
    """
    # Global registry (global environment variable) seems to be the only way 
    # we can get the pointer of model during remote rpc execution.
    assert isinstance(batch, (dict, BatchEncoding))
    assert isinstance(encode_is_query, bool)
    assert isinstance(encoding_kwargs, dict)

    global MODEL_REGISTRY
    model = MODEL_REGISTRY['worker'].model
    return call_batch_encode(model, batch, encode_is_query, encoding_kwargs)

def _cuda_empty_cache():
    if local_rank := os.getenv("LOCAL_RANK", None):
        with torch.cuda.device(f'cuda:{local_rank}'):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()


# === EmbeddingBag Related ===
def _rpc_set_embedding_bag(emb_bag: torch.nn.EmbeddingBag, emb_bag_prompt: Optional[str]):
    global MODEL_REGISTRY
    model = MODEL_REGISTRY['worker'].model

    model.emb_bag = emb_bag
    model.emb_bag.to(model.lm_q.device)
    model.emb_bag_prompt = emb_bag_prompt

def _rpc_lm_q_encode_last_pooling_parallel(batch: dict | BatchEncoding):
    """ Encode with lm_q, then pooling its hidden states at the last position """
    global MODEL_REGISTRY
    lm_q = MODEL_REGISTRY['worker'].model.lm_q_base_unwrap

    # Forward model
    with torch.no_grad(), torch.autocast("cuda"):
        batch = move_to_cuda(batch, device=lm_q.device)
        lm_out: BaseModelOutput = lm_q(
            **batch,
            return_dict=True,
            use_cache=False,    # Do not return `past_key_values`
            output_hidden_states=False
        )
    
    # Fetch eos embedding
    output_eos_embedding = lm_out.last_hidden_state[:, -1].cpu()
    return output_eos_embedding


# === DataCollator ===
@dataclass
class EncodeCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: str = 'only_first'
    max_length: Optional[int] = None
    return_tensors: str = "pt"
    emb_size: Optional[int] = None      # Used to create bow / bce label
    word_tokenizer = ICUWordPreTokenizer(STOPWORD_SETS)
    sparse_remove_stopwords: bool = False
    use_icu_word_pretokenizer: bool = False
    noncontextual_query_embedding: bool = False
    noncontextual_prompt_prefix: Optional[str] = None   # Optional string to prepend in front of each prompts.
    token_id_vector_type: str = "bow"
    
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
        texts_merged = self.format_texts(texts, prepend_prompt=True)
        encoded = dict(self.tokenizer(
            texts_merged,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            # return_token_type_ids=False,
            return_tensors=self.return_tensors,
        ))

        if self.noncontextual_query_embedding:
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
                max_len=self.max_length,
            )
            encoded["nonctx_tok_emb_input_ids"] = q_nonctx_tok_emb_tokenized["input_ids"]
            encoded["nonctx_tok_emb_offsets"] = q_nonctx_tok_emb_tokenized["offsets"]

        # ** Sparse Pooling **
        if self.use_icu_word_pretokenizer:
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
                max_length=self.max_length,
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
