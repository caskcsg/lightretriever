#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
copied from eval/modeling_utils.py

PytorchRPCExactSearchModel encodes embeddings with PyTorch RPC remote calls.

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
from itertools import chain
from threading import Thread
from queue import Queue
from tqdm import tqdm
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

from .exact_search_base import ExactSearchModelBase, EncodeCollator, call_batch_encode
from .arguments import InferenceArguments
from .utils import move_to_device, device_context, empty_cache
from ..finetune.modeling_encoder import EncoderModel
from ..finetune.modeling_hybrid import HybridModel
from ..finetune.nonctx_emb_utils import tokenize_nonctx_qry_tok_emb, tokenize_nonctx_qry_emb_bag, construct_embedding_bag_parallel

logger = logging.getLogger(__name__)

_MODEL_CLS: dict[str, EncoderModel | HybridModel] = {
    "EncoderModel": EncoderModel,
    "HybridModel": HybridModel,
}

@dataclass
class PytorchRPCExactSearchModel(ExactSearchModelBase):
    """
    PytorchRPCExactSearchModel encodes embeddings with PyTorch RPC remote calls.
    
    Exact Search (ES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """
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
                self.tokenizer, 
                encode_is_query=encode_is_query,
                q_max_len=self.args.q_max_len,
                p_max_len=self.args.p_max_len,
                pad_to_max_length=self.args.pad_to_max_length,
                padding=self.args.padding,
                # LightRetriever Dense
                noncontextual_query_embedding=noncontextual_query_embedding,
                noncontextual_prompt_prefix=self.args.noncontextual_prompt_prefix,
                # LightRetriever Sparse
                token_id_vector_type=self.args.token_id_vector_type,
                use_icu_word_pretokenizer=self.args.use_icu_word_pretokenizer,
                sparse_remove_stopwords=self.args.sparse_remove_stopwords,
                emb_size=self.model.lm_p.get_input_embeddings().weight.shape[0],
            ),
        )

        input_cnt, output_cnt = 0, 0
        pbar = tqdm(
            total=len(data_iter), mininterval=10, disable=(not show_progress_bar), 
            desc=f"Encoding {'query' if encode_is_query else 'corpus'} [n_gpu={self.args.world_size}, bs={batch_size}]"
        )

        encoded_embeds: dict[str, Tensor | np.ndarray | dict[int, list]] = dict()    # emb_name -> pre-allocated tensor / {chunk_id -> sparse_reps}

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
                    # If v is Tensor, we need to pre-allocate a very large space on CPU MEM.
                    # Then if new emb comes, we just fill the data to coresponding space.
                    if isinstance(v, Tensor):
                        encoded_embeds[k] = torch.empty((len(dataset), *v.shape[1:]), dtype=v.dtype)
                    elif isinstance(v, np.ndarray):
                        encoded_embeds[k] = np.empty((len(dataset), *v.shape[1:]), dtype=v.dtype)
                    else:
                        encoded_embeds[k] = dict()
                # If v is Tensor / np.ndarray, we just fill the data to coresponding space.
                if isinstance(v, (Tensor, np.ndarray)):
                    encoded_embeds[k][output_chunk_id * batch_size: output_chunk_id * batch_size + v.shape[0]] = v
                # If v is list, we assign the data to coresponding chunk id.
                else:
                    encoded_embeds[k][output_chunk_id] = v

        for chunk_id, batch in enumerate(data_iter):
            if self.args.debug:
                emb = call_batch_encode(self.model, batch, encode_is_query, self.encoding_kwargs)
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

        # Collecting embeddings
        logger.info("Collecting encoded embeddings...")
        for k in encoded_embeds.keys():
            if isinstance(encoded_embeds[k], Tensor):
                if not convert_to_tensor:
                    encoded_embeds[k] = encoded_embeds[k].numpy()
            elif isinstance(encoded_embeds[k], np.ndarray):
                if convert_to_tensor:
                    encoded_embeds[k] = torch.from_numpy(encoded_embeds[k])
            elif isinstance(encoded_embeds[k], dict):
                if isinstance(encoded_embeds[k][0], list):
                    encoded_embeds[k] = list(chain.from_iterable(encoded_embeds[k][i] for i in sorted(encoded_embeds[k].keys())))
                else:
                    raise TypeError(f"Unable to collect embeddings with type {type(encoded_embeds[k][0])}.")
        
        # Clear up cache
        self.empty_cache()
        
        # Compatiable issue: 
        #  - If emb is natually dict format with multiple emb types, we should also return them in dict format
        #  - If emb is single emb type, we should return the emb alone
        if return_dict_format_results := isinstance(emb, dict):
            return encoded_embeds
        else:
            emb_names: list[str] = list(encoded_embeds.keys())
            assert len(emb_names) == 1, f"Not single representations, please encode multi-representations with dict[str, any] format. Current emb names: {emb_names}"
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
                        rpc.rpc_sync(target_device, empty_cache)

                results_queue.put((chunk_id, rets))
            except queue.Empty:
                break
    
    def empty_cache(self, rank: Optional[int] = None):
        if rank is None:
            for rank in range(self.args.world_size):
                rpc.rpc_sync(rank, empty_cache)
        else:
            rpc.rpc_sync(rank, empty_cache)


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

    device = lm_q.device

    # Forward model
    with torch.no_grad(), torch.autocast(device.type):
        batch = move_to_device(batch, device=lm_q.device)
        lm_out: BaseModelOutput = lm_q(
            **batch,
            return_dict=True,
            use_cache=False,    # Do not return `past_key_values`
            output_hidden_states=False
        )
    
    # Fetch eos embedding
    output_eos_embedding = lm_out.last_hidden_state[:, -1].cpu()
    return output_eos_embedding


