#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
copied from eval/modeling_utils.py

RerankerModel scores the sentence pairs with RPC remote calls.

@Time    :   2025/01/02
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import time
import atexit
import logging
import numpy as np
import queue
from threading import Thread
from queue import Queue
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.utils.data
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch import Tensor

import datasets
from transformers.tokenization_utils import PreTrainedTokenizerBase, BatchEncoding, PaddingStrategy

from .arguments import InferenceArguments
from .utils import move_to_device
from ..rerank.modeling import CrossEncoder, CrossEncoderLogits

_MODEL_CLS: dict[str, CrossEncoder | CrossEncoderLogits] = {
    "CrossEncoder": CrossEncoder,
    "CrossEncoderLogits": CrossEncoderLogits
}

logger = logging.getLogger(__name__)

@dataclass
class RerankerModel:
    """
    A wrapper for multiprocessing encoding of re-ranker.

    Rerank in BeIR requires a `predict` method for reranking sentence pairs.
    This class converts a Transformers model into BeIR Rerank format.
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
            self.lower_boundary_of_queue_size: int = 16
            self.up_boundary_of_queue_size: int = 32
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

        model_cls = _MODEL_CLS[self.args.cross_encoder_type]
        self.tokenizer = model_cls.load_tokenizer(self.args.model_name_or_path, self.args)
        self.model = model_cls.from_pretrained(
            self.args.model_name_or_path,
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
    
    def __del__(self):
        """
        Stops all processes started with start_multi_process_pool
        """
        if self.args.local_rank == 0:
            if self.threads is not None:
                # Sending close signal chunk_id = None to all processes
                for _ in range(len(self.threads)):
                    self.input_queue.put([None, None, None])
                
                for p in self.threads:
                    p.join(timeout=10)
            
            if self.input_queue is not None:
                self.input_queue.close()

            if self.output_queue is not None:
                self.output_queue.close()
    
    def _format_sentence_pair(self, sentence_pair: list[str] | tuple[str]):
        assert isinstance(sentence_pair, (list, tuple))
        if len(sentence_pair) == 3:
            # MTEB will pass instruction as the 3rd column, we don't need it because the instruction is pre-defined.
            assert sentence_pair[2] is None, f"You are passing a triple, and the 3rd element is not None."
            sentence_pair = [sentence_pair[0], sentence_pair[1]]
        assert len(sentence_pair) == 2, "Requires sentence inputs as pairs."
        return {
            "text1": self.query_prompt + sentence_pair[0] if self.query_prompt else sentence_pair[0],
            "text2": self.corpus_prompt + sentence_pair[1] if self.corpus_prompt else sentence_pair[1]
        }
    
    def predict(
        self,
        sentences: list[list[str] | tuple[str]],
        batch_size: Optional[int]=None,
        show_progress_bar: bool=True,
        convert_to_tensor: bool=True,
        num_workers: int=8,
    ) -> Tensor | np.ndarray:
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param convert_to_tensor: Convert the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        formated_sentences = list(map(self._format_sentence_pair, sentences))
        dataset = datasets.Dataset.from_list(formated_sentences)     # Switch to pyarrow dataset
        del formated_sentences   # Release MEM
        # del sentences

        if batch_size is None:
            batch_size = self.args.batch_size
        data_iter = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=EncodeCollator(self.tokenizer, padding=self.args.padding, max_length=self.args.max_length),
        )

        input_cnt, output_cnt = 0, 0
        pbar = tqdm(
            total=len(data_iter), mininterval=10, disable=(not show_progress_bar), 
            desc=f"Encoding [n_gpu={self.args.world_size}, bs={batch_size}]"
        )

        rerank_scores: dict[int, Tensor] = dict()
        for chunk_id, batch in enumerate(data_iter):
            if self.args.debug:
                emb = self._rpc_encode(batch, self.encoding_kwargs)
                rerank_scores[chunk_id] = emb
                pbar.update()
            else:
                self.input_queue.put([chunk_id, batch, self.encoding_kwargs])
                input_cnt += 1

                # If queue size exceeds the up_boundary, we will fetch some data before sending
                # more chunk to the input queue. This avoids the overwhelm of Process's queue
                if input_cnt - output_cnt > self.up_boundary_of_queue_size:
                    while input_cnt - output_cnt > self.lower_boundary_of_queue_size:
                        output_chunk_id, emb = self.output_queue.get()
                        rerank_scores[output_chunk_id] = emb
                        output_cnt += 1
                        pbar.update()
        
        for _ in range(input_cnt - output_cnt):
            output_chunk_id, emb = self.output_queue.get()
            rerank_scores[output_chunk_id] = emb
            output_cnt += 1
            pbar.update()

        pbar.close()

        concated_embeds = torch.cat([rerank_scores[i] for i in range(len(rerank_scores))], dim=0)
        return concated_embeds if convert_to_tensor else concated_embeds.numpy()

    
    @classmethod
    def _start_mt(cls, target_device: int, args: InferenceArguments, input_queue: Queue, results_queue: Queue):
        """ Start method of multi-processing """
        while True:
            try:
                chunk_id, batch, encoding_kwargs = input_queue.get()
                if chunk_id is None:  # Use chunk_id == None as close signal
                    logging.warning(f"[{target_device}] Exit Signal received, terminating..")
                    break
                assert isinstance(batch, (dict, BatchEncoding))
                assert isinstance(encoding_kwargs, dict)

                rpc_call_success: bool = False
                retry_cnt: int = 3
                retry_interval: float = 2.0
                while (rpc_call_success is False) and (retry_cnt > 0):
                    try:
                        embeddings = rpc.rpc_sync(
                            target_device, 
                            cls._rpc_encode, 
                            kwargs={
                                "batch": batch,
                                "encoding_kwargs": encoding_kwargs,
                            }
                        )
                        rpc_call_success = True
                    except RuntimeError as e:
                        logger.error(e)
                        logger.error(f"[target_device: {target_device}] Retry counts: {retry_cnt}. Retrying after {retry_interval}s...")
                        retry_cnt -= 1
                        time.sleep(retry_interval)
                        rpc.rpc_sync(target_device, _cuda_empty_cache)

                results_queue.put((chunk_id, embeddings))
            except queue.Empty:
                break
    
    @staticmethod
    def _rpc_encode(batch: dict | BatchEncoding, encoding_kwargs: dict):
        """
        Encode warpper function used by each `worker`
        This function is executed on workers.
        """
        # Global registry (global environment variable) seems to be the only way 
        # we can get the pointer of model during remote rpc execution.
        global MODEL_REGISTRY
        return call_batch_encode(MODEL_REGISTRY['worker'].model, batch, encoding_kwargs)
    
    def empty_cache(self):
        for rank in range(self.args.world_size):
            rpc.rpc_sync(rank, _cuda_empty_cache)
            # rpc.rpc_sync(rank, torch.cuda.empty_cache)


def _cuda_empty_cache():
    if local_rank := os.getenv("LOCAL_RANK", None):
        with torch.cuda.device(f'cuda:{local_rank}'):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()


def call_batch_encode(model: CrossEncoder, batch: dict | BatchEncoding, encoding_kwargs: dict):
    """ Call model encode with tokenized batch inputs, do necessary post-processings: 
        1. Reranked scores (logits): Move to cpu.
    """
    batch = move_to_device(batch, device=model.lm.device)
    with torch.no_grad(), torch.autocast(device_type="cuda"), torch.cuda.device(model.lm.device):
        lm_out = model.forward(batch, **encoding_kwargs)
    return lm_out.logits.detach().squeeze().cpu()


@dataclass
class EncodeCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: str = 'longest_first'
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, text_pairs: list[dict[str, str]]) -> dict[str, Tensor]:
        """ Tokenize for reranker.
            Args:
                text_pairs (list[dict[str, str]]):  List of `text pairs` to rerank. Format: {"text1": str, "text2": str}
        """
        text_pairs_unpacked = [[pair["text1"], pair["text2"]] for pair in text_pairs]
        encoded = dict(self.tokenizer(
            text_pairs_unpacked,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            # return_token_type_ids=False,
            return_tensors=self.return_tensors,
        ))
        return encoded
