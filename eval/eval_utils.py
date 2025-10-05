import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed as dist
torch.set_float32_matmul_precision('high')

from transformers import HfArgumentParser, set_seed
from sentence_transformers import SentenceTransformer
from lightretriever.inference.utils import DEVICE_TYPE, DIST_BACKEND
from lightretriever.inference.exact_search_torchrpc import PytorchRPCExactSearchModel
from lightretriever.inference.rerank import RerankerModel
from lightretriever.inference.dummy import DummyModel

from eval_arguments import EvalArguments


def init_searcher(args: EvalArguments, model: PytorchRPCExactSearchModel | RerankerModel | SentenceTransformer | DummyModel):
    """ Init searcher based on model type. Searcher handles text formatting, processing, 
        call model.encode_queries/encode_corpus(input_ids: Tensor, attention_mask: Tensor, ...), 
        and search logic.

        Note:
            1. RPC Searcher: RPC Master handles search logic. Only MASTER rank need to init RPC searcher.
            2. Single/Distributed Searcher (TODO): Models on each rank handle search logic. No need to init another searcher. We just use model as searcher.

        Args:
            args: EvalArguments
            model: PytorchRPCExactSearchModel | RerankerModel | SentenceTransformer | DummyModel
        Returns:
            searcher: HybridSearch
    """
    if args.model_type in ["RerankerModel", "SentenceTransformer"]:
        # RerankerModel and SentenceTransformer does not use searcher
        return model
    
    if args.inference_arch == "PytorchRPCExactSearchModel":
        # RPC Searcher: RPC Master handles search logic
        # Note: 
        # 1. HybridSearch uses Faiss-gpu only on CUDA device
        if args.model_type == "HybridModel":
            from lightretriever.retriever.hybrid_search import HybridSearch
            searcher = HybridSearch(
                model, 
                batch_size=args.batch_size, 
                corpus_chunk_size=args.corpus_chunk_size, 
                use_multiple_gpu=True if DEVICE_TYPE == "cuda" else False, 
                fuse_weights=args.fuse_weights,
                return_all_results=args.return_all_results
            )
        elif args.model_type == "EncoderModel":
            from lightretriever.retriever.faiss_search import FlatIPFaissSearch
            searcher = FlatIPFaissSearch(
                model, 
                args.batch_size, 
                corpus_chunk_size=args.corpus_chunk_size, 
                use_multiple_gpu=True
            )
        elif args.model_type == "DummyModel":
            from lightretriever.retriever.anserini_search import AnseriniSearch
            searcher = AnseriniSearch(
                model,
                args.batch_size, 
                corpus_chunk_size=args.corpus_chunk_size, 
                # Anserini related
                anserini_lang=args.anserini_lang,
                anserini_vector_type=args.anserini_vector_type,
                anserini_pretokenized=args.anserini_pretokenized,
                anserini_impact_search=args.anserini_impact_search,
                anserini_bm25_k1=args.anserini_bm25_k1,
                anserini_bm25_b=args.anserini_bm25_b,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
    
    else:
        raise ValueError(f"Unknown inference archeticture: {args.inference_arch}")
    
    return searcher


def launch_eval(eval_entry_fn: Callable[[EvalArguments, nn.Module], None]):
    """ Main entry of evaluation. Execute launch_eval on all ranks.

    Args:
        eval_entry_fn: Callable[[EvalArguments, nn.Module], None]. It takes model as inputs, and handle 
                        all the evaluation logic.
    """

    def _parse_args() -> EvalArguments:
        # Wrap args parsing in this function to support type hint
        parser = HfArgumentParser(EvalArguments)
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            (args, ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            (args, ) = parser.parse_args_into_dataclasses()
        return args

    args = _parse_args()
    set_seed(args.seed)

    # Set device
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_device(args.local_rank)
    elif DEVICE_TYPE == "npu":
        torch.npu.set_device(args.local_rank)

    # Init rpc/dist backends
    if args.inference_arch == "PytorchRPCExactSearchModel":
        # Initialize RPC
        if ":" in args.master_addr: # IPV6
            tcp_addr = f"tcp://[{args.master_addr}]:{args.master_port}"
        else:
            tcp_addr = f"tcp://{args.master_addr}:{args.master_port}"
        
        if DEVICE_TYPE == "cuda":
            # # If InfiniBand is working, just use default backend
            # backend = None
            # rpc_backend_options = None

            # Otherwise, disiable InfiniBand
            backend = rpc.backend_registry.BackendType["TENSORPIPE"]
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                init_method=tcp_addr,
                # Disable RDMA to avoid stuck
                # Ref: https://github.com/pytorch/tensorpipe/issues/457#issuecomment-1278720956
                _transports=[
                    # "ibv",      # InfiniBand
                    "shm",      # Shared Memory
                    "uv",       # libuv - TCP
                ],
                _channels=[
                    # "cuda_gdr",     # InfiniBand
                    "cuda_xth",     # CUDA Driver
                    "cuda_ipc",     # CUDA Inter-Process Communication
                    "cuda_basic",    # Basic CUDA API
                    "cma",      # Cross-Memory Attach
                    "mpt_uv",   # Multi-Protocol Transport over UV
                    "basic",      
                ],
            )
        elif DEVICE_TYPE == "npu":
            backend=rpc.backend_registry.BackendType["NPU_TENSORPIPE"]
            rpc_backend_options = None
        else:
            # CPU/MPS: Use default backend
            backend = None
            rpc_backend_options = None
        
        rpc.init_rpc(
            name=f"worker{args.rank}",
            backend=backend,
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=rpc_backend_options,
        )
        logger.warning(f"[Rank{args.rank}] RPC Backends init successful.")

        # Local model on every ranks        
        start_time = time.time()
        if args.model_type == "RerankerModel":
            model = RerankerModel(args)
        elif args.model_type == "SentenceTransformer":
            model = SentenceTransformer(args.model_name_or_path)
        elif args.model_type == "DummyModel":
            model = DummyModel()
        else:
            model = PytorchRPCExactSearchModel(args)
        logger.info(f"[Rank{args.rank}] Model loaded in {time.time()-start_time}s.")

        # Wait for model loading finished
        rpc.api._wait_all_workers(60)

        # Main process handles all data loading, processing, 
        # distribution, gathering encoded outputs and calculate
        # metrics
        if args.rank == 0:
            # Search!
            eval_entry_fn(args, model)
        
        # Block until all RPCs are done
        rpc.shutdown()

    else:
        raise ValueError(f"Unknown inference archeticture: {args.inference_arch}")