#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A script to evaluate the model on MTEB benchmark.
#
# @Author: Ma
#
import os
from pathlib import Path

import logging
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if (int(os.getenv("RANK", -1)) in [0, -1]) else logging.WARN,
    force=True,
)
logger = logging.getLogger(__name__)

BASE_FOLDER_PATH = os.path.split(os.path.realpath(__file__))[0]
ROOT_PATH = str(Path(BASE_FOLDER_PATH).parent.parent.absolute())

import torch
import torch.nn as nn

from eval_arguments import EvalArguments
from eval_utils import launch_eval, init_searcher
from prompts import get_mteb_prompt

import mteb
from mteb.model_meta import ModelMeta
from mteb.overview import get_task, filter_tasks_by_task_types, filter_tasks_by_languages
from mteb.benchmarks.get_benchmark import get_benchmark

# Monkey patch mteb.evaluation.evaluators.RetrievalEvaluator
from mteb_utils.BM25FixedInstructionRetrievalEvaluator import BM25FixedInstructionRetrievalEvaluator
mteb.evaluation.evaluators.InstructionRetrievalEvaluator.InstructionRetrievalEvaluator.__call__ = BM25FixedInstructionRetrievalEvaluator.__call__

def create_output_folder(
    self, model_meta: ModelMeta, output_folder: str | None
) -> Path | None:
    """Patch create output folder for the results."""
    if output_folder is None:
        return None

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

mteb.evaluation.MTEB.create_output_folder = create_output_folder


def call_evaluate(args: EvalArguments, model: nn.Module):
    # `model` handles multi-node multi-GPU computation and return embeddings
    # `searcher` handles text formatting, processing, call 
    # model.encode_queries/encode_corpus(input_ids: Tensor, attention_mask: Tensor, ...), 
    # and search logic.
    #
    # In this example, 
    # `model` is `PytorchRPCExactSearchModel` in `src/lightretriever/inference/exact_search_torchrpc.py`
    # `searcher` is `HybridSearch` in `src/lightretriever/retriever/hybrid_search.py`
    searcher = init_searcher(args, model)

    # MTEB does not support customized similarity functions
    # Hacking with customized search function to support our customized
    # retrievers & similarity functions
    # mteb/evaluation/evaluators/RetrievalEvaluator.py#478
    hackable_meta_of_mteb = mteb.models.bm25.bm25_s
    searcher.mteb_model_meta = hackable_meta_of_mteb

    # Get all tasks
    ## 1. BeIR:  benchmark_name == "BEIR"
    ## 2. CMTEB Retrieval:  benchmark_name == "CMTEB-R", which equals to benchmark_name == "MTEB(cmn, v1)" && task_type == "Retrieval"
    if args.benchmark_name:
        if args.benchmark_name == "CMTEB-R":
            args.benchmark_name = "MTEB(cmn, v1)"
            args.task_type = "Retrieval"
        benchmark = get_benchmark(args.benchmark_name)
        all_tasks = benchmark.tasks
    elif args.task_name:
        all_tasks = [get_task(args.task_name, languages=args.langs)]
    else:
        raise ValueError("Please specify either benchmark_name or task_name.")
    
    if args.task_type:
        all_tasks = filter_tasks_by_task_types(all_tasks, task_types=[args.task_type])
    
    if args.langs:
        all_tasks = filter_tasks_by_languages(all_tasks, languages=args.langs)
    
    for task_cls in all_tasks:
        logger.info(f"Loading task {task_cls}...")
        task_name: str = task_cls.metadata.name
        task_type: str = task_cls.metadata.type

        model.query_prompt, model.corpus_prompt = "", ""
        if args.add_prompt:
            model.query_prompt, model.corpus_prompt = get_mteb_prompt(task_name=task_name, task_type=task_type, prompt_type=args.prompt_type)
            
            logger.info('Set query prompt: \n{}\n\ncorpus prompt: \n{}\n'.format(model.query_prompt, model.corpus_prompt))

        sub_eval = mteb.MTEB(tasks=[task_cls])
        kwargs_sub_eval = {
            "verbosity": 1,
            "output_folder": args.output_dir,
            "overwrite_results": args.overwrite_results, 
            "corpus_chunk_size": args.corpus_chunk_size,
            "k_values": args.k_values,
            "top_k": args.top_k,
            "score_function": args.score_function,
            "save_predictions": args.save_predictions,
        }

        if not args.pred_load_folder:   # No previous predictions
            sub_results = sub_eval.run(searcher, **kwargs_sub_eval)
        else:   # Load previous predictions
            for subset in task_cls.hf_subsets:
                sub_results = sub_eval.run(
                    searcher, **kwargs_sub_eval,
                    eval_subsets=[subset],
                    previous_results=args.pred_load_folder / f"{task_cls.metadata.name}_{subset}_predictions.json",
                )

        # Avoid MEM leaks
        if hasattr(task_cls, "corpus"):
            del task_cls.corpus
        if hasattr(task_cls, "queries"):
            del task_cls.queries
        if hasattr(task_cls, "relevant_docs"):
            del task_cls.relevant_docs
        
        del sub_eval
        del task_cls

    logger.info("--DONE--")


if __name__ == '__main__':
    launch_eval(call_evaluate)
