import logging
from typing import Optional
from dataclasses import dataclass, field

from lightretriever.inference.arguments import InferenceArguments

logger = logging.getLogger(__name__)

@dataclass
class EvalArguments(InferenceArguments):
    """
    Eval Arguments for benchmarks.
    """
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Folder to save the output results."}
    )
    benchmark_name: Optional[str] = field(
        default=None, metadata={
            "help": "Benchmark name of one task collection. "
                    "Note: set one of benchmark_name and task_name is okay, please do not set them both. "
                    " - If benchmark_name is set, then eval all tasks in the benchmark. "
                    " - If task_name is set, then eval the single task. "
        }
    )
    task_name: Optional[str] = field(
        default=None, metadata={
            "help": "Single task name to evaluate. "
                    "Note: set one of benchmark_name and task_name is okay, please do not set them both. "
        }
    )
    task_type: Optional[str] = field(
        default=None, metadata={
            "help": "Pre-defined task type for evaluting multiple tasks. For now, only retrieval task is involved."
        }
    )
    langs: Optional[list[str]] = field(
        default=None, metadata={"help": "Languages to evaluate."}
    )
    corpus_chunk_size: Optional[int] = field(
        default=10_000_000, metadata={"help": "Corpus chunk size for encoding, index and retrieval. Default: 10M."}
    )
    add_prompt: bool = field(
        default=False, metadata={"help": "Whether to add query prompt."}
    )
    prompt_type: str = field(
        default="e5", metadata={"help": "Type of query prompt, choose among e5, instructor, bge, e5_ori."}
    )
    fuse_weights: list[float] = field(
        default_factory=lambda: [0.7, 0.3], metadata={
            "help": "Weights to linear fuse two embedding types' scores. Default to 0.7 for emb (Light Dense), "
                    " 0.3 for tok (Light Sparse). Optimal performances counld be further tuned based on tasks and models."
        }
    )
    return_all_results: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all results when using a Hybrid retriever. "
                    "Only used in evaluation."
        },
    )
    overwrite_results: bool = field(
        default=False, metadata={"help": "Whether to override the existing results."}
    )
    save_predictions: bool = field(
        default=False, metadata={
            "help": "Whether to save preds. MTEB will save the json predictions"
                    "to `(output_folder)/(self.metadata.name)_(hf_subset)_predictions.json`"
        }
    )
    k_values: list[int] = field(
        default_factory=lambda: [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000], metadata={"help": "The top-k scores to retrieve or rerank."}
    )
    top_k: int = field(
        default=1000, metadata={"help": "The top-k threshold to retrieve or rerank."}
    )
    pred_load_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder to load the bi-encoder retrieval results from a local json file. Currently only support retrieval tasks."}
    )

    def __post_init__(self):
        super().__post_init__()
        
        if self.top_k < max(self.k_values):
            self.k_values = [i for i in self.k_values if i <= self.top_k]