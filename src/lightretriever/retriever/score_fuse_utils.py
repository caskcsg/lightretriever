import numpy as np

def fuse_scores_rrf(
    results_list: list[dict[str, dict[str, float]]],
    k: int=60,
):
    """
    Reciprocal Rank Fusion
    Ref: https://dl.acm.org/doi/10.1145/1571941.1572114
            https://github.com/Raudaschl/rag-fusion
            https://safjan.com/implementing-rank-fusion-in-python/

    :param results_list: list[dict[str, dict[str, float]]], a list of retrieval results with
                            dict[query_id -> dict[passage_id -> score]]
    :param k: a ranking constant, controling ranking decrease speed, default to 60
    :return: dict[str, dict[str, float]]. 
            Fused results of dict[query_id -> dict[passage_id -> fused_score]]
    """
    fused_results: dict[str, dict[str, float]] = {}

    for system_results in results_list:
        for query_id, passages in system_results.items():
            if not isinstance(query_id, str):
                query_id = str(query_id)
            
            if query_id not in fused_results:
                fused_results[query_id] = {}

            passage_ids = list(passages.keys())
            scores = np.array([float(passages[pid]) for pid in passage_ids])

            # Sorting in desend order to get ranks: [1, 2, ...]
            sorted_indices = np.argsort(-scores)
            sorted_passage_ids = np.array(passage_ids)[sorted_indices]
            ranks = np.arange(1, len(sorted_passage_ids) + 1)

            # Compute RRF score
            rrf_scores = 1 / (k + ranks)

            for pid, rrf_score in zip(sorted_passage_ids, rrf_scores):
                pid = str(pid)
                if pid not in fused_results[query_id]:
                    fused_results[query_id][pid] = 0.0
                fused_results[query_id][pid] += float(rrf_score)
    
    return fused_results

def fuse_scores_linear(
    results_list: list[dict[str, dict[str, float]]],
    weights: list[int] = [0.7, 0.3],
    eps: float = 1e-8,
):
    """
    Linear interpolation of results with weights
    1. Each scores (q-p scores) of a query will be normalized to [0, 1] by `(i - min) / (max-min)`.
    2. Scores are fused by `score_i * weight_i`

    :param results_list: list[dict[str, dict[str, float]]], a list of retrieval results with
                            dict[query_id -> dict[passage_id -> score]]
    :param weights: The weights to fuse each scores.
    :param eps: Term added to the denominator to improve numerical stability.
    :return: dict[str, dict[str, float]]. 
            Fused results of dict[query_id -> dict[passage_id -> fused_score]]
    """
    assert len(results_list) == len(weights)
    fused_results: dict[str, dict[str, float]] = {}

    for system_results, weight in zip(results_list, weights):
        for query_id, passages in system_results.items():
            if not isinstance(query_id, str):
                query_id = str(query_id)
            
            if query_id not in fused_results:
                fused_results[query_id] = {}

            passage_ids = list(passages.keys())
            scores = np.array([float(passages[pid]) for pid in passage_ids])

            # Linear normalize: (i - min) / (max-min)
            min_val = np.min(scores)
            max_val = np.max(scores)
            scores_normed = (scores - min_val) / (max_val - min_val + eps)
            scores_weighted = scores_normed * weight

            for pid, score in zip(passage_ids, scores_weighted):
                pid = str(pid)
                if pid not in fused_results[query_id]:
                    fused_results[query_id][pid] = 0.0
                fused_results[query_id][pid] += float(score)
    
    return fused_results