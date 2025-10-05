#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Hybrid search based on Faiss (dense) and Tantivy (sparse)

@Time    :   2024/09/03 17:21:29
@Author  :   Ma (Ma787639046@outlook.com)
'''

import time
import heapq
import logging
import numpy as np
from typing import Optional

import torch
from torch import Tensor

import datasets

from .score_fuse_utils import fuse_scores_linear, fuse_scores_rrf

logger = logging.getLogger(__name__)

class HybridSearch:
    def __init__(
        self, 
        model, 
        batch_size: int = 128, 
        corpus_chunk_size: Optional[int] = None, 
        use_multiple_gpu: bool = False,      # Faiss GPU
        faiss_search_map: str = "flat",
        sparse_search_map: str = "anserini",
        score_fuse_method: str = "linear",  # Choose among ['linear', 'rrf']
        fuse_weights: list[float] = [0.7, 0.3],     # Weights to fuse two scores
        return_all_results: bool = False,      # Return dict results of hybrid retrieval, containing single results
        **kwargs,
    ):
        self.model = model  # Model is class that provides encode_corpus() and encode_queries()
        self.batch_size = batch_size
        self.corpus_chunk_size = batch_size * 800 if corpus_chunk_size is None else corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.score_fuse_method = score_fuse_method
        self.fuse_weights = fuse_weights
        self.return_all_results = return_all_results
        
        # ** Dense Index **
        if faiss_search_map == "binary":
            from .faiss_search import BinaryFaissSearch
            den_cls = BinaryFaissSearch
        elif faiss_search_map == "flat":
            from .faiss_search import FlatIPFaissSearch
            den_cls = FlatIPFaissSearch
        elif faiss_search_map == "hnsw":
            from .faiss_search import HNSWFaissSearch
            den_cls = HNSWFaissSearch
        elif faiss_search_map == "hnswsq":
            from .faiss_search import HNSWSQFaissSearch
            den_cls = HNSWSQFaissSearch
        elif faiss_search_map == "pca":
            from .faiss_search import PCAFaissSearch
            den_cls = PCAFaissSearch
        elif faiss_search_map == "pq":
            from .faiss_search import PQFaissSearch
            den_cls = PQFaissSearch
        elif faiss_search_map == "sq":
            from .faiss_search import SQFaissSearch
            den_cls = SQFaissSearch
        else:
            raise NotImplementedError(f"Unsupported faiss_search_map {faiss_search_map}")

        ## Dense
        self.dense_search = den_cls(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size, use_multiple_gpu=use_multiple_gpu, **kwargs)

        # ** Sparse Index **
        # Note: Import on-demand so that we can install only the package we need.
        if sparse_search_map == "anserini":
            from .anserini_search import AnseriniSearch
            spr_cls = AnseriniSearch
        else:
            raise NotImplementedError(f"Unsupported sparse_search_map {sparse_search_map}")
        
        self.sparse_search = spr_cls(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size, **kwargs)
    
    @classmethod
    def name(self):
        return "hybrid_search"
    
    def encode(self, sentences: list[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs):
        return self.model.encode(sentences=sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def encode_queries(self, queries: list[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> list[str]:
        return self.model.encode_queries(queries=queries, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> list[str]:
        return self.model.encode_corpus(corpus=corpus, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def _clear(self, dense=True, sparse=True):
        """ Clear all pre-built index """
        if sparse:
            self.sparse_search._clear()
        
        if dense:
            self.dense_search._clear()

    def index(self, corpus_emb: dict[str, any], corpus_ids: list[str]):
        use_dense_retrieval = "dense_reps" in corpus_emb and corpus_emb["dense_reps"] is not None
        use_sparse_retrieval = "sparse_reps" in corpus_emb and corpus_emb["sparse_reps"] is not None
        assert isinstance(corpus_emb, dict) and (use_dense_retrieval or use_sparse_retrieval)

        if use_dense_retrieval:
            # TODO: Dense index func does not support incremental index. Should refactor it in future.
            logger.info(f"Dense Indexing...")
            self.dense_search.index(corpus_emb["dense_reps"], corpus_ids)
        
        if use_sparse_retrieval:
            logger.info(f"Sparse Dumping...")
            self.sparse_search.index(corpus_emb["sparse_reps"], corpus_ids)
    
    def retrieve_with_emb(
        self, 
        query_emb: dict[str, Tensor | list[str]],
        query_ids: list[str],
        top_k: int,
        dense: bool = True,
        sparse: bool = True,
        **kwargs
    ):
        """
        Retrieve with query embeddings. Please first index all document embeddings,
        then retrieve using this fuction.

        Inputs:
            query_emb (dict[str, Tensor | list[str]]): Query embeddings with shape [batch_size, hidden_dim]
            query_ids (list[str]): list of `query-ids`
            top_k (int): Threthod
            dense (bool): Whether to perform dense vector search
            sparse (bool): Whether to perform sparse vector search / dense-sparse fusion

        Returns:
            dict of `qid -> pid -> score`
        """
        use_dense_retrieval = "dense_reps" in query_emb and query_emb["dense_reps"] is not None
        use_sparse_retrieval = "sparse_reps" in query_emb and query_emb["sparse_reps"] is not None
        use_emb_retrieval = "emb_reps" in query_emb  and query_emb["emb_reps"] is not None
        use_token_id_retrieval = "token_id_reps" in query_emb and query_emb["token_id_reps"] is not None
        assert isinstance(query_emb, dict) and (use_dense_retrieval or use_sparse_retrieval or use_emb_retrieval or use_token_id_retrieval)
        assert dense or sparse, "Please indicate retrieval embedding types."

        results = {}

        if dense:
            if use_dense_retrieval:
                dense_results = self.dense_search.retrieve_with_emb(query_emb["dense_reps"], query_ids, top_k=top_k)
                results["den"] = dense_results
            
            if use_emb_retrieval:
                emb_results = self.dense_search.retrieve_with_emb(query_emb["emb_reps"], query_ids, top_k=top_k)
                results["emb"] = emb_results
        
        if sparse:
            if use_token_id_retrieval:
                imbalanced_sparse_results = self.sparse_search.retrieve_with_emb(query_emb["token_id_reps"], query_ids, top_k=top_k)
                results["tok"] = imbalanced_sparse_results
            
            if use_sparse_retrieval:
                sparse_results = self.sparse_search.retrieve_with_emb(query_emb["sparse_reps"], query_ids, top_k=top_k)
                results["spr"] = sparse_results
            
            if use_dense_retrieval and use_sparse_retrieval:
                dense_sparse_results = self._fuse_results(dense_results, sparse_results, weights=self.fuse_weights)
                results["den_spr"] = dense_sparse_results
            
            if use_emb_retrieval and use_token_id_retrieval:
                imb_embspr_results = self._fuse_results(emb_results, imbalanced_sparse_results, weights=self.fuse_weights)
                results["emb_tok"] = imb_embspr_results
                results["default"] = results["emb_tok"]  # Backward compatiable, output one result as main results
        
        return results
    
    def _add_to_heap(
        self, 
        sub_results: dict[str, dict[str, float]],
        result_heaps: dict[str, list[tuple[float, str]]],
        top_k: int,
        ignore_identical_ids: bool,
    ):
        """ Update results to heaps, then return the result_heaps. """
        # Step4: Collect results
        for qid, pid_to_score in sub_results.items():
            for pid, score in pid_to_score.items():
                if ignore_identical_ids and (qid == pid):   # Ignore identical ids
                    continue
                if qid not in result_heaps:
                    result_heaps[qid] = []

                if len(result_heaps[qid]) < top_k:
                    # Push item on the heap
                    heapq.heappush(result_heaps[qid], (score, pid))
                else:
                    # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                    heapq.heappushpop(result_heaps[qid], (score, pid))
        
        return result_heaps
    
    def _fuse_results(self, dense_results=None, sparse_results=None, weights: list[int] = [0.7, 0.3]):
        # Step5: Rank fusion
        # Note: There are multiple rank fusion methods:
        # 1) Reciprocal Rank Fusion (RRF): Fusing merely based on ranks
        #       Please see: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
        # 2) Linear combination of scores: Linear add the scores.
        #           alpha_den * dense_score + alpha_spr * sparse_score
        #       However, scores from different systems may not compariable. Such as, a dense_score
        #       with cosine similarity ranges in [-1, 1]. But a sparse_score of BM25 may ranges above
        #       10+, even 100+. Thus the linear combination here first normalize the scores to range
        #       [0, 1], then fuse them with linear interpolation .
        if dense_results is None and sparse_results is not None:
            results = sparse_results
        elif dense_results is not None and sparse_results is None:
            results = dense_results
        elif dense_results is not None and sparse_results is not None:
            if self.score_fuse_method == "rrf":
                results = fuse_scores_rrf([dense_results, sparse_results])
            elif self.score_fuse_method == "linear":
                results = fuse_scores_linear([dense_results, sparse_results], weights=weights)
            else:
                raise NotImplementedError(f"score_fuse_method {self.score_fuse_method} is not supported.")
        else:
            raise ValueError("All scores are None. Please check model settings.")
        
        return results

    def search(
        self, 
        corpus: dict[str, dict[str, str]] | datasets.Dataset,
        queries: dict[str, str] | datasets.Dataset, 
        top_k: int = 1000,
        score_function: str = None,     # Unused
        return_sorted: bool = False,    # Unused
        ignore_identical_ids: bool = False,
        **kwargs
    ) -> dict[str, dict[str, float]] | dict[str, dict[str, dict[str, float]]]:
        # Step1: Encoding
        logger.info("Encoding Queries...")
        if isinstance(queries, dict):
            query_ids = list(queries.keys())
            queries_list = [queries[qid] for qid in queries]
        elif isinstance(queries, datasets.Dataset):
            id_name = None
            for _name_choice in ["id", "_id", "query_id"]: 
                if _name_choice in queries.column_names:
                    id_name = _name_choice
            if id_name is None:
                raise KeyError(f"No id column in queries dataset {queries.column_names}")
            query_ids = queries[id_name]
            queries_list = queries
        else:
            raise NotImplementedError(f"Unrecognized type {type(queries)}")
        
        query_embeddings: dict[str, Tensor | np.ndarray | dict | list[str]] = self.model.encode_queries(
            queries_list, 
            batch_size=self.batch_size, 
            show_progress_bar=self.show_progress_bar, 
            convert_to_tensor=self.convert_to_tensor
        )
        use_dense_retrieval = "dense_reps" in query_embeddings
        use_sparse_retrieval = "sparse_reps" in query_embeddings
        use_emb_retrieval = "emb_reps" in query_embeddings
        use_token_id_retrieval = "token_id_reps" in query_embeddings
        assert isinstance(query_embeddings, dict) and (use_dense_retrieval or use_sparse_retrieval or use_emb_retrieval or use_token_id_retrieval)

        logger.info("Sorting Corpus by document length (Longest first)...")
        if isinstance(corpus, dict):
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("text", "")) if isinstance(corpus[k], dict) else len(corpus[k]), reverse=True)
            corpus = [corpus[cid] for cid in corpus_ids]
        elif isinstance(corpus, datasets.Dataset):
            corpus = corpus.map(
                lambda x: {"_length": len(x["text"])}, 
                num_proc=max(2, min(len(corpus) // 100000, 12))
            )
            corpus = corpus.sort("_length", reverse=True).remove_columns("_length")

            id_name = None
            for _name_choice in ["id", "_id", "docid", "doc_id"]: 
                if _name_choice in corpus.column_names:
                    id_name = _name_choice
            if id_name is None:
                raise KeyError(f"No id column in corpus dataset {corpus.column_names}")
            corpus_ids = corpus[id_name]
        else:
            raise NotImplementedError(f"Unrecognized type {type(corpus)}")
        
        num_corpus = len(corpus)
        logger.info(f"Number of corpus: {num_corpus}")

        if use_dense_retrieval:
            logger.info(f"Estimated Dense Index Size ({str(query_embeddings['dense_reps'].dtype)}): {human_readable_size(query_embeddings['dense_reps'].itemsize * query_embeddings['dense_reps'].shape[-1] * num_corpus)}")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        # Keep only the top-k docs for each query
        ## Dense: (Encode -> Index -> Retrieve) in chunks
        ## Sparse: (Encode -> Index) in chunks -> Retrieve at end
        if use_dense_retrieval:
            dense_result_heaps: dict[str, list[tuple[float, str]]] = {qid: [] for qid in query_ids}
        if use_emb_retrieval:
            emb_result_heaps: dict[str, list[tuple[float, str]]] = {qid: [] for qid in query_ids}
        
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Step1: Encode chunk of corpus
            if isinstance(corpus, list):
                corpus_sliced = corpus[corpus_start_idx:corpus_end_idx]
            else:
                corpus_sliced = corpus.select(range(corpus_start_idx, corpus_end_idx))
            
            sub_corpus_embeddings: dict[str, Tensor | np.ndarray | dict | list[str]] = self.model.encode_corpus(
                corpus_sliced,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor=self.convert_to_tensor
            )
            assert isinstance(sub_corpus_embeddings, dict)
            if use_dense_retrieval or use_emb_retrieval:
                assert "dense_reps" in sub_corpus_embeddings
            if use_sparse_retrieval or use_token_id_retrieval:
                assert "sparse_reps" in sub_corpus_embeddings

            # Step2: Index
            self.index(sub_corpus_embeddings, corpus_ids[corpus_start_idx:corpus_end_idx])

            # Step3: Retrieve
            # Note: Only dense retrieval supports indexing and retrieving in chunks!!
            sub_results = self.retrieve_with_emb(query_embeddings, query_ids, top_k=top_k, dense=True, sparse=False)
            if use_dense_retrieval:
                self._add_to_heap(sub_results["den"], dense_result_heaps, top_k=top_k, ignore_identical_ids=ignore_identical_ids)
            if use_emb_retrieval:
                self._add_to_heap(sub_results["emb"], emb_result_heaps, top_k=top_k, ignore_identical_ids=ignore_identical_ids)
            
            self._clear(dense=True, sparse=False) # Remember to clear all index

        # Step5: Rank fusion
        def _parse_heap_results(result_heaps: dict[str, list[tuple[float, str]]]):
            """ Convert result heap to format of dict of `qid -> pid -> score` """
            _results: dict[str, dict[str, float]] = {}
            for qid in result_heaps.keys():
                if qid not in _results:
                    _results[qid] = {}
                for score, corpus_id in result_heaps[qid]:
                    _results[qid][corpus_id] = score
            return _results
        
        dense_results = _parse_heap_results(dense_result_heaps) if use_dense_retrieval else None
        emb_results = _parse_heap_results(emb_result_heaps) if use_emb_retrieval else None

        imbalanced_sparse_results = None
        if use_token_id_retrieval:
            logger.info("Imbalanced Sparse Retrieving...")
            imbalanced_sparse_results = self.sparse_search.retrieve_with_emb(query_emb=query_embeddings["token_id_reps"], query_ids=query_ids, top_k=top_k)
        
        sparse_results = None
        if use_sparse_retrieval:
            # Step3: Retrieve
            logger.info("Sparse Retrieving...")
            sparse_results = self.sparse_search.retrieve_with_emb(query_emb=query_embeddings["sparse_reps"], query_ids=query_ids, top_k=top_k)
        
        if use_dense_retrieval and use_sparse_retrieval:
            dense_sparse_results = self._fuse_results(dense_results, sparse_results, weights=self.fuse_weights)
        
        if use_emb_retrieval and use_token_id_retrieval:
            emb_tok_id_results = self._fuse_results(emb_results, imbalanced_sparse_results, weights=self.fuse_weights)
        
        self._clear()   # Remember to clear all index
        
        # Note:
        # MTEB does not support return multiple emb results at once
        # If we are evaluating MTEB, please set return_all_results=False to only return the default results.
        results = {}
        default_results = None
        if use_dense_retrieval:
            results["den"] = dense_results
            default_results = dense_results
        if use_sparse_retrieval:
            results["spr"] = sparse_results
            default_results = sparse_results
        if use_emb_retrieval:
            results["emb"] = emb_results
            default_results = emb_results
        if use_token_id_retrieval:
            results["tok"] = imbalanced_sparse_results
            default_results = imbalanced_sparse_results
        if use_dense_retrieval and use_sparse_retrieval:
            results["den_spr"] = dense_sparse_results
            default_results = dense_sparse_results
        if use_emb_retrieval and use_token_id_retrieval:
            results["emb_tok"] = emb_tok_id_results
            default_results = emb_tok_id_results
        
        return results if self.return_all_results else default_results

def human_readable_size(size_bytes: int) -> str:
    """ Parse a number in Bytes to human readable string format """
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {units[i]}"
