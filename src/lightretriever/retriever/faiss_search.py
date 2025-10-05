#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Copied from beir/retrieval/search/dense with some api modifications

@Author  :   SentenceTransformers Team

@Edit    :   Ma (Ma787639046@outlook.com)

'''
import os
import csv
import heapq
import numpy as np
from typing import Optional
from tqdm.autonotebook import tqdm

import faiss
from torch import Tensor

import datasets

from .faiss_index import FaissBinaryIndex, FaissTrainIndex, FaissHNSWIndex, FaissIndex

import logging
logger = logging.getLogger(__name__)

def save_dict_to_tsv(_dict: dict, output_path: str, keys=[]):
    with open(output_path, 'w') as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if keys: writer.writerow(keys)
        for key, value in _dict.items():
            writer.writerow([key, value])

def load_tsv_to_dict(input_path: str, header=True):
    mappings = {}
    reader = csv.reader(open(input_path, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    if header: next(reader)
    for row in reader: 
        mappings[row[0]] = int(row[1])
    
    return mappings

#Parent class for any faiss search
class DenseRetrievalFaissSearch:
    def __init__(
            self, 
            model, 
            batch_size: int = 128, 
            corpus_chunk_size: Optional[int] = None, 
            use_single_gpu: bool = False, 
            use_multiple_gpu: bool = False,
            **kwargs
        ):
        self.model = model  # Model is class that provides encode_corpus() and encode_queries()
        self.batch_size = batch_size
        self.corpus_chunk_size = batch_size * 800 if corpus_chunk_size is None else corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.mapping_tsv_keys = ["beir-docid", "faiss-docid"]
        self.faiss_index: Optional[faiss.Index] = None
        self.use_single_gpu = use_single_gpu
        self.use_multiple_gpu = use_multiple_gpu
        self.dim_size = None
        self.mapping = {}
        self.rev_mapping = {}
    
    @classmethod
    def name(self):
        return "faiss_search"
    
    def encode(self, sentences: list[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs):
        return self.model.encode(sentences=sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def encode_queries(self, queries: list[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> list[str]:
        return self.model.encode_queries(queries=queries, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> list[str]:
        return self.model.encode_corpus(corpus=corpus, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def _create_mapping_ids(self, corpus_ids):
        if not all(isinstance(doc_id, int) for doc_id in corpus_ids):
            for idx in range(len(corpus_ids)):
                self.mapping[corpus_ids[idx]] = idx
                self.rev_mapping[idx] = corpus_ids[idx]
    
    def _clear(self):
        """ Clear all pre-built index """
        if self.faiss_index is not None:
            self.faiss_index.reset()
            del self.faiss_index
        
        self.faiss_index = None
        self.dim_size = None
        self.mapping = {}
        self.rev_mapping = {}
    
    def _load(self, input_dir: str, prefix: str, ext: str):
        # Load ID mappings from file
        input_mappings_path = os.path.join(input_dir, "{}.{}.tsv".format(prefix, ext))
        logger.info("Loading Faiss ID-mappings from path: {}".format(input_mappings_path))
        self.mapping = load_tsv_to_dict(input_mappings_path, header=True)
        self.rev_mapping = {v: k for k, v in self.mapping.items()}
        passage_ids = sorted(list(self.rev_mapping))
        
        # Load Faiss Index from disk
        input_faiss_path = os.path.join(input_dir, "{}.{}.faiss".format(prefix, ext))
        logger.info("Loading Faiss Index from path: {}".format(input_faiss_path))
        
        return input_faiss_path, passage_ids

    def save(self, output_dir: str, prefix: str, ext: str):
        # Save BEIR -> Faiss ids mappings
        save_mappings_path = os.path.join(output_dir, "{}.{}.tsv".format(prefix, ext))
        logger.info("Saving Faiss ID-mappings to path: {}".format(save_mappings_path))
        save_dict_to_tsv(self.mapping, save_mappings_path, keys=self.mapping_tsv_keys)

        # Save Faiss Index to disk
        save_faiss_path = os.path.join(output_dir, "{}.{}.faiss".format(prefix, ext))
        logger.info("Saving Faiss Index to path: {}".format(save_faiss_path))
        self.faiss_index.save(save_faiss_path)
        logger.info("Index size: {:.2f}MB".format(os.path.getsize(save_faiss_path)*0.000001))
    
    def _index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str],
        ):
        # Build id mappings
        self._create_mapping_ids(corpus_ids)
        self.dim_size = corpus_emb.shape[1]
        faiss_ids = [self.mapping.get(corpus_id) for corpus_id in corpus_ids]
        return faiss_ids, corpus_emb

    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str]
        ):
        raise NotImplementedError("Base class function. Please implement this depands on index type.")
    
    def retrieve_with_emb(self, 
                          query_emb: np.ndarray,
                          query_ids: list[str],
                          top_k: int,
                          **kwargs
                          ):
        """
        Retrieve with query embeddings. Please first index all document embeddings,
        then retrieve using this fuction.

        Inputs:
            query_emb (np.ndarray): Query embeddings with shape [batch_size, hidden_dim]
            query_ids (list[str]): List of `query-ids`
            top_k (int): Threthod

        Returns:
            Dict of `qid -> pid -> score`
        """
        results: dict[str, dict[str, float]] = dict()   # qid -> pid -> score

        faiss_scores, faiss_doc_ids = self.faiss_index.search(query_emb, top_k, **kwargs)
        
        for idx in range(len(query_ids)):
            scores = [float(score) for score in faiss_scores[idx]]
            if len(self.rev_mapping) != 0:
                doc_ids = [self.rev_mapping[doc_id] for doc_id in faiss_doc_ids[idx]]
            else:
                doc_ids = [str(doc_id) for doc_id in faiss_doc_ids[idx]]
            results[query_ids[idx]] = dict(zip(doc_ids, scores))
        
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
    ) -> dict[str, dict[str, float]]:
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
        
        query_embeddings: Tensor | np.ndarray = self.model.encode_queries(
            queries_list, 
            batch_size=self.batch_size, 
            show_progress_bar=self.show_progress_bar, 
            convert_to_tensor=self.convert_to_tensor
        )

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
        logger.info(f"Estimated Dense Index Size ({str(query_embeddings.dtype)}): {human_readable_size(query_embeddings.itemsize * query_embeddings.shape[-1] * num_corpus)}")
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        # Keep only the top-k docs for each query
        ## Dense: Encode -> Index -> Retrieve in chunks
        dense_result_heaps: dict[str, tuple[float, str]] = {qid: [] for qid in query_ids}
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Step1: Encode chunk of corpus
            if isinstance(corpus, list):
                corpus_sliced = corpus[corpus_start_idx:corpus_end_idx]
            else:
                corpus_sliced = corpus.select(range(corpus_start_idx, corpus_end_idx))
            
            sub_corpus_embeddings: Tensor | np.ndarray = self.model.encode_corpus(
                corpus_sliced,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor=self.convert_to_tensor
            )
            if isinstance(sub_corpus_embeddings, dict):
                if "dense_reps" in sub_corpus_embeddings:
                    logger.warning(f"sub_corpus_embeddings uses `dense_reps` by default. Available keys: {sub_corpus_embeddings.keys()}")
                    sub_corpus_embeddings = sub_corpus_embeddings["dense_reps"]
                else:
                    raise ValueError(f"HybridModel: Return Multi-vector with keys {sub_corpus_embeddings.keys()}, but not `dense_reps`")

            # Step2: Indexing
            logger.info("Dense Indexing...")
            self.index(sub_corpus_embeddings, corpus_ids[corpus_start_idx:corpus_end_idx])
            
            # Step3: Retrieve
            logger.info("Dense Retrieving...")
            sub_results = self.retrieve_with_emb(query_embeddings, query_ids, top_k=top_k)            
            self._clear()   # Remember to clear all index

            # Step4: Collect results
            for qid, pid_to_score in sub_results.items():
                for pid, score in pid_to_score.items():
                    # Ignore identical ids
                    if ignore_identical_ids and (qid == pid):
                        continue

                    if len(dense_result_heaps[qid]) < top_k:
                        # Push item on the heap
                        heapq.heappush(dense_result_heaps[qid], (score, pid))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(dense_result_heaps[qid], (score, pid))

        dense_results: dict[str, dict[str, float]] = {}
        for qid in dense_result_heaps.keys():
            if qid not in dense_results:
                dense_results[qid] = {}
            for score, corpus_id in dense_result_heaps[qid]:
                dense_results[qid][corpus_id] = score
        
        return dense_results


class BinaryFaissSearch(DenseRetrievalFaissSearch):

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "bin"):
        passage_embeddings = []
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index_binary(input_faiss_path)
        logger.info("Reconstructing passage_embeddings back in Memory from Index...")
        for idx in tqdm(range(0, len(passage_ids)), total=len(passage_ids)):
            passage_embeddings.append(base_index.reconstruct(idx))            
        passage_embeddings = np.vstack(passage_embeddings)
        self.faiss_index = FaissBinaryIndex(base_index, passage_ids, passage_embeddings)

    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str],
        ):
        faiss_ids, corpus_embeddings = super()._index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)
        logger.info("Using Binary Hashing in Flat Mode!")
        logger.info("Output Dimension: {}".format(self.dim_size))
        base_index = faiss.IndexBinaryFlat(self.dim_size * 8)
        self.faiss_index = FaissBinaryIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "bin"):
        super().save(output_dir, prefix, ext)
    
    def get_index_name(self):
        return "binary_faiss_index"
    

class PQFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: Optional[int] = None, num_of_centroids: int = 96, 
                 code_size: int = 8, similarity_metric=faiss.METRIC_INNER_PRODUCT, use_rotation: bool = False, **kwargs):
        super(PQFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.num_of_centroids = num_of_centroids
        self.code_size = code_size
        self.similarity_metric = similarity_metric
        self.use_rotation = use_rotation
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "pq"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissTrainIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissTrainIndex(base_index, passage_ids)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str]
        ):
        faiss_ids, corpus_embeddings = super()._index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)  

        logger.info("Using Product Quantization (PQ) in Flat mode!")
        logger.info("Parameters Used: num_of_centroids: {} ".format(self.num_of_centroids))
        logger.info("Parameters Used: code_size: {}".format(self.code_size))          
        
        base_index = faiss.IndexPQ(self.dim_size, self.num_of_centroids, self.code_size, self.similarity_metric)

        if self.use_rotation:
            logger.info("Rotating data before encoding it with a product quantizer...")
            logger.info("Creating OPQ Matrix...")
            logger.info("Input Dimension: {}, Output Dimension: {}".format(self.dim_size, self.num_of_centroids*4))
            opq_matrix = faiss.OPQMatrix(self.dim_size, self.code_size, self.num_of_centroids*4)
            base_index = faiss.IndexPQ(self.num_of_centroids*4, self.num_of_centroids, self.code_size, self.similarity_metric)
            base_index = faiss.IndexPreTransform(opq_matrix, base_index)
        
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        
        else:
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, base_index)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "pq"):
        super().save(output_dir, prefix, ext)
    
    def get_index_name(self):
        return "pq_faiss_index"


class HNSWFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: Optional[int] = None, hnsw_store_n: int = 512, 
                 hnsw_ef_search: int = 128, hnsw_ef_construction: int = 200, similarity_metric=faiss.METRIC_INNER_PRODUCT, **kwargs):
        super(HNSWFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.hnsw_store_n = hnsw_store_n
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.similarity_metric = similarity_metric
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "hnsw"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissHNSWIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissHNSWIndex(base_index, passage_ids)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()
    
    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str],
        ):
        faiss_ids, corpus_embeddings = super()._index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)

        logger.info("Using Approximate Nearest Neighbours (HNSW) in Flat Mode!")
        logger.info("Parameters Required: hnsw_store_n: {}".format(self.hnsw_store_n))
        logger.info("Parameters Required: hnsw_ef_search: {}".format(self.hnsw_ef_search))
        logger.info("Parameters Required: hnsw_ef_construction: {}".format(self.hnsw_ef_construction))
        
        base_index = faiss.IndexHNSWFlat(self.dim_size + 1, self.hnsw_store_n, self.similarity_metric)
        base_index.hnsw.efSearch = self.hnsw_ef_search
        base_index.hnsw.efConstruction = self.hnsw_ef_construction
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissHNSWIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissHNSWIndex.build(faiss_ids, corpus_embeddings, base_index)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "hnsw"):
        super().save(output_dir, prefix, ext)
    
    def get_index_name(self):
        return "hnsw_faiss_index"

class HNSWSQFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: Optional[int] = None, hnsw_store_n: int = 128, 
                 hnsw_ef_search: int = 128, hnsw_ef_construction: int = 200, similarity_metric=faiss.METRIC_INNER_PRODUCT, 
                 quantizer_type: str = "QT_8bit", **kwargs):
        super(HNSWSQFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.hnsw_store_n = hnsw_store_n
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.similarity_metric = similarity_metric
        self.qname = quantizer_type
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "hnsw-sq"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        self.faiss_index = FaissTrainIndex(base_index, passage_ids)
    
    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str],
        ):
        faiss_ids, corpus_embeddings = super()._index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)

        logger.info("Using Approximate Nearest Neighbours (HNSW) in SQ Mode!")
        logger.info("Parameters Required: hnsw_store_n: {}".format(self.hnsw_store_n))
        logger.info("Parameters Required: hnsw_ef_search: {}".format(self.hnsw_ef_search))
        logger.info("Parameters Required: hnsw_ef_construction: {}".format(self.hnsw_ef_construction))
        logger.info("Parameters Required: quantizer_type: {}".format(self.qname))
        
        qtype = getattr(faiss.ScalarQuantizer, self.qname)
        base_index = faiss.IndexHNSWSQ(self.dim_size + 1, qtype, self.hnsw_store_n)
        base_index.hnsw.efSearch = self.hnsw_ef_search
        base_index.hnsw.efConstruction = self.hnsw_ef_construction
        self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "hnsw-sq"):
        super().save(output_dir, prefix, ext)
    
    def get_index_name(self):
        return "hnswsq_faiss_index"

class FlatIPFaissSearch(DenseRetrievalFaissSearch):
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "flat"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissIndex(base_index, passage_ids)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str],
        ):
        faiss_ids, corpus_embeddings = super()._index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)
        base_index = faiss.IndexFlatIP(self.dim_size)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, base_index)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "flat"):
        super().save(output_dir, prefix, ext)
    
    def get_index_name(self):
        return "flat_faiss_index"

class PCAFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, base_index: faiss.Index, output_dimension: int, batch_size: int = 128, 
                corpus_chunk_size: Optional[int] = None, pca_matrix = None, random_rotation: bool = False, 
                eigen_power: float = 0.0, **kwargs):
        super(PCAFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.base_index = base_index
        self.output_dim = output_dimension
        self.pca_matrix = pca_matrix
        self.random_rotation = random_rotation
        self.eigen_power = eigen_power

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "pca"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissTrainIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissTrainIndex(base_index, passage_ids)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str],
        ):
        faiss_ids, corpus_embeddings = super()._index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)
        logger.info("Creating PCA Matrix...")
        logger.info("Input Dimension: {}, Output Dimension: {}".format(self.dim_size, self.output_dim))
        pca_matrix = faiss.PCAMatrix(self.dim_size, self.output_dim, self.eigen_power, self.random_rotation)
        logger.info("Random Rotation in PCA Matrix is set to: {}".format(self.random_rotation))
        logger.info("Whitening in PCA Matrix is set to: {}".format(self.eigen_power))
        if self.pca_matrix is not None:
            pca_matrix = pca_matrix.copy_from(self.pca_matrix)
        self.pca_matrix = pca_matrix
        
        # Final index
        final_index = faiss.IndexPreTransform(pca_matrix, self.base_index)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, final_index)
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, final_index)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "pca"):
        super().save(output_dir, prefix, ext)
    
    def get_index_name(self):
        return "pca_faiss_index"

class SQFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: Optional[int] = None, 
                similarity_metric=faiss.METRIC_INNER_PRODUCT, quantizer_type: str = "QT_fp16", **kwargs):
        super(SQFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.similarity_metric = similarity_metric
        self.qname = quantizer_type

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "sq"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissTrainIndex(gpu_base_index, passage_ids)
        else:
            self.faiss_index = FaissTrainIndex(base_index, passage_ids)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def index(
            self, 
            corpus_emb: Tensor,
            corpus_ids: list[str],
        ):
        faiss_ids, corpus_embeddings = super()._index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)

        logger.info("Using Scalar Quantizer in Flat Mode!")
        logger.info("Parameters Used: quantizer_type: {}".format(self.qname))

        qtype = getattr(faiss.ScalarQuantizer, self.qname)
        base_index = faiss.IndexScalarQuantizer(self.dim_size, qtype, self.similarity_metric)
        if self.use_single_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, base_index)
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissTrainIndex.build(faiss_ids, corpus_embeddings, base_index)
            if self.use_multiple_gpu:
                self.faiss_index.to_gpu()

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "sq"):
        super().save(output_dir, prefix, ext)
    
    def get_index_name(self):
        return "sq_faiss_index"

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
