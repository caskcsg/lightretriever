#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
AnseriniSearch for sparse benchmark evaluation, api based on BaseSearch of SentenceTransformers

Note: 
1) Get Anserini jar: `wget https://repo1.maven.org/maven2/io/anserini/anserini/0.25.0/anserini-0.25.0-fatjar.jar` (Java8)

@Time    :   2024/09/03 14:27:41
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import logging
import orjson
import shutil
import tempfile
from tqdm import tqdm
from pathlib import Path
from typing import Optional

import datasets

# Set anserini jar path & java home before import jni
os.environ['CLASSPATH'] = "PATH_TO/anserini-0.25.0-fatjar.jar"

from jnius import autoclass

logger = logging.getLogger(__name__)

class AnseriniSearch:
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: Optional[int] = None, **kwargs):
        self.model = model  # Model is class that provides encode_corpus() and encode_queries()
        self.batch_size = batch_size
        self.corpus_chunk_size = batch_size * 800 if corpus_chunk_size is None else corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)

        # ** Note: Default values below are Impact Search with Learned Sparse Retrieval Models **

        ## Lang (Optional[String])
        self.anserini_lang = kwargs.get("anserini_lang", None)

        ## JsonVectorCollection (Impact Search only): corpus_emb is a list of dict {tok_id: freq}
        ## JsonCollection (Both): corpus_emb is a list of str "tok_id1 tok_id1 tok_id2 ... "
        self.anserini_vector_type = kwargs.get("anserini_vector_type", "JsonVectorCollection")

        ## Pretokenized (Impact Search only)
        self.anserini_pretokenized = kwargs.get("anserini_pretokenized", True)

        ## Impact Search (Pure dot product). If set to False, will use BM25
        self.anserini_impact_search = kwargs.get("anserini_impact_search", True)

        ## BM25 parameters (Default: k1=0.9, b=0.4; Other options: k1=1.2, b=0.75)
        self.anserini_bm25_k1 = kwargs.get("anserini_bm25_k1", 0.9)
        self.anserini_bm25_b = kwargs.get("anserini_bm25_b", 0.4)

        # Set proper temp path on your own
        # Temps will be deleted by calling `.clear()`
        anserini_temps_folder = Path("results/anserini_temps")
        anserini_temps_folder.mkdir(parents=True, exist_ok=True)
        self.temp_folder = Path(tempfile.mkdtemp(dir=str(anserini_temps_folder.absolute())))

        # Corpus dumping param
        self.index_chunk_size = 40000
        self.n_dumped = 0   # How many corpus have been dumped
        self.is_submit_to_lucene = False    # Whether the dumped vectors are already submitted to lucene
    
    def _clear(self):
        self.n_dumped = 0
        self.is_submit_to_lucene = False
        if self.temp_folder.exists():
            shutil.rmtree(self.temp_folder, ignore_errors=True)

    @classmethod
    def name(self):
        return "anserini_search"
    
    def encode(self, sentences: list[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs):
        return self.model.encode(sentences=sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def encode_queries(self, queries: list[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> list[str]:
        return self.model.encode_queries(queries=queries, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> list[str]:
        return self.model.encode_corpus(corpus=corpus, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def index(self, corpus_emb: list[dict[str, int] | str], corpus_ids: list[str]):
        """ Dump corpus embedding to local folder """
        # Dump corpus
        encoded_corpus_folder = self.temp_folder / "encoded_corpus"
        encoded_corpus_folder.mkdir(parents=True, exist_ok=True)

        f = open(encoded_corpus_folder / f"corpus{self.n_dumped // self.index_chunk_size :04d}.jsonl", "ab")
        for id_, emb_ in tqdm(zip(corpus_ids, corpus_emb), desc=f"Dumping corpus to {encoded_corpus_folder.absolute()}"):
            if self.anserini_vector_type == "JsonVectorCollection":
                line = dict(id=id_, content="", vector=emb_)
            elif self.anserini_vector_type == "JsonCollection":
                line = dict(id=id_, contents=emb_)
            else:
                raise TypeError(f"Unsupported type {self.anserini_vector_type}")
            
            f.write(orjson.dumps(line, option=orjson.OPT_APPEND_NEWLINE))

            self.n_dumped += 1
            if self.n_dumped % self.index_chunk_size == 0:
                f.close()
                f = open(encoded_corpus_folder / f"corpus{self.n_dumped // self.index_chunk_size :04d}.jsonl", "ab")

        f.close()

    def submit_to_lucene(self):
        encoded_corpus_folder = self.temp_folder / "encoded_corpus"
        # Index
        ## Index Args
        ## See: anserini-0.25.0/src/main/java/io/anserini/index/AbstractIndexer.java
        ##      anserini-0.25.0/src/main/java/io/anserini/index/IndexCollection.java
        self.index_folder = self.temp_folder / "index"
        self.index_folder.mkdir(parents=True, exist_ok=True)
        index_args = [
            '-collection', self.anserini_vector_type, 
            '-input', str(encoded_corpus_folder.absolute()), 
            '-index', str(self.index_folder.absolute()),
            '-generator', 'DefaultLuceneDocumentGenerator', 
            '-threads', '64'
        ]

        if self.anserini_lang:
            index_args.extend(['-language', self.anserini_lang])

        if self.anserini_pretokenized:
            index_args.append('-pretokenized')

        if self.anserini_impact_search:
            index_args.append('-impact')
        
        JIndexCollection = autoclass('io.anserini.index.IndexCollection')
        JIndexCollection.main(index_args)
        self.is_submit_to_lucene = True
        logger.warning(f"Index Completed. Size: {human_readable_size(get_path_size(self.index_folder))}")

    def retrieve_with_emb(self, 
                          query_emb: list[str],
                          query_ids: list[str],
                          top_k: int
                          ):
        """
        Retrieve with pseudo-queries. Please first index all pseudo-documents,
        then retrieve using this fuction.

        Inputs:
            query_emb (list[str]): list of `pseudo-query`
            query_ids (list[str]): list of `query-ids`
            top_k (int): Threshold

        Returns:
            dict of `qid -> pid -> score`
        """
        if not self.is_submit_to_lucene:
            self.submit_to_lucene()
        
        results: dict[str, dict[str, float]] = dict()   # qid -> pid -> score

        # Dump query tsv
        encoded_query_path = self.temp_folder / "query.tsv"
        with open(encoded_query_path, "w") as f:
            for id_, q_str_ in tqdm(zip(query_ids, query_emb), desc=f"Dumping queries to {encoded_query_path.absolute()}"):
                line = id_ + '\t' + q_str_ + '\n'
                f.write(line)
        
        # Retrieve
        ## Search Args
        ## See: anserini-0.25.0/src/main/java/io/anserini/search/BaseSearchArgs.java
        ##      anserini-0.25.0/src/main/java/io/anserini/search/SearchCollection.java
        result_path = self.temp_folder / "result.trec"
        search_args = [
            '-hits', str(top_k),
            '-parallelism', '64',
            '-threads', '64',
            '-index', str(self.index_folder.absolute()),
            '-topicReader', 'TsvString', 
            '-topics', str(encoded_query_path.absolute()),
            '-output', str(result_path.absolute()),
            '-format', 'trec',
        ]

        if self.anserini_lang:
            search_args.extend(['-language', self.anserini_lang])

        if self.anserini_pretokenized:
            search_args.append('-pretokenized')

        if self.anserini_impact_search:
            search_args.append('-impact')
        else:
            search_args.extend([
                '-bm25',
                '-bm25.k1', str(self.anserini_bm25_k1),
                '-bm25.b', str(self.anserini_bm25_b),
            ])
        
        JSearchCollection = autoclass('io.anserini.search.SearchCollection')
        JSearchCollection.main(search_args)

        # Parse scores
        with open(result_path, "r") as f:
            for line in f:
                item = line.split(" ")
                qid, pid, score = item[0], item[2], float(item[4])
                if qid not in results:
                    results[qid] = {}
                results[qid][pid] = score
        
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
        
        query_embeddings: dict[str, any] = self.model.encode_queries(
            queries_list, 
            batch_size=self.batch_size, 
            show_progress_bar=self.show_progress_bar, 
            convert_to_tensor=self.convert_to_tensor
        )

        logger.info("Sorting Corpus by document length (Longest first)...")
        if isinstance(corpus, dict):
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("text", "")), reverse=True)
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

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Step1: Encode chunk of corpus
            if isinstance(corpus, dict):
                corpus_sliced = corpus[corpus_start_idx:corpus_end_idx]
            else:
                corpus_sliced = corpus.select(range(corpus_start_idx, corpus_end_idx))
            
            sub_corpus_embeddings: list[dict[str, any]] = self.model.encode_corpus(
                corpus_sliced,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor=self.convert_to_tensor
            )
        
            # Step2: Indexing
            logger.info("Anserini Indexing...")
            self.index(sub_corpus_embeddings, corpus_ids[corpus_start_idx:corpus_end_idx])
        
        # Step3: Retrieve
        logger.info("Retrieving...")
        results = self.retrieve_with_emb(query_embeddings, query_ids, top_k=top_k)

        # Remember to clear all index
        self._clear()
        return results

def get_path_size(path: Path) -> int:
    """ Get disk usage (Bytes) of a path """
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    elif path.is_symlink():
        try:
            target = path.resolve(strict=True)
            return target.stat().st_size if target.is_file() else 0
        except FileNotFoundError:
            return 0
    else:
        return 0

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
