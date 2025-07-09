import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import datasets
from datasets import (
    load_dataset, 
    Value,
    IterableDataset, 
    interleave_datasets, 
    concatenate_datasets
)
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from .homogenous_iterable_dataset import interleave_datasets_homologenous
from .prompts import get_prompt_list
from .stopwords.util import get_lucene_stopword_list, get_unicode_punctuation_list

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if (int(os.getenv("RANK", -1)) in [0, -1]) else logging.WARN)

STOPWORD_SETS = set(get_lucene_stopword_list() + get_unicode_punctuation_list())

# Tokenizer
def load_tokenizer(
    model_name_or_path: str=None, 
    tokenizer_name: str=None, 
    cache_dir: str=None, 
    use_fast: bool=True,
    edit_tokenizer_normalizers: bool=True,
    lowercase: bool=False,
    edit_tokenizer_post_processor: bool=True,
    add_bos_num=-1, 
    add_eos_num=-1, 
    add_pooling_token_num=-1,
    add_pad_token: bool=True,
    pad_token: str='<|pad|>',
    add_sep_token: bool=False,
    sep_token: str='<|sep|>',
):
    """ Load HF GPT tokenizer, add special tokens and modify normalizers or post-processor if necassary. 

        Args:
            model_name_or_path (str): Pretrained tokenizer name or path.
            tokenizer_name (str): Same arg as `model_name_or_path`, kept for backward compatibility.
            cache_dir (str): Where do you want to store the pretrained models downloaded from huggingface.co.
            use_fast (bool): Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. This is 
                            `mandatory` for modifying normalizers or post-processor.
            edit_tokenizer_normalizers (bool): Allow editing normalizers. Enable this to make `lowercase` work.
            lowercase (bool): Whether to lowercase inputs by modifying normalizers.
            edit_tokenizer_post_processor (bool): Allow editing post-processor. Enable this to make all `TemplateProcessing` work.
            add_bos_num (int): Number of <bos> added in the front by modifying post-processor.
            add_eos_num (int): Number of <eos> added in the end by modifying post-processor.
            add_pooling_token_num (int): Number of <|pooling_token_x|> (<|pooling_token_0|>, <|pooling_token_1|>, ...) 
                                added in the end by modifying post-processor. This is likely to expand the vocabulary.
                                Thus resizing embedding layer and saving it will be needed.
            add_pad_token (bool): Whether to add <pad> to the tokenizer vocabulary.
            pad_token (str): The pad token to add. Choosing a preserved token from tokenizer's vocab will avoid enlarging
                            the size of embedding layer. Recommendations:
                            1. Qwen: <|im_end|>
                            2. Llama3: <|reserved_special_token_0|>
                            3. Mistral0.1: <unk>
                            4. Mistral0.3: [control_8]
                            5. Gemma: <|pad|>
            add_sep_token (bool): Whether to add <sep> to the tokenizer vocabulary.
            sep_token (str): The sep token to add. Choosing a preserved token from tokenizer's vocab will avoid enlarging
                            the size of embedding layer. Recommendations:
                            1. Qwen: <|im_start|>
                            2. Llama: <|reserved_special_token_1|>
                            3. Mistral0.1: <s>
                            4. Mistral0.3: [/INST]
                            5. Gemma: <bos>
    """
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, use_fast=use_fast)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, use_fast=use_fast)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Compatiable with GPT Tokenizers
    tokenizer.padding_side = "right"
    
    # Special tokens
    if add_bos_num > 0 and tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<|bos|>'})  # Optional bos
    if add_eos_num > 0 and tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|eos|>'})  # Optional eos
    if add_pad_token and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_token})  # Optional pad
    if add_sep_token and tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': sep_token})  # Optional sep
    
    if add_pooling_token_num > 0:   # Optional pooling token
        tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '<|pooling_token_0|>', 
                '<|pooling_token_1|>', 
                '<|pooling_token_2|>'
            ],
        }, replace_additional_special_tokens=False)

    if edit_tokenizer_normalizers:
        tokenizer = _edit_gpt_normalizers(tokenizer, lowercase)

    if edit_tokenizer_post_processor:
        tokenizer = _edit_gpt_post_processor(
            tokenizer, add_bos_num=add_bos_num, add_eos_num=add_eos_num, add_pooling_token_num=add_pooling_token_num)
    
    return tokenizer


def _edit_gpt_normalizers(
    tokenizer: PreTrainedTokenizerFast, 
    lowercase: bool=False,
):
    """ Edit Normalizers to support lowercase the inputs for embedding training. """
    if not lowercase:   # No need to modify the pretokenizer
        return tokenizer
    
    assert tokenizer.is_fast == True, "Only support editing fast tokenizer."

    lowercase_expr = {"type": "Lowercase"}

    # Replace Normalizers
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if ("normalizer" in tokenizer_json) and isinstance(tokenizer_json["normalizer"], dict) and ("type" in tokenizer_json["normalizer"]):
        if tokenizer_json["normalizer"]["type"] != "Lowercase":
            if tokenizer_json["normalizer"]["type"] == "Sequence":
                # Judge whether Lowercase exists
                lowercase_exists = False
                for _normalizer in tokenizer_json["normalizer"]["normalizers"]:
                    if _normalizer["type"] == "Lowercase":
                        lowercase_exists = True
                        break
                if not lowercase_exists:
                    tokenizer_json["normalizer"]["normalizers"].insert(0, lowercase_expr)
            else:
                ori_expr = tokenizer_json["normalizer"]
                tokenizer_json["normalizer"] = {
                    "type": "Sequence", "normalizers": [lowercase_expr, ori_expr]
                }
    else:
        tokenizer_json["normalizer"] = lowercase_expr
    
    tokenizer._tokenizer = tokenizer._tokenizer.from_str(json.dumps(tokenizer_json))
    return tokenizer


def _edit_gpt_post_processor(
    tokenizer: PreTrainedTokenizerFast, 
    add_bos_num=-1, add_eos_num=-1, add_pooling_token_num=-1,
):
    """ Edit the post processor of a GPT Fast Tokenizer to support adding special tokens for embedding training. """
    if (add_bos_num < 0) and (add_eos_num < 0) and (add_pooling_token_num < 0):
        return tokenizer    # Don't need to edit

    assert tokenizer.is_fast == True, "Only support editing fast tokenizer."

    if add_bos_num > 0:
        bos_token, bos_token_id = tokenizer.bos_token, tokenizer.bos_token_id
        assert bos_token is not None
        assert bos_token_id is not None
    
    eos_token, eos_token_id = tokenizer.eos_token, tokenizer.eos_token_id
    if add_eos_num > 0:
        assert eos_token is not None
        assert eos_token_id is not None
    
    if add_pooling_token_num > 0:
        pooling_tokens = ["<|pooling_token_0|>", "<|pooling_token_1|>", "<|pooling_token_2|>"]
        pooling_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in pooling_tokens]
        for token_id in pooling_token_ids:
            assert token_id != tokenizer.unk_token_id
        assert add_pooling_token_num <= 3, "More add_pooling_token_num is not supported yet."
    
    single = []
    pair = []
    special_tokens = {}

    # bos
    if add_bos_num > 0:
        bos_expr = {'SpecialToken': {'id': bos_token, 'type_id': 0}}
        for _ in range(add_bos_num):
            single.append(bos_expr)
            pair.append(bos_expr)
        special_tokens[bos_token] = {
            'id': bos_token, 
            'ids': [bos_token_id], 
            'tokens': [bos_token]
        }
    
    # text
    a_expr = {'Sequence': {'id': 'A', 'type_id': 0}}
    single.append(a_expr)
    pair.append(a_expr)

    eos_expr = {'SpecialToken': {'id': eos_token, 'type_id': 0}}
    b_expr = {'Sequence': {'id': 'B', 'type_id': 0}}
    ## By default (GPT-alike models), we do not add eos as a separator of A & B, 
    ## please define separation (whitespace, '\n', etc.) on your own
    # pair.append(eos_expr)
    pair.append(b_expr)
    special_tokens[eos_token] = {
        'id': eos_token, 
        'ids': [eos_token_id], 
        'tokens': [eos_token]
    }

    # eos
    if add_eos_num > 0:
        for _ in range(add_eos_num):
            single.append(eos_expr)
            pair.append(eos_expr)
    
    # pooling tokens
    if add_pooling_token_num > 0:
        for i in range(add_pooling_token_num):
            pooling_token_expr = {'SpecialToken': {'id': pooling_tokens[i], 'type_id': 0}}
            single.append(pooling_token_expr)
            pair.append(pooling_token_expr)
            special_tokens[pooling_tokens[i]] = {
                'id': pooling_tokens[i], 
                'ids': [pooling_token_ids[i]], 
                'tokens': [pooling_tokens[i]]
            }

    # Replace post_processor
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if ("post_processor" in tokenizer_json) and ("type" in tokenizer_json["post_processor"]):
        # Contains at least one post_processor
        if tokenizer_json["post_processor"]["type"] == "TemplateProcessing":
            tokenizer_json["post_processor"]["single"] = single
            tokenizer_json["post_processor"]["pair"] = pair
            tokenizer_json["post_processor"]["special_tokens"] = special_tokens
        elif tokenizer_json["post_processor"]["type"] == "Sequence":
            # Locate TemplateProcessing
            for i in range(len(tokenizer_json['post_processor']['processors'])):
                if tokenizer_json['post_processor']['processors'][i]["type"] == "TemplateProcessing":
                    break
            # Replace TemplateProcessing
            if i < len(tokenizer_json['post_processor']['processors']):
                tokenizer_json['post_processor']['processors'][i]["single"] = single
                tokenizer_json['post_processor']['processors'][i]["pair"] = pair
                tokenizer_json['post_processor']['processors'][i]["special_tokens"] = special_tokens
            else:
                tokenizer_json['post_processor']['processors'].append({
                    "type": "TemplateProcessing", "single": single, "pair": pair, "special_tokens": special_tokens,
                })
        else:
            # Wrap the original post_processor with Sequence
            ori_post_processor = tokenizer_json["post_processor"]
            tokenizer_json["post_processor"] = {
                "type": "Sequence", "processors": [
                    ori_post_processor, 
                    {"type": "TemplateProcessing", "single": single, "pair": pair, "special_tokens": special_tokens}
                ]
            }
    else:
        # Directly make TemplateProcessing as post_processor
        tokenizer_json["post_processor"] = {"type": "TemplateProcessing", "single": single, "pair": pair, "special_tokens": special_tokens}

    tokenizer._tokenizer = tokenizer._tokenizer.from_str(json.dumps(tokenizer_json))
    return tokenizer

def resize_emb(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int | None = None):
    """ 
    Some GPT models need to add a [PAD] token. If the tokenizer vocab 
    is expanded, we need to resize embedding size.
    """
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

def build_corpus_idx_to_row(dataset: datasets.Dataset):
    """ Build a dict on memory of corpus id -> row id of hfdataset """
    idx_to_corpus = dict()
    for row_id, corpus_id in enumerate(dataset["_id"]):
        idx_to_corpus[corpus_id] = row_id
    return idx_to_corpus

def read_corpus(corpus_name_or_path: str):
    """ Load HFDataset from local or online """
    corpus_path = None
    # Local file or folders
    if os.path.exists(corpus_name_or_path):
        if os.path.isdir(corpus_name_or_path):
            files = os.listdir(corpus_name_or_path)
            corpus_path = [
                os.path.join(corpus_name_or_path, f)
                for f in files
                if f.endswith('json') or f.endswith('jsonl')
            ]
        else:
            corpus_path = [corpus_name_or_path]
        
        if corpus_path:
            # Load as Json files
            dataset_name = 'json'
            dataset_split = 'train'
            dataset_language = 'default'
        else:
            # Try to load from local HF dataset
            dataset_name = corpus_name_or_path
            dataset_split = 'train'
            dataset_language = None
            corpus_path = None
    # Online huggingface dataset
    else:
        info = corpus_name_or_path.split('/')
        dataset_split = info[-1] if len(info) == 3 else 'train'
        dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
        dataset_language = 'default'
        if ':' in dataset_name:
            dataset_name, dataset_language = dataset_name.split(':')
    
    dataset = load_dataset(
        dataset_name,
        dataset_language,
        data_files=corpus_path,
        split=dataset_split
    )

    # Parse tevatron format Jsonl text column names to sentence transformers format
    for _original_column_name, _new_column_name in [("query_id", "_id"), ("docid", "_id"), ("id", "_id"), ("query", "text"), ("question", "text")]:
        if _original_column_name in dataset.column_names:
            dataset = dataset.rename_column(_original_column_name, _new_column_name)
    
    # Format "_id" to str
    if "_id" in dataset.column_names and dataset.features["_id"].dtype != 'string':
        dataset = dataset.cast_column('_id', Value("string"))
    
    return dataset

def process_tsv_file(tsv_ranks_path: str, depth: int=1000):
    q_p_dict = {}
    ret = []    # (qid, pid) pairs
    print(f"Reading idx from {tsv_ranks_path}")
    for _, line in enumerate(tqdm(open(tsv_ranks_path, 'r'), total=wc_count(tsv_ranks_path))):
        line: list = line.split('\t')
        if len(line) == 4:
            line.pop(1)
        qid: str = line[0].strip()
        pid: str = line[1].strip()
        score: str = float(line[2])     # This score is generated by dual-encoder
        if qid not in q_p_dict:
            q_p_dict[qid] = [(pid, score)]
        else:
            q_p_dict[qid].append((pid, score))
    for k, v in q_p_dict.items():
        q_p_dict[k] = sorted(v, key=lambda x: x[1], reverse=True)
        q_p_dict[k] = q_p_dict[k][:depth]
        ret.extend([(k, pid) for pid, score in q_p_dict[k]]) 
    return ret

def load_domain_datasets(
        domain_names: List[str],
        preprocessed_dir: Union[str, Path],
        prompt_type: str,
        add_domain_id: bool = False,
        add_domain_name: bool = False,
        seed: int = 42,
        domain_to_idx: Optional[Dict[str, int]] = None,
        category_list: Dict[str, List[str]] = None,
    ):
    """
    Load multiple HF dataset by `domain_names` from root dir `preprocessed_dir`
     - `domain_names`: List of each domain names.
     - `preprocessed_dir`: Root folder path of all processed domains.
     - `add_domain_id`: Add domain index.
     - `add_domain_name`: Add domain name.
     - `domain_to_idx`: A dict that map the domain names to coresponding index
    """
    domain_ds: Dict[str, datasets.Dataset] = dict()
    preprocessed_dir = Path(preprocessed_dir)
    for domain_idx, domain_name in enumerate(sorted(domain_names)):
        if (preprocessed_dir / f"{domain_name}.jsonl").exists():
            load_format = "json"
            domain_filepaths = [str(preprocessed_dir / f"{domain_name}.jsonl")]
            logger.info(f"Loading [{domain_idx+1}/{len(domain_names)}] domain {domain_name} from {domain_filepaths}")
        
        elif (preprocessed_dir / f"{domain_name}.json").exists():
            load_format = "json"
            domain_filepaths = [str(preprocessed_dir / f"{domain_name}.json")]
            logger.info(f"Loading [{domain_idx+1}/{len(domain_names)}] domain {domain_name} from {domain_filepaths}")
        
        elif (preprocessed_dir / f"{domain_name}.parquet").exists():
            load_format = "parquet"
            domain_filepaths = [str(preprocessed_dir / f"{domain_name}.parquet")]
            logger.info(f"Loading [{domain_idx+1}/{len(domain_names)}] domain {domain_name} from {domain_filepaths}")
        
        elif (preprocessed_dir / domain_name).exists() and (preprocessed_dir / domain_name).is_dir():
            load_format = str(preprocessed_dir / domain_name)
            domain_filepaths = None
            logger.info(f"Loading [{domain_idx+1}/{len(domain_names)}] domain {domain_name} from {load_format}")
        
        else:
            raise FileNotFoundError(f"{domain_name} does not exists or is not a file. Please check data config.")

        dataset = load_dataset(load_format, data_files=domain_filepaths, split="train")

        if "instruction" not in dataset.column_names:
            # Add pre-defined prompt column (In `prompts.py`)
            instruction_list = get_prompt_list(prompt_type, domain_name, num=len(dataset), seed=seed)
            dataset = dataset.add_column(name="instruction", column=instruction_list)
        else:
            if prompt_type == "e5":     # E5 prompt needs post process the format
                dataset = dataset.map(
                    lambda item: {"instruction": "Instruct: {}\nQuery: ".format(item["instruction"])}
                )

        # Add `_train_dataset_idx` to support reproducable negative passage sampling
        dataset = dataset.add_column(name="_train_dataset_idx", column=np.arange(len(dataset)))

        if add_domain_name:
            dataset = dataset.add_column(name="domain_name", column=[domain_name for _ in range(len(dataset))])
        
        domain_ds[domain_name] = dataset
    
    if category_list is not None:
        logger.info(f"Grouping datasets by category..")
        
        category_ds: Dict[str, datasets.Dataset] = dict()
        for category_name, domain_list in category_list.items():
            category_ds[category_name] = concatenate_datasets([domain_ds[i] for i in domain_list])
        
        if add_domain_id:
            for domain_name in category_ds.keys():
                category_ds[domain_name] = category_ds[domain_name].add_column(
                    name="domain_ids", 
                    column=np.ones(len(category_ds[domain_name]), dtype=np.int8) * domain_to_idx[domain_name]
                )
        return category_ds
    else:
        if add_domain_id:
            for domain_name in domain_ds.keys():
                domain_ds[domain_name] = domain_ds[domain_name].add_column(
                    name="domain_ids", 
                    column=np.ones(len(domain_ds[domain_name]), dtype=np.int8) * domain_to_idx[domain_name]
                )
        return domain_ds

def _average_weights(weights):
    if isinstance(weights, list):
        _sum = sum(weights)
        return [i/_sum for i in weights]
    elif isinstance(weights, dict):
        _sum = sum(weights.values())
        return {k: v/_sum for k, v in weights.items()}
    elif isinstance(weights, np.ndarray):
        _sum = sum(weights)
        return weights / _sum
    else:
        raise NotImplementedError()

def construct_domain_dataset(
        domain_config_path: str,
        preprocessed_dir: Union[str, Path],
        prompt_type: str,
        add_domain_id: bool = False,
        add_domain_name: bool = False,
        seed: int = 42,
        stopping_strategy: str = 'all_exhausted',
        iterable_n_shards: int = 64,
        shuffle: bool = False,
        # Homogenous batch sampling
        homogenous_batch: bool = False,
        global_batch_size: Optional[int] = None,
    ):
    """
    Construct interleavable datasets.
     - `domain_config_path`: Path to json format domain config. 
                             1) domain_ids: A dict that map domain name to index.
                             2) domain_weights: Sampling propobality of each train domain.
     - `preprocessed_dir`: Root folder path of all processed domains.
     - `prompt_type`: Prompt type. Different pre-defined prompts are listed in `prompts.py`.
     - `add_domain_id`: Add domain index.
     - `seed`: Random seed for sampling.
     - `stopping_strategy`: Set to 'first_exhausted' for less sampling or 'all_exhausted' for oversampling.
                            See `datasets.interleave_datasets`
     - `iterable_n_shards`: Defines the shards for each domain datasets when converted to iterable dataset.
     - `shuffle`: Shuffle each domain dataset with `seed`.
    """
    # Load domain weights from local file
    with open(domain_config_path, 'r') as f:
        domain_config: dict = json.load(f)
        domain_to_idx: dict = domain_config['domain_ids']
        train_domain_weights_dict: dict = _average_weights(domain_config['domain_weights'])
        category_list: Dict[str, List[str]] = domain_config.get("category_list", None)

    # whenever we convert dict to array, we sort by key
    domain_list = list(sorted(train_domain_weights_dict.keys()))
    num_domains = len(domain_list)

    # Loading datasets of each domains
    domain_names = domain_list if category_list is None else sum(category_list.values(), [])
    domain_ds = load_domain_datasets(
                    domain_names=domain_names,
                    preprocessed_dir=preprocessed_dir,
                    prompt_type=prompt_type,
                    add_domain_id=add_domain_id,
                    add_domain_name=add_domain_name,
                    seed=seed,
                    domain_to_idx=domain_to_idx,
                    category_list=category_list,
                )
    
    logger.info(f"{len(domain_ds)} domains loaded. Lengths of each domain:")
    logger.info(f"Domain\t Lengths\t Sampling ratio")
    for _domain_name, _domain_dataset in domain_ds.items():
        logger.info(f"{_domain_name}:\t {len(_domain_dataset)}\t {train_domain_weights_dict[_domain_name]}")

    # if homogenous_batch:
    #     # Multiprocess dataloading with Iterable dataset is currently not supported
    #     iterable_n_shards = 1
    
    # Convert to IterableDataset and shuffle
    for _domain_name, _domain_dataset in domain_ds.items():
        logger.info(f"Convert {_domain_name} to IterableDataset with {iterable_n_shards} shards.")
        domain_ds[_domain_name] = _domain_dataset.to_iterable_dataset(iterable_n_shards)
        if shuffle:
            domain_ds[_domain_name] = domain_ds[_domain_name].shuffle(seed=seed, buffer_size=1000)
    
    if homogenous_batch:
        logger.info(f"Using homogenous batch sampling with global batch size {global_batch_size}")
        full_dataset: IterableDataset = interleave_datasets_homologenous(
                    datasets=[domain_ds[_domain] for _domain in domain_list],
                    batch_size=global_batch_size,
                    probabilities=[train_domain_weights_dict[_domain] for _domain in domain_list],
                    seed=seed,
                    stopping_strategy=stopping_strategy,
                )
    else:
        full_dataset: IterableDataset = interleave_datasets(
                        datasets=[domain_ds[_domain] for _domain in domain_list],
                        probabilities=[train_domain_weights_dict[_domain] for _domain in domain_list],
                        seed=seed,
                        stopping_strategy=stopping_strategy,
                    )
    logger.info(f"Finish construct `interleave_datasets`.")
    
    return full_dataset, domain_config


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    tokenizer = load_tokenizer("models/Qwen1.5-0.5B")

    text = "这是一句中文！。This is an English sentence. 여기에 한국어 문장이 있습니다.!" * 100
    text2 = "This is a test sentence. 这是一个测试句子。" * 100
    text3 = "Hello World!!. 这是一个测试句子。" * 100
    text4 = "你好👋。你好👋你好👋你好👋你好👋你好👋" * 100
    texts = [text, text2, text3, text4] * 1_000

    from sparse_emb_util import ICUWordPreTokenizer
    icu_tokenizer = ICUWordPreTokenizer()
    start_time = time.time()
    for i in tqdm(texts):
        token_list = icu_tokenizer(i)
    print(f"ICU Word tokenize text use: {time.time()-start_time}")

    token_list = icu_tokenizer(text)
    token_ids = tokenizer(token_list, is_split_into_words=True, add_special_tokens=False)['input_ids']

    token_recovered = tokenizer.decode(token_ids)

    print()
