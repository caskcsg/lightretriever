#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
DummyModel returns queries & passages as its original string format.

@Time    :   2025/04/03
@Author  :   Ma (Ma787639046@outlook.com)
'''
import datasets

class DummyModel:
    def __init__(self, *args, **kwargs):
        self.encoding_kwargs = {}   # Unused
        pass

    def parse_texts(
        self,
        texts: list[str] | list[dict[str, str]] | datasets.Dataset,
    ) -> list[str]:
        if isinstance(texts, list) and isinstance(texts[0], str):
            return texts
        elif isinstance(texts, list) and isinstance(texts[0], dict):
            processed_texts: list[str] = []
            for item in texts:
                if "title" in item:
                    _text = item["title"] + " " + item["text"]
                else:
                    _text = item["text"]
                if "prompt" in item:
                    _text = item["prompt"] + " " + _text
                processed_texts.append(_text)
            return processed_texts
        elif isinstance(texts, datasets.Dataset):
            # Merge "title" into "text" first, then fetch "text"
            if "title" in texts.column_names:
                texts = texts.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"], num_proc=max(0, min(len(texts) // 100000, 12)) or None)
            if "prompt" in texts.column_names:
                texts = texts.map(lambda x: {"text": x["prompt"] + " " + x["text"]}, remove_columns=["prompt"], num_proc=max(0, min(len(texts) // 100000, 12)) or None)
            processed_texts = list(texts["text"])
            return processed_texts
        else:
            raise NotImplementedError(f"Unrecognized texts type {type(texts)}")
        
    def encode_queries(
        self, 
        queries: list[str] | list[dict[str, str]] | datasets.Dataset, 
        **kwargs
    ) -> list[str]:
        return self.parse_texts(queries)
    
    def encode_corpus(
        self, 
        corpus: list[str] | list[dict[str, str]] | datasets.Dataset, 
        **kwargs
    ) -> list[str]:
        return self.parse_texts(corpus)

    def encode(
        self, 
        sentences: list[str] | list[dict[str, str]] | datasets.Dataset, 
        **kwargs
    ) -> list[str]:
        return self.parse_texts(sentences)
    