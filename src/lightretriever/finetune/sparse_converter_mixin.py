#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Sparse Converter Mixin Implementation.

Sparse Converter is a highly efficient rust-based mapper, which converts
sparse pytorch/numpy matrix to json/string representations.

@Time    :   2025/01/02
@Author  :   Ma (Ma787639046@outlook.com)
'''
from typing import Optional

import torch
from torch import Tensor

from sparse_emb_util import Converter

class SparseConverterMixin:
    def __init__(self, vocab_dict: Optional[dict[str, str]]=None):
        # Rust-based Converter, parsing sparse numpy float32/float16 vector to dict/string format fastly
        self.vocab_dict = vocab_dict
        self.spr_converter = Converter(vocab_dict)
    
    def convert_sparse_reps_to_json(
        self,
        reps: Tensor, 
        quantization_factor: int=100,
        convert_id_to_token: bool=False,
    ):
        """
        Convert sparse representations to quantized json array.
        Format {token: integer value}

        Inputs:
            reps (Tensor): Sparse representations, shape [batch_size, vocab_dim].
            quantization_factor (int): Sparse retrieval engine (like Lucene) only uses integer frequency. 
                                       The reps will be `*quantization_factor` then cut to integer format, 
                                       forming pseudo-documents with number of integer frequency's tokens.
            convert_id_to_token (bool): Whether to convert `integer token id` to `string token` based on vocab.
                                        Note that query syntax error may occur when using **invalid tokens** as 
                                        search query of Lucene or Tantivy
        """
        # f16 resolution is 0.001, enough for sparse retrieval
        if reps.dtype not in [torch.float16, torch.float32]:
            reps = reps.to(dtype=torch.float16)
        
        # Only support operating array in ram
        if reps.device != torch.device('cpu'):
            reps = reps.cpu()
        
        if reps.dtype == torch.float16:
            return self.spr_converter.convert_sparse_reps_to_json_f16(
                reps.numpy(), quantization_factor, convert_id_to_token
            )
        elif reps.dtype == torch.float32:
            return self.spr_converter.convert_sparse_reps_to_json_f32(
                reps.numpy(), quantization_factor, convert_id_to_token
            )
        else:
            raise TypeError(f"Unsupported reps.dtype: {reps.dtype}.")
    
    def convert_sparse_reps_to_pseudo_text(
        self,
        reps: Tensor, 
        quantization_factor: int=100,
        convert_id_to_token: bool=False,
    ):
        """
        Convert sparse representations to quantized pseudo text format 
        by repeating each token for integer frequency times.

        Analysis should using ** whitespace tokenization **, please be sure to avoid cutting the tokens.

        Inputs:
            reps (Tensor): Sparse representations, shape [batch_size, vocab_dim].
            quantization_factor (int): Sparse retrieval engine (like Lucene) only uses integer frequency. 
                                       The reps will be `*quantization_factor` then cut to integer format, 
                                       forming pseudo-documents with number of integer frequency's tokens.
            convert_id_to_token (bool): Whether to convert `integer token id` to `string token` based on vocab.
                                        Note that query syntax error may occur when using **invalid tokens** as 
                                        search query of Lucene or Tantivy
        """
        # f16 resolution is 0.001, enough for sparse retrieval
        if reps.dtype not in [torch.float16, torch.float32]:
            reps = reps.to(dtype=torch.float16)
        
        # Only support operating array in ram
        if reps.device != torch.device('cpu'):
            reps = reps.cpu()
        
        if reps.dtype == torch.float16:
            return self.spr_converter.convert_sparse_reps_to_pseudo_text_f16(
                reps.numpy(), quantization_factor, convert_id_to_token
            )
        elif reps.dtype == torch.float32:
            return self.spr_converter.convert_sparse_reps_to_pseudo_text_f32(
                reps.numpy(), quantization_factor, convert_id_to_token
            )
        else:
            raise TypeError(f"Unsupported reps.dtype: {reps.dtype}.")

    def convert_sparse_reps_to_json_pt(
        self,
        reps: Tensor, 
        quantization_factor: int=100,
        convert_id_to_token: bool=False,
    ):
        """
        Convert sparse representations to quantized json array.
        Format {token: integer value}

        Inputs:
            reps (Tensor): Sparse representations, shape [batch_size, vocab_dim].
            quantization_factor (int): Sparse retrieval engine (like Lucene) only uses integer frequency. 
                                       The reps will be `*quantization_factor` then cut to integer format, 
                                       forming pseudo-documents with number of integer frequency's tokens.
            convert_id_to_token (bool): Whether to convert `integer token id` to `string token` based on vocab.
                                        Note that query syntax error may occur when using **invalid tokens** as 
                                        search query of Lucene or Tantivy
        """
        # Check inputs
        assert isinstance(reps, Tensor)
        if reps.ndim == 1:
            reps = reps.unsqueeze(0)
        assert reps.ndim == 2, f"The shape of reps is {reps.shape}. Please further check input."

        # Frequency larger than 0
        reps = torch.clamp(reps, min=0.0)

        # Quantization
        reps = torch.round(reps * quantization_factor).to(torch.int)    # bs, vocab size

        # Get non-zero position & frequencies
        nonzero_batch_ids, nonzero_vocab_ids = torch.nonzero(reps, as_tuple=True)
        value_freqs = reps[reps != 0]

        # Converting to dict format
        batch_id_to_sparse_reps: dict[int, dict[str, int]] = {i: dict() for i in range(reps.shape[0])}
        for batch_id, vocab_id, value_freq in zip(
            nonzero_batch_ids.tolist(), 
            nonzero_vocab_ids.tolist(), 
            value_freqs.tolist()
        ):
            if convert_id_to_token:
                batch_id_to_sparse_reps[batch_id][self.vocab_dict[vocab_id]] = int(value_freq)
            else:
                batch_id_to_sparse_reps[batch_id][str(vocab_id)] = int(value_freq)
        
        # Check empty dict
        for i in range(len(batch_id_to_sparse_reps)):
            if len(batch_id_to_sparse_reps[i]) == 0:
                if convert_id_to_token:
                    batch_id_to_sparse_reps[i]["[PAD]"] = 1
                else:
                    batch_id_to_sparse_reps[i]["-1"] = 1
        
        sparse_reps: list[dict[str, int]] = [batch_id_to_sparse_reps[i] for i in range(len(batch_id_to_sparse_reps))]
        
        return sparse_reps

    def convert_sparse_reps_to_pseudo_text_pt(
        self,
        reps: Tensor, 
        quantization_factor: int=100,
        convert_id_to_token: bool=False,
    ):
        """
        Convert sparse representations to quantized pseudo text format 
        by repeating each token for integer frequency times.

        Analysis should using ** whitespace tokenization **, please be sure to avoid cutting the tokens.

        Inputs:
            reps (Tensor): Sparse representations, shape [batch_size, vocab_dim].
            quantization_factor (int): Sparse retrieval engine (like Lucene) only uses integer frequency. 
                                       The reps will be `*quantization_factor` then cut to integer format, 
                                       forming pseudo-documents with number of integer frequency's tokens.
            convert_id_to_token (bool): Whether to convert `integer token id` to `string token` based on vocab.
                                        Note that query syntax error may occur when using **invalid tokens** as 
                                        search query of Lucene or Tantivy
        """        
        # Quantization
        json_reps = self.convert_sparse_reps_to_json_pt(reps=reps, quantization_factor=quantization_factor, convert_id_to_token=convert_id_to_token)

        # Text Format
        sparse_reps_text: list[str] = [" ".join(sum([[str(token)] * freq for token, freq in dict_rep.items()], [])) for dict_rep in json_reps]
        return sparse_reps_text