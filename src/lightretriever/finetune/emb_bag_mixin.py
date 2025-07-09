#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
A simple EmbeddingBag inference mixin.

@Time    :   2025/01/02
@Author  :   Ma (Ma787639046@outlook.com)
'''
from typing import Optional
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from .nonctx_emb_utils import construct_embedding_bag

class EmbeddingBagMixin:
    def __init__(self):
        # EmbeddingBag for Inference
        self.emb_bag: Optional[nn.EmbeddingBag] = None
        self.emb_bag_prompt: Optional[str] = None
    
    def construct_embedding_bag(
        self, 
        tokenizer: PreTrainedTokenizerBase,
        prompt: Optional[str] = None, 
        batch_size: int = 2000
    ):
        """ Construct EmbeddingBag for Imbalanced Query Encoder Inference

            Args:
                prompt (Optional[str]): Optional string prompt for token embedding encoding. The inputs 
                    will be formated in `[bos] + [prompts] + [vocab_token_id] + [eos]`.
                batch_size (int): Batch size when forward above inputs through model.
        """
        if self.emb_bag is not None and self.emb_bag_prompt == prompt:
            return
        
        self.emb_bag = construct_embedding_bag(
            model=self.lm_q_base_unwrap, tokenizer=tokenizer, prompt=prompt, batch_size=batch_size
        )
        self.emb_bag.to(self.lm_q.device)
        self.emb_bag_prompt = prompt