import os
import json
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import logging
logger = logging.getLogger(__name__)

class SparseProjectorBase(nn.Module):
    """ Base Class for all Sparse Rep Projectors """
    CONFIG_NAME = 'spr_gate_mlp_config.json'
    WEIGHT_NAME = 'spr_gate_mlp.pt'

    def __init__(self, hidden_dim: int, vocab_size: int, initializer_range: float=0.02):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._config = {'input_dim': hidden_dim, 'output_dim': vocab_size, 'initializer_range': initializer_range}

        self._init_weights(self.linear)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self._config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
    
    @property
    def name(self):
        return "SparseProjectorBase"
    
    @property
    def device(self):
        return self.linear.weight.device

    def forward(self, reps: Tensor) -> Tensor:
        raise NotImplementedError("Base Class. Please implement this function in subclass.")
    
    def gate(self, logits: Tensor, margin: float) -> list[list[int]]:
        """ Input logits obtained from self.forward, then produce a list of unique expansion ids. """
        raise NotImplementedError("Base Class. Please implement this function in subclass.")
    
    def gate_topk(self, logits: Tensor, topk: list[int]) -> list[list[int]]:
        """ Input logits obtained from self.forward, then produce a list of unique expansion ids. """
        raise NotImplementedError("Base Class. Please implement this function in subclass.")
    
    @classmethod
    def build(cls, hidden_dim: int, vocab_size: int, init_weight: Optional[Tensor]=None):
        projector = cls(hidden_dim, vocab_size)
        if init_weight is not None:
            projector.linear.weight.data.copy_(init_weight)
        return projector

    @classmethod
    def load(cls, model_dir: str):
        pooler_config_path = os.path.join(model_dir, cls.CONFIG_NAME)
        if not os.path.exists(pooler_config_path):
            raise FileNotFoundError(f"{cls.__name__} config {pooler_config_path} does not exists.")
        with open(pooler_config_path, 'r') as f:
            pooler_config: dict[str, any] = json.load(f)
        
        pooler = cls(pooler_config['input_dim'], pooler_config['output_dim'])

        pooler_path = os.path.join(model_dir, cls.WEIGHT_NAME)
        if os.path.exists(pooler_path):
            logger.info(f'Loading {cls.__name__} from {pooler_path}')
            state_dict = torch.load(pooler_path, map_location='cpu', weights_only=True)
            pooler.load_state_dict(state_dict)
        return pooler

    def save_pooler(self, save_path: str, state_dict: dict[str, Tensor]=None):
        if not state_dict:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, self.WEIGHT_NAME))
        with open(os.path.join(save_path, self.CONFIG_NAME), 'w') as f:
            json.dump(self._config, f)


class SparseLinearProjector(SparseProjectorBase):
    """ SparseLinearProjector project a hidden state to vocab dimension with only a linear layer """
    CONFIG_NAME = 'linear_projector_config.json'
    WEIGHT_NAME = 'linear_projector.pt'

    @property
    def name(self):
        return "SparseLinearProjector"

    def forward(self, reps: Tensor):
        logits = self.linear(reps)
        return logits


class SparseDownProjector(nn.Module):
    """ A Sparse Down Projector which map `hidden_states` to `a float` number 
        This is used for reproducing BGE-m3 sparse retrieval.
    """
    CONFIG_NAME = None
    WEIGHT_NAME = 'sparse_linear.pt'
    cls_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self, vocab_size: int, hidden_dim: int, output_dim: int=1, initializer_range: float=0.02):
        super().__init__()
        self._config = {'input_dim': hidden_dim, 'output_dim': 1, 'vocab_size': vocab_size, 'initializer_range': initializer_range}
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self._init_weights(self.linear)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self._config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
    
    @property
    def name(self):
        return "SparseDownProjector"
    
    @property
    def device(self):
        return self.linear.weight.device
    
    def forward(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        token_weights = self.linear.forward(hidden_states)
        sparse_embedding = self.scater_sparse_reps(token_weights, input_ids)
        return sparse_embedding
    
    def scater_sparse_reps(self, token_weights: Tensor, input_ids: Tensor):
        """ Scater token_weights to vocab dimension based on input_ids 

            Args:
                token_weights (Tensor): Shape [batch_size, max_len, 1]. Weight of each input ids.
                input_ids (Tensor): Shape [batch_size, max_len]. Input ids.
            
            Returns:
                Tensor of
                Shape [batch_size, vocab_size]. Use `input_ids` as index, then choose the max
                weight of each ids, fill them in the coresponding [batch_size, vocab_size].
                Should keep back-propagation compatible.
        """
        batch_size, max_len = input_ids.shape
        vocab_size = self._config['vocab_size']

        # Shape: [batch_size, vocab_size]
        sparse_embedding = torch.zeros(
            (batch_size, vocab_size), 
            device=token_weights.device, 
            dtype=token_weights.dtype
        )
        sparse_embedding.scatter_reduce_(
            dim=1,
            index=input_ids,
            src=token_weights.squeeze(-1),
            reduce="amax",
            include_self=False,
        )

        # Mask out unused tokens
        unused_tokens = [self.cls_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]
        sparse_embedding[:, unused_tokens] = 0.

        return sparse_embedding
    
    def scater_sparse_reps_deprecated(self, token_weights: Tensor, input_ids: Tensor):
        """ This impl is taken from BGE-m3, it's memory comsuming because of large 
            intermediate tensor shape `bs, maxlen, vocab_size`.
        """
        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self._config['vocab_size'],
                                       dtype=token_weights.dtype,
                                       device=token_weights.device) # bs, maxlen, vocab_size
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.cls_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values # bs, vocab_size
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding
    
    @classmethod
    def build(cls, vocab_size: int, hidden_dim: int, output_dim: int=1, init_weight: Optional[Tensor]=None):
        projector = cls(vocab_size=vocab_size, hidden_dim=hidden_dim, output_dim=output_dim)
        if init_weight is not None:
            projector.linear.weight.data.copy_(init_weight)
        return projector
    
    @classmethod
    def load(cls, model_dir: str, vocab_size: int):
        pooler_path = os.path.join(model_dir, cls.WEIGHT_NAME)
        logger.info(f'Loading {cls.__name__} from {pooler_path}')
        state_dict: dict[str, Tensor] = torch.load(pooler_path, map_location='cpu', weights_only=True)

        if 'weight' in state_dict:
            # Infer input_dim
            input_dim = state_dict['weight'].shape[1]
            output_dim = state_dict['weight'].shape[0]
            # Init pooler
            pooler = cls(vocab_size=vocab_size, hidden_dim=input_dim, output_dim=output_dim)
            pooler.linear.load_state_dict(state_dict)
        else:
            # Infer input_dim
            input_dim = state_dict['linear.weight'].shape[1]
            output_dim = state_dict['linear.weight'].shape[0]
            # Init pooler
            pooler = cls(vocab_size=vocab_size, hidden_dim=input_dim, output_dim=output_dim)
            pooler.load_state_dict(state_dict)

        return pooler
    
    def save_pooler(self, save_path: str, state_dict: dict[str, Tensor]=None):
        if not state_dict:
            state_dict = self.state_dict()
        
        pfx_len = len('linear.')
        parsed_state_dict = {}
        for k, v in state_dict.items():
            if 'linear.' in k:
                parsed_state_dict[k[pfx_len:]] = v
        
        torch.save(state_dict, os.path.join(save_path, self.WEIGHT_NAME))
        with open(os.path.join(save_path, self.CONFIG_NAME), 'w') as f:
            json.dump(self._config, f)