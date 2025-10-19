#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Implementation for MLP Projecters of Dense Representations.

@Time    :   2024/08/29
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import json
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import logging
logger = logging.getLogger(__name__)

class DenseLinearProjector(nn.Module):
    """ A Simple 1-layer MLP Projector """
    CONFIG_NAME = 'pooler_config.json'
    WEIGHT_NAME = 'pooler.pt'

    def __init__(self, input_dim: int = 768, output_dim: int = 768, initializer_range: float=0.02):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'initializer_range': initializer_range}

        self._init_weights(self.linear)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self._config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
    
    @property
    def name(self):
        return "DenseLinearProjector"
    
    @property
    def device(self):
        return self.linear.weight.device

    def forward(self, reps: Tensor = None, **kwargs):
        return self.linear(reps)
    
    @classmethod
    def build(
        cls, 
        input_dim: int, 
        output_dim: int, 
        initializer_range: float=0.02, 
        model_dir: Optional[str]=None, 
        init_weight: Optional[Tensor]=None,
    ):
        """ Build a DenseLinearProjector for training from scratch. 

            Args:
                input_dim (int): The input dimension of the projector.
                output_dim (int): The output dimension of the projector.
                initializer_range (float, optional): The initializer range of the projector. Defaults to 0.02.
                model_dir (Optional[str], optional): The model directory to load the projector from. Defaults to None.
                init_weight (Optional[Tensor], optional): The initial weight of the projector. Defaults to None.
            
            Returns:
                DenseLinearProjector: The built projector.
            
            Notes: Weights Init Priority:
                1) Load from local if model_dir is not None and load successfully.
                2) Use init_weight if it is not None.
                3) Init from scratch if neither of the above works.
        """
        projector = None
        if model_dir is not None:
            try:
                projector = cls.load(model_dir)
                logger.info(f"Loaded projector from {model_dir}.")
            except FileNotFoundError:
                logger.warning(f"Projector not found in {model_dir}, fallback to init.")

        if projector is None:
            projector = cls(input_dim=input_dim, output_dim=output_dim, initializer_range=initializer_range)
            if init_weight is not None:
                projector.linear.weight.data.copy_(init_weight)
                logger.info("Initialized projector with init_weight.")
            else:
                logger.info("Initialized projector from scratch.")
        
        return projector

    @classmethod
    def load(
        cls, 
        model_dir: Optional[str]
    ):
        """ Load a DenseLinearProjector from a model directory. 

            Args:
                model_dir (Optional[str]): The model directory to load the projector from.
            
            Returns:
                DenseLinearProjector: The loaded projector.
        """
        if model_dir is None:
            raise FileNotFoundError(f"model_dir is None.")

        projector_config_path = os.path.join(model_dir, cls.CONFIG_NAME)
        if not os.path.exists(projector_config_path):
            raise FileNotFoundError(f"{cls.__name__} config {projector_config_path} does not exists.")
        
        with open(projector_config_path, 'r') as f:
            projector_config: dict[str, any] = json.load(f)
        
        projector = cls(
            input_dim=projector_config['input_dim'], 
            output_dim=projector_config['output_dim'],
            initializer_range=projector_config.get('initializer_range', 0.02),
        )

        projector_path = os.path.join(model_dir, cls.WEIGHT_NAME)
        if os.path.exists(projector_path):
            logger.info(f'Loading {cls.__name__} from {projector_path}')
            state_dict = torch.load(projector_path, map_location='cpu', weights_only=True)
            projector.load_state_dict(state_dict)
        
        return projector

    def save_pooler(self, save_path: str, state_dict: dict[str, Tensor]=None):
        """ Save the projector to a model directory. 

            Args:
                save_path (str): Path to save the projector.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not state_dict:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, self.WEIGHT_NAME))
        with open(os.path.join(save_path, self.CONFIG_NAME), 'w') as f:
            json.dump(self._config, f)