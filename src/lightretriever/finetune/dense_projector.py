#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Implementation for MLP Projecters of Dense Representations.

@Time    :   2024/08/29
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import json
import torch
import torch.nn as nn
from torch import Tensor

import logging
logger = logging.getLogger(__name__)

class DenseLinearProjector(nn.Module):
    """ A Simpler 1-layer MLP Projector """
    def __init__(self, input_dim: int = 768, output_dim: int = 768, **kwargs):
        super(DenseLinearProjector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, reps: Tensor = None, **kwargs):
        return self.linear(reps)

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if os.path.exists(pooler_path):
            logger.info(f'Loading Pooler from {pooler_path}')
            state_dict = torch.load(pooler_path, map_location='cpu')
            self.load_state_dict(state_dict)
            return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path, state_dict=None):
        if not state_dict:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)