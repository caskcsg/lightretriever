import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import transformers
from transformers.modeling_utils import PreTrainedModel, flash_attention_forward, sdpa_attention_forward

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if int(os.getenv("RANK", -1)) in [0, -1] else logging.WARN)


def apply_bidirectional_attention(model: PreTrainedModel):
    assert model.config._attn_implementation == "flash_attention_2", \
        "Flash attention implementation is needed for bidirectional attention. Because we only modify the is_causal to False in FA2."
    
    for name, module in model.named_modules():
        if hasattr(module, "is_causal") and (module.is_causal == True):
            module.is_causal = False
            logger.info(f"Modifying {name}.is_causal = {module.is_causal}")


def fa2_sdpa_4d_attn_mask_router_func(
    module: nn.Module, 
    query: Tensor, 
    key: Tensor, 
    value: Tensor, 
    attention_mask: Tensor | None,
    **kwargs,
):
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_interface = sdpa_attention_forward
    else:
        attention_interface = flash_attention_forward
    
    return attention_interface(module, query, key, value, attention_mask, **kwargs)

def hacking_fa2_forward_w_4d_attn_mask():
    """ FA2 does not support 4D attention mask. This function makes FA2 fall back SDPA if attention mask is 4D """
    transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fa2_sdpa_4d_attn_mask_router_func
