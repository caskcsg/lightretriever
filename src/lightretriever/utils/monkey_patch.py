import os
from functools import wraps
from typing import Callable, Optional, Tuple, List, Union

import torch
import torch.nn as nn
from torch import Tensor

import transformers
from transformers.modeling_utils import PreTrainedModel, flash_attention_forward, sdpa_attention_forward
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if int(os.getenv("RANK", -1)) in [0, -1] else logging.WARN)

def model_foward_w_bidirectional_attention_mask_for_sdpa(func: Callable[..., Tensor]):
    """ Monkey patch the model forward with SDPA attention to force bidirectional attention mask.

    Args:
        func (Callable[..., Tensor]): The original forward function of the model.

    Returns:
        Callable[..., Tensor]: The patched forward function with forced bidirectional 4D attention mask.
    """
    @wraps(func)
    def _expand_attn_mask_to_4d_f(input_ids: Tensor, attention_mask: Optional[Tensor]=None, *args, **kwargs):
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, dtype=torch.float32)
        return func(input_ids, attention_mask, *args, **kwargs)

    return _expand_attn_mask_to_4d_f


def apply_bidirectional_attention(model: PreTrainedModel):
    """ Apply bidirectional attention to the model.

    Args:
        model (PreTrainedModel): The model to apply bidirectional attention.

    Note: 
        1. For FlashAttention, we only need to modify the is_causal to False.
        2. For SDPA, we need to patch the forward function, converting 2D attention mask to 4D with full attention.
    """
    _attn_implementation = model.config._attn_implementation
    assert _attn_implementation in ["flash_attention_2", "sdpa"], \
        "Flash attention / SDPA implementation is needed for bidirectional attention."
    
    # Modify the is_causal attribute of the attention modules
    for name, module in model.named_modules():
        if hasattr(module, "is_causal") and (module.is_causal == True):
            module.is_causal = False
            logger.info(f"Modifying {name}.is_causal = {module.is_causal}")
    
    if _attn_implementation == "sdpa":
        model.forward = model_foward_w_bidirectional_attention_mask_for_sdpa(model.forward)


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
    transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["flash_attention_3"] = fa2_sdpa_4d_attn_mask_router_func


