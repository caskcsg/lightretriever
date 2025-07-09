# Inspired and supported by https://github.com/huggingface/transformers/pull/31629
import builtins
from functools import wraps
from typing import Callable, Optional
from contextlib import contextmanager

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from einops import rearrange

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

def unpad_to_seqlen_dim(input_ids: Tensor, attention_mask: Tensor):
    """
    Converting input_ids with shape [batch_size, seq_len] -> [1, cu_seq_len]. 
    This function removes all the pad tokens and put useful tokens at the sequence
    length dimension, generating a line of cumulative sequence (cu_seq_len). 
    """
    assert input_ids.ndim == 2
    assert attention_mask.ndim == 2

    dtype, device = input_ids.dtype, input_ids.device
    batch_size, seq_len = input_ids.size(0), input_ids.size(1)

    # Nested input_ids
    input_ids_nested = torch.masked_select(input_ids, attention_mask.bool()).unsqueeze(0)

    # Nested position ids
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)

    position_id_nested = [torch.arange(i, dtype=dtype, device=device) for i in seqlens_in_batch]    
    position_id_nested = torch.cat(position_id_nested).unsqueeze(0)

    # Record flattened indices
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    return input_ids_nested, position_id_nested, indices, batch_size, seq_len


# Copied from flash_attn/bert_padding.py
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, values: Tensor, indices: Tensor, first_axis_dim: int):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2

        # torch.zeros has the copy-on-write feature
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # This assign op will not induce additional memory usage
        output[indices] = values

        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]

        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None

index_put_first_axis = IndexPutFirstAxis.apply


def pad_from_cu_seqlen_dim(hidden_states: Tensor, indices: Tensor, batch_size: int, seq_len: int):
    """
    Arguments:
        hidden_states: (1, total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    assert hidden_states.ndim >= 3 and hidden_states.size(0) == 1, \
        f"hidden_states with shape {hidden_states.shape} may not be hidden states from seqlen nested tensor."

    hidden_states = hidden_states.squeeze(0)        # (total_nnz, ...)

    # dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


def check_seqlen_unpad_condition(model: PreTrainedModel):
    import importlib.metadata
    from packaging import version

    assert version.parse(importlib.metadata.version("transformers")) >= version.parse("4.44.0"), \
        "Please update your transformers version >= 4.44.0"

    assert model.config._attn_implementation == "flash_attention_2", \
        "Flash attention implementation is needed for seqlen unpadding"

def check_is_compatiable_with_passing_cu_seq_lens_forward(model: PreTrainedModel):
    import importlib.metadata
    from packaging import version
    from transformers.models.gemma.modeling_gemma import GemmaPreTrainedModel

    flag_verision_match = version.parse(importlib.metadata.version("transformers")) >= version.parse("4.48.0.dev0")
    flag_not_gemma = not isinstance(model, GemmaPreTrainedModel)    # GemmaModel.forward does not accept kwargs
    return flag_verision_match and flag_not_gemma

def cumulated_forward(func: Callable[..., Tensor], pass_cu_seq_lens: bool=False):
    @wraps(func)
    def _cumulated_f(input_ids: Tensor, attention_mask: Optional[Tensor]=None, *args, **kwargs):
        if getattr(builtins, "DISABLE_CUMULATED_FORWARD", False) or attention_mask is None or attention_mask.ndim == 4:
            return func(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        
        assert kwargs.get("past_key_values", None) is None, \
            "Cumulated forward does not support past_key_values for now.."
        
        if "position_ids" in kwargs:
            kwargs.pop("position_ids")  # position_ids will be generated by cumulative processing

        input_ids_nested, position_id_nested, indices, batch_size, seq_len = unpad_to_seqlen_dim(input_ids, attention_mask)
        if pass_cu_seq_lens:
            position_ids = position_id_nested.flatten()
            indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
            cu_seq_lens = torch.cat(
                (
                    indices_q[position_ids == 0],
                    torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
                )
            )
            max_length = (position_ids.max() + 1).item()
            lm_out: BaseModelOutput = func(
                input_ids=input_ids_nested,
                attention_mask=None,
                position_ids=position_id_nested,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max_length,
                max_length_k=max_length,
                *args, **kwargs
            )
        else:
            lm_out: BaseModelOutput = func(
                input_ids=input_ids_nested,
                attention_mask=None,
                position_ids=position_id_nested,
                *args, **kwargs
            )

        # Pad to original shape for maximum compatiabilty
        for _name in ["last_hidden_state", "logits"]:
            if (_name in lm_out) and (lm_out[_name] != None):
                lm_out[_name] = pad_from_cu_seqlen_dim(lm_out[_name], indices, batch_size, seq_len)
        
        if ("hidden_states" in lm_out) and (lm_out["hidden_states"] != None):
            # Iter through Tuple for padding
            lm_out["hidden_states"] = tuple(pad_from_cu_seqlen_dim(lm_out["hidden_states"][idx], indices, batch_size, seq_len) for idx in range(len(lm_out["hidden_states"])))
        
        return lm_out

    return _cumulated_f


def apply_seqlen_cumulate(model: PreTrainedModel):
    """
    Cumulative sequence removes all pad tokens from original inputs, and stride all other 
    tokens within seq_len dimension. This is very useful to decrease memory usages and speed 
    up training. Flash attention is mandatory during the model forward.
    """
    check_seqlen_unpad_condition(model)
    pass_cu_seq_lens = check_is_compatiable_with_passing_cu_seq_lens_forward(model)
    model.forward = cumulated_forward(model.forward, pass_cu_seq_lens=pass_cu_seq_lens)
    return model


@contextmanager
def disable_cumulated_forward():
    original_state = getattr(builtins, "DISABLE_CUMULATED_FORWARD", None)
    builtins.DISABLE_CUMULATED_FORWARD = True
    try:
        yield
    finally:
        if original_state is None:
            delattr(builtins, "DISABLE_CUMULATED_FORWARD")
        else:
            builtins.DISABLE_CUMULATED_FORWARD = original_state


if __name__ == '__main__':
    input_ids = torch.tensor(
        [
            [1, 3, 101, 0, 0],
            [1, 101, 0, 0, 0],
            [1, 4, 8, 12, 101],
        ]
    ).cuda()

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    ).cuda()

    input_ids_nested, position_id_nested, indices, batch_size, seq_len = unpad_to_seqlen_dim(input_ids, attention_mask)

    hidden_dim = 100000
    hidden_state = torch.rand((1, input_ids_nested.shape[-1], hidden_dim)).cuda()

    hidden_state_padded = pad_from_cu_seqlen_dim(hidden_state, indices, batch_size, seq_len)

    exit()