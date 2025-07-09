#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Sparse Representation Pooling / Aggregation Implementation.

@Time    :   2024/11/02
@Author  :   Ma (Ma787639046@outlook.com)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Below import are used for efficient max aggregation
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead
from .sparse_projector import SparseLinearProjector, SparseDownProjector
from ..utils.max_linear_map import max_linear_mapping

# ========
# Prompt Mask: Remove Prompt Area from attention mask
# ========
def get_sparse_attention_mask(input_ids: Tensor, attention_mask: Tensor, sep_token_id: int):
    """ Remove Prompt, first, last token from attention_mask """
    # 1. Mask logits of following position along sequence length dimension:
    #     first (e.g. [CLS], <bos>)
    #     last token (e.g. [SEP], <eos>)
    #     pad token (e.g. [PAD])   (~attention_mask)
    mask = attention_mask.bool()

    # Prompt mask
    prompt_mask = get_prompt_mask(input_ids, sep_token_id)
    mask = mask.masked_fill(prompt_mask, False)

    bs_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    last_token_indices = attention_mask.sum(dim=1) - 1
    mask[bs_indices, [0] * attention_mask.shape[0]] = False 
    mask[bs_indices, last_token_indices] = False

    return mask

def get_prompt_mask(input_ids: Tensor, sep_token_id: int):
    """ 
    Assume the inputs are `prompt + [SEP] + text`. Locate the sep_token_id first 
    and get a mask to indicate all prompt area.
    """
    assert input_ids.ndim == 2
    if sep_token_id not in input_ids:
        return torch.zeros_like(input_ids, dtype=torch.bool)

    positions = torch.argmax((input_ids == sep_token_id).int(), dim=-1)
    if torch.all(positions == input_ids.shape[-1] - 1):
        # Exclude the last [SEP] token.
        return torch.zeros_like(input_ids, dtype=torch.bool)
    
    col_indices = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
    prompt_mask = col_indices <= positions.unsqueeze(1)
    return prompt_mask

# ========
# Logits Sampling
# ========
def top_p_sampling(
    scores: Tensor,
    top_p: float, 
    filter_value: float = 0., 
    min_tokens_to_keep: int = 1
):
    """
    Adapted from `TopPLogitsWarper` in transformers/generation/logits_process.py
    """
    if top_p <= 0 or top_p >= 1:    # Safety check
        return scores

    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed

def top_k_sampling(
    scores: Tensor,
    top_k: float, 
    filter_value: float = 0., 
    min_tokens_to_keep: int = 1
):
    """
    Adapted from `TopKLogitsWarper` in transformers/generation/logits_process.py
    """
    if top_k <= 0:  # Safety check
        return scores
    
    top_k = max(top_k, min_tokens_to_keep)
    top_k = min(top_k, scores.size(-1))
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed

def count_unique(input_ids: Tensor):
    sorted_ids, _ = torch.sort(input_ids, dim=-1)   # Sort input_ids along the sequence dimension
    diff = sorted_ids[:, 1:] != sorted_ids[:, :-1]  # Compare adjacent elements to identify unique ones
    unique_counts = diff.sum(dim=-1) + 1    # Always count the first element in each row as unique
    return unique_counts

def adaptive_top_k_sampling(
    scores: Tensor,
    input_ids: Tensor, 
    expansion_ratio: float,
    filter_value: float = 0., 
    min_tokens_to_keep: int = 1
):
    """
    Set top-k up-boundary adaptively w.r.t the number of unique tokens within the input.

    Args:
        scores (Tensor): Sparse embeddings. Shape [batch_size, vocab_dim]
        input_ids (Tensor): Input ids. Shape [batch_size, seq_len]
        expansion_ratio (float): The upper boundary of expansion ratio w.r.t the number of unique tokens.
        filter_value (float): Fill the scores exceed top-k margin with this value. Default to 0.
        min_tokens_to_keep (int): Min number of token to preserve.
    """
    cnt = count_unique(input_ids)      # Shape [batch_size]
    top_k = (cnt * expansion_ratio).to(dtype=input_ids.dtype)

    top_k = torch.maximum(top_k, torch.tensor(min_tokens_to_keep, dtype=top_k.dtype, device=top_k.device))
    top_k = torch.minimum(top_k, torch.tensor(scores.size(-1), dtype=top_k.dtype, device=top_k.device))

    # Get k-th largest value of each line of scores with top_k
    sorted_scores, _ = torch.sort(scores, descending=True, dim=-1)
    kth_largest_values = sorted_scores[torch.arange(scores.size(0), dtype=top_k.dtype, device=top_k.device), top_k-1]
    kth_largest_values = kth_largest_values.view(-1, 1)

    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < kth_largest_values
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed

def get_unique_token_ids(input_ids: Tensor, attention_mask: Tensor) -> list[list[int]]:
    """ Get unique token ids from each line of `input_ids` 
        Args:
            input_ids (Tensor): Shape [batch_size, seq_len].
            attention_mask (Tensor): Shape [batch_size, seq_len].
    """
    assert input_ids.ndim == 2
    input_ids = input_ids.masked_fill(~attention_mask.bool(), -1)
    unique_ids = [torch.unique(line[line != -1]).tolist() for line in input_ids]
    return unique_ids

def get_scores_with_indices(
    scores: Tensor, 
    indices: list[list[int]],
    filter_value: float = 0., 
) -> Tensor:
    """ Mask scores by setting values to `filter_value` for tokens not in indices.
        This function handles indices (list[list[int]]) that does not have same
        lengths by itering with a for loop.

        Args:
            scores (Tensor): Shape [batch_size, vocab_size].
            indices (list[list[int]]): Unique token ids for each batch sample.
        Returns:
            Masked scores Tensor with shape [batch_size, vocab_size].
    """
    assert scores.ndim == 2
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for i in range(scores.shape[0]):
        mask[i, indices[i]] = True

    scores = scores.masked_fill(~mask, filter_value)
    return scores

def top_k_sampling_bidirection(
    scores: Tensor,
    top_k: float, 
    use_largest: bool = True,
    use_smallest: bool = True,
    filter_value: float = 0., 
    min_tokens_to_keep: int = 1
):
    """
    Preserve the top-k logits from both `Descend side` and `Ascend side`
    """
    if top_k <= 0:  # Safety check
        return scores
    
    top_k = max(top_k, min_tokens_to_keep)
    top_k = min(top_k, scores.size(-1))
    # Remove all tokens with a probability less than the last token of the top-k
    if use_largest:
        indices_to_remove_dsc = scores < torch.topk(scores, top_k, largest=True)[0][..., -1, None]
    if use_smallest:
        indices_to_remove_asc = scores > torch.topk(scores, top_k, largest=False)[0][..., -1, None]

    if use_largest and use_smallest:
        indices_to_remove = torch.logical_and(indices_to_remove_dsc, indices_to_remove_asc)
    elif use_largest:
        indices_to_remove = indices_to_remove_dsc
    elif use_smallest:
        indices_to_remove = indices_to_remove_asc
    else:
        raise NotImplementedError("Please set at least one of `use_largest`, `use_smallest` to True.")
    
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed


# ========
# Aggregation: Logits -> Representations
# ========
def get_lm_head_weight_bias(hidden_states: Tensor, lm_head: nn.Module):
    """ Forward `hidden_states` with layers before up-projection of `lm_head` (if exists),
        return `weight` and `bias (optional)` of `lm_head` up-projection layer.
    """
    if isinstance(lm_head, BertLMPredictionHead):
        hidden_states = lm_head.forward(hidden_states)
        weight = lm_head.decoder.weight.T
        bias = lm_head.bias
    elif isinstance(lm_head, XLMRobertaLMHead):
        hidden_states = lm_head.dense.forward(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = lm_head.layer_norm.forward(hidden_states)
        weight = lm_head.decoder.weight.T
        bias = lm_head.bias
    elif isinstance(lm_head, (SparseLinearProjector, SparseDownProjector)):
        weight = lm_head.linear.weight.T
        bias = lm_head.linear.bias
    elif isinstance(lm_head, nn.Linear):
        weight = lm_head.weight.T
        bias = lm_head.bias
    else:
        raise NotImplementedError(f"Unrecognized type of lm_head {type(lm_head)}.")
    return hidden_states, weight, bias


def aggregate(
    hidden_states: Tensor,
    lm_head: nn.Module,
    sparse_attention_mask: Tensor,
    sparse_use_max_aggregation: bool = True,
):
    """
    Memory-efficient computation of aggregated values along the sequence dimension using a custom autograd function.

    Args:
        hidden_states (Tensor): Shape [batch_size, seq_len, hidden_dim].
        lm_head (nn.Module): LM head.
        sparse_attention_mask (Tensor, optional): Shape [batch_size, seq_len], where True means valid positions.
        sparse_use_max_aggregation (bool): True for Max Aggregation (GPU MEM Efficient), False for Mean Aggregation (GPU MEM Inefficient)

    Returns:
        aggregated_values (Tensor): Shape [batch_size, vocab_size].
    """
    assert hidden_states.ndim == 3, f"hidden_states.shape is {hidden_states.shape}, maybe you set both `sparse_pooling_strategy` and `sparse_use_max_aggregation`. Please use pooling / aggregation only once at a time."

    hidden_states, weight, bias = get_lm_head_weight_bias(hidden_states, lm_head)

    # ** Aggregate logits **
    # 2. Aggregate along sequence length dimension.
    #    Logits shape: bs, seq_len, vocab_size -> bs, vocab_size
    if sparse_use_max_aggregation:
        logits = max_linear_mapping(hidden_states, weight, bias, sparse_attention_mask)  # bs, seq_len, vocab_size -> bs, vocab_size
    
    # TODO: Add efficient mean aggregation
    else:
        logits: Tensor = lm_head.forward(hidden_states)
        logits = logits.masked_fill(~sparse_attention_mask, torch.finfo(logits.dtype).min)
        logits = torch.mean(logits, dim=1)

    return logits


# ========
# Sparse Converter Mixin: Base Class for converting PyTorch/Numpy array to json/string reps.
# ========


