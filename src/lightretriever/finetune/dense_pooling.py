#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Dense Rep Pooling Function.

@Time    :   2022/11/05
@Author  :   Ma (Ma787639046@outlook.com)
'''
import torch
from torch import Tensor

def pooling(
    last_hidden: Tensor,
    hidden_states: tuple[Tensor]=None, 
    attention_mask: Tensor=None,
    pooling_strategy: str='mean'
):
    """
    Pooling to get the sentence embedding
    'none': Do not pooling
    'cls': [CLS] representation without BERT/RoBERTa's MLP pooler.
    'mean': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    'lasttoken': get the last token representation that is not padding.
    'second_to_last': get the last 2nd token representation that is not padding.
    'third_to_last: get the last 3rd token representation that is not padding.
    """
    if pooling_strategy == 'none':
        return last_hidden

    elif pooling_strategy == 'cls':
        return last_hidden[:, 0]
    
    elif pooling_strategy == "mean":
        return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
    
    elif pooling_strategy == "avg_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        return ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    
    elif pooling_strategy == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        return ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    
    elif pooling_strategy == 'lasttoken':
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1)
            last_token_indices = sequence_lengths - 1
            return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), last_token_indices]
    
    elif pooling_strategy == 'second_to_last':
        assert last_hidden.shape[1] >= 2, f"No enough tokens on dim=1 for last_hidden shape {last_hidden.shape}."
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden[:, -2]
        else:
            sequence_lengths = attention_mask.sum(dim=1)
            last_token_indices = sequence_lengths - 2
            assert torch.all(last_token_indices >= 0), \
                f"last_token_indices {last_token_indices} has a value < 0. Please check your inputs for pooling."
            return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), last_token_indices]
    
    elif pooling_strategy == 'third_to_last':
        assert last_hidden.shape[1] >= 3, f"No enough tokens on dim=1 for last_hidden shape {last_hidden.shape}."
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden[:, -3]
        else:
            sequence_lengths = attention_mask.sum(dim=1)
            last_token_indices = sequence_lengths - 3
            assert torch.all(last_token_indices >= 0), \
                f"last_token_indices {last_token_indices} has a value < 0. Please check your inputs for pooling."
            return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), last_token_indices]
    
    else:
        raise NotImplementedError()


def mean_eos_pooling(
    last_hidden_states: Tensor, 
    input_ids: Tensor, 
    attention_mask: Tensor,
    eos_id: int
):
    """ Find eos position of each sequence from last_hidden_states, 
        mean reduce the values from eos for each sequence

        Args:
            last_hidden_states (Tensor): Shape [batch_size, seq_len, hidden_size].
            input_ids (Tensor): Shape [batch_size, seq_len]. Input ids.
            attention_mask (Tensor): Shape [batch_size, seq_len]. 2-D bool attention mask.
            eos_id (int): Eos token id.
    """
    eos_mask = (input_ids == eos_id) & attention_mask.bool()  # [batch_size, seq_len]
    eos_mask = eos_mask.unsqueeze(-1).type(last_hidden_states.dtype)  # [batch_size, seq_len, 1]
    sum_eos = (last_hidden_states * eos_mask).sum(dim=1)  # [batch_size, hidden_size]
    eos_counts = eos_mask.sum(dim=1)  # [batch_size, 1]
    pooled_output = sum_eos / eos_counts.clamp(min=1)
    return pooled_output

