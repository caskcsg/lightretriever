# Helper functions
from typing import Mapping, Optional

import torch

def move_to_cuda(sample, device: Optional[torch.device]=None):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if isinstance(maybe_tensor, torch.Tensor):
            return maybe_tensor.to(device=device, non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)