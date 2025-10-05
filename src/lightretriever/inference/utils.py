# Helper functions
import os
import logging
import contextlib
from typing import Mapping, Optional

import torch
logger = logging.getLogger(__name__)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", '0'))
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
    DIST_BACKEND = "gloo"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
    DIST_BACKEND = "nccl"
elif hasattr(torch, 'npu') and torch.npu.is_available():
    DEVICE_TYPE = "npu"
    DIST_BACKEND = "hccl"
else:
    DEVICE_TYPE = "cpu"
    DIST_BACKEND = "gloo"


def move_to_device(sample, device: Optional[torch.device]=None):
    if device is None:
        logger.warning(f"Move sample to device without setting a device type is not encouraged, "
                       f"default device type {DEVICE_TYPE} with local rank {LOCAL_RANK} will be used.")
        device = torch.device(DEVICE_TYPE, LOCAL_RANK)
    
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor):
        if isinstance(maybe_tensor, torch.Tensor):
            return maybe_tensor.to(device=device, non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_device(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_device(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_device(sample)


def device_context(device: torch.device):
    """ Get device context for the specified device.
    Args:
        device: The device to get the context for.
    Returns:
        The device context.
    """
    if device.type == "cuda":
        device_ctx = torch.cuda.device(device)
    elif device.type == "npu":
        device_ctx = torch.npu.device(device)
    else:
        device_ctx = contextlib.nullcontext()
    return device_ctx


def empty_cache(
    device_type: str = DEVICE_TYPE,
    index: int = LOCAL_RANK,
):
    """ Empty cache for specified device type. 
    Args:
        device_type: The type of device to empty cache for.
        index: The index of device to empty cache for.
    """
    if device_type == "cuda":
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    elif device_type == "npu":
        with torch.npu.device(index):
            torch.npu.empty_cache()
    elif device_type == "mps":
        pass
    elif device_type == "cpu":
        pass
    else:
        raise NotImplementedError(f"empty_cache not implemented for device type: {device_type}")
