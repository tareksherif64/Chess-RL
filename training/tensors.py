"""numpy <-> torch conversion shared by network, MCTS, and self-play code,
so every training-phase module moves data to the GPU the same way
(channels-first for conv2d, no dtype casts — encode_board/action_mask
already hand back float32).
"""

import numpy as np
import torch


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(8, 8, 18) or (N, 8, 8, 18) board encoding -> (N, 18, 8, 8) tensor on device."""
    if obs.ndim == 3:
        obs = obs[None, ...]
    tensor = torch.from_numpy(obs).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    """(4672,) or (N, 4672) legal-action mask -> tensor on device."""
    if mask.ndim == 1:
        mask = mask[None, :]
    return torch.from_numpy(mask).to(device)
