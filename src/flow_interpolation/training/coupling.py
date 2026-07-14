"""Online source-target coupling for flow-matching training."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@torch.compiler.disable()
@torch.no_grad()
def pair_batch_exact_ot(
    data: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair samples by exact quadratic-cost OT inside one minibatch.

    The Hungarian solve is exact for the empirical minibatch distributions, but
    minibatch sampling makes this only an online approximation to population OT.
    ``paired_noise[i]`` is always ``noise[permutation[i]]``.
    """
    if data.shape != noise.shape:
        raise ValueError(f"Expected matching shapes, got {data.shape} and {noise.shape}")
    if data.device != noise.device:
        raise ValueError(f"Expected matching devices, got {data.device} and {noise.device}")
    if data.ndim == 0:
        raise ValueError("Expected tensors with a batch dimension")

    batch_size = data.shape[0]
    if batch_size < 2:
        permutation = torch.arange(batch_size, device=noise.device)
        return data, noise, permutation

    data_flat = data.detach().float().reshape(batch_size, -1)
    noise_flat = noise.detach().float().reshape(batch_size, -1)
    cost = torch.cdist(data_flat, noise_flat, p=2).square()
    rows, columns = linear_sum_assignment(cost.cpu().numpy())

    permutation_array = np.empty(batch_size, dtype=np.int64)
    permutation_array[rows] = columns
    permutation = torch.as_tensor(
        permutation_array,
        device=noise.device,
        dtype=torch.long,
    )
    return data, noise[permutation], permutation
