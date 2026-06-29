# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn


class DiffNormalizer(nn.Module):
    """Normalize by running mean absolute value.

    The normalization follows MimicKit ADD:
      normalize(x) = x / max(running_mean_abs, min_diff)
    """

    def __init__(
        self,
        shape: tuple[int, ...] | list[int] | torch.Size,
        min_diff: float = 1.0e-4,
        clip: float = math.inf,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        self.min_diff = min_diff
        self.clip = clip
        self.dtype = dtype

        self.register_buffer("_count", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_mean_abs", torch.ones(tuple(shape), dtype=dtype, device=device))

        self.register_buffer("_new_count", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_new_sum_abs", torch.zeros(tuple(shape), dtype=dtype, device=device))

    def record(self, x: torch.Tensor) -> None:
        shape = self._mean_abs.shape
        if len(x.shape) <= len(shape):
            raise ValueError(
                f"Expected input rank > {len(shape)} for shape {tuple(shape)}, got tensor of rank {len(x.shape)}."
            )
        x_flat = x.flatten(start_dim=0, end_dim=len(x.shape) - len(shape) - 1)
        self._new_count += x_flat.shape[0]
        self._new_sum_abs += torch.sum(torch.abs(x_flat), dim=0)

    def update(self) -> None:
        if int(self._new_count.item()) == 0:
            return

        new_count = self._new_count.clone()
        new_sum_abs = self._new_sum_abs.clone()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(new_count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(new_sum_abs, op=torch.distributed.ReduceOp.SUM)

        new_count_value = int(new_count.item())
        if new_count_value == 0:
            self._new_count.zero_()
            self._new_sum_abs.zero_()
            return

        new_mean_abs = new_sum_abs / new_count.to(new_sum_abs.dtype)

        new_total = self._count + new_count
        new_total_f = new_total.to(new_mean_abs.dtype).clamp_min(1.0)
        w_old = self._count.to(new_mean_abs.dtype) / new_total_f
        w_new = new_count.to(new_mean_abs.dtype) / new_total_f

        self._mean_abs[:] = w_old * self._mean_abs + w_new * new_mean_abs
        self._count[:] = new_total

        self._new_count.zero_()
        self._new_sum_abs.zero_()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        diff = torch.clamp_min(self._mean_abs, self.min_diff)
        norm_x = x / diff
        norm_x = torch.clamp(norm_x, -self.clip, self.clip)
        return norm_x.to(self.dtype)

    def unnormalize(self, norm_x: torch.Tensor) -> torch.Tensor:
        diff = torch.clamp_min(self._mean_abs, self.min_diff)
        x = norm_x * diff
        return x.to(self.dtype)

