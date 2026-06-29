# Copyright (c) 2025 ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRU with SRU-style additive transformation gating."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GRUSRUCell(nn.Module):
    """GRU cell with SRU-style input transformation on the candidate state."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_all = nn.Linear(input_size + hidden_size, 2 * hidden_size, bias=bias)
        self.linear_n = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        nn.init.orthogonal_(self.linear_all.weight)
        nn.init.orthogonal_(self.linear_n.weight)
        if bias:
            self.linear_all.bias.data[:hidden_size] = 1.0 + torch.randn(hidden_size)

        self.transform_gate = nn.Linear(input_size, hidden_size, bias=bias)
        nn.init.orthogonal_(self.transform_gate.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        gates = self.linear_all(combined)
        tx = self.transform_gate(x)

        z, r = torch.split(gates, self.hidden_size, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined_new = torch.cat([x, r * h], dim=1)
        n = torch.tanh(tx * self.linear_n(combined_new))
        return (1.0 - z) * n + z * h


class GRU_SRU(nn.Module):
    """Multi-layer GRU with SRU-style candidate modulation."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList(
            [GRUSRUCell(input_size if layer_idx == 0 else hidden_size, hidden_size) for layer_idx in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if state is None:
            state = self.init_state(batch_size, x.device)

        outputs = torch.empty(seq_len, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        h = state[0]
        for time_idx in range(seq_len):
            x_t = x[time_idx]
            next_h = []
            for layer_idx, cell in enumerate(self.cells):
                h_t = cell(x_t, h[layer_idx])
                next_h.append(h_t)
                x_t = h_t
            h = torch.stack(next_h)
            outputs[time_idx] = h[-1]

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, (h, torch.zeros_like(h))

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h_0, torch.zeros_like(h_0)
