# Copyright (c) 2025 ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LSTM with SRU-style gating."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LSTMSRUCell(nn.Module):
    """LSTM cell with SRU-style gating and transformed candidate state."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_all = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        nn.init.orthogonal_(self.linear_all.weight)
        if bias:
            self.linear_all.bias.data[hidden_size : 2 * hidden_size] = 1.0 + torch.randn(hidden_size)

        self.transform_gate = nn.Linear(input_size, hidden_size, bias=bias)
        nn.init.orthogonal_(self.transform_gate.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.linear_all(combined)
        tx = self.transform_gate(x)

        i, f, o, g = torch.split(gates, self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g_t = torch.tanh(tx * g)

        f = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f**2
        c_next = f * c + (1.0 - f) * g_t
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class LSTM_SRU(nn.Module):
    """Multi-layer LSTM with SRU-style gating."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList(
            [LSTMSRUCell(input_size if layer_idx == 0 else hidden_size, hidden_size) for layer_idx in range(num_layers)]
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
        h, c = state
        for time_idx in range(seq_len):
            x_t = x[time_idx]
            next_h = []
            next_c = []
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(x_t, h[layer_idx], c[layer_idx])
                next_h.append(h_t)
                next_c.append(c_t)
                x_t = h_t
            h = torch.stack(next_h)
            c = torch.stack(next_c)
            outputs[time_idx] = h[-1]

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, (h, c)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h_0, c_0
