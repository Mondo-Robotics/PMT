"""PerceptiveMotionAdapter + shared init helpers (split from the 2017-line monolith).

Independent of the tracker classes; imported by both adapter_tracker.py and
token_tracker.py for the shared ``_small_init_last_linear`` / ``_zero_init_last_linear``
helpers.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.registry import register_network


def _small_init_last_linear(module: nn.Module, gain: float = 0.01) -> None:
    last_linear = None
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            last_linear = layer
    if last_linear is not None:
        nn.init.xavier_uniform_(last_linear.weight, gain=gain)
        nn.init.zeros_(last_linear.bias)


def _zero_init_last_linear(module: nn.Module) -> None:
    last_linear = None
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            last_linear = layer
    if last_linear is not None:
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)


@register_network("PerceptiveMotionAdapter", compat_name="perceptive_motion_adapter")
class PerceptiveMotionAdapter(nn.Module):
    """Predict a task latent, preferably as a gated residual over z_flat."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        mode: str = "gated_residual",
        delta_scale: float = 1.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.mode = str(mode)
        self.delta_scale = float(delta_scale)
        self.gate_bias = float(gate_bias)

        valid_modes = {"gated_residual", "residual", "absolute", "none", "no_adapter"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown PMA mode: {self.mode}. Supported modes: {sorted(valid_modes)}")

        if self.mode in {"none", "no_adapter"}:
            self.net = None
        else:
            output_dim = 2 * self.latent_dim if self.mode == "gated_residual" else self.latent_dim
            self.net = MLP(
                input_dim=int(input_dim),
                output_dim=output_dim,
                hidden_dims=list(hidden_dims),
                activation=activation,
            )
            _small_init_last_linear(self.net)

    def forward(self, adapter_input: torch.Tensor, z_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.net is None:
            zeros = torch.zeros_like(z_flat)
            return {
                "z_task": z_flat,
                "delta_z": zeros,
                "gate": zeros,
                "identity_residual": zeros,
            }

        raw = self.net(adapter_input)
        if self.mode == "absolute":
            z_task = raw
            delta_z = z_task - z_flat
            gate = torch.ones_like(delta_z)
            identity_residual = delta_z
        elif self.mode == "residual":
            delta_z = self.delta_scale * raw
            gate = torch.ones_like(delta_z)
            identity_residual = delta_z
            z_task = z_flat + identity_residual
        else:
            delta_raw, gate_logits = torch.chunk(raw, 2, dim=-1)
            delta_z = self.delta_scale * delta_raw
            gate = torch.sigmoid(gate_logits + self.gate_bias)
            identity_residual = gate * delta_z
            z_task = z_flat + identity_residual

        return {
            "z_task": z_task,
            "delta_z": delta_z,
            "gate": gate,
            "identity_residual": identity_residual,
        }
