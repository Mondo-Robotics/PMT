"""PerceptiveMotionTokenTracker + its internal token/encoder/decoder helpers.

Split from the 2017-line monolith. State-dict keys are by attribute-path within
this module; moving these classes here is checkpoint-safe (internal structure
unchanged). The static checkpoint helpers on PerceptiveMotionAdapterTracker are
reused via import (their names are not part of any state_dict).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.networks.actor_critic import FSQ
from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.registry import register_network

from .adapter import _zero_init_last_linear
from .adapter_tracker import PerceptiveMotionAdapterTracker


class _ResidualFSQBehaviorTokenizer(nn.Module):
    """Behavior tokenizer: attention-pool the future-motion window into tokens, then
    apply a residual Finite-Scalar-Quantization (FSQ) stack to make each token DISCRETE.

    This is the P-CaRBT (Perception-Conditioned Contact-Aware Residual Behavior
    Tokenizer) quantizer. It is a drop-in replacement for ``_MotionTokenizer`` — same
    ``forward(flat_future_motion) -> (z_e, z_q)`` signature, same ``[B, num_tokens,
    token_dim]`` output shape — so the downstream decoder/adapter/aux heads are
    untouched. The FSQ primitive is reused from ``actor_critic.FSQ`` (no learned
    codebook → no collapse), per the design doc (latent_space_represent_pmt.md §3.3).

    Residual-FSQ: at each level ``m`` the CURRENT residual ``r`` (in token space) is
    projected to the FSQ grid, quantized with a straight-through estimator, projected
    back up, accumulated into ``z_q``, and subtracted from ``r`` (coarse-to-fine). With
    ``num_residual_levels == 1`` this is plain per-token FSQ.

    Grouped tokenization is intentionally NOT implemented here: the reference joint
    vector ``q_ref(29)`` is in an interleaved Isaac-Lab BFS order (see
    the PCRBT joint-order note), so a hard-coded ``q_ref[0:12]==legs``
    slice would be wrong. Grouping is deferred to a name-resolved future variant.
    """

    def __init__(
        self,
        future_motion_dim: int,
        future_motion_len: int,
        model_dim: int,
        token_dim: int,
        num_tokens: int,
        activation: str,
        num_heads: int,
        fsq_levels: Sequence[int],
        num_residual_levels: int = 1,
    ) -> None:
        super().__init__()
        self.future_motion_dim = int(future_motion_dim)
        self.future_motion_len = int(future_motion_len)
        self.frame_dim = self.future_motion_dim // self.future_motion_len
        self.model_dim = int(model_dim)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.fsq_levels = [int(level) for level in fsq_levels]
        self.fsq_dim = len(self.fsq_levels)
        self.num_residual_levels = int(num_residual_levels)

        if self.future_motion_len <= 0:
            raise ValueError("future_motion_len must be positive.")
        if self.future_motion_dim % self.future_motion_len != 0:
            raise ValueError(
                f"future_motion_dim={self.future_motion_dim} must be divisible by "
                f"future_motion_len={self.future_motion_len}."
            )
        if self.model_dim % int(num_heads) != 0:
            raise ValueError(f"model_dim={self.model_dim} must be divisible by num_heads={num_heads}.")
        if self.num_residual_levels < 1:
            raise ValueError(f"num_residual_levels must be >= 1, got {self.num_residual_levels}.")
        if self.fsq_dim < 1:
            raise ValueError("fsq_levels must be a non-empty list of per-dimension levels.")

        self.frame_embed = MLP(
            input_dim=self.frame_dim,
            output_dim=self.model_dim,
            hidden_dims=[self.model_dim],
            activation=activation,
        )
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, self.model_dim))
        self.attn = nn.MultiheadAttention(self.model_dim, int(num_heads), batch_first=True)
        self.token_proj = MLP(
            input_dim=self.model_dim,
            output_dim=self.token_dim,
            hidden_dims=[self.model_dim],
            activation=activation,
        )
        self.token_norm = nn.LayerNorm(self.token_dim)

        # Residual-FSQ stack: per level a down-projector to the FSQ grid, a (parameter-free)
        # FSQ quantizer, and an up-projector back to token space.
        self.fsq_in = nn.ModuleList(
            [nn.Linear(self.token_dim, self.fsq_dim) for _ in range(self.num_residual_levels)]
        )
        self.fsq_quantizers = nn.ModuleList(
            [FSQ(list(self.fsq_levels)) for _ in range(self.num_residual_levels)]
        )
        self.fsq_out = nn.ModuleList(
            [nn.Linear(self.fsq_dim, self.token_dim) for _ in range(self.num_residual_levels)]
        )
        self._last_code_indices: list[torch.Tensor] = []
        self._last_grids: list[torch.Tensor] = []

    def forward(self, flat_future_motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = flat_future_motion.shape[0]
        frames = flat_future_motion.reshape(batch_size, self.future_motion_len, self.frame_dim)
        frame_features = self.frame_embed(frames.reshape(batch_size * self.future_motion_len, self.frame_dim))
        frame_features = frame_features.reshape(batch_size, self.future_motion_len, self.model_dim)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        token_features, _ = self.attn(queries, frame_features, frame_features, need_weights=False)
        z_e = self.token_proj(token_features.reshape(batch_size * self.num_tokens, self.model_dim))
        z_e = z_e.reshape(batch_size, self.num_tokens, self.token_dim)

        # Residual quantization in token space (coarse-to-fine).
        residual = self.token_norm(z_e)
        z_q = torch.zeros_like(residual)
        code_indices: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []
        for level in range(self.num_residual_levels):
            grid = self.fsq_in[level](residual)
            quantizer = self.fsq_quantizers[level]
            quantized = quantizer.quantize(grid)
            dequantized = self.fsq_out[level](quantized)
            z_q = z_q + dequantized
            residual = residual - dequantized
            grids.append(grid)
            # Indices for usage/entropy diagnostics (no grad).
            code_indices.append(quantizer.encode(grid.detach()))
        self._last_code_indices = code_indices
        self._last_grids = grids
        return z_e, z_q

    def last_code_indices(self) -> list[torch.Tensor]:
        """Per-level FSQ integer indices [B, num_tokens, fsq_dim] from the last forward."""
        return self._last_code_indices

    def soft_usage_entropy(self) -> torch.Tensor | None:
        """Differentiable code-usage entropy surrogate over the last forward's grids.

        For each level and FSQ dimension we form a soft assignment of the bounded
        pre-quant value to that dimension's integer levels (softmax over negative
        squared distance to each level center), average the assignment over the batch
        to get an expected code distribution, and take its entropy normalized by
        ``log(num_levels)``. Maximizing this (the caller negates it) spreads usage
        across codes — an anti-dead-code regularizer that, unlike a histogram of
        detached indices, actually carries gradient into ``fsq_in``.
        """
        grids = getattr(self, "_last_grids", None)
        if not grids:
            return None
        device = grids[0].device
        dtype = grids[0].dtype
        log_levels = torch.log(torch.tensor([float(level) for level in self.fsq_levels], device=device, dtype=dtype))
        entropies: list[torch.Tensor] = []
        for grid in grids:  # grid: [B, num_tokens, fsq_dim]
            flat = grid.reshape(-1, grid.shape[-1])  # [N, fsq_dim]
            quantizer = self.fsq_quantizers[0]
            bounded = quantizer.bound(flat)  # ≈[-(L-1)/2, (L-1)/2] per dim
            half_width = quantizer._half_width.to(flat.device, flat.dtype)  # floor(L/2)
            for dim, num_levels in enumerate(self.fsq_levels):
                # Integer level centers in the bounded space: [-hw, ..., +hw].
                centers = torch.arange(num_levels, device=flat.device, dtype=flat.dtype) - half_width[dim]
                d2 = (bounded[:, dim : dim + 1] - centers.unsqueeze(0)) ** 2  # [N, L]
                soft = torch.softmax(-d2, dim=-1)  # [N, L] soft code assignment
                expected = soft.mean(dim=0)  # [L] expected usage
                ent = -(expected * expected.clamp_min(1.0e-9).log()).sum()
                entropies.append(ent / log_levels[dim])
        return torch.stack(entropies).mean()


class _MotionTokenizer(nn.Module):
    """Encode future flat motion commands into reusable motion tokens."""

    def __init__(
        self,
        future_motion_dim: int,
        future_motion_len: int,
        model_dim: int,
        token_dim: int,
        num_tokens: int,
        activation: str,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.future_motion_dim = int(future_motion_dim)
        self.future_motion_len = int(future_motion_len)
        self.frame_dim = self.future_motion_dim // self.future_motion_len
        self.model_dim = int(model_dim)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)

        if self.future_motion_len <= 0:
            raise ValueError("future_motion_len must be positive.")
        if self.future_motion_dim % self.future_motion_len != 0:
            raise ValueError(
                f"future_motion_dim={self.future_motion_dim} must be divisible by "
                f"future_motion_len={self.future_motion_len}."
            )
        if self.model_dim % int(num_heads) != 0:
            raise ValueError(f"model_dim={self.model_dim} must be divisible by num_heads={num_heads}.")

        self.frame_embed = MLP(
            input_dim=self.frame_dim,
            output_dim=self.model_dim,
            hidden_dims=[self.model_dim],
            activation=activation,
        )
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, self.model_dim))
        self.attn = nn.MultiheadAttention(self.model_dim, int(num_heads), batch_first=True)
        self.token_proj = MLP(
            input_dim=self.model_dim,
            output_dim=self.token_dim,
            hidden_dims=[self.model_dim],
            activation=activation,
        )
        self.token_norm = nn.LayerNorm(self.token_dim)

    def forward(self, flat_future_motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = flat_future_motion.shape[0]
        frames = flat_future_motion.reshape(batch_size, self.future_motion_len, self.frame_dim)
        frame_features = self.frame_embed(frames.reshape(batch_size * self.future_motion_len, self.frame_dim))
        frame_features = frame_features.reshape(batch_size, self.future_motion_len, self.model_dim)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        token_features, _ = self.attn(queries, frame_features, frame_features, need_weights=False)
        z_e = self.token_proj(token_features.reshape(batch_size * self.num_tokens, self.model_dim))
        z_e = z_e.reshape(batch_size, self.num_tokens, self.token_dim)
        z_q = self.token_norm(z_e)
        return z_e, z_q


class _ProprioHistoryEncoder(nn.Module):
    def __init__(self, history_dim: int, embedding_dim: int, activation: str) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=int(history_dim),
            output_dim=int(embedding_dim),
            hidden_dims=[int(embedding_dim), int(embedding_dim)],
            activation=activation,
        )
        self.norm = nn.LayerNorm(int(embedding_dim))

    def forward(self, history_obs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(history_obs))


class _HeightScanEncoder(nn.Module):
    def __init__(
        self,
        height_scan_dim: int,
        context_dim: int,
        num_tokens: int,
        token_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.context_net = MLP(
            input_dim=int(height_scan_dim),
            output_dim=int(context_dim),
            hidden_dims=[int(context_dim), int(context_dim)],
            activation=activation,
        )
        self.token_net = MLP(
            input_dim=int(height_scan_dim),
            output_dim=self.num_tokens * self.token_dim,
            hidden_dims=[int(context_dim), int(context_dim)],
            activation=activation,
        )
        self.context_norm = nn.LayerNorm(int(context_dim))
        self.token_norm = nn.LayerNorm(self.token_dim)

    def forward(self, height_scan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context_norm(self.context_net(height_scan))
        tokens = self.token_net(height_scan).reshape(height_scan.shape[0], self.num_tokens, self.token_dim)
        tokens = self.token_norm(tokens)
        return context, tokens


class _FootEventLatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str,
        latent_key: str = "e_prior",
        contact_key: str = "contact_logits_prior",
        time_key: str = "time_to_event_prior",
    ) -> None:
        super().__init__()
        self.latent_key = str(latent_key)
        self.contact_key = str(contact_key)
        self.time_key = str(time_key)
        self.latent = MLP(
            input_dim=int(input_dim),
            output_dim=int(latent_dim),
            hidden_dims=[int(latent_dim), int(latent_dim)],
            activation=activation,
        )
        self.contact_head = nn.Linear(int(latent_dim), 2)
        self.time_head = nn.Linear(int(latent_dim), 2)
        self.norm = nn.LayerNorm(int(latent_dim))

    def forward(self, foot_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.norm(self.latent(foot_obs))
        return {
            self.latent_key: latent,
            self.contact_key: self.contact_head(latent),
            self.time_key: F.softplus(self.time_head(latent)),
        }


class _TerrainMotionAdapter(nn.Module):
    """Rewrite flat motion tokens with terrain/proprio/foot-event-prior context."""

    def __init__(
        self,
        token_dim: int,
        terrain_context_dim: int,
        history_embedding_dim: int,
        foot_event_dim: int,
        num_motion_tokens: int,
        num_height_tokens: int,
        num_heads: int,
        hidden_dims: Sequence[int],
        activation: str,
        mode: str,
        delta_scale: float,
        gate_bias: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_motion_tokens = int(num_motion_tokens)
        self.num_height_tokens = int(num_height_tokens)
        self.num_heads = int(num_heads)
        self.mode = str(mode)
        self.delta_scale = float(delta_scale)
        self.gate_bias = float(gate_bias)
        valid_modes = {"gated_residual", "residual", "absolute", "none", "no_adapter"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown terrain PMA mode: {self.mode}. Supported modes: {sorted(valid_modes)}")
        if self.num_heads <= 0 or self.token_dim % self.num_heads != 0:
            raise ValueError(
                f"Terrain PMA requires token_dim divisible by num_heads, got {self.token_dim=} {self.num_heads=}."
            )

        if self.mode in {"none", "no_adapter"}:
            self.net = None
        else:
            global_context_dim = int(terrain_context_dim) + int(history_embedding_dim) + int(foot_event_dim)
            self.global_context_proj = nn.Linear(global_context_dim, self.token_dim)
            self.motion_token_pos = nn.Parameter(torch.zeros(1, self.num_motion_tokens, self.token_dim))
            self.height_token_pos = nn.Parameter(torch.zeros(1, self.num_height_tokens, self.token_dim))
            self.terrain_cross_attn = nn.MultiheadAttention(
                embed_dim=self.token_dim,
                num_heads=self.num_heads,
                batch_first=True,
            )
            self.terrain_token_norm = nn.LayerNorm(self.token_dim)
            self.attended_token_norm = nn.LayerNorm(self.token_dim)
            input_dim = 2 * self.token_dim + global_context_dim
            output_dim = 2 * self.token_dim if self.mode == "gated_residual" else self.token_dim
            self.net = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=list(hidden_dims),
                activation=activation,
            )
            _zero_init_last_linear(self.net)

    def forward(
        self,
        z_flat: torch.Tensor,
        terrain_context: torch.Tensor,
        h_prop: torch.Tensor,
        e_prior: torch.Tensor,
        terrain_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.net is None:
            zeros = torch.zeros_like(z_flat)
            return {
                "z_task": z_flat,
                "z_task_q": z_flat,
                "delta_z": zeros,
                "gate": zeros,
                "identity_residual": zeros,
            }

        batch_size = z_flat.shape[0]
        num_tokens = z_flat.shape[1]
        if num_tokens > self.motion_token_pos.shape[1]:
            raise ValueError(
                f"Terrain PMA received {num_tokens} motion tokens but was configured for "
                f"{self.motion_token_pos.shape[1]}."
            )
        if terrain_tokens is None:
            terrain_tokens = terrain_context.new_zeros(batch_size, 1, self.token_dim)
        if terrain_tokens.ndim != 3 or terrain_tokens.shape[0] != batch_size or terrain_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                "terrain_tokens must have shape [batch, num_height_tokens, token_dim], got "
                f"{tuple(terrain_tokens.shape)} for token_dim={self.token_dim}."
            )
        if terrain_tokens.shape[1] > self.height_token_pos.shape[1]:
            raise ValueError(
                f"Terrain PMA received {terrain_tokens.shape[1]} height tokens but was configured for "
                f"{self.height_token_pos.shape[1]}."
            )

        global_context = torch.cat((terrain_context, h_prop, e_prior), dim=-1)
        query = z_flat + self.global_context_proj(global_context).unsqueeze(1)
        query = query + self.motion_token_pos[:, :num_tokens, :]
        terrain_key_value = terrain_tokens + self.height_token_pos[:, : terrain_tokens.shape[1], :]
        terrain_key_value = self.terrain_token_norm(terrain_key_value)
        attended_tokens, _ = self.terrain_cross_attn(
            query=query,
            key=terrain_key_value,
            value=terrain_key_value,
            need_weights=False,
        )
        attended_tokens = self.attended_token_norm(z_flat + attended_tokens)
        context = global_context.unsqueeze(1).expand(-1, num_tokens, -1)
        adapter_input = torch.cat((z_flat, attended_tokens, context), dim=-1)
        raw = self.net(adapter_input.reshape(batch_size * num_tokens, -1))
        raw = raw.reshape(batch_size, num_tokens, -1)

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
            "z_task_q": z_task,
            "delta_z": delta_z,
            "gate": gate,
            "identity_residual": identity_residual,
        }


class _TokenConditionedPMTDecoder(nn.Module):
    """Frozen reusable decoder: proprio/history embedding plus motion tokens to PD targets."""

    def __init__(
        self,
        current_proprio_dim: int,
        history_embedding_dim: int,
        token_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        activation: str,
        init_noise_std: float,
    ) -> None:
        super().__init__()
        self.actor = MLP(
            input_dim=int(current_proprio_dim) + int(history_embedding_dim) + int(token_dim),
            output_dim=int(num_actions),
            hidden_dims=list(hidden_dims),
            activation=activation,
        )
        self.std = nn.Parameter(float(init_noise_std) * torch.ones(int(num_actions)))
        self._distribution: Normal | None = None

    def update_distribution(
        self,
        current_proprio: torch.Tensor,
        h_prop: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> None:
        pooled_token = motion_tokens.mean(dim=1)
        mean = self.actor(torch.cat((current_proprio, h_prop, pooled_token), dim=-1))
        std = self.std.clamp_min(1.0e-6).expand_as(mean)
        self._distribution = Normal(mean, std)

    def act(
        self,
        current_proprio: torch.Tensor,
        h_prop: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.update_distribution(current_proprio, h_prop, motion_tokens)
        return self.distribution.sample()

    def act_inference(
        self,
        current_proprio: torch.Tensor,
        h_prop: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.update_distribution(current_proprio, h_prop, motion_tokens)
        return self.action_mean

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def distribution(self) -> Normal:
        if self._distribution is None:
            raise RuntimeError("Action distribution has not been built. Call update_distribution first.")
        return self._distribution

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)


class _MotionAuxDecoder(nn.Module):
    """Training-only auxiliary decoders; not part of the deployed PMT action path."""

    def __init__(
        self,
        token_dim: int,
        future_motion_dim: int,
        hidden_dim: int,
        activation: str,
        use_phase_head: bool = False,
    ) -> None:
        super().__init__()
        self.motion_head = MLP(
            input_dim=int(token_dim),
            output_dim=int(future_motion_dim),
            hidden_dims=[int(hidden_dim), int(hidden_dim)],
            activation=activation,
        )
        self.contact_head = MLP(
            input_dim=int(token_dim),
            output_dim=2,
            hidden_dims=[int(hidden_dim)],
            activation=activation,
        )
        self.clearance_head = MLP(
            input_dim=int(token_dim),
            output_dim=2,
            hidden_dims=[int(hidden_dim)],
            activation=activation,
        )
        # Optional gait-phase head (sin/cos). Off by default so existing PMT
        # checkpoints (which lack these weights) still load strictly.
        self.use_phase_head = bool(use_phase_head)
        self.phase_head = (
            MLP(
                input_dim=int(token_dim),
                output_dim=2,
                hidden_dims=[int(hidden_dim)],
                activation=activation,
            )
            if self.use_phase_head
            else None
        )

    def forward(self, motion_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled_token = motion_tokens.mean(dim=1)
        outputs = {
            "future_motion_hat": self.motion_head(pooled_token),
            "contact_logits_aux": self.contact_head(pooled_token),
            "clearance_hat": self.clearance_head(pooled_token),
        }
        if self.phase_head is not None:
            # Normalize to the unit circle so the head predicts a valid (sin, cos).
            phase = self.phase_head(pooled_token)
            outputs["phase_hat"] = F.normalize(phase, dim=-1, eps=1.0e-6)
        return outputs


@register_network("PerceptiveMotionTokenTracker", compat_name="perceptive_motion_token_tracker")
class PerceptiveMotionTokenTracker(nn.Module):
    """Final PMA/PMT scaffold with a token-only frozen decoder contract.

    The deployed decoder consumes only current proprio, proprio/history embedding,
    and adapted motion tokens. Future flat q/qdot commands are tokenizer inputs
    only, and height/foot-event cues are folded into the token by PMA.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        policy_set_name: str = "policy",
        history_set_name: str = "policy_history",
        future_motion_set_name: str = "future_motion_window",
        teacher_future_motion_set_name: str = "teacher_future_motion_window",
        height_scan_set_name: str = "height_scan",
        critic_set_name: str = "critic",
        future_motion_len: int = 6,
        num_motion_tokens: int = 4,
        motion_token_dim: int = 64,
        model_dim: int = 128,
        token_num_heads: int = 4,
        history_embedding_dim: int = 128,
        terrain_context_dim: int = 128,
        foot_event_dim: int = 64,
        num_height_tokens: int = 4,
        foot_event_posterior_set_name: str = "foot_event_posterior",
        use_foot_event_posterior: bool = True,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256),
        adapter_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        adapter_mode: str = "gated_residual",
        adapter_delta_scale: float = 0.1,
        adapter_gate_bias: float = -4.0,
        init_noise_std: float = 1.0,
        freeze_pmt: bool = True,
        pmt_only_mode: bool = False,
        require_height_scan: bool = True,
        require_teacher_motion_target: bool = False,
        use_motion_aux_decoder: bool = True,
        flat_identity_obs_key: str | None = None,
        flat_identity_threshold: float | None = None,
        pmt_ckpt_path: str | None = None,
        perceptive_motion_tracker_ckpt_path: str | None = None,
        percaptive_motion_tracker_ckpt_path: str | None = None,
        teacher_ckpt_path: str | None = None,
        require_pmt_checkpoint: bool = True,
        pmt_load_strict: bool = True,
        teacher_load_strict: bool | None = None,
        partial_pmt_load_min_match_fraction: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                "PerceptiveMotionTokenTracker is the final token-only scaffold and does not accept "
                f"v0 command-latent config keys: {unexpected}."
            )

        self.obs_groups = obs_groups
        self.num_actions = int(num_actions)
        self.policy_set_name = str(policy_set_name)
        self.history_set_name = str(history_set_name)
        self.future_motion_set_name = str(future_motion_set_name)
        self.teacher_future_motion_set_name = str(teacher_future_motion_set_name)
        self.height_scan_set_name = str(height_scan_set_name)
        self.critic_set_name = str(critic_set_name)
        self.future_motion_len = int(future_motion_len)
        self.num_motion_tokens = int(num_motion_tokens)
        self.motion_token_dim = int(motion_token_dim)
        self.model_dim = int(model_dim)
        self.token_num_heads = int(token_num_heads)
        self.history_embedding_dim = int(history_embedding_dim)
        self.terrain_context_dim = int(terrain_context_dim)
        self.foot_event_dim = int(foot_event_dim)
        self.num_height_tokens = int(num_height_tokens)
        self.foot_event_posterior_set_name = str(foot_event_posterior_set_name)
        self.use_foot_event_posterior = bool(use_foot_event_posterior)
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.adapter_hidden_dims = tuple(adapter_hidden_dims)
        self.activation = str(activation)
        self.adapter_mode = str(adapter_mode)
        self.adapter_delta_scale = float(adapter_delta_scale)
        self.adapter_gate_bias = float(adapter_gate_bias)
        self.freeze_pmt = bool(freeze_pmt)
        self.pmt_only_mode = bool(pmt_only_mode)
        self.require_height_scan = bool(require_height_scan)
        self.require_teacher_motion_target = bool(require_teacher_motion_target)
        self.flat_identity_obs_key = flat_identity_obs_key
        self.flat_identity_threshold = flat_identity_threshold
        self.require_pmt_checkpoint = bool(require_pmt_checkpoint)
        self.pmt_load_strict = bool(pmt_load_strict if teacher_load_strict is None else teacher_load_strict)
        self.partial_pmt_load_min_match_fraction = float(partial_pmt_load_min_match_fraction)
        if not 0.0 < self.partial_pmt_load_min_match_fraction <= 1.0:
            raise ValueError(
                "partial_pmt_load_min_match_fraction must be in (0, 1], got "
                f"{self.partial_pmt_load_min_match_fraction}."
            )
        if self.pmt_only_mode and self.freeze_pmt:
            raise ValueError("pmt_only_mode trains PMT with PPO and requires freeze_pmt=False.")
        if self.pmt_only_mode and self.require_pmt_checkpoint:
            raise ValueError(
                "pmt_only_mode starts PMT pretraining from the local policy weights and requires "
                "require_pmt_checkpoint=False."
            )
        pmt_checkpoint = next(
            (
                str(path)
                for path in (
                    pmt_ckpt_path,
                    perceptive_motion_tracker_ckpt_path,
                    percaptive_motion_tracker_ckpt_path,
                    teacher_ckpt_path,
                )
                if path is not None and str(path).strip()
            ),
            None,
        )
        self.pmt_ckpt_path = pmt_checkpoint
        self.teacher_ckpt_path = pmt_checkpoint
        self.perceptive_motion_tracker_ckpt_path = pmt_checkpoint
        self.percaptive_motion_tracker_ckpt_path = pmt_checkpoint
        self.pmt_checkpoint_loaded = False
        self.loaded_teacher = False
        self._last_aux_outputs: dict[str, torch.Tensor] = {}
        self._last_bridge_debug: dict[str, float] = {}

        for group_name in (self.policy_set_name, self.history_set_name, self.future_motion_set_name, self.critic_set_name):
            self._validate_obs_group(obs, group_name)
        if self.require_height_scan:
            self._validate_obs_group(obs, self.height_scan_set_name)
        if self.require_teacher_motion_target:
            self._validate_obs_group(obs, self.teacher_future_motion_set_name)

        policy_dim = self._obs_group_dim(obs, self.policy_set_name)
        history_dim = self._obs_group_dim(obs, self.history_set_name)
        future_motion_dim = self._obs_group_dim(obs, self.future_motion_set_name)
        height_scan_dim = self._obs_group_dim(obs, self.height_scan_set_name) if self._has_obs_group(self.height_scan_set_name) else 1
        critic_dim = self._obs_group_dim(obs, self.critic_set_name)

        self.history_encoder = _ProprioHistoryEncoder(history_dim, self.history_embedding_dim, self.activation)
        self.motion_tokenizer = _MotionTokenizer(
            future_motion_dim=future_motion_dim,
            future_motion_len=self.future_motion_len,
            model_dim=self.model_dim,
            token_dim=self.motion_token_dim,
            num_tokens=self.num_motion_tokens,
            activation=self.activation,
            num_heads=self.token_num_heads,
        )
        self.height_scan_encoder = _HeightScanEncoder(
            height_scan_dim=height_scan_dim,
            context_dim=self.terrain_context_dim,
            num_tokens=self.num_height_tokens,
            token_dim=self.motion_token_dim,
            activation=self.activation,
        )
        self.foot_event_encoder = _FootEventLatentEncoder(
            input_dim=policy_dim + self.history_embedding_dim + self.motion_token_dim,
            latent_dim=self.foot_event_dim,
            activation=self.activation,
        )
        self.foot_event_posterior_encoder: _FootEventLatentEncoder | None = None
        if self.use_foot_event_posterior and self._obs_has_group_tensors(obs, self.foot_event_posterior_set_name):
            self.foot_event_posterior_encoder = _FootEventLatentEncoder(
                input_dim=self._obs_group_dim(obs, self.foot_event_posterior_set_name),
                latent_dim=self.foot_event_dim,
                activation=self.activation,
                latent_key="e_post",
                contact_key="contact_logits_post",
                time_key="time_to_event_post",
            )
        self.terrain_motion_adapter = _TerrainMotionAdapter(
            token_dim=self.motion_token_dim,
            terrain_context_dim=self.terrain_context_dim,
            history_embedding_dim=self.history_embedding_dim,
            foot_event_dim=self.foot_event_dim,
            num_motion_tokens=self.num_motion_tokens,
            num_height_tokens=self.num_height_tokens,
            num_heads=self.token_num_heads,
            hidden_dims=self.adapter_hidden_dims,
            activation=self.activation,
            mode=self.adapter_mode,
            delta_scale=self.adapter_delta_scale,
            gate_bias=self.adapter_gate_bias,
        )
        self.pmt_decoder = _TokenConditionedPMTDecoder(
            current_proprio_dim=policy_dim,
            history_embedding_dim=self.history_embedding_dim,
            token_dim=self.motion_token_dim,
            num_actions=self.num_actions,
            hidden_dims=self.actor_hidden_dims,
            activation=self.activation,
            init_noise_std=init_noise_std,
        )
        self.motion_aux_decoder = (
            _MotionAuxDecoder(self.motion_token_dim, future_motion_dim, self.model_dim, self.activation)
            if use_motion_aux_decoder
            else None
        )
        self.pma_critic = MLP(
            input_dim=critic_dim,
            output_dim=1,
            hidden_dims=list(self.critic_hidden_dims),
            activation=self.activation,
        )

        if self.pmt_ckpt_path is not None:
            self._load_pmt_checkpoint(self.pmt_ckpt_path, strict=self.pmt_load_strict)
        elif self.require_pmt_checkpoint:
            raise ValueError(
                "PerceptiveMotionTokenTracker requires pmt_ckpt_path for non-smoke training. "
                "Set require_pmt_checkpoint=False only for synthetic API tests or no-teacher ablations."
            )
        else:
            self.loaded_teacher = True

        if self.freeze_pmt:
            self._freeze_pmt()
        if self.pmt_only_mode:
            self._freeze_pma(include_critic=False)

    @property
    def perceptive_motion_tracker(self) -> _TokenConditionedPMTDecoder:
        return self.pmt_decoder

    @property
    def percaptive_motion_tracker(self) -> _TokenConditionedPMTDecoder:
        return self.pmt_decoder

    @property
    def teacher(self) -> _TokenConditionedPMTDecoder:
        return self.pmt_decoder

    @property
    def student(self) -> _TerrainMotionAdapter:
        return self.terrain_motion_adapter

    def _pmt_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.history_encoder, self.motion_tokenizer, self.pmt_decoder]
        if self.motion_aux_decoder is not None:
            modules.append(self.motion_aux_decoder)
        return modules

    def _pma_modules(self, *, include_critic: bool = False) -> list[nn.Module]:
        modules: list[nn.Module] = [
            self.height_scan_encoder,
            self.foot_event_encoder,
            self.terrain_motion_adapter,
        ]
        if self.foot_event_posterior_encoder is not None:
            modules.append(self.foot_event_posterior_encoder)
        if include_critic:
            modules.append(self.pma_critic)
        return modules

    def _pmt_state_prefixes(self) -> tuple[str, ...]:
        prefixes = ["history_encoder.", "motion_tokenizer.", "pmt_decoder."]
        if self.motion_aux_decoder is not None:
            prefixes.append("motion_aux_decoder.")
        return tuple(prefixes)

    def _pma_state_prefixes(self) -> tuple[str, ...]:
        prefixes = [
            "height_scan_encoder.",
            "foot_event_encoder.",
            "terrain_motion_adapter.",
            "pma_critic.",
        ]
        if self.foot_event_posterior_encoder is not None:
            prefixes.append("foot_event_posterior_encoder.")
        return tuple(prefixes)

    def _clone_state_dict_by_prefixes(self, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def get_pmt_state_dict(self) -> dict[str, torch.Tensor]:
        return self._clone_state_dict_by_prefixes(self._pmt_state_prefixes())

    def get_adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return self._clone_state_dict_by_prefixes(self._pma_state_prefixes())

    def get_pma_state_dict(self) -> dict[str, torch.Tensor]:
        return self.get_adapter_state_dict()

    def _load_pmt_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool) -> None:
        pmt_prefixes = self._pmt_state_prefixes()
        target_state = {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith(pmt_prefixes)
        }
        candidate_prefixes = (
            "",
            "module.",
            "actor_critic.",
            "policy.",
            "student.",
            "teacher.",
            "pmt.",
            "perceptive_motion_token_tracker.",
            "perceptive_motion_tracker.",
            "module.actor_critic.",
            "module.policy.",
        )
        candidates: list[tuple[str, dict[str, torch.Tensor]]] = []
        for prefix in candidate_prefixes:
            candidate = (
                PerceptiveMotionAdapterTracker._strip_prefix_state(state_dict, prefix)
                if prefix
                else dict(state_dict)
            )
            if candidate:
                candidates.append((prefix, candidate))
        for module_prefix in pmt_prefixes:
            candidates.append((f"as_{module_prefix}", {f"{module_prefix}{key}": value for key, value in state_dict.items()}))

        best_match: tuple[str, dict[str, torch.Tensor], list[str], list[str]] | None = None
        for prefix, candidate in candidates:
            filtered = {
                key: value
                for key, value in candidate.items()
                if key in target_state and target_state[key].shape == value.shape
            }
            if not filtered:
                continue
            missing = [key for key in target_state if key not in filtered]
            shape_mismatch = [
                key
                for key, value in candidate.items()
                if key in target_state and target_state[key].shape != value.shape
            ]
            if strict and not missing:
                nn.Module.load_state_dict(self, filtered, strict=False)
                if prefix:
                    print(
                        "[PerceptiveMotionTokenTracker] Loaded PMT state "
                        f"with prefix '{prefix}'"
                    )
                return
            if best_match is None or len(filtered) > len(best_match[1]):
                best_match = (prefix, filtered, missing, shape_mismatch)

        if best_match is None:
            raise RuntimeError("Could not match checkpoint tensors to PMT token state_dict.")

        prefix, filtered, missing, shape_mismatch = best_match
        match_frac = len(filtered) / max(len(target_state), 1)
        if strict or match_frac < self.partial_pmt_load_min_match_fraction:
            missing_sample = ", ".join(missing[:8])
            mismatch_sample = ", ".join(shape_mismatch[:8])
            raise RuntimeError(
                "Strict PMT checkpoint load failed: "
                f"matched={len(filtered)}/{len(target_state)} ({match_frac:.1%}) using prefix '{prefix}'. "
                f"missing=[{missing_sample}] shape_mismatch=[{mismatch_sample}]"
            )

        nn.Module.load_state_dict(self, filtered, strict=False)
        print(
            "[PerceptiveMotionTokenTracker] Loaded partial PMT state: "
            f"matched={len(filtered)} missing_or_shape_mismatch={len(target_state) - len(filtered)} "
            f"using prefix '{prefix}'"
        )

    def _load_pmt_checkpoint(self, ckpt_path: str | Path, strict: bool = True) -> None:
        resolved_path = PerceptiveMotionAdapterTracker._resolve_checkpoint_path(ckpt_path)
        try:
            checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(resolved_path, map_location="cpu")
        state_dict = PerceptiveMotionAdapterTracker._select_checkpoint_state_dict(
            checkpoint,
            ("model_state_dict", "state_dict", "actor_critic", "policy"),
        )
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint format at {resolved_path}")
        self._load_pmt_state_dict(state_dict, strict=strict)
        self.pmt_checkpoint_loaded = True
        self.loaded_teacher = True
        self.pmt_ckpt_path = str(resolved_path)
        self.teacher_ckpt_path = str(resolved_path)
        self.perceptive_motion_tracker_ckpt_path = str(resolved_path)
        self.percaptive_motion_tracker_ckpt_path = str(resolved_path)
        print(f"[PerceptiveMotionTokenTracker] Loaded frozen PMT checkpoint from {resolved_path}")

    def _freeze_pmt(self) -> None:
        for module in self._pmt_modules():
            module.train(False)
            module.requires_grad_(False)

    def _freeze_pma(self, *, include_critic: bool = False) -> None:
        for module in self._pma_modules(include_critic=include_critic):
            module.train(False)
            module.requires_grad_(False)

    def _validate_obs_group(self, obs: TensorDict, set_name: str) -> None:
        if not self._has_obs_group(set_name):
            raise KeyError(f"Missing observation group '{set_name}'.")
        for key in self.obs_groups[set_name]:
            if key not in obs:
                raise KeyError(f"Observation group '{set_name}' references missing tensor '{key}'.")

    def _has_obs_group(self, set_name: str) -> bool:
        return set_name in self.obs_groups and len(self.obs_groups[set_name]) > 0

    def _obs_has_group_tensors(self, obs: TensorDict, set_name: str) -> bool:
        if not self._has_obs_group(set_name):
            return False
        return all(key in obs for key in self.obs_groups[set_name])

    def _obs_group_dim(self, obs: TensorDict, set_name: str) -> int:
        self._validate_obs_group(obs, set_name)
        return int(sum(obs[key].reshape(obs[key].shape[0], -1).shape[-1] for key in self.obs_groups[set_name]))

    def _get_concat_flat(self, obs: TensorDict, set_name: str) -> torch.Tensor:
        self._validate_obs_group(obs, set_name)
        tensors = [obs[key].reshape(obs[key].shape[0], -1) for key in self.obs_groups[set_name]]
        return torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]

    def _flat_identity_mask(self, obs: TensorDict) -> torch.Tensor | None:
        if self.flat_identity_obs_key is None or self.flat_identity_obs_key not in obs:
            return None
        signal = obs[self.flat_identity_obs_key].reshape(obs[self.flat_identity_obs_key].shape[0], -1)
        if self.flat_identity_threshold is None:
            return (signal > 0.5).any(dim=-1, keepdim=True)
        return (signal <= float(self.flat_identity_threshold)).all(dim=-1, keepdim=True)

    def _encode_frozen_foundation(
        self,
        history_obs: torch.Tensor,
        future_motion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freeze_pmt:
            with torch.no_grad():
                h_prop = self.history_encoder(history_obs)
                z_e, z_q = self.motion_tokenizer(future_motion)
        else:
            h_prop = self.history_encoder(history_obs)
            z_e, z_q = self.motion_tokenizer(future_motion)
        return h_prop, z_e, z_q

    def _encode_teacher_token(self, obs: TensorDict, z_flat_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._has_obs_group(self.teacher_future_motion_set_name):
            teacher_future_motion = self._get_concat_flat(obs, self.teacher_future_motion_set_name)
            if self.freeze_pmt:
                with torch.no_grad():
                    z_opt_e, z_opt_q = self.motion_tokenizer(teacher_future_motion)
            else:
                z_opt_e, z_opt_q = self.motion_tokenizer(teacher_future_motion)
            return z_opt_e, z_opt_q
        if self.require_teacher_motion_target:
            raise KeyError(
                f"PMA token training requires '{self.teacher_future_motion_set_name}'. "
                "Set require_teacher_motion_target=False only for identity/no-teacher ablations."
            )
        return z_flat_q, z_flat_q

    def _record_student_debug(
        self,
        outputs: dict[str, torch.Tensor],
        z_flat_q: torch.Tensor,
        z_task_q: torch.Tensor,
        z_opt_q: torch.Tensor,
        adapter_outputs: dict[str, torch.Tensor],
    ) -> None:
        self._last_aux_outputs = {
            key: value.detach() for key, value in outputs.items() if isinstance(value, torch.Tensor)
        }
        z_flat_debug = z_flat_q.detach().reshape(z_flat_q.shape[0], -1)
        z_task_debug = z_task_q.detach().reshape(z_task_q.shape[0], -1)
        z_opt_debug = z_opt_q.detach().reshape(z_opt_q.shape[0], -1)
        delta_z_debug = adapter_outputs["delta_z"].detach().reshape(z_task_q.shape[0], -1)
        gate_debug = adapter_outputs["gate"].detach().reshape(z_task_q.shape[0], -1)
        self._last_bridge_debug = {
            "bridge_z_flat_abs_mean": float(z_flat_debug.abs().mean().item()),
            "bridge_z_task_abs_mean": float(z_task_debug.abs().mean().item()),
            "bridge_z_opt_abs_mean": float(z_opt_debug.abs().mean().item()),
            "bridge_delta_z_abs_mean": float(delta_z_debug.abs().mean().item()),
            "bridge_gate_mean": float(gate_debug.mean().item()),
            "bridge_gate_abs_mean": float(gate_debug.abs().mean().item()),
            "bridge_latent_cosine": float(F.cosine_similarity(z_task_debug, z_opt_debug, dim=-1).mean().item()),
            "bridge_latent_norm_ratio": float(
                (z_task_debug.norm(dim=-1) / z_opt_debug.norm(dim=-1).clamp_min(1.0e-6)).mean().item()
            ),
            "bridge_pmt_only_mode": float(self.pmt_only_mode),
        }

    def _compute_student_outputs(
        self,
        obs: TensorDict,
        actions: torch.Tensor | None = None,
        *,
        include_teacher: bool = True,
    ) -> dict[str, torch.Tensor]:
        del actions
        policy_obs = self._get_concat_flat(obs, self.policy_set_name)
        history_obs = self._get_concat_flat(obs, self.history_set_name)
        future_motion = self._get_concat_flat(obs, self.future_motion_set_name)

        h_prop, z_flat_e, z_flat_q = self._encode_frozen_foundation(history_obs, future_motion)
        if include_teacher:
            z_opt_e, z_opt_q = self._encode_teacher_token(obs, z_flat_q)
        else:
            z_opt_e, z_opt_q = z_flat_e, z_flat_q

        if self.pmt_only_mode:
            z_task_q = z_flat_q
            zero_token = torch.zeros_like(z_flat_q)
            terrain_context = h_prop.new_zeros(h_prop.shape[0], self.terrain_context_dim)
            terrain_tokens = z_flat_q.new_zeros(z_flat_q.shape[0], self.num_height_tokens, self.motion_token_dim)
            e_prior = h_prop.new_zeros(h_prop.shape[0], self.foot_event_dim)
            zero_contact = e_prior.new_zeros(e_prior.shape[0], 2)
            zero_time = e_prior.new_zeros(e_prior.shape[0], 2)
            foot_outputs = {
                "e_prior": e_prior,
                "contact_logits_prior": zero_contact,
                "time_to_event_prior": zero_time,
                "e_foot": e_prior,
                "contact_logits": zero_contact,
                "time_to_event": zero_time,
            }
            adapter_outputs = {
                "z_task": z_task_q,
                "z_task_q": z_task_q,
                "delta_z": zero_token,
                "gate": zero_token,
                "identity_residual": zero_token,
            }
            self.pmt_decoder.update_distribution(policy_obs, h_prop, z_task_q)
            action_mean = self.pmt_decoder.action_mean
            critic_obs = self._get_concat_flat(obs, self.critic_set_name)
            outputs: dict[str, torch.Tensor] = {
                "action": action_mean,
                "actions": action_mean,
                "value": self.pma_critic(critic_obs),
                "z_flat": z_flat_q,
                "z_flat_embedding": z_flat_e,
                "z_flat_token": z_flat_q,
                "motion_token": z_flat_q,
                "z_opt": z_opt_q,
                "z_opt_embedding": z_opt_e,
                "z_opt_token": z_opt_q,
                "delta_z_target": (z_opt_q - z_flat_q).detach(),
                "h_prop": h_prop,
                "terrain_context": terrain_context,
                "terrain_tokens": terrain_tokens,
                **foot_outputs,
                **adapter_outputs,
            }
            if self.motion_aux_decoder is not None:
                outputs.update(self.motion_aux_decoder(z_task_q))
            flat_identity_mask = self._flat_identity_mask(obs)
            if flat_identity_mask is not None:
                outputs["flat_identity_mask"] = flat_identity_mask
            self._record_student_debug(outputs, z_flat_q, z_task_q, z_opt_q, adapter_outputs)
            return outputs

        if self._has_obs_group(self.height_scan_set_name):
            height_scan = self._get_concat_flat(obs, self.height_scan_set_name)
        elif self.require_height_scan:
            raise KeyError(f"Missing required height scan observation group '{self.height_scan_set_name}'.")
        else:
            height_scan = torch.zeros(policy_obs.shape[0], 1, device=policy_obs.device, dtype=policy_obs.dtype)

        terrain_context, terrain_tokens = self.height_scan_encoder(height_scan)
        foot_obs = torch.cat((policy_obs, h_prop, z_flat_q.mean(dim=1)), dim=-1)
        foot_outputs = self.foot_event_encoder(foot_obs)
        e_prior = foot_outputs["e_prior"]
        foot_outputs = {
            **foot_outputs,
            # Backward-compatible aliases for older loss hooks. The action path consumes e_prior.
            "e_foot": e_prior,
            "contact_logits": foot_outputs["contact_logits_prior"],
            "time_to_event": foot_outputs["time_to_event_prior"],
        }
        if self.foot_event_posterior_encoder is not None and self._obs_has_group_tensors(
            obs, self.foot_event_posterior_set_name
        ):
            posterior_obs = self._get_concat_flat(obs, self.foot_event_posterior_set_name)
            posterior_outputs = self.foot_event_posterior_encoder(posterior_obs)
            foot_outputs.update(posterior_outputs)
            foot_outputs["foot_event_prior_post_mse"] = (e_prior - posterior_outputs["e_post"].detach()).pow(2).mean(
                dim=-1,
                keepdim=True,
            )
        adapter_outputs = self.terrain_motion_adapter(
            z_flat=z_flat_q,
            terrain_context=terrain_context,
            terrain_tokens=terrain_tokens,
            h_prop=h_prop,
            e_prior=e_prior,
        )
        z_task_q = adapter_outputs["z_task_q"]

        self.pmt_decoder.update_distribution(policy_obs, h_prop, z_task_q)
        action_mean = self.pmt_decoder.action_mean

        outputs: dict[str, torch.Tensor] = {
            "action": action_mean,
            "actions": action_mean,
            # Distillation's default latent loss compares z_task and z_opt, so these
            # aliases must live in the decoder token space rather than pre-token embeddings.
            "z_flat": z_flat_q,
            "z_flat_embedding": z_flat_e,
            "z_flat_token": z_flat_q,
            "motion_token": z_flat_q,
            "z_opt": z_opt_q,
            "z_opt_embedding": z_opt_e,
            "z_opt_token": z_opt_q,
            "delta_z_target": (z_opt_q - z_flat_q).detach(),
            "h_prop": h_prop,
            "terrain_context": terrain_context,
            "terrain_tokens": terrain_tokens,
            **foot_outputs,
            **adapter_outputs,
        }
        if self.motion_aux_decoder is not None:
            outputs.update(self.motion_aux_decoder(z_task_q))
        flat_identity_mask = self._flat_identity_mask(obs)
        if flat_identity_mask is not None:
            outputs["flat_identity_mask"] = flat_identity_mask

        self._record_student_debug(outputs, z_flat_q, z_task_q, z_opt_q, adapter_outputs)
        return outputs

    def infer_student_outputs(self, obs: TensorDict, actions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self._compute_student_outputs(obs, actions=actions, include_teacher=True)

    def update_distribution(self, obs: TensorDict) -> None:
        self._compute_student_outputs(obs, include_teacher=False)

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        del kwargs
        self._compute_student_outputs(obs, include_teacher=False)
        return self.pmt_decoder.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self._compute_student_outputs(obs, include_teacher=False)["action"]

    @torch.no_grad()
    def infer_teacher_action(self, obs: TensorDict) -> torch.Tensor:
        """Return the frozen PMT teacher action for distillation."""
        policy_obs = self._get_concat_flat(obs, self.policy_set_name)
        history_obs = self._get_concat_flat(obs, self.history_set_name)
        h_prop = self.history_encoder(history_obs)

        if self._has_obs_group(self.teacher_future_motion_set_name):
            teacher_future_motion = self._get_concat_flat(obs, self.teacher_future_motion_set_name)
            _, z_teacher_q = self.motion_tokenizer(teacher_future_motion)
        elif self.require_teacher_motion_target:
            raise KeyError(
                f"PMA token distillation requires '{self.teacher_future_motion_set_name}' "
                "for teacher action inference. Set require_teacher_motion_target=False only for no-teacher ablations."
            )
        else:
            future_motion = self._get_concat_flat(obs, self.future_motion_set_name)
            _, z_teacher_q = self.motion_tokenizer(future_motion)

        self.pmt_decoder.update_distribution(policy_obs, h_prop, z_teacher_q)
        return self.pmt_decoder.action_mean

    def act_inference_from_token(
        self,
        obs: TensorDict,
        motion_token: torch.Tensor,
        h_prop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        policy_obs = self._get_concat_flat(obs, self.policy_set_name)
        if h_prop is None:
            history_obs = self._get_concat_flat(obs, self.history_set_name)
            if self.freeze_pmt:
                with torch.no_grad():
                    h_prop = self.history_encoder(history_obs)
            else:
                h_prop = self.history_encoder(history_obs)
        self.pmt_decoder.update_distribution(policy_obs, h_prop, motion_token)
        return self.pmt_decoder.action_mean

    def act_from_token(
        self,
        obs: TensorDict,
        motion_token: torch.Tensor,
        h_prop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.act_inference_from_token(obs, motion_token, h_prop=h_prop)

    def evaluate(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        del kwargs
        critic_obs = self._get_concat_flat(obs, self.critic_set_name)
        return self.pma_critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.pmt_decoder.get_actions_log_prob(actions)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.pmt_decoder.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.pmt_decoder.action_std

    @property
    def entropy(self) -> torch.Tensor:
        return self.pmt_decoder.entropy

    @property
    def distribution(self) -> Normal:
        return self.pmt_decoder.distribution

    def reset(self, dones: torch.Tensor | None = None, hidden_states: tuple | None = None) -> None:
        del dones, hidden_states

    def get_hidden_states(self) -> tuple[None, None]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        del dones
        pass

    def reset_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        self.reset(dones=dones)

    def update_normalization(self, obs: TensorDict) -> None:
        del obs

    def get_last_aux_outputs(self, clear: bool = True) -> dict[str, torch.Tensor]:
        outputs = dict(self._last_aux_outputs)
        if clear:
            self._last_aux_outputs.clear()
        return outputs

    def get_last_bridge_debug(self, clear: bool = True) -> dict[str, float]:
        debug = dict(self._last_bridge_debug)
        if clear:
            self._last_bridge_debug.clear()
        return debug

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "policy_class": self.__class__.__name__,
            "policy_family": "perceptive_motion_token_tracker",
            "architecture": "PerceptiveMotionTokenTracker",
            "signature": {
                "policy_set_name": self.policy_set_name,
                "history_set_name": self.history_set_name,
                "future_motion_set_name": self.future_motion_set_name,
                "teacher_future_motion_set_name": self.teacher_future_motion_set_name,
                "height_scan_set_name": self.height_scan_set_name,
                "critic_set_name": self.critic_set_name,
                "future_motion_len": self.future_motion_len,
                "num_motion_tokens": self.num_motion_tokens,
                "motion_token_dim": self.motion_token_dim,
                "model_dim": self.model_dim,
                "token_num_heads": self.token_num_heads,
                "history_embedding_dim": self.history_embedding_dim,
                "terrain_context_dim": self.terrain_context_dim,
                "foot_event_dim": self.foot_event_dim,
                "foot_event_posterior_set_name": self.foot_event_posterior_set_name,
                "use_foot_event_posterior": self.use_foot_event_posterior,
                "num_height_tokens": self.num_height_tokens,
                "adapter_mode": self.adapter_mode,
                "adapter_delta_scale": self.adapter_delta_scale,
                "adapter_gate_bias": self.adapter_gate_bias,
                "freeze_pmt": self.freeze_pmt,
                "pmt_only_mode": self.pmt_only_mode,
                "require_height_scan": self.require_height_scan,
                "require_teacher_motion_target": self.require_teacher_motion_target,
                "flat_identity_obs_key": self.flat_identity_obs_key,
                "flat_identity_threshold": self.flat_identity_threshold,
                "require_pmt_checkpoint": self.require_pmt_checkpoint,
                "pmt_load_strict": self.pmt_load_strict,
                "partial_pmt_load_min_match_fraction": self.partial_pmt_load_min_match_fraction,
                "pmt_checkpoint_loaded": self.pmt_checkpoint_loaded,
                "pmt_ckpt_path": self.pmt_ckpt_path,
                "teacher_ckpt_path": self.teacher_ckpt_path,
            },
            "deployed_actor_contract": (
                "C_flat/future q-qdot is tokenizer input only; height scan and deployable e_prior are PMA inputs only; "
                "e_post/contact labels/future-foot labels are supervision-only; "
                "TokenConditionedPMTDecoder consumes current proprio, proprio-history embedding, and adapted motion tokens."
            ),
            "terrain_adapter_contract": (
                "Height scan is encoded into height tokens; PMA cross-attends motion-token queries "
                "to height tokens before the gated residual token rewrite."
            ),
            "decoder_contract": "TokenConditionedPMTDecoder.forward(current_proprio, h_prop, motion_tokens)",
            "training_only_signals": [
                "teacher_future_motion_window",
                "z_terrain_teacher",
                "foot_event_posterior",
                "e_post",
                "contact_gt",
                "time_to_touchdown_labels",
                "terrain_ground_truth",
                "teacher_action",
                "flat_mask",
            ],
        }

    def train(self, mode: bool = True) -> "PerceptiveMotionTokenTracker":
        super().train(mode)
        if self.freeze_pmt:
            self._freeze_pmt()
        if self.pmt_only_mode:
            self._freeze_pma(include_critic=False)
        return self

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        result = nn.Module.load_state_dict(self, state_dict, strict=strict)
        if self.freeze_pmt:
            self._freeze_pmt()
        if self.pmt_only_mode:
            self._freeze_pma(include_critic=False)
        return result
