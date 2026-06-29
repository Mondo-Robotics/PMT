# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.networks.layers import EmpiricalNormalization, MLP
from motion_tracking_rl.registry import register_network
from motion_tracking_rl.utils import build_obs_schema


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, activation: str = "elu") -> None:
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.net = MLP(in_dim, out_dim, [hidden_dim], activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.shape[1]
        return x + self.pe[:t].to(dtype=x.dtype).unsqueeze(0)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.ReLU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x_ln = self.ln_attn(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln_ff(x))
        return self.out_ln(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.ReLU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [B, 1, D], kv: [B, S, D]
        q_ln = self.ln_q(q)
        attn_out, _ = self.attn(q_ln, kv, kv, need_weights=False)
        x = q + attn_out
        x = x + self.ff(self.ln_ff(x))
        return self.out_ln(x)


@register_network("TransformerActorCritic", compat_name="transformer")
class TransformerActorCritic(nn.Module):
    """transformer policy: causal history encoder + dynamics-conditioned cross-attn command encoder."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Normalization
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        history_obs_normalization: bool = False,
        command_obs_normalization: bool = False,
        # Actor/Critic heads
        actor_hidden_dims: tuple[int] | list[int] = (512, 256),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256),
        activation: str = "elu",
        # Velocity estimator (optional)
        use_vel_estimator: bool = False,
        vel_estimator_detach: bool = True,
        vel_estimator_hidden_dims: tuple[int, ...] | None = None,
        vel_estimator_output_dim: int = 3,
        vel_gt_normalization: bool = False,
        # Anchor-position estimator (optional)
        use_anchor_estimator: bool = False,
        anchor_estimator_detach: bool = True,
        anchor_estimator_hidden_dims: tuple[int, ...] | None = None,
        anchor_estimator_output_dim: int = 3,
        anchor_gt_normalization: bool = False,
        anchor_estimator_latent_inputs: tuple[str, ...] = ("h_last", "u_t"),
        # Action distribution
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
        min_std: float = 1e-6,
        validate_args: bool = True,
        # Transformer hyper-parameters
        n_embd: int = 128,
        n_heads: int = 4,
        history_len: int = 10,
        cmd_len: int = 21,
        mlp_ratio: int = 4,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "TransformerActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.n_embd = int(n_embd)
        self.history_len = history_len
        self.cmd_len = cmd_len
        self.noise_std_type = noise_std_type
        self.state_dependent_std = state_dependent_std
        self.log_std_bounds = log_std_bounds
        self.min_std = min_std
        self.validate_args = validate_args
        self._max_std = float(math.exp(log_std_bounds[1]))

        # ---------------------------------------------------------------------
        # Resolve observation dimensions
        # ---------------------------------------------------------------------
        num_actor_obs = 0
        for group in obs_groups.get("policy", []):
            assert len(obs[group].shape) == 2, "TransformerActorCritic expects 1D tensors in obs_groups['policy']."
            num_actor_obs += obs[group].shape[-1]

        num_critic_obs = 0
        for group in obs_groups.get("critic", []):
            assert len(obs[group].shape) == 2, "TransformerActorCritic expects 1D tensors in obs_groups['critic']."
            num_critic_obs += obs[group].shape[-1]

        history_token_dim = self._infer_seq_feature_dim(obs, obs_groups.get("policy_history", []), history_len)
        cmd_token_dim = self._infer_seq_feature_dim(obs, obs_groups.get("command_window", []), cmd_len)
        anchor_estimator_obs_dim = self._infer_2d_feature_dim(obs, obs_groups.get("anchor_estimator", []))
        self.actor_obs_dim = int(num_actor_obs)
        self.critic_obs_dim = int(num_critic_obs)
        self.history_token_dim = int(history_token_dim)
        self.cmd_token_dim = int(cmd_token_dim)
        self.anchor_estimator_obs_dim = int(anchor_estimator_obs_dim)
        self.num_actions = int(num_actions)
        self._obs_schema = build_obs_schema(obs, obs_groups)

        # ---------------------------------------------------------------------
        # Optional velocity estimator head (supervised via separate vel_gt obs group)
        # ---------------------------------------------------------------------
        self.use_vel_estimator = use_vel_estimator
        self.vel_estimator_detach = vel_estimator_detach
        self.vel_output_dim = vel_estimator_output_dim
        self.vel_gt_normalization = vel_gt_normalization

        if self.use_vel_estimator:
            hidden = list(vel_estimator_hidden_dims) if vel_estimator_hidden_dims is not None else [64]
            self.vel_head = MLP(n_embd, vel_estimator_output_dim, hidden, activation)
            self.vel_gt_normalizer = (
                EmpiricalNormalization(vel_estimator_output_dim) if vel_gt_normalization else nn.Identity()
            )
        else:
            self.vel_head = None
            self.vel_gt_normalizer = nn.Identity()

        # ---------------------------------------------------------------------
        # Optional anchor-position estimator head (supervised via separate anchor_gt obs group)
        # ---------------------------------------------------------------------
        self.use_anchor_estimator = use_anchor_estimator
        self.anchor_estimator_detach = anchor_estimator_detach
        self.anchor_output_dim = anchor_estimator_output_dim
        self.anchor_gt_normalization = anchor_gt_normalization
        self.anchor_estimator_latent_inputs = tuple(anchor_estimator_latent_inputs)
        invalid_anchor_inputs = sorted(set(self.anchor_estimator_latent_inputs) - {"h_last", "u_t"})
        if invalid_anchor_inputs:
            raise ValueError(
                "anchor_estimator_latent_inputs contains unsupported values: "
                f"{invalid_anchor_inputs}. Supported values are ('h_last', 'u_t')."
            )
        anchor_estimator_in_dim = len(self.anchor_estimator_latent_inputs) * n_embd + anchor_estimator_obs_dim

        if self.use_anchor_estimator:
            if anchor_estimator_in_dim <= 0:
                raise ValueError(
                    "Anchor estimator is enabled but no input sources were configured. "
                    "Set anchor_estimator_latent_inputs and/or obs_groups['anchor_estimator']."
                )
            hidden = list(anchor_estimator_hidden_dims) if anchor_estimator_hidden_dims is not None else [64]
            self.anchor_head = MLP(anchor_estimator_in_dim, anchor_estimator_output_dim, hidden, activation)
            self.anchor_gt_normalizer = (
                EmpiricalNormalization(anchor_estimator_output_dim) if anchor_gt_normalization else nn.Identity()
            )
        else:
            self.anchor_head = None
            self.anchor_gt_normalizer = nn.Identity()

        self._last_aux_outputs: dict[str, torch.Tensor] | None = None

        # ---------------------------------------------------------------------
        # Encoders (paper-aligned)
        # ---------------------------------------------------------------------
        self.history_embed = TwoLayerMLP(history_token_dim, n_embd, activation=activation)
        self.history_pos = SinusoidalPositionalEncoding(n_embd, max_len=max(64, history_len))
        self.history_block = SelfAttentionBlock(n_embd, n_heads, mlp_ratio=mlp_ratio)
        mask = torch.triu(torch.ones(history_len, history_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        self.q_mlp = TwoLayerMLP(n_embd, n_embd, activation=activation)
        self.cmd_embed = TwoLayerMLP(cmd_token_dim, n_embd, activation=activation)
        self.cmd_pos = SinusoidalPositionalEncoding(n_embd, max_len=max(64, cmd_len))
        self.cmd_block = CrossAttentionBlock(n_embd, n_heads, mlp_ratio=mlp_ratio)

        # ---------------------------------------------------------------------
        # Actor/Critic heads
        # ---------------------------------------------------------------------
        actor_in_dim = (
            num_actor_obs
            + n_embd
            + (vel_estimator_output_dim if self.use_vel_estimator else 0)
            + (anchor_estimator_output_dim if self.use_anchor_estimator else 0)
        )
        trunk_out_dim = int(actor_hidden_dims[-1]) if len(actor_hidden_dims) > 0 else 256
        trunk_hidden_dims = list(actor_hidden_dims[:-1]) if len(actor_hidden_dims) > 1 else [trunk_out_dim]
        self.actor_trunk = MLP(actor_in_dim, trunk_out_dim, trunk_hidden_dims, activation)
        self.mean_head = nn.Linear(trunk_out_dim, num_actions)

        def inv_softplus(x: float) -> float:
            # stable inverse of softplus for x > 0
            x_t = torch.tensor(float(x))
            return float(torch.log(torch.expm1(x_t)))

        if state_dependent_std:
            if self.noise_std_type == "log":
                self.log_std_head = nn.Linear(trunk_out_dim, num_actions)
                nn.init.zeros_(self.log_std_head.weight)
                nn.init.constant_(self.log_std_head.bias, math.log(init_noise_std + 1e-7))
                self.std_head = None
            elif self.noise_std_type == "scalar":
                self.std_head = nn.Linear(trunk_out_dim, num_actions)
                nn.init.zeros_(self.std_head.weight)
                nn.init.constant_(self.std_head.bias, inv_softplus(init_noise_std))
                self.log_std_head = None
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            self.log_std_param = None
            self.raw_std_param = None
        else:
            if self.noise_std_type == "log":
                self.log_std_param = nn.Parameter(torch.ones(num_actions) * math.log(init_noise_std + 1e-7))
                self.raw_std_param = None
            elif self.noise_std_type == "scalar":
                self.raw_std_param = nn.Parameter(torch.ones(num_actions) * inv_softplus(init_noise_std))
                self.log_std_param = None
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            self.log_std_head = None
            self.std_head = None

        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)

        # ---------------------------------------------------------------------
        # Normalizers
        # ---------------------------------------------------------------------
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        self.history_obs_normalization = history_obs_normalization
        self.history_obs_normalizer = (
            EmpiricalNormalization(history_token_dim) if history_obs_normalization else nn.Identity()
        )

        self.command_obs_normalization = command_obs_normalization
        self.command_obs_normalizer = (
            EmpiricalNormalization(cmd_token_dim) if command_obs_normalization else nn.Identity()
        )

        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        # Action distribution
        self.distribution: Normal | None = None

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # -------------------------------------------------------------------------
    # Observation helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _infer_seq_feature_dim(obs: TensorDict, groups: list[str], seq_len: int) -> int:
        if not groups:
            raise ValueError("Missing required obs group set for sequence input (e.g., policy_history/command_window).")

        feature_dim = 0
        for group in groups:
            x = obs[group]
            if x.ndim == 3:
                if x.shape[1] != seq_len:
                    raise ValueError(
                        f"Obs '{group}' has seq_len {x.shape[1]} but expected {seq_len}. Shape: {tuple(x.shape)}"
                    )
                feature_dim += x.shape[-1]
            elif x.ndim == 2:
                if x.shape[-1] % seq_len != 0:
                    raise ValueError(
                        f"Obs '{group}' last dim {x.shape[-1]} is not divisible by seq_len={seq_len}. "
                        f"Shape: {tuple(x.shape)}"
                    )
                feature_dim += x.shape[-1] // seq_len
            else:
                raise ValueError(f"Obs '{group}' must be 2D or 3D. Got shape: {tuple(x.shape)}")
        return feature_dim

    @staticmethod
    def _infer_2d_feature_dim(obs: TensorDict, groups: list[str]) -> int:
        feature_dim = 0
        for group in groups:
            x = obs[group]
            if x.ndim != 2:
                raise ValueError(f"Obs '{group}' must be 2D for anchor-estimator inputs. Got shape: {tuple(x.shape)}")
            feature_dim += x.shape[-1]
        return feature_dim

    def _get_concat_2d(self, obs: TensorDict, set_name: str) -> torch.Tensor:
        obs_list = [obs[k] for k in self.obs_groups.get(set_name, [])]
        if not obs_list:
            raise KeyError(f"obs_groups['{set_name}'] is empty or missing.")
        return torch.cat(obs_list, dim=-1)

    def _get_optional_concat_2d(self, obs: TensorDict, set_name: str) -> torch.Tensor | None:
        groups = self.obs_groups.get(set_name, [])
        if not groups:
            return None
        return self._get_concat_2d(obs, set_name)

    def _get_concat_seq(self, obs: TensorDict, set_name: str, seq_len: int) -> torch.Tensor:
        xs = []
        for k in self.obs_groups.get(set_name, []):
            x = obs[k]
            if x.ndim == 2:
                x = x.reshape(x.shape[0], seq_len, -1)
            elif x.ndim != 3:
                raise ValueError(f"Obs '{k}' must be 2D or 3D. Got shape: {tuple(x.shape)}")
            elif x.shape[1] != seq_len:
                raise ValueError(f"Obs '{k}' has seq_len {x.shape[1]} but expected {seq_len}. Shape: {tuple(x.shape)}")
            xs.append(x)
        if not xs:
            raise KeyError(f"obs_groups['{set_name}'] is empty or missing.")
        # Concatenate per-token features on the last dim.
        return torch.cat(xs, dim=-1)

    # -------------------------------------------------------------------------
    # Core policy methods (PPO interface)
    # -------------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def _encode_history_tokens(self, history_tokens: torch.Tensor) -> torch.Tensor:
        """Encode history tokens into contextual embeddings.

        Args:
            history_tokens: [B, T, Dh]

        Returns:
            tokens: [B, T, D]
        """
        x = self.history_embed(history_tokens)
        x = self.history_pos(x)
        t = x.shape[1]
        mask = self.causal_mask[:t, :t].to(device=x.device)
        return self.history_block(x, attn_mask=mask)

    def _encode_history(self, history_tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_history_tokens(history_tokens)
        # Element-wise max pooling across time (paper).
        return tokens.max(dim=1).values

    def _encode_command(self, h_t: torch.Tensor, cmd_tokens: torch.Tensor) -> torch.Tensor:
        # cmd_tokens: [B, S, Dc]
        q = self.q_mlp(h_t).unsqueeze(1)  # [B, 1, D]
        z = self.cmd_embed(cmd_tokens)
        z = self.cmd_pos(z)
        u = self.cmd_block(q, z)  # [B, 1, D]
        return u.squeeze(1)

    @staticmethod
    def _normalize_seq_tokens(tokens: torch.Tensor, normalizer: nn.Module) -> torch.Tensor:
        if isinstance(normalizer, nn.Identity):
            return tokens
        flat = tokens.reshape(-1, tokens.shape[-1])
        normed = normalizer(flat)
        return normed.reshape(tokens.shape[0], tokens.shape[1], tokens.shape[2])

    def encode_command_latent_from_tokens(
        self,
        history_tokens: torch.Tensor,
        cmd_tokens: torch.Tensor,
        *,
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tokens = self._normalize_seq_tokens(history_tokens, self.history_obs_normalizer)
        cmd_tokens = self._normalize_seq_tokens(cmd_tokens, self.command_obs_normalizer)

        hist_encoded = self._encode_history_tokens(history_tokens)
        h_pool = hist_encoded.max(dim=1).values
        h_last = hist_encoded[:, -1]
        u_t = self._encode_command(h_pool, cmd_tokens)
        if return_history:
            return u_t, h_pool, h_last
        return u_t

    def encode_command_latent(
        self,
        obs: TensorDict,
        *,
        history_set_name: str = "policy_history",
        command_set_name: str = "command_window",
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tokens = self._get_concat_seq(obs, history_set_name, self.history_len)
        cmd_tokens = self._get_concat_seq(obs, command_set_name, self.cmd_len)
        return self.encode_command_latent_from_tokens(
            history_tokens,
            cmd_tokens,
            return_history=return_history,
        )

    def encode_proprio_history(
        self,
        obs: TensorDict,
        *,
        history_set_name: str = "policy_history",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_tokens = self._get_concat_seq(obs, history_set_name, self.history_len)
        history_tokens = self._normalize_seq_tokens(history_tokens, self.history_obs_normalizer)
        hist_encoded = self._encode_history_tokens(history_tokens)
        return hist_encoded.max(dim=1).values, hist_encoded[:, -1]

    def encode_motion_token_from_tokens(
        self,
        history_tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
        *,
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode_command_latent_from_tokens(
            history_tokens,
            motion_tokens,
            return_history=return_history,
        )

    def encode_motion_token(
        self,
        obs: TensorDict,
        *,
        history_set_name: str = "policy_history",
        motion_set_name: str = "command_window",
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tokens = self._get_concat_seq(obs, history_set_name, self.history_len)
        motion_tokens = self._get_concat_seq(obs, motion_set_name, self.cmd_len)
        return self.encode_motion_token_from_tokens(
            history_tokens,
            motion_tokens,
            return_history=return_history,
        )

    def _build_anchor_estimator_input(
        self,
        obs: TensorDict,
        *,
        h_last: torch.Tensor,
        u_t: torch.Tensor,
        anchor_estimator_set_name: str = "anchor_estimator",
    ) -> torch.Tensor:
        inputs = []
        for name in self.anchor_estimator_latent_inputs:
            if name == "h_last":
                inputs.append(h_last)
            elif name == "u_t":
                inputs.append(u_t)
            else:
                raise RuntimeError(f"Unsupported anchor estimator latent input '{name}'.")

        anchor_obs = self._get_optional_concat_2d(obs, anchor_estimator_set_name)
        if anchor_obs is not None:
            inputs.append(anchor_obs)

        if not inputs:
            raise RuntimeError("Anchor estimator input is empty.")
        return torch.cat(inputs, dim=-1)

    def _set_distribution_from_actor_inputs(self, actor_in: torch.Tensor) -> None:
        trunk = self.actor_trunk(actor_in)
        mean = self.mean_head(trunk)

        if self.state_dependent_std:
            if self.noise_std_type == "log":
                log_std = torch.clamp(self.log_std_head(trunk), self.log_std_bounds[0], self.log_std_bounds[1])
                std = torch.exp(log_std)
            elif self.noise_std_type == "scalar":
                std = F.softplus(self.std_head(trunk)) + self.min_std
                std = torch.clamp(std, self.min_std, self._max_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "log":
                log_std = torch.clamp(self.log_std_param, self.log_std_bounds[0], self.log_std_bounds[1])
                std = torch.exp(log_std).expand_as(mean)
            elif self.noise_std_type == "scalar":
                std = (F.softplus(self.raw_std_param) + self.min_std).expand_as(mean)
                std = torch.clamp(std, self.min_std, self._max_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = Normal(mean, std, validate_args=self.validate_args)

    def _update_distribution_from_external_token(
        self,
        obs: TensorDict,
        motion_token: torch.Tensor,
        *,
        policy_set_name: str = "policy",
        history_set_name: str = "policy_history",
        anchor_estimator_set_name: str = "anchor_estimator",
        h_last: torch.Tensor | None = None,
        token_name: str = "motion_token",
    ) -> None:
        if motion_token.ndim != 2 or motion_token.shape[-1] != self.n_embd:
            raise ValueError(
                f"{token_name} must have shape [B, {self.n_embd}], got {tuple(motion_token.shape)}."
            )

        policy_obs = self._get_concat_2d(obs, policy_set_name)
        policy_obs = self.actor_obs_normalizer(policy_obs)

        if h_last is None:
            _, h_last = self.encode_proprio_history(obs, history_set_name=history_set_name)

        aux_outputs: dict[str, torch.Tensor] = {}
        actor_inputs = [policy_obs, motion_token]

        if self.use_vel_estimator:
            v_hat = self.vel_head(h_last)
            aux_outputs["v_hat"] = v_hat
            v_for_actor = v_hat.detach() if self.vel_estimator_detach else v_hat
            actor_inputs.append(v_for_actor)

        if self.use_anchor_estimator:
            anchor_inputs = self._build_anchor_estimator_input(
                obs,
                h_last=h_last,
                u_t=motion_token,
                anchor_estimator_set_name=anchor_estimator_set_name,
            )
            anchor_hat = self.anchor_head(anchor_inputs)
            aux_outputs["anchor_hat"] = anchor_hat
            anchor_for_actor = anchor_hat.detach() if self.anchor_estimator_detach else anchor_hat
            actor_inputs.append(anchor_for_actor)

        self._last_aux_outputs = aux_outputs if aux_outputs else None
        self._set_distribution_from_actor_inputs(torch.cat(actor_inputs, dim=-1))

    def update_distribution_from_latent(
        self,
        obs: TensorDict,
        command_latent: torch.Tensor,
        *,
        policy_set_name: str = "policy",
        history_set_name: str = "policy_history",
        command_set_name: str = "command_window",
        anchor_estimator_set_name: str = "anchor_estimator",
        h_last: torch.Tensor | None = None,
    ) -> None:
        if h_last is None:
            _, _, h_last = self.encode_command_latent(
                obs,
                history_set_name=history_set_name,
                command_set_name=command_set_name,
                return_history=True,
            )
        self._update_distribution_from_external_token(
            obs,
            command_latent,
            policy_set_name=policy_set_name,
            history_set_name=history_set_name,
            anchor_estimator_set_name=anchor_estimator_set_name,
            h_last=h_last,
            token_name="command_latent",
        )

    def update_distribution_from_token(
        self,
        obs: TensorDict,
        motion_token: torch.Tensor,
        *,
        policy_set_name: str = "policy",
        history_set_name: str = "policy_history",
        anchor_estimator_set_name: str = "anchor_estimator",
        h_last: torch.Tensor | None = None,
    ) -> None:
        self._update_distribution_from_external_token(
            obs,
            motion_token,
            policy_set_name=policy_set_name,
            history_set_name=history_set_name,
            anchor_estimator_set_name=anchor_estimator_set_name,
            h_last=h_last,
            token_name="motion_token",
        )

    def _update_distribution(self, obs: TensorDict) -> None:
        u_t, _, h_last = self.encode_command_latent(obs, return_history=True)
        self.update_distribution_from_latent(obs, u_t, h_last=h_last)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.mean

    def act_inference_from_latent(self, obs: TensorDict, command_latent: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        self.update_distribution_from_latent(obs, command_latent, **kwargs)
        return self.distribution.mean

    def act_from_token(self, obs: TensorDict, motion_token: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        self.update_distribution_from_token(obs, motion_token, **kwargs)
        return self.distribution.sample()

    def act_inference_from_token(self, obs: TensorDict, motion_token: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        self.update_distribution_from_token(obs, motion_token, **kwargs)
        return self.distribution.mean

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self._get_concat_2d(obs, "critic")
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_last_aux_outputs(self, *, clear: bool = True) -> dict[str, torch.Tensor]:
        """Return auxiliary outputs from the most recent `act()`/`act_inference()` call.

        This is designed for PPO-style aux losses without changing the `act()` signature.
        """
        aux = self._last_aux_outputs if self._last_aux_outputs is not None else {}
        if clear:
            self._last_aux_outputs = None
        return aux

    def normalize_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize velocities using the vel GT running stats (if enabled)."""
        if not self.vel_gt_normalization:
            return v
        return self.vel_gt_normalizer(v)

    def normalize_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        """Normalize anchor positions using the anchor GT running stats (if enabled)."""
        if not self.anchor_gt_normalization:
            return anchor
        return self.anchor_gt_normalizer(anchor)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "policy_class": self.__class__.__name__,
            "policy_family": "transformer_actor_critic",
            "obs_schema": self._obs_schema,
            "signature": {
                "num_actions": self.num_actions,
                "actor_obs_dim": self.actor_obs_dim,
                "critic_obs_dim": self.critic_obs_dim,
                "history_len": self.history_len,
                "cmd_len": self.cmd_len,
                "history_token_dim": self.history_token_dim,
                "cmd_token_dim": self.cmd_token_dim,
                "use_vel_estimator": bool(self.use_vel_estimator),
                "vel_output_dim": int(self.vel_output_dim),
                "use_anchor_estimator": bool(self.use_anchor_estimator),
                "anchor_output_dim": int(self.anchor_output_dim),
                "anchor_estimator_obs_dim": int(self.anchor_estimator_obs_dim),
            },
        }

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            policy_obs = self._get_concat_2d(obs, "policy")
            self.actor_obs_normalizer.update(policy_obs)
        if self.history_obs_normalization:
            history_tokens = self._get_concat_seq(obs, "policy_history", self.history_len)
            self.history_obs_normalizer.update(history_tokens.reshape(-1, history_tokens.shape[-1]))
        if self.command_obs_normalization:
            cmd_tokens = self._get_concat_seq(obs, "command_window", self.cmd_len)
            self.command_obs_normalizer.update(cmd_tokens.reshape(-1, cmd_tokens.shape[-1]))
        if self.use_vel_estimator and self.vel_gt_normalization and "vel_gt" in self.obs_groups:
            vel_gt = self._get_concat_2d(obs, "vel_gt")
            if vel_gt.shape[-1] != self.vel_output_dim:
                raise ValueError(
                    f"vel_gt dim mismatch: expected {self.vel_output_dim}, got {vel_gt.shape[-1]}. "
                    f"Check obs_groups['vel_gt'] and vel_estimator_output_dim."
                )
            self.vel_gt_normalizer.update(vel_gt)
        if self.use_anchor_estimator and self.anchor_gt_normalization and "anchor_gt" in self.obs_groups:
            anchor_gt = self._get_concat_2d(obs, "anchor_gt")
            if anchor_gt.shape[-1] != self.anchor_output_dim:
                raise ValueError(
                    f"anchor_gt dim mismatch: expected {self.anchor_output_dim}, got {anchor_gt.shape[-1]}. "
                    f"Check obs_groups['anchor_gt'] and anchor_estimator_output_dim."
                )
            self.anchor_gt_normalizer.update(anchor_gt)
        if self.critic_obs_normalization:
            critic_obs = self._get_concat_2d(obs, "critic")
            self.critic_obs_normalizer.update(critic_obs)
