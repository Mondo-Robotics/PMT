# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-backbone SONIC with a vision residual action head.

Action structure:
    a = a_sonic + tanh(delta_a_vision) * residual_scale

The base SONIC branch is inherited from ``SonicActorCritic`` and can be kept
frozen so training only updates the vision residual path (and critic).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.networks.actor_critic import SonicActorCritic


class MapCNN(nn.Module):
    """Small CNN for map feature extraction."""

    def __init__(self, dim_map_embed: int = 64, in_channels: int = 1):
        super().__init__()
        out_ch = dim_map_embed - 3
        if out_ch <= 0:
            raise ValueError(f"dim_map_embed must be > 3, got {dim_map_embed}.")
        mid_ch = max(16, out_ch // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        )

    def forward(self, height_only: torch.Tensor) -> torch.Tensor:
        return self.net(height_only)


# NOTE: This MapTransformer intentionally DIVERGES from the canonical one in
# networks/layers/map_transformer.py (deploy-trimmed ctor, MapCNN(dim_map_embed,
# in_channels=1), forward takes z_intent). State_dict bound to this deploy
# variant's checkpoints, so it is NOT merged into the canonical class.
class MapTransformer(nn.Module):
    """Cross-attention map encoder."""

    def __init__(
        self,
        dim_proprio: int,
        dim_intent: int,
        dim_map_embed: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.dim_map_embed = dim_map_embed
        self.dim_proprio = dim_proprio

        self.norm_proprio = nn.LayerNorm(dim_proprio) if dim_proprio > 0 else nn.Identity()
        self.norm_intent = nn.LayerNorm(dim_intent)
        self.query_proj = nn.Linear(dim_proprio + dim_intent, dim_map_embed)

        self.map_cnn = MapCNN(dim_map_embed=dim_map_embed, in_channels=1)
        self.norm_kv = nn.LayerNorm(dim_map_embed)
        self.cross_attn = nn.MultiheadAttention(dim_map_embed, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim_map_embed, dim_map_embed)
        self.norm_out = nn.LayerNorm(dim_map_embed)

    def forward(
        self,
        map_3d: torch.Tensor,  # [B,3,H,W]
        proprio: torch.Tensor,  # [B,P]
        z_intent: torch.Tensor,  # [B,D]
    ) -> torch.Tensor:
        batch_size = map_3d.shape[0]

        height_only = map_3d[:, 2:3, :, :]
        cnn_feat = self.map_cnn(height_only)
        combined = torch.cat([cnn_feat, map_3d], dim=1)

        kv = combined.reshape(batch_size, self.dim_map_embed, -1).permute(0, 2, 1)
        kv = self.norm_kv(kv)

        z_norm = self.norm_intent(z_intent)
        if self.dim_proprio > 0:
            p_norm = self.norm_proprio(proprio)
            q_in = torch.cat([p_norm, z_norm], dim=-1)
        else:
            q_in = z_norm
        q = self.query_proj(q_in).unsqueeze(1)

        attn_out, _ = self.cross_attn(q, kv, kv)
        out = self.out_proj(attn_out.squeeze(1))
        return self.norm_out(out)


def _zero_init_last_linear(module: nn.Module) -> None:
    last_linear = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            last_linear = submodule
    if last_linear is not None:
        nn.init.constant_(last_linear.weight, 0.0)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0.0)


class DeployResidualVisionSonicActorCritic(SonicActorCritic):
    """Residual vision head on top of deploy-aligned SONIC backbone."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: list[int] = [2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        map_height: int = 17,
        map_width: int = 11,
        map_resolution: float = 0.1,
        dim_map_embed: int = 64,
        num_attn_heads: int = 4,
        residual_hidden_dims: list[int] = [256, 128],
        residual_scale: float = 0.25,
        z_clip: float = 1.0,
        normalize_height: bool = True,
        vision_dropout: float = 0.0,
        freeze_base_sonic: bool = True,
        freeze_base_action_std: bool = True,
        critic_use_vision: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )

        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.map_resolution = float(map_resolution)
        self.dim_map_embed = int(dim_map_embed)
        self.residual_scale = float(residual_scale)
        self.z_clip = float(z_clip)
        self.normalize_height = bool(normalize_height)
        self.vision_dropout = float(vision_dropout)
        self.freeze_base_sonic = bool(freeze_base_sonic)
        self.freeze_base_action_std = bool(freeze_base_action_std)
        self.critic_use_vision = bool(critic_use_vision)

        xs = (torch.arange(self.map_height) - (self.map_height - 1) / 2.0) * self.map_resolution
        ys = (torch.arange(self.map_width) - (self.map_width - 1) / 2.0) * self.map_resolution
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        self.register_buffer("_grid_x", grid_x.clone())
        self.register_buffer("_grid_y", grid_y.clone())

        self.map_transformer = MapTransformer(
            dim_proprio=self.proprio_dim,
            dim_intent=self.latent_dim,
            dim_map_embed=self.dim_map_embed,
            num_heads=num_attn_heads,
        )
        self.residual_mlp = MLP(
            self.proprio_dim + self.dim_map_embed,
            num_actions,
            residual_hidden_dims,
            activation,
        )
        _zero_init_last_linear(self.residual_mlp)

        if self.critic_use_vision:
            critic_hidden = critic_hidden_dims if critic_hidden_dims is not None else actor_hidden_dims
            self.critic = MLP(
                self.num_critic_obs + self.dim_map_embed,
                1,
                critic_hidden,
                activation,
            )

        if self.freeze_base_sonic:
            self._apply_base_freeze()

        self._last_aux: dict[str, torch.Tensor] = {}

    def _apply_base_freeze(self) -> None:
        self._set_trainable(self.robot_encoder, False)
        self._set_trainable(self.robot_encoder_proj, False)
        self._set_trainable(self.human_encoder, False)
        self._set_trainable(self.human_encoder_proj, False)
        self._set_trainable(self.hybrid_encoder, False)
        self._set_trainable(self.hybrid_encoder_proj, False)
        self._set_trainable(self.control_decoder, False)
        if self.freeze_base_action_std:
            self.std.requires_grad_(False)

    def set_warmup_freeze(
        self,
        *,
        freeze_encoders: bool,
        freeze_control_decoder: bool,
        freeze_action_std: bool,
    ) -> dict[str, bool]:
        if not self.freeze_base_sonic:
            return super().set_warmup_freeze(
                freeze_encoders=freeze_encoders,
                freeze_control_decoder=freeze_control_decoder,
                freeze_action_std=freeze_action_std,
            )

        self._apply_base_freeze()
        self._warmup_freeze_state = {
            "encoders_frozen": True,
            "control_decoder_frozen": True,
            "action_std_frozen": self.freeze_base_action_std,
        }
        return dict(self._warmup_freeze_state)

    def _build_map_3d(self, obs: TensorDict) -> torch.Tensor:
        if "vision" not in obs.keys():
            raise KeyError("DeployResidualVisionSonicActorCritic requires observation key 'vision'.")
        height_scan = obs["vision"]
        if height_scan.ndim != 2:
            raise ValueError(
                f"'vision' must be rank-2 [batch, H*W], got shape {tuple(height_scan.shape)}."
            )
        expected = self.map_height * self.map_width
        if height_scan.shape[-1] != expected:
            raise ValueError(
                f"'vision' last dim mismatch: got {height_scan.shape[-1]}, expected {expected} "
                f"({self.map_height}x{self.map_width})."
            )

        batch_size = height_scan.shape[0]
        map_z = height_scan.reshape(batch_size, 1, self.map_height, self.map_width)
        if self.z_clip > 0.0:
            map_z = map_z.clamp(-self.z_clip, self.z_clip)
            if self.normalize_height:
                map_z = map_z / self.z_clip
        map_x = self._grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        map_y = self._grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.cat([map_x, map_y, map_z], dim=1)

    def _compute_token_and_base_mean(self, obs: TensorDict, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.freeze_base_sonic:
            with torch.no_grad():
                token = self._select_action_token(obs)
                token = self._prepare_action_token_for_policy(token)
                policy_input = self.build_control_decoder_input(proprio, token)
                base_mean = self.control_decoder(policy_input)
            return token, base_mean

        token = self._select_action_token(obs)
        token = self._prepare_action_token_for_policy(token)
        policy_input = self.build_control_decoder_input(proprio, token)
        base_mean = self.control_decoder(policy_input)
        return token, base_mean

    def _compute_z_map(self, obs: TensorDict, proprio: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        map_3d = self._build_map_3d(obs)
        z_map = self.map_transformer(map_3d, proprio, token)
        if self.training and self.vision_dropout > 0.0:
            keep_mask = (torch.rand(z_map.shape[0], 1, device=z_map.device) > self.vision_dropout).float()
            z_map = z_map * keep_mask
        return z_map

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        proprio = self.get_actor_obs(obs)
        proprio = self.actor_obs_normalizer(proprio)
        token, base_mean = self._compute_token_and_base_mean(obs, proprio)
        z_map = self._compute_z_map(obs, proprio, token)

        residual_input = torch.cat([proprio, z_map], dim=-1)
        delta_action = torch.tanh(self.residual_mlp(residual_input)) * self.residual_scale
        mean = base_mean + delta_action

        std = self._get_action_std(mean)
        self.distribution = Normal(mean, std)
        self._last_aux = {"base_action": base_mean.detach(), "delta_action": delta_action.detach(), "z_map": z_map.detach()}
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            proprio = self.get_actor_obs(obs)
            proprio = self.actor_obs_normalizer(proprio)
            token, base_mean = self._compute_token_and_base_mean(obs, proprio)
            z_map = self._compute_z_map(obs, proprio, token)

            residual_input = torch.cat([proprio, z_map], dim=-1)
            delta_action = torch.tanh(self.residual_mlp(residual_input)) * self.residual_scale
            mean = base_mean + delta_action

            std = self._get_action_std(mean)
            self.distribution = Normal(mean, std)
            self._last_aux = {
                "base_action": base_mean.detach(),
                "delta_action": delta_action.detach(),
                "z_map": z_map.detach(),
            }
            return mean

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        if not self.critic_use_vision:
            return self.critic(critic_obs)

        proprio = self.get_actor_obs(obs)
        proprio = self.actor_obs_normalizer(proprio)
        token, _ = self._compute_token_and_base_mean(obs, proprio)
        z_map = self._compute_z_map(obs, proprio, token)
        critic_input = torch.cat([critic_obs, z_map], dim=-1)
        return self.critic(critic_input)
