from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.networks.actor_critic import SonicActorCritic
from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.utils import build_obs_schema


def _small_init_last_linear(module: nn.Module) -> None:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    nn.init.xavier_uniform_(linears[-1].weight)
    linears[-1].weight.data.mul_(0.01)
    nn.init.zeros_(linears[-1].bias)


class TerrainMapEncoder(nn.Module):
    def __init__(
        self,
        vision_channels: int,
        proprio_dim: int,
        token_dim: int,
        map_height: int = 17,
        map_width: int = 11,
        map_resolution: float = 0.1,
        embed_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.vision_channels = int(vision_channels)
        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.embed_dim = int(embed_dim)

        xs = (torch.arange(map_height) - (map_height - 1) / 2.0) * map_resolution
        ys = (torch.arange(map_width) - (map_width - 1) / 2.0) * map_resolution
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        self.register_buffer("xy_grid", torch.stack([grid_x, grid_y], dim=0).unsqueeze(0))

        self.map_cnn = nn.Sequential(
            nn.Conv2d(self.vision_channels, self.embed_dim - 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.embed_dim - 2, self.embed_dim - 2, kernel_size=1),
        )
        self.query = nn.Sequential(
            nn.LayerNorm(proprio_dim + token_dim),
            nn.Linear(proprio_dim + token_dim, self.embed_dim),
        )
        self.kv_norm = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.embed_dim), nn.SiLU())

    def forward(self, vision: torch.Tensor, proprio: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        batch = vision.shape[0]
        terrain = vision.view(batch, self.vision_channels, self.map_height, self.map_width)
        xy = self.xy_grid.expand(batch, -1, -1, -1).to(dtype=terrain.dtype)
        kv = torch.cat([self.map_cnn(terrain), xy], dim=1).flatten(2).transpose(1, 2)
        query = self.query(torch.cat([proprio, token], dim=-1)).unsqueeze(1)
        z_vis, _ = self.attn(query, self.kv_norm(kv), self.kv_norm(kv), need_weights=False)
        return self.out(z_vis.squeeze(1))


class ResidualSonicTeacherActorCritic(SonicActorCritic):
    """Blind residual policy on top of the deploy SONIC encoder/decoder."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        residual_hidden_dims: list[int] = [256, 128],
        alpha_init: float = 0.05,
        use_vision: bool = False,
        vision_key: str = "vision",
        map_height: int = 17,
        map_width: int = 11,
        map_resolution: float = 0.1,
        dim_map_embed: int = 64,
        num_attn_heads: int = 4,
        activation: str = "silu",
        freeze_sonic_encoder: bool = True,
        freeze_sonic_decoder: bool = True,
        freeze_action_std: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, activation=activation, **kwargs)
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.activation = activation
        self.freeze_sonic_encoder = bool(freeze_sonic_encoder)
        self.freeze_sonic_decoder = bool(freeze_sonic_decoder)
        self.freeze_action_std = bool(freeze_action_std)
        self.alpha_init = float(alpha_init)
        self.use_vision = bool(use_vision)
        self.vision_key = vision_key
        self._obs_schema = build_obs_schema(obs, obs_groups)

        residual_input_dim = self.proprio_dim + self.latent_dim
        self.vision_encoder = None
        if self.use_vision:
            flat_vision_dim = obs[self.vision_key].shape[-1]
            hw = int(map_height) * int(map_width)
            if flat_vision_dim % hw != 0:
                raise ValueError(f"vision dim {flat_vision_dim} is not divisible by H*W={hw}.")
            vision_channels = flat_vision_dim // hw
            self.vision_encoder = TerrainMapEncoder(
                vision_channels=vision_channels,
                proprio_dim=self.proprio_dim,
                token_dim=self.latent_dim,
                map_height=map_height,
                map_width=map_width,
                map_resolution=map_resolution,
                embed_dim=dim_map_embed,
                num_heads=num_attn_heads,
            )
            residual_input_dim += int(dim_map_embed)

        self.action_residual = MLP(
            residual_input_dim,
            num_actions,
            residual_hidden_dims,
            self.activation,
        )
        self.alpha = nn.Parameter(torch.full((num_actions,), self.alpha_init))
        _small_init_last_linear(self.action_residual)

        self._apply_configured_freeze()
        self._last_aux: dict[str, torch.Tensor] = {}

    def _apply_configured_freeze(self) -> None:
        if self.freeze_sonic_encoder:
            self._set_trainable(self.robot_encoder, False)
            self._set_trainable(self.robot_encoder_proj, False)
            self._set_trainable(self.human_encoder, False)
            self._set_trainable(self.human_encoder_proj, False)
            self._set_trainable(self.hybrid_encoder, False)
            self._set_trainable(self.hybrid_encoder_proj, False)
        if self.freeze_sonic_decoder:
            self._set_trainable(self.control_decoder, False)
        self._set_trainable(self.motion_decoder, False)
        self.std.requires_grad_(not self.freeze_action_std)

    def _compute_token(self, obs: TensorDict) -> torch.Tensor:
        if self.freeze_sonic_encoder:
            with torch.no_grad():
                return self._select_action_token(obs)
        return self._select_action_token(obs)

    def _compute_base_action(self, proprio: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        action_token = self._prepare_action_token_for_policy(token)
        policy_input = self.build_control_decoder_input(proprio, action_token)
        if self.freeze_sonic_decoder:
            with torch.no_grad():
                return self.control_decoder(policy_input)
        return self.control_decoder(policy_input)

    def _actor_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        proprio = self.actor_obs_normalizer(self.get_actor_obs(obs))
        token = self._compute_token(obs)
        base_action = self._compute_base_action(proprio, token)
        residual_inputs = [proprio, token]
        z_vis = None
        if self.use_vision:
            z_vis = self.vision_encoder(obs[self.vision_key], proprio, token)
            residual_inputs.append(z_vis)
        residual = self.action_residual(torch.cat(residual_inputs, dim=-1))
        delta_action = torch.tanh(self.alpha) * residual
        action = base_action + delta_action
        outputs = {
            "action": action,
            "token": token,
            "base_action": base_action,
            "residual": residual,
            "delta_action": delta_action,
            "alpha": self.alpha,
        }
        if z_vis is not None:
            outputs["z_vis"] = z_vis
        return outputs

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        outputs = self._actor_outputs(obs)
        self.distribution = Normal(outputs["action"], self._get_action_std(outputs["action"]))
        self._last_aux = {k: v.detach() for k, v in outputs.items() if k != "action"}
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        outputs = self._actor_outputs(obs)
        self.distribution = Normal(outputs["action"], self._get_action_std(outputs["action"]))
        self._last_aux = {k: v.detach() for k, v in outputs.items() if k != "action"}
        return outputs["action"]

    def infer_teacher_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        outputs = self._actor_outputs(obs)
        latent = self.encode_robot_pre_quant(obs["robot_encoder"])
        return {
            "action": outputs["action"],
            "token": outputs["token"],
            "latent": latent,
            "base_action": outputs["base_action"],
            "delta_action": outputs["delta_action"],
        }

    def get_last_aux_outputs(self, clear: bool = False) -> dict[str, torch.Tensor]:
        aux = dict(self._last_aux)
        if clear:
            self._last_aux.clear()
        return aux

    def get_ppo_log_stats(self) -> dict[str, float]:
        aux = self.get_last_aux_outputs(clear=False)
        if not aux:
            return {}
        base_abs = aux["base_action"].abs().mean()
        delta_abs = aux["delta_action"].abs().mean()
        return {
            "residual_alpha_abs": float(self.alpha.detach().abs().mean().item()),
            "residual_gate_abs": float(torch.tanh(self.alpha.detach()).abs().mean().item()),
            "residual_raw_abs": float(aux["residual"].abs().mean().item()),
            "residual_delta_abs": float(delta_abs.item()),
            "residual_base_action_abs": float(base_abs.item()),
            "residual_delta_base_ratio": float((delta_abs / (base_abs + 1.0e-6)).item()),
        }

    def set_warmup_freeze(
        self,
        *,
        freeze_encoders: bool,
        freeze_control_decoder: bool,
        freeze_action_std: bool,
    ) -> dict[str, bool]:
        if self.freeze_sonic_encoder or freeze_encoders:
            self._set_trainable(self.robot_encoder, False)
            self._set_trainable(self.robot_encoder_proj, False)
            self._set_trainable(self.human_encoder, False)
            self._set_trainable(self.human_encoder_proj, False)
            self._set_trainable(self.hybrid_encoder, False)
            self._set_trainable(self.hybrid_encoder_proj, False)
        else:
            self._set_trainable(self.robot_encoder, self.train_robot_encoder)
            self._set_trainable(self.robot_encoder_proj, self.train_robot_encoder)
            self._set_trainable(self.human_encoder, self.train_human_encoder)
            self._set_trainable(self.human_encoder_proj, self.train_human_encoder)
            self._set_trainable(self.hybrid_encoder, self.train_hybrid_encoder)
            self._set_trainable(self.hybrid_encoder_proj, self.train_hybrid_encoder)

        control_decoder_frozen = self.freeze_sonic_decoder or freeze_control_decoder
        self._set_trainable(self.control_decoder, not control_decoder_frozen)
        action_std_frozen = self.freeze_action_std or freeze_action_std
        self.std.requires_grad_(not action_std_frozen)

        self._warmup_freeze_state = {
            "encoders_frozen": self.freeze_sonic_encoder or freeze_encoders,
            "control_decoder_frozen": control_decoder_frozen,
            "action_std_frozen": action_std_frozen,
        }
        return dict(self._warmup_freeze_state)

    def build_optimizer_param_groups(
        self,
        base_lr: float,
        backbone_lr_scale: float = 1.0,
        vision_adapter_lr_scale: float = 1.0,
        critic_lr_scale: float = 1.0,
    ) -> list[dict[str, object]]:
        groups: dict[str, list[nn.Parameter]] = {"backbone": [], "residual": [], "critic": [], "action_std": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("critic."):
                groups["critic"].append(param)
            elif name.startswith("action_residual.") or name == "alpha":
                groups["residual"].append(param)
            elif name.startswith("vision_encoder."):
                groups["residual"].append(param)
            elif name == "std":
                groups["action_std"].append(param)
            else:
                groups["backbone"].append(param)

        specs = [
            ("backbone", backbone_lr_scale),
            ("residual", vision_adapter_lr_scale),
            ("critic", critic_lr_scale),
            ("action_std", backbone_lr_scale),
        ]
        return [
            {
                "name": name,
                "params": params,
                "lr": float(base_lr) * float(lr_scale),
                "lr_scale": float(lr_scale),
            }
            for name, lr_scale in specs
            if (params := groups[name])
        ]

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "policy_class": self.__class__.__name__,
            "policy_family": "residual_sonic_teacher",
            "obs_schema": self._obs_schema,
            "signature": {
                "num_actions": self.num_actions,
                "proprio_dim": self.proprio_dim,
                "latent_dim": self.latent_dim,
                "alpha_init": self.alpha_init,
                "use_vision": self.use_vision,
                "freeze_sonic_encoder": self.freeze_sonic_encoder,
                "freeze_sonic_decoder": self.freeze_sonic_decoder,
            },
        }
