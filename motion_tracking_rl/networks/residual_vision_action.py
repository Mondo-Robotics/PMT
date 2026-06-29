# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Vision-augmented SONIC Actor-Critic (staged training)
- BASE_ONLY: train blind SONIC only
- VISION_ADAPTER: load base ckpt, freeze base, train vision + adapters + NEW critic
- FINETUNE_ALL: train everything end-to-end

Key stability features:
- proprio_dim==0 safe transformer query
- metric centered coordinates for (x,y)
- gated residuals + zero-init last layers
- optional height clipping/normalization
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.networks.layers.map_transformer import MapCNN, MapTransformer
from motion_tracking_rl.networks.actor_critic import SonicActorCritic

# ``MapCNN``/``MapTransformer`` were extracted to layers/map_transformer.py as the
# canonical variant (PMT plan §2, Phase 0 step 3). They are re-imported above so
# this module's ``MapTransformer`` (and any old reference to it) is the canonical
# class; ``__all__``/qualname compatibility is preserved.


# ---------------------------
# Enums / Small helpers
# ---------------------------

class TrainingStage(Enum):
    BASE_ONLY = auto()
    VISION_ADAPTER = auto()
    FINETUNE_ALL = auto()


class GatedResidual(nn.Module):
    """
    Output = x + tanh(alpha) * residual
    alpha starts at 0 => identity at init
    """
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, residual: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return x + scale * torch.tanh(self.alpha) * residual


def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


def zero_init_last_linear(module: nn.Module) -> None:
    """Zero-init the last Linear layer (stabilizes gated residual training)."""
    lin = _last_linear(module)
    if lin is None:
        return
    nn.init.constant_(lin.weight, 0.0)
    if lin.bias is not None:
        nn.init.constant_(lin.bias, 0.0)



# ---------------------------
# Main staged network
# ---------------------------

class ModularVisionSonicActorCritic(SonicActorCritic):
    """
    Extends SonicActorCritic with:
      - MapTransformer (terrain)
      - intent modulation (z_fsq -> z_modulated)
      - action residual head
      - staged training support with critic reinit for vision stages
    """
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # --- SONIC args ---
        robot_motion_dim: int = 580,
        human_motion_dim: int = 660,
        proprio_dim: int = 0,
        latent_dim: int = 256,
        actor_hidden_dims: list[int] = [2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims: list[int] = [512, 256, 128],
        # --- Staging ---
        training_stage: str = "base_only",
        base_policy_ckpt: Optional[str] = None,
        # --- Vision ---
        map_height: int = 17,
        map_width: int = 11,
        map_resolution: float = 0.1,
        dim_map_embed: int = 64,
        num_attn_heads: int = 4,
        z_clip: float = 1.0,                 # clip height to [-z_clip, z_clip]
        normalize_height: bool = True,       # scale by z_clip -> ~[-1,1]
        freeze_std_in_adapter: bool = False, # optional stability
        **kwargs: Any,
    ) -> None:
        # Stage parse
        try:
            self.stage = TrainingStage[training_stage.upper()]
        except KeyError:
            raise ValueError(f"Invalid training_stage: {training_stage}. "
                             f"Use one of {[s.name for s in TrainingStage]}")

        self.use_vision = (self.stage != TrainingStage.BASE_ONLY)

        # Save vision params
        self.map_height = map_height
        self.map_width = map_width
        self.map_resolution = map_resolution
        self.dim_map_embed = dim_map_embed
        self.z_clip = float(z_clip)
        self.normalize_height = bool(normalize_height)
        self.freeze_std_in_adapter = bool(freeze_std_in_adapter)

        # Parent init (builds base encoders, FSQ, base control_decoder, base critic)
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            robot_motion_dim=robot_motion_dim,
            human_motion_dim=human_motion_dim,
            proprio_dim=proprio_dim,
            latent_dim=latent_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            **kwargs,
        )

        # Build metric centered grids (yaw frame already handled by your pipeline)
        xs = (torch.arange(map_height) - (map_height - 1) / 2.0) * map_resolution
        ys = (torch.arange(map_width)  - (map_width  - 1) / 2.0) * map_resolution
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        # Clone to avoid expanded/overlapping storage that breaks state_dict loading.
        grid_x = grid_x.clone()
        grid_y = grid_y.clone()
        self.register_buffer("_grid_x", grid_x)  # [H,W]
        self.register_buffer("_grid_y", grid_y)  # [H,W]

        self._height_vision_has_mask = self._vision_has_height_mask(obs)

        # Vision encoder
        self.map_transformer = MapTransformer(
            dim_proprio=self.proprio_dim,
            dim_intent=self.latent_dim,
            dim_map_embed=dim_map_embed,
            num_heads=num_attn_heads,
            terrain_input_channels=2 if self._height_vision_has_mask else 1,
            map_coord_channels=4 if self._height_vision_has_mask else 3,
        )

        # Adapters (gated)
        self.intent_modulator = MLP(dim_map_embed, latent_dim, [256, 128], "elu")
        zero_init_last_linear(self.intent_modulator)
        self.intent_gate = GatedResidual(latent_dim)

        residual_in = self.proprio_dim + dim_map_embed
        self.residual_policy = MLP(residual_in, num_actions, [256, 128], "elu")
        zero_init_last_linear(self.residual_policy)
        self.residual_gate = GatedResidual(num_actions)

        # Critic reinit for vision stages (fresh critic sees base critic obs + z_map)
        if self.use_vision:
            base_critic_dim = self._infer_base_critic_dim(obs, obs_groups)
            vision_critic_dim = base_critic_dim + dim_map_embed
            self.critic = MLP(vision_critic_dim, 1, critic_hidden_dims, "elu")
            print(f"[Init] Vision critic: in={vision_critic_dim} (base={base_critic_dim} + map={dim_map_embed})")

        # Load ckpt if provided
        if base_policy_ckpt is not None:
            self._smart_load_checkpoint(base_policy_ckpt)
        elif self.stage == TrainingStage.VISION_ADAPTER:
            raise ValueError("VISION_ADAPTER requires base_policy_ckpt (pretrained blind policy).")

        # Freeze/unfreeze
        self._configure_freezing()

        print(f"[Init] Stage={self.stage.name} use_vision={self.use_vision} "
              f"map={map_height}x{map_width}@{map_resolution} z_clip={self.z_clip}")

    # ---------------------------
    # Init helpers
    # ---------------------------

    def _infer_base_critic_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]]) -> int:
        if "critic" not in obs_groups:
            raise KeyError("obs_groups must include a 'critic' list of keys.")
        dim = 0
        for k in obs_groups["critic"]:
            if k not in obs:
                raise KeyError(f"obs missing critic key '{k}'.")
            dim += obs[k].shape[-1]
        return dim

    def _vision_has_height_mask(self, obs: TensorDict) -> bool:
        if "vision" not in obs:
            raise KeyError("Missing 'vision' observation for terrain encoder initialization.")
        vision = obs["vision"]
        if vision.ndim != 2:
            raise ValueError(f"Expected flat height vision [B,N], got shape {tuple(vision.shape)}.")
        expected = self.map_height * self.map_width
        if vision.shape[-1] == expected:
            return False
        if vision.shape[-1] == 2 * expected:
            return True
        raise ValueError(
            "Expected height vision with either H*W or 2*H*W features, "
            f"got shape {tuple(vision.shape)}."
        )

    def _smart_load_checkpoint(self, path: str) -> None:
        device = next(self.parameters()).device
        print(f"[Loader] Loading checkpoint: {path} (map_location={device})")
        ckpt = torch.load(path, map_location=device)

        # common wrappers
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

        current = self.state_dict()
        compatible = {}
        skipped = []

        for k, v in ckpt.items():
            if k in current and hasattr(v, "shape") and v.shape == current[k].shape:
                compatible[k] = v
            elif k in current:
                skipped.append(k)

        incompat = self.load_state_dict(compatible, strict=False)

        print(f"[Loader] Loaded {len(compatible)} tensors. Skipped {len(skipped)} (expected for critic/vision).")
        # You can print incompat.missing_keys if you want, but it's often noisy.

    def _configure_freezing(self) -> None:
        def set_train(mod: nn.Module, trainable: bool) -> None:
            for p in mod.parameters():
                p.requires_grad = trainable

        base_modules = [
            self.robot_encoder, self.human_encoder,
            self.robot_encoder_proj, self.human_encoder_proj,
            self.control_decoder,
        ]
        if self.hybrid_encoder is not None:
            base_modules.append(self.hybrid_encoder)
        if self.hybrid_encoder_proj is not None:
            base_modules.append(self.hybrid_encoder_proj)
        vision_modules = [
            self.map_transformer,
            self.intent_modulator, self.intent_gate,
            self.residual_policy, self.residual_gate,
        ]

        if self.stage == TrainingStage.BASE_ONLY:
            for m in base_modules:
                set_train(m, True)
            for m in vision_modules:
                set_train(m, False)
            # keep critic/std trainable
            for p in self.critic.parameters():
                p.requires_grad = True
            self.std.requires_grad = True

        elif self.stage == TrainingStage.VISION_ADAPTER:
            for m in base_modules:
                set_train(m, False)
            for m in vision_modules:
                set_train(m, True)

            for p in self.critic.parameters():
                p.requires_grad = True

            # optional: freeze std early for stability
            self.std.requires_grad = (not self.freeze_std_in_adapter)

        elif self.stage == TrainingStage.FINETUNE_ALL:
            for p in self.parameters():
                p.requires_grad = True

    # ---------------------------
    # Encoding helpers
    # ---------------------------

    def _encode_motion_intent(self, obs: TensorDict) -> torch.Tensor:
        return self._select_action_token(obs)

    def _get_policy_proprio(self, obs: TensorDict) -> torch.Tensor:
        keys = self.obs_groups.get("policy", [])
        if len(keys) == 0:
            # allow empty proprio
            B = obs[next(iter(obs.keys()))].shape[0]
            return torch.empty(B, 0, device=next(self.parameters()).device)
        return torch.cat([obs[k] for k in keys], dim=-1)

    def _get_map_encoding(self, obs: TensorDict, proprio: torch.Tensor, z_fsq: torch.Tensor) -> torch.Tensor:
        if "vision" not in obs:
            raise KeyError("Missing 'vision' in obs. Expected flat height scan [B,H*W].")

        height_scan = obs["vision"]
        B = height_scan.shape[0]
        num_cells = self.map_height * self.map_width
        if height_scan.shape[-1] == num_cells:
            map_height = height_scan
            validity_mask = None
        elif height_scan.shape[-1] == 2 * num_cells:
            map_height = height_scan[..., :num_cells]
            validity_mask = height_scan[..., num_cells:]
        else:
            raise ValueError(
                "Expected flat height scan [B,H*W] or [B,2*H*W] for masked terrain input, "
                f"got shape {tuple(height_scan.shape)}."
            )

        # [B,1,H,W]
        map_z = map_height.reshape(B, 1, self.map_height, self.map_width)

        # optional clip/normalize
        if self.z_clip > 0:
            map_z = map_z.clamp(-self.z_clip, self.z_clip)
            if self.normalize_height:
                map_z = map_z / self.z_clip

        # metric coords [B,1,H,W]
        map_x = self._grid_x.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        map_y = self._grid_y.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        if validity_mask is not None:
            map_valid = validity_mask.reshape(B, 1, self.map_height, self.map_width)
            map_valid = (map_valid > 0.5).to(dtype=map_z.dtype)
            map_3d = torch.cat([map_x, map_y, map_z, map_valid], dim=1)  # [B,4,H,W]
        else:
            map_3d = torch.cat([map_x, map_y, map_z], dim=1)  # [B,3,H,W]
        return self.map_transformer(map_3d, proprio, z_fsq)  # [B,dim_map_embed]

    def _get_critic_obs(self, obs: TensorDict, z_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        critic_keys = self.obs_groups.get("critic", [])
        if len(critic_keys) == 0:
            raise KeyError("obs_groups['critic'] is empty. Define critic input keys.")
        base_obs = torch.cat([obs[k] for k in critic_keys], dim=-1)

        if self.use_vision:
            if z_map is None:
                pad = torch.zeros(base_obs.shape[0], self.dim_map_embed, device=base_obs.device)
                return torch.cat([base_obs, pad], dim=-1)
            return torch.cat([base_obs, z_map], dim=-1)

        return base_obs

    # ---------------------------
    # Actor / Critic API
    # ---------------------------

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        proprio = self._get_policy_proprio(obs)

        # encode motion
        if self.stage == TrainingStage.VISION_ADAPTER:
            with torch.no_grad():
                z_fsq = self._encode_motion_intent(obs)
        else:
            z_fsq = self._encode_motion_intent(obs)

        # BASE_ONLY: standard SONIC
        if not self.use_vision:
            x = torch.cat([proprio, z_fsq], dim=-1)
            mean = self.control_decoder(x)
            std = self._get_action_std(mean)
            self.distribution = Normal(mean, std)
            return self.distribution.sample()

        # VISION stages
        z_map = self._get_map_encoding(obs, proprio, z_fsq)

        # gated intent modulation
        delta_z = self.intent_modulator(z_map)
        z_mod = self.intent_gate(z_fsq, delta_z)

        # base action (may be frozen)
        x = torch.cat([proprio, z_mod], dim=-1)
        a_base = self.control_decoder(x)

        # residual action
        r_in = torch.cat([proprio, z_map], dim=-1)
        a_res = self.residual_policy(r_in)

        mean = self.residual_gate(a_base, a_res)

        std = self._get_action_std(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action for deployment (returns mean)."""
        with torch.no_grad():
            proprio = self._get_policy_proprio(obs)
            z_fsq = self._encode_motion_intent(obs)

            if not self.use_vision:
                x = torch.cat([proprio, z_fsq], dim=-1)
                mean = self.control_decoder(x)
                std = self._get_action_std(mean)
                self.distribution = Normal(mean, std)
                return mean

            z_map = self._get_map_encoding(obs, proprio, z_fsq)

            delta_z = self.intent_modulator(z_map)
            z_mod = self.intent_gate(z_fsq, delta_z)

            x = torch.cat([proprio, z_mod], dim=-1)
            a_base = self.control_decoder(x)

            r_in = torch.cat([proprio, z_map], dim=-1)
            a_res = self.residual_policy(r_in)

            mean = self.residual_gate(a_base, a_res)

            std = self._get_action_std(mean)
            self.distribution = Normal(mean, std)
            return mean

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """
        Critic forward. Returns V(s) as shape [B,1] (or [B] depending on your MLP).
        """
        # To compute z_map for critic query we need proprio+z_fsq (policy proprio is fine)
        proprio = self._get_policy_proprio(obs)

        if self.stage == TrainingStage.VISION_ADAPTER:
            with torch.no_grad():
                z_fsq = self._encode_motion_intent(obs)
        else:
            z_fsq = self._encode_motion_intent(obs)

        z_map = None
        if self.use_vision:
            # allow gradients through vision for value learning
            z_map = self._get_map_encoding(obs, proprio, z_fsq)

        critic_in = self._get_critic_obs(obs, z_map)
        return self.critic(critic_in)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Convenient optimizer construction: only params with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]
