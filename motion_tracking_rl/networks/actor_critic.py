# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from motion_tracking_rl.networks.layers import MLP, EmpiricalNormalization
from motion_tracking_rl.registry import register_network


SONIC_AUX_LOSS_COEF = {
    "g1_recon": 0.01,
    "g1_smpl_latent": 1.0,
    "g1_teleop_latent": 1.0,
    "teleop_smpl_latent": 1.0,
    "reencoded_smpl_g1_latent": 1.0,
}


class FSQ(nn.Module):
    """Finite Scalar Quantization (FSQ).

    Ref: Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", 2023.
    """

    def __init__(self, levels: int | list[int], dim: int | None = None, eps: float = 1.0e-3):
        super().__init__()
        if isinstance(levels, int):
            if dim is None:
                raise ValueError("FSQ integer `levels` requires `dim`.")
            levels = [int(levels)] * int(dim)
        else:
            levels = [int(level) for level in levels]
            if dim is not None and len(levels) != int(dim):
                raise ValueError(f"FSQ levels length {len(levels)} must match dim {dim}.")
        if not levels:
            raise ValueError("FSQ requires at least one level.")
        if any(level <= 1 for level in levels):
            raise ValueError(f"FSQ levels must be greater than 1, got {levels}.")

        self.levels = levels
        self.dim = len(levels)
        self.eps = float(eps)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        self.register_buffer("_levels_f", torch.tensor(levels, dtype=torch.float32))
        half_l = (self._levels_f - 1.0) * (1.0 + self.eps) / 2.0
        offset = torch.where((self._levels % 2) == 0, torch.tensor(0.5), torch.tensor(0.0))
        # Match vector_quantize_pytorch.FSQ: the pre-tanh shift is atanh(offset/half_l)
        # (NOT tan), so that tanh(z + shift)*half_l - offset is correctly centered for
        # even-level dims. Using tan() here would mis-center the quantization grid.
        shift = torch.atanh(offset / half_l)
        half_width = torch.div(self._levels, 2, rounding_mode="floor").to(dtype=torch.float32)
        self.register_buffer("_half_l", half_l)
        self.register_buffer("_offset", offset.to(dtype=torch.float32))
        self.register_buffer("_shift", shift.to(dtype=torch.float32))
        self.register_buffer("_half_width", half_width)

        codebook_size = math.prod(levels)
        if codebook_size <= torch.iinfo(torch.int64).max:
            self.register_buffer("_basis", torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int64), dim=0))
            self.codebook_size: int | None = int(codebook_size)
        else:
            self.register_buffer("_basis", torch.empty(0, dtype=torch.int64))
            self.codebook_size = None

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound latents using FSQ's shifted tanh parameterization."""
        half_l = self._half_l.to(z.device, z.dtype)
        offset = self._offset.to(z.device, z.dtype)
        shift = self._shift.to(z.device, z.dtype)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize z to discrete levels with STE (Straight-Through Estimator).

        Args:
            z: Input tensor [..., D]. D must match len(levels).

        Returns:
            z_q: Quantized continuous tensor [..., D] approx in [-1, 1].
        """
        if z.shape[-1] != self.dim:
            raise ValueError(f"FSQ: last dim {z.shape[-1]} must match dim {self.dim}")

        z_bounded = self.bound(z)
        z_rounded = torch.round(z_bounded)
        z_ste = z_bounded + (z_rounded - z_bounded).detach()
        half_width = self._half_width.to(z.device, z.dtype)
        return z_ste / half_width

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode z to integer indices per dimension.

        Returns:
            indices: LongTensor [..., D] with values in [0, L_i-1].
        """
        if z.shape[-1] != self.dim:
            raise ValueError(f"FSQ: last dim {z.shape[-1]} must match dim {self.dim}")

        z_bounded = self.bound(z)
        half_width = self._half_width.to(z.device, z.dtype)
        z_indices = torch.round(z_bounded) + half_width
        z_indices = torch.clamp(z_indices, min=0.0)
        z_indices = torch.minimum(z_indices, (self._levels.to(z.device, z.dtype) - 1.0))
        return z_indices.long()

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert multi-dimensional indices to single codebook indices.

        Args:
            indices: [..., D]
        Returns:
            codes: [..., 1] (or scalar per batch element)
        """
        if self.codebook_size is None:
            raise OverflowError(
                "This FSQ codebook is too large for packed int64 codes; use per-dimension indices instead."
            )
        basis = self._basis.to(indices.device)
        return (indices * basis).sum(dim=-1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode integer indices back to continuous quantized values.

        Args:
            indices: [..., D]
        Returns:
            z_q: [..., D]
        """
        half_width = self._half_width.to(indices.device)
        z_q = (indices.float() - half_width) / half_width
        return z_q


@register_network("SonicActorCritic", compat_name="sonic")
class SonicActorCritic(nn.Module):
    """SONIC Universal Control Policy Architecture."""
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # SONIC specific configs
        robot_motion_dim: int = 580, # 10 frames * (29 pos + 29 vel)
        human_motion_dim: int = 660, # 10 frames * (24 joints * 3 pos)
        proprio_dim: int = 0, # Auto-computed from 'policy' group when <= 0
        latent_dim: int = 64,
        fsq_levels: list[int] | int | None = None,
        num_fsq_levels: int = 32,
        fsq_level_list: list[int] | int = 32,
        max_num_tokens: int = 2,
        actor_hidden_dims: list[int] = [4096, 4096, 2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims: list[int] | None = None,
        encoder_hidden_dims: list[int] = [2048, 1024, 512, 512],
        motion_decoder_hidden_dims: list[int] = [2048, 1024, 512, 512],
        activation: str = "silu",
        init_noise_std: float = 0.05,
        min_action_std: float = 1.0e-3,
        max_action_std: float = 0.5,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        aux_loss_coef: dict[str, float] | None = None,
        reencode_smpl_g1_recon: bool = True,
        train_robot_encoder: bool = True,
        train_human_encoder: bool = True,
        train_hybrid_encoder: bool = False,
        hybrid_motion_dim: int | None = None,
        action_encoder_source: str = "robot",
        encoder_mode_key: str = "encoder_mode_4",
        detach_action_token: bool = True,
        decoder_proprio_layout: str = "none",
        control_decoder_input_order: str = "proprio_token",
        robot_encoder_layout: str = "raw",
        pretrained_encoder_onnx_path: str | None = None,
        pretrained_decoder_onnx_path: str | None = None,
        load_pretrained_robot_encoder: bool = True,
        load_pretrained_human_encoder: bool = True,
        load_pretrained_hybrid_encoder: bool = True,
        load_pretrained_control_decoder: bool = True,
        strict_pretrained_shapes: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "SonicActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.train_robot_encoder = bool(train_robot_encoder)
        self.train_human_encoder = bool(train_human_encoder)
        self.train_hybrid_encoder = bool(train_hybrid_encoder)
        if not (self.train_robot_encoder or self.train_human_encoder or self.train_hybrid_encoder):
            raise ValueError(
                "At least one encoder must be enabled: set one of "
                "`train_robot_encoder`, `train_human_encoder`, `train_hybrid_encoder` to True."
            )

        self.action_encoder_source = action_encoder_source.lower()
        self.encoder_mode_key = encoder_mode_key
        self.detach_action_token = bool(detach_action_token)
        self.decoder_proprio_layout = decoder_proprio_layout.lower()
        self.control_decoder_input_order = control_decoder_input_order.lower()
        self.robot_encoder_layout = robot_encoder_layout.lower()
        valid_sources = {"auto", "robot", "human", "hybrid", "mode"}
        if self.action_encoder_source not in valid_sources:
            raise ValueError(
                f"Invalid action_encoder_source: {action_encoder_source}. "
                f"Expected one of {sorted(valid_sources)}."
            )
        valid_decoder_layouts = {"none", "interleaved_step_history", "grouped_terms"}
        if self.decoder_proprio_layout not in valid_decoder_layouts:
            raise ValueError(
                f"Invalid decoder_proprio_layout: {decoder_proprio_layout}. "
                f"Expected one of {sorted(valid_decoder_layouts)}."
            )
        valid_control_input_orders = {"proprio_token", "token_proprio"}
        if self.control_decoder_input_order not in valid_control_input_orders:
            raise ValueError(
                f"Invalid control_decoder_input_order: {control_decoder_input_order}. "
                f"Expected one of {sorted(valid_control_input_orders)}."
            )
        valid_robot_layouts = {"raw", "g1_onnx_repack"}
        if self.robot_encoder_layout not in valid_robot_layouts:
            raise ValueError(
                f"Invalid robot_encoder_layout: {robot_encoder_layout}. "
                f"Expected one of {sorted(valid_robot_layouts)}."
            )

        # Calculate observation dimensions.
        num_actor_obs = 0
        for obs_group in obs_groups.get("policy", []):
            if obs_group not in obs.keys():
                raise KeyError(
                    f"Policy observation group '{obs_group}' not found. "
                    f"Available observation groups: {list(obs.keys())}"
                )
            num_actor_obs += obs[obs_group].shape[-1]
        if proprio_dim > 0 and proprio_dim != num_actor_obs:
            raise ValueError(
                f"Provided proprio_dim ({proprio_dim}) does not match policy obs dim ({num_actor_obs})."
            )
        self.proprio_dim = num_actor_obs if proprio_dim <= 0 else proprio_dim

        num_critic_obs = 0
        for obs_group in obs_groups.get("critic", []):
            if obs_group not in obs.keys():
                raise KeyError(
                    f"Critic observation group '{obs_group}' not found. "
                    f"Available observation groups: {list(obs.keys())}"
                )
            num_critic_obs += obs[obs_group].shape[-1]
        self.num_critic_obs = num_critic_obs

        # Validate required observation groups for enabled encoders.
        if self.train_robot_encoder and "robot_encoder" not in obs.keys():
            raise KeyError(
                "train_robot_encoder=True requires observation key 'robot_encoder', but it is missing."
            )
        if self.train_human_encoder and "human_encoder" not in obs.keys():
            raise KeyError(
                "train_human_encoder=True requires observation key 'human_encoder', but it is missing."
            )
        if self.train_hybrid_encoder and "hybrid_encoder" not in obs.keys():
            raise KeyError(
                "train_hybrid_encoder=True requires observation key 'hybrid_encoder', but it is missing."
            )
        if self.action_encoder_source == "mode" and self.encoder_mode_key not in obs.keys():
            raise KeyError(
                f"action_encoder_source='mode' requires observation key '{self.encoder_mode_key}', but it is missing."
            )
        if "robot_encoder" in obs.keys() and obs["robot_encoder"].shape[-1] != robot_motion_dim:
            raise ValueError(
                f"robot_encoder observation dim ({obs['robot_encoder'].shape[-1]}) "
                f"does not match robot_motion_dim ({robot_motion_dim})."
            )
        if "human_encoder" in obs.keys() and obs["human_encoder"].shape[-1] != human_motion_dim:
            raise ValueError(
                f"human_encoder observation dim ({obs['human_encoder'].shape[-1]}) "
                f"does not match human_motion_dim ({human_motion_dim})."
            )

        self.num_fsq_levels = int(num_fsq_levels)
        self.max_num_tokens = int(max_num_tokens)
        if self.num_fsq_levels <= 0 or self.max_num_tokens <= 0:
            raise ValueError(
                f"Invalid SONIC FSQ shape: num_fsq_levels={num_fsq_levels}, max_num_tokens={max_num_tokens}."
            )
        self.token_dim = self.num_fsq_levels
        self.token_total_dim = self.max_num_tokens * self.token_dim
        self.encoder_latent_dim = int(latent_dim)
        if self.encoder_latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")

        if fsq_levels is not None:
            if isinstance(fsq_levels, int):
                fsq_level_list = int(fsq_levels)
            else:
                legacy_levels = [int(level) for level in fsq_levels]
                if len(legacy_levels) in (1, self.token_dim):
                    fsq_level_list = legacy_levels
                else:
                    print(
                        "[SONIC][WARN] Ignoring deprecated fsq_levels with length "
                        f"{len(legacy_levels)}; official SONIC uses "
                        f"{self.max_num_tokens}x{self.token_dim} tokens with scalar levels."
                    )
        self.fsq_level_list = self._expand_fsq_level_list(fsq_level_list, self.token_dim)
        self.fsq_levels = list(self.fsq_level_list)
        self.aux_loss_coef = dict(SONIC_AUX_LOSS_COEF)
        if aux_loss_coef is not None:
            self.aux_loss_coef.update({str(name): float(value) for name, value in aux_loss_coef.items()})
        self.reencode_smpl_g1_recon = bool(reencode_smpl_g1_recon)

        # Encoders
        self.robot_encoder = MLP(robot_motion_dim, self.encoder_latent_dim, encoder_hidden_dims, activation)
        self.human_encoder = MLP(human_motion_dim, self.encoder_latent_dim, encoder_hidden_dims, activation)
        self.has_hybrid_encoder = "hybrid_encoder" in obs.keys() or hybrid_motion_dim is not None
        self.hybrid_motion_dim = hybrid_motion_dim if hybrid_motion_dim is not None else (
            obs["hybrid_encoder"].shape[-1] if "hybrid_encoder" in obs.keys() else None
        )
        if self.train_hybrid_encoder and self.hybrid_motion_dim is None:
            raise ValueError("train_hybrid_encoder=True requires `hybrid_motion_dim` or 'hybrid_encoder' observations.")
        if "hybrid_encoder" in obs.keys() and self.hybrid_motion_dim is not None:
            if obs["hybrid_encoder"].shape[-1] != self.hybrid_motion_dim:
                raise ValueError(
                    f"hybrid_encoder observation dim ({obs['hybrid_encoder'].shape[-1]}) "
                    f"does not match hybrid_motion_dim ({self.hybrid_motion_dim})."
                )
        if self.has_hybrid_encoder and self.hybrid_motion_dim is not None:
            self.hybrid_encoder = MLP(self.hybrid_motion_dim, self.encoder_latent_dim, encoder_hidden_dims, activation)
        else:
            self.hybrid_encoder = None

        # Quantizer
        self.quantizer = FSQ(self.fsq_level_list, dim=self.token_dim)

        # Project to FSQ dimension if needed
        self.fsq_dim = self.token_total_dim
        if self.fsq_dim != self.encoder_latent_dim:
            self.robot_encoder_proj = nn.Linear(self.encoder_latent_dim, self.fsq_dim)
            self.human_encoder_proj = nn.Linear(self.encoder_latent_dim, self.fsq_dim)
            self.hybrid_encoder_proj = (
                nn.Linear(self.encoder_latent_dim, self.fsq_dim) if self.hybrid_encoder is not None else None
            )
            # Update effective latent dim for decoders
            self.latent_dim = self.fsq_dim
        else:
            self.robot_encoder_proj = nn.Identity()
            self.human_encoder_proj = nn.Identity()
            self.hybrid_encoder_proj = nn.Identity() if self.hybrid_encoder is not None else None
            self.latent_dim = self.encoder_latent_dim

        # Control Decoder (Policy)
        # Input: Proprioception + Token
        self.control_decoder = MLP(self.proprio_dim + self.latent_dim, num_actions, actor_hidden_dims, activation)

        # Motion Decoder (Aux)
        # Input: Token
        # Output: Robot Motion (reconstruction)
        self.motion_decoder = MLP(self.latent_dim, robot_motion_dim, motion_decoder_hidden_dims, activation)

        # Critic (Standard MLP)
        critic_hidden = actor_hidden_dims if critic_hidden_dims is None else critic_hidden_dims
        self.critic = MLP(num_critic_obs, 1, critic_hidden, activation)

        # Observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization and self.proprio_dim > 0:
            self.actor_obs_normalizer = EmpiricalNormalization(self.proprio_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization and self.num_critic_obs > 0:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action Noise
        if min_action_std <= 0 or max_action_std <= 0 or min_action_std > max_action_std:
            raise ValueError(
                f"Invalid action std clamp range: [{min_action_std}, {max_action_std}]."
            )
        self.min_action_std = float(min_action_std)
        self.max_action_std = float(max_action_std)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # Freeze encoders based on config.
        self._set_trainable(self.robot_encoder, self.train_robot_encoder)
        self._set_trainable(self.robot_encoder_proj, self.train_robot_encoder)
        self._set_trainable(self.human_encoder, self.train_human_encoder)
        self._set_trainable(self.human_encoder_proj, self.train_human_encoder)
        self._set_trainable(self.hybrid_encoder, self.train_hybrid_encoder)
        self._set_trainable(self.hybrid_encoder_proj, self.train_hybrid_encoder)
        self._set_trainable(self.control_decoder, True)
        self.std.requires_grad_(True)
        self._warmup_freeze_state = {
            "encoders_frozen": False,
            "control_decoder_frozen": False,
            "action_std_frozen": False,
        }

        if pretrained_encoder_onnx_path is not None or pretrained_decoder_onnx_path is not None:
            summary = self.load_weights_from_sonic_onnx(
                encoder_onnx_path=pretrained_encoder_onnx_path,
                decoder_onnx_path=pretrained_decoder_onnx_path,
                load_robot_encoder=load_pretrained_robot_encoder,
                load_human_encoder=load_pretrained_human_encoder,
                load_hybrid_encoder=load_pretrained_hybrid_encoder,
                load_control_decoder=load_pretrained_control_decoder,
                strict_shapes=strict_pretrained_shapes,
            )
            print(f"[SONIC] Loaded ONNX pretrained weights summary: {summary}")

        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def forward(self):
        raise NotImplementedError

    @staticmethod
    def _set_trainable(module: nn.Module | None, trainable: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    @staticmethod
    def _expand_fsq_level_list(fsq_level_list: int | list[int], token_dim: int) -> list[int]:
        if isinstance(fsq_level_list, int):
            levels = [int(fsq_level_list)] * int(token_dim)
        else:
            levels = [int(level) for level in fsq_level_list]
            if len(levels) == 1:
                levels = levels * int(token_dim)
        if len(levels) != int(token_dim):
            raise ValueError(
                f"FSQ level list length {len(levels)} must match token dim {token_dim}; "
                "official SONIC uses scalar 32 expanded across each token dimension."
            )
        if any(level <= 1 for level in levels):
            raise ValueError(f"FSQ levels must be greater than 1, got {levels}.")
        return levels

    def get_fsq_token_shape(self) -> tuple[int, int]:
        return self.max_num_tokens, self.token_dim

    @staticmethod
    def _load_onnx_initializers(onnx_path: str) -> dict[str, Any]:
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise ImportError(
                "Loading pretrained ONNX weights requires the `onnx` Python package. "
                "Install it with: `pip install onnx`."
            ) from exc
        model = onnx.load(onnx_path)
        return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    @staticmethod
    def _copy_linear_params(
        linear: nn.Linear,
        weight_array: Any,
        bias_array: Any,
        source_name: str,
        strict_shapes: bool,
        transpose_weight: bool = False,
    ) -> tuple[bool, str | None]:
        # ONNX initializers can be backed by non-writable NumPy arrays.
        # Use torch.tensor(...) to force a writable copy and avoid warnings.
        source_weight = torch.tensor(weight_array)
        if transpose_weight:
            source_weight = source_weight.t()
        source_bias = torch.tensor(bias_array)

        if tuple(source_weight.shape) != tuple(linear.weight.shape):
            message = (
                f"{source_name}: weight shape mismatch "
                f"source={tuple(source_weight.shape)} target={tuple(linear.weight.shape)}"
            )
            if strict_shapes:
                raise ValueError(message)
            return False, message
        if tuple(source_bias.shape) != tuple(linear.bias.shape):
            message = (
                f"{source_name}: bias shape mismatch "
                f"source={tuple(source_bias.shape)} target={tuple(linear.bias.shape)}"
            )
            if strict_shapes:
                raise ValueError(message)
            return False, message

        linear.weight.data.copy_(source_weight.to(device=linear.weight.device, dtype=linear.weight.dtype))
        linear.bias.data.copy_(source_bias.to(device=linear.bias.device, dtype=linear.bias.dtype))
        return True, None

    def _load_encoder_branch_from_onnx(
        self,
        initializers: dict[str, Any],
        source_prefix: str,
        target_encoder: nn.Module | None,
        target_name: str,
        strict_shapes: bool,
    ) -> tuple[int, list[str]]:
        loaded = 0
        skipped: list[str] = []
        if target_encoder is None:
            skipped.append(f"{target_name}: target encoder is None")
            return loaded, skipped

        target_linears = [m for m in target_encoder.modules() if isinstance(m, nn.Linear)]
        source_linear_ids = [0, 2, 4, 6, 8]
        if len(target_linears) != len(source_linear_ids):
            message = (
                f"{target_name}: expected {len(source_linear_ids)} linear layers, "
                f"found {len(target_linears)}"
            )
            if strict_shapes:
                raise ValueError(message)
            skipped.append(message)
            return loaded, skipped

        for idx, source_layer_id in enumerate(source_linear_ids):
            weight_name = f"{source_prefix}.module.{source_layer_id}.weight"
            bias_name = f"{source_prefix}.module.{source_layer_id}.bias"
            if weight_name not in initializers or bias_name not in initializers:
                message = f"{target_name}: missing source tensors ({weight_name}, {bias_name})"
                if strict_shapes:
                    raise KeyError(message)
                skipped.append(message)
                continue

            success, reason = self._copy_linear_params(
                target_linears[idx],
                initializers[weight_name],
                initializers[bias_name],
                source_name=target_name,
                strict_shapes=strict_shapes,
                transpose_weight=False,
            )
            if success:
                loaded += 1
            elif reason is not None:
                skipped.append(reason)

        return loaded, skipped

    def _load_control_decoder_from_onnx(
        self,
        initializers: dict[str, Any],
        strict_shapes: bool,
    ) -> tuple[int, list[str]]:
        loaded = 0
        skipped: list[str] = []

        target_linears = [m for m in self.control_decoder.modules() if isinstance(m, nn.Linear)]
        if len(target_linears) != 7:
            message = (
                "control_decoder: expected 7 linear layers for deploy decoder mapping, "
                f"found {len(target_linears)}"
            )
            if strict_shapes:
                raise ValueError(message)
            skipped.append(message)
            return loaded, skipped

        matmul_names = [name for name in initializers.keys() if name.startswith("onnx::MatMul_")]
        if len(matmul_names) < 7:
            message = (
                "control_decoder: expected at least 7 MatMul tensors in decoder ONNX, "
                f"found {len(matmul_names)}"
            )
            if strict_shapes:
                raise KeyError(message)
            skipped.append(message)
            return loaded, skipped
        matmul_names = sorted(matmul_names, key=lambda n: int(n.split("_")[-1]))[:7]

        bias_names = [
            "module.decoders.g1_dyn.module.0.bias",
            "module.decoders.g1_dyn.module.2.bias",
            "module.decoders.g1_dyn.module.4.bias",
            "module.decoders.g1_dyn.module.6.bias",
            "module.decoders.g1_dyn.module.8.bias",
            "module.decoders.g1_dyn.module.10.bias",
            "module.decoders.g1_dyn.module.12.bias",
        ]

        for idx, (matmul_name, bias_name) in enumerate(zip(matmul_names, bias_names)):
            if bias_name not in initializers:
                message = f"control_decoder: missing source bias tensor {bias_name}"
                if strict_shapes:
                    raise KeyError(message)
                skipped.append(message)
                continue

            success, reason = self._copy_linear_params(
                target_linears[idx],
                initializers[matmul_name],
                initializers[bias_name],
                source_name=f"control_decoder.layer{idx}",
                strict_shapes=strict_shapes,
                transpose_weight=True,
            )
            if success:
                loaded += 1
            elif reason is not None:
                skipped.append(reason)

        return loaded, skipped

    def load_weights_from_sonic_onnx(
        self,
        encoder_onnx_path: str | None = None,
        decoder_onnx_path: str | None = None,
        *,
        load_robot_encoder: bool = True,
        load_human_encoder: bool = True,
        load_hybrid_encoder: bool = True,
        load_control_decoder: bool = True,
        strict_shapes: bool = False,
    ) -> dict[str, Any]:
        """Load compatible pretrained weights from official SONIC ONNX files.

        Supports partial loading for:
        - Encoder branches: g1 -> robot_encoder, teleop -> human_encoder, smpl -> hybrid_encoder
        - Decoder policy head: g1_dyn decoder -> control_decoder

        Notes:
        - `motion_decoder`, `critic`, and `std` are not present in official deploy ONNX and are not loaded.
        - Decoder MatMul weights in ONNX are transposed before assignment to `nn.Linear.weight`.
        """
        if encoder_onnx_path is None and decoder_onnx_path is None:
            raise ValueError("At least one of `encoder_onnx_path` or `decoder_onnx_path` must be provided.")

        summary: dict[str, Any] = {
            "loaded": {
                "robot_encoder": 0,
                "human_encoder": 0,
                "hybrid_encoder": 0,
                "control_decoder": 0,
            },
            "skipped": [],
        }

        if encoder_onnx_path is not None:
            initializers = self._load_onnx_initializers(encoder_onnx_path)
            if load_robot_encoder:
                num_loaded, skipped = self._load_encoder_branch_from_onnx(
                    initializers,
                    source_prefix="module.encoders.g1",
                    target_encoder=self.robot_encoder,
                    target_name="robot_encoder",
                    strict_shapes=strict_shapes,
                )
                summary["loaded"]["robot_encoder"] += num_loaded
                summary["skipped"].extend(skipped)
            if load_human_encoder:
                num_loaded, skipped = self._load_encoder_branch_from_onnx(
                    initializers,
                    source_prefix="module.encoders.teleop",
                    target_encoder=self.human_encoder,
                    target_name="human_encoder",
                    strict_shapes=strict_shapes,
                )
                summary["loaded"]["human_encoder"] += num_loaded
                summary["skipped"].extend(skipped)
            if load_hybrid_encoder:
                num_loaded, skipped = self._load_encoder_branch_from_onnx(
                    initializers,
                    source_prefix="module.encoders.smpl",
                    target_encoder=self.hybrid_encoder,
                    target_name="hybrid_encoder",
                    strict_shapes=strict_shapes,
                )
                summary["loaded"]["hybrid_encoder"] += num_loaded
                summary["skipped"].extend(skipped)

        if decoder_onnx_path is not None and load_control_decoder:
            initializers = self._load_onnx_initializers(decoder_onnx_path)
            num_loaded, skipped = self._load_control_decoder_from_onnx(initializers, strict_shapes=strict_shapes)
            summary["loaded"]["control_decoder"] += num_loaded
            summary["skipped"].extend(skipped)

        return summary

    def _get_action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(self.std).all():
            with torch.no_grad():
                cleaned = torch.nan_to_num(
                    self.std.data,
                    nan=self.min_action_std,
                    posinf=self.max_action_std,
                    neginf=self.min_action_std,
                )
                self.std.data.copy_(torch.clamp(cleaned, min=self.min_action_std, max=self.max_action_std))
            print(
                "[SONIC][WARN] Non-finite action std parameter detected and reset to "
                f"[{self.min_action_std}, {self.max_action_std}] range."
            )

        std = torch.clamp(self.std, min=self.min_action_std, max=self.max_action_std)
        return std.expand_as(mean)

    def quantize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.token_total_dim:
            raise ValueError(
                f"SONIC latent dim {latent.shape[-1]} must match token_total_dim {self.token_total_dim} "
                f"({self.max_num_tokens} tokens x {self.token_dim} dims)."
            )
        token_shape = latent.shape[:-1] + (self.max_num_tokens, self.token_dim)
        z = latent.reshape(token_shape)
        z_q = self.quantizer.quantize(z)
        return z_q.reshape(latent.shape[:-1] + (self.token_total_dim,))

    def _prepare_robot_encoder_input(self, robot_motion: torch.Tensor) -> torch.Tensor:
        if self.robot_encoder_layout != "g1_onnx_repack":
            return robot_motion
        if robot_motion.ndim != 2:
            raise ValueError(
                "robot_encoder_layout='g1_onnx_repack' expects 2D robot motion [batch, 640], "
                f"got shape {tuple(robot_motion.shape)}."
            )
        if robot_motion.shape[-1] != 640:
            raise ValueError(
                "robot_encoder_layout='g1_onnx_repack' expects robot motion dim 640, "
                f"got {robot_motion.shape[-1]}."
            )

        # Match deploy encoder graph:
        # A = x[:580] -> reshape [10,58], B = x[580:640] -> reshape [10,6],
        # concat on last dim -> [10,64], flatten -> [640].
        pos_vel = robot_motion[:, :580].reshape(robot_motion.shape[0], 10, 58)
        anchor = robot_motion[:, 580:640].reshape(robot_motion.shape[0], 10, 6)
        return torch.cat([pos_vel, anchor], dim=-1).reshape(robot_motion.shape[0], 640)

    def encode_robot_pre_quant(self, robot_motion: torch.Tensor) -> torch.Tensor:
        robot_motion = self._prepare_robot_encoder_input(robot_motion)
        z = self.robot_encoder(robot_motion)
        z = self.robot_encoder_proj(z)
        return z

    def encode_robot(self, robot_motion: torch.Tensor) -> torch.Tensor:
        return self.quantize_latent(self.encode_robot_pre_quant(robot_motion))

    def encode_human_pre_quant(self, human_motion: torch.Tensor) -> torch.Tensor:
        z = self.human_encoder(human_motion)
        z = self.human_encoder_proj(z)
        return z

    def encode_human(self, human_motion: torch.Tensor) -> torch.Tensor:
        return self.quantize_latent(self.encode_human_pre_quant(human_motion))

    def encode_hybrid_pre_quant(self, hybrid_motion: torch.Tensor) -> torch.Tensor:
        if self.hybrid_encoder is None or self.hybrid_encoder_proj is None:
            raise ValueError("Hybrid encoder is not initialized.")
        z = self.hybrid_encoder(hybrid_motion)
        z = self.hybrid_encoder_proj(z)
        return z

    def encode_hybrid(self, hybrid_motion: torch.Tensor) -> torch.Tensor:
        return self.quantize_latent(self.encode_hybrid_pre_quant(hybrid_motion))

    def compute_sonic_aux_losses(self, obs: TensorDict) -> dict[str, Any]:
        """Compute SONIC universal-token aux losses in PMT observation space."""
        train_cfg = self.get_sonic_encoder_train_cfg()
        if "robot_encoder" not in obs.keys():
            raise KeyError("SONIC auxiliary losses require observation key 'robot_encoder'.")

        aux_losses: dict[str, torch.Tensor] = {}
        encoded_latents: dict[str, torch.Tensor] = {}
        encoded_tokens: dict[str, torch.Tensor] = {}
        decoded_outputs: dict[str, torch.Tensor] = {}

        g1_motion = obs["robot_encoder"]
        g1_latent = self.encode_robot_pre_quant(g1_motion)
        g1_token = self.quantize_latent(g1_latent)
        g1_recon = self.motion_decoder(g1_token)
        # Official SONIC reconstructs FK-space g1_kin. PMT currently exposes the
        # 580D robot motion-command vector here, which is the decoder output space.
        aux_losses["g1_recon"] = F.mse_loss(g1_recon, g1_motion)
        encoded_latents["g1"] = g1_latent
        encoded_tokens["g1"] = g1_token
        decoded_outputs["g1_recon"] = g1_recon

        smpl_latent: torch.Tensor | None = None
        if train_cfg.get("human", False):
            if "human_encoder" not in obs.keys():
                raise KeyError("train_human_encoder=True requires observation key 'human_encoder'.")
            smpl_motion = obs["human_encoder"]
            smpl_latent = self.encode_human_pre_quant(smpl_motion)
            smpl_token = self.quantize_latent(smpl_latent)
            aux_losses["g1_smpl_latent"] = F.mse_loss(g1_latent, smpl_latent)
            encoded_latents["smpl"] = smpl_latent
            encoded_tokens["smpl"] = smpl_token

            if self.reencode_smpl_g1_recon:
                smpl_g1_recon = self.motion_decoder(smpl_token)
                reencoded_smpl_g1_latent = self.encode_robot_pre_quant(smpl_g1_recon)
                aux_losses["reencoded_smpl_g1_latent"] = F.mse_loss(reencoded_smpl_g1_latent, g1_latent)
                decoded_outputs["smpl_g1_recon"] = smpl_g1_recon

        if train_cfg.get("hybrid", False):
            if "hybrid_encoder" not in obs.keys():
                raise KeyError("train_hybrid_encoder=True requires observation key 'hybrid_encoder'.")
            if self.hybrid_encoder is None or self.hybrid_encoder_proj is None:
                raise ValueError("Hybrid encoder is not initialized.")
            teleop_motion = obs["hybrid_encoder"]
            teleop_latent = self.encode_hybrid_pre_quant(teleop_motion)
            teleop_token = self.quantize_latent(teleop_latent)
            aux_losses["g1_teleop_latent"] = F.mse_loss(g1_latent, teleop_latent)
            if smpl_latent is not None:
                aux_losses["teleop_smpl_latent"] = F.mse_loss(teleop_latent, smpl_latent)
            encoded_latents["teleop"] = teleop_latent
            encoded_tokens["teleop"] = teleop_token

        return {
            "aux_losses": aux_losses,
            "aux_loss_coef": dict(self.aux_loss_coef),
            "encoded_latents": encoded_latents,
            "encoded_tokens": encoded_tokens,
            "decoded_outputs": decoded_outputs,
        }

    def get_sonic_aux_losses(self, obs: TensorDict) -> dict[str, Any]:
        return self.compute_sonic_aux_losses(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[k] for k in self.obs_groups["policy"]]
        actor_obs = torch.cat(obs_list, dim=-1)
        return self._prepare_decoder_proprio(actor_obs)

    def _prepare_decoder_proprio(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Prepare actor proprio input for SONIC decoder compatibility."""
        if self.decoder_proprio_layout != "interleaved_step_history":
            return actor_obs

        if actor_obs.ndim != 2:
            raise ValueError(
                "decoder_proprio_layout='interleaved_step_history' expects 2D actor obs "
                f"[batch, 930], got shape {tuple(actor_obs.shape)}."
            )
        if actor_obs.shape[-1] != 930:
            raise ValueError(
                "decoder_proprio_layout='interleaved_step_history' expects actor obs dim 930, "
                f"got {actor_obs.shape[-1]}."
            )

        # Convert from frame-interleaved [f0(step93), ..., f9(step93)] into
        # deploy decoder grouped order:
        # [his_base_angular_velocity(30), his_body_joint_positions(290),
        #  his_body_joint_velocities(290), his_last_actions(290), his_gravity_dir(30)].
        hist = actor_obs.reshape(actor_obs.shape[0], 10, 93)
        omega = hist[:, :, 0:3].reshape(actor_obs.shape[0], -1)
        q = hist[:, :, 3:32].reshape(actor_obs.shape[0], -1)
        dq = hist[:, :, 32:61].reshape(actor_obs.shape[0], -1)
        a = hist[:, :, 61:90].reshape(actor_obs.shape[0], -1)
        g = hist[:, :, 90:93].reshape(actor_obs.shape[0], -1)
        return torch.cat([omega, q, dq, a, g], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[k] for k in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def build_control_decoder_input(self, proprio: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        if self.control_decoder_input_order == "token_proprio":
            return torch.cat([token, proprio], dim=-1)
        return torch.cat([proprio, token], dim=-1)

    def _encode_from_source(self, obs: TensorDict, source: str) -> torch.Tensor:
        source = source.lower()
        if source == "robot":
            if "robot_encoder" not in obs.keys():
                raise KeyError("Requested robot encoder input, but 'robot_encoder' is missing in observations.")
            return self.encode_robot(obs["robot_encoder"])
        if source == "human":
            if "human_encoder" not in obs.keys():
                raise KeyError("Requested human encoder input, but 'human_encoder' is missing in observations.")
            return self.encode_human(obs["human_encoder"])
        if source == "hybrid":
            if "hybrid_encoder" not in obs.keys():
                raise KeyError("Requested hybrid encoder input, but 'hybrid_encoder' is missing in observations.")
            return self.encode_hybrid(obs["hybrid_encoder"])
        if source == "mode":
            return self._encode_from_mode(obs)
        raise ValueError(f"Unknown source: {source}")

    def _get_encoder_mode_ids(self, obs: TensorDict) -> torch.Tensor:
        if self.encoder_mode_key not in obs.keys():
            raise KeyError(
                f"Encoder mode observation '{self.encoder_mode_key}' is missing. "
                f"Available observation keys: {list(obs.keys())}"
            )
        mode_obs = obs[self.encoder_mode_key]
        if mode_obs.shape[-1] < 1:
            raise ValueError(
                f"Encoder mode observation '{self.encoder_mode_key}' must have at least 1 value per env."
            )
        mode_ids = mode_obs[..., 0]
        if torch.is_floating_point(mode_ids):
            mode_ids = torch.round(mode_ids)
        mode_ids = mode_ids.long()

        valid = (mode_ids >= 0) & (mode_ids <= 2)
        if not torch.all(valid):
            invalid_modes = torch.unique(mode_ids[~valid]).tolist()
            raise ValueError(
                f"Invalid encoder mode id(s): {invalid_modes}. Expected only 0 (g1), 1 (teleop), 2 (smpl)."
            )
        return mode_ids

    def _encode_from_mode(self, obs: TensorDict) -> torch.Tensor:
        mode_ids = self._get_encoder_mode_ids(obs)
        batch_size = mode_ids.shape[0]
        token = None

        # 0 -> g1 branch (robot_encoder)
        g1_mask = mode_ids == 0
        if torch.any(g1_mask):
            if not self.train_robot_encoder:
                raise ValueError(
                    "Mode 0 (g1) requested, but robot encoder is disabled/untrained "
                    "(`train_robot_encoder=False`)."
                )
            if "robot_encoder" not in obs.keys():
                raise KeyError("Mode 0 (g1) requested, but 'robot_encoder' observations are missing.")
            z_g1 = self.encode_robot(obs["robot_encoder"][g1_mask])
            token = z_g1.new_zeros((batch_size, z_g1.shape[-1]))
            token[g1_mask] = z_g1

        # 1 -> teleop branch (human_encoder)
        teleop_mask = mode_ids == 1
        if torch.any(teleop_mask):
            if not self.train_human_encoder:
                raise ValueError(
                    "Mode 1 (teleop) requested, but human encoder is disabled/untrained "
                    "(`train_human_encoder=False`)."
                )
            if "human_encoder" not in obs.keys():
                raise KeyError("Mode 1 (teleop) requested, but 'human_encoder' observations are missing.")
            z_teleop = self.encode_human(obs["human_encoder"][teleop_mask])
            if token is None:
                token = z_teleop.new_zeros((batch_size, z_teleop.shape[-1]))
            token[teleop_mask] = z_teleop

        # 2 -> smpl branch (hybrid_encoder)
        smpl_mask = mode_ids == 2
        if torch.any(smpl_mask):
            if not self.train_hybrid_encoder:
                raise ValueError(
                    "Mode 2 (smpl) requested, but hybrid encoder is disabled/untrained "
                    "(`train_hybrid_encoder=False`)."
                )
            if "hybrid_encoder" not in obs.keys():
                raise KeyError("Mode 2 (smpl) requested, but 'hybrid_encoder' observations are missing.")
            z_smpl = self.encode_hybrid(obs["hybrid_encoder"][smpl_mask])
            if token is None:
                token = z_smpl.new_zeros((batch_size, z_smpl.shape[-1]))
            token[smpl_mask] = z_smpl

        if token is None:
            raise ValueError("Failed to produce encoder token from mode routing.")
        return token

    def _select_action_token(self, obs: TensorDict) -> torch.Tensor:
        if self.action_encoder_source == "mode":
            return self._encode_from_mode(obs)
        if self.action_encoder_source != "auto":
            return self._encode_from_source(obs, self.action_encoder_source)

        candidates: list[tuple[str, bool]] = [
            ("robot", self.train_robot_encoder),
            ("human", self.train_human_encoder),
            ("hybrid", self.train_hybrid_encoder),
        ]
        for source, enabled in candidates:
            if not enabled:
                continue
            key = f"{source}_encoder"
            if key in obs.keys():
                return self._encode_from_source(obs, source)

        enabled_sources = [source for source, enabled in candidates if enabled]
        raise ValueError(
            "No motion input found for enabled encoders. "
            f"Enabled encoders: {enabled_sources}. "
            f"Available observation keys: {list(obs.keys())}"
        )

    def get_token(self, obs: TensorDict, source: str = "robot") -> torch.Tensor:
        if source in ("auto", "mode"):
            return self._select_action_token(obs)
        return self._encode_from_source(obs, source)

    def _prepare_action_token_for_policy(self, token: torch.Tensor) -> torch.Tensor:
        # SONIC paper-style training: PPO/control path uses detached token,
        # while encoder training is driven by auxiliary SONIC losses.
        if self.detach_action_token:
            return token.detach()
        return token

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        # Inference / Rollout
        # Get Proprioception
        proprio = self.get_actor_obs(obs)
        proprio = self.actor_obs_normalizer(proprio)

        # Get token from configured source.
        z = self._select_action_token(obs)
        z = self._prepare_action_token_for_policy(z)

        # Control Decoder
        policy_input = self.build_control_decoder_input(proprio, z)
        mean = self.control_decoder(policy_input)

        # Std
        std = self._get_action_std(mean)
        self.distribution = Normal(mean, std)

        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Inference only action (returns mean)."""
        proprio = self.get_actor_obs(obs)
        proprio = self.actor_obs_normalizer(proprio)
        z = self._select_action_token(obs)
        z = self._prepare_action_token_for_policy(z)

        policy_input = self.build_control_decoder_input(proprio, z)
        mean = self.control_decoder(policy_input)

        # Populate distribution for potential log_prob calls (e.g. in symmetry loss)
        std = self._get_action_std(mean)
        self.distribution = Normal(mean, std)

        return mean

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        pass

    def get_sonic_encoder_train_cfg(self) -> dict[str, bool]:
        return {
            "robot": self.train_robot_encoder,
            "human": self.train_human_encoder,
            "hybrid": self.train_hybrid_encoder,
        }

    def set_warmup_freeze(
        self,
        *,
        freeze_encoders: bool,
        freeze_control_decoder: bool,
        freeze_action_std: bool,
    ) -> dict[str, bool]:
        """Temporarily freeze/unfreeze SONIC policy components for warmup training."""
        freeze_encoders = bool(freeze_encoders)
        freeze_control_decoder = bool(freeze_control_decoder)
        freeze_action_std = bool(freeze_action_std)

        if freeze_encoders:
            self._set_trainable(self.robot_encoder, False)
            self._set_trainable(self.robot_encoder_proj, False)
            self._set_trainable(self.human_encoder, False)
            self._set_trainable(self.human_encoder_proj, False)
            self._set_trainable(self.hybrid_encoder, False)
            self._set_trainable(self.hybrid_encoder_proj, False)
        else:
            # Restore configured trainability for each encoder branch.
            self._set_trainable(self.robot_encoder, self.train_robot_encoder)
            self._set_trainable(self.robot_encoder_proj, self.train_robot_encoder)
            self._set_trainable(self.human_encoder, self.train_human_encoder)
            self._set_trainable(self.human_encoder_proj, self.train_human_encoder)
            self._set_trainable(self.hybrid_encoder, self.train_hybrid_encoder)
            self._set_trainable(self.hybrid_encoder_proj, self.train_hybrid_encoder)

        self._set_trainable(self.control_decoder, not freeze_control_decoder)
        self.std.requires_grad_(not freeze_action_std)

        self._warmup_freeze_state = {
            "encoders_frozen": freeze_encoders,
            "control_decoder_frozen": freeze_control_decoder,
            "action_std_frozen": freeze_action_std,
        }
        return dict(self._warmup_freeze_state)

    def get_warmup_freeze_state(self) -> dict[str, bool]:
        return dict(self._warmup_freeze_state)

    def update_normalization(self, obs):
        if self.actor_obs_normalization and self.proprio_dim > 0:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization and self.num_critic_obs > 0:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters and indicate training resume status."""
        super().load_state_dict(state_dict, strict=strict)
        return True


@register_network("ActorCritic", compat_name="mlp")
class ActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        # Forward/residual network parameters
        forward_hidden_dims: tuple[int] | list[int] | None = None,
        forward_obs_normalization: bool | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        # Check if forward/residual architecture is enabled
        self.use_forward = (
            forward_hidden_dims is not None
            and "forward" in obs_groups
            and len(obs_groups.get("forward", [])) > 0
        )

        if self.use_forward:
            # Calculate forward observation dimensions
            num_forward_obs = 0
            for obs_group in obs_groups["forward"]:
                assert len(obs[obs_group].shape) == 2, "Forward observations must be 1D."
                num_forward_obs += obs[obs_group].shape[-1]
            self.num_forward_obs = num_forward_obs

            # Actor becomes feature extractor
            # Output: actor_hidden_dims[-1], Hidden: actor_hidden_dims[:-1]
            actor_output_dim = actor_hidden_dims[-1]
            actor_hidden = list(actor_hidden_dims[:-1]) if len(actor_hidden_dims) > 1 else []

            # Actor outputs features (no state_dependent_std in actor, moved to forward)
            self.actor = MLP(num_actor_obs, actor_output_dim, actor_hidden, activation)
            print(f"Actor MLP (feature extractor): {self.actor}")

            # Forward network: concat(actor_output, forward_obs) -> action
            if self.state_dependent_std:
                self.forward_net = MLP(
                    actor_output_dim + num_forward_obs,
                    [2, num_actions],
                    list(forward_hidden_dims),
                    activation
                )
            else:
                self.forward_net = MLP(
                    actor_output_dim + num_forward_obs,
                    num_actions,
                    list(forward_hidden_dims),
                    activation
                )
            print(f"Forward MLP: {self.forward_net}")

            # Forward observation normalization
            forward_norm = forward_obs_normalization if forward_obs_normalization is not None else actor_obs_normalization
            self.forward_obs_normalization = forward_norm
            if forward_norm:
                self.forward_obs_normalizer = EmpiricalNormalization(num_forward_obs)
            else:
                self.forward_obs_normalizer = torch.nn.Identity()
        else:
            # Original: Actor directly outputs actions
            self.forward_net = None
            self.forward_obs_normalizer = None
            self.forward_obs_normalization = False

            if self.state_dependent_std:
                self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
            else:
                self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
            print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        if self.state_dependent_std:
            # Initialize std output layer (in forward_net if using forward, else in actor)
            if self.use_forward:
                torch.nn.init.zeros_(self.forward_net[-2].weight[num_actions:])
                if self.noise_std_type == "scalar":
                    torch.nn.init.constant_(self.forward_net[-2].bias[num_actions:], init_noise_std)
                elif self.noise_std_type == "log":
                    torch.nn.init.constant_(
                        self.forward_net[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                    )
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            else:
                torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
                if self.noise_std_type == "scalar":
                    torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
                elif self.noise_std_type == "log":
                    torch.nn.init.constant_(
                        self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                    )
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

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

    def _update_distribution(self, obs: TensorDict) -> None:
        # Get actor observations
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        if self.use_forward:
            # Actor outputs features
            features = self.actor(actor_obs)

            # Get forward observations
            forward_obs = self.get_forward_obs(obs)
            forward_obs = self.forward_obs_normalizer(forward_obs)

            # Concatenate and pass through forward network
            forward_input = torch.cat([features, forward_obs], dim=-1)

            if self.state_dependent_std:
                mean_and_std = self.forward_net(forward_input)
                if self.noise_std_type == "scalar":
                    mean, std = torch.unbind(mean_and_std, dim=-2)
                elif self.noise_std_type == "log":
                    mean, log_std = torch.unbind(mean_and_std, dim=-2)
                    std = torch.exp(log_std)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            else:
                mean = self.forward_net(forward_input)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Original: Actor directly outputs actions
            if self.state_dependent_std:
                mean_and_std = self.actor(actor_obs)
                if self.noise_std_type == "scalar":
                    mean, std = torch.unbind(mean_and_std, dim=-2)
                elif self.noise_std_type == "log":
                    mean, log_std = torch.unbind(mean_and_std, dim=-2)
                    std = torch.exp(log_std)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            else:
                mean = self.actor(actor_obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Guard the action distribution: a transient bad gradient (e.g. from a NaN/Inf
        # observation early in training) can drive the scalar std parameter negative,
        # making Normal() raise "normal expects all elements of std >= 0.0" and killing
        # the run on the first update (seen on cube-interaction obs).
        # Clamp std strictly positive and sanitize NaN/Inf mean — the same defensive
        # posture SonicActorCritic already applies. Harmless for healthy training (std
        # stays well above the floor); lets PPO ride out a transient spike, not crash.
        std = torch.nan_to_num(std, nan=1e-3, posinf=1.0, neginf=1e-3).clamp_min(1e-6)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        if self.use_forward:
            features = self.actor(actor_obs)
            forward_obs = self.get_forward_obs(obs)
            forward_obs = self.forward_obs_normalizer(forward_obs)
            forward_input = torch.cat([features, forward_obs], dim=-1)
            if self.state_dependent_std:
                return self.forward_net(forward_input)[..., 0, :]
            return self.forward_net(forward_input)
        else:
            if self.state_dependent_std:
                return self.actor(actor_obs)[..., 0, :]
            return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_forward_obs(self, obs: TensorDict) -> torch.Tensor:
        """Get concatenated forward observations."""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["forward"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
        if self.use_forward and self.forward_obs_normalization:
            forward_obs = self.get_forward_obs(obs)
            self.forward_obs_normalizer.update(forward_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
