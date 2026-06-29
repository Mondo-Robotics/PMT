from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.networks.residual_vision_action import (
    GatedResidual,
    MapTransformer,
    TrainingStage,
    zero_init_last_linear,
)
from motion_tracking_rl.networks.transformer_actor_critic import TransformerActorCritic
from motion_tracking_rl.networks.layers import GRU_SRU, LSTM_SRU, MLP
from motion_tracking_rl.registry import register_network
from motion_tracking_rl.utils import resolve_nn_activation


class PlainResidual(nn.Module):
    """Residual add without a learned identity gate."""

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + residual


@register_network("VisionTransformerActorCritic", compat_name="vision_transformer")
class VisionTransformerActorCritic(TransformerActorCritic):
    """transformer actor-critic with a terrain-vision correction branch.

    The blind transformer backbone remains intact:
      - policy obs -> actor trunk
      - proprio history -> history encoder
      - command window -> command encoder

    Vision is merged as a correction signal rather than replacing the transformer prior:
      1. a configurable terrain encoder produces a vision latent
      2. the vision latent modulates the command latent
      3. a small residual head adds terrain-specific action corrections
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        map_height: int = 17,
        map_width: int = 11,
        map_resolution: float = 0.1,
        dim_map_embed: int = 64,
        num_attn_heads: int = 4,
        z_clip: float = 3.0,
        normalize_height: bool = True,
        vision_encoder_type: str = "height_map",
        depth_cnn_channels: tuple[int, ...] | list[int] = (16, 32),
        depth_cnn_kernel_sizes: tuple[int, ...] | list[int] = (5, 3),
        depth_cnn_strides: tuple[int, ...] | list[int] = (2, 2),
        depth_frame_feature_dim: int = 64,
        depth_sru_type: str = "lstm_sru",
        depth_sru_num_layers: int = 1,
        critic_use_vision: bool = True,
        use_foot_traj_head: bool = False,
        foot_traj_output_dim: int | None = None,
        foot_traj_target_obs_key: str | None = None,
        foot_traj_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        foot_traj_use_vision: bool = True,
        training_stage: str = "finetune_all",
        base_policy_ckpt: str | None = None,
        allow_partial_base_policy_transfer: bool = False,
        freeze_std_in_adapter: bool = False,
        use_action_residual: bool = True,
        use_map_proprio_cross_attention: bool = False,
        use_identity_gates: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        try:
            self.stage = TrainingStage[training_stage.upper()]
        except KeyError as exc:
            valid = [stage.name.lower() for stage in TrainingStage]
            raise ValueError(f"Invalid training_stage '{training_stage}'. Valid values: {valid}") from exc

        self.use_vision = self.stage != TrainingStage.BASE_ONLY
        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.map_resolution = float(map_resolution)
        self.dim_map_embed = int(dim_map_embed)
        self.z_clip = float(z_clip)
        self.normalize_height = bool(normalize_height)
        self.vision_encoder_type = str(vision_encoder_type).lower()
        if self.vision_encoder_type not in {"height_map", "depth_sru"}:
            raise ValueError(
                f"Invalid vision_encoder_type='{vision_encoder_type}'. "
                "Valid options are: ['height_map', 'depth_sru']."
            )
        self.depth_cnn_channels = tuple(int(channel) for channel in depth_cnn_channels)
        self.depth_cnn_kernel_sizes = tuple(int(kernel) for kernel in depth_cnn_kernel_sizes)
        self.depth_cnn_strides = tuple(int(stride) for stride in depth_cnn_strides)
        self.depth_frame_feature_dim = int(depth_frame_feature_dim)
        self.depth_sru_type = str(depth_sru_type).lower()
        self.depth_sru_num_layers = int(depth_sru_num_layers)
        self.critic_use_vision = bool(critic_use_vision) and self.use_vision
        self.use_foot_traj_head = bool(use_foot_traj_head)
        self.foot_traj_use_vision = bool(foot_traj_use_vision) and self.use_vision
        self.allow_partial_base_policy_transfer = bool(allow_partial_base_policy_transfer)
        self.freeze_std_in_adapter = bool(freeze_std_in_adapter)
        self.use_action_residual = bool(use_action_residual)
        self.use_map_proprio_cross_attention = bool(use_map_proprio_cross_attention)
        self.use_identity_gates = bool(use_identity_gates)
        self.foot_traj_target_obs_key = foot_traj_target_obs_key
        self.foot_traj_output_dim = 0 if foot_traj_output_dim is None else int(foot_traj_output_dim)
        if self.use_foot_traj_head and self.foot_traj_output_dim <= 0:
            if self.foot_traj_target_obs_key is None:
                raise ValueError(
                    "VisionTransformerActorCritic requires foot_traj_output_dim > 0 or "
                    "foot_traj_target_obs_key when use_foot_traj_head=True."
                )
            if self.foot_traj_target_obs_key not in obs.keys():
                raise KeyError(
                    f"foot_traj_target_obs_key='{self.foot_traj_target_obs_key}' not found in observations. "
                    f"Available groups: {list(obs.keys())}"
                )
            self.foot_traj_output_dim = int(obs[self.foot_traj_target_obs_key].shape[-1])

        # Cache dimensions before the parent constructor normalizes/reshapes inputs.
        self._policy_obs_dim = sum(int(obs[group].shape[-1]) for group in obs_groups.get("policy", []))
        self._critic_obs_dim = sum(int(obs[group].shape[-1]) for group in obs_groups.get("critic", []))
        self._n_embd = int(kwargs.get("n_embd", 128))
        critic_hidden_dims = list(kwargs.get("critic_hidden_dims", (512, 256)))
        activation = str(kwargs.get("activation", "elu"))

        super().__init__(obs=obs, obs_groups=obs_groups, num_actions=num_actions, **kwargs)
        if "vision" not in obs.keys():
            raise KeyError("VisionTransformerActorCritic requires observation key 'vision'.")
        self._height_vision_has_mask = False
        self._depth_vision_channels = 0
        if self.vision_encoder_type == "height_map":
            vision_obs = obs["vision"]
            expected_shape = self.map_height * self.map_width
            if vision_obs.ndim != 2:
                raise ValueError(
                    "Height-map vision observations must be shaped [B, H*W] or [B, 2*H*W]. "
                    f"Got {tuple(vision_obs.shape)}."
                )
            if int(vision_obs.shape[-1]) == expected_shape:
                self._height_vision_has_mask = False
            elif int(vision_obs.shape[-1]) == 2 * expected_shape:
                self._height_vision_has_mask = True
            else:
                raise ValueError(
                    "Height-map vision observations must be shaped [B, H*W] or [B, 2*H*W] with "
                    f"H*W={expected_shape}. Got {tuple(vision_obs.shape)}."
                )
        else:
            vision_obs = obs["vision"]
            if vision_obs.ndim not in (4, 5):
                raise ValueError(
                    "Depth-SRU vision observations must be shaped [B, C, H, W] or [B, T, C, H, W]. "
                    f"Got {tuple(vision_obs.shape)}."
                )

        self.map_transformer: MapTransformer | None = None
        self.depth_frame_encoder: nn.Module | None = None
        self.depth_pool: nn.Module | None = None
        self.depth_frame_proj: nn.Module | None = None
        self.depth_frame_norm: nn.Module | None = None
        self.depth_memory: nn.Module | None = None
        self.depth_out_norm: nn.Module | None = None
        self.map_proprio_policy_norm: nn.Module | None = None
        self.map_proprio_policy_proj: nn.Module | None = None
        self.map_proprio_query_norm: nn.Module | None = None
        self.map_proprio_cross_attn: nn.Module | None = None
        self.map_proprio_out_proj: nn.Module | None = None
        self.map_proprio_gate: nn.Module | None = None

        if self.vision_encoder_type == "height_map":
            xs = (torch.arange(self.map_height, dtype=torch.float32) - (self.map_height - 1) / 2.0) * self.map_resolution
            ys = (torch.arange(self.map_width, dtype=torch.float32) - (self.map_width - 1) / 2.0) * self.map_resolution
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
            self.register_buffer("_grid_x", grid_x.clone(), persistent=False)
            self.register_buffer("_grid_y", grid_y.clone(), persistent=False)

            self.map_transformer = MapTransformer(
                dim_proprio=self._policy_obs_dim,
                dim_intent=self._n_embd,
                dim_map_embed=self.dim_map_embed,
                num_heads=num_attn_heads,
                terrain_input_channels=2 if self._height_vision_has_mask else 1,
                map_coord_channels=4 if self._height_vision_has_mask else 3,
            )
        else:
            self.register_buffer("_grid_x", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer("_grid_y", torch.empty(0, dtype=torch.float32), persistent=False)

            depth_in_channels = self._infer_depth_channels(obs)
            self._depth_vision_channels = int(depth_in_channels)
            self.depth_frame_encoder, depth_encoder_out_dim = self._build_depth_frame_encoder(
                in_channels=depth_in_channels,
                channels=self.depth_cnn_channels,
                kernel_sizes=self.depth_cnn_kernel_sizes,
                strides=self.depth_cnn_strides,
                activation=activation,
            )
            self.depth_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.depth_frame_proj = MLP(
                depth_encoder_out_dim,
                self.depth_frame_feature_dim,
                [max(depth_encoder_out_dim, self.depth_frame_feature_dim)],
                activation,
            )
            self.depth_frame_norm = nn.LayerNorm(self.depth_frame_feature_dim)
            if self.depth_sru_type == "lstm_sru":
                self.depth_memory = LSTM_SRU(
                    self.depth_frame_feature_dim,
                    self.dim_map_embed,
                    num_layers=self.depth_sru_num_layers,
                    batch_first=True,
                )
            elif self.depth_sru_type == "gru_sru":
                self.depth_memory = GRU_SRU(
                    self.depth_frame_feature_dim,
                    self.dim_map_embed,
                    num_layers=self.depth_sru_num_layers,
                    batch_first=True,
                )
            else:
                raise ValueError(
                    f"Invalid depth_sru_type='{depth_sru_type}'. Valid options are: ['lstm_sru', 'gru_sru']."
                )
            self.depth_out_norm = nn.LayerNorm(self.dim_map_embed)

        if self.use_map_proprio_cross_attention:
            if self._policy_obs_dim <= 0:
                raise ValueError("use_map_proprio_cross_attention=True requires a non-empty policy observation.")
            self.map_proprio_policy_norm = nn.LayerNorm(self._policy_obs_dim)
            self.map_proprio_policy_proj = nn.Linear(self._policy_obs_dim, self.dim_map_embed)
            self.map_proprio_query_norm = nn.LayerNorm(self.dim_map_embed)
            self.map_proprio_cross_attn = nn.MultiheadAttention(
                self.dim_map_embed,
                num_attn_heads,
                batch_first=True,
            )
            self.map_proprio_out_proj = nn.Linear(self.dim_map_embed, self.dim_map_embed)
            self.map_proprio_gate = GatedResidual(self.dim_map_embed) if self.use_identity_gates else PlainResidual()
            zero_init_last_linear(self.map_proprio_out_proj)

        self.intent_modulator = MLP(self.dim_map_embed, self._n_embd, [256, 128], activation)
        self.intent_gate = GatedResidual(self._n_embd) if self.use_identity_gates else PlainResidual()
        if self.use_action_residual:
            self.residual_policy = MLP(self._policy_obs_dim + self.dim_map_embed, num_actions, [256, 128], activation)
            self.residual_gate = GatedResidual(num_actions) if self.use_identity_gates else PlainResidual()
        else:
            self.residual_policy = None
            self.residual_gate = None
        zero_init_last_linear(self.intent_modulator)
        if self.residual_policy is not None:
            zero_init_last_linear(self.residual_policy)
        if self.use_foot_traj_head:
            foot_traj_in_dim = self._n_embd + (self.dim_map_embed if self.foot_traj_use_vision else 0)
            self.foot_traj_head = MLP(
                foot_traj_in_dim,
                self.foot_traj_output_dim,
                list(foot_traj_hidden_dims),
                activation,
            )
            zero_init_last_linear(self.foot_traj_head)
        else:
            self.foot_traj_head = None

        if self.critic_use_vision:
            self.critic = MLP(self._critic_obs_dim + self.dim_map_embed, 1, critic_hidden_dims, activation)

        if base_policy_ckpt is not None:
            self._smart_load_checkpoint(base_policy_ckpt)
        elif self.stage == TrainingStage.VISION_ADAPTER:
            raise ValueError("VISION_ADAPTER requires base_policy_ckpt so the blind transformer backbone is not frozen at random.")

        self._configure_freezing()

    @staticmethod
    def _format_key_list(keys: list[str], max_items: int = 12) -> str:
        if not keys:
            return "none"
        display = keys[:max_items]
        suffix = "" if len(keys) <= max_items else f", ... (+{len(keys) - max_items} more)"
        return ", ".join(display) + suffix

    @staticmethod
    def _vision_transfer_prefixes() -> tuple[str, ...]:
        return (
            "map_transformer",
            "depth_frame_encoder",
            "depth_pool",
            "depth_frame_proj",
            "depth_frame_norm",
            "depth_memory",
            "depth_out_norm",
            "map_proprio_policy_norm",
            "map_proprio_policy_proj",
            "map_proprio_query_norm",
            "map_proprio_cross_attn",
            "map_proprio_out_proj",
            "map_proprio_gate",
            "intent_modulator",
            "intent_gate",
            "residual_policy",
            "residual_gate",
            "foot_traj_head",
        )

    @classmethod
    def _is_allowed_transfer_skip(cls, key: str, *, allow_vision_module_skip: bool) -> bool:
        allowed_prefixes = [
            "critic",
            "critic_obs_normalizer",
            "vel_head",
            "vel_gt_normalizer",
            "anchor_head",
            "anchor_gt_normalizer",
        ]
        if allow_vision_module_skip:
            allowed_prefixes.extend(cls._vision_transfer_prefixes())
        return key.startswith(tuple(allowed_prefixes))

    def _smart_load_checkpoint(self, path: str) -> None:
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"VisionTransformerActorCritic base checkpoint not found: {ckpt_path}")

        loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format in: {ckpt_path}")
        checkpoint_metadata = loaded.get("policy_metadata") if isinstance(loaded, dict) else None
        source_policy_family = checkpoint_metadata.get("policy_family") if isinstance(checkpoint_metadata, dict) else None

        # Allow loading from distillation checkpoints directly.
        if any(key.startswith("student.") for key in state_dict.keys()):
            state_dict = {
                key.replace("student.", "", 1): value for key, value in state_dict.items() if key.startswith("student.")
            }
        if isinstance(checkpoint_metadata, dict) and checkpoint_metadata.get("policy_family") == "vision_student_teacher":
            nested_student = checkpoint_metadata.get("subpolicies", {}).get("student")
            if nested_student is None:
                nested_student = checkpoint_metadata.get("signature", {}).get("student")
            if isinstance(nested_student, dict):
                checkpoint_metadata = nested_student
                if "signature" not in checkpoint_metadata:
                    checkpoint_metadata = {"signature": checkpoint_metadata}
            source_policy_family = checkpoint_metadata.get("policy_family") if isinstance(checkpoint_metadata, dict) else None

        source_has_vision_tensors = any(key.startswith(self._vision_transfer_prefixes()) for key in state_dict.keys())
        allow_vision_module_skip = not source_has_vision_tensors and source_policy_family != "vision_transformer_actor_critic"
        allow_partial_transfer = (
            self.allow_partial_base_policy_transfer
            and not source_has_vision_tensors
            and source_policy_family != "vision_transformer_actor_critic"
        )

        if isinstance(checkpoint_metadata, dict):
            checkpoint_signature = checkpoint_metadata.get("signature", {})
            current_signature = self.get_checkpoint_metadata().get("signature", {})
            signature_keys = (
                "num_actions",
                "history_len",
                "cmd_len",
                "history_token_dim",
                "cmd_token_dim",
                "actor_obs_dim",
                "anchor_estimator_obs_dim",
                "vision_encoder_type",
                "map_height",
                "map_width",
                "dim_map_embed",
                "height_vision_has_mask",
                "depth_vision_channels",
                "use_action_residual",
                "use_map_proprio_cross_attention",
            )
            signature_issues = [
                f"{key}: checkpoint={checkpoint_signature.get(key)} current={current_signature.get(key)}"
                for key in signature_keys
                if checkpoint_signature.get(key) is not None and checkpoint_signature.get(key) != current_signature.get(key)
            ]
            if signature_issues:
                if allow_partial_transfer:
                    print(
                        "[VisionTransformerActorCritic] Allowing partial blind-teacher transfer despite "
                        "signature mismatches: "
                        f"{self._format_key_list(signature_issues)}"
                    )
                else:
                    raise ValueError(
                        "Base checkpoint is incompatible with the current vision-transformer student. "
                        f"Signature mismatches: {self._format_key_list(signature_issues)}"
                    )
        else:
            print(
                f"[VisionTransformerActorCritic] Checkpoint {ckpt_path} has no policy metadata. "
                "Proceeding with tensor-shape auditing only."
            )

        current = self.state_dict()
        compatible = {}
        skipped = []
        critical_skipped = []
        for key, value in state_dict.items():
            if key in current and hasattr(value, "shape") and tuple(value.shape) == tuple(current[key].shape):
                compatible[key] = value
            elif key in current:
                skipped.append(key)
                if not self._is_allowed_transfer_skip(key, allow_vision_module_skip=allow_vision_module_skip):
                    critical_skipped.append(key)

        if critical_skipped:
            if allow_partial_transfer:
                print(
                    "[VisionTransformerActorCritic] Allowing partial blind-teacher transfer; "
                    "skipping critical mismatched tensors: "
                    f"{self._format_key_list(critical_skipped)}"
                )
            else:
                raise ValueError(
                    "Base checkpoint would skip critical blind-backbone tensors. "
                    f"Critical mismatches: {self._format_key_list(critical_skipped)}"
                )

        nn.Module.load_state_dict(self, compatible, strict=False)
        print(f"[VisionTransformerActorCritic] Loaded {len(compatible)} tensors from {ckpt_path}.")
        if skipped:
            print(
                "[VisionTransformerActorCritic] Skipped incompatible transfer tensors: "
                f"{self._format_key_list(skipped)}"
            )

    @staticmethod
    def _set_module_trainable(module: nn.Module | None, trainable: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def _set_std_trainable(self, trainable: bool) -> None:
        if self.log_std_head is not None:
            self._set_module_trainable(self.log_std_head, trainable)
        if self.std_head is not None:
            self._set_module_trainable(self.std_head, trainable)
        if self.log_std_param is not None:
            self.log_std_param.requires_grad_(trainable)
        if self.raw_std_param is not None:
            self.raw_std_param.requires_grad_(trainable)

    def _configure_freezing(self) -> None:
        if not self.use_vision:
            self._set_module_trainable(self.map_transformer, False)
            self._set_module_trainable(self.depth_frame_encoder, False)
            self._set_module_trainable(self.depth_pool, False)
            self._set_module_trainable(self.depth_frame_proj, False)
            self._set_module_trainable(self.depth_frame_norm, False)
            self._set_module_trainable(self.depth_memory, False)
            self._set_module_trainable(self.depth_out_norm, False)
            self._set_module_trainable(self.map_proprio_policy_norm, False)
            self._set_module_trainable(self.map_proprio_policy_proj, False)
            self._set_module_trainable(self.map_proprio_query_norm, False)
            self._set_module_trainable(self.map_proprio_cross_attn, False)
            self._set_module_trainable(self.map_proprio_out_proj, False)
            self._set_module_trainable(self.map_proprio_gate, False)
            self._set_module_trainable(self.intent_modulator, False)
            self._set_module_trainable(self.intent_gate, False)
            self._set_module_trainable(self.residual_policy, False)
            self._set_module_trainable(self.residual_gate, False)
            self._set_module_trainable(self.foot_traj_head, False)
            return

        base_modules = [
            self.history_embed,
            self.history_block,
            self.q_mlp,
            self.cmd_embed,
            self.cmd_block,
            self.actor_trunk,
            self.mean_head,
            self.vel_head,
            self.anchor_head,
        ]
        vision_modules = [
            self.map_transformer,
            self.depth_frame_encoder,
            self.depth_pool,
            self.depth_frame_proj,
            self.depth_frame_norm,
            self.depth_memory,
            self.depth_out_norm,
            self.map_proprio_policy_norm,
            self.map_proprio_policy_proj,
            self.map_proprio_query_norm,
            self.map_proprio_cross_attn,
            self.map_proprio_out_proj,
            self.map_proprio_gate,
            self.intent_modulator,
            self.intent_gate,
            self.residual_policy,
            self.residual_gate,
            self.foot_traj_head,
        ]

        if self.stage == TrainingStage.VISION_ADAPTER:
            for module in base_modules:
                self._set_module_trainable(module, False)
            for module in vision_modules:
                self._set_module_trainable(module, True)
            self._set_module_trainable(self.critic, True)
            self._set_std_trainable(not self.freeze_std_in_adapter)
        elif self.stage == TrainingStage.FINETUNE_ALL:
            for parameter in self.parameters():
                parameter.requires_grad_(True)
        elif self.stage == TrainingStage.BASE_ONLY:
            for module in base_modules:
                self._set_module_trainable(module, True)
            for module in vision_modules:
                self._set_module_trainable(module, False)
            self._set_module_trainable(self.critic, True)
            self._set_std_trainable(True)

    def _get_map_encoding(self, obs: TensorDict, policy_obs: torch.Tensor, command_latent: torch.Tensor) -> torch.Tensor:
        if "vision" not in obs:
            raise KeyError("VisionTransformerActorCritic requires observation key 'vision'.")
        if self.map_transformer is None:
            raise RuntimeError("Height-map vision path requested, but map_transformer is not initialized.")

        height_scan = obs["vision"]
        batch_size = height_scan.shape[0]
        map_height_flat, validity_mask = self._split_height_scan(height_scan, self.map_height * self.map_width)
        map_z = map_height_flat.reshape(batch_size, 1, self.map_height, self.map_width)

        if self.z_clip > 0.0:
            map_z = map_z.clamp(-self.z_clip, self.z_clip)
            if self.normalize_height:
                map_z = map_z / self.z_clip

        map_x = self._grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        map_y = self._grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        if validity_mask is not None:
            map_valid = validity_mask.reshape(batch_size, 1, self.map_height, self.map_width)
            map_valid = (map_valid > 0.5).to(dtype=map_z.dtype)
            map_3d = torch.cat([map_x, map_y, map_z, map_valid], dim=1)
        else:
            map_3d = torch.cat([map_x, map_y, map_z], dim=1)
        return self.map_transformer(map_3d, policy_obs, command_latent)

    @staticmethod
    def _split_height_scan(height_scan: torch.Tensor, expected_cells: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if height_scan.ndim != 2:
            raise ValueError(f"Height-map vision observations must be 2D [B, N], got {tuple(height_scan.shape)}.")
        if height_scan.shape[-1] == expected_cells:
            return height_scan, None
        if height_scan.shape[-1] == 2 * expected_cells:
            return height_scan[..., :expected_cells], height_scan[..., expected_cells:]
        raise ValueError(
            "Height-map vision observations must contain either H*W or 2*H*W features, "
            f"got {tuple(height_scan.shape)}."
        )

    @staticmethod
    def _build_depth_frame_encoder(
        *,
        in_channels: int,
        channels: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        strides: tuple[int, ...],
        activation: str,
    ) -> tuple[nn.Sequential, int]:
        if not channels:
            raise ValueError("depth_cnn_channels must be non-empty.")
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                "depth_cnn_channels, depth_cnn_kernel_sizes, and depth_cnn_strides must have the same length."
            )

        layers: list[nn.Module] = []
        current_channels = int(in_channels)
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            padding = kernel_size // 2
            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(resolve_nn_activation(activation))
            current_channels = out_channels
        return nn.Sequential(*layers), current_channels

    @staticmethod
    def _infer_depth_channels(obs: TensorDict) -> int:
        if "vision" not in obs:
            raise KeyError("VisionTransformerActorCritic requires observation key 'vision' for depth_sru.")
        vision = obs["vision"]
        if vision.ndim == 4:
            return int(vision.shape[1])
        if vision.ndim == 5:
            return int(vision.shape[2])
        raise ValueError(
            "depth_sru expects vision observations shaped [B, C, H, W] or [B, T, C, H, W], "
            f"but got {tuple(vision.shape)}."
        )

    def _get_depth_encoding(self, obs: TensorDict) -> torch.Tensor:
        if "vision" not in obs:
            raise KeyError("VisionTransformerActorCritic requires observation key 'vision'.")
        if (
            self.depth_frame_encoder is None
            or self.depth_pool is None
            or self.depth_frame_proj is None
            or self.depth_frame_norm is None
            or self.depth_memory is None
            or self.depth_out_norm is None
        ):
            raise RuntimeError("Depth-SRU vision path requested, but the depth encoder modules are not initialized.")

        depth = obs["vision"]
        if depth.ndim == 4:
            depth = depth.unsqueeze(1)
        elif depth.ndim != 5:
            raise ValueError(
                "depth_sru expects vision observations shaped [B, C, H, W] or [B, T, C, H, W], "
                f"but got {tuple(depth.shape)}."
            )

        batch_size, seq_len, channels, height, width = depth.shape
        depth = depth.reshape(batch_size * seq_len, channels, height, width)
        frame_features = self.depth_frame_encoder(depth)
        frame_features = self.depth_pool(frame_features).flatten(start_dim=1)
        frame_features = self.depth_frame_proj(frame_features)
        frame_features = self.depth_frame_norm(frame_features)
        frame_features = frame_features.reshape(batch_size, seq_len, -1)
        depth_encoded, _ = self.depth_memory(frame_features)
        return self.depth_out_norm(depth_encoded[:, -1])

    def _get_vision_encoding(self, obs: TensorDict, policy_obs: torch.Tensor, command_latent: torch.Tensor) -> torch.Tensor:
        if self.vision_encoder_type == "height_map":
            return self._get_map_encoding(obs, policy_obs, command_latent)
        if self.vision_encoder_type == "depth_sru":
            return self._get_depth_encoding(obs)
        raise RuntimeError(f"Unsupported vision_encoder_type='{self.vision_encoder_type}'.")

    def _apply_map_proprio_cross_attention(self, z_vis: torch.Tensor, policy_obs: torch.Tensor) -> torch.Tensor:
        if (
            self.map_proprio_policy_norm is None
            or self.map_proprio_policy_proj is None
            or self.map_proprio_query_norm is None
            or self.map_proprio_cross_attn is None
            or self.map_proprio_out_proj is None
            or self.map_proprio_gate is None
        ):
            raise RuntimeError("Map-proprio cross attention requested, but its modules are not initialized.")

        proprio_token = self.map_proprio_policy_proj(self.map_proprio_policy_norm(policy_obs)).unsqueeze(1)
        map_query = self.map_proprio_query_norm(z_vis).unsqueeze(1)
        kv = torch.cat([proprio_token, map_query], dim=1)
        attn_out, _ = self.map_proprio_cross_attn(map_query, kv, kv)
        delta = self.map_proprio_out_proj(attn_out.squeeze(1))
        return self.map_proprio_gate(z_vis, delta)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().get_checkpoint_metadata()
        metadata["policy_family"] = "vision_transformer_actor_critic"
        metadata["signature"].update(
            {
                "vision_encoder_type": self.vision_encoder_type,
                "map_height": self.map_height,
                "map_width": self.map_width,
                "dim_map_embed": self.dim_map_embed,
                "height_vision_has_mask": bool(self._height_vision_has_mask),
                "depth_vision_channels": int(self._depth_vision_channels),
                "critic_use_vision": bool(self.critic_use_vision),
                "use_foot_traj_head": bool(self.use_foot_traj_head),
                "foot_traj_output_dim": int(self.foot_traj_output_dim),
                "training_stage": self.stage.name.lower(),
                "use_action_residual": bool(self.use_action_residual),
                "use_map_proprio_cross_attention": bool(self.use_map_proprio_cross_attention),
                "use_identity_gates": bool(self.use_identity_gates),
            }
        )
        return metadata

    def build_optimizer_param_groups(
        self,
        *,
        base_lr: float,
        backbone_lr_scale: float = 1.0,
        vision_adapter_lr_scale: float = 1.0,
        critic_lr_scale: float = 1.0,
    ) -> list[dict[str, Any]]:
        modules_by_group = {
            "backbone": [
                self.history_embed,
                self.history_block,
                self.q_mlp,
                self.cmd_embed,
                self.cmd_block,
                self.actor_trunk,
                self.mean_head,
                self.vel_head,
                self.anchor_head,
            ],
            "vision_adapter": [
                self.map_transformer,
                self.depth_frame_encoder,
                self.depth_pool,
                self.depth_frame_proj,
                self.depth_frame_norm,
                self.depth_memory,
                self.depth_out_norm,
                self.map_proprio_policy_norm,
                self.map_proprio_policy_proj,
                self.map_proprio_query_norm,
                self.map_proprio_cross_attn,
                self.map_proprio_out_proj,
                self.map_proprio_gate,
                self.intent_modulator,
                self.intent_gate,
                self.residual_policy,
                self.residual_gate,
                self.foot_traj_head,
            ],
            "critic": [self.critic],
        }
        group_scales = {
            "backbone": float(backbone_lr_scale),
            "vision_adapter": float(vision_adapter_lr_scale),
            "critic": float(critic_lr_scale),
        }

        seen: set[int] = set()
        param_groups: list[dict[str, Any]] = []
        for group_name, modules in modules_by_group.items():
            params: list[nn.Parameter] = []
            for module in modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    if not parameter.requires_grad:
                        continue
                    param_id = id(parameter)
                    if param_id in seen:
                        continue
                    seen.add(param_id)
                    params.append(parameter)
            if params:
                lr_scale = group_scales[group_name]
                param_groups.append(
                    {
                        "name": group_name,
                        "params": params,
                        "lr": float(base_lr) * lr_scale,
                        "lr_scale": lr_scale,
                    }
                )

        std_params: list[nn.Parameter] = []
        for parameter in (self.log_std_param, self.raw_std_param):
            if parameter is None or not parameter.requires_grad:
                continue
            param_id = id(parameter)
            if param_id in seen:
                continue
            seen.add(param_id)
            std_params.append(parameter)
        if std_params:
            param_groups.append(
                {
                    "name": "action_std",
                    "params": std_params,
                    "lr": float(base_lr) * float(backbone_lr_scale),
                    "lr_scale": float(backbone_lr_scale),
                }
            )

        remaining = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in seen
        ]
        if remaining:
            param_groups.append(
                {
                    "name": "remaining",
                    "params": remaining,
                    "lr": float(base_lr),
                    "lr_scale": 1.0,
                }
            )
        return param_groups

    def _compute_actor_context(
        self, obs: TensorDict, *, store_aux_outputs: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        policy_obs = self._get_concat_2d(obs, "policy")
        policy_obs = self.actor_obs_normalizer(policy_obs)

        history_tokens = self._get_concat_seq(obs, "policy_history", self.history_len)
        cmd_tokens = self._get_concat_seq(obs, "command_window", self.cmd_len)
        history_tokens = self.history_obs_normalizer(history_tokens)
        cmd_tokens = self.command_obs_normalizer(cmd_tokens)

        hist_encoded = self._encode_history_tokens(history_tokens)
        h_pool = hist_encoded.max(dim=1).values
        h_last = hist_encoded[:, -1]
        u_t = self._encode_command(h_pool, cmd_tokens)

        aux_outputs: dict[str, torch.Tensor] = {}
        v_for_actor = None
        if self.use_vel_estimator:
            v_hat = self.vel_head(h_last)
            aux_outputs["v_hat"] = v_hat
            v_for_actor = v_hat.detach() if self.vel_estimator_detach else v_hat

        anchor_for_actor = None
        if self.use_anchor_estimator:
            anchor_inputs = self._build_anchor_estimator_input(obs, h_last=h_last, u_t=u_t)
            anchor_hat = self.anchor_head(anchor_inputs)
            aux_outputs["anchor_hat"] = anchor_hat
            anchor_for_actor = anchor_hat.detach() if self.anchor_estimator_detach else anchor_hat

        if store_aux_outputs:
            self._last_aux_outputs = aux_outputs if aux_outputs else None

        z_vis = self._get_vision_encoding(obs, policy_obs, u_t) if self.use_vision else None
        if z_vis is not None and self.use_map_proprio_cross_attention:
            z_vis = self._apply_map_proprio_cross_attention(z_vis, policy_obs)
        return policy_obs, u_t, v_for_actor, anchor_for_actor, z_vis

    def _predict_foot_traj(
        self,
        command_latent: torch.Tensor,
        vision_latent: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_foot_traj_head or self.foot_traj_head is None:
            return None

        if self.foot_traj_use_vision:
            if vision_latent is None:
                raise RuntimeError("foot_traj_use_vision=True requires a valid vision embedding.")
            foot_inputs = torch.cat([command_latent, vision_latent], dim=-1)
        else:
            foot_inputs = command_latent
        return self.foot_traj_head(foot_inputs)

    def _update_distribution(self, obs: TensorDict) -> None:
        policy_obs, command_latent, v_for_actor, anchor_for_actor, z_vis = self._compute_actor_context(
            obs, store_aux_outputs=True
        )

        if z_vis is not None:
            delta_u = self.intent_modulator(z_vis)
            command_latent = self.intent_gate(command_latent, delta_u)

        foot_traj = self._predict_foot_traj(command_latent, z_vis)
        aux_outputs = dict(self._last_aux_outputs) if self._last_aux_outputs is not None else {}
        if foot_traj is not None:
            aux_outputs["foot_traj"] = foot_traj
        self._last_aux_outputs = aux_outputs if aux_outputs else None

        actor_inputs = [policy_obs, command_latent]
        if v_for_actor is not None:
            actor_inputs.append(v_for_actor)
        if anchor_for_actor is not None:
            actor_inputs.append(anchor_for_actor)
        actor_in = torch.cat(actor_inputs, dim=-1)

        trunk = self.actor_trunk(actor_in)
        base_mean = self.mean_head(trunk)

        if z_vis is not None and self.use_action_residual:
            residual_in = torch.cat([policy_obs, z_vis], dim=-1)
            if self.residual_policy is None or self.residual_gate is None:
                raise RuntimeError("use_action_residual=True but residual action modules are not initialized.")
            action_residual = self.residual_policy(residual_in)
            mean = self.residual_gate(base_mean, action_residual)
        else:
            mean = base_mean

        if self.state_dependent_std:
            if self.noise_std_type == "log":
                log_std = torch.clamp(self.log_std_head(trunk), self.log_std_bounds[0], self.log_std_bounds[1])
                std = torch.exp(log_std)
            elif self.noise_std_type == "scalar":
                std = F.softplus(self.std_head(trunk)) + self.min_std
                std = torch.clamp(std, self.min_std, self._max_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        else:
            if self.noise_std_type == "log":
                log_std = torch.clamp(self.log_std_param, self.log_std_bounds[0], self.log_std_bounds[1])
                std = torch.exp(log_std).expand_as(mean)
            elif self.noise_std_type == "scalar":
                std = (F.softplus(self.raw_std_param) + self.min_std).expand_as(mean)
                std = torch.clamp(std, self.min_std, self._max_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        self.distribution = Normal(mean, std, validate_args=self.validate_args)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self._get_concat_2d(obs, "critic")
        critic_obs = self.critic_obs_normalizer(critic_obs)

        if self.critic_use_vision:
            policy_obs, command_latent, _, _, z_vis = self._compute_actor_context(obs, store_aux_outputs=False)
            if z_vis is None:
                z_vis = self._get_vision_encoding(obs, policy_obs, command_latent)
            critic_obs = torch.cat([critic_obs, z_vis], dim=-1)

        return self.critic(critic_obs)

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        self._update_distribution(obs)
        outputs = {"action": self.distribution.mean}
        aux = self.get_last_aux_outputs(clear=False)
        if "foot_traj" in aux:
            outputs["foot_traj"] = aux["foot_traj"]
        return outputs

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any(key.startswith("student.") for key in state_dict.keys()):
            student_state = {
                key.replace("student.", "", 1): value for key, value in state_dict.items() if key.startswith("student.")
            }
            if not student_state:
                raise ValueError("No 'student.*' parameters found in provided state_dict.")
            nn.Module.load_state_dict(self, student_state, strict=strict)
            return False

        nn.Module.load_state_dict(self, state_dict, strict=strict)
        return True
