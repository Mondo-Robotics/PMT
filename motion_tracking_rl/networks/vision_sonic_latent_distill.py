# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.algorithms.distillation import Distillation
from motion_tracking_rl.networks.actor_critic import ActorCritic, SonicActorCritic
from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.networks.residual_vision_action import GatedResidual
from motion_tracking_rl.utils.utils import resolve_optimizer


def _small_init_last_linear(module: nn.Module) -> None:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    nn.init.xavier_uniform_(linears[-1].weight)
    linears[-1].weight.data.mul_(0.01)
    nn.init.zeros_(linears[-1].bias)


def _resolve_class(class_name: str | None, default_class: type[nn.Module]) -> type[nn.Module]:
    if class_name is None:
        return default_class
    return eval(class_name)


class HeightMapEncoder(nn.Module):
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
        self.vision_channels = vision_channels
        self.map_height = map_height
        self.map_width = map_width
        self.embed_dim = embed_dim

        xs = (torch.arange(map_height) - (map_height - 1) / 2.0) * map_resolution
        ys = (torch.arange(map_width) - (map_width - 1) / 2.0) * map_resolution
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        self.register_buffer("xy_grid", torch.stack([grid_x, grid_y], dim=0).unsqueeze(0))

        self.map_cnn = nn.Sequential(
            nn.Conv2d(vision_channels, embed_dim - 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim - 2, embed_dim - 2, kernel_size=1),
        )
        self.query = nn.Sequential(
            nn.LayerNorm(proprio_dim + token_dim),
            nn.Linear(proprio_dim + token_dim, embed_dim),
        )
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.SiLU())

    def forward(self, vision: torch.Tensor, proprio: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        batch = vision.shape[0]
        height = vision.view(batch, self.vision_channels, self.map_height, self.map_width)
        xy = self.xy_grid.expand(batch, -1, -1, -1).to(dtype=height.dtype)
        kv = torch.cat([self.map_cnn(height), xy], dim=1).flatten(2).transpose(1, 2)
        kv = self.kv_norm(kv)
        query = self.query(torch.cat([proprio, token], dim=-1)).unsqueeze(1)
        z_vis, _ = self.attn(query, kv, kv, need_weights=False)
        return self.out(z_vis.squeeze(1))


class VisionLatentSonicActorCritic(SonicActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        robot_motion_dim: int = 640,
        actor_hidden_dims: list[int] = [2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims: list[int] | None = None,
        activation: str = "silu",
        map_height: int = 17,
        map_width: int = 11,
        map_resolution: float = 0.1,
        dim_map_embed: int = 64,
        num_attn_heads: int = 4,
        token_adapter_hidden_dims: list[int] = [256, 128],
        action_residual_hidden_dims: list[int] = [256, 128],
        q_ref_delta_hidden_dims: list[int] = [512, 256],
        use_foot_traj_head: bool = False,
        foot_traj_target_obs_key: str | None = None,
        foot_traj_output_dim: int | None = None,
        foot_traj_hidden_dims: list[int] = [256, 128],
        foot_traj_detach_features: bool = True,
        token_residual_scale: float = 0.5,
        action_residual_scale: float = 0.25,
        use_action_residual: bool = True,
        freeze_sonic_encoder: bool = True,
        freeze_sonic_decoder: bool = True,
        freeze_action_std: bool = True,
        load_action_std_from_checkpoint: bool = True,
        store_loaded_policy_as_prior: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            robot_motion_dim=robot_motion_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )
        self.num_actions = num_actions
        self.robot_motion_dim = robot_motion_dim
        self.token_residual_scale = token_residual_scale
        self.action_residual_scale = action_residual_scale
        self.use_action_residual = use_action_residual
        self.use_foot_traj_head = bool(use_foot_traj_head)
        self.foot_traj_target_obs_key = foot_traj_target_obs_key
        self.foot_traj_output_dim = 0 if foot_traj_output_dim is None else int(foot_traj_output_dim)
        self.foot_traj_detach_features = bool(foot_traj_detach_features)
        self.freeze_sonic_encoder = freeze_sonic_encoder
        self.freeze_sonic_decoder = freeze_sonic_decoder
        self.freeze_action_std = freeze_action_std
        self.load_action_std_from_checkpoint = load_action_std_from_checkpoint
        self.store_loaded_policy_as_prior = store_loaded_policy_as_prior
        object.__setattr__(self, "_loaded_policy_prior", None)

        vision_channels = obs["vision"].shape[-1] // (map_height * map_width)
        self.vision_encoder = HeightMapEncoder(
            vision_channels=vision_channels,
            proprio_dim=self.proprio_dim,
            token_dim=self.latent_dim,
            map_height=map_height,
            map_width=map_width,
            map_resolution=map_resolution,
            embed_dim=dim_map_embed,
            num_heads=num_attn_heads,
        )
        token_input_dim = self.latent_dim + dim_map_embed + self.proprio_dim
        self.token_adapter = MLP(token_input_dim, self.latent_dim, token_adapter_hidden_dims, activation)
        self.token_gate = GatedResidual(self.latent_dim)
        _small_init_last_linear(self.token_adapter)

        action_input_dim = self.proprio_dim + self.latent_dim + dim_map_embed
        self.action_residual = MLP(action_input_dim, num_actions, action_residual_hidden_dims, activation)
        _small_init_last_linear(self.action_residual)

        self.q_ref_delta_head = MLP(action_input_dim, num_actions, q_ref_delta_hidden_dims, activation)
        _small_init_last_linear(self.q_ref_delta_head)

        if self.use_foot_traj_head:
            if self.foot_traj_output_dim <= 0:
                self.foot_traj_output_dim = int(obs[self.foot_traj_target_obs_key].shape[-1])
            self.foot_traj_head = MLP(action_input_dim, self.foot_traj_output_dim, foot_traj_hidden_dims, activation)
            _small_init_last_linear(self.foot_traj_head)
        else:
            self.foot_traj_head = None

        if freeze_sonic_encoder:
            self._set_trainable(self.robot_encoder, False)
            self._set_trainable(self.robot_encoder_proj, False)
            self._set_trainable(self.human_encoder, False)
            self._set_trainable(self.human_encoder_proj, False)
            self._set_trainable(self.hybrid_encoder, False)
            self._set_trainable(self.hybrid_encoder_proj, False)
        if freeze_sonic_decoder:
            self._set_trainable(self.control_decoder, False)
        self._set_trainable(self.motion_decoder, False)
        self.std.requires_grad_(not freeze_action_std)
        self._last_aux: dict[str, torch.Tensor] = {}

    def _base_latent(self, obs: TensorDict) -> torch.Tensor:
        if self.freeze_sonic_encoder:
            with torch.no_grad():
                return self.encode_robot_pre_quant(obs["robot_encoder"])
        return self.encode_robot_pre_quant(obs["robot_encoder"])

    def _actor_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        proprio = self.actor_obs_normalizer(self.get_actor_obs(obs))
        latent_base = self._base_latent(obs)
        token_base = self.quantize_latent(latent_base)
        z_vis = self.vision_encoder(obs["vision"], proprio, token_base)
        adapter_input = torch.cat([latent_base, z_vis, proprio], dim=-1)
        delta_latent = self.token_adapter(adapter_input)
        latent = self.token_gate(latent_base, delta_latent, self.token_residual_scale)
        token = self.quantize_latent(latent)
        base_action = self.control_decoder(self.build_control_decoder_input(proprio, token))
        residual_input = torch.cat([proprio, token, z_vis], dim=-1)
        delta_action = torch.tanh(self.action_residual(residual_input)) * self.action_residual_scale
        action = base_action + delta_action if self.use_action_residual else base_action
        outputs = {
            "action": action,
            "token": token,
            "token_base": token_base,
            "latent": latent,
            "latent_base": latent_base,
            "delta_latent": delta_latent,
            "z_vis": z_vis,
            "base_action": base_action,
            "delta_action": delta_action,
            "q_ref_delta": self.q_ref_delta_head(residual_input),
        }
        if self.foot_traj_head is not None:
            foot_traj_input = residual_input.detach() if self.foot_traj_detach_features else residual_input
            outputs["foot_traj"] = self.foot_traj_head(foot_traj_input)
        return outputs

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        outputs = self._actor_outputs(obs)
        self.distribution = Normal(outputs["action"], self._get_action_std(outputs["action"]))
        self._last_aux = {k: v for k, v in outputs.items() if k != "action"}
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        outputs = self._actor_outputs(obs)
        self.distribution = Normal(outputs["action"], self._get_action_std(outputs["action"]))
        self._last_aux = {k: v for k, v in outputs.items() if k != "action"}
        return outputs["action"]

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        return self._actor_outputs(obs)

    def get_last_aux_outputs(self, clear: bool = False) -> dict[str, torch.Tensor]:
        aux = dict(self._last_aux)
        if clear:
            self._last_aux.clear()
        return aux

    def set_warmup_freeze(
        self,
        *,
        freeze_encoders: bool,
        freeze_control_decoder: bool,
        freeze_action_std: bool,
    ) -> dict[str, bool]:
        if self.freeze_sonic_encoder:
            self._set_trainable(self.robot_encoder, False)
            self._set_trainable(self.robot_encoder_proj, False)
        if self.freeze_sonic_decoder:
            self._set_trainable(self.control_decoder, False)
        self.std.requires_grad_(not (freeze_action_std or self.freeze_action_std))
        self._warmup_freeze_state = {
            "encoders_frozen": self.freeze_sonic_encoder,
            "control_decoder_frozen": self.freeze_sonic_decoder,
            "action_std_frozen": freeze_action_std or self.freeze_action_std,
        }
        return dict(self._warmup_freeze_state)

    def build_optimizer_param_groups(
        self,
        base_lr: float,
        backbone_lr_scale: float = 1.0,
        vision_adapter_lr_scale: float = 1.0,
        critic_lr_scale: float = 1.0,
    ) -> list[dict[str, object]]:
        groups = {"backbone": [], "vision": [], "critic": []}
        vision_prefixes = (
            "vision_encoder.",
            "token_adapter.",
            "token_gate.",
            "action_residual.",
            "q_ref_delta_head.",
            "foot_traj_head.",
        )
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("critic."):
                groups["critic"].append(param)
            elif name.startswith(vision_prefixes):
                groups["vision"].append(param)
            else:
                groups["backbone"].append(param)
        return [
            {"params": groups["backbone"], "lr": base_lr * backbone_lr_scale},
            {"params": groups["vision"], "lr": base_lr * vision_adapter_lr_scale},
            {"params": groups["critic"], "lr": base_lr * critic_lr_scale},
        ]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any(key.startswith("student.") for key in state_dict):
            student_state = {
                key.replace("student.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("student.")
            }
            if not self.load_action_std_from_checkpoint and "std" in student_state:
                student_state["std"] = self.std.detach().clone()
            self._load_transfer_state_dict(student_state)
            if self.store_loaded_policy_as_prior:
                self._store_current_policy_as_prior()
            return False
        self._load_transfer_state_dict(state_dict) if strict else nn.Module.load_state_dict(self, state_dict, strict=False)
        if self.store_loaded_policy_as_prior:
            self._store_current_policy_as_prior()
        return True

    def _load_transfer_state_dict(self, state_dict: dict) -> None:
        current = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        skipped = sorted(set(state_dict) - set(compatible))
        missing = sorted(set(current) - set(compatible))
        if skipped or missing:
            print(
                f"[VisionSONIC] Transfer-loaded {len(compatible)} tensors; "
                f"skipped {len(skipped)} checkpoint tensors and left {len(missing)} current tensors initialized."
            )
            nn.Module.load_state_dict(self, compatible, strict=False)
            return
        nn.Module.load_state_dict(self, state_dict, strict=True)

    def _store_current_policy_as_prior(self) -> None:
        object.__setattr__(self, "_loaded_policy_prior", None)
        prior = copy.deepcopy(self)
        object.__setattr__(prior, "_loaded_policy_prior", None)
        prior.requires_grad_(False)
        prior.eval()
        object.__setattr__(self, "_loaded_policy_prior", prior)

    def reference_act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self._loaded_policy_prior.act_inference(obs)


class VisionSonicStudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_cfg: dict,
        teacher_cfg: dict,
        teacher_robot_encoder_key: str = "teacher_robot_encoder",
        student_q_ref_key: str = "student_q_ref",
        teacher_q_ref_key: str = "teacher_q_ref",
        align_teacher_to_student_reference: bool = True,
        teacher_ckpt_path: str | None = None,
        teacher_load_strict: bool = True,
        teacher_obs_groups: dict[str, list[str]] | None = None,
        warm_start_student_residual_from_teacher: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups
        self.teacher_robot_encoder_key = teacher_robot_encoder_key
        self.student_q_ref_key = student_q_ref_key
        self.teacher_q_ref_key = teacher_q_ref_key
        self.align_teacher_to_student_reference = align_teacher_to_student_reference

        student_kwargs = {k: v for k, v in student_cfg.items() if k != "class_name"}
        teacher_class = _resolve_class(teacher_cfg.get("class_name"), SonicActorCritic)
        teacher_kwargs = {k: v for k, v in teacher_cfg.items() if k != "class_name"}
        self.teacher_obs_groups = teacher_obs_groups if teacher_obs_groups is not None else obs_groups
        has_onnx_teacher = bool(
            teacher_kwargs.get("pretrained_encoder_onnx_path") and teacher_kwargs.get("pretrained_decoder_onnx_path")
        )
        if teacher_ckpt_path is None and not has_onnx_teacher:
            raise ValueError("VisionSonicStudentTeacher requires teacher_ckpt_path or encoder/decoder ONNX paths.")
        self.student = VisionLatentSonicActorCritic(obs, obs_groups, num_actions, **student_kwargs)
        self.teacher = teacher_class(obs, self.teacher_obs_groups, num_actions, **teacher_kwargs)
        if teacher_ckpt_path is not None:
            state = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
            state = state.get("model_state_dict", state)
            if any(key.startswith("teacher.") for key in state):
                state = {key[len("teacher.") :]: value for key, value in state.items() if key.startswith("teacher.")}
            self.teacher.load_state_dict(state, strict=teacher_load_strict)
        if warm_start_student_residual_from_teacher:
            self._warm_start_student_residual_from_teacher()
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.loaded_teacher = True
        self.teacher_ckpt_path = teacher_ckpt_path

    def _warm_start_student_residual_from_teacher(self) -> None:
        if not hasattr(self.teacher, "action_residual"):
            return
        teacher_state = self.teacher.action_residual.state_dict()
        student_state = self.student.action_residual.state_dict()
        copied = 0
        for key, value in teacher_state.items():
            if key.startswith("0."):
                continue
            if key in student_state and student_state[key].shape == value.shape:
                student_state[key].copy_(value.to(device=student_state[key].device, dtype=student_state[key].dtype))
                copied += 1
        if copied > 0:
            self.student.action_residual.load_state_dict(student_state)
            print(f"[VisionSONIC] Warm-started {copied} student residual tensors from teacher.")

    @property
    def action_mean(self) -> torch.Tensor:
        return self.student.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.student.action_std

    @property
    def entropy(self) -> torch.Tensor:
        return self.student.entropy

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.student.get_actions_log_prob(actions)

    def _teacher_obs(self, obs: TensorDict) -> TensorDict:
        teacher_obs = obs.clone()
        teacher_obs["robot_encoder"] = obs[self.teacher_robot_encoder_key]
        return teacher_obs

    def _align_teacher_action(self, obs: TensorDict, action: torch.Tensor) -> torch.Tensor:
        if not self.align_teacher_to_student_reference:
            return action
        return action + obs[self.teacher_q_ref_key] - obs[self.student_q_ref_key]

    def act(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act(obs)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act_inference(obs)

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        return self.student.infer_student_outputs(obs)

    def infer_teacher_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        teacher_obs = self._teacher_obs(obs)
        with torch.no_grad():
            if hasattr(self.teacher, "infer_teacher_outputs"):
                teacher = dict(self.teacher.infer_teacher_outputs(teacher_obs))
            elif hasattr(self.teacher, "encode_robot_pre_quant"):
                latent = self.teacher.encode_robot_pre_quant(teacher_obs["robot_encoder"])
                token = self.teacher.quantize_latent(latent)
                proprio = self.teacher.actor_obs_normalizer(self.teacher.get_actor_obs(teacher_obs))
                action_token = self.teacher._prepare_action_token_for_policy(token)
                policy_input = self.teacher.build_control_decoder_input(proprio, action_token)
                teacher = {
                    "action": self.teacher.control_decoder(policy_input),
                    "token": token,
                    "latent": latent,
                }
            else:
                latent = self.student.encode_robot_pre_quant(teacher_obs["robot_encoder"])
                token = self.student.quantize_latent(latent)
                teacher = {
                    "action": self.teacher.act_inference(teacher_obs),
                    "token": token,
                    "latent": latent,
                }
            teacher["action"] = self._align_teacher_action(obs, teacher["action"])
            teacher["q_ref"] = obs[self.teacher_q_ref_key]
        return teacher

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        return self.infer_teacher_outputs(obs)["action"]

    def update_normalization(self, obs: TensorDict) -> None:
        self.student.update_normalization(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_states: tuple | None = None) -> None:
        self.student.reset(dones)
        self.teacher.reset(dones)

    def get_hidden_states(self):
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        return None

    def train(self, mode: bool = True):
        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()
        return self

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        nn.Module.load_state_dict(self, state_dict, strict=strict)
        self.teacher.requires_grad_(False)
        self.loaded_teacher = True
        return True


class VisionSonicDistillation(Distillation):
    def __init__(
        self,
        *args,
        token_loss_coef: float = 1.0,
        token_loss_delta: float = 0.1,
        q_ref_delta_loss_coef: float = 0.0,
        q_ref_delta_loss_delta: float = 0.1,
        residual_l2_coef: float = 0.0,
        backbone_lr_scale: float = 1.0,
        vision_adapter_lr_scale: float = 1.0,
        critic_lr_scale: float = 1.0,
        optimizer: str = "adam",
        **kwargs,
    ) -> None:
        super().__init__(*args, optimizer=optimizer, **kwargs)
        param_groups = self.policy.student.build_optimizer_param_groups(
            self.learning_rate,
            backbone_lr_scale=backbone_lr_scale,
            vision_adapter_lr_scale=vision_adapter_lr_scale,
            critic_lr_scale=critic_lr_scale,
        )
        self.optimizer = resolve_optimizer(optimizer)(param_groups, lr=self.learning_rate)
        self.token_loss_coef = token_loss_coef
        self.token_loss_delta = token_loss_delta
        self.q_ref_delta_loss_coef = q_ref_delta_loss_coef
        self.q_ref_delta_loss_delta = q_ref_delta_loss_delta
        self.residual_l2_coef = residual_l2_coef

    def update(self) -> dict[str, float]:
        mean_behavior_loss = 0.0
        mean_token_loss = 0.0
        mean_q_ref_delta_loss = 0.0
        mean_residual_l2 = 0.0
        loss_accum = 0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                student = self.policy.infer_student_outputs(obs)
                teacher = self.policy.infer_teacher_outputs(obs)

                behavior_loss = self.loss_fn(student["action"], privileged_actions)
                token_loss = F.huber_loss(student["latent"], teacher["latent"], delta=self.token_loss_delta)
                q_ref_delta_target = teacher["q_ref"] - obs[self.policy.student_q_ref_key]
                q_ref_delta_loss = F.huber_loss(
                    student["q_ref_delta"], q_ref_delta_target, delta=self.q_ref_delta_loss_delta
                )
                residual_l2 = student["delta_action"].square().mean()
                total_loss = (
                    behavior_loss
                    + self.token_loss_coef * token_loss
                    + self.q_ref_delta_loss_coef * q_ref_delta_loss
                    + self.residual_l2_coef * residual_l2
                )

                loss_accum = loss_accum + total_loss
                mean_behavior_loss += behavior_loss.item()
                mean_token_loss += token_loss.item()
                mean_q_ref_delta_loss += q_ref_delta_loss.item()
                mean_residual_l2 += residual_l2.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss_accum.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss_accum = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()
        return {
            "behavior": mean_behavior_loss / cnt,
            "token": mean_token_loss / cnt,
            "q_ref_delta": mean_q_ref_delta_loss / cnt,
            "residual_l2": mean_residual_l2 / cnt,
        }
