# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from motion_tracking_rl.networks.layers import MLP, EmpiricalNormalization, HiddenState
from motion_tracking_rl.registry import register_network


@register_network("StudentTeacher", compat_name="student_teacher")
class StudentTeacher(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        teacher_ckpt_path: str | None = None,
        motion_target_obs_key: str | None = None,
        anchor_target_obs_key: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False
        self.teacher_ckpt_path = teacher_ckpt_path
        self.obs_groups = obs_groups
        self.motion_target_obs_key = motion_target_obs_key
        self.anchor_target_obs_key = anchor_target_obs_key

        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.motion_target_dim = 0
        if motion_target_obs_key is not None:
            if motion_target_obs_key not in obs.keys():
                raise KeyError(
                    f"motion_target_obs_key='{motion_target_obs_key}' not found in observations. "
                    f"Available groups: {list(obs.keys())}"
                )
            self.motion_target_dim = int(obs[motion_target_obs_key].shape[-1])

        self.anchor_target_dim = 0
        if anchor_target_obs_key is not None:
            if anchor_target_obs_key not in obs.keys():
                raise KeyError(
                    f"anchor_target_obs_key='{anchor_target_obs_key}' not found in observations. "
                    f"Available groups: {list(obs.keys())}"
                )
            self.anchor_target_dim = int(obs[anchor_target_obs_key].shape[-1])

        self.has_auxiliary_heads = self.motion_target_dim > 0 or self.anchor_target_dim > 0

        if self.has_auxiliary_heads:
            feature_dim = int(student_hidden_dims[-1])
            backbone_hidden_dims = list(student_hidden_dims[:-1]) or [feature_dim]
            self.student_backbone = MLP(num_student_obs, feature_dim, backbone_hidden_dims, activation)
            self.student_action_head = nn.Linear(feature_dim, num_actions)
            self.student_motion_head = (
                nn.Linear(feature_dim, self.motion_target_dim) if self.motion_target_dim > 0 else None
            )
            self.student_anchor_head = (
                nn.Linear(feature_dim, self.anchor_target_dim) if self.anchor_target_dim > 0 else None
            )
            self.student = None
            print(f"Student backbone: {self.student_backbone}")
        else:
            self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
            self.student_backbone = None
            self.student_action_head = None
            self.student_motion_head = None
            self.student_anchor_head = None
            print(f"Student MLP: {self.student}")

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        print(f"Teacher MLP: {self.teacher}")

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        if teacher_ckpt_path is not None:
            self._load_teacher_from_checkpoint(teacher_ckpt_path)

    def _load_teacher_from_checkpoint(self, ckpt_path: str) -> None:
        import os

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
        teacher_state_dict = {}
        teacher_obs_normalizer_state_dict = {}
        for key, value in state_dict.items():
            if "actor." in key:
                teacher_state_dict[key.replace("actor.", "")] = value
            if "actor_obs_normalizer." in key:
                teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
        self.teacher.load_state_dict(teacher_state_dict, strict=True)
        if teacher_obs_normalizer_state_dict:
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=True)
        self.loaded_teacher = True
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()
        print(f"[StudentTeacher] Loaded teacher from checkpoint: {ckpt_path}")

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
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

    def _student_outputs_from_normalized_obs(self, normalized_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.has_auxiliary_heads:
            features = self.student_backbone(normalized_obs)
            outputs = {"action": self.student_action_head(features)}
            if self.student_motion_head is not None:
                outputs["motion"] = self.student_motion_head(features)
            if self.student_anchor_head is not None:
                outputs["anchor"] = self.student_anchor_head(features)
            return outputs
        return {"action": self.student(normalized_obs)}

    def _update_distribution(self, obs: TensorDict) -> None:
        normalized_obs = self.student_obs_normalizer(self.get_student_obs(obs))
        mean = self._student_outputs_from_normalized_obs(normalized_obs)["action"]
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.infer_student_outputs(obs)["action"]

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        return self._student_outputs_from_normalized_obs(student_obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any("actor" in key for key in state_dict):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any("student" in key or "student_backbone" in key for key in state_dict):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
