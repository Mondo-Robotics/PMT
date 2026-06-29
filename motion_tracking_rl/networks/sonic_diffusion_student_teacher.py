# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn
import warnings

import torch
import torch.nn as nn
from tensordict import TensorDict

from motion_tracking_rl.networks.actor_critic import SonicActorCritic
from motion_tracking_rl.networks.diffusion_actor_critic import DiffusionActorCritic


class SonicDiffusionStudentTeacher(nn.Module):
    """Student-teacher wrapper with a diffusion student and SONIC teacher."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_cfg: dict[str, Any] | None = None,
        student_use_inference_for_rollout: bool = False,
        student_extra_policy_obs_groups: list[str] | None = None,
        teacher_cfg: dict[str, Any] | None = None,
        teacher_ckpt_path: str | None = None,
        teacher_load_strict: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "SonicDiffusionStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if "policy" not in obs_groups:
            raise ValueError("SonicDiffusionStudentTeacher requires obs_groups['policy'].")
        if "teacher" not in obs_groups:
            raise ValueError("SonicDiffusionStudentTeacher requires obs_groups['teacher'].")

        self.obs_groups = obs_groups
        self.loaded_teacher = False
        self.teacher_ckpt_path = teacher_ckpt_path

        student_cfg = {} if student_cfg is None else dict(student_cfg)
        teacher_cfg = {} if teacher_cfg is None else dict(teacher_cfg)
        self.student_use_inference_for_rollout = bool(student_use_inference_for_rollout)
        student_extra_policy_obs_groups = [] if student_extra_policy_obs_groups is None else list(
            student_extra_policy_obs_groups
        )

        student_policy_groups = list(obs_groups["policy"])
        for group_name in student_extra_policy_obs_groups:
            if group_name not in obs.keys():
                raise KeyError(
                    f"Student extra policy group '{group_name}' is missing in observations. "
                    f"Available keys: {list(obs.keys())}"
                )
            if group_name not in student_policy_groups:
                student_policy_groups.append(group_name)
        self._student_policy_groups = student_policy_groups

        # Student gets the same observation contract as standard FPO training.
        student_obs_groups = {
            "policy": self._student_policy_groups,
            "critic": list(obs_groups.get("critic", obs_groups["policy"])),
        }
        self.student = DiffusionActorCritic(
            obs=obs,
            obs_groups=student_obs_groups,
            num_actions=num_actions,
            **student_cfg,
        )

        # Teacher keeps standard SONIC policy/critic observation contract.
        # Additional encoder groups are read directly from `obs` by SonicActorCritic.
        teacher_obs_groups = {
            "policy": list(obs_groups["policy"]),
            "critic": list(obs_groups.get("critic", obs_groups["policy"])),
        }
        self.teacher = SonicActorCritic(
            obs=obs,
            obs_groups=teacher_obs_groups,
            num_actions=num_actions,
            **teacher_cfg,
        )
        self._freeze_teacher()

        if teacher_ckpt_path:
            self._load_teacher_checkpoint(Path(teacher_ckpt_path), strict=teacher_load_strict)
        elif self._teacher_cfg_has_pretrained_source(teacher_cfg):
            # SONIC was initialized from ONNX teacher parameters inside constructor.
            self.loaded_teacher = True
        else:
            warnings.warn(
                "SonicDiffusionStudentTeacher teacher has no checkpoint or ONNX source configured. "
                "loaded_teacher remains False.",
                stacklevel=2,
            )

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.student.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.student.action_std

    @property
    def entropy(self) -> torch.Tensor:
        return self.student.entropy

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _teacher_cfg_has_pretrained_source(teacher_cfg: dict[str, Any]) -> bool:
        return bool(
            teacher_cfg.get("pretrained_encoder_onnx_path") is not None
            or teacher_cfg.get("pretrained_decoder_onnx_path") is not None
        )

    def _load_teacher_checkpoint(self, ckpt_path: Path, strict: bool) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
        loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported teacher checkpoint format in: {ckpt_path}")
        self._load_teacher_state_dict(state_dict, strict=strict)
        self.loaded_teacher = True
        print(f"[Distill] Loaded SONIC teacher checkpoint: {ckpt_path}")

    def _load_teacher_state_dict(self, state_dict: dict[str, Any], strict: bool) -> None:
        # Accept both plain SONIC keys and "teacher.*" prefixed keys.
        if any(key.startswith("teacher.") for key in state_dict):
            teacher_state = {k.replace("teacher.", "", 1): v for k, v in state_dict.items() if k.startswith("teacher.")}
        else:
            teacher_state = state_dict
        self.teacher.load_state_dict(teacher_state, strict=strict)
        self._freeze_teacher()

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.student.get_actor_obs(obs)

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.student_use_inference_for_rollout:
            return self.student.act_inference(obs)
        return self.student.act(obs)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act_inference(obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher.act_inference(obs)

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
        # Distillation checkpoint resume path.
        if any(key.startswith("student.") for key in state_dict):
            super().load_state_dict(state_dict, strict=strict)
            self._freeze_teacher()
            self.loaded_teacher = True
            return True

        # Plain SONIC teacher checkpoint path.
        self._load_teacher_state_dict(state_dict, strict=strict)
        self.loaded_teacher = True
        return False
