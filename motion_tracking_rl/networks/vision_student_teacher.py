from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict

from motion_tracking_rl.networks._teacher_alignment import align_teacher_actions
from motion_tracking_rl.networks.transformer_actor_critic import TransformerActorCritic
from motion_tracking_rl.networks.vision_transformer_actor_critic import VisionTransformerActorCritic
from motion_tracking_rl.registry import register_network
from motion_tracking_rl.utils import build_obs_schema


@register_network("VisionStudentTeacher", compat_name="vision_student_latent_anchor")
class VisionStudentTeacher(nn.Module):
    """Distillation wrapper for a vision-augmented transformer student and transformer teacher."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_cfg: dict[str, Any] | None = None,
        teacher_cfg: dict[str, Any] | None = None,
        teacher_class_name: str = "TransformerActorCritic",
        teacher_ckpt_path: str | None = None,
        teacher_load_strict: bool = True,
        align_teacher_to_student_reference: bool = True,
        foot_traj_target_obs_key: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "VisionStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        required_sets = ["policy", "policy_history", "command_window", "teacher", "teacher_policy_history", "teacher_command_window"]
        missing_sets = [name for name in required_sets if name not in obs_groups]
        if missing_sets:
            raise ValueError(
                f"VisionStudentTeacher requires observation sets {required_sets}. Missing: {missing_sets}"
            )

        self.obs_groups = obs_groups
        self.loaded_teacher = False
        self.teacher_ckpt_path = teacher_ckpt_path
        self.teacher_class_name = str(teacher_class_name)
        self.align_teacher_to_student_reference = bool(align_teacher_to_student_reference)
        self.foot_traj_target_obs_key = foot_traj_target_obs_key
        self._obs_schema = build_obs_schema(obs, obs_groups)
        self._last_bridge_debug: dict[str, float] = {}

        student_cfg = {} if student_cfg is None else dict(student_cfg)
        teacher_cfg = {} if teacher_cfg is None else dict(teacher_cfg)

        if bool(student_cfg.get("use_foot_traj_head", False)):
            if foot_traj_target_obs_key is None:
                raise ValueError(
                    "VisionStudentTeacher requires foot_traj_target_obs_key when use_foot_traj_head=True."
                )
            if foot_traj_target_obs_key not in obs.keys():
                raise KeyError(
                    f"foot_traj_target_obs_key='{foot_traj_target_obs_key}' not found in observations. "
                    f"Available groups: {list(obs.keys())}"
                )
            inferred_dim = int(obs[foot_traj_target_obs_key].shape[-1])
            configured_dim = student_cfg.get("foot_traj_output_dim", None)
            if configured_dim is None:
                student_cfg["foot_traj_output_dim"] = inferred_dim
            elif int(configured_dim) != inferred_dim:
                raise ValueError(
                    f"Configured foot_traj_output_dim={configured_dim} does not match "
                    f"observation '{foot_traj_target_obs_key}' dim {inferred_dim}."
                )

        student_obs_groups = {
            "policy": list(obs_groups["policy"]),
            "policy_history": list(obs_groups["policy_history"]),
            "command_window": list(obs_groups["command_window"]),
            "critic": list(obs_groups.get("critic", obs_groups["policy"])),
        }
        if "vel_gt" in obs_groups:
            student_obs_groups["vel_gt"] = list(obs_groups["vel_gt"])
        if "anchor_gt" in obs_groups:
            student_obs_groups["anchor_gt"] = list(obs_groups["anchor_gt"])
        if "anchor_estimator" in obs_groups:
            student_obs_groups["anchor_estimator"] = list(obs_groups["anchor_estimator"])

        teacher_obs_groups = {
            "policy": list(obs_groups["teacher"]),
            "policy_history": list(obs_groups["teacher_policy_history"]),
            "command_window": list(obs_groups["teacher_command_window"]),
            "critic": list(obs_groups.get("teacher_critic", obs_groups.get("critic", obs_groups["teacher"]))),
        }
        if "teacher_vel_gt" in obs_groups:
            teacher_obs_groups["vel_gt"] = list(obs_groups["teacher_vel_gt"])
        elif "vel_gt" in obs_groups:
            teacher_obs_groups["vel_gt"] = list(obs_groups["vel_gt"])
        if "teacher_anchor_gt" in obs_groups:
            teacher_obs_groups["anchor_gt"] = list(obs_groups["teacher_anchor_gt"])
        elif "anchor_gt" in obs_groups:
            teacher_obs_groups["anchor_gt"] = list(obs_groups["anchor_gt"])
        if "teacher_anchor_estimator" in obs_groups:
            teacher_anchor_estimator = list(obs_groups["teacher_anchor_estimator"])
            if teacher_anchor_estimator:
                teacher_obs_groups["anchor_estimator"] = teacher_anchor_estimator
        if "vision" in obs_groups:
            student_obs_groups["vision"] = list(obs_groups["vision"])
            teacher_obs_groups["vision"] = list(obs_groups.get("teacher_vision", obs_groups["vision"]))

        self.student = VisionTransformerActorCritic(
            obs=obs,
            obs_groups=student_obs_groups,
            num_actions=num_actions,
            **student_cfg,
        )
        teacher_classes = {
            "TransformerActorCritic": TransformerActorCritic,
            "VisionTransformerActorCritic": VisionTransformerActorCritic,
        }
        if self.teacher_class_name not in teacher_classes:
            raise ValueError(
                "VisionStudentTeacher teacher_class_name must be one of "
                f"{sorted(teacher_classes)}. Got: {self.teacher_class_name}"
            )
        if self.teacher_class_name == "VisionTransformerActorCritic" and "vision" not in obs.keys():
            raise KeyError("vision-transformer teacher requires observation key 'vision'.")
        teacher_class = teacher_classes[self.teacher_class_name]
        self.teacher = teacher_class(
            obs=obs,
            obs_groups=teacher_obs_groups,
            num_actions=num_actions,
            **teacher_cfg,
        )
        self._freeze_teacher()

        if teacher_ckpt_path:
            self._load_teacher_checkpoint(Path(teacher_ckpt_path), strict=teacher_load_strict)

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
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def _load_teacher_checkpoint(self, ckpt_path: Path, strict: bool) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
        loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported teacher checkpoint format in: {ckpt_path}")
        checkpoint_metadata = loaded.get("policy_metadata") if isinstance(loaded, dict) else None
        if isinstance(checkpoint_metadata, dict) and checkpoint_metadata.get("policy_family") == "vision_student_teacher":
            nested_teacher = checkpoint_metadata.get("subpolicies", {}).get("teacher")
            if nested_teacher is None:
                nested_teacher = checkpoint_metadata.get("signature", {}).get("teacher")
            if isinstance(nested_teacher, dict):
                checkpoint_metadata = nested_teacher
                if "signature" not in checkpoint_metadata:
                    checkpoint_metadata = {"signature": checkpoint_metadata}
        if isinstance(checkpoint_metadata, dict):
            checkpoint_signature = checkpoint_metadata.get("signature", {})
            current_signature = self.teacher.get_checkpoint_metadata().get("signature", {})
            signature_keys = [
                "num_actions",
                "actor_obs_dim",
                "history_len",
                "cmd_len",
                "history_token_dim",
                "cmd_token_dim",
                "use_vel_estimator",
                "vel_output_dim",
                "use_anchor_estimator",
                "anchor_output_dim",
                "anchor_estimator_obs_dim",
            ]
            if isinstance(self.teacher, VisionTransformerActorCritic):
                signature_keys.extend(
                    [
                        "vision_encoder_type",
                        "map_height",
                        "map_width",
                        "dim_map_embed",
                        "height_vision_has_mask",
                        "depth_vision_channels",
                        "critic_use_vision",
                        "use_foot_traj_head",
                        "foot_traj_output_dim",
                    ]
                )
            signature_issues = [
                f"{key}: checkpoint={checkpoint_signature.get(key)} current={current_signature.get(key)}"
                for key in signature_keys
                if checkpoint_signature.get(key) is not None and checkpoint_signature.get(key) != current_signature.get(key)
            ]
            if signature_issues:
                raise ValueError(
                    f"{self.teacher_class_name} checkpoint does not match the configured teacher schema. "
                    + "; ".join(signature_issues)
                )
        else:
            print(
                f"[VisionStudentTeacher] Teacher checkpoint {ckpt_path} has no policy metadata. "
                "Relying on strict tensor loading only."
            )
        self._load_teacher_state_dict(state_dict, strict=strict)
        self.loaded_teacher = True
        print(f"[VisionStudentTeacher] Loaded {self.teacher_class_name} teacher checkpoint: {ckpt_path}")

    def _load_teacher_state_dict(self, state_dict: dict[str, Any], strict: bool) -> None:
        if any(key.startswith("teacher.") for key in state_dict.keys()):
            teacher_state = {
                key.replace("teacher.", "", 1): value for key, value in state_dict.items() if key.startswith("teacher.")
            }
        else:
            teacher_state = state_dict
        self.teacher.load_state_dict(teacher_state, strict=strict)
        self._freeze_teacher()

    def act(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act(obs)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act_inference(obs)

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        return self.student.infer_student_outputs(obs)

    def get_last_aux_outputs(self, *, clear: bool = True) -> dict[str, torch.Tensor]:
        return self.student.get_last_aux_outputs(clear=clear)

    def get_last_bridge_debug(self, *, clear: bool = True) -> dict[str, float]:
        debug = dict(self._last_bridge_debug)
        if clear:
            self._last_bridge_debug = {}
        return debug

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        student_metadata = self.student.get_checkpoint_metadata()
        teacher_metadata = self.teacher.get_checkpoint_metadata()
        return {
            "policy_class": self.__class__.__name__,
            "policy_family": "vision_student_teacher",
            "obs_schema": self._obs_schema,
            "signature": {
                "align_teacher_to_student_reference": bool(self.align_teacher_to_student_reference),
                "foot_traj_target_obs_key": self.foot_traj_target_obs_key,
                "teacher_class_name": self.teacher_class_name,
                "student": student_metadata.get("signature", {}),
                "teacher": teacher_metadata.get("signature", {}),
            },
            "subpolicies": {
                "student": student_metadata,
                "teacher": teacher_metadata,
            },
        }

    def normalize_velocity(self, v: torch.Tensor) -> torch.Tensor:
        return self.student.normalize_velocity(v)

    def normalize_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        return self.student.normalize_anchor(anchor)

    @staticmethod
    def _get_concat_seq(obs: TensorDict, groups: list[str], expected_seq_len: int | None = None) -> torch.Tensor:
        xs = []
        seq_len = expected_seq_len
        for key in groups:
            value = obs[key]
            if value.ndim == 2:
                if seq_len is None:
                    raise ValueError(
                        f"Cannot infer sequence length for observation '{key}'."
                    )
                if value.shape[-1] % seq_len != 0:
                    raise ValueError(
                        f"Observation '{key}' last dim {value.shape[-1]} is not divisible by seq_len={seq_len}."
                    )
                value = value.reshape(value.shape[0], seq_len, -1)
            elif value.ndim != 3:
                raise ValueError(f"Observation '{key}' must be 2D or 3D. Got shape: {tuple(value.shape)}")
            if seq_len is None:
                seq_len = value.shape[1]
            elif value.shape[1] != seq_len:
                raise ValueError(
                    f"Command-window obs groups must share the same sequence length. "
                    f"Got {seq_len} and {value.shape[1]}."
                )
            xs.append(value)
        if not xs:
            raise KeyError("Command-window observation set is empty.")
        return torch.cat(xs, dim=-1)

    def _align_teacher_actions(self, obs: TensorDict, teacher_actions: torch.Tensor) -> torch.Tensor:
        return align_teacher_actions(self, obs, teacher_actions)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            teacher_actions = self.teacher.act_inference(obs)
            return self._align_teacher_actions(obs, teacher_actions)

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
        if any(key.startswith("student.") for key in state_dict.keys()):
            nn.Module.load_state_dict(self, state_dict, strict=strict)
            self._freeze_teacher()
            self.loaded_teacher = True
            return True

        self._load_teacher_state_dict(state_dict, strict=strict)
        self.loaded_teacher = True
        return False
