# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from motion_tracking_rl.networks import (
    PerceptiveMotionAdapterTracker,
    PerceptiveMotionTokenTracker,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from motion_tracking_rl.registry import register_algorithm
from motion_tracking_rl.storage import RolloutStorage
from motion_tracking_rl.utils.utils import resolve_optimizer


@register_algorithm("Distillation", compat_name="distillation")
class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent | PerceptiveMotionAdapterTracker | PerceptiveMotionTokenTracker

    def __init__(
        self,
        policy: StudentTeacher | StudentTeacherRecurrent | PerceptiveMotionAdapterTracker | PerceptiveMotionTokenTracker,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        behavior_loss_coef: float = 1.0,
        latent_loss_coef: float = 0.0,
        delta_z_loss_coef: float = 0.0,
        latent_cosine_loss_coef: float = 0.0,
        latent_norm_loss_coef: float = 0.0,
        flat_identity_loss_coef: float = 0.0,
        delta_smooth_loss_coef: float = 0.0,
        motion_loss_coef: float = 0.0,
        anchor_loss_coef: float = 0.0,
        vel_loss_coef: float = 0.0,
        vel_loss_type: str = "huber",
        vel_loss_delta: float = 1.0,
        anchor_est_loss_coef: float = 0.0,
        anchor_est_loss_type: str = "huber",
        anchor_est_loss_delta: float = 1.0,
        foot_traj_loss_coef: float = 0.0,
        foot_traj_loss_type: str = "huber",
        foot_traj_loss_delta: float = 1.0,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.policy.to(self.device)
        self.storage = None

        self.trainable_params = [param for param in self.policy.parameters() if param.requires_grad]
        self.has_trainable_params = bool(self.trainable_params)
        if self.has_trainable_params:
            self.optimizer = resolve_optimizer(optimizer)(self.trainable_params, lr=learning_rate)
        else:
            self.optimizer = None
            print("[Distillation] No trainable parameters; running metric-only updates.")

        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.behavior_loss_coef = float(behavior_loss_coef)
        self.latent_loss_coef = float(latent_loss_coef)
        self.delta_z_loss_coef = float(delta_z_loss_coef)
        self.latent_cosine_loss_coef = float(latent_cosine_loss_coef)
        self.latent_norm_loss_coef = float(latent_norm_loss_coef)
        self.flat_identity_loss_coef = float(flat_identity_loss_coef)
        self.delta_smooth_loss_coef = float(delta_smooth_loss_coef)
        self.motion_loss_coef = float(motion_loss_coef)
        self.anchor_loss_coef = float(anchor_loss_coef)
        self.vel_loss_coef = float(vel_loss_coef)
        self.vel_loss_type = str(vel_loss_type)
        self.vel_loss_delta = float(vel_loss_delta)
        self.anchor_est_loss_coef = float(anchor_est_loss_coef)
        self.anchor_est_loss_type = str(anchor_est_loss_type)
        self.anchor_est_loss_delta = float(anchor_est_loss_delta)
        self.foot_traj_loss_coef = float(foot_traj_loss_coef)
        self.foot_traj_loss_type = str(foot_traj_loss_type)
        self.foot_traj_loss_delta = float(foot_traj_loss_delta)

        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")
        if self.vel_loss_type not in ("mse", "huber"):
            raise ValueError(f"Unknown vel_loss_type: {vel_loss_type}. Supported types are: ['mse', 'huber']")
        if self.anchor_est_loss_type not in ("mse", "huber"):
            raise ValueError(
                f"Unknown anchor_est_loss_type: {anchor_est_loss_type}. Supported types are: ['mse', 'huber']"
            )
        if self.foot_traj_loss_type not in ("mse", "huber"):
            raise ValueError(
                f"Unknown foot_traj_loss_type: {foot_traj_loss_type}. Supported types are: ['mse', 'huber']"
            )

        self.num_updates = 0

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int],
    ) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.actions = self.policy.act(obs).detach()
        infer_teacher_action = getattr(self.policy, "infer_teacher_action", None)
        if callable(infer_teacher_action):
            privileged_actions = infer_teacher_action(obs)
        else:
            privileged_actions = self.policy.evaluate(obs)
        self.transition.privileged_actions = privileged_actions.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _resolve_obs_group_keys(self, group_name: str) -> list[str]:
        obs_groups = getattr(self.policy, "obs_groups", None)
        if isinstance(obs_groups, dict) and group_name in obs_groups:
            return list(obs_groups[group_name])

        student = getattr(self.policy, "student", None)
        student_obs_groups = getattr(student, "obs_groups", None)
        if isinstance(student_obs_groups, dict) and group_name in student_obs_groups:
            return list(student_obs_groups[group_name])

        return []

    def _compute_velocity_loss(self, aux: dict[str, torch.Tensor], obs: TensorDict) -> torch.Tensor:
        v_hat = aux.get("v_hat", None)
        if v_hat is None:
            raise ValueError("Velocity loss was enabled, but the policy did not emit auxiliary output 'v_hat'.")

        v_gt = None
        vel_keys = self._resolve_obs_group_keys("vel_gt")
        if vel_keys and all(key in obs.keys() for key in vel_keys):
            v_gt = torch.cat([obs[key] for key in vel_keys], dim=-1)
        elif "vel_gt" in obs.keys():
            v_gt = obs["vel_gt"]

        if v_gt is None:
            raise KeyError(
                "Velocity GT observations not found. Expected obs_groups['vel_gt'] to map to an observation "
                "group present in env observations (for example 'vel_gt_xyz')."
            )

        if v_hat.shape != v_gt.shape:
            raise ValueError(
                f"Velocity shapes mismatch: v_hat {tuple(v_hat.shape)} vs v_gt {tuple(v_gt.shape)}. "
                "Check vel_estimator_output_dim and obs_groups['vel_gt']."
            )

        if hasattr(self.policy, "normalize_velocity"):
            v_hat = self.policy.normalize_velocity(v_hat)
            v_gt = self.policy.normalize_velocity(v_gt)

        if self.vel_loss_type == "huber":
            return F.huber_loss(v_hat, v_gt, delta=self.vel_loss_delta)
        return F.mse_loss(v_hat, v_gt)

    def _compute_anchor_est_loss(self, aux: dict[str, torch.Tensor], obs: TensorDict) -> torch.Tensor:
        anchor_hat = aux.get("anchor_hat", None)
        if anchor_hat is None:
            raise ValueError(
                "Anchor estimator loss was enabled, but the policy did not emit auxiliary output 'anchor_hat'."
            )

        anchor_gt = None
        anchor_keys = self._resolve_obs_group_keys("anchor_gt")
        if anchor_keys and all(key in obs.keys() for key in anchor_keys):
            anchor_gt = torch.cat([obs[key] for key in anchor_keys], dim=-1)
        elif "anchor_gt" in obs.keys():
            anchor_gt = obs["anchor_gt"]

        if anchor_gt is None:
            raise KeyError(
                "Anchor GT observations not found. Expected obs_groups['anchor_gt'] to map to an observation "
                "group present in env observations (for example 'anchor_gt')."
            )

        if anchor_hat.shape != anchor_gt.shape:
            raise ValueError(
                f"Anchor shapes mismatch: anchor_hat {tuple(anchor_hat.shape)} vs anchor_gt {tuple(anchor_gt.shape)}. "
                "Check anchor_estimator_output_dim and obs_groups['anchor_gt']."
            )

        if hasattr(self.policy, "normalize_anchor"):
            anchor_hat = self.policy.normalize_anchor(anchor_hat)
            anchor_gt = self.policy.normalize_anchor(anchor_gt)

        if self.anchor_est_loss_type == "huber":
            return F.huber_loss(anchor_hat, anchor_gt, delta=self.anchor_est_loss_delta)
        return F.mse_loss(anchor_hat, anchor_gt)

    def _compute_foot_traj_loss(
        self, outputs: dict[str, torch.Tensor], obs: TensorDict
    ) -> tuple[torch.Tensor, float, float]:
        foot_traj_key = getattr(self.policy, "foot_traj_target_obs_key", None)
        if foot_traj_key is None:
            raise ValueError("foot_traj_loss was enabled, but the policy does not define foot_traj_target_obs_key.")
        if foot_traj_key not in obs.keys():
            raise KeyError(
                f"foot_traj_target_obs_key='{foot_traj_key}' not found in observations. Available groups: {list(obs.keys())}"
            )
        if "foot_traj" not in outputs:
            raise ValueError("foot_traj_loss was enabled, but the policy did not emit output 'foot_traj'.")

        foot_traj_pred = outputs["foot_traj"]
        foot_traj_target = obs[foot_traj_key]
        if foot_traj_pred.shape != foot_traj_target.shape:
            raise ValueError(
                f"Foot trajectory shapes mismatch: pred {tuple(foot_traj_pred.shape)} vs "
                f"target {tuple(foot_traj_target.shape)}."
            )

        target_abs_mean = float(foot_traj_target.detach().abs().mean().item())
        pred_abs_mean = float(foot_traj_pred.detach().abs().mean().item())
        if self.foot_traj_loss_type == "huber":
            loss = F.huber_loss(foot_traj_pred, foot_traj_target, delta=self.foot_traj_loss_delta)
        else:
            loss = F.mse_loss(foot_traj_pred, foot_traj_target)
        return loss, target_abs_mean, pred_abs_mean

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_motion_loss = 0.0
        mean_anchor_loss = 0.0
        mean_vel_loss = 0.0
        mean_anchor_est_loss = 0.0
        mean_foot_traj_loss = 0.0
        mean_foot_traj_target_abs = 0.0
        mean_foot_traj_pred_abs = 0.0
        mean_latent_loss = 0.0
        mean_latent_cosine_loss = 0.0
        mean_latent_norm_loss = 0.0
        mean_latent_cosine = 0.0
        mean_z_task_abs = 0.0
        mean_z_opt_abs = 0.0
        mean_z_flat_abs = 0.0
        mean_delta_z_abs = 0.0
        mean_delta_z_loss = 0.0
        mean_delta_z_target_abs = 0.0
        mean_identity_residual_abs = 0.0
        mean_latent_norm_ratio = 0.0
        mean_flat_identity_loss = 0.0
        mean_delta_smooth_loss = 0.0
        mean_gate = 0.0
        mean_gate_abs = 0.0
        accum_loss: torch.Tensor | None = None
        accum_steps = 0
        cnt = 0
        motion_cnt = 0
        anchor_cnt = 0
        vel_cnt = 0
        anchor_est_cnt = 0
        foot_traj_cnt = 0
        latent_cnt = 0
        delta_z_cnt = 0
        flat_identity_cnt = 0
        delta_smooth_cnt = 0
        gate_cnt = 0

        def optimizer_step() -> None:
            nonlocal accum_loss, accum_steps
            if accum_steps == 0 or accum_loss is None:
                return

            if self.optimizer is not None and self.has_trainable_params:
                self.optimizer.zero_grad()
                (accum_loss / accum_steps).backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                self.optimizer.step()

            self.policy.detach_hidden_states()
            accum_loss = None
            accum_steps = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                outputs = self.policy.infer_student_outputs(obs)
                actions = outputs["action"]
                behavior_loss = self.loss_fn(actions, privileged_actions)
                total_loss = self.behavior_loss_coef * behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if "z_task" in outputs and "z_opt" in outputs:
                    z_task = outputs["z_task"]
                    z_opt = outputs["z_opt"].detach()
                    latent_loss = F.mse_loss(z_task, z_opt)
                    latent_cosine_values = F.cosine_similarity(z_task, z_opt, dim=-1)
                    latent_cosine_loss = (1.0 - latent_cosine_values).mean()
                    latent_norm_loss = F.mse_loss(z_task.norm(dim=-1), z_opt.norm(dim=-1))

                    total_loss = total_loss + self.latent_loss_coef * latent_loss
                    total_loss = total_loss + self.latent_cosine_loss_coef * latent_cosine_loss
                    total_loss = total_loss + self.latent_norm_loss_coef * latent_norm_loss

                    mean_latent_loss += latent_loss.item()
                    mean_latent_cosine_loss += latent_cosine_loss.item()
                    mean_latent_norm_loss += latent_norm_loss.item()
                    mean_latent_cosine += latent_cosine_values.detach().mean().item()
                    mean_z_task_abs += z_task.detach().abs().mean().item()
                    mean_z_opt_abs += z_opt.detach().abs().mean().item()
                    if "z_flat" in outputs:
                        mean_z_flat_abs += outputs["z_flat"].detach().abs().mean().item()
                    if "delta_z" in outputs:
                        mean_delta_z_abs += outputs["delta_z"].detach().abs().mean().item()
                    if "identity_residual" in outputs:
                        mean_identity_residual_abs += outputs["identity_residual"].detach().abs().mean().item()
                    mean_latent_norm_ratio += (
                        z_task.detach().norm(dim=-1) / z_opt.detach().norm(dim=-1).clamp_min(1.0e-6)
                    ).mean().item()
                    latent_cnt += 1

                if self.delta_z_loss_coef != 0.0:
                    if "delta_z" not in outputs or "delta_z_target" not in outputs:
                        raise ValueError(
                            "delta_z_loss_coef was enabled, but the policy did not emit both "
                            "'delta_z' and 'delta_z_target'."
                        )

                    delta_z = outputs["delta_z"]
                    delta_z_target = outputs["delta_z_target"].detach()
                    if delta_z.shape != delta_z_target.shape:
                        raise ValueError(
                            f"delta_z shape {tuple(delta_z.shape)} does not match "
                            f"delta_z_target shape {tuple(delta_z_target.shape)}."
                        )

                    delta_z_loss = F.mse_loss(delta_z, delta_z_target)
                    total_loss = total_loss + self.delta_z_loss_coef * delta_z_loss
                    mean_delta_z_loss += delta_z_loss.item()
                    mean_delta_z_target_abs += delta_z_target.abs().mean().item()
                    delta_z_cnt += 1

                if self.flat_identity_loss_coef != 0.0 and "identity_residual" in outputs:
                    identity_residual = outputs["identity_residual"]
                    residual_per_env = identity_residual.pow(2).mean(dim=-1)
                    flat_identity_mask = outputs.get("flat_identity_mask", None)
                    if flat_identity_mask is not None:
                        flat_identity_mask = flat_identity_mask.float().view(-1)
                        flat_identity_loss = (
                            (residual_per_env * flat_identity_mask).sum()
                            / flat_identity_mask.sum().clamp_min(1.0)
                        )
                        total_loss = total_loss + self.flat_identity_loss_coef * flat_identity_loss
                        mean_flat_identity_loss += flat_identity_loss.item()
                        flat_identity_cnt += 1

                if self.delta_smooth_loss_coef != 0.0 and "delta_z" in outputs and "delta_z_prev" in outputs:
                    delta_prev_is_temporal = outputs.get("delta_z_prev_is_temporal", False)
                    if torch.is_tensor(delta_prev_is_temporal):
                        delta_prev_is_temporal = bool(delta_prev_is_temporal.detach().all().item())

                    if bool(delta_prev_is_temporal):
                        delta_smooth_loss = F.mse_loss(outputs["delta_z"], outputs["delta_z_prev"].detach())
                        total_loss = total_loss + self.delta_smooth_loss_coef * delta_smooth_loss
                        mean_delta_smooth_loss += delta_smooth_loss.item()
                        delta_smooth_cnt += 1

                if "gate" in outputs:
                    gate = outputs["gate"].detach()
                    mean_gate += gate.mean().item()
                    mean_gate_abs += gate.abs().mean().item()
                    gate_cnt += 1

                motion_key = getattr(self.policy, "motion_target_obs_key", None)
                if self.motion_loss_coef != 0.0 and motion_key is not None and "motion" in outputs:
                    motion_loss = self.loss_fn(outputs["motion"], obs[motion_key])
                    total_loss = total_loss + self.motion_loss_coef * motion_loss
                    mean_motion_loss += motion_loss.item()
                    motion_cnt += 1

                anchor_key = getattr(self.policy, "anchor_target_obs_key", None)
                if self.anchor_loss_coef != 0.0 and anchor_key is not None and "anchor" in outputs:
                    anchor_loss = self.loss_fn(outputs["anchor"], obs[anchor_key])
                    total_loss = total_loss + self.anchor_loss_coef * anchor_loss
                    mean_anchor_loss += anchor_loss.item()
                    anchor_cnt += 1

                aux = None
                if self.vel_loss_coef != 0.0 or self.anchor_est_loss_coef != 0.0:
                    if not hasattr(self.policy, "get_last_aux_outputs"):
                        raise AttributeError(
                            "Auxiliary estimator loss was enabled, but the policy does not expose get_last_aux_outputs()."
                        )
                    aux = self.policy.get_last_aux_outputs(clear=True)

                if self.vel_loss_coef != 0.0:
                    vel_loss = self._compute_velocity_loss(aux, obs)
                    total_loss = total_loss + self.vel_loss_coef * vel_loss
                    mean_vel_loss += vel_loss.item()
                    vel_cnt += 1

                if self.anchor_est_loss_coef != 0.0:
                    anchor_est_loss = self._compute_anchor_est_loss(aux, obs)
                    total_loss = total_loss + self.anchor_est_loss_coef * anchor_est_loss
                    mean_anchor_est_loss += anchor_est_loss.item()
                    anchor_est_cnt += 1

                if self.foot_traj_loss_coef != 0.0:
                    foot_traj_loss, foot_traj_target_abs, foot_traj_pred_abs = self._compute_foot_traj_loss(outputs, obs)
                    total_loss = total_loss + self.foot_traj_loss_coef * foot_traj_loss
                    mean_foot_traj_loss += foot_traj_loss.item()
                    mean_foot_traj_target_abs += foot_traj_target_abs
                    mean_foot_traj_pred_abs += foot_traj_pred_abs
                    foot_traj_cnt += 1

                accum_loss = total_loss if accum_loss is None else accum_loss + total_loss
                accum_steps += 1

                if accum_steps >= self.gradient_length:
                    optimizer_step()

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        optimizer_step()

        mean_behavior_loss /= max(cnt, 1)
        mean_motion_loss = mean_motion_loss / motion_cnt if motion_cnt > 0 else 0.0
        mean_anchor_loss = mean_anchor_loss / anchor_cnt if anchor_cnt > 0 else 0.0
        mean_vel_loss = mean_vel_loss / vel_cnt if vel_cnt > 0 else 0.0
        mean_anchor_est_loss = mean_anchor_est_loss / anchor_est_cnt if anchor_est_cnt > 0 else 0.0
        mean_foot_traj_loss = mean_foot_traj_loss / foot_traj_cnt if foot_traj_cnt > 0 else 0.0
        mean_foot_traj_target_abs = mean_foot_traj_target_abs / foot_traj_cnt if foot_traj_cnt > 0 else 0.0
        mean_foot_traj_pred_abs = mean_foot_traj_pred_abs / foot_traj_cnt if foot_traj_cnt > 0 else 0.0
        mean_latent_loss = mean_latent_loss / latent_cnt if latent_cnt > 0 else 0.0
        mean_latent_cosine_loss = mean_latent_cosine_loss / latent_cnt if latent_cnt > 0 else 0.0
        mean_latent_norm_loss = mean_latent_norm_loss / latent_cnt if latent_cnt > 0 else 0.0
        mean_latent_cosine = mean_latent_cosine / latent_cnt if latent_cnt > 0 else 0.0
        mean_z_task_abs = mean_z_task_abs / latent_cnt if latent_cnt > 0 else 0.0
        mean_z_opt_abs = mean_z_opt_abs / latent_cnt if latent_cnt > 0 else 0.0
        mean_z_flat_abs = mean_z_flat_abs / latent_cnt if latent_cnt > 0 else 0.0
        mean_delta_z_abs = mean_delta_z_abs / latent_cnt if latent_cnt > 0 else 0.0
        mean_delta_z_loss = mean_delta_z_loss / delta_z_cnt if delta_z_cnt > 0 else 0.0
        mean_delta_z_target_abs = mean_delta_z_target_abs / delta_z_cnt if delta_z_cnt > 0 else 0.0
        mean_identity_residual_abs = mean_identity_residual_abs / latent_cnt if latent_cnt > 0 else 0.0
        mean_latent_norm_ratio = mean_latent_norm_ratio / latent_cnt if latent_cnt > 0 else 0.0
        mean_flat_identity_loss = mean_flat_identity_loss / flat_identity_cnt if flat_identity_cnt > 0 else 0.0
        mean_delta_smooth_loss = mean_delta_smooth_loss / delta_smooth_cnt if delta_smooth_cnt > 0 else 0.0
        mean_gate = mean_gate / gate_cnt if gate_cnt > 0 else 0.0
        mean_gate_abs = mean_gate_abs / gate_cnt if gate_cnt > 0 else 0.0
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        loss_dict = {
            "behavior": mean_behavior_loss,
            "behavior_loss": mean_behavior_loss,
            "action_loss": mean_behavior_loss,
            "motion": mean_motion_loss,
            "anchor": mean_anchor_loss,
        }
        if self.vel_loss_coef != 0.0:
            loss_dict["vel_estimator"] = mean_vel_loss
        if self.anchor_est_loss_coef != 0.0:
            loss_dict["anchor_estimator"] = mean_anchor_est_loss
        if self.foot_traj_loss_coef != 0.0:
            loss_dict["foot_traj"] = mean_foot_traj_loss
            loss_dict["foot_traj_target_abs_mean"] = mean_foot_traj_target_abs
            loss_dict["foot_traj_pred_abs_mean"] = mean_foot_traj_pred_abs
        if latent_cnt > 0:
            loss_dict["latent"] = mean_latent_loss
            loss_dict["latent_loss"] = mean_latent_loss
            loss_dict["latent_cosine_loss"] = mean_latent_cosine_loss
            loss_dict["latent_norm"] = mean_latent_norm_loss
            loss_dict["latent_norm_loss"] = mean_latent_norm_loss
            loss_dict["latent_cosine"] = mean_latent_cosine
            loss_dict["z_task_abs_mean"] = mean_z_task_abs
            loss_dict["z_opt_abs_mean"] = mean_z_opt_abs
            loss_dict["z_flat_abs_mean"] = mean_z_flat_abs
            loss_dict["delta_z_abs_mean"] = mean_delta_z_abs
            loss_dict["identity_residual_abs_mean"] = mean_identity_residual_abs
            loss_dict["latent_norm_ratio"] = mean_latent_norm_ratio
        if delta_z_cnt > 0:
            loss_dict["delta_z"] = mean_delta_z_loss
            loss_dict["delta_z_loss"] = mean_delta_z_loss
            loss_dict["delta_z_target_abs_mean"] = mean_delta_z_target_abs
        if flat_identity_cnt > 0:
            loss_dict["flat_identity"] = mean_flat_identity_loss
            loss_dict["identity_loss"] = mean_flat_identity_loss
        if delta_smooth_cnt > 0:
            loss_dict["delta_smooth"] = mean_delta_smooth_loss
            loss_dict["delta_smoothness_loss"] = mean_delta_smooth_loss
        if gate_cnt > 0:
            loss_dict["gate_mean"] = mean_gate
            loss_dict["gate_abs_mean"] = mean_gate_abs
        return loss_dict

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.trainable_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.trainable_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
