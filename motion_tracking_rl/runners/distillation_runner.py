# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F
from collections import deque
from tensordict import TensorDict

import motion_tracking_rl
from motion_tracking_rl.algorithms import Distillation
from motion_tracking_rl.env import VecEnv
from motion_tracking_rl.networks import (
    StudentTeacher,
    StudentTeacherRecurrent,
    SonicDiffusionStudentTeacher,
    VisionStudentTeacher,
    VisionAblationStudentTeacher,
    PerceptiveMotionAdapterTracker,
    PerceptiveMotionTokenTracker,
)
from motion_tracking_rl.registry import NETWORKS, ALGORITHMS, register_runner
from motion_tracking_rl.runners import OnPolicyRunner
from motion_tracking_rl.runners.on_policy_runner import _resolve_class
from motion_tracking_rl.utils import resolve_obs_groups, store_code_state


@register_runner("distillation")
@register_runner("DistillationRunner")
class DistillationRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.debug_rollout_action_stats = bool(self.cfg.get("debug_rollout_action_stats", False))
        self.debug_rollout_action_print_freq = max(int(self.cfg.get("debug_rollout_action_print_freq", 1)), 1)
        self.debug_rollout_action_steps = max(int(self.cfg.get("debug_rollout_action_steps", 1)), 1)
        self.debug_use_teacher_actions_for_env_step = bool(self.cfg.get("debug_use_teacher_actions_for_env_step", False))

        # DAgger-style mixed rollout
        self.student_mean_for_env_step = bool(self.cfg.get("student_mean_for_env_step", False))
        self.teacher_mix_start = float(self.cfg.get("teacher_mix_start", 1.0))
        self.teacher_mix_end = float(self.cfg.get("teacher_mix_end", 0.0))
        self.teacher_mix_anneal_iters = int(self.cfg.get("teacher_mix_anneal_iters", 0))
        self.teacher_env_mask: torch.Tensor | None = None

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["teacher"])

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [motion_tracking_rl.__file__]

    @staticmethod
    def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
        flat = tensor.detach().float().reshape(-1)
        finite_mask = torch.isfinite(flat)
        finite_count = int(finite_mask.sum().item())
        total = int(flat.numel())
        if finite_count == 0:
            return f"{name}: shape={tuple(tensor.shape)} finite=0/{total}"
        finite = flat[finite_mask]
        return (
            f"{name}: shape={tuple(tensor.shape)} finite={finite_count}/{total} "
            f"min={finite.min().item():.3e} max={finite.max().item():.3e} "
            f"mean={finite.mean().item():.3e} absmax={finite.abs().max().item():.3e}"
        )

    def _sample_teacher_env_mask(self, teacher_mix: float, env_ids: torch.Tensor | None = None) -> None:
        """Sample a per-environment hard-switch mask for teacher vs student rollout control."""
        if self.teacher_env_mask is None:
            self.teacher_env_mask = torch.zeros(self.env.num_envs, 1, dtype=torch.bool, device=self.device)

        teacher_mix = float(min(1.0, max(0.0, teacher_mix)))
        if env_ids is None:
            target = slice(None)
            num_targets = self.env.num_envs
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
            if env_ids.numel() == 0:
                return
            target = env_ids
            num_targets = int(env_ids.numel())

        if teacher_mix <= 0.0:
            self.teacher_env_mask[target] = False
        elif teacher_mix >= 1.0:
            self.teacher_env_mask[target] = True
        else:
            self.teacher_env_mask[target] = torch.rand(num_targets, 1, device=self.device) < teacher_mix

    @staticmethod
    def _bridge_debug_scalar(value) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _collect_bridge_debug(self) -> dict[str, float]:
        debug: dict[str, float] = {}
        policy = getattr(self.alg, "policy", None)
        if policy is not None and hasattr(policy, "get_last_bridge_debug"):
            try:
                for name, value in policy.get_last_bridge_debug(clear=True).items():
                    scalar = self._bridge_debug_scalar(value)
                    if scalar is not None:
                        debug[name] = scalar
            except Exception as exc:
                if self.debug_rollout_action_stats:
                    print(f"[Distill][DEBUG] Failed to read policy bridge debug: {exc}")

        env = getattr(self.env, "unwrapped", self.env)
        command_manager = getattr(env, "command_manager", None)
        action_manager = getattr(env, "action_manager", None)
        if command_manager is None or action_manager is None:
            return debug

        try:
            motion_cmd = command_manager.get_term("motion")
            student_cmd = command_manager.get_term("student_motion")
            action_term = action_manager.get_term("joint_pos")
        except Exception as exc:
            if self.debug_rollout_action_stats:
                print(f"[Distill][DEBUG] Failed to access env bridge terms: {exc}")
            return debug

        motion_ids = getattr(motion_cmd, "motion_ids", None)
        student_motion_ids = getattr(student_cmd, "motion_ids", None)
        if isinstance(motion_ids, torch.Tensor) and isinstance(student_motion_ids, torch.Tensor):
            debug["bridge_motion_id_mismatch_frac"] = float((motion_ids != student_motion_ids).float().mean().item())

        motion_frames = getattr(motion_cmd, "frame_ids", None)
        student_frames = getattr(student_cmd, "frame_ids", None)
        if isinstance(motion_frames, torch.Tensor) and isinstance(student_frames, torch.Tensor):
            debug["bridge_frame_id_abs_diff"] = float((motion_frames.float() - student_frames.float()).abs().mean().item())

        motion_qref = getattr(motion_cmd, "joint_pos", None)
        student_qref = getattr(student_cmd, "joint_pos", None)
        if isinstance(motion_qref, torch.Tensor) and isinstance(student_qref, torch.Tensor):
            qref_delta = motion_qref.float() - student_qref.float()
            debug["bridge_env_qref_mae"] = float(qref_delta.abs().mean().item())
            debug["bridge_env_qref_max_abs"] = float(qref_delta.abs().max().item())

            offset = getattr(action_term, "_offset", None)
            if isinstance(offset, torch.Tensor):
                motion_err = (offset.float() - motion_qref.float()).abs().mean(dim=-1)
                student_err = (offset.float() - student_qref.float()).abs().mean(dim=-1)
                debug["bridge_offset_vs_motion_mae"] = float(motion_err.mean().item())
                debug["bridge_offset_vs_student_mae"] = float(student_err.mean().item())
                debug["bridge_offset_prefers_motion_frac"] = float((motion_err < student_err).float().mean().item())
                debug["bridge_offset_prefers_student_frac"] = float((student_err < motion_err).float().mean().item())
                debug["bridge_offset_tie_frac"] = float((motion_err == student_err).float().mean().item())

        return debug

    def _print_bridge_config_debug(self) -> None:
        env = getattr(self.env, "unwrapped", self.env)
        command_manager = getattr(env, "command_manager", None)
        action_manager = getattr(env, "action_manager", None)
        if command_manager is None or action_manager is None:
            print("[Distill][DEBUG] bridge_config unavailable: missing command_manager or action_manager")
            return

        try:
            motion_cmd = command_manager.get_term("motion")
            student_cmd = command_manager.get_term("student_motion")
            action_term = action_manager.get_term("joint_pos")
            action_scale = getattr(getattr(action_term, "cfg", None), "scale", None)
            print(
                "[Distill][DEBUG] bridge_config "
                f"motion_cls={motion_cmd.__class__.__name__} "
                f"student_cls={student_cmd.__class__.__name__} "
                f"action_scale={action_scale} "
                f"motion_offset_with_ref={getattr(motion_cmd.cfg, 'update_action_offset_with_ref', None)} "
                f"student_offset_with_ref={getattr(student_cmd.cfg, 'update_action_offset_with_ref', None)}"
            )
            if student_cmd.__class__.__name__ == "SyncedStudentMultiMotionCommandV2":
                print(
                    "[Distill][DEBUG] student_motion is SyncedStudentMultiMotionCommandV2; "
                    "use the rollout bridge metrics to verify whether the live joint_pos offset follows "
                    "student_motion or motion."
                )
        except Exception as exc:
            print(f"[Distill][DEBUG] bridge_config unavailable: {exc}")


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")
        if self.debug_rollout_action_stats:
            teacher_param_count = sum(p.numel() for p in self.alg.policy.teacher.parameters())
            teacher_trainable_count = sum(p.numel() for p in self.alg.policy.teacher.parameters() if p.requires_grad)
            student_param_count = sum(
                p.numel() for name, p in self.alg.policy.named_parameters() if p.requires_grad and not name.startswith("teacher.")
            )
            print(
                "[Distill][DEBUG] teacher_loaded="
                f"{self.alg.policy.loaded_teacher}, teacher_ckpt_path={self.alg.policy.teacher_ckpt_path}, "
                f"teacher_params={teacher_param_count}, teacher_trainable={teacher_trainable_count}, "
                f"student_params={student_param_count}, use_teacher_env_step={self.debug_use_teacher_actions_for_env_step}"
            )
            self._print_bridge_config_debug()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Pre-compute whether we use the new mixed-rollout path
        use_mixed_rollout = self.teacher_mix_anneal_iters > 0

        if use_mixed_rollout:
            print(
                f"[Distill] Mixed rollout enabled: teacher_prob {self.teacher_mix_start:.2f} → "
                f"{self.teacher_mix_end:.2f} over {self.teacher_mix_anneal_iters} iters, "
                f"student_mean_for_env_step={self.student_mean_for_env_step}"
            )
            self._sample_teacher_env_mask(self.teacher_mix_start)

        for it in range(start_iter, tot_iter):
            start = time.time()
            rollout_student_env_action_mse_sum = 0.0
            rollout_student_env_action_mae_sum = 0.0
            rollout_env_action_mse_sum = 0.0
            rollout_env_action_mae_sum = 0.0
            rollout_teacher_env_frac_sum = 0.0
            rollout_control_steps = 0
            rollout_teacher_reward_sum = 0.0
            rollout_teacher_reward_count = 0
            rollout_student_reward_sum = 0.0
            rollout_student_reward_count = 0
            rollout_bridge_metric_sums: dict[str, float] = {}
            rollout_bridge_metric_counts: dict[str, int] = {}

            # Compute current teacher mixing ratio
            if use_mixed_rollout:
                progress = min(1.0, max(0.0, (it - start_iter) / self.teacher_mix_anneal_iters))
                teacher_mix = self.teacher_mix_start + (self.teacher_mix_end - self.teacher_mix_start) * progress
            else:
                teacher_mix = 0.0

            # Rollout
            with torch.inference_mode():
                for rollout_step in range(self.num_steps_per_env):
                    # Sample actions (stores transition.actions and transition.privileged_actions)
                    actions = self.alg.act(obs)
                    teacher_actions = self.alg.transition.privileged_actions

                    should_debug = (
                        self.debug_rollout_action_stats
                        and (it % self.debug_rollout_action_print_freq == 0)
                        and (rollout_step < self.debug_rollout_action_steps)
                    )
                    student_env_actions = self.alg.policy.act_inference(obs) if self.student_mean_for_env_step else actions
                    if should_debug:
                        print(
                            f"[Distill][DEBUG][it={it}][rollout_step={rollout_step}] "
                            + self._tensor_stats("student_action", actions)
                        )
                        print(
                            f"[Distill][DEBUG][it={it}][rollout_step={rollout_step}] "
                            + self._tensor_stats("student_env_action", student_env_actions)
                        )
                        print(
                            f"[Distill][DEBUG][it={it}][rollout_step={rollout_step}] "
                            + self._tensor_stats("teacher_action", teacher_actions)
                        )
                        print(
                            f"[Distill][DEBUG][it={it}][rollout_step={rollout_step}] "
                            + self._tensor_stats("student_env_delta", student_env_actions - teacher_actions)
                        )
                        mse = F.mse_loss(student_env_actions, teacher_actions).item()
                        mae = (student_env_actions - teacher_actions).abs().mean().item()
                        clip_limit = getattr(self.alg.policy.student, "action_clip", None) if self.alg.policy.student is not None else None
                        student_clip_frac = 0.0
                        teacher_clip_frac = 0.0
                        if clip_limit is not None:
                            student_clip_frac = (student_env_actions.abs() > clip_limit).float().mean().item()
                            teacher_clip_frac = (teacher_actions.abs() > clip_limit).float().mean().item()
                        mode_info = "mode_ids=NA"
                        if "encoder_mode_4" in obs.keys():
                            mode_ids = torch.round(obs["encoder_mode_4"][:, 0]).long()
                            uniq, counts = torch.unique(mode_ids, return_counts=True)
                            mode_pairs = ", ".join(f"{int(u.item())}:{int(c.item())}" for u, c in zip(uniq, counts))
                            mode_info = f"mode_ids={{ {mode_pairs} }}"
                        teacher_env_frac = (
                            self.teacher_env_mask.float().mean().item()
                            if use_mixed_rollout and self.teacher_env_mask is not None
                            else float(self.debug_use_teacher_actions_for_env_step)
                        )
                        print(
                            f"[Distill][DEBUG][it={it}][rollout_step={rollout_step}] "
                            f"mse={mse:.3e} mae={mae:.3e} teacher_mix={teacher_mix:.3f} "
                            f"teacher_env_frac={teacher_env_frac:.3f} "
                            f"student_clip_frac={student_clip_frac:.3e} teacher_clip_frac={teacher_clip_frac:.3e} "
                            + mode_info
                        )

                    student_env_delta = student_env_actions - teacher_actions
                    rollout_student_env_action_mse_sum += float(F.mse_loss(student_env_actions, teacher_actions).item())
                    rollout_student_env_action_mae_sum += float(student_env_delta.abs().mean().item())

                    # Determine the action used to step the environment
                    if self.debug_use_teacher_actions_for_env_step:
                        # Legacy: 100% teacher
                        teacher_step_mask = torch.ones(self.env.num_envs, 1, dtype=torch.bool, device=self.device)
                        env_actions = teacher_actions
                    elif use_mixed_rollout:
                        # DAgger-style annealing: per-environment hard switch between teacher and student.
                        if self.teacher_env_mask is None:
                            self._sample_teacher_env_mask(teacher_mix)
                        teacher_step_mask = self.teacher_env_mask.clone()
                        teacher_mask = teacher_step_mask.expand_as(teacher_actions)
                        env_actions = torch.where(teacher_mask, teacher_actions, student_env_actions)
                    elif self.student_mean_for_env_step:
                        # Pure student mean (no mixing, no noise)
                        teacher_step_mask = torch.zeros(self.env.num_envs, 1, dtype=torch.bool, device=self.device)
                        env_actions = self.alg.policy.act_inference(obs)
                    else:
                        # Default: student noisy sample
                        teacher_step_mask = torch.zeros(self.env.num_envs, 1, dtype=torch.bool, device=self.device)
                        env_actions = actions
                    env_action_delta = env_actions - teacher_actions
                    rollout_env_action_mse_sum += float(F.mse_loss(env_actions, teacher_actions).item())
                    rollout_env_action_mae_sum += float(env_action_delta.abs().mean().item())
                    rollout_teacher_env_frac_sum += float(teacher_step_mask.float().mean().item())
                    rollout_control_steps += 1
                    bridge_debug = self._collect_bridge_debug()
                    for name, value in bridge_debug.items():
                        scalar = self._bridge_debug_scalar(value)
                        if scalar is None:
                            continue
                        rollout_bridge_metric_sums[name] = rollout_bridge_metric_sums.get(name, 0.0) + scalar
                        rollout_bridge_metric_counts[name] = rollout_bridge_metric_counts.get(name, 0) + 1
                    if should_debug and bridge_debug:
                        bridge_info = ", ".join(f"{key}={value:.4e}" for key, value in sorted(bridge_debug.items()))
                        print(
                            f"[Distill][DEBUG][it={it}][rollout_step={rollout_step}] "
                            f"bridge: {bridge_info}"
                        )
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(env_actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    teacher_reward_mask = teacher_step_mask.view(-1).bool()
                    reward_values = rewards.view(-1)
                    if teacher_reward_mask.any():
                        rollout_teacher_reward_sum += float(reward_values[teacher_reward_mask].sum().item())
                        rollout_teacher_reward_count += int(teacher_reward_mask.sum().item())
                    student_reward_mask = ~teacher_reward_mask
                    if student_reward_mask.any():
                        rollout_student_reward_sum += float(reward_values[student_reward_mask].sum().item())
                        rollout_student_reward_count += int(student_reward_mask.sum().item())
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    if use_mixed_rollout:
                        done_env_ids = (dones.view(-1) > 0).nonzero(as_tuple=False).squeeze(-1)
                        self._sample_teacher_env_mask(teacher_mix, done_env_ids)
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # Update policy
            loss_dict = self.alg.update()
            if rollout_control_steps > 0:
                teacher_env_frac = rollout_teacher_env_frac_sum / rollout_control_steps
                loss_dict["teacher_mix_target"] = teacher_mix
                loss_dict["teacher_env_frac"] = teacher_env_frac
                loss_dict["student_env_frac"] = 1.0 - teacher_env_frac
                loss_dict["student_env_vs_teacher_mse"] = rollout_student_env_action_mse_sum / rollout_control_steps
                loss_dict["student_env_vs_teacher_mae"] = rollout_student_env_action_mae_sum / rollout_control_steps
                loss_dict["env_action_vs_teacher_mse"] = rollout_env_action_mse_sum / rollout_control_steps
                loss_dict["env_action_vs_teacher_mae"] = rollout_env_action_mae_sum / rollout_control_steps
            if rollout_teacher_reward_count > 0:
                loss_dict["teacher_step_reward"] = rollout_teacher_reward_sum / rollout_teacher_reward_count
            if rollout_student_reward_count > 0:
                loss_dict["student_step_reward"] = rollout_student_reward_sum / rollout_student_reward_count
            for name, total in rollout_bridge_metric_sums.items():
                count = rollout_bridge_metric_counts.get(name, 0)
                if count > 0:
                    loss_dict[name] = total / count

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """Construct the distillation algorithm."""
        # Initialize the policy
        student_teacher_class = _resolve_class(self.policy_cfg.pop("class_name"), NETWORKS, globals())
        student_teacher: StudentTeacher | StudentTeacherRecurrent | SonicDiffusionStudentTeacher | VisionStudentTeacher | VisionAblationStudentTeacher | PerceptiveMotionAdapterTracker | PerceptiveMotionTokenTracker = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = _resolve_class(self.alg_cfg.pop("class_name"), ALGORITHMS, globals())
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        alg.init_storage(
            "distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg
