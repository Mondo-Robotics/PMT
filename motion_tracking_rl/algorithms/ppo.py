# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from motion_tracking_rl.networks import ActorCritic, ActorCriticRecurrent
from motion_tracking_rl.networks.rnd import RandomNetworkDistillation
from motion_tracking_rl.registry import register_algorithm
from motion_tracking_rl.storage import RolloutStorage
from motion_tracking_rl.utils import string_to_callable


@register_algorithm("PPO", compat_name="ppo")
class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        backbone_lr_scale: float = 1.0,
        vision_adapter_lr_scale: float = 1.0,
        critic_lr_scale: float = 1.0,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # Optional supervised auxiliary losses
        vel_loss_coef: float = 0.0,
        vel_loss_type: str = "huber",
        vel_loss_delta: float = 1.0,
        anchor_est_loss_coef: float = 0.0,
        anchor_est_loss_type: str = "huber",
        anchor_est_loss_delta: float = 1.0,
        foot_traj_loss_coef: float = 0.0,
        foot_traj_loss_type: str = "huber",
        foot_traj_loss_delta: float = 1.0,
        foot_traj_target_obs_key: str | None = None,
        aux_loss_scale: float = 1.0,
        aux_loss_coef: dict[str, float] | None = None,
        sonic_loss_coef: float | None = None,
        use_mean_action_for_rollout: bool = False,
        mean_action_rollout_iters: int = 0,
        value_only_warmup_iters: int = 0,
        action_prior_loss_coef: float = 0.0,
        warmup_freeze_iters: int = 0,
        warmup_freeze_encoders: bool = True,
        warmup_freeze_control_decoder: bool = True,
        warmup_freeze_action_std: bool = True,
        warmup_reset_optimizer_state: bool = False,
        warmup_reset_rnd_optimizer_state: bool = False,
        warmup_unfreeze_lr_scale: float = 1.0,
        debug_numeric: bool = False,
        debug_print_freq: int = 1,
        debug_rollout_steps: int = 1,
        debug_update_batches: int = 2,
        debug_raise_on_nonfinite: bool = False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        if backbone_lr_scale <= 0.0:
            raise ValueError(f"backbone_lr_scale must be > 0.0, got {backbone_lr_scale}.")
        if vision_adapter_lr_scale <= 0.0:
            raise ValueError(f"vision_adapter_lr_scale must be > 0.0, got {vision_adapter_lr_scale}.")
        if critic_lr_scale <= 0.0:
            raise ValueError(f"critic_lr_scale must be > 0.0, got {critic_lr_scale}.")
        self.backbone_lr_scale = float(backbone_lr_scale)
        self.vision_adapter_lr_scale = float(vision_adapter_lr_scale)
        self.critic_lr_scale = float(critic_lr_scale)

        # Create optimizer
        optimizer_params = self.policy.parameters()
        if hasattr(self.policy, "build_optimizer_param_groups"):
            optimizer_params = self.policy.build_optimizer_param_groups(
                base_lr=learning_rate,
                backbone_lr_scale=self.backbone_lr_scale,
                vision_adapter_lr_scale=self.vision_adapter_lr_scale,
                critic_lr_scale=self.critic_lr_scale,
            )
        self.optimizer = optim.Adam(optimizer_params, lr=learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.base_learning_rate = float(learning_rate)
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.vel_loss_coef = vel_loss_coef
        self.vel_loss_type = vel_loss_type
        self.vel_loss_delta = vel_loss_delta
        self.anchor_est_loss_coef = float(anchor_est_loss_coef)
        self.anchor_est_loss_type = str(anchor_est_loss_type)
        self.anchor_est_loss_delta = float(anchor_est_loss_delta)
        self.foot_traj_loss_coef = float(foot_traj_loss_coef)
        self.foot_traj_loss_type = str(foot_traj_loss_type)
        self.foot_traj_loss_delta = float(foot_traj_loss_delta)
        self.foot_traj_target_obs_key = foot_traj_target_obs_key
        if self.vel_loss_type not in ("mse", "huber"):
            raise ValueError(f"Unknown vel_loss_type: {vel_loss_type}. Use 'huber' or 'mse'.")
        if self.anchor_est_loss_type not in ("mse", "huber"):
            raise ValueError(f"Unknown anchor_est_loss_type: {anchor_est_loss_type}. Use 'huber' or 'mse'.")
        if self.foot_traj_loss_type not in ("mse", "huber"):
            raise ValueError(f"Unknown foot_traj_loss_type: {foot_traj_loss_type}. Use 'huber' or 'mse'.")
        if sonic_loss_coef is not None:
            warnings.warn(
                "`sonic_loss_coef` is deprecated; use `aux_loss_scale` with `aux_loss_coef` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            aux_loss_scale = float(sonic_loss_coef)
        if aux_loss_scale < 0.0:
            raise ValueError(f"aux_loss_scale must be >= 0.0, got {aux_loss_scale}.")
        self.aux_loss_scale = float(aux_loss_scale)
        self.aux_loss_coef = {str(name): float(value) for name, value in (aux_loss_coef or {}).items()}
        self.sonic_loss_coef = self.aux_loss_scale
        if action_prior_loss_coef < 0.0:
            raise ValueError(f"action_prior_loss_coef must be >= 0.0, got {action_prior_loss_coef}.")
        if mean_action_rollout_iters < 0:
            raise ValueError(f"mean_action_rollout_iters must be >= 0, got {mean_action_rollout_iters}.")
        if value_only_warmup_iters < 0:
            raise ValueError(f"value_only_warmup_iters must be >= 0, got {value_only_warmup_iters}.")
        self.use_mean_action_for_rollout = bool(use_mean_action_for_rollout)
        self.mean_action_rollout_iters = int(mean_action_rollout_iters)
        self.value_only_warmup_iters = int(value_only_warmup_iters)
        self.action_prior_loss_coef = float(action_prior_loss_coef)

        if warmup_freeze_iters < 0:
            raise ValueError(f"warmup_freeze_iters must be >= 0, got {warmup_freeze_iters}.")
        self.warmup_freeze_iters = int(warmup_freeze_iters)
        self.warmup_freeze_encoders = bool(warmup_freeze_encoders)
        self.warmup_freeze_control_decoder = bool(warmup_freeze_control_decoder)
        self.warmup_freeze_action_std = bool(warmup_freeze_action_std)
        self.warmup_reset_optimizer_state = bool(warmup_reset_optimizer_state)
        self.warmup_reset_rnd_optimizer_state = bool(warmup_reset_rnd_optimizer_state)
        if warmup_unfreeze_lr_scale <= 0.0:
            raise ValueError(f"warmup_unfreeze_lr_scale must be > 0.0, got {warmup_unfreeze_lr_scale}.")
        self.warmup_unfreeze_lr_scale = float(warmup_unfreeze_lr_scale)

        if debug_print_freq <= 0:
            raise ValueError(f"debug_print_freq must be >= 1, got {debug_print_freq}.")
        if debug_rollout_steps < 0:
            raise ValueError(f"debug_rollout_steps must be >= 0, got {debug_rollout_steps}.")
        if debug_update_batches < 0:
            raise ValueError(f"debug_update_batches must be >= 0, got {debug_update_batches}.")
        self.debug_numeric = bool(debug_numeric)
        self.debug_print_freq = int(debug_print_freq)
        self.debug_rollout_steps = int(debug_rollout_steps)
        self.debug_update_batches = int(debug_update_batches)
        self.debug_raise_on_nonfinite = bool(debug_raise_on_nonfinite)
        self._debug_iter_enabled = False
        self._debug_rollout_step = 0
        self._debug_update_batch_idx = 0
        self._debug_nonfinite_reported = False

        self.current_learning_iteration = 0
        self._warmup_freeze_active = False
        self._warned_warmup_unsupported = False
        self.set_learning_iteration(0)

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        sampled_actions = self.policy.act(obs)
        use_mean_rollout = (
            self.use_mean_action_for_rollout
            or (
                self.mean_action_rollout_iters > 0
                and self.current_learning_iteration < self.mean_action_rollout_iters
            )
        )
        rollout_actions = self.policy.action_mean if use_mean_rollout else sampled_actions
        self.transition.actions = rollout_actions.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        should_log_rollout_debug = (
            self.debug_numeric
            and self._debug_iter_enabled
            and self._debug_rollout_step < self.debug_rollout_steps
        )
        action_is_finite = bool(torch.isfinite(self.transition.actions).all().item())
        value_is_finite = bool(torch.isfinite(self.transition.values).all().item())
        log_prob_is_finite = bool(torch.isfinite(self.transition.actions_log_prob).all().item())
        reward_is_finite = bool(torch.isfinite(self.transition.rewards).all().item())
        dones_is_finite = bool(torch.isfinite(dones).all().item())
        rollout_has_nonfinite = not (
            action_is_finite and value_is_finite and log_prob_is_finite and reward_is_finite and dones_is_finite
        )
        obs_nonfinite_keys: list[str] = []
        if should_log_rollout_debug or rollout_has_nonfinite:
            obs_nonfinite_keys = self._nonfinite_obs_keys(obs, max_keys=8)
            rollout_has_nonfinite = rollout_has_nonfinite or bool(obs_nonfinite_keys)

        should_log_nonfinite_rollout = rollout_has_nonfinite and (not self._debug_nonfinite_reported)
        if (should_log_rollout_debug or should_log_nonfinite_rollout) and self.gpu_global_rank == 0:
            done_frac = float(dones.float().mean().item())
            timeout_frac = float("nan")
            if "time_outs" in extras:
                timeout_frac = float(extras["time_outs"].float().mean().item())
            self._debug_print(
                f"[PPO][DEBUG][it={self.current_learning_iteration}][rollout_step={self._debug_rollout_step}] "
                f"done_frac={done_frac:.3e} timeout_frac={timeout_frac:.3e} obs_nonfinite_keys={obs_nonfinite_keys}"
            )
            self._debug_print("[PPO][DEBUG] " + self._tensor_debug_stats("action", self.transition.actions))
            self._debug_print("[PPO][DEBUG] " + self._tensor_debug_stats("action_mean", self.transition.action_mean))
            self._debug_print("[PPO][DEBUG] " + self._tensor_debug_stats("action_sigma", self.transition.action_sigma))
            self._debug_print("[PPO][DEBUG] " + self._tensor_debug_stats("value", self.transition.values))
            self._debug_print("[PPO][DEBUG] " + self._tensor_debug_stats("action_log_prob", self.transition.actions_log_prob))
            self._debug_print("[PPO][DEBUG] " + self._tensor_debug_stats("reward", self.transition.rewards))

        if rollout_has_nonfinite:
            self._debug_nonfinite_reported = True
            if self.debug_raise_on_nonfinite:
                raise FloatingPointError(
                    f"Non-finite rollout tensors detected at iteration {self.current_learning_iteration}, "
                    f"rollout step {self._debug_rollout_step}."
                )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
        self._debug_rollout_step += 1

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def _debug_print(self, message: str) -> None:
        if self.gpu_global_rank == 0:
            print(message)

    @staticmethod
    def _tensor_debug_stats(name: str, tensor: torch.Tensor) -> str:
        tensor_detached = tensor.detach()
        shape = tuple(tensor_detached.shape)
        numel = tensor_detached.numel()
        if numel == 0:
            return f"{name}: shape={shape} numel=0"

        finite_mask = torch.isfinite(tensor_detached)
        finite_count = int(finite_mask.sum().item())
        if finite_count == 0:
            return f"{name}: shape={shape} finite=0/{numel}"

        finite_vals = tensor_detached[finite_mask].float()
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
        mean_val = float(finite_vals.mean().item())
        absmax_val = float(finite_vals.abs().max().item())
        return (
            f"{name}: shape={shape} finite={finite_count}/{numel} "
            f"min={min_val:.3e} max={max_val:.3e} mean={mean_val:.3e} absmax={absmax_val:.3e}"
        )

    @staticmethod
    def _nonfinite_obs_keys(obs: TensorDict, max_keys: int = 12) -> list[str]:
        nonfinite_keys: list[str] = []
        for key in obs.keys():
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                continue
            if not torch.isfinite(value).all():
                nonfinite_keys.append(str(key))
                if len(nonfinite_keys) >= max_keys:
                    break
        return nonfinite_keys

    def _resolve_obs_group_keys(self, group_name: str) -> list[str]:
        obs_groups = getattr(self.policy, "obs_groups", None)
        if isinstance(obs_groups, dict) and group_name in obs_groups:
            return list(obs_groups[group_name])
        return []

    def _compute_velocity_loss(
        self, aux_outputs: dict[str, torch.Tensor], obs_batch: TensorDict, original_batch_size: int
    ) -> torch.Tensor | None:
        v_hat = aux_outputs.get("v_hat", None)
        if v_hat is None:
            return None

        v_hat = v_hat[:original_batch_size]
        v_gt = None
        vel_keys = self._resolve_obs_group_keys("vel_gt")
        if vel_keys and all(k in obs_batch.keys() for k in vel_keys):
            v_gt = torch.cat([obs_batch[k] for k in vel_keys], dim=-1)
        elif "vel_gt" in obs_batch.keys():
            v_gt = obs_batch["vel_gt"]

        if v_gt is None:
            raise KeyError(
                "Velocity GT observations not found. Expected obs_groups['vel_gt'] to map to an "
                "observation group present in env observations (e.g. 'vel_gt_xyz')."
            )

        v_gt = v_gt[:original_batch_size]
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

    def _compute_anchor_est_loss(
        self, aux_outputs: dict[str, torch.Tensor], obs_batch: TensorDict, original_batch_size: int
    ) -> torch.Tensor | None:
        anchor_hat = aux_outputs.get("anchor_hat", None)
        if anchor_hat is None:
            return None

        anchor_hat = anchor_hat[:original_batch_size]
        anchor_gt = None
        anchor_keys = self._resolve_obs_group_keys("anchor_gt")
        if anchor_keys and all(k in obs_batch.keys() for k in anchor_keys):
            anchor_gt = torch.cat([obs_batch[k] for k in anchor_keys], dim=-1)
        elif "anchor_gt" in obs_batch.keys():
            anchor_gt = obs_batch["anchor_gt"]

        if anchor_gt is None:
            raise KeyError(
                "Anchor GT observations not found. Expected obs_groups['anchor_gt'] to map to an "
                "observation group present in env observations (e.g. 'anchor_gt')."
            )

        anchor_gt = anchor_gt[:original_batch_size]
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
        self, aux_outputs: dict[str, torch.Tensor], obs_batch: TensorDict, original_batch_size: int
    ) -> torch.Tensor | None:
        foot_traj_pred = aux_outputs.get("foot_traj", None)
        if foot_traj_pred is None:
            return None

        target_key = self.foot_traj_target_obs_key
        if target_key is None:
            target_key = getattr(self.policy, "foot_traj_target_obs_key", None)

        foot_traj_target = None
        if target_key is not None and target_key in obs_batch.keys():
            foot_traj_target = obs_batch[target_key]
        else:
            foot_keys = self._resolve_obs_group_keys("foot_traj_target")
            if foot_keys and all(k in obs_batch.keys() for k in foot_keys):
                foot_traj_target = torch.cat([obs_batch[k] for k in foot_keys], dim=-1)
            elif "foot_traj_target" in obs_batch.keys():
                foot_traj_target = obs_batch["foot_traj_target"]

        if foot_traj_target is None:
            raise KeyError(
                "Foot trajectory target observations not found. Expected `foot_traj_target_obs_key` "
                "or obs_groups['foot_traj_target'] to point to an observation group present in the batch."
            )

        foot_traj_pred = foot_traj_pred[:original_batch_size]
        foot_traj_target = foot_traj_target[:original_batch_size]
        if foot_traj_pred.shape != foot_traj_target.shape:
            raise ValueError(
                f"Foot trajectory shapes mismatch: pred {tuple(foot_traj_pred.shape)} vs "
                f"target {tuple(foot_traj_target.shape)}."
            )
        if self.foot_traj_loss_type == "huber":
            return F.huber_loss(foot_traj_pred, foot_traj_target, delta=self.foot_traj_loss_delta)
        return F.mse_loss(foot_traj_pred, foot_traj_target)

    def _compute_surrogate_loss(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        value_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the policy objective.

        Subclasses can override this hook while reusing PPO's rollout, auxiliary losses,
        distributed synchronization, logging, and optimizer handling.
        """
        del returns_batch, value_batch
        log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        ratio = torch.exp(log_ratio)
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        stats = {
            "ratio_mean": ratio.detach().mean(),
            "ratio_std": ratio.detach().float().std(unbiased=False),
            "ratio_max_abs": ratio.detach().abs().max(),
            "approx_kl": ((ratio.detach() - 1.0) - log_ratio.detach()).mean(),
            "clip_fraction": (torch.abs(ratio.detach() - 1.0) > self.clip_param).float().mean(),
        }
        return surrogate_loss, ratio, stats

    def _compute_policy_aux_loss(self, obs_batch: TensorDict) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute policy-provided auxiliary losses with per-term coefficients."""
        sonic_loss = torch.tensor(0.0, device=self.device)
        aux_stats: dict[str, float] = {}
        if self.aux_loss_scale <= 0.0:
            return sonic_loss, aux_stats

        aux_getter = getattr(self.policy, "compute_sonic_aux_losses", None)
        if not callable(aux_getter):
            aux_getter = getattr(self.policy, "get_sonic_aux_losses", None)
        if not callable(aux_getter):
            return sonic_loss, aux_stats

        aux_result = aux_getter(obs_batch)
        if aux_result is None:
            return sonic_loss, aux_stats
        aux_losses = aux_result.get("aux_losses", {})
        if not aux_losses:
            return sonic_loss, aux_stats

        aux_coefs = dict(aux_result.get("aux_loss_coef", {}) or {})
        aux_coefs.update(self.aux_loss_coef)
        weighted_aux_loss = torch.tensor(0.0, device=self.device)
        for name, aux_value in aux_losses.items():
            if torch.is_tensor(aux_value):
                aux_tensor = aux_value.to(self.device).mean()
            else:
                aux_tensor = torch.as_tensor(aux_value, device=self.device, dtype=torch.float32).mean()
            # Match official gear_sonic ppo_trainer_aux_loss: a loss term with NO
            # configured coefficient defaults to 0.0 (disabled), not 1.0. This avoids
            # silently giving an unconfigured aux term full weight.
            coef = float(aux_coefs.get(name, 0.0))
            weighted_aux_loss = weighted_aux_loss + coef * aux_tensor

            detached = aux_tensor.detach().float()
            aux_stats[f"sonic/{name}"] = float(detached.item())
            aux_stats[f"sonic_weighted/{name}"] = float((detached * coef).item())

        sonic_loss = self.aux_loss_scale * weighted_aux_loss
        aux_stats["sonic/aux_loss_scale"] = self.aux_loss_scale
        return sonic_loss, aux_stats

    def _debug_dump_std_optimizer_state(self) -> None:
        if not hasattr(self.policy, "std"):
            return
        std_param = getattr(self.policy, "std", None)
        if std_param is None or not isinstance(std_param, torch.Tensor):
            return
        self._debug_print(
            f"[PPO][DEBUG][it={self.current_learning_iteration}] "
            + self._tensor_debug_stats("policy.std_param", std_param)
        )
        if not isinstance(std_param, nn.Parameter):
            return
        state = self.optimizer.state.get(std_param, None)
        if state is None:
            self._debug_print(f"[PPO][DEBUG][it={self.current_learning_iteration}] optimizer state for std is empty.")
            return
        for state_name in ("exp_avg", "exp_avg_sq"):
            state_tensor = state.get(state_name, None)
            if state_tensor is not None:
                self._debug_print(
                    f"[PPO][DEBUG][it={self.current_learning_iteration}] "
                    + self._tensor_debug_stats(f"optimizer.std.{state_name}", state_tensor)
                )

    def _set_optimizer_lr(self, learning_rate: float) -> None:
        self.learning_rate = float(learning_rate)
        for param_group in self.optimizer.param_groups:
            lr_scale = float(param_group.get("lr_scale", 1.0))
            param_group["lr"] = self.learning_rate * lr_scale

    def get_optimizer_lrs(self) -> dict[str, float]:
        optimizer_lrs: dict[str, float] = {}
        for index, param_group in enumerate(self.optimizer.param_groups):
            group_name = str(param_group.get("name", f"group_{index}"))
            optimizer_lrs[group_name] = float(param_group["lr"])
        return optimizer_lrs

    def _reset_optimizer_state(self) -> None:
        self.optimizer.state.clear()
        if self.rnd_optimizer is not None and self.warmup_reset_rnd_optimizer_state:
            self.rnd_optimizer.state.clear()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_sonic_loss = 0
        mean_vel_loss = 0
        mean_anchor_est_loss = 0
        mean_foot_traj_loss = 0
        mean_action_prior_loss = 0
        mean_surrogate_stats: dict[str, float] = {}
        mean_policy_stats: dict[str, float] = {}
        mean_sonic_aux_stats: dict[str, float] = {}
        debug_nonfinite_batches = 0
        self._debug_update_batch_idx = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            if hasattr(self.policy, "get_ppo_log_stats"):
                for stat_name, stat_value in self.policy.get_ppo_log_stats().items():
                    mean_policy_stats[stat_name] = mean_policy_stats.get(stat_name, 0.0) + float(stat_value)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive" and not self._warmup_freeze_active:
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-7, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    self._set_optimizer_lr(self.learning_rate)

            # Surrogate loss
            surrogate_loss, ratio, surrogate_stats = self._compute_surrogate_loss(
                actions_log_prob_batch=actions_log_prob_batch,
                old_actions_log_prob_batch=old_actions_log_prob_batch,
                advantages_batch=advantages_batch,
                returns_batch=returns_batch,
                value_batch=value_batch,
            )

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            value_only_warmup_active = self.current_learning_iteration < self.value_only_warmup_iters
            if value_only_warmup_active:
                loss = self.value_loss_coef * value_loss
            else:
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            sonic_loss, sonic_aux_stats = self._compute_policy_aux_loss(obs_batch)
            loss += sonic_loss

            # Velocity estimator loss (optional; policy-specific)
            aux = None
            vel_loss = None
            anchor_est_loss = None
            foot_traj_loss = None
            action_prior_loss = None
            if (
                self.vel_loss_coef > 0.0
                or self.anchor_est_loss_coef > 0.0
                or self.foot_traj_loss_coef > 0.0
            ) and hasattr(self.policy, "get_last_aux_outputs"):
                aux = self.policy.get_last_aux_outputs(clear=True)
                if self.vel_loss_coef > 0.0:
                    vel_loss = self._compute_velocity_loss(aux, obs_batch, original_batch_size)
                    if vel_loss is not None:
                        loss += self.vel_loss_coef * vel_loss
                if self.anchor_est_loss_coef > 0.0:
                    anchor_est_loss = self._compute_anchor_est_loss(aux, obs_batch, original_batch_size)
                    if anchor_est_loss is not None:
                        loss += self.anchor_est_loss_coef * anchor_est_loss
                if self.foot_traj_loss_coef > 0.0:
                    foot_traj_loss = self._compute_foot_traj_loss(aux, obs_batch, original_batch_size)
                    if foot_traj_loss is not None:
                        loss += self.foot_traj_loss_coef * foot_traj_loss

            if self.action_prior_loss_coef > 0.0 and not value_only_warmup_active:
                with torch.no_grad():
                    reference_actions = self.policy.reference_act_inference(obs_batch[:original_batch_size])
                action_prior_loss = F.mse_loss(mu_batch, reference_actions)
                loss += self.action_prior_loss_coef * action_prior_loss
            
            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            nonfinite_tensors: list[tuple[str, torch.Tensor]] = []
            for name, tensor in (
                ("actions_log_prob", actions_log_prob_batch),
                ("value_batch", value_batch),
                ("mu_batch", mu_batch),
                ("sigma_batch", sigma_batch),
                ("entropy_batch", entropy_batch),
                ("ratio", ratio),
                ("surrogate_loss", surrogate_loss),
                ("value_loss", value_loss),
                ("total_loss", loss),
            ):
                if not torch.isfinite(tensor).all():
                    nonfinite_tensors.append((name, tensor))
            if isinstance(sonic_loss, torch.Tensor) and not torch.isfinite(sonic_loss).all():
                nonfinite_tensors.append(("sonic_loss", sonic_loss))
            for stat_name, stat_tensor in surrogate_stats.items():
                if isinstance(stat_tensor, torch.Tensor) and not torch.isfinite(stat_tensor).all():
                    nonfinite_tensors.append((stat_name, stat_tensor))
            if vel_loss is not None and not torch.isfinite(vel_loss).all():
                nonfinite_tensors.append(("vel_loss", vel_loss))
            if anchor_est_loss is not None and not torch.isfinite(anchor_est_loss).all():
                nonfinite_tensors.append(("anchor_est_loss", anchor_est_loss))
            if foot_traj_loss is not None and not torch.isfinite(foot_traj_loss).all():
                nonfinite_tensors.append(("foot_traj_loss", foot_traj_loss))
            if action_prior_loss is not None and not torch.isfinite(action_prior_loss).all():
                nonfinite_tensors.append(("action_prior_loss", action_prior_loss))
            if self.rnd and not torch.isfinite(rnd_loss).all():
                nonfinite_tensors.append(("rnd_loss", rnd_loss))
            if self.symmetry and not torch.isfinite(symmetry_loss).all():
                nonfinite_tensors.append(("symmetry_loss", symmetry_loss))

            should_log_batch_debug = (
                self.debug_numeric
                and self._debug_iter_enabled
                and self._debug_update_batch_idx < self.debug_update_batches
            )
            if should_log_batch_debug and self.gpu_global_rank == 0:
                ratio_mean = float(ratio.mean().item())
                ratio_max_abs = float(ratio.abs().max().item())
                entropy_mean = float(entropy_batch.mean().item())
                sonic_value = float(sonic_loss.item()) if isinstance(sonic_loss, torch.Tensor) else 0.0
                self._debug_print(
                    f"[PPO][DEBUG][it={self.current_learning_iteration}][mb={self._debug_update_batch_idx}] "
                    f"loss={float(loss.item()):.3e} surr={float(surrogate_loss.item()):.3e} "
                    f"vf={float(value_loss.item()):.3e} ent={entropy_mean:.3e} "
                    f"sonic={sonic_value:.3e} ratio_mean={ratio_mean:.3e} ratio_absmax={ratio_max_abs:.3e} "
                    f"lr={self.learning_rate:.3e}"
                )
                self._debug_print(
                    "[PPO][DEBUG] "
                    + self._tensor_debug_stats("advantages", advantages_batch)
                    + " | "
                    + self._tensor_debug_stats("returns", returns_batch)
                )
                self._debug_print(
                    "[PPO][DEBUG] "
                    + self._tensor_debug_stats("mu_batch", mu_batch)
                    + " | "
                    + self._tensor_debug_stats("sigma_batch", sigma_batch)
                )
                self._debug_dump_std_optimizer_state()

            if nonfinite_tensors:
                debug_nonfinite_batches += 1
                if not self._debug_nonfinite_reported and self.gpu_global_rank == 0:
                    names = [name for name, _ in nonfinite_tensors]
                    self._debug_print(
                        f"[PPO][ERROR][it={self.current_learning_iteration}][mb={self._debug_update_batch_idx}] "
                        f"non-finite tensors detected: {names}"
                    )
                    for name, tensor in nonfinite_tensors:
                        self._debug_print("[PPO][ERROR] " + self._tensor_debug_stats(name, tensor))
                    obs_nonfinite_keys = self._nonfinite_obs_keys(obs_batch)
                    if obs_nonfinite_keys:
                        self._debug_print(
                            f"[PPO][ERROR][it={self.current_learning_iteration}] "
                            f"obs batch has non-finite keys: {obs_nonfinite_keys}"
                        )
                    self._debug_dump_std_optimizer_state()
                self._debug_nonfinite_reported = True
                if self.debug_raise_on_nonfinite:
                    raise FloatingPointError(
                        f"Non-finite tensors detected in PPO update iteration {self.current_learning_iteration}."
                    )

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if should_log_batch_debug and self.gpu_global_rank == 0:
                self._debug_print(
                    f"[PPO][DEBUG][it={self.current_learning_iteration}][mb={self._debug_update_batch_idx}] "
                    f"policy_grad_norm={float(grad_norm):.3e} clip={self.max_grad_norm:.3e}"
                )
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if isinstance(sonic_loss, torch.Tensor):
                mean_sonic_loss += sonic_loss.item()
            for stat_name, stat_value in sonic_aux_stats.items():
                mean_sonic_aux_stats[stat_name] = mean_sonic_aux_stats.get(stat_name, 0.0) + stat_value
            if vel_loss is not None:
                mean_vel_loss += float(vel_loss.item())
            if anchor_est_loss is not None:
                mean_anchor_est_loss += float(anchor_est_loss.item())
            if foot_traj_loss is not None:
                mean_foot_traj_loss += float(foot_traj_loss.item())
            if action_prior_loss is not None:
                mean_action_prior_loss += float(action_prior_loss.item())
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            for stat_name, stat_tensor in surrogate_stats.items():
                mean_surrogate_stats[stat_name] = mean_surrogate_stats.get(stat_name, 0.0) + float(
                    stat_tensor.detach().float().mean().item()
                )
            self._debug_update_batch_idx += 1

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        denom = num_updates
        mean_value_loss /= denom
        mean_surrogate_loss /= denom
        mean_entropy /= denom
        mean_sonic_loss /= denom
        mean_vel_loss /= denom
        mean_anchor_est_loss /= denom
        mean_foot_traj_loss /= denom
        mean_action_prior_loss /= denom
        if mean_rnd_loss is not None:
            mean_rnd_loss /= denom
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= denom
        for stat_name in list(mean_surrogate_stats.keys()):
            mean_surrogate_stats[stat_name] /= denom
        for stat_name in list(mean_policy_stats.keys()):
            mean_policy_stats[stat_name] /= denom
        for stat_name in list(mean_sonic_aux_stats.keys()):
            mean_sonic_aux_stats[stat_name] /= denom

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "sonic": mean_sonic_loss,
        }
        loss_dict.update(mean_surrogate_stats)
        loss_dict.update(mean_policy_stats)
        loss_dict.update(mean_sonic_aux_stats)
        if self.vel_loss_coef > 0.0:
            loss_dict["vel_estimator"] = mean_vel_loss
        if self.anchor_est_loss_coef > 0.0:
            loss_dict["anchor_estimator"] = mean_anchor_est_loss
        if self.foot_traj_loss_coef > 0.0:
            loss_dict["foot_traj"] = mean_foot_traj_loss
        if self.action_prior_loss_coef > 0.0:
            loss_dict["action_prior"] = mean_action_prior_loss
        if self.value_only_warmup_iters > 0:
            loss_dict["value_only_warmup_active"] = float(
                self.current_learning_iteration < self.value_only_warmup_iters
            )
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.debug_numeric or debug_nonfinite_batches > 0:
            loss_dict["debug_nonfinite_batches"] = float(debug_nonfinite_batches)

        return loss_dict

    def set_learning_iteration(self, iteration: int) -> None:
        """Set learning iteration and apply warmup freeze transitions when needed."""
        self.current_learning_iteration = int(iteration)
        self._debug_rollout_step = 0
        self._debug_update_batch_idx = 0
        self._debug_nonfinite_reported = False
        self._debug_iter_enabled = self.debug_numeric and (self.current_learning_iteration % self.debug_print_freq == 0)
        if self._debug_iter_enabled and self.gpu_global_rank == 0:
            self._debug_print(f"[PPO][DEBUG] Numeric debug enabled for iteration {self.current_learning_iteration}.")

        should_freeze = self.warmup_freeze_iters > 0 and self.current_learning_iteration < self.warmup_freeze_iters

        if should_freeze == self._warmup_freeze_active:
            return

        was_frozen = self._warmup_freeze_active
        self._warmup_freeze_active = should_freeze
        transition_to_unfreeze = was_frozen and not should_freeze

        target_lr = self.base_learning_rate
        if transition_to_unfreeze:
            target_lr = self.base_learning_rate * self.warmup_unfreeze_lr_scale
        self._set_optimizer_lr(target_lr)

        if transition_to_unfreeze and self.warmup_reset_optimizer_state:
            self._reset_optimizer_state()
            if self.gpu_global_rank == 0:
                print(
                    "[PPO] Warmup transition reset optimizer state "
                    f"(reset_rnd={self.warmup_reset_rnd_optimizer_state})."
                )
        if not hasattr(self.policy, "set_warmup_freeze"):
            if should_freeze and not self._warned_warmup_unsupported:
                warnings.warn(
                    "warmup_freeze_* was configured, but the current policy does not implement "
                    "`set_warmup_freeze`; warmup freeze is ignored.",
                    stacklevel=2,
                )
                self._warned_warmup_unsupported = True
            return

        state = self.policy.set_warmup_freeze(
            freeze_encoders=should_freeze and self.warmup_freeze_encoders,
            freeze_control_decoder=should_freeze and self.warmup_freeze_control_decoder,
            freeze_action_std=should_freeze and self.warmup_freeze_action_std,
        )
        if self.gpu_global_rank == 0:
            phase = "warmup-frozen" if should_freeze else "warmup-unfrozen"
            print(
                f"[PPO] Iteration {self.current_learning_iteration}: {phase} "
                f"(target iters: {self.warmup_freeze_iters}, lr_reset_to={self.learning_rate:.3e}, state: {state})"
            )

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
