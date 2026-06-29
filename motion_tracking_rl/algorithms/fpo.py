# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flow Policy Optimization (FPO) algorithm for diffusion policies."""

from __future__ import annotations

import warnings
from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from motion_tracking_rl.algorithms.ppo import PPO


def _ste_clamp_max(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Forward clamps x to max_val, backward passes identity gradient."""
    clamped = torch.clamp(x, max=max_val)
    return x + (clamped - x).detach()


def _resolve_cfm_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lookup = {
            "float": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if value not in lookup:
            raise ValueError(f"Unsupported cfm_storage_dtype: {value}")
        return lookup[value]
    raise ValueError(f"Unsupported cfm_storage_dtype type: {type(value)}")


def _resolve_cfm_device(value: str | torch.device | None) -> torch.device | None:
    if value is None:
        return None
    if isinstance(value, torch.device):
        return value
    if isinstance(value, str):
        return torch.device(value)
    raise ValueError(f"Unsupported cfm_storage_device type: {type(value)}")


class FPO(PPO):
    """Internal FPO-style base used by FPOPlus.

    Plain FPO is intentionally not registered as a usable algorithm.  This
    class keeps the rollout and storage helpers that FPOPlus builds on.
    """

    def __init__(
        self,
        policy,
        num_fpo_samples: int = 16,
        positive_advantage: bool = False,
        cfm_storage_dtype: str | torch.dtype | None = None,
        cfm_storage_device: str | torch.device | None = None,
        bound_coef: float = 0.0,
        cfm_diff_clamp_max: float = 10.0,
        storage_action_noise_std: float = 0.0,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        if rnd_cfg is not None:
            raise NotImplementedError("RND is not supported in FPO++.")
        if symmetry_cfg is not None:
            raise NotImplementedError("Symmetry is not supported in FPO++.")

        super().__init__(policy, rnd_cfg=None, symmetry_cfg=None, **kwargs)

        if getattr(policy, "is_recurrent", False):
            raise NotImplementedError("Recurrent policies are not supported in FPO++.")

        self.num_fpo_samples = num_fpo_samples
        self.positive_advantage = positive_advantage
        self.bound_coef = bound_coef
        self.cfm_diff_clamp_max = cfm_diff_clamp_max
        self.cfm_storage_dtype = _resolve_cfm_dtype(cfm_storage_dtype)
        self.cfm_storage_device = _resolve_cfm_device(cfm_storage_device)
        self.storage_action_noise_std = float(storage_action_noise_std)

        if self.num_fpo_samples <= 0:
            raise ValueError("num_fpo_samples must be positive.")

        if self.cfm_storage_device is not None and self.cfm_storage_device.type == "cpu":
            warnings.warn(
                "CFM storage on CPU is enabled. Expect slower updates due to host-to-device transfers.",
                stacklevel=2,
            )

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        storage = cast(Any, self.storage)

        action_dim = int(actions_shape[0])
        device = self.cfm_storage_device if self.cfm_storage_device is not None else self.device
        dtype = self.cfm_storage_dtype if self.cfm_storage_dtype is not None else torch.float

        storage.cfm_device = device
        storage.cfm_dtype = dtype
        storage.cfm_initial_loss = torch.zeros(
            num_transitions_per_env, num_envs, self.num_fpo_samples, device=device, dtype=dtype
        )
        storage.cfm_loss_eps = torch.zeros(
            num_transitions_per_env, num_envs, self.num_fpo_samples, action_dim, device=device, dtype=dtype
        )
        storage.cfm_loss_t = torch.zeros(
            num_transitions_per_env, num_envs, self.num_fpo_samples, 1, device=device, dtype=dtype
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        policy = cast(Any, self.policy)
        actions, cfm_info = policy.act_with_cfm_info(obs, self.num_fpo_samples)

        if self.storage_action_noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * self.storage_action_noise_std
            cfm_info = dict(cfm_info)
            with torch.no_grad():
                actor_obs = self._get_actor_obs(obs)
                normalizer = getattr(policy, "actor_obs_normalizer", None)
                if normalizer is not None:
                    actor_obs = normalizer(actor_obs)
                eps_sample = cfm_info["loss_eps"]
                t_sample = cfm_info["loss_t"]
                batch_size, num_samples, action_dim = eps_sample.shape
                obs_tile = (
                    actor_obs.unsqueeze(1)
                    .expand(batch_size, num_samples, -1)
                    .reshape(batch_size * num_samples, -1)
                )
                actions_tile = (
                    actions.unsqueeze(1)
                    .expand(batch_size, num_samples, -1)
                    .reshape(batch_size * num_samples, -1)
                )
                initial_cfm_loss = policy.compute_cfm_loss(
                    obs_tile,
                    actions_tile,
                    eps_sample.reshape(batch_size * num_samples, action_dim),
                    t_sample.reshape(batch_size * num_samples, 1),
                ).view(batch_size, num_samples)
            cfm_info["initial_cfm_loss"] = initial_cfm_loss.detach()

        self.transition.actions = actions.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = torch.zeros(actions.shape[0], 1, device=self.device).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.cfm_info = cfm_info

        return cast(torch.Tensor, self.transition.actions)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.transition.cfm_info is not None:
            storage = cast(Any, self.storage)
            step = storage.step
            cfm_device = storage.cfm_initial_loss.device
            cfm_dtype = storage.cfm_initial_loss.dtype
            storage.cfm_initial_loss[step].copy_(
                self.transition.cfm_info["initial_cfm_loss"].to(device=cfm_device, dtype=cfm_dtype)
            )
            storage.cfm_loss_eps[step].copy_(
                self.transition.cfm_info["loss_eps"].to(device=cfm_device, dtype=cfm_dtype)
            )
            storage.cfm_loss_t[step].copy_(
                self.transition.cfm_info["loss_t"].to(device=cfm_device, dtype=cfm_dtype)
            )
        super().process_env_step(obs, rewards, dones, extras)

    def _get_actor_obs(self, obs_batch: TensorDict | dict | torch.Tensor) -> torch.Tensor:
        if hasattr(self.policy, "get_actor_obs"):
            return cast(Any, self.policy).get_actor_obs(obs_batch)
        if isinstance(obs_batch, (dict, TensorDict)):
            return obs_batch["policy"]
        return obs_batch

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cfm_ratio = 0.0
        mean_old_cfm_loss = 0.0
        mean_current_cfm_loss = 0.0
        mean_clip_fraction = 0.0
        mean_ratio_std = 0.0
        mean_logratio = 0.0
        mean_approx_kl = 0.0
        mean_adv_std = 0.0
        mean_cfm_diff_absmax = 0.0
        debug_nonfinite_batches = 0
        old_cfm_loss_cache: list[torch.Tensor] = []
        batch_index = 0

        if self.policy.is_recurrent:
            raise NotImplementedError("Recurrent policies are not supported in FPO++.")

        storage = cast(Any, self.storage)
        policy = cast(Any, self.policy)
        generator = storage.fpo_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            _old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            _hidden_states_batch,
            _masks_batch,
            cfm_info_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            elif self.positive_advantage:
                advantages_batch = torch.nn.functional.softplus(advantages_batch)

            if cfm_info_batch is None:
                raise RuntimeError("CFM info is missing. Ensure FPO++ storage is initialized.")

            _rollout_cfm_loss, eps_sample, t_sample = cfm_info_batch
            target_dtype = actions_batch.dtype
            eps_sample = eps_sample.to(self.device, dtype=target_dtype, non_blocking=True)
            t_sample = t_sample.to(self.device, dtype=target_dtype, non_blocking=True)

            actor_obs = self._get_actor_obs(obs_batch)
            if hasattr(self.policy, "actor_obs_normalizer"):
                actor_obs = self.policy.actor_obs_normalizer(actor_obs)

            B, N, D = eps_sample.shape
            flat_obs = actor_obs.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            flat_acts = actions_batch.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            flat_eps = eps_sample.reshape(B * N, D)
            flat_t = t_sample.reshape(B * N, 1)

            epoch_idx = batch_index // self.num_mini_batches
            mini_idx = batch_index % self.num_mini_batches

            if epoch_idx == 0:
                with torch.no_grad():
                    old_cfm_loss = policy.compute_cfm_loss(flat_obs, flat_acts, flat_eps, flat_t)
                old_cfm_loss = old_cfm_loss.view(B, N).detach()
                old_cfm_loss_cache.append(old_cfm_loss)
            else:
                old_cfm_loss = old_cfm_loss_cache[mini_idx]

            current_cfm_loss = policy.compute_cfm_loss(flat_obs, flat_acts, flat_eps, flat_t)
            current_cfm_loss = current_cfm_loss.view(B, N)

            # CFM loss returns negative log-prob; use per-sample log-ratio as in PPO.
            # Keep [B, N] shape so each of the N samples contributes its own gradient.
            cfm_difference = old_cfm_loss - current_cfm_loss
            # One-sided STE clamp on the upper bound only, matching the reference FPO
            # implementation: caps ratio blow-up when current_cfm_loss collapses, while
            # leaving gradients intact through the straight-through estimator.
            logratio = _ste_clamp_max(cfm_difference, max_val=self.cfm_diff_clamp_max)
            ratio = torch.exp(logratio)

            value_batch = self.policy.evaluate(obs_batch)

            # Broadcast advantages across the N samples for per-sample surrogate.
            adv = advantages_batch.view(-1, 1).expand(-1, ratio.shape[1])
            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss
            if self.bound_coef > 0 and hasattr(self.policy, "mean_bound_loss") and self.policy.mean_bound_loss is not None:
                loss = loss + self.bound_coef * cast(torch.Tensor, self.policy.mean_bound_loss)

            has_nonfinite = not (
                torch.isfinite(old_cfm_loss).all()
                and torch.isfinite(current_cfm_loss).all()
                and torch.isfinite(cfm_difference).all()
                and torch.isfinite(logratio).all()
                and torch.isfinite(ratio).all()
                and torch.isfinite(value_batch).all()
                and torch.isfinite(surrogate_loss).all()
                and torch.isfinite(value_loss).all()
                and torch.isfinite(loss).all()
            )
            if has_nonfinite:
                debug_nonfinite_batches += 1
                if not self._debug_nonfinite_reported and self.gpu_global_rank == 0:
                    self._debug_print(
                        f"[FPO][ERROR][it={self.current_learning_iteration}][mb={batch_index}] Non-finite tensor detected."
                    )
                    self._debug_print("[FPO][ERROR] " + self._tensor_debug_stats("old_cfm_loss", old_cfm_loss))
                    self._debug_print("[FPO][ERROR] " + self._tensor_debug_stats("current_cfm_loss", current_cfm_loss))
                    self._debug_print("[FPO][ERROR] " + self._tensor_debug_stats("cfm_difference", cfm_difference))
                    self._debug_print("[FPO][ERROR] " + self._tensor_debug_stats("logratio", logratio))
                    self._debug_print("[FPO][ERROR] " + self._tensor_debug_stats("ratio", ratio))
                    self._debug_print("[FPO][ERROR] " + self._tensor_debug_stats("advantages", advantages_batch))
                self._debug_nonfinite_reported = True
                if self.debug_raise_on_nonfinite:
                    raise RuntimeError(
                        "Non-finite tensors detected in FPO update. "
                        f"iteration={self.current_learning_iteration}, mini_batch={batch_index}"
                    )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                ratio_std = ratio.std(unbiased=False).item()
                adv_std = advantages_batch.std(unbiased=False).item()
                logratio_mean = logratio.mean().item()
                approx_kl = ((ratio - 1.0) - logratio).mean().item()
                old_cfm_mean = old_cfm_loss.mean().item()
                current_cfm_mean = current_cfm_loss.mean().item()
                cfm_diff_absmax = cfm_difference.abs().max().item()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cfm_ratio += ratio.mean().item()
            mean_old_cfm_loss += old_cfm_mean
            mean_current_cfm_loss += current_cfm_mean
            mean_clip_fraction += clip_frac
            mean_ratio_std += ratio_std
            mean_logratio += logratio_mean
            mean_approx_kl += approx_kl
            mean_adv_std += adv_std
            mean_cfm_diff_absmax += cfm_diff_absmax
            batch_index += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_cfm_ratio /= num_updates
        mean_old_cfm_loss /= num_updates
        mean_current_cfm_loss /= num_updates
        mean_clip_fraction /= num_updates
        mean_ratio_std /= num_updates
        mean_logratio /= num_updates
        mean_approx_kl /= num_updates
        mean_adv_std /= num_updates
        mean_cfm_diff_absmax /= num_updates

        # Clear rollout storage for the next iteration.
        storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "cfm_ratio": mean_cfm_ratio,
            "old_cfm_loss": mean_old_cfm_loss,
            "current_cfm_loss": mean_current_cfm_loss,
            "clip_fraction": mean_clip_fraction,
            "ratio_std": mean_ratio_std,
            "logratio": mean_logratio,
            "approx_kl": mean_approx_kl,
            "adv_std": mean_adv_std,
            "cfm_diff_absmax": mean_cfm_diff_absmax,
        }
        if self.debug_numeric or debug_nonfinite_batches > 0:
            loss_dict["debug_nonfinite_batches"] = float(debug_nonfinite_batches)
        return loss_dict
