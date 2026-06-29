# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial Distillation for Discrimination (ADD) on top of PPO."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from motion_tracking_rl.algorithms.ppo import PPO
from motion_tracking_rl.networks import ActorCritic, ActorCriticRecurrent, DiffNormalizer
from motion_tracking_rl.networks import MLP
from motion_tracking_rl.registry import register_algorithm
from motion_tracking_rl.storage import RolloutStorage


class ReplayBuffer:
    """Simple ring buffer for discriminator negative samples."""

    def __init__(self, capacity: int, obs_dim: int, device: str | torch.device, dtype: torch.dtype = torch.float32):
        if capacity <= 0:
            raise ValueError(f"Replay buffer capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.buffer = torch.zeros(self.capacity, self.obs_dim, device=self.device, dtype=dtype)
        self.size = 0
        self.head = 0

    def is_full(self) -> bool:
        return self.size >= self.capacity

    def push(self, data: torch.Tensor) -> None:
        if data.ndim != 2 or data.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected [N, {self.obs_dim}] data, got {tuple(data.shape)}.")
        if data.numel() == 0:
            return

        data = data.to(self.buffer.dtype)
        n = data.shape[0]
        if n >= self.capacity:
            self.buffer.copy_(data[-self.capacity :])
            self.size = self.capacity
            self.head = 0
            return

        end = self.head + n
        if end <= self.capacity:
            self.buffer[self.head : end].copy_(data)
        else:
            first = self.capacity - self.head
            self.buffer[self.head :].copy_(data[:first])
            self.buffer[: end % self.capacity].copy_(data[first:])

        self.head = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = torch.randint(0, self.size, (batch_size,), device=self.buffer.device)
        return self.buffer[idx]


@register_algorithm("ADDPPO", compat_name="add_ppo")
class ADDPPO(PPO):
    """PPO + ADD discriminator over observation differences."""

    policy: ActorCritic | ActorCriticRecurrent

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        disc_hidden_dims: list[int] | tuple[int, ...] = (1024, 512),
        disc_activation: str = "relu",
        disc_obs_group: str = "add_disc_obs",
        disc_demo_group: str = "add_disc_demo",
        task_reward_weight: float = 0.0,
        disc_reward_weight: float = 1.0,
        disc_reward_scale: float = 2.0,
        disc_batch_size: int = 2,
        disc_epochs: int = 2,
        disc_learning_rate: float = 2.5e-4,
        disc_replay_buffer_size: int = 200000,
        disc_replay_samples: int = 1000,
        disc_loss_weight: float = 1.0,
        disc_logit_reg: float = 0.01,
        disc_grad_penalty: float = 2.0,
        disc_weight_decay: float = 1.0e-4,
        **ppo_kwargs,
    ) -> None:
        if getattr(policy, "is_recurrent", False):
            raise NotImplementedError("Recurrent policies are not supported in ADDPPO v1.")

        super().__init__(policy, **ppo_kwargs)

        if disc_batch_size <= 0:
            raise ValueError(f"disc_batch_size must be positive, got {disc_batch_size}.")
        if disc_epochs <= 0:
            raise ValueError(f"disc_epochs must be positive, got {disc_epochs}.")

        self.disc_hidden_dims = list(disc_hidden_dims)
        self.disc_activation = disc_activation
        self.disc_obs_group = disc_obs_group
        self.disc_demo_group = disc_demo_group

        self.task_reward_weight = float(task_reward_weight)
        self.disc_reward_weight = float(disc_reward_weight)
        self.disc_reward_scale = float(disc_reward_scale)

        self.disc_batch_size = int(disc_batch_size)
        self.disc_epochs = int(disc_epochs)
        self.disc_learning_rate = float(disc_learning_rate)
        self.disc_replay_buffer_size = int(disc_replay_buffer_size)
        self.disc_replay_samples = int(disc_replay_samples)
        self.disc_loss_weight = float(disc_loss_weight)
        self.disc_logit_reg = float(disc_logit_reg)
        self.disc_grad_penalty = float(disc_grad_penalty)
        self.disc_weight_decay = float(disc_weight_decay)

        self.discriminator: MLP | None = None
        self.disc_optimizer: optim.Optimizer | None = None
        self.diff_normalizer: DiffNormalizer | None = None
        self.disc_replay_buffer: ReplayBuffer | None = None
        self.pos_diff: torch.Tensor | None = None
        self.disc_obs_dim: int | None = None
        self._bce = nn.BCEWithLogitsLoss()

        self._last_disc_reward_mean = 0.0
        self._last_disc_reward_std = 0.0

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        self._build_disc_modules(obs)

    def _build_disc_modules(self, obs: TensorDict) -> None:
        if self.disc_obs_group not in obs.keys():
            raise KeyError(
                f"ADDPPO expects observation group '{self.disc_obs_group}'. Available keys: {list(obs.keys())}"
            )
        if self.disc_demo_group not in obs.keys():
            raise KeyError(
                f"ADDPPO expects observation group '{self.disc_demo_group}'. Available keys: {list(obs.keys())}"
            )

        disc_obs_dim = int(obs[self.disc_obs_group].shape[-1])
        disc_demo_dim = int(obs[self.disc_demo_group].shape[-1])
        if disc_obs_dim != disc_demo_dim:
            raise ValueError(
                f"ADDPPO disc groups must match dims: {self.disc_obs_group}={disc_obs_dim}, "
                f"{self.disc_demo_group}={disc_demo_dim}."
            )

        self.disc_obs_dim = disc_obs_dim
        self.discriminator = MLP(disc_obs_dim, 1, self.disc_hidden_dims, self.disc_activation).to(self.device)
        self._init_disc_logit_layer()
        self.disc_optimizer = optim.SGD(
            self.discriminator.parameters(),
            lr=self.disc_learning_rate,
            weight_decay=self.disc_weight_decay,
        )
        self.diff_normalizer = DiffNormalizer(shape=(disc_obs_dim,), device=self.device, dtype=torch.float32)
        self.disc_replay_buffer = ReplayBuffer(
            self.disc_replay_buffer_size, disc_obs_dim, device=self.device, dtype=torch.float32
        )
        self.pos_diff = torch.zeros(1, disc_obs_dim, device=self.device, dtype=torch.float32)

        # Freeze action noise std to match MimicKit's FIXED std behaviour.
        # Without this, the optimizer shrinks std → entropy collapse → exploration death.
        if hasattr(self.policy, "std") and isinstance(self.policy.std, nn.Parameter):
            self.policy.std.requires_grad_(False)
        if hasattr(self.policy, "log_std") and isinstance(self.policy.log_std, nn.Parameter):
            self.policy.log_std.requires_grad_(False)

    def compute_returns(self, obs: TensorDict) -> None:
        if self.storage is None or self.discriminator is None or self.diff_normalizer is None or self.disc_replay_buffer is None:
            raise RuntimeError("ADDPPO storage/discriminator is not initialized. Call init_storage first.")

        disc_obs = self.storage.observations[self.disc_obs_group]
        disc_demo = self.storage.observations[self.disc_demo_group]
        obs_diff = (disc_demo - disc_obs).flatten(0, 1).detach()

        self._update_replay_buffer(obs_diff)
        # MimicKit parity: rewards in this iteration are computed with frozen/old stats.
        norm_diff = self.diff_normalizer.normalize(obs_diff)
        disc_r_flat = self._compute_disc_rewards(norm_diff)
        disc_r = disc_r_flat.view(self.storage.num_transitions_per_env, self.storage.num_envs, 1)

        task_r = self.storage.rewards
        self.storage.rewards = self.task_reward_weight * task_r + self.disc_reward_weight * disc_r

        # Accumulate current rollout stats; commit happens once at end of update().
        self.diff_normalizer.record(obs_diff)

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r_flat)
        self._last_disc_reward_mean = float(disc_reward_mean.item())
        self._last_disc_reward_std = float(disc_reward_std.item())

        super().compute_returns(obs)

    def _update_replay_buffer(self, obs_diff: torch.Tensor) -> None:
        if self.disc_replay_buffer is None:
            return
        n = obs_diff.shape[0]
        if n == 0:
            return
        rand_idx = torch.randperm(n, device=obs_diff.device, dtype=torch.long)
        if self.disc_replay_buffer.is_full():
            num_samples = min(n, self.disc_replay_samples)
        else:
            num_samples = n
        replay_data = obs_diff[rand_idx[:num_samples]]
        self.disc_replay_buffer.push(replay_data)

    def _compute_disc_rewards(self, norm_diff: torch.Tensor) -> torch.Tensor:
        if self.discriminator is None:
            raise RuntimeError("Discriminator has not been initialized.")
        with torch.no_grad():
            logits = self.discriminator(norm_diff).squeeze(-1)
            prob = torch.sigmoid(logits)
            rewards = -torch.log(torch.clamp(1.0 - prob, min=1.0e-4))
            rewards = rewards * self.disc_reward_scale
        return rewards

    def _get_disc_weights(self) -> list[torch.Tensor]:
        if self.discriminator is None:
            return []
        return [m.weight for m in self.discriminator.modules() if isinstance(m, nn.Linear)]

    def _get_disc_logit_layer(self) -> nn.Linear:
        if self.discriminator is None:
            raise RuntimeError("Discriminator has not been initialized.")
        for module in reversed(list(self.discriminator.modules())):
            if isinstance(module, nn.Linear):
                return module
        raise RuntimeError("No linear layer found in discriminator.")

    def _init_disc_logit_layer(self) -> None:
        logit_layer = self._get_disc_logit_layer()
        nn.init.uniform_(logit_layer.weight, -1.0, 1.0)
        if logit_layer.bias is not None:
            nn.init.zeros_(logit_layer.bias)

    def _get_disc_logit_weights(self) -> torch.Tensor:
        return self._get_disc_logit_layer().weight

    def _effective_disc_batch_size(self) -> int:
        if self.storage is None:
            return self.disc_batch_size
        return int(math.ceil(self.disc_batch_size * self.storage.num_envs))

    def _sample_disc_negatives(self, batch_size: int) -> torch.Tensor:
        if self.discriminator is None or self.diff_normalizer is None or self.disc_replay_buffer is None or self.pos_diff is None:
            raise RuntimeError("ADDPPO modules are not initialized.")

        if self.storage is None:
            raise RuntimeError("ADDPPO storage is not initialized.")

        rollout_diff = (
            self.storage.observations[self.disc_demo_group] - self.storage.observations[self.disc_obs_group]
        ).flatten(0, 1).detach()
        if rollout_diff.shape[0] == 0:
            raise RuntimeError("Empty discriminator rollout buffer.")

        rollout_idx = torch.randint(0, rollout_diff.shape[0], (batch_size,), device=rollout_diff.device)
        return rollout_diff[rollout_idx]

    def _compute_disc_loss(self, diff_obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.discriminator is None or self.diff_normalizer is None or self.pos_diff is None:
            raise RuntimeError("ADDPPO modules are not initialized.")

        if diff_obs.ndim != 2:
            raise ValueError(f"Expected discriminator negatives shaped [N, D], got {tuple(diff_obs.shape)}.")

        if self.disc_replay_buffer is not None and self.disc_replay_buffer.size > 0:
            replay_diff_obs = self.disc_replay_buffer.sample(diff_obs.shape[0])
            diff_obs = torch.cat([diff_obs, replay_diff_obs], dim=0)

        norm_diff_obs = self.diff_normalizer.normalize(diff_obs).detach()
        norm_diff_obs.requires_grad_(True)
        pos_diff = self.pos_diff.detach().expand(norm_diff_obs.shape[0], -1).clone()
        pos_diff.requires_grad_(True)

        disc_neg_logit = self.discriminator(norm_diff_obs).squeeze(-1)
        disc_pos_logit = self.discriminator(pos_diff).squeeze(-1)

        disc_loss_pos = self._bce(disc_pos_logit, torch.ones_like(disc_pos_logit))
        disc_loss_neg = self._bce(disc_neg_logit, torch.zeros_like(disc_neg_logit))
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        logit_weights = self._get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss = disc_loss + self.disc_logit_reg * disc_logit_loss

        disc_neg_grad = torch.autograd.grad(
            disc_neg_logit,
            norm_diff_obs,
            grad_outputs=torch.ones_like(disc_neg_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_pos_grad = torch.autograd.grad(
            disc_pos_logit,
            pos_diff,
            grad_outputs=torch.ones_like(disc_pos_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_neg_grad_penalty = torch.mean(torch.sum(torch.square(disc_neg_grad), dim=-1))
        disc_pos_grad_penalty = torch.mean(torch.sum(torch.square(disc_pos_grad), dim=-1))
        disc_grad_penalty = 0.5 * (disc_neg_grad_penalty + disc_pos_grad_penalty)
        disc_loss = disc_loss + self.disc_grad_penalty * disc_grad_penalty

        disc_weight_decay = torch.tensor(0.0, device=self.device)

        disc_neg_acc = (disc_neg_logit < 0).float().mean()
        disc_pos_acc = (disc_pos_logit > 0).float().mean()

        disc_info = {
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_grad_penalty_neg": disc_neg_grad_penalty.detach(),
            "disc_grad_penalty_pos": disc_pos_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_weight_decay": disc_weight_decay.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": torch.mean(disc_pos_logit).detach(),
            "disc_neg_logit": torch.mean(disc_neg_logit).detach(),
        }
        return disc_loss, disc_info

    def update(self) -> dict[str, float]:  # noqa: C901
        if self.discriminator is None or self.disc_optimizer is None:
            raise RuntimeError("Discriminator has not been initialized.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_sonic_loss = 0.0
        mean_vel_loss = 0.0
        mean_disc_loss = 0.0
        mean_disc_grad_penalty = 0.0
        mean_disc_logit_loss = 0.0
        mean_disc_weight_decay = 0.0
        mean_disc_pos_acc = 0.0
        mean_disc_neg_acc = 0.0
        mean_disc_pos_logit = 0.0
        mean_disc_neg_logit = 0.0
        mean_sonic_aux_stats: dict[str, float] = {}

        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

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
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            sonic_loss, sonic_aux_stats = self._compute_policy_aux_loss(obs_batch)
            loss += sonic_loss

            vel_loss = None
            if self.vel_loss_coef > 0.0 and hasattr(self.policy, "get_last_aux_outputs"):
                aux = self.policy.get_last_aux_outputs(clear=True)
                v_hat = aux.get("v_hat", None)
                if v_hat is not None:
                    v_hat = v_hat[:original_batch_size]
                    v_gt = None
                    if hasattr(self.policy, "obs_groups"):
                        vel_keys = self.policy.obs_groups.get("vel_gt", [])
                        if vel_keys and all(k in obs_batch.keys() for k in vel_keys):
                            v_gt = torch.cat([obs_batch[k] for k in vel_keys], dim=-1)
                    if v_gt is None and "vel_gt" in obs_batch.keys():
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
                        vel_loss = F.huber_loss(v_hat, v_gt, delta=self.vel_loss_delta)
                    elif self.vel_loss_type == "mse":
                        vel_loss = F.mse_loss(v_hat, v_gt)
                    else:
                        raise ValueError(f"Unknown vel_loss_type: {self.vel_loss_type}. Use 'huber' or 'mse'.")
                    loss += self.vel_loss_coef * vel_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()

            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if isinstance(sonic_loss, torch.Tensor):
                mean_sonic_loss += sonic_loss.item()
            for stat_name, stat_value in sonic_aux_stats.items():
                mean_sonic_aux_stats[stat_name] = mean_sonic_aux_stats.get(stat_name, 0.0) + stat_value
            if vel_loss is not None:
                mean_vel_loss += float(vel_loss.item())
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_sonic_loss /= num_updates
        mean_vel_loss /= num_updates
        for stat_name in list(mean_sonic_aux_stats.keys()):
            mean_sonic_aux_stats[stat_name] /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        disc_batch_size = self._effective_disc_batch_size()
        rollout_sample_count = self.storage.observations[self.disc_obs_group].flatten(0, 1).shape[0]
        num_disc_batches = max(1, int(math.ceil(float(rollout_sample_count) / disc_batch_size)))
        num_disc_steps = num_disc_batches * self.disc_epochs
        for _ in range(num_disc_steps):
            diff_obs = self._sample_disc_negatives(disc_batch_size)
            disc_loss, disc_info = self._compute_disc_loss(diff_obs)
            disc_step_loss = self.disc_loss_weight * disc_loss

            self.disc_optimizer.zero_grad()
            disc_step_loss.backward()

            if self.is_multi_gpu:
                self._reduce_disc_parameters()

            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.disc_optimizer.step()

            mean_disc_loss += disc_loss.item()
            mean_disc_grad_penalty += float(disc_info["disc_grad_penalty"].item())
            mean_disc_logit_loss += float(disc_info["disc_logit_loss"].item())
            mean_disc_weight_decay += float(disc_info["disc_weight_decay"].item())
            mean_disc_pos_acc += float(disc_info["disc_pos_acc"].item())
            mean_disc_neg_acc += float(disc_info["disc_neg_acc"].item())
            mean_disc_pos_logit += float(disc_info["disc_pos_logit"].item())
            mean_disc_neg_logit += float(disc_info["disc_neg_logit"].item())

        mean_disc_loss /= num_disc_steps
        mean_disc_grad_penalty /= num_disc_steps
        mean_disc_logit_loss /= num_disc_steps
        mean_disc_weight_decay /= num_disc_steps
        mean_disc_pos_acc /= num_disc_steps
        mean_disc_neg_acc /= num_disc_steps
        mean_disc_pos_logit /= num_disc_steps
        mean_disc_neg_logit /= num_disc_steps

        # MimicKit parity: update normalizer once per training iteration (after optimization).
        if self.diff_normalizer is not None:
            self.diff_normalizer.update()

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "sonic": mean_sonic_loss,
            "disc": mean_disc_loss,
            "disc_grad_penalty": mean_disc_grad_penalty,
            "disc_logit_reg": mean_disc_logit_loss,
            "disc_weight_decay": mean_disc_weight_decay,
            "disc_pos_acc": mean_disc_pos_acc,
            "disc_neg_acc": mean_disc_neg_acc,
            "disc_pos_logit": mean_disc_pos_logit,
            "disc_neg_logit": mean_disc_neg_logit,
            "disc_reward_mean": self._last_disc_reward_mean,
            "disc_reward_std": self._last_disc_reward_std,
        }
        if self.vel_loss_coef > 0.0:
            loss_dict["vel_estimator"] = mean_vel_loss
        loss_dict.update(mean_sonic_aux_stats)
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        if self.discriminator is None or self.diff_normalizer is None:
            return
        model_params = [self.discriminator.state_dict(), self.diff_normalizer.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.discriminator.load_state_dict(model_params[0])
        self.diff_normalizer.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]

        if len(grads) == 0:
            return

        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def _reduce_disc_parameters(self) -> None:
        if self.discriminator is None:
            return

        grads = [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
        if len(grads) == 0:
            return

        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.discriminator.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
