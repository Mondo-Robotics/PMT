# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flow Policy Optimization++ (FPO++) algorithm for diffusion policies."""

from __future__ import annotations

import torch
import torch.nn as nn

from motion_tracking_rl.algorithms.fpo import FPO
from motion_tracking_rl.registry import register_algorithm


class _ActorEMA:
    """Exponential moving average of a subset of module parameters.

    Stores a shadow copy of each tracked parameter in-place; swap() temporarily
    moves the shadow weights into the live module and returns a restore handle.
    """

    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self._shadow: dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        d = self.decay
        for name, param in module.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            shadow.mul_(d).add_(param.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_into(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Swap shadow weights into module, returning the previous live weights."""
        backup: dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            backup[name] = param.detach().clone()
            param.data.copy_(shadow)
        return backup

    @torch.no_grad()
    def restore(self, module: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in module.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self._shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name in self._shadow:
                self._shadow[name].copy_(tensor)


def _ste_clamp(
    x: torch.Tensor,
    min_val: float | None = None,
    max_val: float | None = None,
) -> torch.Tensor:
    """Clamp in forward pass while preserving identity gradient."""
    clamped = torch.clamp(x, min=min_val, max=max_val)
    return x + (clamped - x).detach()


@register_algorithm("FPOPlus", compat_name="fpo_plus")
class FPOPlus(FPO):
    """FPO++ algorithm with per-sample ratios and ASPO trust region."""

    def __init__(
        self,
        policy,
        trust_region_mode: str = "aspo",
        use_aspo: bool | None = None,
        advantage_clamp: tuple[float, float] = (100.0, 100.0),
        cfm_loss_clamp: float | None = 20.0,
        cfm_loss_clamp_negative_advantages: bool = True,
        cfm_loss_clamp_negative_advantages_max: float = 20.0,
        cfm_diff_clamp: float | None = 10.0,
        cfm_diff_clamp_use_ste: bool = True,
        recompute_old_cfm_loss: bool = False,
        storage_action_noise_std: float = 0.0,
        ema_decay: float = 0.95,
        ema_warmup_steps: int = 500,
        optimizer: str = "adamw",
        weight_decay: float = 1e-4,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        **kwargs,
    ) -> None:
        kwargs.setdefault("num_fpo_samples", 16)
        kwargs.setdefault("clip_param", 0.05)
        kwargs.setdefault("num_learning_epochs", 16)
        kwargs.setdefault("learning_rate", 1e-4)
        kwargs.setdefault("schedule", "fixed")
        kwargs.setdefault("use_clipped_value_loss", False)
        kwargs.setdefault("bound_coef", 0.0)
        storage_action_noise_std = float(kwargs.setdefault("storage_action_noise_std", storage_action_noise_std))
        super().__init__(policy, **kwargs)

        if use_aspo is not None:
            trust_region_mode = "aspo" if use_aspo else "ppo"
        if trust_region_mode not in {"ppo", "spo", "aspo"}:
            raise ValueError(
                f"Unknown trust_region_mode: {trust_region_mode}. Supported: 'ppo', 'spo', 'aspo'."
            )

        self.trust_region_mode = trust_region_mode
        self.use_aspo = self.trust_region_mode == "aspo"
        self.advantage_clamp = advantage_clamp
        self.cfm_loss_clamp = cfm_loss_clamp
        self.cfm_loss_clamp_negative_advantages = cfm_loss_clamp_negative_advantages
        self.cfm_loss_clamp_negative_advantages_max = cfm_loss_clamp_negative_advantages_max
        self.cfm_diff_clamp = cfm_diff_clamp
        self.cfm_diff_clamp_use_ste = cfm_diff_clamp_use_ste
        self.recompute_old_cfm_loss = recompute_old_cfm_loss
        self.storage_action_noise_std = storage_action_noise_std
        self.ema_decay = float(ema_decay)
        self.ema_warmup_steps = int(ema_warmup_steps)
        self._ema_update_count = 0
        self.ema: _ActorEMA | None = None
        if self.ema_decay > 0.0:
            self.ema = _ActorEMA(self.policy, decay=self.ema_decay)
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        if self.trust_region_mode in {"spo", "aspo"} and self.positive_advantage:
            raise ValueError("trust_region_mode in {'spo', 'aspo'} requires positive_advantage=False.")
        if self.clip_param <= 0.0:
            raise ValueError("FPOPlus requires clip_param > 0.")

        if self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}. Supported: 'adam', 'adamw'.")

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cfm_ratio = 0.0
        mean_clip_fraction = 0.0
        mean_ratio_std = 0.0
        mean_ratio_absmax = 0.0
        mean_adv_std = 0.0
        mean_action_clip_frac = 0.0
        mean_old_cfm_loss = 0.0
        mean_current_cfm_loss = 0.0
        mean_cfm_diff_absmax = 0.0
        mean_cfm_diff_clamp_hit_frac = 0.0
        mean_cfm_loss_clamp_hit_frac = 0.0
        debug_nonfinite_batches = 0
        self._debug_update_batch_idx = 0
        old_cfm_loss_cache: list[torch.Tensor] = []
        batch_index = 0

        if self.policy.is_recurrent:
            raise NotImplementedError("Recurrent policies are not supported in FPO++.")

        generator = self.storage.fpo_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

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

            with torch.no_grad():
                positive_clamp, negative_clamp = self.advantage_clamp
                advantages_batch = advantages_batch.clamp(-negative_clamp, positive_clamp)

            if cfm_info_batch is None:
                raise RuntimeError("CFM info is missing. Ensure FPO++ storage is initialized.")

            rollout_cfm_loss, eps_sample, t_sample = cfm_info_batch
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

            if self.recompute_old_cfm_loss:
                epoch_idx = batch_index // self.num_mini_batches
                mini_idx = batch_index % self.num_mini_batches
                if epoch_idx == 0:
                    with torch.no_grad():
                        old_cfm_loss = self.policy.compute_cfm_loss(flat_obs, flat_acts, flat_eps, flat_t)
                    old_cfm_loss = old_cfm_loss.view(B, N).detach()
                    old_cfm_loss_cache.append(old_cfm_loss)
                else:
                    old_cfm_loss = old_cfm_loss_cache[mini_idx]
            else:
                old_cfm_loss = rollout_cfm_loss.to(self.device, dtype=target_dtype, non_blocking=True)

            current_cfm_loss = self.policy.compute_cfm_loss(flat_obs, flat_acts, flat_eps, flat_t)
            current_cfm_loss = current_cfm_loss.view(B, N)

            old_cfm_loss_raw = old_cfm_loss
            current_cfm_loss_raw = current_cfm_loss
            if self.cfm_loss_clamp is not None:
                old_cfm_loss = torch.clamp(old_cfm_loss, max=self.cfm_loss_clamp)
                current_cfm_loss = torch.clamp(current_cfm_loss, max=self.cfm_loss_clamp)

            if self.cfm_loss_clamp_negative_advantages:
                current_cfm_loss = torch.where(
                    advantages_batch.reshape(B, 1) < 0,
                    current_cfm_loss.clamp(max=self.cfm_loss_clamp_negative_advantages_max),
                    current_cfm_loss,
                )

            log_ratio_unclamped = old_cfm_loss - current_cfm_loss
            log_ratio = log_ratio_unclamped
            if self.cfm_diff_clamp is not None:
                if self.cfm_diff_clamp_use_ste:
                    log_ratio = _ste_clamp(log_ratio, max_val=self.cfm_diff_clamp)
                else:
                    log_ratio = torch.clamp(log_ratio, max=self.cfm_diff_clamp)
            ratio = torch.exp(log_ratio)

            value_batch = self.policy.evaluate(obs_batch)

            adv = advantages_batch.reshape(B, 1).expand(-1, N)

            if self.trust_region_mode == "ppo":
                surrogate = -adv * ratio
                surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            elif self.trust_region_mode == "spo":
                surrogate_loss = -torch.mean(
                    ratio * adv - (torch.abs(adv) / (2.0 * self.clip_param)) * (ratio - 1.0).pow(2)
                )
            elif self.trust_region_mode == "aspo":
                surrogate = -adv * ratio
                surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                ppo_loss = torch.max(surrogate, surrogate_clipped)
                spo_loss = -(ratio * adv - (torch.abs(adv) / (2.0 * self.clip_param)) * (ratio - 1.0).pow(2))
                surrogate_loss = torch.where(adv > 0, ppo_loss, spo_loss).mean()
            else:
                raise ValueError(f"Unknown trust_region_mode: {self.trust_region_mode}")

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
                loss = loss + self.bound_coef * self.policy.mean_bound_loss

            has_nonfinite = not (
                torch.isfinite(old_cfm_loss).all()
                and torch.isfinite(current_cfm_loss).all()
                and torch.isfinite(log_ratio).all()
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
                        f"[FPO+][ERROR][it={self.current_learning_iteration}][mb={self._debug_update_batch_idx}] "
                        "Non-finite tensor detected."
                    )
                    self._debug_print("[FPO+][ERROR] " + self._tensor_debug_stats("old_cfm_loss", old_cfm_loss))
                    self._debug_print("[FPO+][ERROR] " + self._tensor_debug_stats("current_cfm_loss", current_cfm_loss))
                    self._debug_print("[FPO+][ERROR] " + self._tensor_debug_stats("log_ratio", log_ratio))
                    self._debug_print("[FPO+][ERROR] " + self._tensor_debug_stats("ratio", ratio))
                    self._debug_print("[FPO+][ERROR] " + self._tensor_debug_stats("advantages", advantages_batch))
                self._debug_nonfinite_reported = True
                if self.debug_raise_on_nonfinite:
                    raise RuntimeError(
                        "Non-finite tensors detected in FPOPlus update. "
                        f"iteration={self.current_learning_iteration}, mini_batch={self._debug_update_batch_idx}"
                    )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                ratio_std = ratio.std(unbiased=False).item()
                ratio_absmax = ratio.abs().max().item()
                adv_std = advantages_batch.std(unbiased=False).item()
                old_cfm_mean = old_cfm_loss.mean().item()
                current_cfm_mean = current_cfm_loss.mean().item()
                cfm_diff_absmax = log_ratio_unclamped.abs().max().item()
                if self.cfm_diff_clamp is not None:
                    cfm_diff_clamp_hit_frac = (log_ratio_unclamped > self.cfm_diff_clamp).float().mean().item()
                else:
                    cfm_diff_clamp_hit_frac = 0.0
                if self.cfm_loss_clamp is not None:
                    cfm_loss_clamp_hit_frac = (
                        (old_cfm_loss_raw > self.cfm_loss_clamp).float().mean()
                        + (current_cfm_loss_raw > self.cfm_loss_clamp).float().mean()
                    ).mul(0.5).item()
                else:
                    cfm_loss_clamp_hit_frac = 0.0

                action_clip_limit = getattr(self.policy, "action_clip", None)
                if action_clip_limit is not None:
                    action_clip_frac = (actions_batch.abs() >= (float(action_clip_limit) - 1.0e-6)).float().mean().item()
                else:
                    action_clip_frac = 0.0

            should_log_batch_debug = (
                self.debug_numeric
                and self._debug_iter_enabled
                and self._debug_update_batch_idx < self.debug_update_batches
            )
            if should_log_batch_debug and self.gpu_global_rank == 0:
                self._debug_print(
                    f"[FPO+][DEBUG][it={self.current_learning_iteration}][mb={self._debug_update_batch_idx}] "
                    f"loss={loss.item():.3e} surr={surrogate_loss.item():.3e} vf={value_loss.item():.3e} "
                    f"ratio_mean={ratio.mean().item():.3e} ratio_absmax={ratio_absmax:.3e} ratio_std={ratio_std:.3e} "
                    f"adv_std={adv_std:.3e} action_clip_frac={action_clip_frac:.3e} "
                    f"old_cfm={old_cfm_mean:.3e} cur_cfm={current_cfm_mean:.3e} "
                    f"cfm_diff_absmax={cfm_diff_absmax:.3e} diff_clamp_hit={cfm_diff_clamp_hit_frac:.3e} "
                    f"loss_clamp_hit={cfm_loss_clamp_hit_frac:.3e}"
                )
                self._debug_print("[FPO+][DEBUG] " + self._tensor_debug_stats("actions_batch", actions_batch))
                self._debug_print("[FPO+][DEBUG] " + self._tensor_debug_stats("ratio", ratio))
                self._debug_print("[FPO+][DEBUG] " + self._tensor_debug_stats("advantages", advantages_batch))
                self._debug_print("[FPO+][DEBUG] " + self._tensor_debug_stats("old_cfm_loss", old_cfm_loss))
                self._debug_print("[FPO+][DEBUG] " + self._tensor_debug_stats("current_cfm_loss", current_cfm_loss))
                self._debug_print("[FPO+][DEBUG] " + self._tensor_debug_stats("log_ratio", log_ratio))

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cfm_ratio += ratio.mean().item()
            mean_clip_fraction += clip_frac
            mean_ratio_std += ratio_std
            mean_ratio_absmax += ratio_absmax
            mean_adv_std += adv_std
            mean_action_clip_frac += action_clip_frac
            mean_old_cfm_loss += old_cfm_mean
            mean_current_cfm_loss += current_cfm_mean
            mean_cfm_diff_absmax += cfm_diff_absmax
            mean_cfm_diff_clamp_hit_frac += cfm_diff_clamp_hit_frac
            mean_cfm_loss_clamp_hit_frac += cfm_loss_clamp_hit_frac
            self._debug_update_batch_idx += 1
            batch_index += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_cfm_ratio /= num_updates
        mean_clip_fraction /= num_updates
        mean_ratio_std /= num_updates
        mean_ratio_absmax /= num_updates
        mean_adv_std /= num_updates
        mean_action_clip_frac /= num_updates
        mean_old_cfm_loss /= num_updates
        mean_current_cfm_loss /= num_updates
        mean_cfm_diff_absmax /= num_updates
        mean_cfm_diff_clamp_hit_frac /= num_updates
        mean_cfm_loss_clamp_hit_frac /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "cfm_ratio": mean_cfm_ratio,
            "clip_fraction": mean_clip_fraction,
            "ratio_std": mean_ratio_std,
            "ratio_absmax": mean_ratio_absmax,
            "adv_std": mean_adv_std,
            "action_clip_frac": mean_action_clip_frac,
            "old_cfm_loss": mean_old_cfm_loss,
            "current_cfm_loss": mean_current_cfm_loss,
            "cfm_diff_absmax": mean_cfm_diff_absmax,
            "cfm_diff_clamp_hit_frac": mean_cfm_diff_clamp_hit_frac,
            "cfm_loss_clamp_hit_frac": mean_cfm_loss_clamp_hit_frac,
        }
        if self.debug_numeric or debug_nonfinite_batches > 0:
            loss_dict["debug_nonfinite_batches"] = float(debug_nonfinite_batches)

        if self.ema is not None:
            self._ema_update_count += 1
            if self._ema_update_count > self.ema_warmup_steps:
                self.ema.update(self.policy)
        return loss_dict

    # ----- EMA checkpoint hooks used by the runner -----
    def ema_state_dict(self) -> dict | None:
        if self.ema is None:
            return None
        return {
            "shadow": self.ema.state_dict(),
            "decay": self.ema_decay,
            "warmup_steps": self.ema_warmup_steps,
            "update_count": self._ema_update_count,
        }

    def load_ema_state_dict(self, state: dict | None) -> None:
        if state is None or self.ema is None:
            return
        if "shadow" in state:
            self.ema.load_state_dict(state["shadow"])
        self._ema_update_count = int(state.get("update_count", self._ema_update_count))

    def swap_ema_into_policy(self):
        """Context-manager style helper: returns backup, caller must call restore()."""
        if self.ema is None:
            return None
        return self.ema.swap_into(self.policy)

    def restore_from_ema_swap(self, backup) -> None:
        if self.ema is None or backup is None:
            return
        self.ema.restore(self.policy, backup)
