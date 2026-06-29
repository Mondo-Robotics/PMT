# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from motion_tracking_rl.algorithms.ppo import PPO
from motion_tracking_rl.registry import register_algorithm


@register_algorithm("BPO", compat_name="bpo")
class BPO(PPO):
    """Bounded Policy Optimization using the existing PPO training infrastructure.

    This implementation keeps the rollout/storage/auxiliary-loss path from the local
    PPO class and replaces PPO's clipped policy objective with BPO's bounded-ratio
    regression objective. It intentionally uses the value function as the BPO
    baseline so existing actor-critic modules can train without adding a median
    critic head.
    """

    def __init__(
        self,
        *args,
        reg_w: float = 0.01,
        use_median: bool = False,
        online_adv: bool = True,
        loss_type: str = "adv_TV",
        tv_loss_coef: float = 0.0,
        bpo_advantage_clip: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if reg_w <= 0.0:
            raise ValueError(f"reg_w must be > 0.0, got {reg_w}.")
        if bpo_advantage_clip <= 0.0:
            raise ValueError(f"bpo_advantage_clip must be > 0.0, got {bpo_advantage_clip}.")
        if loss_type not in ("adv_TV", "TV", "log_TV", "MSE", "RKL", "FKL", "JS"):
            raise ValueError(
                f"Unknown BPO loss_type: {loss_type}. "
                "Use one of 'adv_TV', 'TV', 'log_TV', 'MSE', 'RKL', 'FKL', or 'JS'."
            )
        if use_median:
            warnings.warn(
                "BPO use_median=True was requested, but this inherited PPO implementation "
                "does not add a median critic head. Falling back to the value-function baseline.",
                stacklevel=2,
            )

        self.reg_w = float(reg_w)
        self.use_median = False
        self.online_adv = bool(online_adv)
        self.loss_type = str(loss_type)
        self.tv_loss_coef = float(tv_loss_coef)
        self.bpo_advantage_clip = float(bpo_advantage_clip)

    def _compute_surrogate_loss(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        value_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        ratio = torch.exp(log_ratio)

        q_values = returns_batch.flatten()
        values = value_batch.flatten()
        value_advantage = q_values - values.detach()
        target_advantage = torch.clamp(
            value_advantage / self.reg_w,
            min=-self.bpo_advantage_clip,
            max=self.bpo_advantage_clip,
        )

        target_ratio = F.sigmoid(target_advantage) * (2.0 * self.clip_param) + (1.0 - self.clip_param)
        target_log_ratio = torch.log(target_ratio)
        ratio_error = target_ratio - ratio.flatten()

        if self.loss_type == "adv_TV":
            if self.online_adv:
                weights = torch.abs(value_advantage)
            else:
                weights = torch.abs(advantages_batch.flatten().detach())
            surrogate_loss = torch.mean(torch.abs(ratio_error) * weights)
            if self.tv_loss_coef > 0.0:
                surrogate_loss = surrogate_loss + self.tv_loss_coef * torch.mean(torch.abs(ratio_error))
        elif self.loss_type == "TV":
            surrogate_loss = torch.mean(torch.abs(ratio_error))
        elif self.loss_type == "log_TV":
            surrogate_loss = torch.mean(torch.abs(target_log_ratio - log_ratio.flatten()))
        elif self.loss_type == "MSE":
            surrogate_loss = torch.mean(torch.square(ratio_error))
        else:
            reverse_kl = torch.mean(target_ratio * (target_log_ratio - log_ratio.flatten()))
            forward_kl = torch.mean(ratio.flatten() * (log_ratio.flatten() - target_log_ratio))
            if self.loss_type == "RKL":
                surrogate_loss = reverse_kl
            elif self.loss_type == "FKL":
                surrogate_loss = forward_kl
            else:
                surrogate_loss = 0.5 * (reverse_kl + forward_kl)

        stats = {
            "ratio_mean": ratio.detach().mean(),
            "ratio_std": ratio.detach().float().std(unbiased=False),
            "ratio_max_abs": ratio.detach().abs().max(),
            "approx_kl": ((ratio.detach() - 1.0) - log_ratio.detach()).mean(),
            "clip_fraction": (torch.abs(ratio.detach() - 1.0) > self.clip_param).float().mean(),
            "bpo_target_ratio_mean": target_ratio.detach().mean(),
            "bpo_target_ratio_std": target_ratio.detach().float().std(unbiased=False),
            "bpo_abs_ratio_error": torch.abs(ratio_error.detach()).mean(),
            "bpo_abs_target_advantage": torch.abs(target_advantage.detach()).mean(),
        }
        return surrogate_loss, ratio, stats
