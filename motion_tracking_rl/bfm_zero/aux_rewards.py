"""BFM-Zero auxiliary (safety) reward terms.

The faithful BFM-Zero agent (``FBcprAuxAgent``) trains an auxiliary critic ``Q_R`` on a set of
safety/penalty rewards that are NOT part of the FB latent objective. These are emitted by the env
into ``info["aux_rewards"]`` (raw, unscaled, per-env), and the agent applies its own scaling.

This module provides:
  1. Pure-torch formulas (``aux_*``) that take explicit tensors — testable WITHOUT Isaac Sim.
  2. ``compute_aux_rewards``: a thin orchestrator over a small tensor bundle.
  3. ``AUX_REWARD_KEYS`` and the default scaling used by the agent config.

All formulas are verbatim ports of the BFM-Zero implementations in
``humanoidverse/envs/legged_base_task/legged_robot_base.py`` and
``humanoidverse/envs/legged_robot_motions/legged_robot_motions.py``.

The IsaacLab-side extractor that gathers the required tensors from the env managers lives in
``bfm_zero_aux_rewards.py`` under ``tasks/tracking/mdp`` (imports isaaclab); this module stays pure.
"""

from __future__ import annotations

import torch

# Order matches the BFM-Zero default aux_rewards list in humanoidverse/train.py.
AUX_REWARD_KEYS: tuple[str, ...] = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)

# Default per-term scaling used by FBcprAuxAgentConfig.aux_rewards_scaling (from train.py).
DEFAULT_AUX_REWARDS_SCALING: dict[str, float] = {
    "penalty_action_rate": -0.1,
    "penalty_feet_ori": -0.4,
    "penalty_ankle_roll": -4.0,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "penalty_undesired_contact": -1.0,
    "penalty_torques": 0.0,
    "limits_torque": 0.0,
}

# Soft-limit factors from reward_bfm_zero.yaml.
SOFT_DOF_POS_LIMIT = 0.95
SOFT_TORQUE_LIMIT = 0.95

CONTACT_FORCE_THRESHOLD = 1.0  # N; matches BFM-Zero "> 1." checks


def aux_penalty_torques(torques: torch.Tensor) -> torch.Tensor:
    """sum_j torque_j^2."""
    return torch.sum(torch.square(torques), dim=1)


def aux_penalty_action_rate(last_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """sum_j (last_action_j - action_j)^2."""
    return torch.sum(torch.square(last_actions - actions), dim=1)


def aux_limits_dof_pos(
    dof_pos: torch.Tensor,
    dof_pos_limits_low: torch.Tensor,
    dof_pos_limits_high: torch.Tensor,
) -> torch.Tensor:
    """Soft joint-position limit excess (lower clip + upper clip), summed over joints.

    Mirrors ``_reward_limits_dof_pos`` (no-curriculum branch): the env's ``dof_pos_limits`` are
    already the soft limits (hard * soft_factor); pass soft limits directly.
    """
    out = -(dof_pos - dof_pos_limits_low).clip(max=0.0)
    out += (dof_pos - dof_pos_limits_high).clip(min=0.0)
    return torch.sum(out, dim=1)


def aux_limits_torque(torques: torch.Tensor, torque_limits: torch.Tensor, soft_factor: float = SOFT_TORQUE_LIMIT) -> torch.Tensor:
    """sum_j clip(|torque_j| - torque_limit_j * soft, min=0). Mirrors ``_reward_limits_torque``."""
    return torch.sum((torch.abs(torques) - torque_limits * soft_factor).clip(min=0.0), dim=1)


def aux_penalty_undesired_contact(contact_forces_penalised: torch.Tensor) -> torch.Tensor:
    """1.0 if any penalised body has |contact force| > threshold else 0.

    ``contact_forces_penalised``: [N, K, 3] forces on the penalised contact bodies.
    Mirrors ``_reward_penalty_undesired_contact``.
    """
    n = contact_forces_penalised.shape[0]
    res = torch.zeros(n, dtype=torch.float, device=contact_forces_penalised.device)
    undesired = torch.any(torch.abs(contact_forces_penalised) > CONTACT_FORCE_THRESHOLD, dim=(1, 2))
    res[undesired] = 1.0
    return res


def aux_penalty_ankle_roll(left_ankle_roll: torch.Tensor, right_ankle_roll: torch.Tensor) -> torch.Tensor:
    """sum(left_roll^2 + right_roll^2). Inputs are [N, 1] ankle-roll joint positions."""
    return torch.sum(torch.square(left_ankle_roll) + torch.square(right_ankle_roll), dim=1)


def aux_penalty_feet_ori(
    left_foot_quat_xyzw: torch.Tensor,
    right_foot_quat_xyzw: torch.Tensor,
    gravity_vec: torch.Tensor,
    feet_contact: torch.Tensor,
) -> torch.Tensor:
    """Foot-flatness penalty weighted by contact. Mirrors ``_reward_penalty_feet_ori``.

    ``feet_contact``: [N, 2] boolean/float contact flags (left, right).
    Uses BFM's ``quat_rotate_inverse`` (w_last=True).
    """
    from ._vendor.torch_utils import quat_rotate_inverse  # type: ignore

    left_g = quat_rotate_inverse(left_foot_quat_xyzw, gravity_vec, w_last=True)
    right_g = quat_rotate_inverse(right_foot_quat_xyzw, gravity_vec, w_last=True)
    left_term = torch.sum(torch.square(left_g[:, :2]), dim=1) ** 0.5 * feet_contact[:, 0]
    right_term = torch.sum(torch.square(right_g[:, :2]), dim=1) ** 0.5 * feet_contact[:, 1]
    return left_term + right_term


def aux_penalty_slippage(foot_vel: torch.Tensor, foot_contact_forces: torch.Tensor) -> torch.Tensor:
    """sum_f ||foot_vel_f|| * (||contact_force_f|| > threshold). Mirrors ``_reward_penalty_slippage``.

    ``foot_vel``: [N, F, 3]; ``foot_contact_forces``: [N, F, 3].
    """
    speed = torch.norm(foot_vel, dim=-1)
    in_contact = torch.norm(foot_contact_forces, dim=-1) > CONTACT_FORCE_THRESHOLD
    return torch.sum(speed * in_contact, dim=1)
