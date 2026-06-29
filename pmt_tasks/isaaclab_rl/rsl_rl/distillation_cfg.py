# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rl_cfg import RslRlBaseRunnerCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlDistillationStudentTeacherCfg:
    """Configuration for the distillation student-teacher networks."""

    class_name: str = "StudentTeacher"
    """The policy class name. Default is StudentTeacher."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the student policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    student_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the student network."""

    teacher_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the teacher network."""

    student_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the student network."""

    teacher_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the teacher network."""

    activation: str = MISSING
    """The activation function for the student and teacher networks."""

    teacher_ckpt_path: str | None = None
    """Absolute path to a pretrained teacher checkpoint (PPO ActorCritic).

    When provided, the teacher weights are loaded directly from this file during
    ``StudentTeacher.__init__``, bypassing the runner's ``--resume`` / ``--load_run`` logic.
    """

    motion_target_obs_key: str | None = None
    """Observation-group key providing the auxiliary future-motion target."""

    anchor_target_obs_key: str | None = None
    """Observation-group key providing the auxiliary anchor target."""


@configclass
class RslRlDistillationStudentTeacherRecurrentCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher recurrent networks."""

    class_name: str = "StudentTeacherRecurrent"
    """The policy class name. Default is StudentTeacherRecurrent."""

    rnn_type: str = MISSING
    """The type of the RNN network. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The hidden dimension of the RNN network."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN network."""

    teacher_recurrent: bool = MISSING
    """Whether the teacher network is recurrent too."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name. Default is Distillation."""

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use for the student policy."""

    loss_type: Literal["mse", "huber"] = "mse"
    """The loss type to use for the student policy."""

    behavior_loss_coef: float = 1.0
    """Coefficient for the behavior-cloning action loss."""

    latent_loss_coef: float = 0.0
    """Coefficient for PMA/PMT latent MSE loss."""

    delta_z_loss_coef: float = 0.0
    """Coefficient for PMA residual-token target loss."""

    latent_cosine_loss_coef: float = 0.0
    """Coefficient for one-minus-cosine latent alignment loss."""

    latent_norm_loss_coef: float = 0.0
    """Coefficient for PMA/PMT latent norm matching loss."""

    flat_identity_loss_coef: float = 0.0
    """Coefficient for identity residual penalty on flat or identity samples."""

    delta_smooth_loss_coef: float = 0.0
    """Coefficient for PMA residual smoothness loss."""

    motion_loss_coef: float = 0.0
    """Coefficient for the auxiliary future-motion loss."""

    anchor_loss_coef: float = 0.0
    """Coefficient for the auxiliary anchor loss."""

    vel_loss_coef: float = 0.0
    """Coefficient for the auxiliary velocity-estimator loss."""

    vel_loss_type: Literal["mse", "huber"] = "huber"
    """Loss type for the auxiliary velocity-estimator loss."""

    vel_loss_delta: float = 1.0
    """Delta parameter for Huber velocity loss."""

    anchor_est_loss_coef: float = 0.0
    """Coefficient for the auxiliary anchor-position-estimator loss."""

    anchor_est_loss_type: Literal["mse", "huber"] = "huber"
    """Loss type for the auxiliary anchor-position-estimator loss."""

    anchor_est_loss_delta: float = 1.0
    """Delta parameter for Huber anchor-position loss."""

    foot_traj_loss_coef: float = 0.0
    """Coefficient for the auxiliary foot-trajectory loss."""

    foot_traj_loss_type: Literal["mse", "huber"] = "huber"
    """Loss type for the auxiliary foot-trajectory loss."""

    foot_traj_loss_delta: float = 1.0
    """Delta parameter for Huber foot-trajectory loss."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for distillation algorithms."""

    class_name: str = "DistillationRunner"
    """The runner class name. Default is DistillationRunner."""

    policy: RslRlDistillationStudentTeacherCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""

    debug_use_teacher_actions_for_env_step: bool = False
    """Use teacher actions to step the environment during rollout (stage 1 distillation).

    When True, this overrides the mixed-rollout path and forces 100% teacher control.
    """

    debug_rollout_action_stats: bool = False
    """Print action statistics during rollout for debugging."""

    # -- DAgger-style mixed rollout ------------------------------------------

    student_mean_for_env_step: bool = False
    """Use the student's deterministic mean action (``act_inference``) instead of a
    noisy sample for environment stepping."""

    teacher_mix_start: float = 1.0
    """Probability of using teacher rollout control at the beginning of training."""

    teacher_mix_end: float = 0.0
    """Probability of using teacher rollout control at the end of the annealing window."""

    teacher_mix_anneal_iters: int = 0
    """Number of iterations over which to linearly anneal the teacher rollout probability
    from ``teacher_mix_start`` to ``teacher_mix_end``.  0 disables annealing."""
