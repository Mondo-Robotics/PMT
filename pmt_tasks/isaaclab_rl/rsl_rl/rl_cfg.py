# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    forward_hidden_dims: list[int] | None = None
    """Hidden dimensions for the forward/residual network. Default is None.

    When provided, the actor becomes a feature extractor outputting actor_hidden_dims[-1] dimensions,
    and the forward network takes concat(actor_output, forward_obs) to produce actions.
    
    For example, if actor_hidden_dims=[256, 256, 256] and forward_hidden_dims=[128, 128]:
    - Actor: obs_dim -> 256 -> 256 -> 256 (outputs 256-dim features)
    - Forward: (256 + forward_obs_dim) -> 128 -> 128 -> num_actions
    
    If None (default), the original single-stage actor is used.
    """

    forward_obs_normalization: bool | None = None
    """Whether to normalize forward observations. Default is None.
    
    If None, uses the same setting as actor_obs_normalization.
    """


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    backbone_lr_scale: float = 1.0
    """Relative LR multiplier for pretrained backbone parameter groups when the policy exposes them."""

    vision_adapter_lr_scale: float = 1.0
    """Relative LR multiplier for vision-adapter parameter groups when the policy exposes them."""

    critic_lr_scale: float = 1.0
    """Relative LR multiplier for critic parameter groups when the policy exposes them."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    # Optional supervised auxiliary losses (policy-specific)
    vel_loss_coef: float = 0.0
    """Coefficient for velocity estimator loss. Default is 0.0 (disabled)."""

    vel_loss_type: Literal["huber", "mse"] = "huber"
    """Loss type for velocity estimator. Default is huber."""

    vel_loss_delta: float = 1.0
    """Huber delta for velocity estimator loss. Default is 1.0."""

    anchor_est_loss_coef: float = 0.0
    """Coefficient for anchor-position estimator loss. Default is 0.0 (disabled)."""

    anchor_est_loss_type: Literal["huber", "mse"] = "huber"
    """Loss type for anchor-position estimator loss. Default is huber."""

    anchor_est_loss_delta: float = 1.0
    """Huber delta for anchor-position estimator loss. Default is 1.0."""

    foot_traj_loss_coef: float = 0.0
    """Coefficient for foot-trajectory auxiliary loss. Default is 0.0 (disabled)."""

    foot_traj_loss_type: Literal["huber", "mse"] = "huber"
    """Loss type for foot-trajectory auxiliary loss. Default is huber."""

    foot_traj_loss_delta: float = 1.0
    """Huber delta for foot-trajectory auxiliary loss. Default is 1.0."""

    foot_traj_target_obs_key: str | None = None
    """Optional observation-group key for foot-trajectory supervision target."""

    aux_loss_scale: float = 1.0
    """Global coefficient for policy auxiliary losses such as SONIC."""

    aux_loss_coef: dict[str, float] = field(default_factory=dict)
    """Per-term policy auxiliary loss coefficients."""

    sonic_loss_coef: float | None = None
    """Deprecated compatibility alias for aux_loss_scale."""

    use_mean_action_for_rollout: bool = False
    """If True, store and step the mean action while still evaluating PPO log-probability under the policy."""

    mean_action_rollout_iters: int = 0
    """Number of initial iterations to force mean-action rollout before using sampled policy actions."""

    value_only_warmup_iters: int = 0
    """Number of initial iterations to update only the value function before actor updates begin."""

    action_prior_loss_coef: float = 0.0
    """Coefficient for MSE against a policy-provided reference action, if available."""

    warmup_freeze_iters: int = 0
    """Number of initial learning iterations to run with configured policy module freezing."""

    warmup_freeze_encoders: bool = True
    """If True, freeze SONIC encoders during warmup iterations."""

    warmup_freeze_control_decoder: bool = True
    """If True, freeze SONIC control decoder during warmup iterations."""

    warmup_freeze_action_std: bool = True
    """If True, freeze action std parameter during warmup iterations."""

    warmup_reset_optimizer_state: bool = False
    """If True, reset Adam optimizer state when warmup transitions from frozen to unfrozen."""

    warmup_reset_rnd_optimizer_state: bool = False
    """If True, also reset RND optimizer state when warmup transitions from frozen to unfrozen."""

    warmup_unfreeze_lr_scale: float = 1.0
    """LR scale applied at frozen->unfrozen transition: lr = base_lr * scale."""

    debug_numeric: bool = False
    """If True, print PPO numeric diagnostics during rollout and update."""

    debug_print_freq: int = 1
    """Print debug diagnostics every N learning iterations."""

    debug_rollout_steps: int = 1
    """Number of rollout steps per iteration to log when debug is enabled."""

    debug_update_batches: int = 2
    """Number of PPO mini-batches per iteration to log when debug is enabled."""

    debug_raise_on_nonfinite: bool = False
    """If True, raise immediately when non-finite tensors are detected in rollout or update."""


@configclass
class RslRlBpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the BPO algorithm."""

    class_name: str = "BPO"
    """The algorithm class name. Default is BPO."""

    reg_w: float = 0.01
    """Temperature for converting value advantage into BPO target ratio."""

    use_median: bool = False
    """Whether to use a median critic baseline.

    The local inherited-PPO BPO implementation currently falls back to value
    baseline because existing actor-critic modules do not expose a median head.
    """

    online_adv: bool = True
    """Whether to weight BPO ratio regression by online return-minus-value advantage."""

    loss_type: Literal["adv_TV", "TV", "log_TV", "MSE", "RKL", "FKL", "JS"] = "adv_TV"
    """Policy loss form used by BPO."""

    tv_loss_coef: float = 0.0
    """Optional unweighted total-variation penalty added to adv_TV loss."""

    bpo_advantage_clip: float = 10.0
    """Clamp applied to scaled BPO target advantage before sigmoid."""


@configclass
class RslRlFpoDiffusionActorCriticCfg:
    """Configuration for the FPO++ diffusion actor-critic networks."""

    class_name: str = "DiffusionActorCritic"
    """The policy class name. Default is DiffusionActorCritic."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the diffusion actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    num_steps: int = 64
    """Number of solver steps for diffusion sampling."""

    solver_method: str = "euler"
    """ODE solver method for sampling: euler, midpoint, heun3, etc."""

    parameterization: str = "velocity"
    """Flow parameterization: 'data' (predict x1) or 'velocity'."""

    timestep_embed_dim: int = 8
    """Timestep embedding size used for AdaLN conditioning."""

    sample_t_strategy: str = "uniform"
    """Timestep sampling strategy: 'uniform' or 'lognormal'."""

    p_mean: float = -1.2
    """Lognormal sampling mean parameter (only for lognormal)."""

    p_std: float = 1.2
    """Lognormal sampling std parameter (only for lognormal)."""

    perturb_action_std: float = 0.02
    """Exploration noise added to sampled actions during training."""

    cfm_target_std: float | None = None
    """Deprecated Gaussian-NLL target scale; unused by the default velocity-CFM loss."""

    cfm_loss_reduction: Literal["mean", "sum", "sqrt"] = "sqrt"
    """Reduction for per-action CFM squared error."""

    cfm_loss_t_inverse_cdf_beta: float = 1.0
    """Beta inverse-CDF parameter for rollout CFM-loss timestep sampling."""

    actor_scale: float | list[float] = 1.0
    """Scale mapping latent diffusion actions to environment actions."""

    action_bound: float = 0.9
    """Soft action bound used for bound loss (actions are normalized to [-1, 1])."""

    action_clip: float | None = None
    """Optional hard clip for actions before storage/env step (match env clip_actions)."""

    zero_sampling_inference: bool = False
    """Whether act_inference should initialize sampling from zero noise."""

    rollout_zero_noise: bool = False
    """Whether to use zero noise during rollout action sampling (overrides perturb_action_std)."""

    loss_dim_mask: list[float] | None = None
    """Optional per-dimension loss mask for CFM loss (1.0=use, 0.0=ignore)."""


@configclass
class RslRlFpoPlusAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the FPO++ algorithm."""

    class_name: str = "FPOPlus"
    """The algorithm class name. Default is FPOPlus."""

    num_learning_epochs: int = 16
    """Number of learning epochs per update."""

    learning_rate: float = 1.0e-4
    """Learning rate for optimizer."""

    schedule: str = "fixed"
    """Learning rate schedule."""

    use_clipped_value_loss: bool = False
    """Whether to use PPO's clipped value loss."""

    clip_param: float = 0.05
    """Surrogate trust-region clipping parameter."""

    num_fpo_samples: int = 16
    """Number of (eps, t) samples for CFM loss computation."""

    positive_advantage: bool = False
    """Whether to apply softplus to advantages instead of normalization."""

    cfm_storage_dtype: str | None = None
    """Optional dtype for CFM storage (e.g., 'float16', 'bfloat16')."""

    cfm_storage_device: str | None = None
    """Optional device for CFM storage (e.g., 'cpu', 'cuda:0')."""

    bound_coef: float = 0.0
    """Coefficient for optional action-bound loss. Disabled by default."""

    cfm_diff_clamp_max: float = 10.0
    """One-sided STE clamp applied to the per-sample CFM log-ratio before exp()."""

    trust_region_mode: Literal["ppo", "spo", "aspo"] = "aspo"
    """Trust region method used for the FPO surrogate."""

    use_aspo: bool | None = None
    """Deprecated alias for trust_region_mode. True->'aspo', False->'ppo'."""

    advantage_clamp: tuple[float, float] = (100.0, 100.0)
    """Clamp advantages to [-negative_max, positive_max] before the surrogate loss."""

    cfm_loss_clamp: float | None = 20.0
    """Clamp individual CFM losses before differencing. None disables."""

    cfm_loss_clamp_negative_advantages: bool = True
    """Clamp current CFM loss for transitions with negative advantages."""

    cfm_loss_clamp_negative_advantages_max: float = 20.0
    """Maximum current CFM loss allowed for negative-advantage transitions."""

    cfm_diff_clamp: float | None = 10.0
    """Upper-clamp the log-ratio before exponentiation. None disables."""

    cfm_diff_clamp_use_ste: bool = True
    """Use straight-through gradient for diff clamp (forward clamp + identity backward)."""

    recompute_old_cfm_loss: bool = False
    """Recompute old CFM loss instead of using rollout-stored old CFM loss."""

    storage_action_noise_std: float = 0.0
    """Std of Gaussian noise added to stored rollout actions before old-CFM storage."""

    ema_decay: float = 0.95
    """EMA decay for the actor weights. Set >0 to enable an exponential moving
    average copy of the diffusion actor used for checkpoint export and eval."""

    ema_warmup_steps: int = 500
    """Number of update() calls before the EMA starts accumulating.

    Prevents very noisy early weights from contaminating the EMA. Ignored when
    ``ema_decay <= 0``.
    """

    optimizer: Literal["adam", "adamw"] = "adamw"
    """Optimizer type."""

    weight_decay: float = 1e-4
    """Weight decay used by AdamW."""

    adam_beta1: float = 0.9
    """Adam beta1 coefficient."""

    adam_beta2: float = 0.999
    """Adam beta2 coefficient (matches original FPO++ Adam betas (0.9, 0.999))."""


@configclass
class RslRlAddPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the ADD algorithm (PPO + discriminator on obs differences)."""

    class_name: str = "ADDPPO"
    """The algorithm class name. Default is ADDPPO."""

    disc_obs_group: str = "add_disc_obs"
    """Observation group containing agent discriminator observations."""

    disc_demo_group: str = "add_disc_demo"
    """Observation group containing demo discriminator observations."""

    disc_hidden_dims: list[int] = [1024, 512]
    """Discriminator hidden dimensions."""

    disc_activation: str = "relu"
    """Discriminator activation."""

    task_reward_weight: float = 0.0
    """Weight for original task reward."""

    disc_reward_weight: float = 1.0
    """Weight for discriminator reward."""

    disc_reward_scale: float = 2.0
    """Scale applied to discriminator reward."""

    disc_batch_size: int = 2
    """MimicKit discriminator batch multiplier; effective batch is ceil(value * num_envs)."""

    disc_epochs: int = 2
    """Number of discriminator-only epochs per PPO update."""

    disc_learning_rate: float = 2.5e-4
    """Learning rate for the separate discriminator optimizer."""

    disc_replay_buffer_size: int = 200000
    """Replay buffer capacity for difference samples."""

    disc_replay_samples: int = 1000
    """Maximum number of rollout samples pushed per update when replay is full."""

    disc_loss_weight: float = 1.0
    """Compatibility scalar applied only inside the discriminator optimizer step."""

    disc_logit_reg: float = 0.01
    """L2 regularization on discriminator logit layer weights."""

    disc_grad_penalty: float = 2.0
    """Gradient penalty coefficient."""

    disc_weight_decay: float = 1.0e-4
    """L2 regularization on all discriminator weights."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool | None = None
    """This parameter is deprecated and will be removed in the future.

    Use `actor_obs_normalization` and `critic_obs_normalization` instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "policy": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the policy will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    motion_resample_frequency: int = 0
    """Iterations between streaming motion working-set swaps. 0 disables swapping.

    When > 0, after every ``motion_resample_frequency`` PPO updates the runner asks
    any command term exposing ``resample_working_set()`` to swap its resident
    working set and force-reset all envs (see StreamingMultiMotionCommand). Only
    has an effect with a streaming command; ignored otherwise.
    """

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Default is False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""

    reset_action_std_on_load: float = -1.0
    """If positive, reset the loaded policy's action std to this value after loading a checkpoint.

    This is useful when PPO finetuning from deterministic distillation checkpoints whose
    stored exploration std is intentionally very small.
    """

    policy: RslRlPpoActorCriticCfg | RslRlFpoDiffusionActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlBpoAlgorithmCfg | RslRlFpoPlusAlgorithmCfg = MISSING
    """The algorithm configuration."""
