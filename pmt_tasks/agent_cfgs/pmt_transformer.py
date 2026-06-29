"""PMT transformer PPO agent cfgs."""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PMTAdaptiveSamplingTransformerActorCriticCfg(RslRlPpoActorCriticCfg):
    """Policy configuration for ``TransformerActorCritic``."""

    class_name: str = "TransformerActorCritic"

    n_embd: int = 128
    n_heads: int = 4
    history_len: int = 10
    cmd_len: int = 21
    mlp_ratio: int = 4

    state_dependent_std: bool = False
    log_std_bounds: tuple[float, float] = (-5.0, 2.0)
    min_std: float = 1e-6
    validate_args: bool = True

    history_obs_normalization: bool = True
    command_obs_normalization: bool = True

    use_vel_estimator: bool = False
    vel_estimator_detach: bool = True
    vel_estimator_hidden_dims: tuple[int, ...] = (64,)
    vel_estimator_output_dim: int = 3
    vel_gt_normalization: bool = True
    use_anchor_estimator: bool = False
    anchor_estimator_detach: bool = True
    anchor_estimator_hidden_dims: tuple[int, ...] = (64,)
    anchor_estimator_output_dim: int = 3
    anchor_gt_normalization: bool = True
    anchor_estimator_latent_inputs: tuple[str, ...] = ("h_last", "u_t")


@configclass
class G1PMTAdaptiveSamplingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Transformer PPO runner for the PMT adaptive-sampling task."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    motion_resample_frequency = 0
    experiment_name = "pmt_adaptive_sampling"
    obs_groups = {
        "policy": ["proprio"],
        "policy_history": ["proprio_history"],
        "command_window": ["command_window"],
        "critic": ["critic"],
    }
    policy = PMTAdaptiveSamplingTransformerActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        history_obs_normalization=True,
        command_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        n_embd=128,
        n_heads=4,
        history_len=10,
        cmd_len=21,
        mlp_ratio=4,
        state_dependent_std=False,
        log_std_bounds=(-5.0, 2.0),
        min_std=1e-6,
        validate_args=True,
        use_vel_estimator=False,
        use_anchor_estimator=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        vel_loss_coef=0.0,
        vel_loss_type="huber",
        vel_loss_delta=1.0,
        anchor_est_loss_coef=0.0,
        anchor_est_loss_type="huber",
        anchor_est_loss_delta=1.0,
    )
