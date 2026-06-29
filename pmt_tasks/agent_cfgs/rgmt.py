"""RGMT PPO agent cfgs."""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.agent_cfgs.transformer import RslRlPpoTransformerActorCriticCfg
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class G1RGMTPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Paper-faithful RGMT deploy runner.

    TransformerActorCritic is intentionally the compact RGMT architecture used
    in this repo: one causal history self-attention block, one command
    cross-attention block, and an MLP actor/critic head with n_embd=128.
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "rgmt"
    obs_groups = {
        "policy": ["proprio"],
        "policy_history": ["proprio_history"],
        "command_window": ["command_window"],
        "critic": ["critic"],
    }
    policy = RslRlPpoTransformerActorCriticCfg(
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
