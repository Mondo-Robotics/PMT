"""ADD PPO agent cfg (ported from the old rsl_rl_add_cfg:G1AddFlatRunnerCfg).

PMT plan §10 PART A / §6 Phase-1 slice. ``build_agent_cfg`` returns a fresh instance
per call (no module-level singleton; §10/D). The algorithm ``class_name`` is "ADDPPO"
(registry.ALGORITHMS key, compat_name="add_ppo"); the policy is the plain
``ActorCritic`` (compat network "mlp"). obs_groups declares policy/critic + the two
discriminator sets, matching compat.SPECS["add_ppo"].required_obs_sets.

Values follow MimicKit's G1 ADD agent config.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlAddPpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)


@configclass
class G1AddFlatRunnerCfg(RslRlOnPolicyRunnerCfg):
    """ADD runner configuration for Unitree G1 multi-motion tracking."""

    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 200
    experiment_name = "g1_add_multimotionv2_flat"
    resume = False

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "add_disc_obs": ["add_disc_obs"],
        "add_disc_demo": ["add_disc_demo"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 512, 256],
        critic_hidden_dims=[1024, 512, 512, 256],
        activation="elu",
    )

    algorithm = RslRlAddPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        disc_obs_group="add_disc_obs",
        disc_demo_group="add_disc_demo",
        disc_hidden_dims=[1024, 512],
        disc_activation="relu",
        task_reward_weight=0.0,  # match working reference ADD run (pure discriminator)
        disc_reward_weight=1.0,
        disc_reward_scale=2.0,
        disc_batch_size=2,
        disc_epochs=2,
        disc_learning_rate=2.5e-4,
        disc_replay_buffer_size=200000,
        disc_replay_samples=1000,
        disc_loss_weight=1.0,
        disc_logit_reg=0.01,
        disc_grad_penalty=2.0,
        disc_weight_decay=1.0e-4,
    )
