"""MultiMotion/Flat PPO + BPO agent cfgs (ported from rsl_rl_ppo_cfg /
rsl_rl_bpo_cfg G1FlatMultiMotion*RunnerCfg).

PMT plan §10 PART A / §6 Phase 2.1. ``build_agent_cfg`` returns a fresh instance per
call (no module-level singleton; §10/D). Both runners use the plain ``ActorCritic``
MLP policy (compat network "mlp") on the on_policy runner. The base, sampler-variant,
and streaming tasks all share ``G1FlatMultiMotionPPORunnerCfg``; the BPO task swaps in
``G1FlatMultiMotionBPORunnerCfg`` (algorithm-only delta).

Values are the verified ground-truth from the spec (rsl_rl_ppo_cfg.py:39 /
rsl_rl_bpo_cfg.py:7).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlBpoAlgorithmCfg,
    RslRlFpoDiffusionActorCriticCfg,
    RslRlFpoPlusAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1FlatMultiMotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the MultiMotion V2 flat family (base/uniform/adaptive/streaming)."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_multi_motion_flat"
    resume = False

    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatMultiMotionBPORunnerCfg(G1FlatMultiMotionPPORunnerCfg):
    """BPO runner: same env + MLP policy as the PPO runner; algorithm swap only."""

    experiment_name = "g1_multi_motion_flat_bpo"

    algorithm = RslRlBpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=3,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        reg_w=0.01,
        use_median=False,
        online_adv=True,
        loss_type="adv_TV",
        tv_loss_coef=0.0,
        bpo_advantage_clip=10.0,
    )


@configclass
class G1FlatSingleClipFpoPlusRunnerCfg(RslRlOnPolicyRunnerCfg):
    """FPO++ runner for single-clip flat motion tracking with diffusion policy.

    Hyperparameters ported from the reference FPO++ G1 whole-body motion-tracking
    config (amazon-far/fpo-control, isaaclab_fpo/task_cfgs.py:G1FlatMotionTrackingFlow
    PPORunnerCfg). The earlier PMT defaults (clip_param 0.05, advantage_clamp 100,
    cfm clamps 20/10, fixed LR) DIVERGED (reward collapsed to -157); the reference
    tracking task uses MUCH tighter clamps + adaptive LR because tracking advantages
    are high-variance. Key reference values: clip_param=0.01, advantage_clamp=(5,5),
    cfm_loss_clamp=3.0, cfm_diff_clamp_max=3.0, schedule=adaptive, num_mini_batches=6,
    num_steps_per_env=48, actor/critic [1024,512,256], cfm_loss_reduction=mean,
    action_perturb_std=0.1, no action clipping (PD targets can exceed joint limits).
    """

    num_steps_per_env = 48
    max_iterations = 20000
    save_interval = 500
    experiment_name = "g1_fpo_plus_flat"
    resume = False

    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlFpoDiffusionActorCriticCfg(
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        num_steps=64,
        solver_method="euler",
        parameterization="velocity",
        actor_scale=1.0,
        action_clip=None,          # tracking: PD targets may exceed joint limits (ref)
        cfm_loss_reduction="mean",  # ref G1 tracking uses mean (not sqrt)
        perturb_action_std=0.1,     # ref action_perturb_std for tracking exploration
    )

    algorithm = RslRlFpoPlusAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=False,
        num_learning_epochs=16,
        num_mini_batches=6,         # ref tracking
        learning_rate=1.0e-4,
        schedule="adaptive",        # ref tracking (was fixed → no LR adaptation)
        gamma=0.99,
        lam=0.95,
        desired_kl=1.0e-4,          # ref FPO default (tighter than PPO's 0.01)
        max_grad_norm=1.0,
        clip_param=0.01,            # ref tracking (PMT 0.05 was 5x too loose)
        num_fpo_samples=16,
        trust_region_mode="aspo",
        advantage_clamp=(5.0, 5.0),  # ref tracking (PMT default 100 → exploding grads)
        cfm_loss_clamp=3.0,          # ref tracking (PMT default 20 → CFM ratio blowup)
        cfm_diff_clamp_max=3.0,      # ref tracking (PMT default 10)
    )
