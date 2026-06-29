"""PMT (PerceptiveMotionTokenTracker) PPO-pretrain agent cfg (plan §6 Phase 2.5).

Faithful copy of the old
``G1PerceptiveMotionTokenTrackerPMTPretrainWalkDancePPORunnerCfg``
(rsl_rl_ppo_cfg.py): runs the token tracker in ``pmt_only_mode=True`` /
``freeze_pmt=False`` / ``require_pmt_checkpoint=False`` / ``require_height_scan=False``,
so PPO trains the PMT decoder + tokenizer FROM SCRATCH with no pretrained PMT
checkpoint. This is the un-gated (no ckpt blocker) gate target for the family.

``build_agent_cfg`` returns a fresh instance per call (§10/D). The policy
``class_name`` resolves via the runner's ``eval()`` to the registered
``PerceptiveMotionTokenTracker`` (== compat axis ``perceptive_motion_token_tracker``).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PerceptiveMotionTokenTrackerPMTPretrainCfg:
    """Policy config for the PMT-pretrain (pmt_only_mode, from-scratch).

    Standalone @configclass (NOT inheriting RslRlPpoActorCriticCfg): the token
    tracker is a strict scaffold that rejects the base actor-critic normalization /
    noise-std / forward keys (it raises TypeError on unexpected kwargs). This mirrors
    the old standalone PerceptiveMotionTokenTrackerPMTPretrainCfg (rsl_rl_ppo_cfg.py).
    """

    class_name: str = "PerceptiveMotionTokenTracker"
    policy_set_name: str = "policy"
    history_set_name: str = "policy_history"
    future_motion_set_name: str = "future_motion_window"
    teacher_future_motion_set_name: str = "teacher_future_motion_window"
    height_scan_set_name: str = "height_scan"
    critic_set_name: str = "critic"
    future_motion_len: int = 21
    num_motion_tokens: int = 4
    motion_token_dim: int = 64
    model_dim: int = 128
    token_num_heads: int = 4
    history_embedding_dim: int = 128
    terrain_context_dim: int = 128
    foot_event_dim: int = 64
    num_height_tokens: int = 4
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256)
    activation: str = "elu"
    adapter_mode: str = "no_adapter"
    adapter_delta_scale: float = 0.0
    adapter_gate_bias: float = -4.0
    init_noise_std: float = 1.0
    freeze_pmt: bool = False
    pmt_only_mode: bool = True
    require_height_scan: bool = False
    require_teacher_motion_target: bool = False
    use_motion_aux_decoder: bool = True
    pmt_ckpt_path: str | None = None
    teacher_ckpt_path: str | None = None
    require_pmt_checkpoint: bool = False
    pmt_load_strict: bool = True


@configclass
class G1PerceptiveMotionTokenTrackerPMTPretrainRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Token-tracker PMT PPO-pretrain runner (from-scratch, no ckpt)."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_perceptive_motion_token_tracker_pmt_pretrain"
    resume = False

    obs_groups = {
        "policy": ["policy", "proprio"],
        "policy_history": ["proprio_history"],
        "future_motion_window": ["command_window", "motion_anchor_delta_window"],
        "height_scan": ["vision"],
        "critic": ["critic"],
    }

    policy = PerceptiveMotionTokenTrackerPMTPretrainCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
