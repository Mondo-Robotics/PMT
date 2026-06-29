"""P-CaRBT (PerceptiveResidualBehaviorTokenTracker) PPO-pretrain agent cfg.

Flat-ground feasibility run for the FSQ behavior tokenizer (docs/pcrbt_implementation_plan.md).
Mirrors the PMT-pretrain token-tracker runner (perceptive_motion_token.py) but:
  - swaps the policy ``class_name`` to ``PerceptiveResidualBehaviorTokenTracker``;
  - drops the ``height_scan`` obs set (reduced OBS; ``pmt_only_mode`` skips the adapter);
  - adds FSQ knobs (``fsq_levels``, ``num_residual_levels``) + the differentiable
    ``aux_loss_scale``/``aux_loss_coef`` so the FSQ-usage-entropy and motion-recon aux
    losses are actually weighted non-zero (ppo.py:559/576 — unset coef ⇒ silent 0).

From-scratch: ``pmt_only_mode=True`` / ``freeze_pmt=False`` / ``require_pmt_checkpoint=False``.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PerceptiveResidualBehaviorTokenTrackerCfg:
    """Policy config for P-CaRBT PPO-pretrain (pmt_only_mode, from-scratch, flat).

    Standalone @configclass (the token tracker rejects base actor-critic kwargs). Holds
    the inherited token-tracker knobs PLUS the new FSQ behavior-tokenizer knobs.
    """

    class_name: str = "PerceptiveResidualBehaviorTokenTracker"
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
    require_pmt_checkpoint: bool = False
    pmt_load_strict: bool = True
    # --- P-CaRBT FSQ behavior tokenizer knobs ---
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    num_residual_levels: int = 1
    use_phase_head: bool = False
    # Per-term aux coefs returned by compute_sonic_aux_losses (the algorithm's
    # aux_loss_coef OVERRIDES this; both are set, kept consistent).
    aux_loss_coef: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.aux_loss_coef is None:
            self.aux_loss_coef = {"fsq_usage_entropy": 0.01, "motion_recon": 0.1}


@configclass
class G1PerceptiveResidualBehaviorTokenTrackerRunnerCfg(RslRlOnPolicyRunnerCfg):
    """P-CaRBT PPO-pretrain runner (FSQ behavior tokens, flat lafan1, from-scratch)."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_pcrbt_flat_pretrain"
    resume = False

    # No height_scan set (reduced OBS; pmt_only_mode skips the terrain adapter).
    obs_groups = {
        "policy": ["policy", "proprio"],
        "policy_history": ["proprio_history"],
        "future_motion_window": ["command_window", "motion_anchor_delta_window"],
        "critic": ["critic"],
    }

    policy = PerceptiveResidualBehaviorTokenTrackerCfg()

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
        # Activate the behavior-tokenizer aux losses (ppo.py:559/576).
        aux_loss_scale=1.0,
        aux_loss_coef={"fsq_usage_entropy": 0.01, "motion_recon": 0.1},
    )
