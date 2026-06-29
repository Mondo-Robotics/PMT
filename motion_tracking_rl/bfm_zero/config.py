"""Build the BFM-Zero ``FBcprAuxAgentConfig`` for the IsaacLab terrain+flat streaming G1 task.

This mirrors ``humanoidverse/train.py:train_bfm_zero``'s agent block but is parameterized for this
port. The BFM-Zero networks/agent are now **vendored** inside PMT
(``motion_tracking_rl.bfm_zero._vendor``), so no external ``BFM-Zero`` repo / ``sys.path`` injection
is required.

The agent consumes a Gymnasium ``Dict`` observation space with keys:
  - ``state``            (64)
  - ``privileged_state`` (208)
  - ``last_action``      (29)
  - ``history_actor``    (372)
and a 29-D continuous action space.
"""

from __future__ import annotations

import os
from pathlib import Path

# Kept only for backward-compatible env overrides / diagnostics; no longer used to locate code.
_PMT_REPO_ROOT = Path(os.environ.get("PMT_REPO_ROOT", Path(__file__).resolve().parents[2])).expanduser()


def ensure_bfm_zero_on_path(repo: str | None = None) -> str:
    """Deprecated no-op: the BFM-Zero (FB-CPR-Aux) code is vendored under
    ``motion_tracking_rl.bfm_zero._vendor`` and imported directly, so nothing needs to be added to
    ``sys.path``. Retained as a no-op so existing callers/launchers keep working.

    Returns the PMT repo root (purely informational).
    """
    return str(_PMT_REPO_ROOT)


# Default aux-reward set + scaling (verbatim from train_bfm_zero).
AUX_REWARDS = [
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
]
AUX_REWARDS_SCALING = {
    "penalty_action_rate": -0.1,
    "penalty_feet_ori": -0.4,
    "penalty_ankle_roll": -4.0,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "penalty_undesired_contact": -1.0,
    "penalty_torques": 0.0,
    "limits_torque": 0.0,
}


def build_agent_config(
    *,
    device: str = "cuda",
    z_dim: int = 256,
    seq_length: int = 8,
    actor_std: float = 0.05,
    compile_agent: bool = True,
    cudagraphs: bool = False,
    bfm_zero_repo: str | None = None,
    batch_size: int = 1024,
    rollout_expert_trajectories_length: int = 250,
    hidden_dim: int = 2048,
    hidden_layers: int = 6,
    embedding_layers: int = 2,
    disc_hidden_dim: int = 1024,
    disc_hidden_layers: int = 3,
    backward_hidden_dim: int = 256,
    backward_hidden_layers: int = 1,
):
    """Construct the ``FBcprAuxAgentConfig`` used to train BFM-Zero on this task.

    Network input filters reproduce the BFM-Zero defaults:
      - actor:       [state, last_action, history_actor]
      - backward/disc:[state, privileged_state]
      - forward/critic/aux_critic: [state, privileged_state, last_action, history_actor]
    """
    ensure_bfm_zero_on_path(bfm_zero_repo)

    # The BFM model config's ``device`` is a Literal['cpu', 'cuda']; normalize ``cuda:N`` -> 'cuda'
    # (with a single visible GPU, 'cuda' resolves to the same device the env/runner use).
    model_device = "cuda" if str(device).startswith("cuda") else "cpu"

    from ._vendor.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
    from ._vendor.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from ._vendor.agents.nn_filters import DictInputFilterConfig
    from ._vendor.agents.nn_models import (
        ActorArchiConfig,
        BackwardArchiConfig,
        DiscriminatorArchiConfig,
        ForwardArchiConfig,
        RewardNormalizerConfig,
    )
    from ._vendor.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig

    def _filter(*keys):
        return DictInputFilterConfig(name="DictInputFilterConfig", key=list(keys))

    # NOTE: ``num_parallel`` MUST stay 2 — the ensemble uncertainty divides by
    # ``num_parallel**2 - num_parallel`` (so 1 -> divide-by-zero / NaN). Only hidden width/depth
    # are shrunk for the smoke preset; the FB-CPR math is invariant to these.
    archi = FBcprAuxModelArchiConfig(
        name="FBcprAuxModelArchiConfig",
        z_dim=z_dim,
        norm_z=True,
        f=ForwardArchiConfig(
            name="ForwardArchi", hidden_dim=hidden_dim, model="residual", hidden_layers=hidden_layers,
            embedding_layers=embedding_layers, num_parallel=2, ensemble_mode="batch",
            input_filter=_filter("state", "privileged_state", "last_action", "history_actor"),
        ),
        b=BackwardArchiConfig(
            name="BackwardArchi", hidden_dim=backward_hidden_dim, hidden_layers=backward_hidden_layers, norm=True,
            input_filter=_filter("state", "privileged_state"),
        ),
        actor=ActorArchiConfig(
            name="actor", model="residual", hidden_dim=hidden_dim, hidden_layers=hidden_layers,
            embedding_layers=embedding_layers,
            input_filter=_filter("state", "last_action", "history_actor"),
        ),
        critic=ForwardArchiConfig(
            name="ForwardArchi", hidden_dim=hidden_dim, model="residual", hidden_layers=hidden_layers,
            embedding_layers=embedding_layers, num_parallel=2, ensemble_mode="batch",
            input_filter=_filter("state", "privileged_state", "last_action", "history_actor"),
        ),
        discriminator=DiscriminatorArchiConfig(
            name="DiscriminatorArchi", hidden_dim=disc_hidden_dim, hidden_layers=disc_hidden_layers,
            input_filter=_filter("state", "privileged_state"),
        ),
        aux_critic=ForwardArchiConfig(
            name="ForwardArchi", hidden_dim=hidden_dim, model="residual", hidden_layers=hidden_layers,
            embedding_layers=embedding_layers, num_parallel=2, ensemble_mode="batch",
            input_filter=_filter("state", "privileged_state", "last_action", "history_actor"),
        ),
    )

    model = FBcprAuxModelConfig(
        name="FBcprAuxModel",
        device=model_device,
        archi=archi,
        obs_normalizer=ObsNormalizerConfig(
            name="ObsNormalizerConfig",
            normalizers={
                "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
            },
            allow_mismatching_keys=True,
        ),
        inference_batch_size=500000,
        seq_length=seq_length,
        actor_std=actor_std,
        amp=False,
        norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
    )

    train = FBcprAuxAgentTrainConfig(
        name="FBcprAuxAgentTrainConfig",
        lr_f=0.0003, lr_b=1e-05, lr_actor=0.0003,
        weight_decay=0.0, clip_grad_norm=0.0,
        fb_target_tau=0.01, ortho_coef=100.0, train_goal_ratio=0.2,
        fb_pessimism_penalty=0.0, actor_pessimism_penalty=0.5, stddev_clip=0.3,
        q_loss_coef=0.0, batch_size=batch_size, discount=0.98,
        use_mix_rollout=True, update_z_every_step=100, z_buffer_size=8192,
        rollout_expert_trajectories=True, rollout_expert_trajectories_length=rollout_expert_trajectories_length,
        rollout_expert_trajectories_percentage=0.5,
        lr_discriminator=1e-05, lr_critic=0.0003, critic_target_tau=0.005,
        critic_pessimism_penalty=0.5, reg_coeff=0.05, scale_reg=True,
        expert_asm_ratio=0.6, relabel_ratio=0.8,
        grad_penalty_discriminator=10.0, weight_decay_discriminator=0.0,
        lr_aux_critic=0.0003, reg_coeff_aux=0.02, aux_critic_pessimism_penalty=0.5,
    )

    return FBcprAuxAgentConfig(
        name="FBcprAuxAgent",
        model=model,
        train=train,
        aux_rewards=list(AUX_REWARDS),
        aux_rewards_scaling=dict(AUX_REWARDS_SCALING),
        cudagraphs=cudagraphs,
        compile=compile_agent,
    )


def build_obs_space(*, device: str = "cpu"):
    """Build the Gymnasium Dict observation space the agent expects (sans ``time``)."""
    ensure_bfm_zero_on_path()
    import gymnasium
    import numpy as np

    from . import obs_math  # local import; pure-torch, no isaac

    def box(dim):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    return gymnasium.spaces.Dict(
        {
            "state": box(obs_math.STATE_DIM),
            "privileged_state": box(obs_math.PRIVILEGED_STATE_DIM),
            "last_action": box(obs_math.LAST_ACTION_DIM),
            "history_actor": box(obs_math.HistoryActorBuffer(1, 4).dim),
        }
    )
