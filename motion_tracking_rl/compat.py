"""Compatibility matrix + AlgorithmSpec declarations (PMT plan §3b).

Invalid algorithm/network/feature combinations are declared here and validated at
config-build time (fail loud), rather than crashing at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    runner: str                      # e.g. "on_policy" | "distillation"
    compatible_networks: frozenset[str]  # {"mlp","transformer",...} or {"diffusion"} for fpo_plus
    requires_paired_command: bool    # distillation: teacher/student pair
    supports_rnd: bool
    supports_symmetry: bool
    supports_recurrent: bool
    required_obs_sets: frozenset[str]  # {"policy","critic"} | + {"discriminator"} | + {"teacher"}


SPECS: dict[str, AlgorithmSpec] = {
    # NOTE: bfm_zero RL-core is motion_tracking_rl.bfm_zero (+ scripts/bfm_zero/train.py);
    # its IsaacLab env cfg lives in pmt_tasks.env_cfgs.bfm_zero. It has its own FB-CPR-Aux
    # runner and intentionally stays outside the rsl_rl matrix.
    "ppo": AlgorithmSpec(
        "ppo", "on_policy",
        # perceptive_motion_token_tracker: PMT pretrain runs PPO from scratch
        # (rsl_rl_ppo_cfg PerceptiveMotionTokenTracker PMTPretrain runner, pmt_only_mode).
        # vision_ablation: architecture ablations (flat_mlp/split_mlp/split_cnn) run PPO.
        frozenset({
            "mlp", "transformer", "vision_transformer", "sonic",
            "perceptive_motion_token_tracker", "vision_ablation",
        }),
        False, True, True, True,
        frozenset({"policy", "critic"}),
    ),
    "bpo": AlgorithmSpec(
        "bpo", "on_policy",
        frozenset({"mlp", "transformer", "vision_transformer", "sonic"}),
        False, True, True, True,
        frozenset({"policy", "critic"}),
    ),
    # add_ppo rejects recurrent: add_ppo.py:97-98 raises NotImplementedError
    # ("Recurrent policies are not supported in ADDPPO v1.").
    # Real discriminator obs-group names come from RslRlAddPpoAlgorithmCfg
    # (rl_cfg.py:433 disc_obs_group="add_disc_obs", :436 disc_demo_group="add_disc_demo").
    "add_ppo": AlgorithmSpec(
        "add_ppo", "on_policy",
        frozenset({"mlp", "transformer"}),
        False, False, False, False,
        frozenset({"policy", "critic", "add_disc_obs", "add_disc_demo"}),
    ),
    "fpo_plus": AlgorithmSpec(
        "fpo_plus", "on_policy",
        frozenset({"diffusion"}),
        False, False, False, False,
        frozenset({"policy", "critic"}),
    ),
    "distillation": AlgorithmSpec(
        "distillation", "distillation",
        # PMA distill: PerceptiveMotionAdapterTracker / PerceptiveMotionTokenTracker
        # are distilled from a frozen tracker/teacher (Distillation runner). The
        # perceptive_motion_adapter building block + vision_ablation matched-
        # distill students are part of the same distillation family.
        frozenset({
            "student_teacher", "vision_student_latent_anchor",
            "perceptive_motion_adapter", "perceptive_motion_adapter_tracker",
            "perceptive_motion_token_tracker", "vision_ablation",
        }),
        True, False, False, True,
        frozenset({"policy", "teacher"}),
    ),
}


def validate(algorithm: str, network: str, *, rnd=False, symmetry=False, recurrent=False, obs_sets=None):
    """Raise ValueError with a clear message if the (algorithm, network, feature) combo is
    invalid; else return the resolved runner name."""
    if algorithm not in SPECS:
        raise ValueError(
            f"unknown algorithm '{algorithm}'; allowed: {sorted(SPECS)}"
        )
    spec = SPECS[algorithm]

    if network not in spec.compatible_networks:
        raise ValueError(
            f"algorithm '{algorithm}' is incompatible with network '{network}'; "
            f"compatible_networks: {sorted(spec.compatible_networks)}"
        )

    if rnd and not spec.supports_rnd:
        raise ValueError(
            f"algorithm '{algorithm}' does not support feature 'rnd'"
        )
    if symmetry and not spec.supports_symmetry:
        raise ValueError(
            f"algorithm '{algorithm}' does not support feature 'symmetry'"
        )
    if recurrent and not spec.supports_recurrent:
        raise ValueError(
            f"algorithm '{algorithm}' does not support feature 'recurrent'"
        )

    if obs_sets is not None:
        provided = set(obs_sets)
        missing = set(spec.required_obs_sets) - provided
        if missing:
            raise ValueError(
                f"algorithm '{algorithm}' requires obs_sets {sorted(spec.required_obs_sets)}; "
                f"missing: {sorted(missing)} (provided: {sorted(provided)})"
            )

    return spec.runner
