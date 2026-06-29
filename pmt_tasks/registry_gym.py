"""Gym registration via factory closures (PMT plan §10 PART B / §10/C).

Isaac Lab keys its whole CLI chain on the gym task id (gym.spec(task).kwargs,
@hydra_task_config(task,...), gym.make(task, cfg=)). So PMT registers ONE
gym.register per configs/task/*.yaml, each with FRESH-per-call factory closures as
entry points (functools.partial over the builder + the yaml stem). load_cfg_from_
registry resolves a callable entry point natively, so no codegen is needed (§10/D).

Map: configs/task/<stem>.yaml  ->  gym id "PMT-<DerivedId>-v0".
For the slice we map the known stems to their derived ids deterministically; any
other task stem falls back to a "PMT-<stem>-v0" id.

Call ``register_pmt_tasks()`` AFTER the Isaac Lab app is launched (it imports the
env/agent cfg builders lazily inside the closures, so registration itself does not
require isaaclab — but gym.make later does).
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from pmt_tasks.builder import build_env_cfg, build_agent_cfg

CONFIGS_TASK_DIR = Path(__file__).resolve().parent.parent / "configs" / "task"

# Deterministic stem -> gym id map for the Phase-1 slice. Mirrors the old ids.
_TASK_ID_MAP = {
    "pmt_stepping_stone": "PMT-SteppingStone-G1-v0",
    "rgmt": "RGMT-G1-v0",
    "pmt_adaptive_sampling": "PMT-AdaptiveSampling-G1-v0",
    "pmt_adaptive_sampling_baseline": "PMT-AdaptiveSampling-Baseline-G1-v0",
    "distill_stepping_stone_latent_anchor": "PMT-Distill-SteppingStone-LatentAnchor-G1-v0",
    "ppofinetune_vision_teacher_stepping_stone_latent_anchor": (
        "PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0"
    ),
    "add_multimotion_flat": "PMT-ADD-MultiMotionV2-Flat-v0",
    "backflip": "PMT-Backflip-G1-v0",
    # First real training run: transformer teacher on big_map terrain + walk/dance clips.
    "walk_dance_bigmap": "PMT-WalkDanceBigMap-G1-v0",
    # transformer teacher on big_map terrain + terrain-anchored cartwheel clips (same stack).
    "cartwheel_bigmap": "PMT-CartwheelBigMap-G1-v0",
    # Phase 2.1 MultiMotion/Flat family (mirror the old G1-MultiMotionV2-* ids).
    "multimotionv2_flat": "PMT-G1-MultiMotionV2-Flat-v0",
    "multimotionv2_uniform_flat": "PMT-G1-MultiMotionV2-Uniform-Flat-v0",
    "multimotionv2_adaptive_flat": "PMT-G1-MultiMotionV2-Adaptive-Flat-v0",
    "multimotionv2_streaming_flat": "PMT-G1-MultiMotionV2-Streaming-Flat-v0",
    "multimotionv2_streaming_100style": "PMT-G1-MultiMotionV2-Streaming-100style-Flat-v0",
    "multimotionv2_100style_flat": "PMT-G1-MultiMotionV2-100style-Flat-v0",
    "bpo_multimotionv2_flat": "PMT-G1-BPO-MultiMotionV2-Flat-v0",
    "fpo_plus_flat": "PMT-G1-FPOPlus-SingleClip-Flat-v0",
    # Phase 2.2 SONIC family + distillation runner path.
    "sonic_multimotion_flat": "PMT-SONIC-G1-MultiMotionV2-Flat-v0",
    "distill_stepping_stone": "PMT-Distill-SteppingStone-G1-v0",
    # Phase 2.3b TerrainFlatMix (unified per-clip origin+noise, ONE store/sampler).
    "terrain_flat_mix": "PMT-TerrainFlatMix-G1-v0",
    # Phase 2.4 Omniretarget tasks DELETED (2026-06-25): training failed on the cluster
    # (legacy OmniretargetMultiMotionCommand produced recurring NaN rewards). The env cfg,
    # command/event code, scene/objects axis, and builder fn were removed entirely.
    # Phase 2.5 PerceptiveMotion family (token-tracker PPO pretrain, from-scratch).
    "perceptive_motion_token_tracker": "PMT-PerceptiveMotionTokenTracker-G1-v0",
    # P-CaRBT: FSQ behavior-token tracker, flat lafan1 PPO pretrain (from-scratch).
    "pmt_pcrbt": "PMT-PCaRBT-G1-v0",
    "pmt_pcrbt_100style": "PMT-PCaRBT-100style-G1-v0",
}


def gym_id_for_stem(stem: str) -> str:
    return _TASK_ID_MAP.get(stem, f"PMT-{stem}-v0")


def _make_env_factory(stem: str):
    """Return a FRESH-per-call env-cfg factory (a real function, not a partial).

    Isaac Lab's load_cfg_from_registry calls ``inspect.getfile(entry_point)`` on a
    callable entry point BEFORE invoking it; ``functools.partial`` has no
    ``__code__`` and trips getfile (TypeError). A nested ``def`` closure does have
    ``__code__`` (-> this module's file) and still re-invokes the builder each call.
    """

    def _env_cfg_entry_point():
        return build_env_cfg(stem)

    _env_cfg_entry_point.__name__ = f"build_env_cfg__{stem}"
    return _env_cfg_entry_point


def _make_agent_factory(stem: str):
    """Return a FRESH-per-call agent-cfg factory (a real function, not a partial)."""

    def _agent_cfg_entry_point():
        return build_agent_cfg(stem)

    _agent_cfg_entry_point.__name__ = f"build_agent_cfg__{stem}"
    return _agent_cfg_entry_point


def register_pmt_tasks() -> list[str]:
    """Loop configs/task/*.yaml and gym.register each (idempotent). Returns the ids.

    Most task YAMLs have direct env/agent builders and are launchable via their
    gym ids. A small number of registered slice/config targets intentionally raise
    at build time because they require extra assets or runtime wiring.
    """
    registered: list[str] = []
    for yaml_path in sorted(CONFIGS_TASK_DIR.glob("*.yaml")):
        stem = yaml_path.stem
        gym_id = gym_id_for_stem(stem)
        if gym_id in gym.registry:
            registered.append(gym_id)
            continue
        gym.register(
            id=gym_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                # FRESH per call (§10/D): each closure re-invokes the builder. We use
                # real nested-def closures (not functools.partial) so Isaac Lab's
                # inspect.getfile(entry_point) in load_cfg_from_registry succeeds.
                "env_cfg_entry_point": _make_env_factory(stem),
                "rsl_rl_cfg_entry_point": _make_agent_factory(stem),
            },
        )
        registered.append(gym_id)

    # BFM-Zero uses its own FB-CPR-Aux runner and agent config, not an rsl_rl
    # agent cfg. Register only the env cfg so scripts/bfm_zero/train.py can build
    # the task through Isaac Lab's normal parse_env_cfg/gym.make path.
    bfm_zero_id = "BFM-Zero-Flat-MultiMotionV2-G1-v0"
    if bfm_zero_id in gym.registry:
        registered.append(bfm_zero_id)
    else:
        gym.register(
            id=bfm_zero_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "pmt_tasks.env_cfgs.bfm_zero.bfm_zero:BFMZeroG1FlatMultiMotionV2EnvCfg"
                )
            },
        )
        registered.append(bfm_zero_id)
    return registered
