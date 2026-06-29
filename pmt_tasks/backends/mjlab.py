"""mjlab backend env emitter (MJLAB_BACKEND_PLAN.md Phase C).

Builds an mjlab ``ManagerBasedRlEnvCfg`` for a PMT task by *populating mjlab's proven
tracking template* (`mjlab.tasks.tracking`) from PMT's resolved OmegaConf config — rather
than re-deriving the env structure. This keeps the mjlab hot path identical to mjlab's own
well-tested task while letting PMT's YAML stay the source of truth for the values that vary
(motion dir, decimation/dt, reward weights, tracked bodies, anchor).

Scope (MVP): the G1 flat-tracking family (multimotionv2_*_flat, sonic_multimotion_flat).
Terrain / vision / distill tasks are out of scope until mjlab parity exists (see plan).

Clip handling: PMT/SONIC npz are in Isaac-Lab BFS axis order; mjlab indexes positionally
against MJCF order. We remap once at ingest (scripts/pmt_npz_to_mjlab.py) into a cached
mjlab-order dir and point the command at that.
"""

from __future__ import annotations

import os
from pathlib import Path

# PMT reward-weight key -> mjlab reward term name. PMT and mjlab descend from the same
# BeyondMimic reward stack, so most map 1:1. Where PMT uses a name mjlab lacks (joint_pos/
# joint_vel reward variants), we leave the mjlab default weight (documented, Phase E).
_REWARD_KEY_MAP = {
    "motion_global_anchor_pos": "motion_global_root_pos",
    "motion_global_anchor_ori": "motion_global_root_ori",
    "body_pos": "motion_body_pos",
    "body_ori": "motion_body_ori",
    "body_lin_vel": "motion_body_lin_vel",
    "body_ang_vel": "motion_body_ang_vel",
    "action_rate": "action_rate_l2",
    # PMT-only reward terms with no mjlab 1:1 equivalent (kept at mjlab defaults):
    #   joint_pos, joint_vel, torque
}

# mjlab-order clip cache root (remapped npz live here).
_MJLAB_CLIP_CACHE = Path(
    os.environ.get("PMT_MJLAB_CLIP_CACHE", "/tmp/pmt_mjlab_clips")
)


def _ensure_mjlab_order_clips(src: str) -> str:
    """Remap a clip / dir-of-clips from BFS (PMT) to MJCF (mjlab) order, cached.

    Returns the path to the mjlab-order clip (or dir). Idempotent.
    """
    from scripts.pmt_npz_to_mjlab import convert  # type: ignore

    src_p = Path(src)
    if src_p.is_dir():
        out_dir = _MJLAB_CLIP_CACHE / src_p.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for npz in sorted(src_p.rglob("*.npz")):
            rel = npz.relative_to(src_p)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                convert(str(npz), str(dst))
        return str(out_dir)
    out = _MJLAB_CLIP_CACHE / src_p.name
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        convert(str(src_p), str(out))
    return str(out)


def _first_clip(path: str) -> str:
    """mjlab's stock MotionCommand is single-clip; pick a representative clip for MVP."""
    p = Path(path)
    if p.is_dir():
        clips = sorted(p.rglob("*.npz"))
        if not clips:
            raise FileNotFoundError(f"no .npz under {path}")
        return str(clips[0])
    return str(p)


def build_flat_tracking_env(cfg, *, remap_clips: bool = True, single_clip: bool = True):
    """Emit an mjlab G1 flat-tracking env cfg from a PMT resolved config dict.

    Args:
        cfg: resolved OmegaConf (dict-like) from ``build_task_config``.
        remap_clips: convert PMT BFS-order clips to mjlab MJCF order at ingest.
        single_clip: MVP — mjlab stock MotionCommand is single-clip; multi-clip is Phase E.
    """
    from mjlab.tasks.tracking.config.g1.env_cfgs import (
        unitree_g1_flat_tracking_env_cfg,
    )
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    motion = cfg["motion"]
    robot = cfg.get("robot", {})

    env_cfg = unitree_g1_flat_tracking_env_cfg()

    # dt / decimation from PMT config (§3a dt-from-motion).
    decimation = int(robot.get("decimation", motion.get("decimation", 4)))
    sim_dt = float(robot.get("sim_dt", motion.get("sim_dt", 0.005)))
    env_cfg.decimation = decimation
    env_cfg.sim.mujoco.timestep = sim_dt

    # motion clips: remap BFS->MJCF, then point command at them. Reuse the builder's
    # canonical path-list normalization so a list-valued motion_files (e.g. a future
    # terrain/flat-mix clip set) is NOT stringified whole into one bogus path. The 7
    # flat tasks resolve motion_files to a single dir string today; if a list ever
    # reaches here we take the first entry (mjlab's stock command is single-source).
    from pmt_tasks.builder import _as_path_list

    paths = _as_path_list(motion["motion_files"])
    src = paths[0]
    clip_path = _ensure_mjlab_order_clips(src) if remap_clips else src
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = _first_clip(clip_path) if single_clip else clip_path

    # reward weights from PMT config (mapped names; unmapped keep mjlab defaults).
    weights = cfg.get("reward_weights") or {}
    for pmt_key, w in weights.items():
        mj_key = _REWARD_KEY_MAP.get(pmt_key)
        if mj_key and mj_key in env_cfg.rewards:
            env_cfg.rewards[mj_key].weight = float(w)

    return env_cfg


# Dispatch table parallel to builder._ENV_BUILDERS, keyed by task stem.
_MJLAB_ENV_BUILDERS = {
    "multimotionv2_flat": build_flat_tracking_env,
    "multimotionv2_uniform_flat": build_flat_tracking_env,
    "multimotionv2_adaptive_flat": build_flat_tracking_env,
    "multimotionv2_streaming_flat": build_flat_tracking_env,
    "multimotionv2_streaming_100style": build_flat_tracking_env,
    "multimotionv2_100style_flat": build_flat_tracking_env,
    "sonic_multimotion_flat": build_flat_tracking_env,
}
