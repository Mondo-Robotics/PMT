"""Phase C (MJLAB_BACKEND_PLAN.md): mjlab backend emitter + builder dispatch.

Dispatch-resolution tests run anywhere. The emitter test that builds a real mjlab env
is skipped unless mjlab is installed (the mjlab venv).
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_HAS_MJLAB = importlib.util.find_spec("mjlab") is not None


def test_resolve_backend_precedence():
    from pmt_tasks.builder import _resolve_backend

    os.environ.pop("PMT_BACKEND", None)
    assert _resolve_backend({}, None) == "isaaclab"  # default
    assert _resolve_backend({}, "mjlab") == "mjlab"  # explicit arg
    assert _resolve_backend({"backend": {"name": "mjlab"}}, None) == "mjlab"  # axis
    try:
        os.environ["PMT_BACKEND"] = "mjlab"
        assert _resolve_backend({}, None) == "mjlab"  # env var
        assert _resolve_backend({}, "isaaclab") == "isaaclab"  # arg beats env
    finally:
        os.environ.pop("PMT_BACKEND", None)


def test_reward_key_map_targets_exist_in_mjlab_template():
    """Every mapped PMT reward key must name a real mjlab reward term."""
    if not _HAS_MJLAB:
        pytest.skip("mjlab not installed")
    from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

    from pmt_tasks.backends.mjlab import _REWARD_KEY_MAP

    rewards = set(make_tracking_env_cfg().rewards.keys())
    for pmt_key, mj_key in _REWARD_KEY_MAP.items():
        assert mj_key in rewards, f"{pmt_key} -> {mj_key} not a mjlab reward term"


@pytest.mark.skipif(not _HAS_MJLAB, reason="needs mjlab")
def test_build_flat_tracking_env_from_pmt_cfg(tmp_path):
    """Emitter populates mjlab template from a PMT-style resolved config."""
    import numpy as np

    # tiny fake BFS-order clip (30 bodies, 29 joints) so the remap + load path runs.
    src = tmp_path / "clips"
    src.mkdir()
    T = 8
    np.savez(
        src / "fake.npz",
        fps=np.array([50]),
        joint_pos=np.zeros((T, 29), np.float32),
        joint_vel=np.zeros((T, 29), np.float32),
        body_pos_w=np.zeros((T, 30, 3), np.float32),
        body_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (T, 30, 1)),
        body_lin_vel_w=np.zeros((T, 30, 3), np.float32),
        body_ang_vel_w=np.zeros((T, 30, 3), np.float32),
    )

    from pmt_tasks.backends.mjlab import build_flat_tracking_env

    cfg = {
        "motion": {"motion_files": str(src), "decimation": 4, "sim_dt": 0.005},
        "robot": {"name": "g1", "decimation": 4, "sim_dt": 0.005},
        "reward_weights": {
            "motion_global_anchor_pos": 0.5,
            "action_rate": -0.1,
        },
    }
    env_cfg = build_flat_tracking_env(cfg)
    assert env_cfg.decimation == 4
    assert abs(env_cfg.sim.mujoco.timestep - 0.005) < 1e-9
    # mapped reward weights applied
    assert env_cfg.rewards["motion_global_root_pos"].weight == 0.5
    assert env_cfg.rewards["action_rate_l2"].weight == -0.1
    # clip got remapped into the cache and pointed at
    assert env_cfg.commands["motion"].motion_file.endswith(".npz")


@pytest.mark.skipif(not _HAS_MJLAB, reason="needs mjlab")
def test_build_flat_tracking_env_handles_list_motion_files(tmp_path):
    """A list-valued ``motion_files`` must NOT be stringified whole into one bogus path.

    Regression for the ``str(motion["motion_files"])`` shortcut: the emitter now routes
    through the builder's ``_as_path_list`` helper and takes the first entry (mjlab's
    stock MotionCommand is single-source).
    """
    import numpy as np

    src = tmp_path / "clips"
    src.mkdir()
    T = 8
    npz = src / "fake.npz"
    np.savez(
        npz,
        fps=np.array([50]),
        joint_pos=np.zeros((T, 29), np.float32),
        joint_vel=np.zeros((T, 29), np.float32),
        body_pos_w=np.zeros((T, 30, 3), np.float32),
        body_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (T, 30, 1)),
        body_lin_vel_w=np.zeros((T, 30, 3), np.float32),
        body_ang_vel_w=np.zeros((T, 30, 3), np.float32),
    )

    from pmt_tasks.backends.mjlab import build_flat_tracking_env

    cfg = {
        "motion": {"motion_files": [str(npz)], "decimation": 4, "sim_dt": 0.005},
        "robot": {"name": "g1", "decimation": 4, "sim_dt": 0.005},
        "reward_weights": {},
    }
    env_cfg = build_flat_tracking_env(cfg)
    # the first list entry was used (a real .npz), not a stringified list like "['...']"
    mf = env_cfg.commands["motion"].motion_file
    assert mf.endswith(".npz")
    assert "[" not in mf and "'" not in mf
