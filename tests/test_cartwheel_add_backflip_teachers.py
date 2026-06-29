"""Gate for the 2026-06-24 three-teacher batch (cartwheel teacher, ADD task-reward
blend, backflip curated-success repoint).

Pure path only (wbt-safe): exercises build_task_config under the cluster profile and
asserts the config/wiring changes that the cluster jobs depend on.
"""
from __future__ import annotations

from pmt_tasks.builder import build_task_config, load_paths
from pmt_tasks.registry_gym import gym_id_for_stem


def test_cartwheel_task_resolves_cluster_path():
    """Cartwheel teacher resolves to the verified 500-clip weight500 optimized dir."""
    cfg = build_task_config("cartwheel_bigmap", profile="cluster")
    paths = load_paths("cluster")
    assert (
        cfg["motion"]["motion_files"]
        == f"{paths.TERRAIN_MOTION_ROOT}/terrain/"
        "cartwheel_bigmap_n100_full_weight500/optimized"
    )
    # same big_map mesh as walk_dance teacher.
    assert cfg["terrain"]["mesh_path"].endswith("g1_29dof_big_map.stl")


def test_cartwheel_gym_id():
    assert gym_id_for_stem("cartwheel_bigmap") == "PMT-CartwheelBigMap-G1-v0"


def test_add_task_reward_weight_is_pure_disc():
    """ADD uses pure-discriminator reward (task_reward_weight=0.0), matching the proven
    working reference run (md_rl/ADD/2026-02-10_12-54-22_iter1, 40k iters at 0.0). ADD's
    single-digit reward scale is by design; blending a task reward deviated from the recipe."""
    cfg = build_task_config("add_multimotion_flat", profile="cluster")
    assert float(cfg["algorithm"]["task_reward_weight"]) == 0.0
    assert float(cfg["algorithm"]["disc_reward_weight"]) == 1.0


def test_backflip_uses_merged_optimized_set():
    """Backflip teacher trains on the full back_flip_merged/optimized set (cluster).
    The earlier plateau was a termination bug (full-XYZ ee_body_pos), now fixed to z-only,
    so the merged set is the correct data per user (2026-06-25)."""
    cfg = build_task_config("backflip", profile="cluster")
    assert cfg["motion"]["motion_files"] == load_paths("cluster").BACKFLIP_MOTION
    # §3a control rate preserved.
    assert int(cfg["motion"]["decimation"]) == 10
    assert float(cfg["motion"]["sim_dt"]) == 0.002
