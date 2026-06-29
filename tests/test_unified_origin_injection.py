"""Pure tests for UnifiedMotionCommandV2 per-clip env-origin injection (Phase 2.3b).

The §9b plan of record extends ``UnifiedMultiMotionCommand`` with per-clip
``env_origins`` injection so it subsumes the grouped terrain+flat use-case WITHOUT a
hard env partition: terrain clips get a zero origin (they are world-placed on the mesh),
flat clips get ``flat_origin`` (shifted onto the dedicated flat patch -> no collision).

These tests exercise the pure, isaaclab-free helpers (``terrain_flag_for_files``,
``per_env_origin``). They load the command module DIRECTLY as a top-level module so the
heavy isaaclab block (gated on ``__package__``) is skipped -> runs in the wbt env.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

_MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "pmt_tasks", "mdp", "commands", "unified_motion_command.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ucmd_pure", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ucmd = _load_module()


# --------------------------------------------------------------------------
# terrain_flag_for_files: correctly tags a mixed (terrain + flat) clip list
# --------------------------------------------------------------------------
def test_terrain_flag_mixed_list():
    files = [
        "/data/terrain/a.npz",
        "/data/flat/x.npz",
        "/data/terrain/b.npz",
        "/data/flat/y.npz",
    ]
    terrain = ["/data/terrain/a.npz", "/data/terrain/b.npz"]
    flag = ucmd.terrain_flag_for_files(files, terrain)
    assert flag.tolist() == [True, False, True, False]


def test_terrain_flag_normalizes_paths():
    # membership is by absolute path; relative + ./ forms must still match.
    files = ["./terrain/a.npz", "flat/x.npz"]
    terrain = ["terrain/a.npz"]
    flag = ucmd.terrain_flag_for_files(files, terrain)
    assert flag.tolist() == [True, False]


def test_terrain_flag_all_flat_when_no_terrain():
    files = ["/f/1.npz", "/f/2.npz"]
    flag = ucmd.terrain_flag_for_files(files, [])
    assert flag.tolist() == [False, False]


# --------------------------------------------------------------------------
# strip_chunk_suffix + terrain_flag_for_paths: the eager store chunks long clips
# (source_file -> "<path>[chunk_i]") and may skip unreadable clips, so the per-clip
# flag must be rebuilt from the store's RESIDENT clip list, keyed by stripped path,
# and its length must equal the resident clip count (== motion_ids index space).
# This is the root-cause fix for the terrain_flat_mix CUDA device-side assert.
# --------------------------------------------------------------------------
def test_strip_chunk_suffix():
    assert ucmd.strip_chunk_suffix("/data/terrain/a.npz[chunk_0]") == "/data/terrain/a.npz"
    assert ucmd.strip_chunk_suffix("/data/terrain/a.npz[chunk_12]") == "/data/terrain/a.npz"
    assert ucmd.strip_chunk_suffix("/data/flat/x.npz") == "/data/flat/x.npz"


def test_terrain_flag_for_paths_handles_chunks_and_length():
    # cfg.motion_files order: 1 terrain clip + 1 flat clip. The store chunked the
    # terrain clip into 3 chunks (more entries than source clips) -> the flag MUST be
    # length 4 (== resident clip count), not 2 (== len(cfg.motion_files)).
    terrain_files = ["/data/terrain/a.npz"]
    resident_paths = [
        "/data/terrain/a.npz[chunk_0]",
        "/data/terrain/a.npz[chunk_1]",
        "/data/terrain/a.npz[chunk_2]",
        "/data/flat/x.npz",
    ]
    flag = ucmd.terrain_flag_for_paths(resident_paths, terrain_files)
    assert flag.tolist() == [True, True, True, False]
    # invariant: flag length == resident clip count (motion_ids index space).
    assert len(flag) == len(resident_paths)


# --------------------------------------------------------------------------
# per_env_origin: terrain rows -> 0, flat rows -> flat_origin
# --------------------------------------------------------------------------
def test_per_env_origin_split():
    is_terrain_env = torch.tensor([True, False, True, False, False])
    env_ids = torch.arange(5)
    flat_origin = [90.0, 0.0, 0.0]
    origins = ucmd.per_env_origin(is_terrain_env, env_ids, flat_origin, "cpu")

    assert origins.shape == (5, 3)
    # terrain rows zero
    assert torch.equal(origins[0], torch.zeros(3))
    assert torch.equal(origins[2], torch.zeros(3))
    # flat rows == flat_origin
    expected_flat = torch.tensor(flat_origin)
    assert torch.equal(origins[1], expected_flat)
    assert torch.equal(origins[3], expected_flat)
    assert torch.equal(origins[4], expected_flat)


def test_per_env_origin_all_terrain_is_zero():
    is_terrain_env = torch.tensor([True, True, True])
    env_ids = torch.arange(3)
    origins = ucmd.per_env_origin(is_terrain_env, env_ids, [90.0, 0.0, 0.0], "cpu")
    assert torch.equal(origins, torch.zeros(3, 3))


def test_per_env_origin_all_flat():
    is_terrain_env = torch.tensor([False, False])
    env_ids = torch.arange(2)
    flat_origin = [12.0, -3.0, 0.5]
    origins = ucmd.per_env_origin(is_terrain_env, env_ids, flat_origin, "cpu")
    assert torch.equal(origins, torch.tensor([flat_origin, flat_origin]))


def test_per_env_origin_subset_env_ids_order():
    # env_ids may be a SUBSET in arbitrary order; output rows follow env_ids order and
    # read each env's flag from the global is_terrain_env tensor.
    is_terrain_env = torch.tensor([True, False, True, False])
    env_ids = torch.tensor([3, 0])  # flat, terrain
    flat_origin = [90.0, 0.0, 0.0]
    origins = ucmd.per_env_origin(is_terrain_env, env_ids, flat_origin, "cpu")
    assert torch.equal(origins[0], torch.tensor(flat_origin))  # env 3 -> flat
    assert torch.equal(origins[1], torch.zeros(3))             # env 0 -> terrain


# --------------------------------------------------------------------------
# consistency: per_env_origin and per_env_pose_velocity_noise key on the SAME flag,
# so a terrain env gets BOTH zero origin AND zero noise (the no-collision invariant).
# --------------------------------------------------------------------------
def test_origin_and_noise_share_the_terrain_flag():
    is_terrain_env = torch.tensor([True, False])
    env_ids = torch.arange(2)
    flat_origin = [90.0, 0.0, 0.0]
    pose_range = {"x": (1.0, 1.0), "y": (1.0, 1.0)}
    vel_range = {}

    origins = ucmd.per_env_origin(is_terrain_env, env_ids, flat_origin, "cpu")
    pose_rand, _ = ucmd.per_env_pose_velocity_noise(
        is_terrain_env, env_ids, pose_range, vel_range, "cpu"
    )

    # terrain env (row 0): zero origin AND zero pose noise -> stays mesh-aligned.
    assert torch.equal(origins[0], torch.zeros(3))
    assert torch.equal(pose_rand[0], torch.zeros(6))
    # flat env (row 1): flat origin AND nonzero pose noise.
    assert torch.equal(origins[1], torch.tensor(flat_origin))
    assert pose_rand[1, 0] == 1.0 and pose_rand[1, 1] == 1.0
