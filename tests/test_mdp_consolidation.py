"""Pure equivalence tests for the §9b MDP consolidation (Phase 2.3a).

These prove the consolidation is BEHAVIOR-PRESERVING: each consolidated
parametrized function produces output IDENTICAL to the old per-variant function
it replaced, on a synthetic command. The old variant names are kept as thin
wrappers, so we also assert wrapper == param-func.

Runs in the pure ``wbt`` env. The mdp modules import isaaclab/omni at load time,
so we install minimal stubs in ``sys.modules`` BEFORE importing them. The only
isaaclab math the consolidated paths touch (``subtract_frame_transforms``,
``matrix_from_quat``) is provided here with a real torch implementation, so the
equivalence check exercises the genuine numerics, not a no-op.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------
# Minimal isaaclab / omni stubs (installed before importing the mdp modules).
# --------------------------------------------------------------------------
def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    # q = (w, x, y, z), real torch implementation matching isaaclab convention.
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    m = torch.stack(
        [
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    vq = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return _quat_mul(_quat_mul(q, vq), _quat_inv(q))[..., 1:].contiguous()


def _subtract_frame_transforms(t01, q01, t02=None, q02=None):
    # Express frame 2 in frame 1: t12 = R(q01)^-1 (t02 - t01); q12 = q01^-1 * q02
    q01_inv = _quat_inv(q01)
    t12 = _quat_apply(q01_inv, (t02 - t01))
    q12 = _quat_mul(q01_inv, q02)
    return t12, q12


def _install_stubs():
    if "_pmt_consolidation_stubs" in sys.modules:
        return

    omni = types.ModuleType("omni")
    sys.modules["omni"] = omni

    isaaclab = types.ModuleType("isaaclab")
    sys.modules["isaaclab"] = isaaclab

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.matrix_from_quat = _matrix_from_quat
    math_mod.quat_apply = _quat_apply
    math_mod.quat_inv = _quat_inv
    math_mod.quat_mul = _quat_mul
    math_mod.subtract_frame_transforms = _subtract_frame_transforms
    math_mod.quat_error_magnitude = lambda a, b: torch.zeros(a.shape[0])
    math_mod.yaw_quat = lambda q: q
    math_mod.quat_apply_inverse = lambda q, v: _quat_apply(_quat_inv(q), v)
    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.math = math_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod

    managers_mod = types.ModuleType("isaaclab.managers")

    class _SceneEntityCfg:  # used as a default-arg value at def time (rewards.py)
        def __init__(self, *a, **k):
            self.args = a
            self.kwargs = k

    managers_mod.SceneEntityCfg = _SceneEntityCfg
    sys.modules["isaaclab.managers"] = managers_mod

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = object
    assets_mod.RigidObject = object
    sys.modules["isaaclab.assets"] = assets_mod

    sensors_mod = types.ModuleType("isaaclab.sensors")
    for name in ("Camera", "Imu", "RayCaster", "RayCasterCamera", "TiledCamera", "ContactSensor"):
        setattr(sensors_mod, name, object)
    sys.modules["isaaclab.sensors"] = sensors_mod

    envs_mod = types.ModuleType("isaaclab.envs")
    envs_mod.__path__ = []  # mark as package so `isaaclab.envs.mdp` can resolve
    envs_mod.ManagerBasedEnv = object
    envs_mod.ManagerBasedRLEnv = object
    sys.modules["isaaclab.envs"] = envs_mod

    # pmt_tasks.mdp.__init__ does `from isaaclab.envs.mdp import *`.
    envs_mdp_mod = types.ModuleType("isaaclab.envs.mdp")
    envs_mod.mdp = envs_mdp_mod
    sys.modules["isaaclab.envs.mdp"] = envs_mdp_mod

    # pmt_tasks.mdp.commands / multi_motion_command: only the type names are
    # referenced (annotations). rewards._get_body_indexes is REAL (pure torch),
    # so do NOT stub pmt_tasks.mdp.rewards.
    cmd_mod = types.ModuleType("pmt_tasks.mdp.commands")
    cmd_mod.MotionCommand = object
    cmd_mod.MultiMotionCommand = object
    # observations.py now imports MultiMotionCommandV2 from the commands subpackage
    # (PMT reorg 2026-06-25: command stack moved to mdp/commands/).
    cmd_mod.MultiMotionCommandV2 = object
    sys.modules["pmt_tasks.mdp.commands"] = cmd_mod

    mmc_mod = types.ModuleType("pmt_tasks.mdp.multi_motion_command")
    mmc_mod.MultiMotionCommandV2 = object
    sys.modules["pmt_tasks.mdp.multi_motion_command"] = mmc_mod

    # Install a LIGHTWEIGHT pmt_tasks.mdp package so importing the consolidated
    # submodules does NOT trigger the heavy real __init__ (which imports the full
    # command stack). The submodules are then loaded by file path below.
    import os

    pmt_pkg = __import__("pmt_tasks")
    mdp_dir = os.path.join(os.path.dirname(pmt_pkg.__file__), "mdp")
    mdp_pkg = types.ModuleType("pmt_tasks.mdp")
    mdp_pkg.__path__ = [mdp_dir]
    sys.modules["pmt_tasks.mdp"] = mdp_pkg

    sys.modules["_pmt_consolidation_stubs"] = types.ModuleType("_pmt_consolidation_stubs")


@pytest.fixture(scope="module", autouse=True)
def _stubs():
    _install_stubs()
    yield


# --------------------------------------------------------------------------
# Synthetic command + env doubles.
# --------------------------------------------------------------------------
BODY_NAMES = ["pelvis", "left_foot", "right_foot"]


class _Cfg:
    body_names = BODY_NAMES


class _FakeCommand:
    """Duck-typed command with the buffers the consolidated funcs read."""

    def __init__(self, n=4, b=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.cfg = _Cfg()
        self.motion_anchor_body_index = 0
        self.anchor_pos_w = torch.randn(n, 3, generator=g)
        self.robot_anchor_pos_w = torch.randn(n, 3, generator=g)
        self.anchor_quat_w = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
        self.robot_anchor_quat_w = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
        self.body_pos_relative_w = torch.randn(n, b, 3, generator=g)
        self.robot_body_pos_w = torch.randn(n, b, 3, generator=g)
        self.robot_body_quat_w = torch.nn.functional.normalize(torch.randn(n, b, 4, generator=g), dim=-1)
        self.raw_body_pos_w_with_terrain_height = torch.randn(n, b, 3, generator=g)


class _FakeEnv:
    def __init__(self, command, num_envs=4):
        self.num_envs = num_envs
        self._command = command
        self.command_manager = types.SimpleNamespace(get_term=lambda name: command)


@pytest.fixture()
def env():
    cmd = _FakeCommand()
    return _FakeEnv(cmd, num_envs=cmd.anchor_pos_w.shape[0])


# --------------------------------------------------------------------------
# OBSERVATIONS: tracking_obs(obs_type) == the 4 old wrappers (same value+shape).
# --------------------------------------------------------------------------
def test_tracking_obs_matches_four_wrappers(env):
    from pmt_tasks.mdp import observations as obs

    pairs = [
        ("motion_anchor_pos", obs.motion_anchor_pos_b),
        ("motion_anchor_ori", obs.motion_anchor_ori_b),
        ("robot_body_pos", obs.robot_body_pos_b),
        ("robot_body_ori", obs.robot_body_ori_b),
    ]
    for obs_type, wrapper in pairs:
        param_out = obs.tracking_obs(env, "motion", obs_type)
        wrap_out = wrapper(env, "motion")
        assert param_out.shape == wrap_out.shape, obs_type
        assert torch.equal(param_out, wrap_out), obs_type


def test_tracking_obs_shapes(env):
    from pmt_tasks.mdp import observations as obs

    n = env.num_envs
    b = len(BODY_NAMES)
    assert obs.tracking_obs(env, "motion", "motion_anchor_pos").shape == (n, 3)
    assert obs.tracking_obs(env, "motion", "motion_anchor_ori").shape == (n, 6)  # 3x2
    assert obs.tracking_obs(env, "motion", "robot_body_pos").shape == (n, b * 3)
    assert obs.tracking_obs(env, "motion", "robot_body_ori").shape == (n, b * 6)


def test_tracking_obs_unknown_type_raises(env):
    from pmt_tasks.mdp import observations as obs

    with pytest.raises(ValueError):
        obs.tracking_obs(env, "motion", "bogus")


# --------------------------------------------------------------------------
# TERMINATIONS: exceeded_tracking_error(dimensions=...) == the 3 anchor funcs.
# --------------------------------------------------------------------------
def test_exceeded_tracking_error_matches_anchor_wrappers(env):
    from pmt_tasks.mdp import terminations as term

    thr = 0.5
    # xyz
    assert torch.equal(
        term.exceeded_tracking_error(env, "motion", thr, dimensions="xyz"),
        term.bad_anchor_pos(env, "motion", thr),
    )
    # z only
    assert torch.equal(
        term.exceeded_tracking_error(env, "motion", thr, dimensions="z"),
        term.bad_anchor_pos_z_only(env, "motion", thr),
    )
    # xy, terrain
    assert torch.equal(
        term.exceeded_tracking_error(env, "motion", thr, dimensions="xy", use_terrain=True),
        term.bad_raw_terrain_anchor_pos_xy(env, "motion", thr),
    )


def test_exceeded_tracking_error_matches_old_inline_logic(env):
    """Match the pre-consolidation inline math exactly (independent recompute)."""
    from pmt_tasks.mdp import terminations as term

    cmd = env._command
    thr = 0.5
    old_xyz = torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=1) > thr
    old_z = torch.abs(cmd.anchor_pos_w[:, -1] - cmd.robot_anchor_pos_w[:, -1]) > thr
    raw_xy = cmd.raw_body_pos_w_with_terrain_height[:, cmd.motion_anchor_body_index, :2]
    old_xy = torch.norm(raw_xy - cmd.robot_anchor_pos_w[:, :2], dim=1) > thr

    assert torch.equal(term.exceeded_tracking_error(env, "motion", thr, dimensions="xyz"), old_xyz)
    assert torch.equal(term.exceeded_tracking_error(env, "motion", thr, dimensions="z"), old_z)
    assert torch.equal(
        term.exceeded_tracking_error(env, "motion", thr, dimensions="xy", use_terrain=True), old_xy
    )


def test_exceeded_tracking_error_bad_dim_raises(env):
    from pmt_tasks.mdp import terminations as term

    with pytest.raises(ValueError):
        term.exceeded_tracking_error(env, "motion", 0.5, dimensions="w")


def test_bad_motion_body_pos_equals_exceeded_body_pos(env):
    from pmt_tasks.mdp import terminations as term

    thr = 0.5
    assert torch.equal(
        term.bad_motion_body_pos(env, "motion", thr),
        term.exceeded_body_pos(env, "motion", thr),
    )
    # with body_names subset too
    sub = ["pelvis", "left_foot"]
    assert torch.equal(
        term.bad_motion_body_pos(env, "motion", thr, body_names=sub),
        term.exceeded_body_pos(env, "motion", thr, body_names=sub),
    )


def test_bad_motion_body_pos_z_only_matches_old_inline(env):
    """z-only wrapper now delegates to exceeded_body_height; assert == old inline."""
    from pmt_tasks.mdp import terminations as term

    cmd = env._command
    thr = 0.5
    idx = list(range(len(BODY_NAMES)))
    old = torch.any(
        torch.abs(cmd.body_pos_relative_w[:, idx, -1] - cmd.robot_body_pos_w[:, idx, -1]) > thr,
        dim=-1,
    )
    assert torch.equal(term.bad_motion_body_pos_z_only(env, "motion", thr), old)


# --------------------------------------------------------------------------
# REWARDS: apply_reward_weight_set sets weights (and param overrides) correctly.
# --------------------------------------------------------------------------
class _Term:
    def __init__(self, weight=0.0, params=None):
        self.weight = weight
        self.params = params if params is not None else {}


class _RewardsCfg:
    def __init__(self):
        self.term_a = _Term(weight=1.0, params={"std": 0.3})
        self.term_b = _Term(weight=2.0)
        self.term_c = None  # disabled term -> must be skipped, not crash


def test_apply_reward_weight_set_scalar():
    from pmt_tasks.mdp.rewards import apply_reward_weight_set

    cfg = _RewardsCfg()
    apply_reward_weight_set(cfg, {"term_a": 5.0, "term_b": 0.0, "term_c": 9.0, "missing": 7.0})
    assert cfg.term_a.weight == 5.0
    assert cfg.term_b.weight == 0.0
    assert cfg.term_c is None  # skipped, no crash


def test_apply_reward_weight_set_mapping_param_override():
    from pmt_tasks.mdp.rewards import apply_reward_weight_set

    cfg = _RewardsCfg()
    apply_reward_weight_set(cfg, {"term_a": {"weight": 3.0, "std": 0.9}})
    assert cfg.term_a.weight == 3.0
    assert cfg.term_a.params["std"] == 0.9


def test_apply_reward_weight_set_empty_noop():
    from pmt_tasks.mdp.rewards import apply_reward_weight_set

    cfg = _RewardsCfg()
    apply_reward_weight_set(cfg, None)
    apply_reward_weight_set(cfg, {})
    assert cfg.term_a.weight == 1.0  # unchanged
