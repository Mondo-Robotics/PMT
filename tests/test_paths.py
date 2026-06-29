"""Phase-0 path-resolution gate (PMT plan §5, §6).

Asserts PMT_PROFILE=local vs cluster select different roots, and that the
${paths.*} interpolations RESOLVE to concrete strings (the v1 ${${}} form did not).
"""
import os

from omegaconf import OmegaConf

from pmt_tasks.builder import load_paths

_PATH_ENV_VARS = (
    "PMT_DATA_ROOT",
    "PMT_MOTION_ROOT",
    "PMT_CKPT_ROOT",
    "PMT_DATASET_ROOT",
    "PMT_TERRAIN_MOTION_ROOT",
    "PMT_SONIC_ROOT",
    "PMT_MULTIMOTION_FLAT_MOTION",
    "PMT_BACKFLIP_MOTION",
)


def _clear_path_env(monkeypatch):
    for name in _PATH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _home_path(*parts):
    return os.path.join(os.environ["HOME"], *parts)


def _assert_resolved(block):
    """No unresolved ${...} left in any value."""
    container = OmegaConf.to_container(block, resolve=True)
    for k, v in container.items():
        assert isinstance(v, str), f"{k} not a string: {v!r}"
        assert "${" not in v, f"{k} left unresolved: {v!r}"


def test_local_profile_default(monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.delenv("PMT_PROFILE", raising=False)
    p = load_paths()  # no env, no arg -> local
    assert p.DATA_ROOT == _home_path("whole_body_tracking")
    assert p.MOTION_ROOT == _home_path("whole_body_tracking", "motions")
    assert p.CKPT_ROOT == _home_path("whole_body_tracking", "logs", "rsl_rl")
    _assert_resolved(p)


def test_cluster_profile_env(monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("PMT_PROFILE", "cluster")
    p = load_paths()
    assert p.DATA_ROOT == _home_path("pmt_cluster_data", "whole_body_tracking")
    assert p.MOTION_ROOT == _home_path("pmt_cluster_data", "motions")
    assert p.TERRAIN_ROOT == _home_path("pmt_cluster_data", "whole_body_tracking")
    assert p.SONIC_ROOT == _home_path("pmt_cluster_data", "sonic")
    assert p.TERRAIN_MOTION_ROOT == _home_path("pmt_cluster_data", "motions")
    _assert_resolved(p)


def test_profile_arg_overrides_env(monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("PMT_PROFILE", "cluster")
    p = load_paths(profile="local")  # explicit arg wins
    assert p.DATA_ROOT == _home_path("whole_body_tracking")


def test_local_and_cluster_differ(monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.delenv("PMT_PROFILE", raising=False)
    local = load_paths("local")
    cluster = load_paths("cluster")
    assert local.DATA_ROOT != cluster.DATA_ROOT
    assert local.MOTION_ROOT != cluster.MOTION_ROOT
    assert local.CKPT_ROOT != cluster.CKPT_ROOT


def test_path_env_overrides(monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.delenv("PMT_PROFILE", raising=False)
    monkeypatch.setenv("PMT_DATA_ROOT", "/tmp/pmt_data")
    monkeypatch.setenv("PMT_SONIC_ROOT", "/tmp/pmt_sonic")
    p = load_paths("local")
    assert p.DATA_ROOT == "/tmp/pmt_data"
    assert p.MOTION_ROOT == "/tmp/pmt_data/motions"
    assert p.SONIC_ROOT == "/tmp/pmt_sonic"
    _assert_resolved(p)


def test_unknown_profile_raises():
    try:
        load_paths("does_not_exist")
    except ValueError as e:
        assert "unknown PMT profile" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown profile")
