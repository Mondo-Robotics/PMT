"""Pure-wbt tests for the Phase 2.1 MultiMotion/Flat family (plan §6 Phase 2.1).

Covers the 5 tasks: base / uniform / adaptive / streaming / bpo. Asserts the builder
resolves each task yaml, the runner derives to on_policy, the sampler/storage flags
flow from the motion axis, the streaming yaml sets storage_mode=streaming, and the
bpo task passes compat + derives the on_policy runner.
"""
from omegaconf import OmegaConf

import pytest

from pmt_tasks.builder import build_task_config, load_paths

_FAMILY = [
    "multimotionv2_flat",
    "multimotionv2_uniform_flat",
    "multimotionv2_adaptive_flat",
    "multimotionv2_streaming_flat",
    "bpo_multimotionv2_flat",
]


def _resolved(cfg):
    s = OmegaConf.to_yaml(cfg, resolve=True)
    assert "${" not in s, f"unresolved interpolation left:\n{s}"


@pytest.mark.parametrize("task", _FAMILY)
def test_family_builds_on_policy(monkeypatch, task):
    """Each of the 5 family yamls composes, fully resolves, and derives on_policy
    with the standard {policy,critic} obs groups (MLP actor-critic)."""
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config(task)
    _resolved(cfg)
    assert cfg.runner.name == "on_policy"
    assert set(cfg.obs_groups.keys()) == {"policy", "critic"}
    # motion path is profile-driven (${paths.MOTION_ROOT}/...)
    assert cfg.motion.motion_files == load_paths("local").MULTIMOTION_FLAT_MOTION


@pytest.mark.parametrize(
    "task,expected_sampler",
    [
        ("multimotionv2_flat", "bin_adaptive"),
        ("multimotionv2_uniform_flat", "uniform"),
        ("multimotionv2_adaptive_flat", "adaptive"),
    ],
)
def test_sampler_is_motion_axis_param(monkeypatch, task, expected_sampler):
    """Uniform/adaptive/bin_adaptive are a MOTION-axis param, not separate env classes."""
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config(task)
    assert cfg.motion.sampler == expected_sampler
    assert cfg.motion.storage_mode == "eager"


def test_streaming_sets_storage_mode(monkeypatch):
    """The streaming task's motion yaml flips storage_mode=streaming (plan §9b)."""
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("multimotionv2_streaming_flat")
    assert cfg.motion.storage_mode == "streaming"
    assert cfg.motion.command_class == "StreamingMultiMotionCommand"
    # streaming-only knobs are present for the env builder to consume
    assert "max_working_set" in cfg.motion


def test_bpo_runner_on_policy_and_compat_ok(monkeypatch):
    """BPO task: algorithm swap only — derives on_policy + compat(bpo, mlp) passes."""
    from motion_tracking_rl import compat

    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("bpo_multimotionv2_flat")
    assert cfg.algorithm.name == "BPO"
    assert cfg.runner.name == "on_policy"
    # compat.validate(bpo, mlp) must succeed and report the on_policy runner
    runner = compat.validate("bpo", "mlp", obs_sets={"policy", "critic"})
    assert runner == "on_policy"
