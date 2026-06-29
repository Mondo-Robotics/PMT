"""Pure-wbt tests for the vision transformer latent-anchor distillation task (plan §6 / §10).

The latent-anchor distill task wires a BLIND TransformerActorCritic teacher into a
vision-augmented transformer student (VisionStudentTeacher) via the DistillationRunner.
These tests exercise the pure builder path (no isaaclab): the task composes, fully
resolves, derives the distillation runner, exposes the teacher+policy obs sets, and
the named teacher checkpoint resolves to a concrete ``model_*.pt`` path.
"""
import re

from omegaconf import OmegaConf

from motion_tracking_rl import compat, registry
from pmt_tasks.builder import build_task_config


def _resolved(cfg):
    s = OmegaConf.to_yaml(cfg, resolve=True)
    assert "${" not in s, f"unresolved interpolation left:\n{s}"


def test_latent_anchor_task_derives_distillation_runner(monkeypatch):
    monkeypatch.setenv("PMT_PROFILE", "cluster")
    cfg = build_task_config("distill_stepping_stone_latent_anchor")
    _resolved(cfg)
    assert cfg.network.name == "VisionStudentTeacher"
    assert cfg.algorithm.name == "Distillation"
    # KEY assertion: runner derives to DISTILLATION (not on_policy).
    assert cfg.runner.name == "distillation"
    assert cfg._derived.runner == "distillation"
    # distillation required obs sets present.
    assert "teacher" in cfg.obs_groups
    assert "policy" in cfg.obs_groups


def test_latent_anchor_teacher_ckpt_resolves_to_model_pt(monkeypatch):
    """The named ckpt ${checkpoints.ss_teacher} must resolve to a concrete
    model_*.pt path the VisionStudentTeacher wrapper can load (not a dangling
    interpolation or a bare run dir)."""
    monkeypatch.setenv("PMT_PROFILE", "cluster")
    cfg = build_task_config("distill_stepping_stone_latent_anchor")
    teacher_ckpt = cfg.network.teacher_ckpt
    assert teacher_ckpt is not None
    assert "${" not in str(teacher_ckpt)
    assert re.search(r"model_\d+\.pt$", str(teacher_ckpt)), (
        f"teacher_ckpt did not resolve to a model_*.pt file: {teacher_ckpt}"
    )
    # Resolves under the PMT logs CKPT_ROOT (cluster profile).
    assert "/PMT/logs/rsl_rl/" in str(teacher_ckpt)


def test_latent_anchor_compat_vision_student_ok():
    """compat(distillation, vision_student_latent_anchor) is valid and yields the
    distillation runner."""
    runner = compat.validate(
        "distillation", "vision_student_latent_anchor", obs_sets={"policy", "teacher"}
    )
    assert runner == "distillation"


def test_latent_anchor_network_registered():
    registry.autoload()
    assert "VisionStudentTeacher" in registry.NETWORKS
    assert (
        registry.network_compat_name("VisionStudentTeacher")
        == "vision_student_latent_anchor"
    )


def test_latent_anchor_paired_motion_paths(monkeypatch):
    """The env requires paired optimized/raw clips: the task must surface both
    motion_files (optimized -> teacher cmd) and raw_motion_files (raw -> student cmd)."""
    monkeypatch.setenv("PMT_PROFILE", "cluster")
    cfg = build_task_config("distill_stepping_stone_latent_anchor")
    assert str(cfg.motion.motion_files).endswith("/optimized")
    assert str(cfg.motion.raw_motion_files).endswith("/raw")
