"""Pure-wbt tests for Phase 2.2: SONIC family + distillation runner path (plan §6).

PART A (SONIC): the sonic_multimotion_flat task composes, fully resolves, derives the
on_policy runner, and selects the SonicActorCritic network + PPO(+aux losses).
PART B (distill): the distill_stepping_stone task composes, fully resolves, derives
the DISTILLATION runner (NOT on_policy), exposes the teacher+policy obs sets, and
compat(distillation, student_teacher) succeeds. Also asserts VisionStudentTeacher
is now registered so assert_compat_consistency passes (KNOWN_PENDING cleared).
"""
from omegaconf import OmegaConf

import pytest

from motion_tracking_rl import compat, registry
from pmt_tasks.builder import build_task_config


def _resolved(cfg):
    s = OmegaConf.to_yaml(cfg, resolve=True)
    assert "${" not in s, f"unresolved interpolation left:\n{s}"


# --- PART A: SONIC ----------------------------------------------------------

def test_sonic_task_builds_on_policy(monkeypatch):
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("sonic_multimotion_flat")
    _resolved(cfg)
    assert cfg.network.name == "SonicActorCritic"
    assert cfg.algorithm.name == "PPO"
    assert cfg.runner.name == "on_policy"
    # SonicActorCritic reads encoder groups from the full obs dict; obs_groups is
    # still {policy, critic}.
    assert set(cfg.obs_groups.keys()) == {"policy", "critic"}


def test_sonic_aux_coefficients_and_paired_human(monkeypatch):
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("sonic_multimotion_flat")
    # SONIC auxiliary losses use the official global scale and per-term weights.
    assert float(cfg.algorithm.aux_loss_scale) == 1.0
    assert float(cfg.algorithm.aux_loss_coef["g1_recon"]) == 0.01
    assert float(cfg.algorithm.aux_loss_coef["g1_smpl_latent"]) == 1.0
    assert float(cfg.algorithm.aux_loss_coef["reencoded_smpl_g1_latent"]) == 1.0
    # GATE: scratch encoders — no external ONNX required.
    assert cfg.network.load_pretrained_robot_encoder is False
    assert cfg.network.load_pretrained_control_decoder is False
    assert cfg.network.pretrained_encoder_onnx_path is None
    assert int(cfg.network.latent_dim) == 64
    assert int(cfg.network.num_fsq_levels) == 32
    assert int(cfg.network.max_num_tokens) == 2
    assert cfg.network.activation == "silu"
    # paired human motion is requested via the motion axis.
    assert cfg.motion.load_human_motion is True
    assert cfg.network.robot_motion_dim == 580
    assert cfg.network.human_motion_dim == 660


def test_sonic_compat_ppo_sonic_ok():
    runner = compat.validate("ppo", "sonic", obs_sets={"policy", "critic"})
    assert runner == "on_policy"


# --- PART B: distillation runner path ---------------------------------------

def test_distill_task_derives_distillation_runner(monkeypatch):
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("distill_stepping_stone")
    _resolved(cfg)
    assert cfg.network.name == "StudentTeacher"
    assert cfg.algorithm.name == "Distillation"
    # The KEY assertion: runner derives to DISTILLATION (not on_policy).
    assert cfg.runner.name == "distillation"
    assert cfg._derived.runner == "distillation"
    # teacher + policy obs sets present (distillation required_obs_sets).
    assert "teacher" in cfg.obs_groups
    assert "policy" in cfg.obs_groups


def test_distill_compat_distillation_student_teacher_ok():
    runner = compat.validate(
        "distillation", "student_teacher", obs_sets={"policy", "teacher"}
    )
    assert runner == "distillation"


def test_distill_runner_class_is_distillation_runner():
    """The agent cfg's class_name resolves to DistillationRunner via the registry
    (this is what scripts/train.py dispatches on)."""
    registry.autoload()
    assert "DistillationRunner" in registry.RUNNERS
    assert "distillation" in registry.RUNNERS
    assert registry.get_runner("DistillationRunner") is registry.get_runner("distillation")


def test_vision_student_teacher_registered_and_consistent():
    """Phase 2.2 cleared the VisionStudentTeacher KNOWN_PENDING: it is now
    @register_network'd and assert_compat_consistency passes."""
    registry.autoload()
    assert "VisionStudentTeacher" in registry.NETWORKS
    assert registry.network_compat_name("VisionStudentTeacher") == "vision_student_latent_anchor"
    # consistency must not raise (would fail if compat tables drifted).
    registry.assert_compat_consistency()
