"""Builder end-to-end on the 3 slice tasks + an invalid combo (PMT plan §3, §3b)."""
import os

import pytest
from omegaconf import OmegaConf

from pmt_tasks.builder import build_task_config

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


def _resolved(cfg):
    """Assert no unresolved ${...} remain anywhere in the config."""
    s = OmegaConf.to_yaml(cfg, resolve=True)
    assert "${" not in s, f"unresolved interpolation left:\n{s}"


def test_pmt_stepping_stone_builds_on_policy(monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("pmt_stepping_stone")
    _resolved(cfg)
    assert cfg.runner.name == "on_policy"
    assert set(cfg.obs_groups.keys()) >= {"policy", "critic"}
    assert "teacher" not in cfg.obs_groups
    # ${paths.*} resolved into the terrain mesh path
    assert cfg.terrain.mesh_path.startswith(_home_path("whole_body_tracking"))


def test_distill_derives_distillation_runner_and_teacher(monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("distill_stepping_stone_latent_anchor")
    _resolved(cfg)
    assert cfg.runner.name == "distillation"
    # GROUNDED: distillation_runner.py:61 default_sets=["teacher"] — teacher ONLY,
    # not teacher+student (the old invented derive.py semantics added "student").
    assert "teacher" in set(cfg.obs_groups.keys())
    assert "student" not in set(cfg.obs_groups.keys())
    # named checkpoint resolved into a concrete path string
    assert cfg.network.teacher_ckpt.startswith(_home_path("whole_body_tracking"))
    # fair reward weight-set selected; anchor obs dropped for the student
    assert cfg._derived.anchor_in_actor_obs is False


def test_ppofinetune_derives_on_policy(monkeypatch):
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("ppofinetune_vision_teacher_stepping_stone_latent_anchor")
    _resolved(cfg)
    assert cfg.runner.name == "on_policy"
    # teacher obs present (vision teacher keeps anchor)
    assert cfg._derived.anchor_in_actor_obs is True


def test_add_derives_discriminator_obs_sets(monkeypatch):
    # Phase-1 widened slice: ADD (add_ppo) must emit the discriminator obs sets so
    # compat.validate's required_obs_sets check passes (plan §3b, §9b).
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("add_multimotion_flat")
    _resolved(cfg)
    assert cfg.runner.name == "on_policy"
    keys = set(cfg.obs_groups.keys())
    assert {"policy", "critic", "add_disc_obs", "add_disc_demo"} <= keys, keys
    # ADD flat clips run at the terrain-flat default rate (§3a dt-from-motion).
    assert cfg.motion.decimation == 4
    assert cfg.motion.sim_dt == 0.005


def test_sonic_agent_cfg_modes_resolve_release_onnx(monkeypatch, tmp_path):
    pytest.importorskip("isaaclab")
    from pmt_tasks.builder import build_agent_cfg

    onnx_dir = tmp_path / "sonic_release"
    onnx_dir.mkdir()
    encoder = onnx_dir / "model_encoder.onnx"
    decoder = onnx_dir / "model_decoder.onnx"
    encoder.write_bytes(b"placeholder")
    decoder.write_bytes(b"placeholder")
    monkeypatch.setenv("PMT_PROFILE", "local")
    monkeypatch.setenv("PMT_SONIC_ONNX_DIR", str(onnx_dir))

    scratch_cfg = build_agent_cfg(
        "sonic_multimotion_flat",
        profile="local",
        overrides=["sonic_mode=scratch"],
    )
    assert scratch_cfg.policy.class_name == "SonicActorCritic"
    assert scratch_cfg.policy.pretrained_encoder_onnx_path is None
    assert scratch_cfg.policy.pretrained_decoder_onnx_path is None
    assert scratch_cfg.policy.robot_motion_dim == 580
    assert scratch_cfg.policy.human_motion_dim == 660
    assert scratch_cfg.obs_groups == {"policy": ["policy"], "critic": ["critic"]}

    expected = {
        "finetune_all": {
            "detach_action_token": False,
            "freeze_robot_encoder": False,
            "freeze_control_decoder": False,
            "freeze_motion_decoder": False,
            "freeze_critic": False,
            "freeze_action_std": False,
            "aux_train_robot_encoder": True,
        },
        "finetune_decoder": {
            "detach_action_token": True,
            "freeze_robot_encoder": True,
            "freeze_control_decoder": False,
            "freeze_motion_decoder": False,
            "freeze_critic": False,
            "freeze_action_std": False,
            "aux_train_robot_encoder": False,
        },
        "play": {
            "detach_action_token": True,
            "freeze_robot_encoder": True,
            "freeze_control_decoder": True,
            "freeze_motion_decoder": True,
            "freeze_critic": True,
            "freeze_action_std": True,
            "aux_train_robot_encoder": False,
        },
    }
    for mode, fields in expected.items():
        cfg = build_agent_cfg(
            "sonic_multimotion_flat",
            profile="local",
            overrides=[f"sonic_mode={mode}"],
        )
        policy = cfg.policy
        assert policy.class_name == "OfficialSonicActorCritic"
        assert policy.training_mode == mode
        assert policy.pretrained_encoder_onnx_path == str(encoder)
        assert policy.pretrained_decoder_onnx_path == str(decoder)
        assert policy.load_pretrained_robot_encoder is True
        assert policy.load_pretrained_control_decoder is True
        assert policy.load_pretrained_human_encoder is False
        assert policy.load_pretrained_hybrid_encoder is False
        assert policy.strict_pretrained_shapes is True
        assert policy.robot_motion_dim == 640
        assert policy.latent_dim == 64
        assert policy.num_fsq_levels == 32
        assert policy.fsq_level_list == 32
        assert policy.max_num_tokens == 2
        assert policy.action_encoder_source == "mode"
        assert policy.encoder_mode_key == "encoder_mode_4"
        assert policy.decoder_proprio_layout == "interleaved_step_history"
        assert policy.control_decoder_input_order == "token_proprio"
        assert policy.robot_encoder_layout == "g1_onnx_repack"
        assert set(cfg.obs_groups) >= {"policy", "critic", "robot_encoder", "encoder_mode_4"}
        for name, value in fields.items():
            assert getattr(policy, name) is value


def test_backflip_decimation_dt_from_config(monkeypatch):
    # Phase-1 widened slice: the DECISIVE coupling — per-task control rate flows from
    # the motion axis config, NOT a hard-coded subclass field (plan §3a dt-from-motion).
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("backflip")
    _resolved(cfg)
    assert cfg.runner.name == "on_policy"
    # GROUNDED: old G1BackFlipMerged...EnvCfg sets dec=10/dt=0.002
    # (distill_stepping_stone_env_cfg.py:1095-1096). Here they come from config.
    assert cfg.motion.decimation == 10
    assert cfg.motion.sim_dt == 0.002
    # knee negative-power safety term present in the reward weight-set.
    assert "knee_negative_power" in dict(cfg.reward_weights)


def test_invalid_combo_raises():
    # fpo_plus requires network in {diffusion}; transformer is invalid (plan §3b).
    # Compose through an existing algorithm YAML, then override the algorithm under test.
    bad = {
        "defaults": [
            {"robot": "g1"},
            {"terrain": "stepping_stone"},
            {"motion": "multi"},
            {"sensor": "none"},
            {"obs": "transformer_hist"},
            {"reward": "deepmimic_anchor"},
            {"network": "transformer"},
            {"algorithm": "ppo"},
            {"stage": "scratch"},
        ],
        "algorithm": {"compat_name": "fpo_plus", "name": "FPOPlus"},
        "experiment_name": "bad_fpo_plus_transformer",
    }
    try:
        build_task_config(bad)
    except ValueError as e:
        assert "fpo_plus" in str(e) and "incompatible" in str(e)
    else:
        raise AssertionError("expected ValueError for fpo_plus+transformer")
