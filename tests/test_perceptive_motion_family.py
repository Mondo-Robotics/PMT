"""Phase 2.5 — PerceptiveMotion family + ablations (PMT plan §6 Phase 2.5, §3b).

Pure (wbt-env) tests, no isaaclab/omni:
  1. The 2017-line monolith split is checkpoint/import safe: the OLD module path
     re-exports the SAME class objects as the new subpackage (is-identity).
  2. The 3 PM networks + VisionAblation are @register_network'd and
     assert_compat_consistency passes (KNOWN_PENDING reduced to diffusion only).
  3. The PerceptiveMotion token-tracker task composes, exposes the
     future_motion_window + height_scan obs groups, and compat-validates.
  4. Ablations are config field-flips: building the base ablation task with each
     network override yields the correct flipped field (Type I architecture +
     Type II VisionTransformer toggles) AND a --multirun dotlist flips it too.
"""
from __future__ import annotations

import pytest

from motion_tracking_rl import registry
from pmt_tasks.builder import build_task_config


@pytest.fixture(scope="module", autouse=True)
def _autoload():
    registry.autoload()


# --- 1. monolith split is checkpoint/import safe (same class objects) ---------

def test_monolith_split_same_class_objects():
    import motion_tracking_rl.networks.perceptive_motion as new
    import motion_tracking_rl.networks.perceptive_motion_adapter_tracker as old

    for name in (
        "PerceptiveMotionAdapter",
        "PerceptiveMotionAdapterTracker",
        "PerceptiveMotionTokenTracker",
        "PerceptiveMotionTracker",
        "PercaptiveMotionTracker",
    ):
        assert getattr(old, name) is getattr(new, name), f"{name}: old IS NOT new"

    # top-level networks package re-exports the same objects too
    from motion_tracking_rl import networks
    assert networks.PerceptiveMotionTokenTracker is new.PerceptiveMotionTokenTracker
    assert networks.PerceptiveMotionAdapterTracker is new.PerceptiveMotionAdapterTracker


def test_misspelled_alias_is_transformer():
    import motion_tracking_rl.networks.perceptive_motion as new
    from motion_tracking_rl.networks.transformer_actor_critic import TransformerActorCritic

    assert new.PercaptiveMotionTracker is TransformerActorCritic
    assert new.PerceptiveMotionTracker is TransformerActorCritic


# --- 2. networks registered + compat consistent -------------------------------

def test_pm_networks_registered():
    for cls_name, compat_name in (
        ("PerceptiveMotionAdapter", "perceptive_motion_adapter"),
        ("PerceptiveMotionAdapterTracker", "perceptive_motion_adapter_tracker"),
        ("PerceptiveMotionTokenTracker", "perceptive_motion_token_tracker"),
        ("VisionAblationActorCritic", "vision_ablation"),
    ):
        assert cls_name in registry.NETWORKS, f"{cls_name} not registered"
        assert registry.NETWORK_COMPAT[cls_name] == compat_name


def test_compat_consistency_passes():
    # assert_compat_consistency raises if registry compat tables drift from SPECS.
    registry.assert_compat_consistency()


def test_compat_specs_accept_pm_networks():
    from motion_tracking_rl import compat

    assert "perceptive_motion_token_tracker" in compat.SPECS["ppo"].compatible_networks
    assert "vision_ablation" in compat.SPECS["ppo"].compatible_networks
    assert "perceptive_motion_adapter_tracker" in compat.SPECS["distillation"].compatible_networks
    assert "perceptive_motion_token_tracker" in compat.SPECS["distillation"].compatible_networks


# --- 3. PMT token-tracker task composes + obs groups present ------------------

def test_pmt_task_composes_with_required_obs_groups():
    cfg = build_task_config("perceptive_motion_token_tracker")
    obs_groups = dict(cfg["obs_groups"])
    assert "future_motion_window" in obs_groups
    assert "height_scan" in obs_groups
    assert list(obs_groups["future_motion_window"]) == [
        "command_window",
        "motion_anchor_delta_window",
    ]
    assert list(obs_groups["height_scan"]) == ["vision"]
    # compat-valid (PPO on_policy) + correct network selected
    assert cfg["_derived"]["runner"] == "on_policy"
    assert cfg["network"]["name"] == "PerceptiveMotionTokenTracker"
    # from-scratch gate knobs (no pretrained ckpt blocker)
    assert cfg["network"]["require_pmt_checkpoint"] is False
    assert cfg["network"]["pmt_only_mode"] is True


# --- 4. ablations are config field-flips (the thesis) -------------------------

_ABLATION_BASE_DEFAULTS = [
    {"robot": "g1"}, {"terrain": "stepping_stone"}, {"motion": "multi"},
    {"scene": "none"}, {"sensor": "none"}, {"obs": "perceptive_motion_token"},
    {"reward": "deepmimic_anchor"}, {"algorithm": "ppo"}, {"stage": "scratch"},
]


def _compose_with_network(network_choice: str):
    defaults = list(_ABLATION_BASE_DEFAULTS) + [{"network": network_choice}]
    return build_task_config({"defaults": defaults, "experiment_name": "ablation"})


@pytest.mark.parametrize(
    "network_choice,field,expected",
    [
        ("vision_transformer", "use_action_residual", True),
        ("vision_transformer", "use_identity_gates", True),
        ("vision_transformer", "use_map_proprio_cross_attention", False),
        ("vision_transformer_no_residual", "use_action_residual", False),
        ("vision_transformer_no_identity_gate", "use_identity_gates", False),
        ("vision_transformer_map_proprio_cross", "use_map_proprio_cross_attention", True),
    ],
)
def test_type_ii_visiontransformer_field_flips(network_choice, field, expected):
    cfg = _compose_with_network(network_choice)
    assert cfg["network"][field] is expected
    # composition still compat-validates (PPO accepts vision_transformer)
    assert cfg["_derived"]["runner"] == "on_policy"


@pytest.mark.parametrize(
    "network_choice,expected_arch",
    [
        ("vision_ablation", "flat_mlp"),
        ("vision_ablation_split_mlp", "split_mlp"),
        ("vision_ablation_split_cnn", "split_cnn"),
    ],
)
def test_type_i_architecture_field_flips(network_choice, expected_arch):
    cfg = _compose_with_network(network_choice)
    assert cfg["network"]["architecture"] == expected_arch
    assert cfg["network"]["name"] == "VisionAblationActorCritic"
    assert cfg["_derived"]["runner"] == "on_policy"


def test_gru_ablation_is_class_selection_not_field_flip():
    """GRU is a CLASS SELECTION (different registered class_name), not an
    `architecture` field flip. It still composes + compat-validates."""
    cfg = _compose_with_network("vision_ablation_gru")
    assert cfg["network"]["name"] == "VisionAblationRecurrentActorCritic"
    assert cfg["network"]["rnn_type"] == "gru"
    assert cfg["_derived"]["runner"] == "on_policy"


def test_multirun_dotlist_flips_fields():
    """A --multirun-style dotlist override flips the field with NO variant yaml."""
    cfg = build_task_config(
        "vision_ablation_base",
        overrides=["network.use_identity_gates=false", "network.use_action_residual=false"],
    )
    assert cfg["network"]["use_identity_gates"] is False
    assert cfg["network"]["use_action_residual"] is False
