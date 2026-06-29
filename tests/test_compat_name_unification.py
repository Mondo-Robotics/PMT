"""Unify the `name` vs `compat_name` two-namespace seam (PMT prereq #7, plan §3b/§4).

Before: registry keyed on CLASS names ("PPO", "TransformerActorCritic"); compat.SPECS
keyed on AXIS names ("ppo", "transformer"); the YAMLs bridged them with TWO free-text
fields (`name` + `compat_name`) that nothing validated -> they could silently drift.

After: the registry decorators carry the compat axis name as the SINGLE source of truth
(registry.ALGORITHM_COMPAT / NETWORK_COMPAT). The builder derives compat_name from the
YAML `name` via that mapping; an explicit YAML `compat_name` is only a fallback for
not-yet-registered classes; an unknown name with no fallback FAILS LOUD.

These tests exercise registry + compat + builder.compat-name-resolution ONLY (no
isaaclab/omni). They do NOT touch derive.py (a concurrent agent owns that).
"""
from __future__ import annotations

import pytest

from motion_tracking_rl import compat, registry
from pmt_tasks.builder import _resolve_compat_name, build_task_config


@pytest.fixture(scope="module", autouse=True)
def _autoload():
    registry.autoload()


# ---------------------------------------------------------------------------
# 1. The registry compat tables are consistent with compat.SPECS (no drift).
# ---------------------------------------------------------------------------
def test_registry_compat_tables_consistent_with_specs():
    """assert_compat_consistency must pass: every registered algorithm compat_name is a
    compat.SPECS key, and every registered network compat_name is referenced by some
    algorithm's compatible_networks set. This is the automatic drift guard."""
    tables = registry.assert_compat_consistency()
    # the tables are non-empty (decorators ran) and contain the slice we unified
    assert tables["ALGORITHM_COMPAT"]["PPO"] == "ppo"
    assert tables["ALGORITHM_COMPAT"]["Distillation"] == "distillation"
    assert tables["NETWORK_COMPAT"]["TransformerActorCritic"] == "transformer"
    assert tables["NETWORK_COMPAT"]["ActorCritic"] == "mlp"


def test_every_registered_algorithm_compat_is_a_specs_key():
    for cls_name, axis in registry.ALGORITHM_COMPAT.items():
        assert axis in compat.SPECS, (
            f"algorithm '{cls_name}' compat_name '{axis}' is not a compat.SPECS key "
            f"{sorted(compat.SPECS)}"
        )


def test_every_registered_network_compat_is_referenced_by_some_spec():
    referenced = set().union(*(s.compatible_networks for s in compat.SPECS.values()))
    for cls_name, axis in registry.NETWORK_COMPAT.items():
        assert axis in referenced, (
            f"network '{cls_name}' compat_name '{axis}' is referenced by no algorithm's "
            f"compatible_networks {sorted(referenced)} -> orphan"
        )


def test_assert_compat_consistency_fails_loud_on_injected_drift(monkeypatch):
    """If a bogus algorithm compat_name (not in SPECS) is injected, the consistency check
    must raise a clear ValueError naming the offender."""
    monkeypatch.setitem(registry.ALGORITHM_COMPAT, "BogusAlg", "no_such_axis")
    with pytest.raises(ValueError) as exc:
        registry.assert_compat_consistency()
    assert "no_such_axis" in str(exc.value) and "compat.SPECS" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. The builder resolves compat_name from `name` ALONE for the registered slice.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind,name,expected",
    [
        ("algorithm", "PPO", "ppo"),
        ("algorithm", "FPOPlus", "fpo_plus"),
        ("algorithm", "Distillation", "distillation"),
        ("network", "TransformerActorCritic", "transformer"),
        ("network", "VisionTransformerActorCritic", "vision_transformer"),
        ("network", "ActorCritic", "mlp"),
        ("network", "StudentTeacher", "student_teacher"),
        ("network", "SonicActorCritic", "sonic"),
    ],
)
def test_resolve_compat_from_name_only(kind, name, expected):
    """A YAML block carrying ONLY `name` (no compat_name) resolves via the registry."""
    table = registry.ALGORITHM_COMPAT if kind == "algorithm" else registry.NETWORK_COMPAT
    resolved = _resolve_compat_name(kind=kind, cfg_block={"name": name}, registry_table=table)
    assert resolved == expected


def test_registered_yamls_have_no_compat_name_field():
    """The redundant `compat_name` field is removed from the YAMLs whose class IS
    registered (so the registry is the only link). Only KNOWN_PENDING classes keep it."""
    import pathlib

    root = pathlib.Path(registry.__file__).resolve().parent.parent / "configs"
    for sub in ("algorithm", "network"):
        for f in (root / sub).glob("*.yaml"):
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(f)
            name = cfg.get("name")
            has_compat = "compat_name" in cfg
            registered = name in registry.ALGORITHM_COMPAT or name in registry.NETWORK_COMPAT
            if registered:
                assert not has_compat, (
                    f"{f.name}: class '{name}' is registered; remove redundant compat_name"
                )
            else:
                # not-yet-registered (KNOWN_PENDING) MUST keep an explicit override
                assert has_compat, (
                    f"{f.name}: class '{name}' is NOT registered; it must keep an explicit "
                    f"compat_name override until decorated"
                )


# ---------------------------------------------------------------------------
# 3. Fail-loud paths.
# ---------------------------------------------------------------------------
def test_unregistered_name_without_override_fails_loud():
    """A name that is neither in the registry table nor accompanied by a YAML
    compat_name override raises a clear ValueError."""
    with pytest.raises(ValueError) as exc:
        _resolve_compat_name(
            kind="network",
            cfg_block={"name": "TotallyUnknownNet"},
            registry_table=registry.NETWORK_COMPAT,
        )
    msg = str(exc.value)
    assert "TotallyUnknownNet" in msg
    assert "no compat axis" in msg
    assert "register_network" in msg  # tells the user how to fix


def test_unregistered_name_with_override_uses_override():
    """A not-yet-registered class falls back to its explicit YAML compat_name."""
    resolved = _resolve_compat_name(
        kind="network",
        cfg_block={"name": "VisionStudentTeacher", "compat_name": "vision_student_latent_anchor"},
        registry_table=registry.NETWORK_COMPAT,
    )
    assert resolved == "vision_student_latent_anchor"


def test_override_disagreeing_with_registry_fails_loud():
    """If a YAML provides a compat_name that contradicts the registry mapping, the
    builder must raise (drift guard at the per-task level)."""
    with pytest.raises(ValueError) as exc:
        _resolve_compat_name(
            kind="network",
            cfg_block={"name": "TransformerActorCritic", "compat_name": "wrong_axis"},
            registry_table=registry.NETWORK_COMPAT,
        )
    assert "disagrees with the registry mapping" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. End-to-end: the slice task YAMLs (with compat_name removed) still build.
# ---------------------------------------------------------------------------
def test_slice_tasks_resolve_compat_from_registry(monkeypatch):
    """pmt_stepping_stone (transformer/ppo) and ppofinetune (vision_transformer/ppo)
    carry NO compat_name in their network/algorithm YAMLs yet still resolve the runner.
    (The distill task uses the KNOWN_PENDING vision_student network whose YAML keeps an
    explicit override; covered by test_builder_slice.py.)"""
    monkeypatch.setenv("PMT_PROFILE", "local")
    cfg = build_task_config("pmt_stepping_stone")
    assert cfg.runner.name == "on_policy"
    # the composed network/algorithm blocks carry no compat_name field
    assert "compat_name" not in cfg.network
    assert "compat_name" not in cfg.algorithm
