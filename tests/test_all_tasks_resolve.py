"""Phase 3 — CI completeness gate (PMT plan §6 Phase 3, §9).

The success criterion "CI loads EVERY task/*.yaml" lives here. For each
configs/task/*.yaml this asserts the PURE build_task_config path composes the
defaults list, runs the §3a derivation, resolves all ${paths.*}/${checkpoints.*}
interpolations, and validates the (algorithm, network, feature, obs_sets) tuple
against compat.py (§3b) — WITHOUT instantiating Isaac Lab.

This is wbt-safe because build_task_config imports no isaaclab/omni at module or
call time (the @configclass-emitting helpers build_env_cfg/build_agent_cfg import
isaaclab lazily and are NOT exercised here). See builder.py module docstring.

It also pins the two consistency invariants the plan calls out:
  * registry.assert_compat_consistency() passes (registry compat tables match SPECS).
  * the only network with a real backing class but no @register_network is `diffusion`
    (KNOWN_PENDING is now empty; all networks including diffusion are registered).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from motion_tracking_rl import compat, registry
from pmt_tasks import builder
from pmt_tasks.builder import build_task_config

CONFIGS_TASK_DIR = Path(builder.__file__).resolve().parent.parent / "configs" / "task"

# Networks with a real backing class that are intentionally not yet
# @register_network'd (their config yamls carry an explicit compat_name override).
# Plan §9c/§9d: this set is down to exactly {diffusion} (no FPO task wired).
KNOWN_PENDING = set()


def _all_task_stems() -> list[str]:
    return sorted(p.stem for p in CONFIGS_TASK_DIR.glob("*.yaml"))


def test_task_dir_is_discoverable():
    """Sanity: there is a non-trivial set of task yamls to load."""
    stems = _all_task_stems()
    assert len(stems) >= 15, f"expected the full wired task set, found {stems}"


def test_builder_does_not_require_isaac_runtime():
    """The pure build path must not require the Isaac *runtime* (omni/isaacsim).

    Guards the wbt-safety claim. We check two things robustly (not via global
    sys.modules, which sibling tests pollute and which legitimately holds pure
    isaaclab.utils.* submodules in the wbt env):
      1. neither builder.py nor build_task_config's own source imports omni/isaacsim;
      2. build_task_config's source does not statically import isaaclab at all —
         the @configclass-emitting helpers defer that to call time (lazy).
    """
    builder_src = inspect.getsource(builder)
    for runtime in ("import omni", "from omni", "import isaacsim", "from isaacsim"):
        assert runtime not in builder_src, f"builder.py statically imports {runtime!r}"

    # build_task_config itself (the pure path) must not statically import isaaclab.
    fn_src = inspect.getsource(builder.build_task_config)
    assert "import isaaclab" not in fn_src and "from isaaclab" not in fn_src


@pytest.mark.parametrize("stem", _all_task_stems())
def test_every_task_yaml_resolves(stem):
    """build_task_config composes + derives + validates every task yaml (pure path)."""
    cfg = build_task_config(stem)

    # the derivation must have attached the coupled fields (§3a) ...
    assert "obs_groups" in cfg and len(cfg["obs_groups"]) >= 1
    assert "reward_weights" in cfg
    # ... and validate must have produced a runner (§3b).
    runner = cfg["_derived"]["runner"]
    assert runner in {"on_policy", "distillation"}
    # the resolved runner is echoed onto the runner axis.
    assert cfg["runner"]["name"] == runner
    # every task selects an algorithm + network and they validated together.
    assert cfg["algorithm"]["name"]
    assert cfg["network"]["name"]


def test_compat_consistency_holds():
    """registry.assert_compat_consistency() passes (plan §3b/§4)."""
    registry.autoload()
    tables = registry.assert_compat_consistency()
    # every registered algorithm compat name is a SPECS key.
    assert set(tables["ALGORITHM_COMPAT"].values()) <= set(compat.SPECS)


def test_known_pending_is_exactly_diffusion():
    """All networks are now registered; KNOWN_PENDING is empty (§9c/§9d).

    Derive the pending set from compat: networks referenced by some algorithm spec
    minus networks the registry actually maps. It must be empty now.
    """
    registry.autoload()
    referenced = set().union(*(s.compatible_networks for s in compat.SPECS.values()))
    registered = set(registry.NETWORK_COMPAT.values())
    pending = referenced - registered
    assert pending == KNOWN_PENDING, (
        f"KNOWN_PENDING drifted: expected {KNOWN_PENDING}, got {pending}. "
        f"(referenced={sorted(referenced)}, registered={sorted(registered)})"
    )
