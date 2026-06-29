"""Name->class registries replacing eval()-based dispatch (PMT plan §4)."""
from __future__ import annotations

ALGORITHMS: dict[str, type] = {}
NETWORKS: dict[str, type] = {}
RUNNERS: dict[str, type] = {}

# Single source of truth linking a registered CLASS name (registry key, == old
# `class_name` / YAML `name`) to its COMPAT AXIS name (compat.SPECS key). This is
# the bridge that used to be two free-text YAML fields (`name` + `compat_name`).
# Populated by the decorators below; consumed by builder.py and validated by
# assert_compat_consistency() at autoload (PMT plan §3b/§4, prereq #7).
ALGORITHM_COMPAT: dict[str, str] = {}
NETWORK_COMPAT: dict[str, str] = {}


def register_algorithm(name, *, compat_name=None):
    """Register an algorithm class under `name` (== old class_name / YAML `name`).

    `compat_name` is the compat.SPECS axis key (e.g. "ppo", "add_ppo"). It must be
    given EXPLICITLY — class names like "ADDPPO"/"FPOPlus" do NOT lowercase cleanly
    into axis names, so a heuristic would be fragile. Stored in ALGORITHM_COMPAT.
    """
    def deco(cls):
        if name in ALGORITHMS and ALGORITHMS[name] is not cls:
            raise ValueError(f"algorithm '{name}' already registered to {ALGORITHMS[name]}")
        ALGORITHMS[name] = cls
        if compat_name is not None:
            existing = ALGORITHM_COMPAT.get(name)
            if existing is not None and existing != compat_name:
                raise ValueError(
                    f"algorithm '{name}' compat_name conflict: {existing!r} vs {compat_name!r}"
                )
            ALGORITHM_COMPAT[name] = compat_name
        return cls
    return deco


def register_network(name, *, compat_name=None):
    """Register a network class under `name` (== old class_name / YAML `name`).

    `compat_name` is the compat axis key (e.g. "transformer", "mlp"). Explicit for
    the same reason as algorithms ("TransformerActorCritic" -> "transformer").
    Stored in NETWORK_COMPAT.
    """
    def deco(cls):
        if name in NETWORKS and NETWORKS[name] is not cls:
            raise ValueError(f"network '{name}' already registered to {NETWORKS[name]}")
        NETWORKS[name] = cls
        if compat_name is not None:
            existing = NETWORK_COMPAT.get(name)
            if existing is not None and existing != compat_name:
                raise ValueError(
                    f"network '{name}' compat_name conflict: {existing!r} vs {compat_name!r}"
                )
            NETWORK_COMPAT[name] = compat_name
        return cls
    return deco


def register_runner(name):
    def deco(cls):
        if name in RUNNERS and RUNNERS[name] is not cls:
            raise ValueError(f"runner '{name}' already registered to {RUNNERS[name]}")
        RUNNERS[name] = cls
        return cls
    return deco


def get_algorithm(name):
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm '{name}'; registered: {sorted(ALGORITHMS)}")
    return ALGORITHMS[name]


def get_network(name):
    if name not in NETWORKS:
        raise KeyError(f"unknown network '{name}'; registered: {sorted(NETWORKS)}")
    return NETWORKS[name]


def get_runner(name):
    if name not in RUNNERS:
        raise KeyError(f"unknown runner '{name}'; registered: {sorted(RUNNERS)}")
    return RUNNERS[name]


def algorithm_compat_name(name):
    """Compat axis name for a registered algorithm `name`, or None if not mapped."""
    return ALGORITHM_COMPAT.get(name)


def network_compat_name(name):
    """Compat axis name for a registered network `name`, or None if not mapped."""
    return NETWORK_COMPAT.get(name)


def assert_compat_consistency():
    """Fail loud if the registry compat tables drift from compat.SPECS (PMT §3b).

    Checks (run after autoload):
      1. Every registered algorithm compat_name is a key in compat.SPECS.
      2. Every registered network compat_name appears in SOME algorithm's
         compatible_networks set (i.e. it is a real axis some algorithm accepts).
         A registered network not referenced by any spec is an orphan -> fail.

    Returns a dict of the resolved tables (for debuggability / reporting).
    """
    # imported here to avoid a circular import at module load
    from motion_tracking_rl import compat

    spec_algorithms = set(compat.SPECS)
    referenced_networks = set().union(
        *(s.compatible_networks for s in compat.SPECS.values())
    ) if compat.SPECS else set()

    # 1. algorithm compat names must be SPECS keys
    bad_alg = {n: c for n, c in ALGORITHM_COMPAT.items() if c not in spec_algorithms}
    if bad_alg:
        raise ValueError(
            "registry ALGORITHM_COMPAT has compat names absent from compat.SPECS: "
            f"{bad_alg} (SPECS keys: {sorted(spec_algorithms)})"
        )

    # 2. network compat names must be referenced by at least one spec
    bad_net = {n: c for n, c in NETWORK_COMPAT.items() if c not in referenced_networks}
    if bad_net:
        raise ValueError(
            "registry NETWORK_COMPAT has compat names not referenced by any algorithm's "
            f"compatible_networks: {bad_net} "
            f"(referenced networks: {sorted(referenced_networks)})"
        )

    return {"ALGORITHM_COMPAT": dict(ALGORITHM_COMPAT), "NETWORK_COMPAT": dict(NETWORK_COMPAT)}


def autoload():
    """Import algorithm/runner/slice-network modules to trigger decorator registration.

    Algorithms import cleanly (per Wave 1). Network/runner modules may pull in
    isaaclab/omni at import time, so those imports are wrapped and failures recorded.
    Returns a dict describing what loaded and what failed (for debuggability).
    """
    result: dict[str, object] = {"algorithms": [], "networks": [], "runners": [], "failed": {}}

    # Algorithms first — these import cleanly without omni/isaacsim.
    for mod in (
        "motion_tracking_rl.algorithms.ppo",
        "motion_tracking_rl.algorithms.bpo",
        "motion_tracking_rl.algorithms.fpo_plus",
        "motion_tracking_rl.algorithms.add_ppo",
        "motion_tracking_rl.algorithms.distillation",
    ):
        try:
            __import__(mod)
            result["algorithms"].append(mod)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - debug aid
            result["failed"][mod] = repr(exc)  # type: ignore[index]

    # Slice networks — may need omni at import time.
    for mod in (
        "motion_tracking_rl.networks.actor_critic",
        "motion_tracking_rl.networks.transformer_actor_critic",
        "motion_tracking_rl.networks.vision_transformer_actor_critic",
        "motion_tracking_rl.networks.student_teacher",
        "motion_tracking_rl.networks.vision_student_teacher",
        "motion_tracking_rl.networks.perceptive_motion",
        "motion_tracking_rl.networks.vision_ablation_actor_critic",
    ):
        try:
            __import__(mod)
            result["networks"].append(mod)  # type: ignore[union-attr]
        except Exception as exc:
            result["failed"][mod] = repr(exc)  # type: ignore[index]

    # Runners — may need omni at import time.
    for mod in (
        "motion_tracking_rl.runners.on_policy_runner",
        "motion_tracking_rl.runners.distillation_runner",
    ):
        try:
            __import__(mod)
            result["runners"].append(mod)  # type: ignore[union-attr]
        except Exception as exc:
            result["failed"][mod] = repr(exc)  # type: ignore[index]

    return result
