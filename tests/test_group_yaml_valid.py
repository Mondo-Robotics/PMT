"""Every group YAML loads via OmegaConf; network/algorithm/runner names map to
the registry (or a known-pending set). (PMT plan §6 Phase-0 verify.)
"""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from motion_tracking_rl import registry

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# Networks that are real classes but not yet @register_network'd / autoloaded.
# As of Phase 2.2 VisionStudentTeacher is decorated, so this is empty.
KNOWN_PENDING_NETWORKS: set[str] = set()

GROUP_AXES = [
    "robot", "terrain", "motion", "scene", "sensor", "obs",
    "reward", "network", "algorithm", "stage", "runner",
]


@pytest.fixture(scope="module", autouse=True)
def _autoload():
    registry.autoload()


def _yaml_files(axis):
    return sorted((CONFIGS / axis).glob("*.yaml"))


@pytest.mark.parametrize("axis", GROUP_AXES)
def test_axis_yamls_load(axis):
    files = _yaml_files(axis)
    assert files, f"no yamls under configs/{axis}/"
    for f in files:
        cfg = OmegaConf.load(f)  # must not raise
        assert cfg is not None


def test_network_names_in_registry_or_pending():
    for f in _yaml_files("network"):
        cfg = OmegaConf.load(f)
        name = cfg.get("name")
        assert name is not None, f"{f} missing 'name'"
        assert name in registry.NETWORKS or name in KNOWN_PENDING_NETWORKS, (
            f"network '{name}' ({f.name}) not in registry.NETWORKS "
            f"{sorted(registry.NETWORKS)} nor known-pending {sorted(KNOWN_PENDING_NETWORKS)}"
        )


def test_algorithm_names_in_registry():
    for f in _yaml_files("algorithm"):
        cfg = OmegaConf.load(f)
        name = cfg.get("name")
        assert name is not None, f"{f} missing 'name'"
        assert name in registry.ALGORITHMS, (
            f"algorithm '{name}' ({f.name}) not in registry.ALGORITHMS "
            f"{sorted(registry.ALGORITHMS)}"
        )


def test_runner_names_in_registry():
    for f in _yaml_files("runner"):
        cfg = OmegaConf.load(f)
        name = cfg.get("name")
        assert name is not None, f"{f} missing 'name'"
        assert name in registry.RUNNERS, (
            f"runner '{name}' ({f.name}) not in registry.RUNNERS "
            f"{sorted(registry.RUNNERS)}"
        )
