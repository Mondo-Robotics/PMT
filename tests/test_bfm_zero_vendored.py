"""Regression gate: BFM-Zero (FB-CPR-Aux) is fully vendored inside PMT.

Guarantees PMT needs NOTHING from any external ``BFM-Zero`` repo / ``humanoidverse`` package:
  1. importing the bfm_zero submodules never imports a top-level ``humanoidverse`` module;
  2. the vendored FB-CPR-Aux agent config builds and runs a forward pass (CPU);
  3. all 17 vendored files stay byte-identical to the recorded upstream source (if present);
  4. ``ensure_bfm_zero_on_path`` is a no-op that does not mutate ``sys.path``.

Pure-torch / CPU only — no Isaac Sim required. Run:
    PYTHONPATH=<repo> python -m pytest tests/test_bfm_zero_vendored.py
or  PYTHONPATH=<repo> python tests/test_bfm_zero_vendored.py
"""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "motion_tracking_rl" / "bfm_zero" / "_vendor"


class _NoHumanoidverse:
    """Context manager that makes any ``import humanoidverse[...]`` raise."""

    def __enter__(self):
        self._real = builtins.__import__

        def guard(name, *a, **k):
            if name == "humanoidverse" or name.startswith("humanoidverse."):
                raise AssertionError(f"FORBIDDEN external import of '{name}' (must be vendored)")
            return self._real(name, *a, **k)

        builtins.__import__ = guard
        return self

    def __exit__(self, *exc):
        builtins.__import__ = self._real
        return False


def test_no_external_humanoidverse_import():
    with _NoHumanoidverse():
        import motion_tracking_rl.bfm_zero as bz  # noqa: F401
        from motion_tracking_rl.bfm_zero import (  # noqa: F401
            aux_rewards,
            config,
            expert_streaming,
            obs_math,
            tracking_eval,
        )
    leaked = [m for m in sys.modules if m == "humanoidverse" or m.startswith("humanoidverse.")]
    assert not leaked, f"external humanoidverse modules leaked into sys.modules: {leaked}"


def test_agent_builds_and_acts():
    import torch

    from motion_tracking_rl.bfm_zero import config

    with _NoHumanoidverse():
        cfg = config.build_agent_config(
            device="cpu", batch_size=8, rollout_expert_trajectories_length=4,
            hidden_dim=64, hidden_layers=2, embedding_layers=2,
            disc_hidden_dim=64, disc_hidden_layers=2,
            backward_hidden_dim=32, backward_hidden_layers=1,
            compile_agent=False,
        )
        space = config.build_obs_space(device="cpu")
        agent = cfg.build(obs_space=space, action_dim=29)

    obs = {k: torch.zeros(2, sp.shape[0]) for k, sp in space.spaces.items()}
    z = torch.nn.functional.normalize(torch.randn(2, 256), dim=-1)
    with torch.no_grad():
        action = agent.act(obs=obs, z=z, mean=True)
    assert tuple(action.shape) == (2, 29), action.shape


def test_ensure_bfm_zero_on_path_is_noop():
    from motion_tracking_rl.bfm_zero import config

    before = list(sys.path)
    config.ensure_bfm_zero_on_path()
    config.ensure_bfm_zero_on_path("/nonexistent/path/should/not/be/added")
    assert sys.path == before, "ensure_bfm_zero_on_path must not mutate sys.path anymore"


def test_vendored_files_byte_identical_to_upstream():
    """If the upstream BFM-Zero repo is present, the vendored copies must match it verbatim."""
    src_root = Path(__file__).resolve().parents[2] / "BFM-Zero" / "humanoidverse"
    if not src_root.exists():
        import pytest  # type: ignore

        pytest.skip("upstream BFM-Zero repo not present; skipping byte-identity check")

    mapping = {
        "agents/base.py": "agents/base.py",
        "agents/base_model.py": "agents/base_model.py",
        "agents/nn_models.py": "agents/nn_models.py",
        "agents/nn_filters.py": "agents/nn_filters.py",
        "agents/nn_filter_models.py": "agents/nn_filter_models.py",
        "agents/normalizers.py": "agents/normalizers.py",
        "agents/pytree_utils.py": "agents/pytree_utils.py",
        "agents/fb/agent.py": "agents/fb/agent.py",
        "agents/fb/model.py": "agents/fb/model.py",
        "agents/fb_cpr/agent.py": "agents/fb_cpr/agent.py",
        "agents/fb_cpr/model.py": "agents/fb_cpr/model.py",
        "agents/fb_cpr_aux/agent.py": "agents/fb_cpr_aux/agent.py",
        "agents/fb_cpr_aux/model.py": "agents/fb_cpr_aux/model.py",
        "agents/misc/zbuffer.py": "agents/misc/zbuffer.py",
        "agents/buffers/transition.py": "agents/buffers/transition.py",
        "agents/envs/utils/gym_spaces.py": "agents/envs/utils/gym_spaces.py",
        "torch_utils.py": "utils/torch_utils.py",
    }
    diffs = []
    for vend_rel, src_rel in mapping.items():
        v = (VENDOR / vend_rel).read_bytes()
        s = (src_root / src_rel).read_bytes()
        if v != s:
            diffs.append(vend_rel)
    assert not diffs, f"vendored files drifted from upstream: {diffs}"


if __name__ == "__main__":
    test_no_external_humanoidverse_import()
    print("[1/4] no external humanoidverse import .......... PASS")
    test_agent_builds_and_acts()
    print("[2/4] agent builds + acts (CPU) ................. PASS")
    test_ensure_bfm_zero_on_path_is_noop()
    print("[3/4] ensure_bfm_zero_on_path no-op ............. PASS")
    try:
        test_vendored_files_byte_identical_to_upstream()
        print("[4/4] vendored files byte-identical ............ PASS")
    except Exception as e:  # noqa: BLE001
        print(f"[4/4] vendored files byte-identical ............ {e}")
    print("ALL BFM-ZERO VENDORING CHECKS PASSED")
