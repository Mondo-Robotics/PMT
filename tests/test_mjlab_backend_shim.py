"""Phase B invariants (MJLAB_BACKEND_PLAN.md): mjlab backend import hygiene + shim.

These tests are designed to run in the *mjlab* venv (no isaaclab/omni installed):
  <mjlab-repo>/.venv/bin/python -m pytest tests/test_mjlab_backend_shim.py

They guard the two Phase-B guarantees:
  1. `import pmt_tasks.mdp` must NOT drag in isaaclab/omni/pxr (USD-free import).
  2. The `_backend` shim resolves math + field aliases on whichever backend is active.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_MJLAB = importlib.util.find_spec("mjlab") is not None
_HAS_ISAACLAB = importlib.util.find_spec("isaaclab") is not None


def _backend_math_importable() -> bool:
    """The _backend shim needs ONE backend's math package actually importable.

    (wbt-style envs have neither: isaaclab's spec exists but importing it needs omni,
    which is absent; mjlab isn't installed.) We must actually import, not just find_spec.
    """
    try:
        from pmt_tasks.mdp import _backend  # noqa: F401

        return True
    except Exception:
        return False


_BACKEND_READY = _backend_math_importable()
_skip_no_backend = pytest.mark.skipif(
    not _BACKEND_READY, reason="no backend math importable (e.g. isaaclab needs omni)"
)


def test_mdp_import_is_usd_free():
    """`import pmt_tasks.mdp` must succeed without isaaclab/omni present."""
    import pmt_tasks.mdp as m  # should not raise

    # When isaaclab is absent the heavy command stack is intentionally not loaded.
    if not _HAS_ISAACLAB:
        assert m._HAS_ISAACLAB is False


@_skip_no_backend
def test_backend_detection():
    from pmt_tasks.mdp import _backend as B

    assert B.BACKEND in ("isaaclab", "mjlab")
    if _HAS_MJLAB and not _HAS_ISAACLAB:
        assert B.BACKEND == "mjlab"


@_skip_no_backend
def test_backend_math_identity_quat():
    import torch

    from pmt_tasks.mdp import _backend as B

    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity (wxyz)
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(B.quat_apply(q, v), v)
    assert torch.allclose(B.quat_rotate(q, v), v)
    assert torch.allclose(B.quat_rotate_inverse(q, v), v)


@_skip_no_backend
def test_applied_torque_maps_to_qfrc_actuator_on_mjlab():
    from pmt_tasks.mdp import _backend as B

    # review correction: mjlab .joint_torques raises; shim must use qfrc_actuator.
    if B.IS_MJLAB:
        assert B._MJLAB_FIELD_MAP["applied_torque"] == "qfrc_actuator"
        assert B._MJLAB_FIELD_MAP["body_pos_w"] == "body_link_pos_w"
        assert B._MJLAB_FIELD_MAP["root_lin_vel_b"] == "root_link_lin_vel_b"


@pytest.mark.skipif(not (_HAS_MJLAB and not _HAS_ISAACLAB), reason="mjlab-only env")
def test_robotview_aliases_against_real_entity_data():
    """RobotView must alias canonical PMT names onto mjlab EntityData properties."""
    from mjlab.entity.data import EntityData

    from pmt_tasks.mdp import _backend as B

    props = set(dir(EntityData))
    for pmt_name, mj_name in B._MJLAB_FIELD_MAP.items():
        # every aliased target must actually exist on mjlab EntityData
        assert mj_name in props, f"{pmt_name} -> {mj_name} not on mjlab EntityData"
