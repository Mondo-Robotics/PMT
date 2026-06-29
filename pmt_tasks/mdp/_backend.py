"""Backend abstraction shim (MJLAB_BACKEND_PLAN.md Phase B).

PMT's MDP term functions were written against Isaac Lab. To run the SAME term math on
mjlab (MuJoCo-Warp), the only real divergences are:

  1. **Math utils**: `isaaclab.utils.math.*`  vs  `mjlab.utils.lab_api.math.*`
     (near-identical APIs; both use wxyz quaternions).
  2. **Robot data field names**: PMT reads `robot.data.body_pos_w`, `root_lin_vel_b`,
     `applied_torque`; mjlab exposes `body_link_pos_w`, `root_link_lin_vel_b`,
     `qfrc_actuator` (and `joint_torques` RAISES NotImplementedError — review correction).
  3. **SceneEntityCfg field**: isaaclab `asset_name` vs mjlab `name` (config-time).

This module resolves (1) and (2) behind one import so term functions stay backend-agnostic:

    from pmt_tasks.mdp import _backend as B
    pos = B.quat_apply(q, v)                 # math, either backend
    view = B.RobotView(env, "robot")         # data, canonical PMT names
    p = view.body_pos_w                       # -> body_link_pos_w on mjlab

The active backend is detected once from whichever sim package is importable, overridable
via env var ``PMT_BACKEND=isaaclab|mjlab``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

# --- backend detection -----------------------------------------------------


def _detect_backend() -> str:
    forced = os.environ.get("PMT_BACKEND")
    if forced in ("isaaclab", "mjlab"):
        return forced
    if importlib.util.find_spec("isaaclab") is not None:
        return "isaaclab"
    if importlib.util.find_spec("mjlab") is not None:
        return "mjlab"
    # default: assume isaaclab (PMT's historical backend) so pure tests import
    return "isaaclab"


BACKEND: str = _detect_backend()
IS_MJLAB: bool = BACKEND == "mjlab"
IS_ISAACLAB: bool = BACKEND == "isaaclab"


# --- math utils ------------------------------------------------------------
# Resolve a single `math_utils`-like namespace for either backend.

if IS_MJLAB:
    from mjlab.utils.lab_api import math as _m  # type: ignore

    # mjlab is missing the quat_rotate aliases; quat_apply IS quat_rotate (wxyz).
    quat_apply = _m.quat_apply
    quat_rotate = _m.quat_apply
    quat_inv = _m.quat_inv

    def quat_rotate_inverse(q, v):  # noqa: ANN001
        return _m.quat_apply(_m.quat_inv(q), v)

    quat_mul = _m.quat_mul
    quat_error_magnitude = _m.quat_error_magnitude
    yaw_quat = _m.yaw_quat
    matrix_from_quat = _m.matrix_from_quat
    subtract_frame_transforms = _m.subtract_frame_transforms
    convert_quat = _m.convert_quat
    wrap_to_pi = _m.wrap_to_pi
    math_utils = _m
else:
    import isaaclab.utils.math as _m  # type: ignore

    quat_apply = _m.quat_apply
    quat_rotate = getattr(_m, "quat_rotate", _m.quat_apply)
    quat_rotate_inverse = getattr(_m, "quat_rotate_inverse", None)
    quat_inv = _m.quat_inv
    quat_mul = _m.quat_mul
    quat_error_magnitude = _m.quat_error_magnitude
    yaw_quat = _m.yaw_quat
    matrix_from_quat = _m.matrix_from_quat
    subtract_frame_transforms = _m.subtract_frame_transforms
    convert_quat = _m.convert_quat
    wrap_to_pi = _m.wrap_to_pi
    math_utils = _m

    if quat_rotate_inverse is None:  # very old isaaclab

        def quat_rotate_inverse(q, v):  # noqa: ANN001,F811
            return _m.quat_apply(_m.quat_inv(q), v)


# --- robot data field aliasing ---------------------------------------------
# canonical PMT name -> mjlab EntityData attribute (isaaclab is identity).

_MJLAB_FIELD_MAP = {
    "body_pos_w": "body_link_pos_w",
    "body_quat_w": "body_link_quat_w",
    "body_lin_vel_w": "body_link_lin_vel_w",
    "body_ang_vel_w": "body_link_ang_vel_w",
    "root_lin_vel_b": "root_link_lin_vel_b",
    "root_ang_vel_b": "root_link_ang_vel_b",
    "root_pos_w": "root_link_pos_w",
    "root_quat_w": "root_link_quat_w",
    "root_lin_vel_w": "root_link_lin_vel_w",
    "root_ang_vel_w": "root_link_ang_vel_w",
    # review correction: mjlab .joint_torques RAISES; use actuator force.
    "applied_torque": "qfrc_actuator",
}


class RobotView:
    """Exposes a backend asset's ``.data`` under PMT's canonical field names.

    Identity passthrough on isaaclab; field-aliased on mjlab. Unknown attributes
    fall through to the underlying ``data`` object, so already-matching names
    (joint_pos, joint_vel, default_joint_pos, projected_gravity_b, ...) just work.
    """

    __slots__ = ("_data", "_asset")

    def __init__(self, env: Any, name: str = "robot"):
        self._asset = env.scene[name]
        self._data = self._asset.data

    @property
    def asset(self) -> Any:
        return self._asset

    def __getattr__(self, item: str) -> Any:
        if IS_MJLAB:
            item = _MJLAB_FIELD_MAP.get(item, item)
        return getattr(self._data, item)


def resolve_scene_entity_cfg(cfg: Any) -> Any:
    """Return a backend SceneEntityCfg, mapping PMT's ``asset_name`` to mjlab's ``name``.

    Accepts a PMT-style cfg (with ``asset_name``) and adapts it at config time.
    """
    if IS_MJLAB:
        from mjlab.managers.scene_entity_config import SceneEntityCfg  # type: ignore

        name = getattr(cfg, "asset_name", None) or getattr(cfg, "name", "robot")
        return SceneEntityCfg(name=name)
    return cfg
