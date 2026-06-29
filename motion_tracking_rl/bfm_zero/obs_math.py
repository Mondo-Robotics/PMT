"""Pure-torch builders for the BFM-Zero observation dict (no Isaac Sim dependency).

These functions reproduce the BFM-Zero observation contract used by ``humanoidverse`` so the
same ``FBcprAuxAgent`` networks can consume IsaacLab tensors *and* expert-mocap tensors with an
identical layout.

Contract for G1 29-DOF, 14 tracked bodies (TRACKED_BODY_NAMES order, root = index 0):

- ``state``           : 64  = joint_pos_rel(29) + joint_vel(29) + projected_gravity(3) + base_ang_vel(3)
- ``privileged_state``: 208 = compute_humanoid_observations_max over 14 bodies
                              (root_height 1 + local_body_pos 39 + tan_norm_rot 84
                               + local_body_vel 42 + local_body_ang_vel 42)
- ``last_action``     : 29
- ``history_actor``   : per-key blocks in SORTED key order, each block [N, H*key_dim] newest-first.
                        keys/dims: actions(29), base_ang_vel(3), dof_pos(29), dof_vel(29),
                        projected_gravity(3). For H=4 -> 4*29+4*3+4*29+4*29+4*3 = 372.
                        Matches ``humanoidverse._get_obs_history_actor``: iterate
                        ``sorted(history_config.keys())`` and concat per-key reshaped blocks.

All quaternion math reuses ``humanoidverse.utils.torch_utils`` (w_last=True convention,
i.e. xyzw). IsaacLab stores quaternions wxyz, so callers MUST convert before passing body
quaternions in here (see ``wxyz_to_xyzw``).
"""

from __future__ import annotations

import torch

# Reuse BFM-Zero's exact quaternion conventions (xyzw / w_last=True).
# Vendored verbatim under ._vendor (no external BFM-Zero repo required).
from ._vendor.torch_utils import (  # type: ignore
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_rotate_inverse,
    quat_to_tan_norm,
)

# Per-frame component dims (G1 29-DOF).
NUM_JOINTS = 29
# privileged_state body count. GROUND TRUTH (from the official pretrained model's init_kwargs):
# privileged_state = 463 = 31 bodies = 30 real robot bodies + 1 EXTENDED virtual "head_link"
# (parent=torso_link, offset [0,0,0.35], identity rot). The original BFM-Zero ``max_local_self``
# used exactly this 31-body set; matching it is required for obs-contract parity (a 14- or 30-body
# subset changes the latent geometry the backward map / discriminator learned).
NUM_REAL_BODIES = 30
NUM_EXTEND_BODIES = 1  # virtual head_link appended after the 30 real bodies (index 30)
NUM_PRIVILEGED_BODIES = NUM_REAL_BODIES + NUM_EXTEND_BODIES  # 31
# Kept for backwards-compat references.
NUM_TRACKED_BODIES = NUM_REAL_BODIES

# Extended-body definition (matches the original g1 extend_config head_link).
EXTEND_PARENT_BODY = "torso_link"
EXTEND_POS_IN_PARENT = (0.0, 0.0, 0.35)  # body-frame offset from torso

STATE_DIM = NUM_JOINTS + NUM_JOINTS + 3 + 3  # 64
LAST_ACTION_DIM = NUM_JOINTS  # 29
# privileged_state = 1 + (B-1)*3 + B*6 + B*3 + B*3 with B=31 -> 1+90+186+93+93 = 463
PRIVILEGED_STATE_DIM = (
    1
    + (NUM_PRIVILEGED_BODIES - 1) * 3
    + NUM_PRIVILEGED_BODIES * 6
    + NUM_PRIVILEGED_BODIES * 3
    + NUM_PRIVILEGED_BODIES * 3
)

# Sum of per-frame history component dims (used for dim bookkeeping): 93.
HISTORY_FRAME_DIM = NUM_JOINTS + 3 + NUM_JOINTS + NUM_JOINTS + 3


# Canonical pelvis-first 30-body order (import-light module; no torch/isaac).
try:
    from .body_names import PRIVILEGED_BODY_NAMES  # noqa: E402
except ImportError:  # loaded by file path (tests) without package context
    import importlib.util as _ilu
    import os as _os

    _spec = _ilu.spec_from_file_location(
        "_bfm_body_names", _os.path.join(_os.path.dirname(__file__), "body_names.py")
    )
    _bn = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_bn)
    PRIVILEGED_BODY_NAMES = _bn.PRIVILEGED_BODY_NAMES

# PRIVILEGED_BODY_NAMES lists the 30 REAL robot bodies (the extended head is synthetic, appended
# at runtime by append_extended_head_body, so it is NOT in this name list).
assert len(PRIVILEGED_BODY_NAMES) == NUM_REAL_BODIES
assert PRIVILEGED_BODY_NAMES[0] == "pelvis"


def wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert IsaacLab (w, x, y, z) quaternions to BFM/torch_utils (x, y, z, w)."""
    return torch.cat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)


# Index of the extend-body parent (torso_link) within PRIVILEGED_BODY_NAMES (the 30 real bodies).
TORSO_BODY_INDEX = 15


def append_extended_head_body(
    body_pos_w: torch.Tensor,
    body_quat_xyzw: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    *,
    parent_index: int = TORSO_BODY_INDEX,
    pos_in_parent: tuple[float, float, float] = EXTEND_POS_IN_PARENT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append the virtual ``head_link`` body (index 30) to the 30 real bodies -> 31 bodies.

    Replicates ``legged_robot_motions._init_motion_extend`` exactly (head_link: parent=torso_link,
    pos_in_parent=[0,0,0.35], rot_in_parent=identity):

      extend_pos     = my_quat_rotate(parent_rot_xyzw, pos_in_parent) + parent_pos
      extend_rot     = quat_mul(parent_rot_xyzw, identity_xyzw) = parent_rot
      extend_ang_vel = parent_ang_vel  (copied)
      extend_lin_vel = parent_lin_vel + cross(parent_ang_vel, pos_in_parent)  # raw offset (orig quirk)

    Inputs are [N, 30, *] tensors (xyzw quats). Returns [N, 31, *] tensors with the head at index 30.
    """
    n = body_pos_w.shape[0]
    device = body_pos_w.device
    parent_pos = body_pos_w[:, parent_index, :]
    parent_rot = body_quat_xyzw[:, parent_index, :]
    parent_lin = body_lin_vel_w[:, parent_index, :]
    parent_ang = body_ang_vel_w[:, parent_index, :]

    offset = torch.tensor(pos_in_parent, dtype=body_pos_w.dtype, device=device).reshape(1, 3).expand(n, 3)

    # rot_in_parent is identity, so the original's outer my_quat_rotate(identity, .) is a no-op.
    extend_pos = my_quat_rotate(parent_rot, offset) + parent_pos
    extend_rot = parent_rot  # quat_mul(parent_rot, identity) == parent_rot
    extend_ang = parent_ang
    extend_lin = parent_lin + torch.cross(parent_ang, offset, dim=-1)

    out_pos = torch.cat([body_pos_w, extend_pos.unsqueeze(1)], dim=1)
    out_rot = torch.cat([body_quat_xyzw, extend_rot.unsqueeze(1)], dim=1)
    out_lin = torch.cat([body_lin_vel_w, extend_lin.unsqueeze(1)], dim=1)
    out_ang = torch.cat([body_ang_vel_w, extend_ang.unsqueeze(1)], dim=1)
    return out_pos, out_rot, out_lin, out_ang


def build_privileged_state_with_extend(
    body_pos_w: torch.Tensor,
    body_quat_xyzw: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Append the extended head body then build the 463-dim privileged_state (31 bodies)."""
    p, q, lv, av = append_extended_head_body(body_pos_w, body_quat_xyzw, body_lin_vel_w, body_ang_vel_w)
    return build_privileged_state(p, q, lv, av, **kwargs)


def build_state(
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_ang_vel: torch.Tensor,
) -> torch.Tensor:
    """state (64) = [joint_pos_rel(29), joint_vel(29), projected_gravity(3), base_ang_vel(3)].

    Mirrors the ``g1env_state`` tensor assembled in
    ``humanoidverse.agents.envs.humanoidverse_isaac._get_g1env_observation`` which concatenates
    the obs-SCALED actor observations. Note ``base_ang_vel`` is scaled by ``BASE_ANG_VEL_SCALE``
    (0.25) in the env, so callers should pass an already-scaled ``base_ang_vel`` (or use
    ``build_state_from_raw``).
    """
    return torch.cat([joint_pos_rel, joint_vel, projected_gravity, base_ang_vel], dim=-1)


def build_state_from_raw(
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_ang_vel_raw: torch.Tensor,
) -> torch.Tensor:
    """Same as :func:`build_state` but applies the env obs scale to ``base_ang_vel`` (×0.25)."""
    return build_state(joint_pos_rel, joint_vel, projected_gravity, base_ang_vel_raw * BASE_ANG_VEL_SCALE)


def build_privileged_state(
    body_pos_w: torch.Tensor,
    body_quat_xyzw: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    *,
    root_height_obs: bool = True,
    local_root_obs: bool = True,
) -> torch.Tensor:
    """privileged_state (208): max-info body observation over the tracked bodies.

    Faithful re-implementation of
    ``humanoidverse.envs.legged_robot_motions.compute_humanoid_observations_max`` (w_last=True),
    producing the concatenation [root_height, local_body_pos, local_body_rot_tan_norm,
    local_body_vel, local_body_ang_vel].

    Args:
        body_pos_w:      [N, B, 3] world body positions; body 0 = root/pelvis.
        body_quat_xyzw:  [N, B, 4] world body quaternions in xyzw.
        body_lin_vel_w:  [N, B, 3] world body linear velocities.
        body_ang_vel_w:  [N, B, 3] world body angular velocities.
    """
    n, b, _ = body_pos_w.shape
    root_pos = body_pos_w[:, 0, :]
    root_rot = body_quat_xyzw[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2).repeat((1, b, 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(n * b, 4)

    # Local body positions (root-relative, heading-aligned), drop root pos (=0).
    local_body_pos = body_pos_w - root_pos.unsqueeze(-2)
    flat_local_body_pos = local_body_pos.reshape(n * b, 3)
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(n, b * 3)[..., 3:]

    # Local body rotations -> tan/norm (6D per body).
    flat_body_rot = body_quat_xyzw.reshape(n * b, 4)
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(n, b * 6)

    if not local_root_obs:
        root_rot_obs = quat_to_tan_norm(root_rot, w_last=True)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_lin_vel_w.reshape(n * b, 3)
    local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel).reshape(n, b * 3)

    flat_body_ang_vel = body_ang_vel_w.reshape(n * b, 3)
    local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel).reshape(n, b * 3)

    parts = []
    if root_height_obs:
        parts.append(root_h)
    parts.extend([local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel])
    return torch.cat(parts, dim=-1)


def projected_gravity_from_root_quat(root_quat_xyzw: torch.Tensor, gravity_vec: torch.Tensor) -> torch.Tensor:
    """Projected gravity in the base frame from a root quaternion (xyzw).

    ``gravity_vec`` is the world gravity direction, typically [0, 0, -1] broadcast to [N, 3].
    """
    return quat_rotate_inverse(root_quat_xyzw, gravity_vec, w_last=True)


def base_ang_vel_from_world(root_quat_xyzw: torch.Tensor, ang_vel_w: torch.Tensor) -> torch.Tensor:
    """Rotate a WORLD-frame root angular velocity into the BASE frame (matches online env).

    The online env exposes ``robot.data.root_ang_vel_b`` (base frame); motion data provides
    world-frame ``body_ang_vel_w[:, 0]``. Use this to align the expert ``state`` convention.
    """
    return quat_rotate_inverse(root_quat_xyzw, ang_vel_w, w_last=True)


# history_actor per-key components (key -> dim). Iterated in SORTED key order to match
# ``humanoidverse._get_obs_history_actor`` (which iterates ``sorted(history_config.keys())``).
HISTORY_KEY_DIMS: dict[str, int] = {
    "actions": NUM_JOINTS,
    "base_ang_vel": 3,
    "dof_pos": NUM_JOINTS,
    "dof_vel": NUM_JOINTS,
    "projected_gravity": 3,
}

# BFM obs scales applied to the per-key history values (from config/obs/bfm_zero_obs.yaml).
# Only base_ang_vel is scaled (0.25); others are 1.0. The same scales apply to the ``state``
# tensor's base_ang_vel component for parity.
HISTORY_OBS_SCALES: dict[str, float] = {
    "actions": 1.0,
    "base_ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 1.0,
    "projected_gravity": 1.0,
}

BASE_ANG_VEL_SCALE: float = 0.25


class HistoryActorBuffer:
    """Rolling per-env, per-key history flattened into ``history_actor``.

    Faithfully mirrors ``humanoidverse``'s ``HistoryHandler`` + ``_get_obs_history_actor``:
    each proprio key has its own ring buffer of length ``history_len`` (index 0 = newest), and the
    flattened ``history_actor`` is the concatenation, in SORTED key order, of each key's
    ``[N, history_len * key_dim]`` block. Values pushed are already obs-SCALED to match the env.
    On reset, an env's history is zeroed.

    TIMING CONTRACT (must match BFM-Zero):
        history_actor = buf.flatten()   # observation uses history BEFORE current frame
        buf.push(current_frame)         # then current frame is appended for next step
    i.e. the ``history_actor`` accompanying step ``t``'s observation contains frames
    ``t-1, t-2, ...`` (zeros after reset). Callers MUST flatten before pushing.

    SCALING CONTRACT:
        - pass RAW proprio values and keep ``apply_scale=True`` (default), OR
        - pass already actor-obs-scaled values with ``apply_scale=False``.
        Mixing (scaled values + apply_scale=True) double-scales base_ang_vel.
    """

    def __init__(self, num_envs: int, history_len: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.history_len = history_len
        self.device = torch.device(device)
        self.keys = sorted(HISTORY_KEY_DIMS.keys())
        # key -> [N, H, key_dim], index 0 = newest.
        self.bufs: dict[str, torch.Tensor] = {
            k: torch.zeros(num_envs, history_len, HISTORY_KEY_DIMS[k], device=self.device) for k in self.keys
        }

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for k in self.keys:
            if env_ids is None:
                self.bufs[k].zero_()
            else:
                self.bufs[k][env_ids] = 0.0

    def push(self, frames: dict[str, torch.Tensor], *, apply_scale: bool = True) -> None:
        """Insert newest per-key values; shift older frames back by one.

        ``frames`` maps each history key to a ``[N, key_dim]`` tensor of RAW (unscaled) values.
        Obs scales are applied here so the stored/flattened history matches the env exactly.
        """
        for k in self.keys:
            v = frames[k]
            if apply_scale:
                v = v * HISTORY_OBS_SCALES[k]
            self.bufs[k] = torch.roll(self.bufs[k], shifts=1, dims=1)
            self.bufs[k][:, 0, :] = v

    def flatten(self) -> torch.Tensor:
        """Return ``history_actor`` ([N, dim]) as sorted-key per-key blocks, newest-first."""
        blocks = [self.bufs[k].reshape(self.num_envs, -1) for k in self.keys]
        return torch.cat(blocks, dim=1)

    @property
    def dim(self) -> int:
        return self.history_len * sum(HISTORY_KEY_DIMS.values())
