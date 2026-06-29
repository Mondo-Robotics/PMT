"""Expert buffer over the streaming terrain+flat motion store (no Isaac Sim dependency).

The BFM-Zero discriminator ``Q_D`` and backward map ``B`` only consume ``state`` and
``privileged_state`` (see the input-filter keys in ``humanoidverse/train.py``). So the expert
buffer must serve transitions whose observation dicts carry ``state``/``privileged_state``
(plus a zero ``last_action`` for schema compatibility) built with the SAME ``obs_math`` layout
as the online env.

This module builds those observations from the ragged ``FlatMotionStore`` and exposes a
``sample(batch_size)`` API compatible with the BFM ``TrajectoryDictBuffer`` consumers
(returns ``observation`` and ``next.observation`` plus ``terminated``/``truncated``/``motion_id``).

Motion ``body_quat_w`` is stored in IsaacLab wxyz convention; we convert to xyzw via
``obs_math.wxyz_to_xyzw`` before the (w_last=True) builders.

Two implementations are provided:
  - ``StaticExpertBuffer``: materializes per-clip obs into a flat tensor store (good for small
    datasets / tests). Sampling draws contiguous ``seq_length`` windows that never cross clip
    boundaries.
  - ``StreamingExpertSlicer``: lazily samples ``(motion_id, start_frame)`` pairs and gathers
    windows on demand from a ``FlatMotionStore`` working set (memory-bounded, for real training).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import obs_math


def _gravity_vec(n: int, device: torch.device) -> torch.Tensor:
    g = torch.zeros(n, 3, device=device)
    g[:, 2] = -1.0
    return g


def build_expert_obs_from_frames(
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_wxyz: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the BFM observation dict for a batch of expert frames.

    Args:
        joint_pos_rel: [M, 29] joint positions already relative to the default pose.
        joint_vel:     [M, 29] joint velocities.
        body_pos_w:    [M, B, 3] tracked-body world positions (body 0 = root/pelvis).
        body_quat_wxyz:[M, B, 4] tracked-body world quaternions in wxyz (IsaacLab convention).
        body_lin_vel_w:[M, B, 3] tracked-body world linear velocities.
        body_ang_vel_w:[M, B, 3] tracked-body world angular velocities.

    Returns dict with ``state`` (64), ``privileged_state`` (208), ``last_action`` (29 zeros).
    """
    m = joint_pos_rel.shape[0]
    device = joint_pos_rel.device
    body_quat_xyzw = obs_math.wxyz_to_xyzw(body_quat_wxyz)

    root_quat_xyzw = body_quat_xyzw[:, 0, :]
    proj_grav = obs_math.projected_gravity_from_root_quat(root_quat_xyzw, _gravity_vec(m, device))
    # IMPORTANT: the ONLINE env state uses base-frame root angular velocity
    # (robot.data.root_ang_vel_b). The motion store provides WORLD-frame root ang vel
    # (body_ang_vel_w[:, 0]); rotate it into the base frame so the expert and online ``state``
    # share the same convention. A world-vs-base frame leak here lets the discriminator/backward
    # map trivially separate expert from online, collapsing the CPR reward.
    base_ang_vel = obs_math.base_ang_vel_from_world(root_quat_xyzw, body_ang_vel_w[:, 0, :])

    state = obs_math.build_state_from_raw(joint_pos_rel, joint_vel, proj_grav, base_ang_vel)
    # 30 real bodies + 1 extended head body -> 463-dim privileged_state (official contract).
    privileged = obs_math.build_privileged_state_with_extend(
        body_pos_w, body_quat_xyzw, body_lin_vel_w, body_ang_vel_w
    )
    last_action = torch.zeros(m, obs_math.LAST_ACTION_DIM, device=device)
    return {"state": state, "privileged_state": privileged, "last_action": last_action}


@dataclass
class _ClipObs:
    """Per-clip precomputed observation tensors (state/privileged/last_action)."""

    state: torch.Tensor  # [T, 64]
    privileged_state: torch.Tensor  # [T, 208]
    last_action: torch.Tensor  # [T, 29]
    motion_id: int
    is_flat: bool


class StaticExpertBuffer:
    """In-memory expert buffer that serves seq_length windows for the discriminator/backward map.

    Suitable for small datasets and unit tests. Each clip's observations are precomputed once.
    ``sample`` returns a BFM-style batch where each row is a contiguous ``seq_length`` window that
    stays within a single clip (so ``next`` is the next frame of the same clip).
    """

    def __init__(self, clips: list[_ClipObs], seq_length: int, device: torch.device | str = "cpu"):
        if not clips:
            raise ValueError("StaticExpertBuffer requires at least one clip")
        self.seq_length = seq_length
        self.device = torch.device(device)
        self.clips = clips
        self.motion_ids = [c.motion_id for c in clips]
        self._clip_lengths = [c.state.shape[0] for c in clips]
        self._max_len = max(self._clip_lengths)
        # Precompute default-seq_length starts (used by sample()).
        self._starts = self._build_starts(seq_length)
        if not self._starts:
            raise ValueError(
                f"No clip is long enough for seq_length={seq_length}+1; longest clip has "
                f"{self._max_len} frames"
            )

    def _build_starts(self, seq_length: int) -> list[tuple[int, int]]:
        """Valid (clip_idx, start) so frames [start .. start+seq_length] stay in-clip."""
        starts: list[tuple[int, int]] = []
        for ci, t in enumerate(self._clip_lengths):
            max_start = t - seq_length - 1  # need start..start+seq_length (next obs at +seq_length)
            for s in range(max(0, max_start + 1)):
                starts.append((ci, s))
        return starts

    def __len__(self) -> int:
        return len(self._starts)

    def _gather_seq(self, starts: list[tuple[int, int]], seq_length: int, offset: int) -> tuple[dict, torch.Tensor]:
        """Gather ``len(starts) * seq_length`` rows: for each start, frames offset..offset+L-1.

        Output rows are ordered [seq0_f0, seq0_f1, ..., seq0_f(L-1), seq1_f0, ...] so a
        downstream ``view(B//L, L, d)`` recovers contiguous expert sub-trajectories — exactly
        what ``FBcprAgent.encode_expert`` expects.
        """
        # Use contiguous per-sequence SLICES (views) instead of per-frame stacking: for each start
        # we take c.state[f0 : f0+seq_length], which keeps frames sequence-contiguous and avoids
        # thousands of tiny tensor ops per update (large throughput win).
        states, privs, acts, mids = [], [], [], []
        for ci, s in starts:
            c = self.clips[ci]
            f0 = s + offset
            f1 = f0 + seq_length
            states.append(c.state[f0:f1])
            privs.append(c.privileged_state[f0:f1])
            acts.append(c.last_action[f0:f1])
            mids.append(torch.full((seq_length,), c.motion_id, dtype=torch.long))
        obs = {
            "state": torch.cat(states, dim=0).to(self.device),
            "privileged_state": torch.cat(privs, dim=0).to(self.device),
            "last_action": torch.cat(acts, dim=0).to(self.device),
        }
        return obs, torch.cat(mids, dim=0).to(self.device)

    def sample(self, batch_size: int, seq_length: int | None = None) -> dict:
        """Return a BFM-style batch of ``batch_size`` rows grouped into contiguous sequences.

        ``batch_size`` must be divisible by ``seq_length`` (defaults to the buffer's seq_length).
        Rows are ``B // seq_length`` contiguous in-clip windows of length ``seq_length`` (obs at
        offsets 0..L-1, ``next`` at 1..L), flattened in sequence-contiguous order. This matches
        ``TrajectoryDictBuffer.sample(batch_size, seq_length)`` so both ``encode_expert``'s
        ``view(B//L, L, d)`` and ``_sample_tracking_z``'s ``view(batch_dim, traj_length, d)``
        reshapes are correct.
        """
        seq_length = seq_length or self.seq_length
        if batch_size % seq_length != 0:
            raise ValueError(f"batch_size {batch_size} must be divisible by seq_length {seq_length}")
        starts_pool = self._starts if seq_length == self.seq_length else self._build_starts(seq_length)
        if not starts_pool:
            raise ValueError(f"No clip long enough for seq_length={seq_length}; longest has {self._max_len} frames")
        num_slices = batch_size // seq_length
        pick_rows = torch.randint(0, len(starts_pool), (num_slices,)).tolist()
        starts = [starts_pool[r] for r in pick_rows]
        obs, mids = self._gather_seq(starts, seq_length, offset=0)
        next_obs, _ = self._gather_seq(starts, seq_length, offset=1)
        terminated = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
        return {
            "observation": obs,
            "next": {"observation": next_obs, "terminated": terminated},
            "terminated": terminated,
            "truncated": truncated,
            "motion_id": mids,
        }


def build_static_expert_buffer_from_store(
    store,
    *,
    seq_length: int,
    default_joint_pos: torch.Tensor,
    is_flat_flags: list[bool] | None = None,
    device: torch.device | str = "cpu",
) -> StaticExpertBuffer:
    """Materialize a ``StaticExpertBuffer`` from a loaded ``FlatMotionStore`` working set.

    ``store`` must expose ``num_motions``, per-motion lengths (``lengths`` or ``motion_lengths``),
    and ``get_motion_data`` over the currently-committed working set (resident clips).
    ``default_joint_pos`` is [29] subtracted to form ``joint_pos_rel`` (matching the env).
    """
    default_joint_pos = default_joint_pos.to(device).reshape(1, -1)
    clips: list[_ClipObs] = []
    lengths = getattr(store, "lengths", None)
    if lengths is None:
        lengths = getattr(store, "motion_lengths", None)
    num = store.num_motions
    for mid in range(num):
        t = int(lengths[mid]) if lengths is not None else None
        if t is None:
            raise ValueError("store must expose per-motion lengths")
        frame_ids = torch.arange(t, device=device)
        motion_ids = torch.full((t,), mid, dtype=torch.long, device=device)
        jp, jv, bp, bq, blv, bav = store.get_motion_data(motion_ids, frame_ids)
        # The streaming store may hold fp16 tensors; upcast to float32 for the obs math (which
        # mixes with float32 constants like the gravity vector).
        jp = jp.to(device=device, dtype=torch.float32) - default_joint_pos.float()
        obs = build_expert_obs_from_frames(
            jp,
            jv.to(device=device, dtype=torch.float32),
            bp.to(device=device, dtype=torch.float32),
            bq.to(device=device, dtype=torch.float32),
            blv.to(device=device, dtype=torch.float32),
            bav.to(device=device, dtype=torch.float32),
        )
        is_flat = bool(is_flat_flags[mid]) if is_flat_flags is not None else False
        clips.append(
            _ClipObs(
                state=obs["state"],
                privileged_state=obs["privileged_state"],
                last_action=obs["last_action"],
                motion_id=mid,
                is_flat=is_flat,
            )
        )
    return StaticExpertBuffer(clips, seq_length=seq_length, device=device)
