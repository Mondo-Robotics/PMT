"""Offline per-bin difficulty prior (pure torch, no isaaclab).

Phase 2 of the adaptive-sampling plan (adaptive_sampling_discussion.md §3 Phase 2):
precompute a per-clip, per-bin difficulty prior in [0,1] from cheap kinematic
signals already present in every motion npz, and inject it into the hybrid sampler's
``failed_bin_count`` so high-dynamic / high-frequency segments are sampled MORE even
before any online failure statistics exist.

Signals (all derivable from joint_vel + body velocities — no extra FK):
  * high-frequency energy ratio of joint velocities (rfft per bin): fast, jittery,
    bursty segments (jumps, spins, foot strikes) score high.
  * RMS jerk (finite-difference of joint velocity): transient / aggressive control.
  * body acceleration magnitude proxy (finite-diff of body linear velocity): whole-
    body dynamics (takeoff, landing, direction changes).

These are combined per bin, then the WHOLE-DATASET distribution of each feature is
normalized to [0,1] via robust min/quantile scaling so the prior is comparable across
clips. The result is a ``[num_clips, max_bins]`` tensor (zeros on invalid/padding bins).

This module is import-light (only torch) so it is unit-testable without Isaac Sim and
can be driven by the offline script ``scripts/precompute_motion_prior.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Cache format version — bump if the feature math or layout changes so a stale cache
# is rejected rather than silently mismatched.
PRIOR_CACHE_VERSION = 1


@dataclass
class ClipBinFeatures:
    """Per-bin raw (un-normalized) features for one clip. Each tensor is [num_bins]."""

    hf_ratio: Tensor      # high-frequency energy ratio of joint velocity
    rms_jerk: Tensor      # RMS of joint-velocity finite difference (jerk proxy)
    body_accel: Tensor    # mean body linear-acceleration magnitude proxy
    num_bins: int


def _bin_slices(num_frames: int, bin_size: int) -> list[tuple[int, int]]:
    """[(start, end)] frame ranges for each bin (last bin may be short). >=1 bin."""
    n_bins = max(1, (num_frames + bin_size - 1) // bin_size)
    out = []
    for b in range(n_bins):
        s = b * bin_size
        e = min((b + 1) * bin_size, num_frames)
        if s >= num_frames:
            s, e = max(0, num_frames - 1), num_frames
        out.append((s, e))
    return out


def _high_freq_ratio(x: Tensor) -> float:
    """Fraction of spectral energy in the upper half of the frequency band.

    x: [T, D] (time-major). Returns a scalar in [0,1]. For T < 2 returns 0 (no
    dynamics resolvable). Energy is summed over the rfft bins above the median
    frequency index, divided by total spectral energy (excluding the DC term so a
    constant offset does not dominate).
    """
    T = x.shape[0]
    if T < 4:
        return 0.0
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    xc = x - x.mean(dim=0, keepdim=True)          # remove DC / mean pose-rate
    spec = torch.fft.rfft(xc, dim=0).abs()         # [F, D]
    spec = spec[1:]                                 # drop DC bin
    if spec.shape[0] < 2:
        return 0.0
    energy = (spec ** 2).sum(dim=1)                 # [F-1] over channels
    total = energy.sum()
    if total <= 0:
        return 0.0
    half = spec.shape[0] // 2
    hi = energy[half:].sum()
    return float((hi / total).clamp(0.0, 1.0).item())


def compute_clip_bin_features(
    joint_vel: Tensor,
    body_lin_vel_w: Tensor | None,
    fps: float,
    bin_size: int,
) -> ClipBinFeatures:
    """Compute per-bin raw features for one clip.

    Args:
        joint_vel: [T, J] joint velocities.
        body_lin_vel_w: [T, B, 3] body linear velocities (or None -> body_accel=0).
        fps: clip frame rate (for jerk/accel scaling to per-second units).
        bin_size: frames per bin.
    """
    T = int(joint_vel.shape[0])
    dt = 1.0 / float(fps) if fps > 0 else 1.0
    slices = _bin_slices(T, bin_size)
    n_bins = len(slices)

    hf = torch.zeros(n_bins, dtype=torch.float32)
    jerk = torch.zeros(n_bins, dtype=torch.float32)
    accel = torch.zeros(n_bins, dtype=torch.float32)

    # Sanitize non-finite inputs up front (a single NaN/Inf would otherwise poison the
    # whole feature channel's dataset-wide quantile normalization downstream).
    jv = torch.nan_to_num(joint_vel.float(), nan=0.0, posinf=0.0, neginf=0.0)
    # Joint acceleration (jerk proxy) = d(joint_vel)/dt across the whole clip, then
    # bin-averaged RMS. ``jacc[k]`` is the diff for the frame transition k->k+1, so we
    # attribute it to the bin owning the LEFT frame k (left-frame convention). This
    # keeps every interior diff in exactly one bin AND includes the boundary diff
    # k=e-1 (transition into the next bin) in the current bin, so sharp transitions at
    # a bin edge are not dropped (Codex Phase-2 finding).
    if T >= 2:
        jacc = (jv[1:] - jv[:-1]) / dt              # [T-1, J]; jacc[k] = frame k->k+1
    else:
        jacc = torch.zeros((0, jv.shape[1]), dtype=torch.float32)

    if body_lin_vel_w is not None and T >= 2:
        blv = torch.nan_to_num(body_lin_vel_w.float(), nan=0.0, posinf=0.0, neginf=0.0)
        bacc = (blv[1:] - blv[:-1]) / dt            # [T-1, B, 3]
        bacc_mag = bacc.norm(dim=-1).mean(dim=-1)   # [T-1] mean over bodies
    else:
        bacc_mag = None

    n_diff = jacc.shape[0]
    for b, (s, e) in enumerate(slices):
        if e - s >= 1:
            hf[b] = _high_freq_ratio(jv[s:e])
        # Diffs owned by this bin: left frame in [s, e) -> diff indices [s, e) capped at
        # n_diff. This assigns the boundary diff (frame e-1 -> e) to THIS bin exactly
        # once and never double-counts it in the next bin (whose range starts at e).
        a_s, a_e = min(s, n_diff), min(e, n_diff)
        if a_e > a_s:
            seg = jacc[a_s:a_e]
            jerk[b] = float(torch.sqrt((seg ** 2).mean()).item())
            if bacc_mag is not None:
                accel[b] = float(bacc_mag[a_s:a_e].mean().item())

    return ClipBinFeatures(hf_ratio=hf, rms_jerk=jerk, body_accel=accel, num_bins=n_bins)


def _robust_unit_norm(values: Tensor, valid: Tensor, q_lo: float = 0.05, q_hi: float = 0.95) -> Tensor:
    """Scale ``values`` to [0,1] using dataset quantiles over VALID entries.

    Robust to outliers: maps [q_lo, q_hi] quantile range -> [0,1], clamps outside.
    Invalid entries are left as 0. If the valid range is degenerate, returns zeros.
    """
    out = torch.zeros_like(values)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    v = values[valid]
    if v.numel() == 0:
        return out
    lo = torch.quantile(v, q_lo)
    hi = torch.quantile(v, q_hi)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or (hi - lo) <= 1e-12:
        return out
    scaled = ((values - lo) / (hi - lo)).clamp(0.0, 1.0)
    out[valid] = scaled[valid]
    return out


def combine_features_to_prior(
    per_clip_features: list[ClipBinFeatures],
    max_bins: int,
    *,
    w_hf: float = 0.4,
    w_jerk: float = 0.3,
    w_accel: float = 0.3,
) -> Tensor:
    """Combine per-clip features into a dataset-normalized prior ``[N, max_bins]``.

    Each feature is normalized to [0,1] across ALL valid bins of the WHOLE dataset
    (so clips are comparable), then combined with the given weights. Padding bins
    (beyond a clip's num_bins, or beyond max_bins) stay 0.
    """
    n = len(per_clip_features)
    hf = torch.zeros(n, max_bins, dtype=torch.float32)
    jerk = torch.zeros(n, max_bins, dtype=torch.float32)
    accel = torch.zeros(n, max_bins, dtype=torch.float32)
    valid = torch.zeros(n, max_bins, dtype=torch.bool)

    for i, f in enumerate(per_clip_features):
        nb = min(f.num_bins, max_bins)
        hf[i, :nb] = f.hf_ratio[:nb]
        jerk[i, :nb] = f.rms_jerk[:nb]
        accel[i, :nb] = f.body_accel[:nb]
        valid[i, :nb] = True

    hf_n = _robust_unit_norm(hf, valid)
    jerk_n = _robust_unit_norm(jerk, valid)
    accel_n = _robust_unit_norm(accel, valid)

    wsum = max(w_hf + w_jerk + w_accel, 1e-8)
    prior = (w_hf * hf_n + w_jerk * jerk_n + w_accel * accel_n) / wsum
    prior[~valid] = 0.0
    return prior.clamp(0.0, 1.0)


def slice_prior_to_working_set(
    global_prior: Tensor,
    working_to_global: Tensor,
    local_max_bins: int,
) -> Tensor:
    """Slice a global ``[N_global, max_bins_global]`` prior to the resident set.

    Rows are gathered by ``working_to_global`` (local id -> global id); columns are
    truncated or zero-padded to ``local_max_bins`` (the resident sampler's bin count).
    Returns ``[num_resident, local_max_bins]`` on the same device as ``global_prior``.
    """
    w2g = working_to_global.to(global_prior.device).long()
    rows = global_prior[w2g]                        # [R, max_bins_global]
    R = rows.shape[0]
    out = torch.zeros(R, local_max_bins, dtype=rows.dtype, device=rows.device)
    c = min(local_max_bins, rows.shape[1])
    out[:, :c] = rows[:, :c]
    return out
