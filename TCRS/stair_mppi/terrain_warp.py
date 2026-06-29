"""
Terrain reference warping for stair climbing.

Three core operations:
  1. Stance foot lock — latch (x,y,z) to support pose at touchdown
  2. Swing clearance warp — raise mid-swing z above max terrain along path
  3. Support-aware root z — derive pelvis z from warped support feet + EMA

Usage:
    from stair_mppi.terrain_warp import TerrainReferenceWarper
    warper = TerrainReferenceWarper(terrain, foot_contact_z=0.049)
    wp, wl, wr, debug = warper.warp_segment(
        raw_pelvis, raw_lfoot, raw_rfoot,
        left_contact, right_contact,
        left_support_xy, right_support_xy,
    )
"""

import numpy as np

from stair_mppi.terrain import StairTerrain

# G1 leg geometry constants (verified from XML)
HIP_OFFSET_Z = -0.251   # pelvis origin to hip_yaw joint, z component (m)
L_UPPER_LEG = 0.194      # hip_yaw to knee (m)
L_LOWER_LEG = 0.300      # knee to ankle_roll (m)
L_LEG_MAX = L_UPPER_LEG + L_LOWER_LEG  # 0.494m, hip to ankle chain
# Pelvis-body origin to ankle-roll-link reach is larger than |HIP_OFFSET_Z| +
# leg length because the hip chain has non-zero lateral/forward offsets before
# the main thigh/shin links. The planner/IK works in pelvis-body ↔ ankle-body
# coordinates, so use a reach consistent with that chain.
PELVIS_ANKLE_REACH = abs(HIP_OFFSET_Z) + L_LEG_MAX + 0.06


def _iter_mask_runs(mask: np.ndarray):
    """Yield (start, end, value) for consecutive runs in a boolean mask."""
    i = 0
    n = mask.shape[0]
    while i < n:
        j = i + 1
        while j < n and mask[j] == mask[i]:
            j += 1
        yield i, j, bool(mask[i])
        i = j


class SupportAwareRootZFilter:
    """Smooth a precomputed absolute pelvis-z target with EMA + rate limit.

    New API (``step``): caller constructs z_target externally; filter only
    applies asymmetric EMA + per-frame rate clamping.

    Legacy API (``_step_one_frame``): retained for backward compatibility with
    older planner scripts that still pass raw/warped foot data into the filter.
    """

    def __init__(
        self,
        alpha_up: float = 0.35,
        alpha_down: float = 0.10,
        max_delta: float = 0.02,
        weight_smooth: float = 1,
        flight_alpha: float = 0.6,
    ):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.max_delta = max_delta
        self.weight_smooth = weight_smooth
        self.flight_alpha = flight_alpha
        self._ema_z = None
        self._prev_w = np.array([0.5, 0.5], dtype=np.float64)

    def reset(self):
        self._ema_z = None
        self._prev_w = np.array([0.5, 0.5], dtype=np.float64)

    # ------------------------------------------------------------------
    # New clean API: pure smoother, caller constructs z_target externally
    # ------------------------------------------------------------------

    def step(self, z_target: float, is_flight: bool = False) -> float:
        """Smooth *one* externally-constructed z_target.

        Args:
            z_target: absolute pelvis z target for this frame.
            is_flight: if True, use faster tracking alpha.

        Returns:
            Smoothed pelvis z.
        """
        if self._ema_z is None:
            self._ema_z = float(z_target)
            return self._ema_z

        alpha = self.alpha_up if z_target > self._ema_z else self.alpha_down
        if is_flight:
            alpha = max(alpha, self.flight_alpha)

        ema_target = self._ema_z + alpha * (z_target - self._ema_z)
        dz = np.clip(ema_target - self._ema_z, -self.max_delta, self.max_delta)
        self._ema_z = self._ema_z + dz
        return self._ema_z

    # ------------------------------------------------------------------
    # Legacy API: retained for backward compatibility
    # ------------------------------------------------------------------

    def _step_one_frame(
        self,
        ema_z,
        prev_w,
        raw_pelvis_z: float,
        raw_lfoot_z: float,
        raw_rfoot_z: float,
        warp_lfoot_z: float,
        warp_rfoot_z: float,
        left_weight: float,
        right_weight: float,
        is_flight: bool = False,
    ) -> tuple:
        """Advance filter by one frame. Returns (out_z, new_ema_z, new_prev_w).

        Accepts continuous support weights [0,1] instead of binary contact bools.
        Backward compatible: bool True/False auto-casts to 1.0/0.0.

        Flight phase: uses raw_pelvis_z directly (ballistic trajectory from
        mocap) with higher EMA alpha for fast tracking.  Triggered by
        ``is_flight=True`` (explicit phase-clock signal) **or** when both
        support weights are near zero (``w_sum < 1e-6``).
        """
        lw = float(left_weight)
        rw = float(right_weight)
        w_sum = lw + rw

        if is_flight:
            # TRUE FLIGHT: keep raw ballistic root, but don't let it sink
            # below terrain-aware estimate
            raw_sup_z = 0.5 * (raw_lfoot_z + raw_rfoot_z)
            delta_nom = raw_pelvis_z - raw_sup_z
            warp_sup_z = min(warp_lfoot_z, warp_rfoot_z)
            z_target = max(raw_pelvis_z, delta_nom + warp_sup_z)
            new_prev_w = prev_w.copy()
        elif w_sum < 1e-6:
            # No reliable support weights but not explicitly flight:
            # use terrain-aware fallback instead of raw pelvis
            raw_sup_z = 0.5 * (raw_lfoot_z + raw_rfoot_z)
            delta_nom = raw_pelvis_z - raw_sup_z
            warp_sup_z = min(warp_lfoot_z, warp_rfoot_z)
            z_target = delta_nom + warp_sup_z
            new_prev_w = prev_w.copy()
        else:
            # Normalize weights for blending
            w = np.array([lw / w_sum, rw / w_sum])

            # Smooth weight transition (only for non-flight)
            w = (1.0 - self.weight_smooth) * prev_w + self.weight_smooth * w
            new_prev_w = w.copy()

            raw_sup_z = w[0] * raw_lfoot_z + w[1] * raw_rfoot_z
            delta_nom = raw_pelvis_z - raw_sup_z

            warp_sup_z = w[0] * warp_lfoot_z + w[1] * warp_rfoot_z
            z_target = delta_nom + warp_sup_z

        # Reachability clamp: pelvis must not be so high that either foot
        # is unreachable. Use pelvis-body ↔ ankle-body reach, not the simpler
        # hip-z + leg-length estimate.
        # During flight, use raw foot z (warp foot z may have terrain artifacts).
        max_reach = PELVIS_ANKLE_REACH - 0.01
        if is_flight:
            lowest_foot_z = min(raw_lfoot_z, raw_rfoot_z)
        else:
            lowest_foot_z = min(warp_lfoot_z, warp_rfoot_z)
        z_target = min(z_target, lowest_foot_z + max_reach)

        if ema_z is None:
            new_ema_z = z_target
        else:
            alpha = self.alpha_up if z_target > ema_z else self.alpha_down
            # During flight, use higher alpha for fast ballistic tracking
            if is_flight:
                alpha = max(alpha, self.flight_alpha)
            ema_target = ema_z + alpha * (z_target - ema_z)
            dz = np.clip(ema_target - ema_z, -self.max_delta, self.max_delta)
            new_ema_z = ema_z + dz

        return new_ema_z, new_ema_z, new_prev_w

    def filter_segment(
        self,
        raw_pelvis: np.ndarray,
        raw_lfoot: np.ndarray,
        raw_rfoot: np.ndarray,
        warp_lfoot: np.ndarray,
        warp_rfoot: np.ndarray,
        left_contact: np.ndarray,
        right_contact: np.ndarray,
        left_weight: np.ndarray = None,
        right_weight: np.ndarray = None,
        flight_mask: np.ndarray = None,
    ) -> np.ndarray:
        """Compute warped pelvis z for the segment.

        Only frame 0 commits state to the filter. Frames 1..H-1 are
        previewed using a cloned state so that future horizon does not
        leak into the current-frame EMA.

        Args:
            raw_pelvis: (H, 3) raw pelvis positions
            raw_lfoot: (H, 3) raw left foot positions
            raw_rfoot: (H, 3) raw right foot positions
            warp_lfoot: (H, 3) warped left foot positions
            warp_rfoot: (H, 3) warped right foot positions
            left_contact: (H,) bool contact mask
            right_contact: (H,) bool contact mask
            left_weight: (H,) optional continuous support weights [0,1]
            right_weight: (H,) optional continuous support weights [0,1]
            flight_mask: (H,) optional bool — True = both feet airborne

        Returns:
            (H,) warped pelvis z values
        """
        H = raw_pelvis.shape[0]
        out_z = np.zeros(H, dtype=np.float64)

        # Use continuous weights if provided, else derive from bool contact
        if left_weight is None:
            left_weight = left_contact.astype(np.float64)
        if right_weight is None:
            right_weight = right_contact.astype(np.float64)
        if flight_mask is None:
            flight_mask = np.zeros(H, dtype=bool)

        # --- Frame 0: commit to real state ---
        out_z_val, self._ema_z, self._prev_w = self._step_one_frame(
            self._ema_z, self._prev_w,
            raw_pelvis[0, 2], raw_lfoot[0, 2], raw_rfoot[0, 2],
            warp_lfoot[0, 2], warp_rfoot[0, 2],
            left_weight[0], right_weight[0],
            is_flight=bool(flight_mask[0]),
        )
        out_z[0] = out_z_val

        # --- Frames 1..H-1: preview with cloned state (no commit) ---
        preview_ema = self._ema_z
        preview_w = self._prev_w.copy()
        for i in range(1, H):
            out_z_val, preview_ema, preview_w = self._step_one_frame(
                preview_ema, preview_w,
                raw_pelvis[i, 2], raw_lfoot[i, 2], raw_rfoot[i, 2],
                warp_lfoot[i, 2], warp_rfoot[i, 2],
                left_weight[i], right_weight[i],
                is_flight=bool(flight_mask[i]),
            )
            out_z[i] = out_z_val

        return out_z


def apply_swing_clearance_warp(
    raw_foot: np.ndarray,
    warped_foot: np.ndarray,
    contact_mask: np.ndarray,
    support_xy: np.ndarray,
    terrain: StairTerrain,
    foot_contact_z: float,
    clearance: float = 0.03,
    n_path_samples: int = 11,
) -> np.ndarray:
    """Raise swing-phase foot z to clear terrain obstacles.

    Only modifies z during swing phases. Leaves x/y unchanged.
    Uses a parabolic bump profile that peaks at mid-swing.

    Args:
        raw_foot: (H, 3) original flat-ground foot positions
        warped_foot: (H, 3) foot positions after stance lock (modified in-place copy)
        contact_mask: (H,) bool — True = stance
        support_xy: (H, 2) latched support x,y per frame
        terrain: terrain height lookup
        foot_contact_z: ankle height above ground during stance
        clearance: additional clearance above max terrain (m)
        n_path_samples: number of points to sample along swing path

    Returns:
        (H, 3) warped foot with swing clearance applied
    """
    out = warped_foot.copy()
    H = contact_mask.shape[0]

    for s, e, is_contact in _iter_mask_runs(contact_mask):
        if is_contact:
            continue  # skip stance phases

        # --- liftoff point ---
        if s > 0:
            x0, y0, z0 = out[s - 1]
        else:
            x0, y0, z0 = raw_foot[s]

        # --- landing point ---
        if e < H:
            x_td = support_xy[e, 0] if np.isfinite(support_xy[e, 0]) else raw_foot[e, 0]
            y_td = support_xy[e, 1] if np.isfinite(support_xy[e, 1]) else raw_foot[e, 1]
            z_td = terrain.height_at(x_td, y_td) + foot_contact_z
        else:
            x_td, y_td = raw_foot[e - 1, :2]
            z_td = max(
                raw_foot[e - 1, 2],
                terrain.height_at(x_td, y_td) + foot_contact_z,
            )

        # --- max terrain along swing path ---
        xs = np.linspace(x0, x_td, n_path_samples)
        ys = np.linspace(y0, y_td, n_path_samples)
        h_path_max = float(np.max(terrain.height_batch(xs, ys)))

        # --- required peak z ---
        z_peak_req = max(
            float(np.max(raw_foot[s:e, 2])),
            h_path_max + foot_contact_z + clearance,
            z0,
            z_td,
        )

        # --- apply parabolic bump ---
        swing_len = e - s
        for k in range(s, e):
            u = (k - s + 0.5) / swing_len  # phase in (0, 1)
            bump = 4.0 * u * (1.0 - u)
            z_line = (1.0 - u) * z0 + u * z_td
            z_candidate = z_line + bump * (z_peak_req - max(z0, z_td))
            out[k, 2] = max(raw_foot[k, 2], z_candidate)

    return out


class TerrainReferenceWarper:
    """Unified terrain reference warper.

    Performs stance lock -> swing clearance -> support-aware root z
    in the correct order.
    """

    def __init__(
        self,
        terrain: StairTerrain,
        foot_contact_z: float,
        clearance: float = 0.04,
        root_alpha_up: float = 0.35,
        root_alpha_down: float = 0.10,
        root_max_delta: float = 0.02,
        weight_smooth: float = 0.25,
        terrain_z_reduction: float = 0.10,
    ):
        self.terrain = terrain
        self.foot_contact_z = foot_contact_z
        self.clearance = clearance
        self.terrain_z_reduction = terrain_z_reduction
        self.root_filter = SupportAwareRootZFilter(
            alpha_up=root_alpha_up,
            alpha_down=root_alpha_down,
            max_delta=root_max_delta,
            weight_smooth=weight_smooth,
        )

    def reset(self):
        self.root_filter.reset()

    def warp_segment(
        self,
        raw_pelvis: np.ndarray,
        raw_lfoot: np.ndarray,
        raw_rfoot: np.ndarray,
        left_contact: np.ndarray,
        right_contact: np.ndarray,
        left_support_xy: np.ndarray,
        right_support_xy: np.ndarray,
        left_weight: np.ndarray = None,
        right_weight: np.ndarray = None,
    ) -> tuple:
        """Warp a raw motion segment onto terrain.

        Order of operations:
          1. Stance lock both feet
          2. Swing clearance warp both feet
          3. Support-aware pelvis z from warped feet
          4. Apply terrain_z_reduction to lower pelvis target

        Args:
            raw_pelvis: (H, 3) raw pelvis positions
            raw_lfoot: (H, 3) raw left foot positions
            raw_rfoot: (H, 3) raw right foot positions
            left_contact: (H,) bool
            right_contact: (H,) bool
            left_support_xy: (H, 2)
            right_support_xy: (H, 2)
            left_weight: (H,) optional continuous support weights [0,1]
            right_weight: (H,) optional continuous support weights [0,1]

        Returns:
            (warp_pelvis, warp_lfoot, warp_rfoot, debug_dict)
        """
        # Import here to avoid circular dependency
        from stair_mppi.terrain_footstep_planner import (
            apply_stance_lock,
            compute_support_surface_heights,
        )

        wp = raw_pelvis.copy()
        wl = raw_lfoot.copy()
        wr = raw_rfoot.copy()

        # 1. Stance lock
        wl, left_support_z = apply_stance_lock(
            wl, left_contact, left_support_xy, self.terrain, self.foot_contact_z
        )
        wr, right_support_z = apply_stance_lock(
            wr, right_contact, right_support_xy, self.terrain, self.foot_contact_z
        )

        # 2. Swing clearance warp
        wl = apply_swing_clearance_warp(
            raw_lfoot, wl, left_contact, left_support_xy,
            self.terrain, self.foot_contact_z, self.clearance,
        )
        wr = apply_swing_clearance_warp(
            raw_rfoot, wr, right_contact, right_support_xy,
            self.terrain, self.foot_contact_z, self.clearance,
        )

        # 3. Support-aware pelvis z
        wp[:, 2] = self.root_filter.filter_segment(
            raw_pelvis, raw_lfoot, raw_rfoot,
            wl, wr, left_contact, right_contact,
            left_weight=left_weight, right_weight=right_weight,
        )

        # 4. Reduce pelvis z target by terrain_z_reduction to allow more knee flexion
        if self.terrain_z_reduction > 0.0:
            wp[:, 2] -= self.terrain_z_reduction

        # Keep raw x, y for pelvis
        # (pelvis x/y follow the raw motion, only z is warped)

        debug = {
            "left_support_z": left_support_z,
            "right_support_z": right_support_z,
        }

        return wp, wl, wr, debug
