"""
MuJoCo Jacobian IK for visualizing G1 ghost poses from MPPI foot targets.

The root pose is fixed to the target root pose. Upper-body joints are held to
the provided reference joint angles. Only the 12 leg joints are updated.
"""

import json
from dataclasses import dataclass

import mujoco
import numpy as np


LEFT_FOOT_BODY_NAME = "left_ankle_roll_link"
RIGHT_FOOT_BODY_NAME = "right_ankle_roll_link"

# Foot contact geometry offsets in ankle_roll_link local frame.
# Derived from G1 MuJoCo model contact spheres (r=5mm):
#   Heel spheres at [-0.05, ±0.025, -0.03], Toe spheres at [0.12, ±0.03, -0.03]
# We use the midpoints as representative toe/heel points.
TOE_OFFSET = np.array([0.12, 0.0, -0.03])
HEEL_OFFSET = np.array([-0.05, 0.0, -0.03])

LEFT_LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]

RIGHT_LEG_JOINT_NAMES = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


@dataclass
class GhostIKResult:
    qpos: np.ndarray
    leg_qpos: np.ndarray
    left_pos_err: float
    right_pos_err: float
    left_rot_err: float
    right_rot_err: float
    converged: bool
    iterations: int
    root_pos_offset: np.ndarray = None  # (3,) how much root moved from target
    cost: float = 0.0
    seed_index: int = 0
    used_multiseed: bool = False


@dataclass
class FrameIKTargets:
    """Per-frame IK targets for trajectory optimization."""
    root_pos: np.ndarray          # (3,)
    root_quat: np.ndarray         # (4,) wxyz
    left_pos: np.ndarray          # (3,)
    left_quat: np.ndarray         # (4,) wxyz
    right_pos: np.ndarray         # (3,)
    right_quat: np.ndarray        # (4,) wxyz
    fixed_upper_body_qpos: np.ndarray  # (n_joints - 7,)
    ref_leg_qpos: np.ndarray = None    # (12,)
    left_point_scale: np.ndarray = None
    right_point_scale: np.ndarray = None
    root_xy_weight: float = 2.0
    root_z_weight: float = 10.0


class G1GhostJacobianIK:
    """Differential IK for the articulated G1 ghost visualization.

    Supports per-body position and rotation constraints loaded from JSON config.
    Each body in the config can have:
      - pos_weight: weight for 3D position tracking (0 = disabled)
      - rot_weight: weight for 3D orientation tracking (0 = disabled)

    Special entries (left_toe, left_heel, right_toe, right_heel) define virtual
    constraint points on the ankle_roll_link with offsets.
    """

    # Bodies that are part of the foot multi-point system (handled separately)
    _FOOT_MULTIPOINT_KEYS = {
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_toe", "left_heel", "right_toe", "right_heel",
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        max_iters: int = 30,
        tol: float = 1e-3,
        damping: float = 0.05,
        step_size: float = 0.5,
        orientation_weight: float = 0.2,
        posture_weight: float = 0.02,
        max_joint_step: float = 0.20,
        penetration_weight: float = 10.0,
        ankle_weight: float = 1.0,
        toe_weight: float = 1.0,
        heel_weight: float = 1.0,
        ankle_rot_weight: float = 0.0,
        multiseed_enabled: bool = True,
        multiseed_num_seeds: int = 3,
        multiseed_trigger_pos_err: float = 0.20,
        multiseed_trigger_joint_step: float = 0.80,
        multiseed_seed_span: float = 0.35,
        multiseed_continuity_weight: float = 0.25,
        multiseed_posture_weight: float = 0.02,
    ):
        self.model = model
        self.max_iters = max(1, int(max_iters))
        self.tol = float(tol)
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.orientation_weight = float(orientation_weight)
        self.posture_weight = max(float(posture_weight), 0.0)
        self.max_joint_step = max(float(max_joint_step), 1e-6)
        self.penetration_weight = max(float(penetration_weight), 1.0)
        self.multiseed_enabled = bool(multiseed_enabled)
        self.multiseed_num_seeds = max(1, int(multiseed_num_seeds))
        self.multiseed_trigger_pos_err = max(float(multiseed_trigger_pos_err), 0.0)
        self.multiseed_trigger_joint_step = max(float(multiseed_trigger_joint_step), 0.0)
        self.multiseed_seed_span = max(float(multiseed_seed_span), 0.0)
        self.multiseed_continuity_weight = max(float(multiseed_continuity_weight), 0.0)
        self.multiseed_posture_weight = max(float(multiseed_posture_weight), 0.0)
        # Per-point weights for multi-point IK: [ankle, toe, heel] per foot.
        # Applied as diagonal scaling to error and Jacobian rows.
        self.point_weights = np.array([
            ankle_weight, ankle_weight, ankle_weight,
            toe_weight, toe_weight, toe_weight,
            heel_weight, heel_weight, heel_weight,
        ], dtype=np.float64)
        # Ankle orientation weight: explicit rotation tracking via axis-angle error.
        # Complements the multi-point (toe/heel) position constraints which only
        # weakly constrain yaw.  Set > 0 to enable (e.g. 6.0).
        self.ankle_rot_weight = max(float(ankle_rot_weight), 0.0)

        # Extra body constraints: list of (body_id, pos_weight, rot_weight)
        # Populated by from_config(). Bodies not in _FOOT_MULTIPOINT_KEYS
        # with non-zero weights get added here.
        self.body_constraints = []

        # Knee-forward penalty weight (loaded from config solver section).
        # Penalizes configurations where the knee is behind the pelvis
        # in the body's forward (x) direction. Prevents leg-flip solutions.
        self.knee_forward_weight = 0.0

        self.left_foot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FOOT_BODY_NAME
        )
        self.right_foot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FOOT_BODY_NAME
        )
        if self.left_foot_body_id == -1 or self.right_foot_body_id == -1:
            raise ValueError("Failed to locate ghost foot bodies in MuJoCo model")

        # Knee body IDs for knee-forward penalty
        self.left_knee_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link"
        )
        self.right_knee_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_knee_link"
        )
        self.pelvis_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        left_joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in LEFT_LEG_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        right_joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in RIGHT_LEG_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(left_joint_ids < 0) or np.any(right_joint_ids < 0):
            raise ValueError("Failed to locate one or more ghost leg joints")

        self.leg_joint_ids = np.concatenate([left_joint_ids, right_joint_ids], axis=0)
        self.leg_qpos_adr = np.array(
            [model.jnt_qposadr[jid] for jid in self.leg_joint_ids],
            dtype=np.int32,
        )
        self.leg_dof_adr = np.array(
            [model.jnt_dofadr[jid] for jid in self.leg_joint_ids],
            dtype=np.int32,
        )
        self.leg_limits = model.jnt_range[self.leg_joint_ids].copy()
        # Tighten joint limits to prevent non-physical solutions (e.g. hip rotating 180°).
        # URDF limits are mechanical extremes; walking/dancing uses a much smaller range.
        # Joint order: [L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
        #               R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]
        # No static joint limit tightening — dynamic protection is done in
        # solve_with_root via max_joint_step clamping and continuity cost.
        self._leg_limit_margin = 1e-3 * (self.leg_limits[:, 1] - self.leg_limits[:, 0])
        self._effective_multiseed_num = max(self.multiseed_num_seeds, 5)
        self._halton_seed_offsets = self._build_halton_offsets(
            max(self._effective_multiseed_num, 1), len(self.leg_qpos_adr)
        )

        all_joint_qpos_adr = np.arange(7, model.nq, dtype=np.int32)
        self.non_leg_qpos_adr = np.setdiff1d(all_joint_qpos_adr, self.leg_qpos_adr)
        self.non_leg_local_idx = self.non_leg_qpos_adr - 7

        # Reusable Jacobian buffers
        self._jacp_left = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr_left = np.zeros((3, model.nv), dtype=np.float64)
        self._jacp_right = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr_right = np.zeros((3, model.nv), dtype=np.float64)
        self._quat_left = np.zeros(4, dtype=np.float64)
        self._quat_right = np.zeros(4, dtype=np.float64)
        self._err_rot_left = np.zeros(3, dtype=np.float64)
        self._err_rot_right = np.zeros(3, dtype=np.float64)

        # Multi-point IK buffers: toe and heel Jacobians for each foot
        self._jacp_left_toe = np.zeros((3, model.nv), dtype=np.float64)
        self._jacp_left_heel = np.zeros((3, model.nv), dtype=np.float64)
        self._jacp_right_toe = np.zeros((3, model.nv), dtype=np.float64)
        self._jacp_right_heel = np.zeros((3, model.nv), dtype=np.float64)
        # MuJoCo point buffer (3,) for mj_jac calls
        self._point_buf = np.zeros(3, dtype=np.float64)

        # Generic body constraint buffers (allocated on first use or by from_config)
        self._jacp_body = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr_body = np.zeros((3, model.nv), dtype=np.float64)
        self._quat_err_buf = np.zeros(4, dtype=np.float64)
        self._rot_err_buf = np.zeros(3, dtype=np.float64)

        # Second MjData for computing reference body poses via FK
        self._ref_data = mujoco.MjData(model)

    @classmethod
    def from_config(cls, model: mujoco.MjModel, config_path: str) -> "G1GhostJacobianIK":
        """Create IK solver from a JSON config file (GMR-style per-body weights).

        The config has two sections:
          - body_weights: per-body pos_weight/rot_weight
          - solver: global solver parameters

        Real MuJoCo bodies with non-zero pos/rot weights (excluding the foot
        multi-point entries) are added as extra body constraints.
        """
        with open(config_path, "r") as f:
            cfg = json.load(f)

        solver_cfg = cfg.get("solver", {})
        body_cfg = cfg.get("body_weights", {})

        # Extract foot multi-point weights from body_weights
        ankle_w = body_cfg.get("left_ankle_roll_link", {}).get("pos_weight", 1.0)
        ankle_rw = body_cfg.get("left_ankle_roll_link", {}).get("rot_weight", 0.0)
        toe_w = body_cfg.get("left_toe", {}).get("pos_weight", 1.0)
        heel_w = body_cfg.get("left_heel", {}).get("pos_weight", 1.0)

        instance = cls(
            model=model,
            max_iters=solver_cfg.get("max_iters", 30),
            tol=solver_cfg.get("tol", 1e-3),
            damping=solver_cfg.get("damping", 0.05),
            step_size=solver_cfg.get("step_size", 0.5),
            orientation_weight=solver_cfg.get("orientation_weight", 0.2),
            posture_weight=solver_cfg.get("posture_weight", 0.02),
            max_joint_step=solver_cfg.get("max_joint_step", 0.20),
            penetration_weight=solver_cfg.get("penetration_weight", 10.0),
            ankle_weight=ankle_w,
            toe_weight=toe_w,
            heel_weight=heel_w,
            ankle_rot_weight=ankle_rw,
            multiseed_enabled=solver_cfg.get("multiseed_enabled", True),
            multiseed_num_seeds=solver_cfg.get("multiseed_num_seeds", 3),
            multiseed_trigger_pos_err=solver_cfg.get("multiseed_trigger_pos_err", 0.20),
            multiseed_trigger_joint_step=solver_cfg.get("multiseed_trigger_joint_step", 0.80),
            multiseed_seed_span=solver_cfg.get("multiseed_seed_span", 0.35),
            multiseed_continuity_weight=solver_cfg.get("multiseed_continuity_weight", 0.25),
            multiseed_posture_weight=solver_cfg.get("multiseed_posture_weight", 0.02),
        )

        # Collect extra body constraints (non-foot-multipoint bodies)
        for body_name, bw in body_cfg.items():
            if body_name in cls._FOOT_MULTIPOINT_KEYS:
                continue
            if body_name.startswith("_"):
                continue
            pw = bw.get("pos_weight", 0.0)
            rw = bw.get("rot_weight", 0.0)
            if pw == 0.0 and rw == 0.0:
                continue
            # pelvis is handled separately via root_xy/z_weight
            if body_name == "pelvis":
                continue
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                print(f"  [IK config] Warning: body '{body_name}' not found, skipping")
                continue
            instance.body_constraints.append((bid, float(pw), float(rw), body_name))

        if instance.body_constraints:
            names = [bc[3] for bc in instance.body_constraints]
            print(f"  [IK config] Extra body constraints: {names}")

        # Knee-forward penalty
        kfw = solver_cfg.get("knee_forward_weight", 0.0)
        instance.knee_forward_weight = float(kfw)
        if kfw > 0:
            print(f"  [IK config] Knee-forward penalty weight: {kfw}")

        if instance.ankle_rot_weight > 0:
            print(f"  [IK config] Ankle rotation weight: {instance.ankle_rot_weight}")

        return instance

    @staticmethod
    def _first_primes(n: int) -> list:
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    @staticmethod
    def _van_der_corput(index: int, base: int) -> float:
        value = 0.0
        denom = 1.0
        while index > 0:
            index, remainder = divmod(index, base)
            denom *= base
            value += remainder / denom
        return value

    @classmethod
    def _build_halton_offsets(cls, n_samples: int, dim: int) -> np.ndarray:
        """Deterministic low-discrepancy offsets in [-1, 1].

        cuRobo uses Halton-like seeds to cover joint space without random
        clumping.  This CPU version keeps the same idea, but uses the samples
        as local offsets around the continuous/ref seed instead of launching a
        full global search every frame.
        """
        if n_samples <= 0 or dim <= 0:
            return np.zeros((0, dim), dtype=np.float64)
        primes = cls._first_primes(dim)
        samples = np.zeros((n_samples, dim), dtype=np.float64)
        for i in range(n_samples):
            # Start from index 2 so the first non-primary sample is not all zeros.
            idx = i + 2
            for d, base in enumerate(primes):
                samples[i, d] = 2.0 * cls._van_der_corput(idx, base) - 1.0
        return samples

    def extract_leg_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return np.asarray(qpos[self.leg_qpos_adr], dtype=np.float64).copy()

    def _clip_leg_qpos(self, leg_qpos: np.ndarray) -> np.ndarray:
        lower = self.leg_limits[:, 0] + self._leg_limit_margin
        upper = self.leg_limits[:, 1] - self._leg_limit_margin
        return np.clip(np.asarray(leg_qpos, dtype=np.float64), lower, upper)

    @staticmethod
    def _seed_is_duplicate(seed: np.ndarray, seeds: list, tol: float = 1e-5) -> bool:
        return any(float(np.max(np.abs(seed - old))) < tol for old in seeds)

    def _candidate_leg_seeds(
        self,
        initial_leg_qpos: np.ndarray = None,
        ref_leg_qpos: np.ndarray = None,
    ) -> list:
        """Build deterministic candidate seeds, ordered by continuity priority."""
        seeds = []
        n_candidates = self._effective_multiseed_num

        initial = None if initial_leg_qpos is None else self._clip_leg_qpos(initial_leg_qpos)
        ref = None if ref_leg_qpos is None else self._clip_leg_qpos(ref_leg_qpos)

        for seed in (initial, ref):
            if seed is not None and not self._seed_is_duplicate(seed, seeds):
                seeds.append(seed)

        if initial is not None:
            base = initial
        elif ref is not None:
            base = ref
        else:
            base = 0.5 * (self.leg_limits[:, 0] + self.leg_limits[:, 1])
            base = self._clip_leg_qpos(base)

        joint_span = self.leg_limits[:, 1] - self.leg_limits[:, 0]
        local_span = np.minimum(self.multiseed_seed_span, 0.25 * joint_span)
        for offset_unit in self._halton_seed_offsets:
            candidate = self._clip_leg_qpos(base + offset_unit * local_span)
            if not self._seed_is_duplicate(candidate, seeds):
                seeds.append(candidate)
            if len(seeds) >= n_candidates:
                break

        if initial is not None and ref is not None:
            for alpha in (0.50, 0.75, 0.25):
                if len(seeds) >= n_candidates:
                    break
                blended = self._clip_leg_qpos(alpha * initial + (1.0 - alpha) * ref)
                if not self._seed_is_duplicate(blended, seeds):
                    seeds.append(blended)

        return seeds[: n_candidates]

    def _ik_result_cost(
        self,
        result: GhostIKResult,
        initial_leg_qpos: np.ndarray = None,
        ref_leg_qpos: np.ndarray = None,
        initial_root_pos: np.ndarray = None,
        root_continuity_weight: float = 0.0,
    ) -> float:
        """Stable top-k style scalar score for candidate IK results."""
        pos_err = max(float(result.left_pos_err), float(result.right_pos_err))
        rot_err = max(float(result.left_rot_err), float(result.right_rot_err))
        root_offset = 0.0
        if result.root_pos_offset is not None:
            root_offset = float(np.linalg.norm(result.root_pos_offset))

        cost = pos_err * pos_err + 0.10 * rot_err * rot_err + 0.25 * root_offset * root_offset
        if initial_leg_qpos is not None:
            dq = np.asarray(result.leg_qpos, dtype=np.float64) - np.asarray(initial_leg_qpos, dtype=np.float64)
            max_dq = float(np.max(np.abs(dq)))
            rms_dq = float(np.linalg.norm(dq) / np.sqrt(max(len(dq), 1)))
            excess = max(max_dq - self.multiseed_trigger_joint_step, 0.0)
            cost += self.multiseed_continuity_weight * rms_dq * rms_dq + 5.0 * excess * excess
        if initial_root_pos is not None and root_continuity_weight > 0.0:
            root_dev = np.asarray(result.qpos[:3], dtype=np.float64) - np.asarray(initial_root_pos, dtype=np.float64)
            cost += float(root_continuity_weight) * float(np.dot(root_dev, root_dev))
        if ref_leg_qpos is not None:
            dev = np.asarray(result.leg_qpos, dtype=np.float64) - np.asarray(ref_leg_qpos, dtype=np.float64)
            rms_ref = float(np.linalg.norm(dev) / np.sqrt(max(len(dev), 1)))
            cost += self.multiseed_posture_weight * rms_ref * rms_ref
        if not np.isfinite(cost):
            return 1e16
        return float(cost)

    def _needs_multiseed(self, result: GhostIKResult, initial_leg_qpos: np.ndarray = None) -> bool:
        if not self.multiseed_enabled or self.multiseed_num_seeds <= 1:
            return False
        pos_err = max(float(result.left_pos_err), float(result.right_pos_err))
        pos_trigger = min(self.multiseed_trigger_pos_err, 0.04)
        if pos_err > pos_trigger:
            return True
        if result.root_pos_offset is not None and pos_err > 0.015:
            if float(np.linalg.norm(result.root_pos_offset)) > 0.035:
                return True
        if initial_leg_qpos is not None:
            max_dq = float(np.max(np.abs(result.leg_qpos - np.asarray(initial_leg_qpos, dtype=np.float64))))
            if max_dq > self.multiseed_trigger_joint_step:
                return True
        return False

    def _resolve_point_weights(self, point_scale: np.ndarray = None) -> np.ndarray:
        """Combine static foot weights with optional per-call support scaling."""
        if point_scale is None:
            return self.point_weights
        point_scale = np.asarray(point_scale, dtype=np.float64)
        if point_scale.shape == (3,):
            point_scale = np.repeat(point_scale, 3)
        if point_scale.shape != self.point_weights.shape:
            raise ValueError(
                f"point_scale must have shape (3,) or {self.point_weights.shape}, "
                f"got {point_scale.shape}"
            )
        return self.point_weights * np.maximum(point_scale, 0.0)

    def _foot_multipoint_targets(
        self, target_pos: np.ndarray, target_quat: np.ndarray,
    ) -> tuple:
        """Compute world positions for ankle, toe, and heel from target foot pose.

        Returns (ankle_pos, toe_pos, heel_pos) each as (3,) arrays.
        """
        # Build rotation matrix from target quaternion
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, target_quat)
        R = rot.reshape(3, 3)
        toe_world = target_pos + R @ TOE_OFFSET
        heel_world = target_pos + R @ HEEL_OFFSET
        return np.asarray(target_pos, dtype=np.float64), toe_world, heel_world

    def _foot_multipoint_current(
        self, data: mujoco.MjData, body_id: int,
    ) -> tuple:
        """Get current world positions for ankle, toe, and heel from MuJoCo state.

        Returns (ankle_pos, toe_pos, heel_pos) each as (3,) arrays.
        """
        ankle_pos = data.xpos[body_id].copy()
        R = data.xmat[body_id].reshape(3, 3)
        toe_world = ankle_pos + R @ TOE_OFFSET
        heel_world = ankle_pos + R @ HEEL_OFFSET
        return ankle_pos, toe_world, heel_world

    def _foot_multipoint_jac(
        self,
        data: mujoco.MjData,
        body_id: int,
        dof_adr: np.ndarray,
        jacp_ankle: np.ndarray,
        jacp_toe: np.ndarray,
        jacp_heel: np.ndarray,
    ) -> np.ndarray:
        """Compute stacked Jacobian (9 x n_dof) for ankle + toe + heel.

        Uses mj_jac with offset points for toe and heel.
        """
        # Ankle Jacobian (body origin)
        _jacr_dummy = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, data, jacp_ankle, _jacr_dummy, body_id)

        # Toe point in world frame
        R = data.xmat[body_id].reshape(3, 3)
        ankle_pos = data.xpos[body_id]

        self._point_buf[:] = ankle_pos + R @ TOE_OFFSET
        mujoco.mj_jac(self.model, data, jacp_toe, None, self._point_buf, body_id)

        self._point_buf[:] = ankle_pos + R @ HEEL_OFFSET
        mujoco.mj_jac(self.model, data, jacp_heel, None, self._point_buf, body_id)

        return np.vstack([
            jacp_ankle[:, dof_adr],
            jacp_toe[:, dof_adr],
            jacp_heel[:, dof_adr],
        ])

    def _ankle_rot_error_and_jac(
        self,
        data: mujoco.MjData,
        body_id: int,
        target_quat: np.ndarray,
        dof_adr: np.ndarray,
    ) -> tuple:
        """Compute weighted ankle orientation error (3,) and Jacobian (3, n_dof).

        Uses quaternion error → axis-angle (3D), scaled by ankle_rot_weight.
        Returns empty arrays if ankle_rot_weight <= 0.
        """
        if self.ankle_rot_weight <= 0.0:
            return np.zeros(0), np.zeros((0, len(dof_adr)))

        w = self.ankle_rot_weight
        # Quaternion error: q_err = q_target * conj(q_current)
        cur_quat = data.xquat[body_id]
        neg_cur = np.zeros(4, dtype=np.float64)
        mujoco.mju_negQuat(neg_cur, cur_quat)
        q_err = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(q_err, target_quat, neg_cur)
        # axis-angle (3D)
        rot_err = np.zeros(3, dtype=np.float64)
        mujoco.mju_quat2Vel(rot_err, q_err, 1.0)

        # Rotational Jacobian
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        jacp_dummy = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, data, jacp_dummy, jacr, body_id)

        return rot_err * w, jacr[:, dof_adr] * w

    def _apply_penetration_weights(self, err, jac):
        """Apply asymmetric weighting: upweight z-error for points below target.

        For each 3D point in the error vector, if err_z > 0 (current z is below
        target z — the foot needs to move UP), scale the z component by
        penetration_weight. This makes IK strongly avoid configurations where
        toe/heel is underground.

        Args:
            err: (M,) error vector where M is a multiple of 3 (N points × 3D).
            jac: (M, n_dof) Jacobian matrix.

        Returns:
            weighted_err: (M,) modified error.
            weighted_jac: (M, n_dof) modified Jacobian.
        """
        if self.penetration_weight <= 1.0:
            return err, jac
        w_err = err.copy()
        w_jac = jac.copy()
        n_points = len(err) // 3
        for i in range(n_points):
            z_idx = i * 3 + 2  # z component of the i-th point
            if err[z_idx] > 0:  # current z is below target z → penalize
                w_err[z_idx] *= self.penetration_weight
                w_jac[z_idx] *= self.penetration_weight
        return w_err, w_jac

    def _knee_forward_error_and_jac(
        self, data: mujoco.MjData, dof_adr: np.ndarray,
    ) -> tuple:
        """Penalize leg pointing backward: knee must not be behind the hip.

        Computes the vector from hip to knee in the pelvis local frame.
        In normal stance, the knee is below and slightly behind the hip
        (local x ~ -0.08). When the leg flips backward, the knee goes
        far behind (local x << -0.15). We penalize when the hip-to-knee
        vector projected onto pelvis local -z (downward) is too small,
        i.e., the knee is not sufficiently below the hip.

        More precisely: in pelvis local frame, the knee should be mostly
        downward (-z). We penalize when the knee's local x component is
        too negative (too far behind).

        Returns:
            err: (N,) penalty errors
            jac: (N, n_dof) corresponding Jacobian rows
        """
        if self.knee_forward_weight <= 0.0:
            return np.zeros(0), np.zeros((0, len(dof_adr)))
        if self.pelvis_body_id < 0:
            return np.zeros(0), np.zeros((0, len(dof_adr)))

        R_pelvis = data.xmat[self.pelvis_body_id].reshape(3, 3)
        pelvis_pos = data.xpos[self.pelvis_body_id]
        # Pelvis forward (x) and down (-z) directions in world frame
        fwd = R_pelvis[:, 0]  # pelvis x-axis = forward

        err_list = []
        jac_list = []
        n_dof = len(dof_adr)

        for knee_bid in [self.left_knee_body_id, self.right_knee_body_id]:
            if knee_bid < 0:
                continue
            knee_pos = data.xpos[knee_bid]
            d = knee_pos - pelvis_pos
            # Project onto pelvis forward direction
            fwd_proj = np.dot(d, fwd)

            # In normal G1 stance, knee local x ≈ -0.08 (slightly behind).
            # When leg flips backward, fwd_proj << -0.15.
            # Penalize when fwd_proj < threshold (knee is too far behind).
            threshold = -0.12  # allow slightly behind, penalize extreme
            if fwd_proj < threshold:
                w = self.knee_forward_weight
                err_val = (threshold - fwd_proj) * w
                err_list.append(err_val)

                # Jacobian: d(fwd_proj)/dq = fwd^T @ J_knee
                jacp_knee = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacBody(self.model, data, jacp_knee, None, knee_bid)
                jac_row = (fwd @ jacp_knee[:, dof_adr]) * w
                jac_list.append(jac_row.reshape(1, n_dof))

        if not err_list:
            return np.zeros(0), np.zeros((0, n_dof))

        return np.array(err_list), np.vstack(jac_list)

    def _compute_ref_body_poses(
        self,
        data: mujoco.MjData,
        target_root_pos: np.ndarray,
        target_root_quat: np.ndarray,
        fixed_upper_body_qpos: np.ndarray,
        ref_leg_qpos: np.ndarray,
    ) -> dict:
        """Compute target body poses by FK with reference leg joints.

        Sets up the reference data with root + upper-body + reference leg joints,
        does mj_forward, and stores xpos/xquat for each constrained body.

        Returns dict mapping body_id → (target_pos(3,), target_quat(4,)).
        """
        if not self.body_constraints:
            return {}
        rd = self._ref_data
        rd.qpos[:3] = target_root_pos
        rd.qpos[3:7] = target_root_quat
        rd.qpos[7:] = fixed_upper_body_qpos
        rd.qpos[self.leg_qpos_adr] = ref_leg_qpos
        mujoco.mj_forward(self.model, rd)
        targets = {}
        for (bid, pw, rw, _name) in self.body_constraints:
            targets[bid] = (rd.xpos[bid].copy(), rd.xquat[bid].copy())
        return targets

    def _body_constraint_errors_and_jac(
        self, data: mujoco.MjData, ref_poses: dict, dof_adr: np.ndarray,
    ) -> tuple:
        """Compute error vector and Jacobian for extra body constraints.

        For each body in self.body_constraints with non-zero pos/rot weight:
          - pos error: (target_pos - current_pos) * pos_weight  → 3D
          - rot error: quaternion error axis-angle * rot_weight  → 3D

        Returns:
            err: (M,) stacked error vector
            jac: (M, n_dof) stacked Jacobian
        """
        if not self.body_constraints or not ref_poses:
            return np.zeros(0), np.zeros((0, len(dof_adr)))

        err_list = []
        jac_list = []
        n_dof = len(dof_adr)

        for (bid, pw, rw, _name) in self.body_constraints:
            if bid not in ref_poses:
                continue
            tgt_pos, tgt_quat = ref_poses[bid]

            if pw > 0:
                pos_err = (tgt_pos - data.xpos[bid]) * pw
                mujoco.mj_jacBody(self.model, data, self._jacp_body, self._jacr_body, bid)
                jac_pos = self._jacp_body[:, dof_adr] * pw  # (3, n_dof)
                err_list.append(pos_err)
                jac_list.append(jac_pos)

            if rw > 0:
                # Quaternion error: q_err = q_target * conj(q_current)
                cur_quat = data.xquat[bid]
                mujoco.mju_negQuat(self._quat_err_buf, cur_quat)
                q_err = np.zeros(4, dtype=np.float64)
                mujoco.mju_mulQuat(q_err, tgt_quat, self._quat_err_buf)
                # Convert to axis-angle (3D rotation error)
                mujoco.mju_quat2Vel(self._rot_err_buf, q_err, 1.0)
                rot_err = self._rot_err_buf * rw

                mujoco.mj_jacBody(self.model, data, self._jacp_body, self._jacr_body, bid)
                jac_rot = self._jacr_body[:, dof_adr] * rw  # (3, n_dof)
                err_list.append(rot_err)
                jac_list.append(jac_rot)

        if not err_list:
            return np.zeros(0), np.zeros((0, n_dof))

        return np.concatenate(err_list), np.vstack(jac_list)

    def _restore_fixed_state(
        self,
        data: mujoco.MjData,
        target_root_pos: np.ndarray,
        target_root_quat: np.ndarray,
        fixed_joint_qpos: np.ndarray,
    ):
        data.qpos[:3] = target_root_pos
        data.qpos[3:7] = target_root_quat
        data.qpos[self.non_leg_qpos_adr] = fixed_joint_qpos[self.non_leg_local_idx]

    def _position_warmstart(
        self,
        data: mujoco.MjData,
        target_root_pos: np.ndarray,
        target_root_quat: np.ndarray,
        target_lf_pos: np.ndarray,
        target_rf_pos: np.ndarray,
        fixed_upper_body_qpos: np.ndarray,
        warmstart_iters: int = 10,
    ):
        """Run position-only IK (no orientation, no posture) to get a seed
        that places the feet close to the target positions. This prevents the
        posture regularization from anchoring the solution to joint angles that
        are far from the position-feasible configuration (e.g. flat-ground
        joints when the pelvis has been lifted for stairs)."""
        identity = np.eye(12, dtype=np.float64)
        warmstart_damping = 0.01
        for _ in range(warmstart_iters):
            mujoco.mj_forward(self.model, data)
            left_pos_err = np.asarray(target_lf_pos, dtype=np.float64) - data.xpos[self.left_foot_body_id]
            right_pos_err = np.asarray(target_rf_pos, dtype=np.float64) - data.xpos[self.right_foot_body_id]
            if max(np.linalg.norm(left_pos_err), np.linalg.norm(right_pos_err)) < self.tol:
                break

            mujoco.mj_jacBody(
                self.model, data, self._jacp_left, self._jacr_left, self.left_foot_body_id
            )
            mujoco.mj_jacBody(
                self.model, data, self._jacp_right, self._jacr_right, self.right_foot_body_id
            )
            jac = np.vstack(
                [
                    self._jacp_left[:, self.leg_dof_adr],
                    self._jacp_right[:, self.leg_dof_adr],
                ]
            )
            jac_t = jac.T
            pos_err = np.concatenate([left_pos_err, right_pos_err], axis=0)
            lhs = jac_t @ jac + (warmstart_damping ** 2) * identity
            delta_leg = np.linalg.solve(lhs, jac_t @ pos_err)
            delta_leg = np.clip(delta_leg, -self.max_joint_step, self.max_joint_step)
            delta_norm = float(np.linalg.norm(delta_leg))
            if delta_norm > 1.0:
                delta_leg *= 1.0 / delta_norm

            qvel = np.zeros(self.model.nv, dtype=np.float64)
            qvel[self.leg_dof_adr] = delta_leg
            mujoco.mj_integratePos(self.model, data.qpos, qvel, 1.0)
            data.qpos[self.leg_qpos_adr] = np.clip(
                data.qpos[self.leg_qpos_adr],
                self.leg_limits[:, 0],
                self.leg_limits[:, 1],
            )
            self._restore_fixed_state(
                data,
                np.asarray(target_root_pos, dtype=np.float64),
                np.asarray(target_root_quat, dtype=np.float64),
                fixed_upper_body_qpos,
            )

    def solve(
        self,
        data: mujoco.MjData,
        target_root_pos: np.ndarray,
        target_root_quat: np.ndarray,
        target_lf_pos: np.ndarray,
        target_lf_quat: np.ndarray,
        target_rf_pos: np.ndarray,
        target_rf_quat: np.ndarray,
        fixed_upper_body_qpos: np.ndarray,
        initial_leg_qpos: np.ndarray = None,
        ref_leg_qpos: np.ndarray = None,
        left_point_scale: np.ndarray = None,
        right_point_scale: np.ndarray = None,
    ) -> GhostIKResult:
        """Solve the ghost leg joints with MuJoCo Jacobians."""
        fixed_upper_body_qpos = np.asarray(fixed_upper_body_qpos, dtype=np.float64)
        if fixed_upper_body_qpos.shape[0] != (self.model.nq - 7):
            raise ValueError(
                f"fixed_upper_body_qpos must have length {self.model.nq - 7}, "
                f"got {fixed_upper_body_qpos.shape[0]}"
            )

        data.qpos[:3] = target_root_pos
        data.qpos[3:7] = target_root_quat
        data.qpos[7:] = fixed_upper_body_qpos
        if initial_leg_qpos is not None:
            data.qpos[self.leg_qpos_adr] = np.asarray(initial_leg_qpos, dtype=np.float64)
        data.qpos[self.leg_qpos_adr] = np.clip(
            data.qpos[self.leg_qpos_adr],
            self.leg_limits[:, 0],
            self.leg_limits[:, 1],
        )

        # Phase 1: position-only warm-start to escape flat-ground local minima.
        # When the pelvis is elevated for stairs, the initial joint angles from
        # flat-ground motion place the feet far from the targets. Running a few
        # position-only iterations (no orientation, no posture pull) moves the
        # joints to a position-feasible region first.
        mujoco.mj_forward(self.model, data)
        init_left_err = np.linalg.norm(
            np.asarray(target_lf_pos, dtype=np.float64) - data.xpos[self.left_foot_body_id]
        )
        init_right_err = np.linalg.norm(
            np.asarray(target_rf_pos, dtype=np.float64) - data.xpos[self.right_foot_body_id]
        )
        warmstart_thresh = 0.02
        if initial_leg_qpos is None and max(init_left_err, init_right_err) > warmstart_thresh:
            self._position_warmstart(
                data, target_root_pos, target_root_quat,
                target_lf_pos, target_rf_pos, fixed_upper_body_qpos,
            )

        # Posture seed: prefer ref_leg_qpos (original motion-clip joints)
        if ref_leg_qpos is not None:
            seed_leg_qpos = np.asarray(ref_leg_qpos, dtype=np.float64).copy()
        elif initial_leg_qpos is not None:
            seed_leg_qpos = np.asarray(initial_leg_qpos, dtype=np.float64).copy()
        else:
            seed_leg_qpos = self.extract_leg_qpos(data.qpos)

        identity = np.eye(12, dtype=np.float64)

        # Pre-compute multi-point target positions for each foot
        target_lf_pos = np.asarray(target_lf_pos, dtype=np.float64)
        target_rf_pos = np.asarray(target_rf_pos, dtype=np.float64)
        target_lf_quat = np.asarray(target_lf_quat, dtype=np.float64)
        target_rf_quat = np.asarray(target_rf_quat, dtype=np.float64)
        lf_point_weights = self._resolve_point_weights(left_point_scale)
        rf_point_weights = self._resolve_point_weights(right_point_scale)
        lf_tgt_ankle, lf_tgt_toe, lf_tgt_heel = self._foot_multipoint_targets(
            target_lf_pos, target_lf_quat,
        )
        rf_tgt_ankle, rf_tgt_toe, rf_tgt_heel = self._foot_multipoint_targets(
            target_rf_pos, target_rf_quat,
        )

        # Pre-compute reference body poses for extra body constraints
        ref_body_poses = self._compute_ref_body_poses(
            data, np.asarray(target_root_pos), np.asarray(target_root_quat),
            fixed_upper_body_qpos, seed_leg_qpos,
        )

        last_left_pos_err = 0.0
        last_right_pos_err = 0.0
        last_left_rot_err = 0.0
        last_right_rot_err = 0.0
        converged = False
        it_used = self.max_iters

        for it in range(self.max_iters):
            mujoco.mj_forward(self.model, data)

            # Current multi-point positions
            lf_cur_ankle, lf_cur_toe, lf_cur_heel = self._foot_multipoint_current(
                data, self.left_foot_body_id,
            )
            rf_cur_ankle, rf_cur_toe, rf_cur_heel = self._foot_multipoint_current(
                data, self.right_foot_body_id,
            )

            # Multi-point errors (9D per foot)
            lf_err = np.concatenate([
                lf_tgt_ankle - lf_cur_ankle,
                lf_tgt_toe - lf_cur_toe,
                lf_tgt_heel - lf_cur_heel,
            ])
            rf_err = np.concatenate([
                rf_tgt_ankle - rf_cur_ankle,
                rf_tgt_toe - rf_cur_toe,
                rf_tgt_heel - rf_cur_heel,
            ])
            lf_raw_err = lf_err.copy()
            rf_raw_err = rf_err.copy()

            # Apply per-point weights (ankle/toe/heel scaling)
            lf_err *= lf_point_weights
            rf_err *= rf_point_weights

            weighted_err = np.concatenate([lf_err, rf_err])

            # Extra body constraints (hip/knee pos+rot)
            body_err, body_jac = self._body_constraint_errors_and_jac(
                data, ref_body_poses, self.leg_dof_adr,
            )
            if len(body_err) > 0:
                weighted_err = np.concatenate([weighted_err, body_err])

            err_norm = float(np.linalg.norm(weighted_err))
            last_left_pos_err = float(np.linalg.norm(lf_raw_err[:3]))
            last_right_pos_err = float(np.linalg.norm(rf_raw_err[:3]))
            last_left_rot_err = float(np.linalg.norm(lf_raw_err[3:]))
            last_right_rot_err = float(np.linalg.norm(rf_raw_err[3:]))
            if err_norm < self.tol:
                converged = True
                it_used = it + 1
                break

            # Multi-point Jacobians (9 x n_dof per foot)
            lf_jac = self._foot_multipoint_jac(
                data, self.left_foot_body_id, self.leg_dof_adr,
                self._jacp_left, self._jacp_left_toe, self._jacp_left_heel,
            )
            rf_jac = self._foot_multipoint_jac(
                data, self.right_foot_body_id, self.leg_dof_adr,
                self._jacp_right, self._jacp_right_toe, self._jacp_right_heel,
            )
            # Apply same per-point weights to Jacobian rows
            lf_jac *= lf_point_weights[:, None]
            rf_jac *= rf_point_weights[:, None]
            jac = np.vstack([lf_jac, rf_jac])  # (18, 12)

            # Append body constraint Jacobian rows
            if len(body_err) > 0:
                jac = np.vstack([jac, body_jac])

            # Knee-forward penalty
            knee_err, knee_jac = self._knee_forward_error_and_jac(data, self.leg_dof_adr)
            if len(knee_err) > 0:
                weighted_err = np.concatenate([weighted_err, knee_err])
                jac = np.vstack([jac, knee_jac])

            # Ankle orientation constraint (explicit rotation tracking)
            for _bid, _tgt_quat in [
                (self.left_foot_body_id, target_lf_quat),
                (self.right_foot_body_id, target_rf_quat),
            ]:
                _rot_err, _rot_jac = self._ankle_rot_error_and_jac(
                    data, _bid, _tgt_quat, self.leg_dof_adr,
                )
                if len(_rot_err) > 0:
                    weighted_err = np.concatenate([weighted_err, _rot_err])
                    jac = np.vstack([jac, _rot_jac])

            # Asymmetric weighting: penalize points below target z (penetration)
            # Only apply to the first 18 rows (foot multi-point), not body constraints
            foot_err = weighted_err[:18]
            foot_jac = jac[:18]
            foot_err, foot_jac = self._apply_penetration_weights(foot_err, foot_jac)
            weighted_err[:18] = foot_err
            jac[:18] = foot_jac

            jac_t = jac.T
            cur_leg_qpos = self.extract_leg_qpos(data.qpos)
            # Null-space posture regularization keeps the solution on a stable branch
            # near the position-converged seed (from warm-start phase).
            lhs = jac_t @ jac + ((self.damping ** 2) + self.posture_weight) * identity
            rhs = jac_t @ weighted_err + self.posture_weight * (seed_leg_qpos - cur_leg_qpos)
            delta_leg = np.linalg.solve(lhs, rhs)
            delta_leg = np.clip(delta_leg, -self.max_joint_step, self.max_joint_step)

            delta_norm = float(np.linalg.norm(delta_leg))
            if delta_norm > 1.0:
                delta_leg *= 1.0 / delta_norm

            qvel = np.zeros(self.model.nv, dtype=np.float64)
            qvel[self.leg_dof_adr] = self.step_size * delta_leg
            mujoco.mj_integratePos(self.model, data.qpos, qvel, 1.0)
            data.qpos[self.leg_qpos_adr] = np.clip(
                data.qpos[self.leg_qpos_adr],
                self.leg_limits[:, 0],
                self.leg_limits[:, 1],
            )
            self._restore_fixed_state(
                data,
                np.asarray(target_root_pos, dtype=np.float64),
                np.asarray(target_root_quat, dtype=np.float64),
                fixed_upper_body_qpos,
            )

        mujoco.mj_forward(self.model, data)
        return GhostIKResult(
            qpos=data.qpos.copy(),
            leg_qpos=self.extract_leg_qpos(data.qpos),
            left_pos_err=last_left_pos_err,
            right_pos_err=last_right_pos_err,
            left_rot_err=last_left_rot_err,
            right_rot_err=last_right_rot_err,
            converged=converged,
            iterations=it_used,
        )

    def solve_with_root(
        self,
        data: mujoco.MjData,
        target_root_pos: np.ndarray,
        target_root_quat: np.ndarray,
        target_lf_pos: np.ndarray,
        target_lf_quat: np.ndarray,
        target_rf_pos: np.ndarray,
        target_rf_quat: np.ndarray,
        fixed_upper_body_qpos: np.ndarray,
        initial_leg_qpos: np.ndarray = None,
        ref_leg_qpos: np.ndarray = None,
        root_xy_weight: float = 2.0,
        root_z_weight: float = 10.0,
        left_point_scale: np.ndarray = None,
        right_point_scale: np.ndarray = None,
        allow_multiseed: bool = True,
        force_multiseed: bool = False,
        initial_root_pos: np.ndarray = None,
        root_continuity_weight: float = 0.0,
    ) -> GhostIKResult:
        """Solve IK with cuRobo-style adaptive multi-seed fallback.

        The fast path remains differential IK from the previous frame seed.  If
        that candidate has high pose error or a large joint jump, run a small
        deterministic seed bank and select the lowest-cost result.
        """
        primary = self._solve_with_root_single(
            data,
            target_root_pos=target_root_pos,
            target_root_quat=target_root_quat,
            target_lf_pos=target_lf_pos,
            target_lf_quat=target_lf_quat,
            target_rf_pos=target_rf_pos,
            target_rf_quat=target_rf_quat,
            fixed_upper_body_qpos=fixed_upper_body_qpos,
            initial_leg_qpos=initial_leg_qpos,
            ref_leg_qpos=ref_leg_qpos,
            root_xy_weight=root_xy_weight,
            root_z_weight=root_z_weight,
            left_point_scale=left_point_scale,
            right_point_scale=right_point_scale,
        )
        primary.cost = self._ik_result_cost(
            primary,
            initial_leg_qpos,
            ref_leg_qpos,
            initial_root_pos=initial_root_pos,
            root_continuity_weight=root_continuity_weight,
        )

        need_multiseed = force_multiseed or self._needs_multiseed(primary, initial_leg_qpos)
        if (not allow_multiseed) or (not need_multiseed):
            data.qpos[:] = primary.qpos
            mujoco.mj_forward(self.model, data)
            return primary

        best = primary
        best_cost = primary.cost
        primary_seed = None if initial_leg_qpos is None else np.asarray(initial_leg_qpos, dtype=np.float64)
        if initial_leg_qpos is not None:
            candidate = self._solve_with_root_single(
                data,
                target_root_pos=target_root_pos,
                target_root_quat=target_root_quat,
                target_lf_pos=target_lf_pos,
                target_lf_quat=target_lf_quat,
                target_rf_pos=target_rf_pos,
                target_rf_quat=target_rf_quat,
                fixed_upper_body_qpos=fixed_upper_body_qpos,
                initial_leg_qpos=None,
                ref_leg_qpos=ref_leg_qpos,
                root_xy_weight=root_xy_weight,
                root_z_weight=root_z_weight,
                left_point_scale=left_point_scale,
                right_point_scale=right_point_scale,
            )
            candidate.cost = self._ik_result_cost(
                candidate,
                initial_leg_qpos,
                ref_leg_qpos,
                initial_root_pos=initial_root_pos,
                root_continuity_weight=root_continuity_weight,
            )
            candidate.seed_index = -1
            candidate.used_multiseed = True
            if candidate.cost < best_cost:
                best = candidate
                best_cost = candidate.cost
        for seed_idx, seed in enumerate(self._candidate_leg_seeds(initial_leg_qpos, ref_leg_qpos)):
            if primary_seed is not None and float(np.max(np.abs(seed - primary_seed))) < 1e-5:
                continue
            candidate = self._solve_with_root_single(
                data,
                target_root_pos=target_root_pos,
                target_root_quat=target_root_quat,
                target_lf_pos=target_lf_pos,
                target_lf_quat=target_lf_quat,
                target_rf_pos=target_rf_pos,
                target_rf_quat=target_rf_quat,
                fixed_upper_body_qpos=fixed_upper_body_qpos,
                initial_leg_qpos=seed,
                ref_leg_qpos=ref_leg_qpos,
                root_xy_weight=root_xy_weight,
                root_z_weight=root_z_weight,
                left_point_scale=left_point_scale,
                right_point_scale=right_point_scale,
            )
            candidate.cost = self._ik_result_cost(
                candidate,
                initial_leg_qpos,
                ref_leg_qpos,
                initial_root_pos=initial_root_pos,
                root_continuity_weight=root_continuity_weight,
            )
            candidate.seed_index = seed_idx + 1
            candidate.used_multiseed = True
            if candidate.cost < best_cost:
                best = candidate
                best_cost = candidate.cost

        best.cost = best_cost
        best.used_multiseed = True
        data.qpos[:] = best.qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        return best

    def _solve_with_root_single(
        self,
        data: mujoco.MjData,
        target_root_pos: np.ndarray,
        target_root_quat: np.ndarray,
        target_lf_pos: np.ndarray,
        target_lf_quat: np.ndarray,
        target_rf_pos: np.ndarray,
        target_rf_quat: np.ndarray,
        fixed_upper_body_qpos: np.ndarray,
        initial_leg_qpos: np.ndarray = None,
        ref_leg_qpos: np.ndarray = None,
        root_xy_weight: float = 2.0,
        root_z_weight: float = 10.0,
        left_point_scale: np.ndarray = None,
        right_point_scale: np.ndarray = None,
    ) -> GhostIKResult:
        """Solve legs + root xyz together — feet-first, root follows smoothly.

        The solver prioritizes reaching foot targets with natural joint
        configurations. The root translates as needed to keep feet reachable.

        Key improvements over basic IK:
        - Decaying step size: large early steps → fine late adjustments
        - Soft joint limits: extra damping as joints approach range boundaries
        - Separate root/leg delta clamping: prevents leg joints from jumping
        - Higher posture regularization: keeps joints close to motion-clip poses

        Args:
            initial_leg_qpos: Warm-start seed (e.g., previous frame IK result).
            ref_leg_qpos: Posture regularization target (e.g., original motion-clip
                joints). If None, falls back to initial_leg_qpos or warmstart output.
            root_xy_weight: Moderate (1-5), keeps pelvis tracking forward direction.
            root_z_weight:  Higher (5-20), prevents pelvis from sinking into ground.
        """
        fixed_upper_body_qpos = np.asarray(fixed_upper_body_qpos, dtype=np.float64)
        target_root_pos = np.asarray(target_root_pos, dtype=np.float64)
        target_root_quat = np.asarray(target_root_quat, dtype=np.float64)
        target_lf_pos = np.asarray(target_lf_pos, dtype=np.float64)
        target_rf_pos = np.asarray(target_rf_pos, dtype=np.float64)
        target_lf_quat = np.asarray(target_lf_quat, dtype=np.float64)
        target_rf_quat = np.asarray(target_rf_quat, dtype=np.float64)
        lf_point_weights = self._resolve_point_weights(left_point_scale)
        rf_point_weights = self._resolve_point_weights(right_point_scale)

        root_dof_adr = np.array([0, 1, 2], dtype=np.int32)
        root_weights = np.array([root_xy_weight, root_xy_weight, root_z_weight],
                                dtype=np.float64)
        n_root = 3
        n_leg = len(self.leg_dof_adr)

        all_dof_adr = np.concatenate([root_dof_adr, self.leg_dof_adr])
        n_dof = len(all_dof_adr)

        # Initialize
        data.qpos[:3] = target_root_pos
        data.qpos[3:7] = target_root_quat
        data.qpos[7:] = fixed_upper_body_qpos
        if initial_leg_qpos is not None:
            data.qpos[self.leg_qpos_adr] = np.asarray(initial_leg_qpos, dtype=np.float64)
        data.qpos[self.leg_qpos_adr] = np.clip(
            data.qpos[self.leg_qpos_adr], self.leg_limits[:, 0], self.leg_limits[:, 1],
        )

        # Warmstart: position-only, root fixed.
        # Only use it when there is no previous-frame seed.  With a continuous
        # seed, the main solver can handle moderate errors; running the
        # unconstrained warmstart mid-trajectory can jump to the wrong knee
        # branch on stair transitions.
        mujoco.mj_forward(self.model, data)
        init_left_err = np.linalg.norm(target_lf_pos - data.xpos[self.left_foot_body_id])
        init_right_err = np.linalg.norm(target_rf_pos - data.xpos[self.right_foot_body_id])
        warmstart_thresh = 0.02
        if initial_leg_qpos is None and max(init_left_err, init_right_err) > warmstart_thresh:
            self._position_warmstart(
                data, target_root_pos, target_root_quat,
                target_lf_pos, target_rf_pos, fixed_upper_body_qpos,
            )

        # Posture seed: prefer ref_leg_qpos (original motion-clip joints)
        # to prevent locking onto a flipped warmstart solution.
        if ref_leg_qpos is not None:
            seed_leg_qpos = np.asarray(ref_leg_qpos, dtype=np.float64).copy()
        elif initial_leg_qpos is not None:
            seed_leg_qpos = np.asarray(initial_leg_qpos, dtype=np.float64).copy()
        else:
            seed_leg_qpos = self.extract_leg_qpos(data.qpos)
        identity = np.eye(n_dof, dtype=np.float64)

        # Pre-compute joint range info for soft limits
        leg_range = self.leg_limits[:, 1] - self.leg_limits[:, 0]  # (n_leg,)
        # Margin: within 10% of range boundary, start adding extra damping
        soft_limit_margin = 0.10 * leg_range

        # Pre-compute multi-point target positions for each foot
        # 3 points per foot: ankle, toe, heel → 9D per foot, 18D total
        lf_tgt_ankle, lf_tgt_toe, lf_tgt_heel = self._foot_multipoint_targets(
            target_lf_pos, target_lf_quat,
        )
        rf_tgt_ankle, rf_tgt_toe, rf_tgt_heel = self._foot_multipoint_targets(
            target_rf_pos, target_rf_quat,
        )

        # Pre-compute reference body poses for extra body constraints
        ref_body_poses = self._compute_ref_body_poses(
            data, target_root_pos, target_root_quat,
            fixed_upper_body_qpos, seed_leg_qpos,
        )

        last_left_pos_err = 0.0
        last_right_pos_err = 0.0
        last_left_rot_err = 0.0
        last_right_rot_err = 0.0
        converged = False
        it_used = self.max_iters

        for it in range(self.max_iters):
            mujoco.mj_forward(self.model, data)

            # Current multi-point positions
            lf_cur_ankle, lf_cur_toe, lf_cur_heel = self._foot_multipoint_current(
                data, self.left_foot_body_id,
            )
            rf_cur_ankle, rf_cur_toe, rf_cur_heel = self._foot_multipoint_current(
                data, self.right_foot_body_id,
            )

            # Multi-point errors (9D per foot)
            lf_err = np.concatenate([
                lf_tgt_ankle - lf_cur_ankle,
                lf_tgt_toe - lf_cur_toe,
                lf_tgt_heel - lf_cur_heel,
            ])
            rf_err = np.concatenate([
                rf_tgt_ankle - rf_cur_ankle,
                rf_tgt_toe - rf_cur_toe,
                rf_tgt_heel - rf_cur_heel,
            ])
            lf_raw_err = lf_err.copy()
            rf_raw_err = rf_err.copy()

            root_pos_err = target_root_pos[root_dof_adr] - data.qpos[root_dof_adr]

            # Apply per-point weights (ankle/toe/heel scaling)
            lf_err *= lf_point_weights
            rf_err *= rf_point_weights

            weighted_err = np.concatenate([lf_err, rf_err])

            # Extra body constraints (hip/knee pos+rot)
            body_err, body_jac = self._body_constraint_errors_and_jac(
                data, ref_body_poses, all_dof_adr,
            )
            if len(body_err) > 0:
                weighted_err = np.concatenate([weighted_err, body_err])

            err_norm = float(np.linalg.norm(weighted_err))
            last_left_pos_err = float(np.linalg.norm(lf_raw_err[:3]))  # ankle pos err
            last_right_pos_err = float(np.linalg.norm(rf_raw_err[:3]))
            # Rotation error approximated from toe/heel deviations
            last_left_rot_err = float(np.linalg.norm(lf_raw_err[3:]))
            last_right_rot_err = float(np.linalg.norm(rf_raw_err[3:]))
            if err_norm < self.tol:
                converged = True
                it_used = it + 1
                break

            # Multi-point Jacobians (9 x n_dof per foot)
            lf_jac = self._foot_multipoint_jac(
                data, self.left_foot_body_id, all_dof_adr,
                self._jacp_left, self._jacp_left_toe, self._jacp_left_heel,
            )
            rf_jac = self._foot_multipoint_jac(
                data, self.right_foot_body_id, all_dof_adr,
                self._jacp_right, self._jacp_right_toe, self._jacp_right_heel,
            )
            # Apply same per-point weights to Jacobian rows
            lf_jac *= lf_point_weights[:, None]
            rf_jac *= rf_point_weights[:, None]
            jac = np.vstack([lf_jac, rf_jac])  # (18, n_dof)

            # Append body constraint Jacobian rows
            if len(body_err) > 0:
                jac = np.vstack([jac, body_jac])

            # Knee-forward penalty
            knee_err, knee_jac = self._knee_forward_error_and_jac(data, all_dof_adr)
            if len(knee_err) > 0:
                weighted_err = np.concatenate([weighted_err, knee_err])
                jac = np.vstack([jac, knee_jac])

            # Ankle orientation constraint (explicit rotation tracking)
            for _bid, _tgt_quat in [
                (self.left_foot_body_id, target_lf_quat),
                (self.right_foot_body_id, target_rf_quat),
            ]:
                _rot_err, _rot_jac = self._ankle_rot_error_and_jac(
                    data, _bid, _tgt_quat, all_dof_adr,
                )
                if len(_rot_err) > 0:
                    weighted_err = np.concatenate([weighted_err, _rot_err])
                    jac = np.vstack([jac, _rot_jac])

            # Asymmetric weighting: penalize points below target z (penetration)
            # Only apply to the first 18 rows (foot multi-point), not body constraints
            foot_err = weighted_err[:18]
            foot_jac = jac[:18]
            foot_err, foot_jac = self._apply_penetration_weights(foot_err, foot_jac)
            weighted_err[:18] = foot_err
            jac[:18] = foot_jac

            jac_t = jac.T
            cur_leg_qpos = self.extract_leg_qpos(data.qpos)

            # --- Build LHS ---
            lhs = jac_t @ jac + (self.damping ** 2) * identity

            # Posture regularization for legs (higher = more natural poses)
            lhs[n_root:, n_root:] += self.posture_weight * np.eye(n_leg)

            # Root position regularization
            lhs[:n_root, :n_root] += np.diag(root_weights)

            # Soft joint limits: extra damping when joints are near boundaries.
            # This prevents joints from hitting hard limits and flipping to
            # unnatural configurations (e.g., knee hyperextension causing twist).
            dist_lo = cur_leg_qpos - self.leg_limits[:, 0]  # distance from lower limit
            dist_hi = self.leg_limits[:, 1] - cur_leg_qpos  # distance from upper limit
            dist_min = np.minimum(dist_lo, dist_hi)
            # Smooth activation: 0 when far from limit, increases near boundary
            soft_limit_penalty = np.where(
                dist_min < soft_limit_margin,
                self.damping * (1.0 - dist_min / np.maximum(soft_limit_margin, 1e-8)),
                0.0,
            )
            lhs[n_root:, n_root:] += np.diag(soft_limit_penalty)

            # --- Build RHS ---
            rhs = jac_t @ weighted_err
            rhs[n_root:] += self.posture_weight * (seed_leg_qpos - cur_leg_qpos)
            rhs[:n_root] += root_weights * root_pos_err

            delta = np.linalg.solve(lhs, rhs)

            # Decaying step size: large steps early, fine adjustments later.
            current_step_size = self.step_size / (1.0 + it * 0.15)

            # Separate clamping for root and leg deltas.
            delta[:n_root] = np.clip(delta[:n_root], -0.1, 0.1)
            # Total root offset clamp: prevent root from drifting far from
            # target across accumulated iterations. Allow a larger envelope
            # only when foot errors are already large; otherwise keep the
            # tighter 50mm bound to avoid gratuitous root chasing.
            prospective_root = data.qpos[:3] + current_step_size * delta[:n_root]
            root_offset = prospective_root - target_root_pos
            foot_err_mag = max(last_left_pos_err, last_right_pos_err)
            extra_root_drift = 0.05 * np.clip((foot_err_mag - 0.04) / 0.16, 0.0, 1.0)
            max_root_drift = 0.05 + extra_root_drift
            drift_norm = np.linalg.norm(root_offset)
            if drift_norm > max_root_drift:
                allowed = target_root_pos + root_offset * (max_root_drift / drift_norm)
                delta[:n_root] = (allowed - data.qpos[:3]) / max(current_step_size, 1e-8)

            leg_delta = delta[n_root:]
            leg_delta_norm = float(np.linalg.norm(leg_delta))
            if leg_delta_norm > self.max_joint_step:
                leg_delta *= self.max_joint_step / leg_delta_norm
            delta[n_root:] = leg_delta

            qvel = np.zeros(self.model.nv, dtype=np.float64)
            qvel[all_dof_adr] = current_step_size * delta
            mujoco.mj_integratePos(self.model, data.qpos, qvel, 1.0)

            # Clamp legs
            data.qpos[self.leg_qpos_adr] = np.clip(
                data.qpos[self.leg_qpos_adr], self.leg_limits[:, 0], self.leg_limits[:, 1],
            )
            # Restore root quat + upper body (root pos is free)
            data.qpos[3:7] = target_root_quat
            data.qpos[self.non_leg_qpos_adr] = fixed_upper_body_qpos[self.non_leg_local_idx]

        mujoco.mj_forward(self.model, data)
        root_offset = data.qpos[:3].copy() - target_root_pos
        return GhostIKResult(
            qpos=data.qpos.copy(),
            leg_qpos=self.extract_leg_qpos(data.qpos),
            left_pos_err=last_left_pos_err,
            right_pos_err=last_right_pos_err,
            left_rot_err=last_left_rot_err,
            right_rot_err=last_right_rot_err,
            converged=converged,
            iterations=it_used,
            root_pos_offset=root_offset,
        )

    # ------------------------------------------------------------------
    # Trajectory IK interface: stateless residual evaluation
    # ------------------------------------------------------------------

    def pack_state(self, root_pos: np.ndarray, leg_qpos: np.ndarray) -> np.ndarray:
        """Pack root_xyz (3) + leg_q (12) into a flat (15,) vector."""
        return np.concatenate([
            np.asarray(root_pos, dtype=np.float64)[:3],
            np.asarray(leg_qpos, dtype=np.float64)[:12],
        ])

    def unpack_state(self, x: np.ndarray) -> tuple:
        """Unpack (15,) → (root_pos (3,), leg_qpos (12,))."""
        x = np.asarray(x, dtype=np.float64)
        return x[:3].copy(), x[3:15].copy()

    def evaluate_frame_residual(
        self,
        data: mujoco.MjData,
        root_pos: np.ndarray,
        leg_qpos: np.ndarray,
        targets: "FrameIKTargets",
    ) -> dict:
        """Evaluate IK residual for a given state WITHOUT iterating.

        Sets qpos from (root_pos, targets.root_quat, targets.fixed_upper_body_qpos, leg_qpos),
        runs mj_forward, then computes all residual components matching solve_with_root's
        internal cost structure.

        Returns dict of named residual arrays (all weighted).
        """
        root_pos = np.asarray(root_pos, dtype=np.float64)
        leg_qpos = np.asarray(leg_qpos, dtype=np.float64)

        # Write state into MuJoCo
        data.qpos[:3] = root_pos
        data.qpos[3:7] = targets.root_quat
        data.qpos[7:] = targets.fixed_upper_body_qpos
        data.qpos[self.leg_qpos_adr] = np.clip(
            leg_qpos, self.leg_limits[:, 0], self.leg_limits[:, 1])
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)

        # --- Foot multi-point tracking ---
        lf_point_weights = self._resolve_point_weights(targets.left_point_scale)
        rf_point_weights = self._resolve_point_weights(targets.right_point_scale)

        lf_tgt_ankle, lf_tgt_toe, lf_tgt_heel = self._foot_multipoint_targets(
            targets.left_pos, targets.left_quat)
        rf_tgt_ankle, rf_tgt_toe, rf_tgt_heel = self._foot_multipoint_targets(
            targets.right_pos, targets.right_quat)

        lf_cur_ankle, lf_cur_toe, lf_cur_heel = self._foot_multipoint_current(
            data, self.left_foot_body_id)
        rf_cur_ankle, rf_cur_toe, rf_cur_heel = self._foot_multipoint_current(
            data, self.right_foot_body_id)

        lf_err = np.concatenate([
            lf_tgt_ankle - lf_cur_ankle,
            lf_tgt_toe - lf_cur_toe,
            lf_tgt_heel - lf_cur_heel,
        ]) * lf_point_weights
        rf_err = np.concatenate([
            rf_tgt_ankle - rf_cur_ankle,
            rf_tgt_toe - rf_cur_toe,
            rf_tgt_heel - rf_cur_heel,
        ]) * rf_point_weights
        foot_track = np.concatenate([lf_err, rf_err])

        # --- Foot rotation (ankle orientation) ---
        all_dof_adr = np.concatenate([np.array([0, 1, 2], dtype=np.int32), self.leg_dof_adr])
        rot_residuals = []
        for _bid, _tgt_quat in [
            (self.left_foot_body_id, targets.left_quat),
            (self.right_foot_body_id, targets.right_quat),
        ]:
            _rot_err, _ = self._ankle_rot_error_and_jac(data, _bid, _tgt_quat, all_dof_adr)
            if len(_rot_err) > 0:
                rot_residuals.append(_rot_err)
        foot_rot = np.concatenate(rot_residuals) if rot_residuals else np.zeros(0)

        # --- Root tracking ---
        root_weights = np.array([
            targets.root_xy_weight, targets.root_xy_weight, targets.root_z_weight
        ], dtype=np.float64)
        root_track = np.sqrt(root_weights) * (targets.root_pos[:3] - data.qpos[:3])

        # --- Posture regularization ---
        seed = targets.ref_leg_qpos if targets.ref_leg_qpos is not None else np.zeros(12)
        cur_leg = data.qpos[self.leg_qpos_adr]
        posture = np.sqrt(self.posture_weight) * (seed - cur_leg)

        # --- Knee forward penalty (fixed size: always 2 values for L/R) ---
        knee_err_raw, _ = self._knee_forward_error_and_jac(data, all_dof_adr)
        # Pad to fixed size 2 (left, right) for consistent residual length
        knee_err = np.zeros(2, dtype=np.float64)
        if len(knee_err_raw) > 0:
            knee_err[:len(knee_err_raw)] = knee_err_raw[:2]

        # --- Body constraints ---
        ref_body_poses = self._compute_ref_body_poses(
            data, targets.root_pos, targets.root_quat,
            targets.fixed_upper_body_qpos,
            seed,
        )
        body_err, _ = self._body_constraint_errors_and_jac(data, ref_body_poses, all_dof_adr)

        return {
            "foot_track": foot_track,
            "foot_rot": foot_rot,
            "root_track": root_track,
            "posture": posture,
            "knee_forward": knee_err,
            "body_constraints": body_err,
        }
