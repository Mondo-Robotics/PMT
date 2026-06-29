"""
Minimal MPPI demo for stair terrain.

Keeps only the essentials:
  1. load a motion clip from .npz
  2. build a simple terrain-lifted reference trajectory
  3. run a tiny MPPI planner over time-varying z-residual profiles
  4. solve main-robot IK to track current MPPI pelvis/foot targets
  5. show a ghost robot for terrain-lifted reference pose

This script is intentionally simple. It does not include:
  - high-fidelity contact-state estimation
  - detailed debug and diagnostic utilities

Usage:
    python -m stair_mppi.minimal_mppi_demo
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline

from stair_mppi.ghost_ik import G1GhostJacobianIK
from stair_mppi.terrain import StairTerrain
from stair_mppi.terrain import RaycastTerrain

JOINT_NAMES_POLICY_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

NPZ_PELVIS = 0
NPZ_LFOOT = 18
NPZ_RFOOT = 19
CONTACT_Z_THRESHOLD = 0.06
CONTACT_SPEED_THRESHOLD = 0.5
TOE_OFFSET_X = 0.14

# Body names in the motion npz order (BeyondMimic convention, interleaved L/R).
# MuJoCo uses kinematic chain order (left chain, right chain, waist, arms),
# so we need a mapping to reorder body data for export.
NPZ_BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
    "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]


def build_joint_mapping(model):
    mapping = []
    for name in JOINT_NAMES_POLICY_ORDER:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise ValueError(f"Actuator '{name}' not found")
        mapping.append(aid)
    return mapping


def build_body_mapping(model):
    """Build mapping from npz body index to MuJoCo body index (skipping world).

    Returns: list where mapping[npz_idx] = mujoco_body_id.
    """
    mapping = []
    for name in NPZ_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise ValueError(f"Body '{name}' not found in MuJoCo model")
        mapping.append(bid)
    return mapping


def reorder_to_mujoco(values, mapping):
    out = np.zeros_like(values)
    for policy_idx, mujoco_idx in enumerate(mapping):
        out[mujoco_idx] = values[policy_idx]
    return out


def reorder_from_mujoco(values, mapping):
    out = np.zeros_like(values)
    for policy_idx, mujoco_idx in enumerate(mapping):
        out[policy_idx] = values[mujoco_idx]
    return out


def quat_ang_vel(quats, fps):
    """Compute angular velocity from quaternion time series via finite differences.

    quats: (N, 4) in [w, x, y, z] convention.
    Returns: (N, 3) angular velocity in world frame.
    """
    n = quats.shape[0]
    omega = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return omega
    for i in range(1, n - 1):
        dq = quats[i + 1] - quats[i - 1]
        q_conj = quats[i].copy()
        q_conj[1:] *= -1.0
        # omega = 2 * dq * conj(q) * fps/2 = dq * conj(q) * fps
        # quaternion multiply: p*q
        w0, x0, y0, z0 = dq
        w1, x1, y1, z1 = q_conj
        rw = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        rx = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        ry = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        rz = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        omega[i] = np.array([rx, ry, rz]) * fps
    omega[0] = omega[1]
    omega[-1] = omega[-2]
    return omega


def _quat_rotate(quat_wxyz, vec):
    """Rotate a 3D vector by a single quaternion [w, x, y, z]."""
    w, x, y, z = quat_wxyz
    xyz = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(xyz, vec)
    return vec + w * t + np.cross(xyz, t)


def quat_rotate_batch(quats_wxyz, vec):
    """Rotate a 3D vector by N quaternions [w, x, y, z].

    Args:
        quats_wxyz: (N, 4) quaternions in [w, x, y, z] convention.
        vec: (3,) vector in local frame.

    Returns:
        (N, 3) rotated vectors in world frame.
    """
    w = quats_wxyz[:, 0:1]
    xyz = quats_wxyz[:, 1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + w * t + np.cross(xyz, t)


def add_sphere(scene, pos, rgba, size=0.008):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[size, 0.0, 0.0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_box(scene, pos, rgba, half_size):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.asarray(half_size, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_trail(scene, positions, rgba, size=0.006, stride=2):
    for pos in positions[::max(1, stride)]:
        add_sphere(scene, pos, rgba, size=size)


def add_current_markers(scene, pelvis, lfoot, rfoot):
    add_box(scene, pelvis, [0.0, 0.8, 1.0, 0.9], [0.014, 0.014, 0.014])
    add_box(scene, lfoot, [0.2, 0.5, 1.0, 0.9], [0.016, 0.016, 0.010])
    add_box(scene, rfoot, [0.0, 0.2, 0.8, 0.9], [0.016, 0.016, 0.010])


def parse_rgb(text):
    parts = [float(x.strip()) for x in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"RGB must have exactly 3 comma-separated values, got: {text}")
    rgb = np.asarray(parts, dtype=np.float64)
    if np.any((rgb < 0.0) | (rgb > 1.0)):
        raise ValueError(f"RGB values must be in [0, 1], got: {text}")
    return rgb


def update_robot_pose(data, root_pos, root_quat, joint_qpos):
    data.qpos[:3] = np.asarray(root_pos, dtype=np.float64)
    data.qpos[3:7] = np.asarray(root_quat, dtype=np.float64)
    data.qpos[7:] = np.asarray(joint_qpos, dtype=np.float64)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0


def stabilize_contact_mask(mask, min_contact_frames=4, min_swing_frames=2):
    out = np.asarray(mask, dtype=bool).copy()
    n = out.shape[0]
    if n == 0:
        return out

    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if out[i] and (j - i) < min_contact_frames:
            out[i:j] = False
        i = j

    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if (not out[i]) and (j - i) < min_swing_frames:
            left_true = (i > 0) and out[i - 1]
            right_true = (j < n) and out[j]
            if left_true and right_true:
                out[i:j] = True
        i = j

    return out


class MotionClip:
    def __init__(
        self,
        motion_path,
        start_frame=0,
        n_frames=300,
        contact_z_threshold=CONTACT_Z_THRESHOLD,
        contact_speed_threshold=CONTACT_SPEED_THRESHOLD,
    ):
        raw = np.load(motion_path)
        total_frames = raw["body_pos_w"].shape[0]
        if start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {start_frame}")
        if n_frames <= 0:
            raise ValueError(f"n_frames must be > 0, got {n_frames}")
        if start_frame >= total_frames:
            raise ValueError(
                f"start_frame ({start_frame}) is out of range for motion with {total_frames} frames"
            )

        end_frame = min(start_frame + n_frames, total_frames)
        if end_frame <= start_frame:
            raise ValueError(
                f"Empty motion segment: start_frame={start_frame}, end_frame={end_frame}"
            )
        segment = slice(start_frame, end_frame)

        self.fps = float(raw["fps"].item())
        self.n_frames = end_frame - start_frame
        self.joint_pos = raw["joint_pos"][segment].copy()
        self.body_pos = raw["body_pos_w"][segment].copy()
        self.body_quat = raw["body_quat_w"][segment].copy()
        if "body_lin_vel_w" in raw:
            self.body_lin_vel = raw["body_lin_vel_w"][segment].copy()
        else:
            self.body_lin_vel = np.gradient(self.body_pos, axis=0) * self.fps

        lfoot_speed = np.linalg.norm(self.body_lin_vel[:, NPZ_LFOOT], axis=1)
        rfoot_speed = np.linalg.norm(self.body_lin_vel[:, NPZ_RFOOT], axis=1)
        lfoot_z = self.body_pos[:, NPZ_LFOOT, 2]
        rfoot_z = self.body_pos[:, NPZ_RFOOT, 2]

        # Adaptive contact threshold: percentile-based.
        # Use the 25th percentile of foot z + margin as threshold.
        # This is robust to motions where the ground plane is not at z=0.
        all_foot_z = np.concatenate([lfoot_z, rfoot_z])
        _p25 = float(np.percentile(all_foot_z, 25.0))
        _adaptive_z_thresh = max(float(contact_z_threshold), _p25 + 0.02)
        left_contact_raw = (lfoot_z < _adaptive_z_thresh) & (
            lfoot_speed < float(contact_speed_threshold)
        )
        right_contact_raw = (rfoot_z < _adaptive_z_thresh) & (
            rfoot_speed < float(contact_speed_threshold)
        )
        self.left_contact = stabilize_contact_mask(left_contact_raw)
        self.right_contact = stabilize_contact_mask(right_contact_raw)

        # Phase clock fitting — for gaits with flight phases (running/hopping)
        from stair_mppi.gait_phase import fit_phase_clock, compute_per_frame_phase
        self.phase_params = fit_phase_clock(
            lfoot_z, rfoot_z, lfoot_speed, rfoot_speed,
            self.fps, contact_z_threshold, contact_speed_threshold,
        )
        if self.phase_params is not None:
            self.per_frame_phase = compute_per_frame_phase(
                self.phase_params, self.n_frames,
            )
            # Do NOT override contact masks — phase clock can misdetect boundaries.
            pass
        else:
            self.per_frame_phase = None

        contact_z_samples = []
        if np.any(self.left_contact):
            contact_z_samples.append(lfoot_z[self.left_contact])
        if np.any(self.right_contact):
            contact_z_samples.append(rfoot_z[self.right_contact])
        if contact_z_samples:
            _estimated = float(np.median(np.concatenate(contact_z_samples)))
        else:
            foot_z = np.concatenate([lfoot_z, rfoot_z])
            _estimated = float(np.percentile(foot_z, 10.0))
        # 实测 G1 ankle_roll_link 到脚底的高度为 0.06m，覆盖动捕估计值
        self.foot_nominal_z = 0.06

        # Compute nominal pelvis height above foot contact surface (flat ground).
        # This is the target for pelvis height regulation on terrain.
        pelvis_z = self.body_pos[:, NPZ_PELVIS, 2]
        self.pelvis_height_above_foot = float(np.median(pelvis_z) - self.foot_nominal_z)

    def get_pelvis_pos(self, frame):
        return self.body_pos[frame, NPZ_PELVIS].copy()

    def _future_indices(self, frame, horizon):
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        return (int(frame) + np.arange(int(horizon), dtype=np.int64)) % self.n_frames

    def get_future_indices(self, frame, horizon):
        return self._future_indices(frame, horizon)

    def get_future_segment(self, frame, horizon):
        idx = self._future_indices(frame, horizon)
        return (
            self.body_pos[idx, NPZ_PELVIS].copy(),
            self.body_pos[idx, NPZ_LFOOT].copy(),
            self.body_pos[idx, NPZ_RFOOT].copy(),
        )

    def get_future_contacts(self, frame, horizon):
        idx = self._future_indices(frame, horizon)
        return (
            self.left_contact[idx].copy(),
            self.right_contact[idx].copy(),
        )

    def get_future_support_weights(self, frame, horizon):
        """Return continuous support weights for pelvis z computation.

        When phase clock is available, returns smooth cosine-decay weights.
        Otherwise, falls back to binary weights from contact masks (1.0/0.0).
        """
        idx = self._future_indices(frame, horizon)
        if self.per_frame_phase is not None:
            return (
                self.per_frame_phase.support_weight_left[idx].copy(),
                self.per_frame_phase.support_weight_right[idx].copy(),
            )
        return (
            self.left_contact[idx].astype(np.float64),
            self.right_contact[idx].astype(np.float64),
        )

    def get_future_foot_quats(self, frame, horizon):
        idx = self._future_indices(frame, horizon)
        return (
            self.body_quat[idx, NPZ_LFOOT].copy(),
            self.body_quat[idx, NPZ_RFOOT].copy(),
        )


class FootstepResolver:
    """落脚点解算器：将动捕原始脚踝位置安全化到地形台阶表面。

    核心职责:
    1. 将所有台阶/地面分解为 x 方向的支撑面段 (surface_segments)
    2. 对每个触地帧 (touchdown)，调用 _resolve_landing_pose() 找到最佳支撑面
       并将 ankle x 约束到安全区间内
    3. 将落脚点广播到整个支撑阶段 (contact_support)

    注意: 目前所有约束都只在 x 方向上，y 直接沿用动捕原始值。
    """
    def __init__(
        self,
        terrain,       # 地形对象，需要 height_at(x, y) 和 height_batch(x, y) 方法
        raw_foot,      # (N, 3) 动捕原始脚踝位置 (世界坐标)
        contact_mask,  # (N,) 布尔触地掩码，True = 支撑期
        foot_nominal_z,  # 标量: 脚踝在平地 (z=0) 上的标称 z 高度
        foot_quats=None,       # (N, 4) 脚踝四元数 [w,x,y,z]，用于将足部偏移旋转到世界坐标
        foot_point_offsets=None,  # (M,) 足部关键点相对脚踝的 x 方向局部偏移
                                  # 默认 = [heel_offset_x, mid_offset_x, toe_offset_x]
                                  # 每个偏移都会通过四元数旋转到世界坐标，然后检查是否超出台阶边缘
        edge_margin=0.03,    # 台阶边缘安全裕量 (米): ankle x 不能离边缘太近
        toe_offset_x=TOE_OFFSET_X,  # 脚尖相对脚踝的 x 偏移 (局部坐标)
        toe_margin=0.005,    # 脚尖专用安全裕量: 脚尖比脚跟允许更靠近边缘
        max_landing_shift=0.05,  # _resolve_landing_pose 沿 fwd 方向的最大搜索位移 (米)
    ):
        self.terrain = terrain
        self.raw_foot = np.asarray(raw_foot, dtype=np.float64)   # 动捕原始脚踝位置
        self.contact_mask = np.asarray(contact_mask, dtype=bool)  # 触地掩码
        self.foot_nominal_z = float(foot_nominal_z)  # 平地上脚踝标称 z (例: ~0.049m)
        self.edge_margin = max(float(edge_margin), 0.0)  # 台阶边缘安全距离
        self.toe_offset_x = max(float(toe_offset_x), 0.0)  # 脚尖 x 偏移量
        self.toe_margin = max(float(toe_margin), 0.0)  # 脚尖专用裕量 (比 edge_margin 小)
        self.max_landing_shift = max(float(max_landing_shift), 0.005)  # 最大搜索位移
        self.n_frames = int(self.contact_mask.shape[0])  # 总帧数
        if foot_quats is not None:
            self.foot_quats = np.asarray(foot_quats, dtype=np.float64)
        else:
            # 如果没有提供四元数，默认朝向前方 (单位四元数)
            self.foot_quats = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                (self.n_frames, 1),
            )
        if foot_point_offsets is not None:
            self.foot_point_offsets = np.asarray(foot_point_offsets, dtype=np.float64)
        else:
            self.foot_point_offsets = np.array([toe_offset_x], dtype=np.float64)

        # 构建所有候选支撑面段: 每个台阶顶面 + 台阶之间的地面间隙
        self.surface_segments = self._build_surface_segments()

        # 检测触地事件: 从非触地→触地的上升沿
        # contact_mask[i] = True 且 contact_mask[i-1] = False → 这一帧是触地帧
        self.touchdown = self.contact_mask & ~np.r_[False, self.contact_mask[:-1]]
        # 如果存在触地帧但没有检测到上升沿 (整段都是触地)，将第一个触地帧标记为 touchdown
        if np.any(self.contact_mask) and not np.any(self.touchdown):
            first_contact = int(np.flatnonzero(self.contact_mask)[0])
            self.touchdown[first_contact] = True

        # 对每个触地帧，解算安全落脚位置 (x 约束到台阶安全区间, z = 地形高度 + foot_nominal_z)
        self.touchdown_support = np.full((self.n_frames, 3), np.nan, dtype=np.float64)
        for td in np.flatnonzero(self.touchdown):
            self.touchdown_support[td] = self._resolve_landing_pose(
                self.raw_foot[td], self.foot_quats[td]
            )

        # 构建前/后触地帧索引映射 (用于 swing 阶段查找相邻的支撑帧)
        (
            self.prev_contact_idx,  # prev_contact_idx[i] = 帧 i 之前最近的触地帧索引
            self.next_contact_idx,  # next_contact_idx[i] = 帧 i 之后最近的触地帧索引
        ) = self._build_prev_next_indices(self.contact_mask)
        # 构建每帧的"下一个 touchdown 帧"索引 (wrap=True: 到末尾后循环到开头)
        self.next_touchdown_idx = self._build_next_event_indices(self.touchdown, wrap=True)

        # 将 touchdown 落脚点广播到整个支撑阶段: contact_support[i] = 该支撑段的锁定位置
        self.contact_support = np.full((self.n_frames, 3), np.nan, dtype=np.float64)
        self._build_contact_support()

    def _build_surface_segments(self):
        """构建候选支撑面段列表: 每个台阶顶面 + 台阶之间的地面间隙。

        将地形在 x 方向上切分为连续的段:
        - 台阶段: (x_lo, x_hi, top_z) — 台阶顶面
        - 间隙段: (prev_hi, next_lo, 0.0) — 台阶之间的平地

        返回:
            [(x_lo, x_hi, surface_z), ...] 按 x 递增排列的段列表。
            段之间无间隙、不重叠，覆盖 (-inf, +inf) 整个 x 轴。

        注意: 只在 x 方向分段，完全不考虑 y 方向的台阶边界。
        如果台阶在 y 方向有限宽度，脚可能从 y 方向伸出台阶，
        但这里不会检测到。
        """
        # 按 x 范围排序所有台阶
        steps = sorted(
            getattr(self.terrain, "steps", []),
            key=lambda s: (float(s.x_lo), float(s.x_hi)),
        )
        if not steps:
            # 没有台阶 → 整个 x 轴都是 z=0 的平地
            return [(-np.inf, np.inf, 0.0)]

        segments = []
        prev_hi = -np.inf  # 上一个台阶的右端 x
        for step in steps:
            x_lo = float(step.x_lo)  # 当前台阶左端 x
            x_hi = float(step.x_hi)  # 当前台阶右端 x
            if x_lo > prev_hi:
                # 台阶之间有间隙 → 插入一段 z=0 的平地
                segments.append((prev_hi, x_lo, 0.0))
            # 添加台阶顶面段
            segments.append((x_lo, x_hi, float(step.top_z)))
            prev_hi = max(prev_hi, x_hi)
        # 最后一个台阶之后到 +∞ 的平地
        segments.append((prev_hi, np.inf, 0.0))
        return segments

    def _ankle_interval_on_surface(self, x_lo, x_hi):
        """计算脚踝本身在某个支撑面上的 x 安全区间。

        从支撑面的左右边界各缩进 edge_margin:
            ankle_x ∈ [x_lo + margin, x_hi - margin]
        确保脚踝不会太靠近台阶边缘。

        参数:
            x_lo, x_hi: 支撑面的 x 范围。

        返回:
            (lo, hi): 脚踝允许的 x 区间。
        """
        lo = x_lo + self.edge_margin if np.isfinite(x_lo) else -np.inf
        hi = x_hi - self.edge_margin if np.isfinite(x_hi) else np.inf
        return lo, hi

    def _point_interval_on_surface(self, x_lo, x_hi, dx):
        """计算足部某个关键点 (脚尖/脚跟/中足) 对 ankle x 的约束区间。

        关键点在世界坐标下相对脚踝有 dx 的 x 方向偏移 (由四元数旋转得到)。
        如果关键点位于脚踝前方 (dx >= 0, 即脚尖方向):
            - 左边界: x_lo + edge_margin (防止脚跟超出左侧)
            - 右边界: x_hi - toe_margin (脚尖允许更靠近右侧边缘)
        如果关键点位于脚踝后方 (dx < 0, 即脚跟方向):
            - 左边界: x_lo + toe_margin (脚跟允许更靠近左侧边缘)
            - 右边界: x_hi - edge_margin (防止脚尖超出右侧)

        将关键点约束转换回 ankle x 的约束: ankle_x ∈ [lo - dx, hi - dx]

        参数:
            x_lo, x_hi: 支撑面的 x 范围。
            dx: 关键点相对脚踝的世界坐标系 x 方向偏移。

        返回:
            (ankle_lo, ankle_hi): 为保证该关键点在支撑面内，ankle x 的允许区间。
        """
        if dx >= 0.0:
            lo = x_lo + self.edge_margin if np.isfinite(x_lo) else -np.inf
            hi = x_hi - self.toe_margin if np.isfinite(x_hi) else np.inf
        else:
            lo = x_lo + self.toe_margin if np.isfinite(x_lo) else -np.inf
            hi = x_hi - self.edge_margin if np.isfinite(x_hi) else np.inf
        # 关键点约束: point_x = ankle_x + dx ∈ [lo, hi]
        # → ankle_x ∈ [lo - dx, hi - dx]
        return lo - dx, hi - dx

    @staticmethod
    def _project_to_interval(x, lo, hi):
        """将 x 投影到区间 [lo, hi] 上。

        如果区间有效 (lo <= hi): 标准 clip。
        如果区间退化 (lo > hi): 取中点 (尽可能折中)。
        如果区间半无穷: 取有限端。
        """
        if lo <= hi:
            return float(np.clip(x, lo, hi))
        if np.isfinite(lo) and np.isfinite(hi):
            return 0.5 * (lo + hi)  # 退化区间: 取中点
        if np.isfinite(lo):
            return float(lo)
        if np.isfinite(hi):
            return float(hi)
        return float(x)  # 双无穷: 不约束

    def _score_surface_candidate(self, raw_x, foot_quat, surface):
        """评估某个支撑面是否能安全放下整只脚，并返回最佳 ankle x。

        评估逻辑:
        1. 计算 ankle 本身的安全 x 区间
        2. 对每个足部关键点 (脚跟/中足/脚尖)，用四元数将局部偏移旋转到世界坐标，
           计算该关键点对 ankle x 的约束区间
        3. 取所有约束区间的交集 → 如果交集非空，表示整只脚都能放在这个面上
        4. 如果交集为空，表示无法完全放下 → 退而求其次，只保证 ankle 在安全区间内

        得分元组 (violated, total_violation, shift):
        - violated: 超出支撑面的关键点数 (0 = 完全安全)
        - total_violation: 总超出距离 (m)
        - shift: ankle x 相对动捕原始 x 的偏移量 (越小越好)
        得分越小越好 (元组逐元素比较)。

        参数:
            raw_x: 动捕原始 ankle x 坐标。
            foot_quat: 脚踝四元数 [w,x,y,z]，用于将局部偏移旋转到世界坐标。
            surface: (x_lo, x_hi, surface_z) 候选支撑面。

        返回:
            (score_tuple, x_safe): 得分元组和安全化后的 ankle x。
        """
        x_lo, x_hi, _surface_z = surface
        # 1. ankle 本身的安全 x 区间
        ankle_lo, ankle_hi = self._ankle_interval_on_surface(x_lo, x_hi)
        # 初始化综合约束区间为 ankle 区间
        req_lo, req_hi = ankle_lo, ankle_hi
        dx_list = []  # 记录每个关键点的世界坐标 x 偏移

        # 2. 逐个关键点收紧约束区间
        for x_off in self.foot_point_offsets:
            # 将局部 x 偏移通过脚踝四元数旋转到世界坐标
            off_world = _quat_rotate(foot_quat, np.array([x_off, 0.0, 0.0]))
            dx = float(off_world[0])  # 世界坐标下的 x 方向偏移
            dx_list.append(dx)
            # 该关键点对 ankle x 的约束区间
            pt_lo, pt_hi = self._point_interval_on_surface(x_lo, x_hi, dx)
            # 取交集: 逐步收紧
            req_lo = max(req_lo, pt_lo)
            req_hi = min(req_hi, pt_hi)

        # 3. 交集非空 → 整只脚都能放下 → 完美得分 (0 个违规)
        if req_lo <= req_hi:
            x_safe = float(np.clip(raw_x, req_lo, req_hi))  # 尽量接近原始 x
            return (0, 0.0, abs(x_safe - raw_x)), x_safe

        # 4. 交集为空 → 无法完全放下 → 退回只保证 ankle 安全，统计违规
        x_safe = self._project_to_interval(raw_x, ankle_lo, ankle_hi)
        violated = 0           # 超出支撑面的关键点数
        total_violation = 0.0  # 总超出距离
        for dx in dx_list:
            pt_lo, pt_hi = self._point_interval_on_surface(x_lo, x_hi, dx)
            # 计算 x_safe 处该关键点超出安全区间的距离
            violation = max(pt_lo - x_safe, 0.0) + max(x_safe - pt_hi, 0.0)
            if violation > 1e-9:
                violated += 1
                total_violation += violation
        return (violated, total_violation, abs(x_safe - raw_x)), x_safe

    def _resolve_landing_pose(self, raw_pose, foot_quat):
        """解算单帧的安全落脚位置 — 沿脚的 yaw 方向做边缘约束。

        改进: 不再假设台阶边缘沿世界 x 轴排列。
        而是从 foot_quat 提取 yaw → 得到脚的前方方向 fwd，
        沿 fwd 方向搜索最优 ankle 偏移，使所有关键点 (脚跟/中足/脚尖)
        都落在同一个支撑面上，且距边缘有足够裕量。

        算法:
        1. 计算 ankle 和各关键点的世界 xy 坐标
        2. 查询各点处的地形高度
        3. 沿 fwd 方向做 1D 搜索 (离散步进)，找到使所有关键点
           都在同一台阶面上 (高度差 < 容差) 的最小偏移量
        4. z 取所有关键点地形高度的 max + foot_nominal_z

        参数:
            raw_pose: (3,) 动捕原始脚踝位置 [x, y, z]。
            foot_quat: (4,) 脚踝四元数 [w,x,y,z]。

        返回:
            (3,) 安全落脚位置 [x_safe, y_safe, z_safe]。
        """
        ax = float(raw_pose[0])
        ay = float(raw_pose[1])

        # ---- 从 foot_quat 提取 yaw → 脚的前方方向 (世界 xy 平面) ----
        w, qx, qy, qz = foot_quat
        yaw = np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        fwd_x = np.cos(yaw)  # 脚前方方向的世界 x 分量
        fwd_y = np.sin(yaw)  # 脚前方方向的世界 y 分量

        # ---- 预计算各关键点相对 ankle 的世界 xy 偏移 ----
        # foot_point_offsets = [heel_x, mid_x, toe_x], 都是脚局部坐标的 x 方向偏移
        point_offsets_world = []  # [(dx, dy), ...] 世界坐标偏移
        for x_off in self.foot_point_offsets:
            off_world = _quat_rotate(foot_quat, np.array([x_off, 0.0, 0.0]))
            point_offsets_world.append((float(off_world[0]), float(off_world[1])))

        # ---- 辅助函数: 在给定 ankle (px, py) 处评估所有关键点 ----
        def _eval_at(px, py):
            """返回 (h_ankle, h_points[], h_max, all_same_surface)"""
            h_ankle = float(self.terrain.height_at(px, py))
            h_max = h_ankle
            h_pts = []
            for (dx, dy) in point_offsets_world:
                h = float(self.terrain.height_at(px + dx, py + dy))
                h_pts.append(h)
                if h > h_max:
                    h_max = h
            # 所有点是否在同一台阶面上 (高度差 < 1cm)
            all_h = [h_ankle] + h_pts
            all_same = (max(all_h) - min(all_h)) < 0.01
            return h_ankle, h_pts, h_max, all_same

        # ---- 先评估原始位置 ----
        h_ankle_0, h_pts_0, h_max_0, same_0 = _eval_at(ax, ay)
        if same_0:
            # 所有关键点都在同一台阶面上 → 不需要调整
            z_safe = h_max_0 + self.foot_nominal_z
            return np.array([ax, ay, z_safe], dtype=np.float64)

        # 原始位置 ankle 处的地形高度 — 作为目标台阶面高度参考
        # 搜索时优先留在同一高度的台阶面上，防止被推到更低的平地
        ref_h = h_ankle_0

        # ---- 沿脚的前方方向 (±fwd) 搜索最优偏移 ----
        # 搜索策略: 沿 fwd 方向以 step_size 步进，直到找到所有关键点
        # 在同一台阶面上的位置，或达到搜索上限。
        # 负向 fwd = 脚整体后退 (让脚尖远离前方边缘)
        # 正向 fwd = 脚整体前进 (让脚跟远离后方边缘)
        step_size = 0.005  # 搜索步长 (m)
        max_shift = self.max_landing_shift  # 使用配置的最大搜索位移
        n_steps = int(max_shift / step_size) + 1

        best_shift = 0.0
        best_score = self._landing_score(h_ankle_0, h_pts_0, same_0, 0.0, ref_h)
        best_xy = (ax, ay)
        best_h_max = h_max_0

        for sign in (-1.0, 1.0):
            for i in range(1, n_steps + 1):
                d = sign * i * step_size
                px = ax + d * fwd_x
                py = ay + d * fwd_y
                h_ankle, h_pts, h_max, same = _eval_at(px, py)
                score = self._landing_score(h_ankle, h_pts, same, abs(d), ref_h)
                if score < best_score:
                    best_score = score
                    best_shift = d
                    best_xy = (px, py)
                    best_h_max = h_max
                # 如果已找到完全一致的台阶面，且后续 shift 只会增大，提前停止
                if same and abs(d) > abs(best_shift) + step_size:
                    break

        x_safe, y_safe = best_xy
        z_safe = best_h_max + self.foot_nominal_z
        return np.array([x_safe, y_safe, z_safe], dtype=np.float64)

    @staticmethod
    def _landing_score(h_ankle, h_pts, same, shift, ref_h):
        """落脚点评分: 越小越好。

        优先级 (元组字典序比较):
        1. same_surface: 所有关键点是否在同一台阶面 (0=是, 1=否)
        2. height_spread: 各关键点高度差的极差 (越小越好)
        3. ref_deviation: ankle 处地形高度与原始参考高度的偏差
           (优先留在原始台阶面上，防止被推到更低/更高的面)
        4. shift: 沿 fwd 方向的偏移距离 (越小越好，尽量不偏离原始位置)

        返回:
            (int, float, float, float) 得分元组。
        """
        all_h = [h_ankle] + list(h_pts)
        spread = max(all_h) - min(all_h)
        ref_dev = abs(h_ankle - ref_h)
        return (0 if same else 1, spread, ref_dev, shift)

    def _build_prev_next_indices(self, mask):
        """为每一帧构建"前一个事件帧"和"后一个事件帧"的索引映射。

        用二分搜索 (searchsorted) 在事件帧列表中高效查找:
        - prev_idx[i]: 帧 i 之前 (含 i) 最近的事件帧索引，不存在则为 -1
        - next_idx[i]: 帧 i 之后 (含 i) 最近的事件帧索引，到末尾则循环到开头

        参数:
            mask: (N,) 布尔数组，True 的位置为事件帧。

        返回:
            (prev_idx, next_idx): 两个 (N,) int64 索引数组。
        """
        n = int(mask.shape[0])
        event_idx = np.flatnonzero(mask)  # 所有事件帧的索引 (已排序)
        prev_idx = np.full(n, -1, dtype=np.int64)
        next_idx = np.full(n, -1, dtype=np.int64)
        if event_idx.size == 0:
            return prev_idx, next_idx

        for i in range(n):
            # 前一个事件帧: 在 event_idx 中找 <= i 的最大值
            p = np.searchsorted(event_idx, i, side="right") - 1
            prev_idx[i] = int(event_idx[p]) if p >= 0 else -1
            # 后一个事件帧: 在 event_idx 中找 >= i 的最小值
            q = np.searchsorted(event_idx, i, side="left")
            next_idx[i] = int(event_idx[q]) if q < event_idx.size else int(event_idx[0])
        return prev_idx, next_idx

    def _build_next_event_indices(self, mask, wrap=False):
        """为每一帧构建"下一个事件帧"索引映射。

        与 _build_prev_next_indices 类似，但只构建 next 方向。
        wrap=True 时，序列末尾的帧会循环到第一个事件帧。

        参数:
            mask: (N,) 布尔数组，True 的位置为事件帧。
            wrap: 是否在到达末尾时循环。

        返回:
            (N,) int64 索引数组，-1 表示无后续事件帧。
        """
        n = int(mask.shape[0])
        event_idx = np.flatnonzero(mask)
        next_idx = np.full(n, -1, dtype=np.int64)
        if event_idx.size == 0:
            return next_idx

        for i in range(n):
            q = np.searchsorted(event_idx, i, side="left")
            if q < event_idx.size:
                next_idx[i] = int(event_idx[q])
            elif wrap:
                next_idx[i] = int(event_idx[0])  # 循环到序列开头
            else:
                next_idx[i] = -1  # 无后续事件
        return next_idx

    def _build_contact_support(self):
        """将 touchdown 落脚点广播到整个支撑阶段。

        遍历所有帧:
        1. 当检测到"触地起始帧"(从非触地→触地的上升沿):
           - 优先使用 touchdown_support (在 __init__ 中预计算的安全落脚位置)
           - 如果 touchdown_support 无效 (NaN)，重新调用 _resolve_landing_pose() 计算
        2. 在整个支撑阶段 (连续触地) 内，所有帧都锁定到同一个 current 位置
           → 支撑期间 xy 和 z 完全固定，脚不会滑动

        结果写入 self.contact_support: (N, 3) 数组，
        支撑帧有有效值，摆动帧保持 NaN。
        """
        if not np.any(self.contact_mask):
            return  # 没有任何触地帧，无需处理

        current = None  # 当前支撑段的锁定位置
        for i in range(self.n_frames):
            prev_i = i - 1
            # 检测触地起始帧: 当前帧触地 且 (是第一帧 或 前一帧非触地)
            if self.contact_mask[i] and (i == 0 or not self.contact_mask[prev_i]):
                # 新的支撑段开始 → 更新 current 落脚位置
                if np.isfinite(self.touchdown_support[i, 2]):
                    # 有预计算的安全落脚位置 → 直接使用
                    current = self.touchdown_support[i].copy()
                else:
                    # 预计算值无效 → 重新解算
                    current = self._resolve_landing_pose(
                        self.raw_foot[i], self.foot_quats[i]
                    )
            # 触地帧 → 写入当前支撑段的锁定位置
            if self.contact_mask[i]:
                if current is None:
                    # 极端情况: 序列开头就是触地但没被识别为 touchdown
                    current = self._resolve_landing_pose(
                        self.raw_foot[i], self.foot_quats[i]
                    )
                # 整个支撑段所有帧共享同一个 (x, y, z) → 脚不滑动
                self.contact_support[i] = current.copy()

    def prev_support_pose(self, frame_idx):
        """查询帧 frame_idx 之前最近的支撑位置。

        用于摆动阶段确定 liftoff (抬脚) 位置:
        摆动前的最后一个触地帧的 contact_support 就是抬脚点。

        返回:
            (3,) 位置数组 [x, y, z]，或 None (无前序支撑帧)。
        """
        idx = int(frame_idx) % self.n_frames
        prev_contact = int(self.prev_contact_idx[idx])
        if prev_contact < 0:
            return None
        pose = self.contact_support[prev_contact]
        if not np.isfinite(pose[2]):
            return None
        return pose.copy()

    def next_touchdown_pose(self, frame_idx):
        """查询帧 frame_idx 之后最近的 touchdown (着地) 位置。

        用于摆动阶段确定 landing (落脚) 目标:
        摆动结束时脚要到达的安全位置。

        返回:
            (3,) 位置数组 [x, y, z]，或 None (无后续 touchdown)。
        """
        idx = int(frame_idx) % self.n_frames
        td = int(self.next_touchdown_idx[idx])
        if td < 0:
            return None
        pose = self.touchdown_support[td]
        if not np.isfinite(pose[2]):
            return None
        return pose.copy()


class TerrainReference:
    """地形参考轨迹构建器：将动捕原始脚位适配到目标地形。

    核心输出:
        left_precomputed / right_precomputed: (N, 3) 左/右脚踝参考位置
        left_precomputed_quats / right_precomputed_quats: (N, 4) 左/右脚踝参考四元数

    处理流程:
        1. 创建 FootstepResolver → 为每个触地帧解算安全落脚位置
        2. _precompute_foot_reference() → 支撑帧用锁定落脚位置，
           摆动帧用 Hermite/quintic 插值 + 地面 clamp
        3. _flatten_stance_quats() → 支撑期脚面保持水平 (仅保留 yaw)
    """
    def __init__(
        self,
        terrain,             # 地形对象
        motion,              # MotionClip 动捕数据
        lookahead=0.0,       # 地形前瞻距离 (米)，用于提前感知即将到来的台阶
        smoothing_alpha=0.08,  # 骨盆 z 的 EMA 平滑系数
        footstep_margin=0.03,  # 台阶边缘安全裕量 (米)，传递给 FootstepResolver.edge_margin
        toe_offset_x=TOE_OFFSET_X,  # 脚尖相对脚踝的 x 偏移 (局部坐标)
        heel_offset_x=-0.04,  # 脚跟相对脚踝的 x 偏移 (局部坐标，负值=后方)
        mid_offset_x=0.10,    # 中足相对脚踝的 x 偏移 (局部坐标)
        toe_margin=0.005,      # 脚尖专用安全裕量 (比 edge_margin 小)
        swing_floor_margin=0.003,  # 摆动阶段额外的地面间距裕量 (米)
        soft_reach_limit=0.70,     # 骨盆到脚的软可达距离限制 (米)
        max_landing_shift=0.05,    # 落脚点沿 fwd 方向的最大搜索位移 (米)
    ):
        self.terrain = terrain
        self.motion = motion
        self.lookahead = float(lookahead)
        self.alpha = float(smoothing_alpha)
        self.swing_floor_margin = max(float(swing_floor_margin), 0.0)
        self.soft_reach_limit = max(float(soft_reach_limit), 0.1)
        self._ema_z = None  # 骨盆 z 的指数移动平均值

        # 足部关键点的 x 方向局部偏移: [脚跟, 中足, 脚尖]
        # 这三个点会被 FootstepResolver 用来检查整只脚是否能放在台阶上
        foot_point_offsets = np.array(
            [heel_offset_x, mid_offset_x, toe_offset_x], dtype=np.float64
        )
        # 创建左脚落脚点解算器
        self.left_resolver = FootstepResolver(
            terrain=terrain,
            raw_foot=motion.body_pos[:, NPZ_LFOOT],     # 动捕原始左脚踝位置
            contact_mask=motion.left_contact,             # 左脚触地掩码
            foot_nominal_z=motion.foot_nominal_z,
            foot_quats=motion.body_quat[:, NPZ_LFOOT],  # 动捕原始左脚四元数
            foot_point_offsets=foot_point_offsets,
            edge_margin=footstep_margin,
            toe_offset_x=toe_offset_x,
            toe_margin=toe_margin,
            max_landing_shift=max_landing_shift,
        )
        # 创建右脚落脚点解算器
        self.right_resolver = FootstepResolver(
            terrain=terrain,
            raw_foot=motion.body_pos[:, NPZ_RFOOT],
            contact_mask=motion.right_contact,
            foot_nominal_z=motion.foot_nominal_z,
            foot_quats=motion.body_quat[:, NPZ_RFOOT],
            foot_point_offsets=foot_point_offsets,
            edge_margin=footstep_margin,
            toe_offset_x=toe_offset_x,
            toe_margin=toe_margin,
            max_landing_shift=max_landing_shift,
        )
        # 预计算完整的脚位参考轨迹 (支撑帧=锁定落点, 摆动帧=插值)
        self.left_precomputed = self._precompute_foot_reference(self.left_resolver)
        self.right_precomputed = self._precompute_foot_reference(self.right_resolver)
        # 预计算脚踝四元数参考 (初始为动捕原始值)
        self.left_precomputed_quats = motion.body_quat[:, NPZ_LFOOT].copy()
        self.right_precomputed_quats = motion.body_quat[:, NPZ_RFOOT].copy()
        # 支撑期脚面保持水平: 提取 yaw 角、去掉 pitch/roll
        # 在台阶上脚应该平放，不要跟随动捕中可能存在的脚面倾斜
        self._flatten_stance_quats(
            self.left_precomputed_quats, motion.left_contact
        )
        self._flatten_stance_quats(
            self.right_precomputed_quats, motion.right_contact
        )

        # Enforce no lateral crossover between left and right foot references.
        # Check in pelvis local frame: left foot must stay on left side.
        self._enforce_no_crossover(motion, min_sep=0.08)
        # Enforce step-order consistency: the cross product of consecutive
        # left/right landing vectors must match the original motion's sign.
        self._enforce_step_order(motion)

    def _enforce_step_order(self, motion):
        """Enforce that the relative left/right order of consecutive landings
        matches the original motion.

        For each stance phase of each foot, we know the locked landing xy.
        The vector from left_landing to right_landing, crossed with pelvis
        forward direction, must have the same sign as in the original motion.
        If not, push the offending foot back to the correct side.
        """
        N = len(self.left_precomputed)
        pelvis_quats = motion.body_quat[:, NPZ_PELVIS]
        raw_lfoot = motion.body_pos[:, NPZ_LFOOT]
        raw_rfoot = motion.body_pos[:, NPZ_RFOOT]

        # Get landing positions: the foot position at the START of each stance phase
        left_stances = []  # (frame_idx, position)
        right_stances = []
        # Left stance phases
        in_stance = False
        for i in range(N):
            if motion.left_contact[i] and not in_stance:
                left_stances.append((i, self.left_precomputed[i, :2].copy()))
                in_stance = True
            elif not motion.left_contact[i]:
                in_stance = False
        in_stance = False
        for i in range(N):
            if motion.right_contact[i] and not in_stance:
                right_stances.append((i, self.right_precomputed[i, :2].copy()))
                in_stance = True
            elif not motion.right_contact[i]:
                in_stance = False

        if len(left_stances) < 1 or len(right_stances) < 1:
            return

        # For each frame, find the nearest left and right stance landing
        # and check cross product consistency
        n_fixed = 0
        for i in range(N):
            w, x, y, z = pelvis_quats[i]
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            fwd = np.array([np.cos(yaw), np.sin(yaw)])  # pelvis forward in xy

            # Vector from left to right foot (precomputed)
            lr_vec = self.right_precomputed[i, :2] - self.left_precomputed[i, :2]
            # Cross product z-component: fwd × lr_vec
            cross_precomp = fwd[0] * lr_vec[1] - fwd[1] * lr_vec[0]

            # Same for original motion
            lr_raw = raw_rfoot[i, :2] - raw_lfoot[i, :2]
            cross_raw = fwd[0] * lr_raw[1] - fwd[1] * lr_raw[0]

            # If original has no crossover (cross_raw < 0 means right is to the right
            # of the forward direction, which is normal), enforce same sign
            if cross_raw < -0.01 and cross_precomp > 0:
                # Precomputed crossed over — push apart
                lat = np.array([-np.sin(yaw), np.cos(yaw)])  # left direction
                deficit = 0.08  # push to at least 8cm separation
                self.left_precomputed[i, :2] += 0.5 * deficit * lat
                self.right_precomputed[i, :2] -= 0.5 * deficit * lat
                n_fixed += 1
            elif cross_raw > 0.01 and cross_precomp < 0:
                lat = np.array([-np.sin(yaw), np.cos(yaw)])
                deficit = 0.08
                self.left_precomputed[i, :2] += 0.5 * deficit * lat
                self.right_precomputed[i, :2] -= 0.5 * deficit * lat
                n_fixed += 1

        if n_fixed > 0:
            print(f"  [TerrainRef] Step-order fix: {n_fixed}/{N} frames corrected")

    def _enforce_no_crossover(self, motion, min_sep=0.08):
        """Enforce that left foot stays left and right foot stays right.

        For each frame, project both feet onto the pelvis lateral axis.
        If the left foot's lateral coordinate is less than the right foot's
        (i.e. they crossed), push them apart to maintain min_sep.

        Only modifies frames where the original motion had no crossover
        (respects the original motion's intent for intentional crossovers).
        """
        N = len(self.left_precomputed)
        pelvis_pos = motion.body_pos[:, NPZ_PELVIS]
        pelvis_quats = motion.body_quat[:, NPZ_PELVIS]
        raw_lfoot = motion.body_pos[:, NPZ_LFOOT]
        raw_rfoot = motion.body_pos[:, NPZ_RFOOT]

        n_fixed = 0
        for i in range(N):
            # Pelvis lateral direction
            w, x, y, z = pelvis_quats[i]
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            lat = np.array([-np.sin(yaw), np.cos(yaw)])
            pel_xy = pelvis_pos[i, :2]

            # Check original motion: did it have crossover at this frame?
            raw_lf_lat = np.dot(raw_lfoot[i, :2] - pel_xy, lat)
            raw_rf_lat = np.dot(raw_rfoot[i, :2] - pel_xy, lat)
            if raw_lf_lat - raw_rf_lat < 0.02:
                # Original motion already has crossover — don't enforce
                continue

            # Check precomputed reference
            lf_lat = np.dot(self.left_precomputed[i, :2] - pel_xy, lat)
            rf_lat = np.dot(self.right_precomputed[i, :2] - pel_xy, lat)
            if lf_lat - rf_lat < min_sep:
                deficit = min_sep - (lf_lat - rf_lat)
                self.left_precomputed[i, :2] += 0.5 * deficit * lat
                self.right_precomputed[i, :2] -= 0.5 * deficit * lat
                n_fixed += 1

        if n_fixed > 0:
            print(f"  [TerrainRef] Fixed {n_fixed}/{N} frames lateral crossover (min_sep={min_sep*100:.0f}cm)")

    @staticmethod
    def _flatten_stance_quats(quats, contact_mask):
        """将支撑期的脚踝四元数"压平"：去掉 pitch/roll，仅保留 yaw。

        原理: 在离散台阶地形上，支撑期脚面应该水平贴在台阶表面上，
        不需要跟随动捕数据中可能存在的脚面倾斜 (pitch/roll)。
        仅保留 yaw (绕 z 轴旋转) 来维持脚面朝向。

        操作: 从四元数 [w,x,y,z] 中提取 yaw = atan2(2(wz+xy), 1-2(y²+z²))，
        然后重建纯 yaw 四元数: [cos(yaw/2), 0, 0, sin(yaw/2)]。

        参数:
            quats: (N, 4) 四元数数组 [w,x,y,z]，会被原地修改。
            contact_mask: (N,) 布尔触地掩码，仅修改触地帧。
        """
        for i in range(quats.shape[0]):
            if not contact_mask[i]:
                continue  # 摆动帧不修改，保留原始动捕四元数
            w, x, y, z = quats[i]
            # 从四元数中提取 yaw 角 (绕 z 轴旋转角)
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            half = yaw * 0.5
            # 重建纯 yaw 四元数: pitch = roll = 0 → 脚面水平
            quats[i] = [np.cos(half), 0.0, 0.0, np.sin(half)]

    def reset(self):
        """重置骨盆 z 的 EMA 状态。"""
        self._ema_z = None

    def _target_height(self, x, y):
        """查询目标地形高度 = max(当前位置高度, 前瞻位置高度)。

        前瞻 (lookahead): 提前感知前方即将到来的台阶，让骨盆提前抬升。
        取 max 是为了保守 — 永远不会低估前方的地形高度。
        """
        h_here = self.terrain.height_at(x, y)       # 当前 xy 处的地形高度
        h_ahead = self.terrain.height_at(x + self.lookahead, y)  # 前方 lookahead 米处的高度
        return max(h_here, h_ahead)

    def update(self, pelvis_x, pelvis_y):
        """单帧更新骨盆 z 的指数移动平均值 (EMA)。

        公式: ema_z += alpha * (target - ema_z)
        alpha 越小，骨盆 z 变化越平滑，但响应越慢。
        用于实时逐帧场景 (非预计算)。

        返回:
            当前帧的平滑骨盆 z 参考值。
        """
        target = self._target_height(pelvis_x, pelvis_y)
        if self._ema_z is None:
            self._ema_z = target  # 第一帧直接初始化
        else:
            self._ema_z += self.alpha * (target - self._ema_z)  # EMA 更新
        return self._ema_z

    def segment_offsets(self, pelvis_positions):
        """批量计算一段骨盆位置序列的 EMA z 偏移量。

        参数:
            pelvis_positions: (N, 3) 骨盆位置序列。

        返回:
            (N,) 每帧的 EMA 平滑 z 参考值。
        """
        offsets = np.zeros(pelvis_positions.shape[0], dtype=np.float64)
        ema = 0.0 if self._ema_z is None else float(self._ema_z)
        for i, pelvis in enumerate(pelvis_positions):
            target = self._target_height(pelvis[0], pelvis[1])
            ema += self.alpha * (target - ema)
            offsets[i] = ema
        return offsets

    def _swing_indices(self, start, end, n_frames):
        """生成摆动阶段的帧索引数组 [start, start+1, ..., end-1]。

        参数:
            start, end: 摆动阶段的起止帧索引。
            n_frames: 总帧数 (用于边界检查)。

        返回:
            (n_swing,) int64 索引数组，如果无效则返回空数组。
        """
        if start < 0 or end < 0:
            return np.zeros(0, dtype=np.int64)
        if start < end:
            return np.arange(start, end, dtype=np.int64)
        return np.zeros(0, dtype=np.int64)

    def _apply_floor_clamp(self, points, foot_nominal_z):
        """地面 clamp: 确保脚踝 z 不低于地形高度 + 标称高度 + 裕量。

        公式: z = max(z, terrain_height(x, y) + foot_nominal_z + swing_floor_margin)

        注意: 地形高度是在 ankle 的 (x, y) 处查询的，
        不是在脚尖或脚跟的 (x, y) 处查询。
        如果脚尖/脚跟偏离 ankle 较远且地形在该处更高，此 clamp 不会感知到。

        参数:
            points: (M, 3) 脚踝位置数组。
            foot_nominal_z: 平地上脚踝标称 z。

        返回:
            (M, 3) clamp 后的位置数组 (副本，不修改原始数组)。
        """
        floor = (
            self.terrain.height_batch(points[:, 0], points[:, 1])  # 每个 ankle xy 处的地形高度
            + float(foot_nominal_z)       # 加上脚踝标称高度
            + self.swing_floor_margin     # 加上额外裕量
        )
        out = points.copy()
        out[:, 2] = np.maximum(out[:, 2], floor)  # z 取较大值 (不允许低于地面)
        return out

    def _precompute_foot_reference(self, resolver):
        """预计算完整的脚位参考轨迹：支撑帧用锁定落点，摆动帧用插值。

        处理流程:
        1. 初始化: 从动捕原始脚位开始
        2. 覆盖支撑帧: 用 FootstepResolver 计算的安全落脚位置 (contact_support)
        3. 插值摆动帧: 在 liftoff (抬脚) 和 landing (落脚) 之间:
           - xy: cubic Hermite 插值 (端点速度=0，保证平滑过渡)
           - z:  quintic 基线插值 + 动捕原始摆动形状的渐变混入
        4. 地面 clamp: 确保所有帧 z 不低于地形

        参数:
            resolver: FootstepResolver 实例 (含 contact_support, touchdown_support 等)。

        返回:
            (N, 3) 脚踝参考位置数组。
        """
        raw = resolver.raw_foot      # 动捕原始脚踝位置 (N, 3)
        contact = resolver.contact_mask  # 触地掩码 (N,)
        n_frames = int(resolver.n_frames)
        out = raw.copy()  # 从原始数据开始，逐步覆盖

        # === 第 1 步: 用安全落脚位置覆盖所有有效的支撑帧 ===
        # contact_support 中有效 (非 NaN) 的帧 = FootstepResolver 已解算的安全位置
        valid_contact = np.isfinite(resolver.contact_support[:, 2])
        out[valid_contact] = resolver.contact_support[valid_contact]

        # === 第 2 步: 找到所有摆动阶段的起止帧 ===
        # swing_start: 从触地→非触地的下降沿 (摆动开始帧)
        prev_contact = np.r_[False, contact[:-1]]  # 前一帧的触地状态
        swing_start = np.flatnonzero((~contact) & prev_contact)  # 当前帧非触地 且 前一帧触地
        # touchdown: 从非触地→触地的上升沿 (落脚帧)
        touchdown = np.flatnonzero(contact & (~prev_contact))

        if touchdown.size == 0:
            # 没有落脚事件 → 只做地面 clamp 后返回
            self._apply_floor_clamp(out, resolver.foot_nominal_z)
            return out

        # === 第 3 步: 对每个摆动阶段做插值 ===
        touchdown_sorted = np.sort(touchdown)
        for s in swing_start:
            # 找到当前 swing_start 之后最近的 touchdown (落脚帧)
            q = np.searchsorted(touchdown_sorted, s + 1, side="left")
            if q >= touchdown_sorted.size:
                continue  # 没有后续落脚帧 → 跳过
            e = int(touchdown_sorted[q])  # 落脚帧索引
            span = self._swing_indices(int(s), e, n_frames)  # 摆动帧索引数组
            if span.size == 0:
                continue

            # liftoff = 摆动前最后一个触地帧 (抬脚帧)
            liftoff_idx = int(s) - 1
            # landing = 摆动后第一个触地帧 (落脚帧)
            landing_idx = int(e)
            if liftoff_idx < 0:
                continue  # 序列开头就是摆动 → 无 liftoff 参考，跳过

            liftoff_pose = out[liftoff_idx]  # 抬脚位置 (已被 contact_support 覆盖)
            # 落脚位置: 优先使用 touchdown_support (预解算的安全位置)
            landing_pose = resolver.touchdown_support[landing_idx]
            if not np.isfinite(landing_pose[2]):
                landing_pose = out[landing_idx]  # 预解算无效 → 退回到当前 out 值

            # 边界检查: liftoff 或 landing 无效则只做 clamp
            if not np.isfinite(liftoff_pose[2]) or not np.isfinite(landing_pose[2]):
                out[span] = self._apply_floor_clamp(out[span], resolver.foot_nominal_z)
                continue

            n = int(span.size)  # 摆动帧数
            if n == 1:
                t = np.array([0.5], dtype=np.float64)  # 只有 1 帧 → 取中点
            else:
                t = np.linspace(0.0, 1.0, n, dtype=np.float64)  # 0→1 的归一化时间

            # --- xy 插值: cubic Hermite (端点速度=0, 保证平滑过渡) ---
            # h(t) = 3t² - 2t³, h(0)=0, h(1)=1, h'(0)=h'(1)=0
            xy_blend = (3.0 * t ** 2) - (2.0 * t ** 3)
            out[span, 0:2] = (
                (1.0 - xy_blend)[:, np.newaxis] * liftoff_pose[0:2]  # 从 liftoff xy
                + xy_blend[:, np.newaxis] * landing_pose[0:2]         # 到 landing xy
            )

            # --- z 插值: quintic 基线 + 动捕摆动形状的渐变混入 ---
            # quintic: q(t) = 6t⁵ - 15t⁴ + 10t³, 满足 C2 (位置+速度+加速度连续)
            z_blend = (6.0 * t ** 5) - (15.0 * t ** 4) + (10.0 * t ** 3)
            # z 基线: 从 liftoff z 到 landing z 的 quintic 插值
            z_base = (1.0 - z_blend) * liftoff_pose[2] + z_blend * landing_pose[2]

            # 动捕摆动形状: 从原始 z 中提取 "超出线性插值的部分"
            # 这部分包含了动捕数据中自然的抬脚弧线形状
            raw_z0 = raw[liftoff_idx, 2]   # 原始 liftoff z
            raw_z1 = raw[landing_idx, 2]   # 原始 landing z
            raw_linear = (1.0 - t) * raw_z0 + t * raw_z1  # 原始数据的线性基线
            raw_shape = raw[span, 2] - raw_linear  # 超出线性基线的摆动形状 (弧线)

            # 渐变窗口: 16t²(1-t)², 在 t=0 和 t=1 处为 0 (不影响端点),
            # 在 t=0.5 处最大=1 (中间最大混入)
            shape_taper = 16.0 * (t ** 2) * ((1.0 - t) ** 2)
            # 最终 z = quintic 基线 + 渐变 × 动捕弧线形状
            out[span, 2] = z_base + shape_taper * raw_shape

            # 地面 clamp: 确保摆动阶段脚踝 z 不低于地形
            out[span] = self._apply_floor_clamp(out[span], resolver.foot_nominal_z)

        # 全局地面 clamp: 最后一次确保所有帧都不穿地
        return self._apply_floor_clamp(out, resolver.foot_nominal_z)

    def build_segment(
        self,
        pelvis,
        lfoot,
        rfoot,
        left_contact,
        right_contact,
        indices,
    ):
        pelvis_z_offsets = self.segment_offsets(pelvis)

        ref_pelvis = pelvis.copy()
        ref_pelvis[:, 2] += pelvis_z_offsets
        idx = np.asarray(indices, dtype=np.int64)
        ref_lfoot = self.left_precomputed[idx].copy()
        ref_rfoot = self.right_precomputed[idx].copy()
        ref_lfoot_quats = self.left_precomputed_quats[idx].copy()
        ref_rfoot_quats = self.right_precomputed_quats[idx].copy()

        # Layer 1: enforce reachability by lowering pelvis z when either foot
        # exceeds the soft reach limit.  Compute pelvis-to-foot distance and
        # lower pelvis z per horizon step so that both legs stay within L_soft.
        L = self.soft_reach_limit
        for i in range(ref_pelvis.shape[0]):
            for foot in (ref_lfoot[i], ref_rfoot[i]):
                dxy = np.sqrt(
                    (ref_pelvis[i, 0] - foot[0]) ** 2
                    + (ref_pelvis[i, 1] - foot[1]) ** 2
                )
                # max allowed vertical distance given horizontal offset
                max_dz = np.sqrt(max(L ** 2 - dxy ** 2, 0.0))
                # pelvis must be above foot, so cap: pelvis_z <= foot_z + max_dz
                ref_pelvis[i, 2] = min(ref_pelvis[i, 2], foot[2] + max_dz)

        return ref_pelvis, ref_lfoot, ref_rfoot, ref_lfoot_quats, ref_rfoot_quats


class TinyMPPI:
    IDX_PELVIS_DZ = 0
    IDX_LFOOT_DX = 1
    IDX_LFOOT_DY = 2
    IDX_LFOOT_DZ = 3
    IDX_RFOOT_DX = 4
    IDX_RFOOT_DY = 5
    IDX_RFOOT_DZ = 6
    ACTION_DIM = 7

    def __init__(
        self,
        terrain,
        foot_nominal_z,
        pelvis_height_above_foot,
        n_samples=32,
        n_iterations=1,
        noise_std=0.03,
        noise_xy_scale=0.35,
        temperature=0.5,
        residual_limit=0.12,
        residual_limit_xy=0.06,
        n_knots=4,
        swing_clearance=0.02,
        toe_offset_x=TOE_OFFSET_X,
        heel_offset_x=-0.04,
        mid_offset_x=0.10,
        contact_point_weight=1000.0,
        contact_point_pen_weight=3000.0,
        stance_slip_weight=2000.0,
        action_w_pelvis_z=2.0,
        action_w_foot_x=150.0,
        action_w_foot_y=500.0,
        action_w_foot_z=2.0,
        pelvis_height_weight=500.0,
        soft_reach_limit=0.70,
        reach_penalty_weight=5000.0,
        seed=42,
    ):
        self.terrain = terrain
        self.foot_nominal_z = float(foot_nominal_z)
        self.pelvis_height_above_foot = float(pelvis_height_above_foot)
        self.pelvis_height_weight = float(pelvis_height_weight)
        self.soft_reach_limit = max(float(soft_reach_limit), 0.1)
        self.reach_penalty_weight = max(float(reach_penalty_weight), 0.0)
        self.n_samples = max(2, int(n_samples))
        self.n_iterations = max(1, int(n_iterations))
        self.noise_std = float(noise_std)
        self.noise_xy_scale = max(float(noise_xy_scale), 0.0)
        self.temperature = max(float(temperature), 1e-6)
        self.residual_limit = max(float(residual_limit), 1e-6)
        self.residual_limit_xy = max(float(residual_limit_xy), 1e-6)
        self.n_knots = max(2, int(n_knots))
        self.swing_clearance = max(float(swing_clearance), 0.0)
        self.toe_offset_x = max(float(toe_offset_x), 0.0)
        self.heel_offset_x = float(heel_offset_x)
        self.mid_offset_x = float(mid_offset_x)
        self.contact_point_weight = max(float(contact_point_weight), 0.0)
        self.contact_point_pen_weight = max(float(contact_point_pen_weight), 0.0)
        self.stance_slip_weight = max(float(stance_slip_weight), 0.0)
        self.action_weights = np.asarray(
            [
                action_w_pelvis_z,
                action_w_foot_x,
                action_w_foot_y,
                action_w_foot_z,
                action_w_foot_x,
                action_w_foot_y,
                action_w_foot_z,
            ],
            dtype=np.float64,
        )
        self.action_weights = np.maximum(self.action_weights, 0.0)
        self.noise_scale = np.asarray(
            [
                1.0,
                self.noise_xy_scale,
                self.noise_xy_scale,
                1.0,
                self.noise_xy_scale,
                self.noise_xy_scale,
                1.0,
            ],
            dtype=np.float64,
        )
        self.residual_limit_vec = np.asarray(
            [
                self.residual_limit,
                self.residual_limit_xy,
                self.residual_limit_xy,
                self.residual_limit,
                self.residual_limit_xy,
                self.residual_limit_xy,
                self.residual_limit,
            ],
            dtype=np.float64,
        )
        self.foot_point_offsets_x = np.asarray(
            [self.heel_offset_x, self.mid_offset_x, self.toe_offset_x], dtype=np.float64
        )
        self.rng = np.random.default_rng(seed)
        self.current_knots = np.zeros((self.n_knots, self.ACTION_DIM), dtype=np.float64)
        self.current_residual = np.zeros((1, self.ACTION_DIM), dtype=np.float64)
        self.current_action = np.zeros(self.ACTION_DIM, dtype=np.float64)
        self.last_cost = np.inf
        self.last_cost_terms = {}

        # Precompute spline basis matrices for fast expansion/compression.
        # Since knot locations are fixed (evenly spaced in [0,1]), the
        # CubicSpline mapping from knot values to dense values is linear:
        #   dense = B @ knots  (for each action channel).
        # We build B once here, then _expand_action is a single matmul.
        #
        # To preserve C2 smoothness, we NEVER clip the dense output.
        # Instead, knot-space limits are tightened by the spline overshoot
        # factor so that B @ knots stays within physical bounds.
        self._basis_cache = {}  # horizon -> (B, B_pinv)
        self._knot_limit_vec = None  # set on first _get_basis call

    def _clip_knots(self, knots):
        """Clip in knot space using tightened limits that guarantee dense bounds."""
        if self._knot_limit_vec is not None:
            return np.clip(knots, -self._knot_limit_vec, self._knot_limit_vec)
        return np.clip(knots, -self.residual_limit_vec, self.residual_limit_vec)

    def _clip_action(self, action):
        """Clip in knot space (backward-compat alias used in replan for knots)."""
        return self._clip_knots(action)

    def _get_basis(self, horizon):
        """Get (or build and cache) the spline basis matrix for a given horizon.

        Returns (B, B_pinv) where:
          B:      (horizon, n_knots) — maps knot values to dense trajectory
          B_pinv: (n_knots, horizon) — least-squares inverse for compression
        """
        horizon = max(int(horizon), 2)
        if horizon in self._basis_cache:
            return self._basis_cache[horizon]

        knot_t = np.linspace(0.0, 1.0, self.n_knots, dtype=np.float64)
        traj_t = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
        B = np.zeros((horizon, self.n_knots), dtype=np.float64)
        for k in range(self.n_knots):
            unit = np.zeros(self.n_knots, dtype=np.float64)
            unit[k] = 1.0
            cs = CubicSpline(knot_t, unit, bc_type='clamped')
            B[:, k] = cs(traj_t)
        B_pinv = np.linalg.pinv(B)
        self._basis_cache[horizon] = (B, B_pinv)

        # Compute the worst-case overshoot factor and tighten knot limits.
        # If all knots are at ±L, the dense output can reach ±L*overshoot.
        # To guarantee dense stays within ±residual_limit, knot limits must
        # be residual_limit / overshoot.  We compute overshoot as the max
        # L1 row norm of B (worst case when all knots conspire).
        if self._knot_limit_vec is None:
            overshoot = np.max(np.sum(np.abs(B), axis=1))
            # overshoot >= 1.0 always (rows sum to 1, but |entries| can exceed)
            overshoot = max(overshoot, 1.0)
            self._knot_limit_vec = self.residual_limit_vec / overshoot

        return B, B_pinv

    def _expand_action(self, knots, horizon):
        """Expand knots to dense trajectory via spline basis. No dense clipping."""
        horizon = max(int(horizon), 1)
        if horizon == 1:
            return knots[[0]].copy()

        B, _ = self._get_basis(horizon)
        # B: (horizon, n_knots), knots: (n_knots, ACTION_DIM)
        return B @ knots

    def _compress_action(self, action):
        """Compress dense trajectory back to knots via pseudoinverse."""
        horizon = max(int(action.shape[0]), 1)
        if horizon == 1:
            knots = np.repeat(action[[0]], self.n_knots, axis=0)
            return self._clip_knots(knots)

        _, B_pinv = self._get_basis(horizon)
        # B_pinv: (n_knots, horizon), action: (horizon, ACTION_DIM)
        knots = B_pinv @ action
        return self._clip_knots(knots)

    def _apply_action(self, action, pelvis, lfoot, rfoot):
        if action.shape[0] != pelvis.shape[0]:
            raise ValueError(
                f"Action horizon ({action.shape[0]}) does not match trajectory horizon ({pelvis.shape[0]})"
            )
        out_pelvis = pelvis.copy()
        out_lfoot = lfoot.copy()
        out_rfoot = rfoot.copy()
        out_pelvis[:, 2] += action[:, self.IDX_PELVIS_DZ]

        out_lfoot[:, 0] += action[:, self.IDX_LFOOT_DX]
        out_lfoot[:, 1] += action[:, self.IDX_LFOOT_DY]
        out_lfoot[:, 2] += action[:, self.IDX_LFOOT_DZ]

        out_rfoot[:, 0] += action[:, self.IDX_RFOOT_DX]
        out_rfoot[:, 1] += action[:, self.IDX_RFOOT_DY]
        out_rfoot[:, 2] += action[:, self.IDX_RFOOT_DZ]
        return out_pelvis, out_lfoot, out_rfoot

    def _cost_terms(
        self,
        knots,
        prev_action,
        ref_pelvis,
        ref_lfoot,
        ref_rfoot,
        left_contact,
        right_contact,
        lfoot_quats,
        rfoot_quats,
    ):
        horizon = int(ref_pelvis.shape[0])
        if horizon <= 0:
            return {
                "surface_raw": 0.0,
                "penetration_raw": 0.0,
                "stance_raw": 0.0,
                "point_contact_raw": 0.0,
                "point_pen_raw": 0.0,
                "clearance_raw": 0.0,
                "slip_raw": 0.0,
                "pelvis_raw": 0.0,
                "action_raw": 0.0,
                "smooth_raw": 0.0,
                "reach_raw": 0.0,
                "surface": 0.0,
                "penetration": 0.0,
                "stance": 0.0,
                "point_contact": 0.0,
                "point_pen": 0.0,
                "clearance": 0.0,
                "slip": 0.0,
                "pelvis": 0.0,
                "action": 0.0,
                "smooth": 0.0,
                "reach": 0.0,
                "total": 0.0,
            }
        action = self._expand_action(knots, horizon)
        pelvis, lfoot, rfoot = self._apply_action(action, ref_pelvis, ref_lfoot, ref_rfoot)

        left_contact = np.asarray(left_contact, dtype=bool)
        right_contact = np.asarray(right_contact, dtype=bool)
        if left_contact.shape[0] != horizon or right_contact.shape[0] != horizon:
            raise ValueError(
                "Contact masks must have the same horizon as the reference segment"
            )
        left_swing = ~left_contact
        right_swing = ~right_contact

        l_terrain = self.terrain.height_batch(lfoot[:, 0], lfoot[:, 1]) + self.foot_nominal_z
        r_terrain = self.terrain.height_batch(rfoot[:, 0], rfoot[:, 1]) + self.foot_nominal_z

        l_surface_err = lfoot[:, 2] - l_terrain
        r_surface_err = rfoot[:, 2] - r_terrain
        l_penetration = np.minimum(l_surface_err, 0.0)
        r_penetration = np.minimum(r_surface_err, 0.0)
        penetration_raw = 1200.0 * (np.sum(l_penetration ** 2) + np.sum(r_penetration ** 2))
        stance_raw = 600.0 * (
            np.sum(l_surface_err[left_contact] ** 2) +
            np.sum(r_surface_err[right_contact] ** 2)
        )

        # Multi-point support cost (heel, midfoot, toe).
        # Rotate each foot-local offset by the foot's world-frame quaternion
        # so that terrain is queried at the actual support point, not just +x.
        n_points = self.foot_point_offsets_x.shape[0]
        l_point_err = np.zeros((n_points, horizon), dtype=np.float64)
        r_point_err = np.zeros((n_points, horizon), dtype=np.float64)
        for k, x_off in enumerate(self.foot_point_offsets_x):
            off_local = np.array([x_off, 0.0, 0.0])
            l_off_w = quat_rotate_batch(lfoot_quats, off_local)
            r_off_w = quat_rotate_batch(rfoot_quats, off_local)
            l_point_terrain = (
                self.terrain.height_batch(
                    lfoot[:, 0] + l_off_w[:, 0], lfoot[:, 1] + l_off_w[:, 1]
                ) + self.foot_nominal_z
            )
            r_point_terrain = (
                self.terrain.height_batch(
                    rfoot[:, 0] + r_off_w[:, 0], rfoot[:, 1] + r_off_w[:, 1]
                ) + self.foot_nominal_z
            )
            l_point_err[k] = (lfoot[:, 2] + l_off_w[:, 2]) - l_point_terrain
            r_point_err[k] = (rfoot[:, 2] + r_off_w[:, 2]) - r_point_terrain

        point_contact_raw = self.contact_point_weight * (
            np.sum(l_point_err[:, left_contact] ** 2) +
            np.sum(r_point_err[:, right_contact] ** 2)
        )
        point_pen_raw = self.contact_point_pen_weight * (
            np.sum(np.minimum(l_point_err, 0.0) ** 2) +
            np.sum(np.minimum(r_point_err, 0.0) ** 2)
        )

        l_toe_err = l_point_err[-1]
        r_toe_err = r_point_err[-1]
        l_clearance_err = np.minimum(l_toe_err - self.swing_clearance, 0.0)
        r_clearance_err = np.minimum(r_toe_err - self.swing_clearance, 0.0)
        clearance_raw = 220.0 * (
            np.sum(l_clearance_err[left_swing] ** 2) +
            np.sum(r_clearance_err[right_swing] ** 2)
        )

        slip_raw = 0.0
        if horizon > 1:
            l_pair = left_contact[1:] & left_contact[:-1]
            if np.any(l_pair):
                l_dxy = lfoot[1:, 0:2] - lfoot[:-1, 0:2]
                slip_raw += self.stance_slip_weight * np.sum(l_dxy[l_pair] ** 2)
            r_pair = right_contact[1:] & right_contact[:-1]
            if np.any(r_pair):
                r_dxy = rfoot[1:, 0:2] - rfoot[:-1, 0:2]
                slip_raw += self.stance_slip_weight * np.sum(r_dxy[r_pair] ** 2)

        # Pelvis height cost: target a fixed height above the mean stance-foot
        # terrain surface. Uses the average of both foot terrain heights as the
        # ground reference so the pelvis stays centered between split-level feet.
        pelvis_terrain = 0.5 * (l_terrain + r_terrain)
        pelvis_target_z = pelvis_terrain + self.pelvis_height_above_foot
        pelvis_err = pelvis[:, 2] - pelvis_target_z
        pelvis_raw = self.pelvis_height_weight * np.sum(pelvis_err ** 2)

        # Layer 2: reach penalty — penalize pelvis-to-foot distances that
        # exceed the soft reach limit.  Higher weight during stance so MPPI
        # avoids asking IK for impossible contact targets.
        L = self.soft_reach_limit
        l_dist = np.sqrt(np.sum((pelvis - lfoot) ** 2, axis=1))
        r_dist = np.sqrt(np.sum((pelvis - rfoot) ** 2, axis=1))
        l_over = np.maximum(l_dist - L, 0.0)
        r_over = np.maximum(r_dist - L, 0.0)
        # 2x weight during stance (must be reachable for contact)
        l_reach_w = np.where(left_contact, 2.0, 1.0)
        r_reach_w = np.where(right_contact, 2.0, 1.0)
        reach_raw = self.reach_penalty_weight * (
            np.sum(l_reach_w * l_over ** 2) + np.sum(r_reach_w * r_over ** 2)
        )

        action_raw = np.sum((action ** 2) * self.action_weights[np.newaxis, :])

        smooth_step = (
            np.sum((np.diff(action, axis=0) ** 2) * self.action_weights[np.newaxis, :])
            if horizon > 1
            else 0.0
        )
        smooth_transition = np.sum(((action[0] - prev_action[0]) ** 2) * self.action_weights)
        smooth_raw = 2.0 * (smooth_step + smooth_transition)

        surface_raw = (
            penetration_raw +
            stance_raw +
            point_contact_raw +
            point_pen_raw +
            clearance_raw
        )

        return {
            "surface_raw": float(surface_raw),
            "penetration_raw": float(penetration_raw),
            "stance_raw": float(stance_raw),
            "point_contact_raw": float(point_contact_raw),
            "point_pen_raw": float(point_pen_raw),
            "clearance_raw": float(clearance_raw),
            "slip_raw": float(slip_raw),
            "pelvis_raw": float(pelvis_raw),
            "action_raw": float(action_raw),
            "smooth_raw": float(smooth_raw),
            "reach_raw": float(reach_raw),
            "surface": float(surface_raw / horizon),
            "penetration": float(penetration_raw / horizon),
            "stance": float(stance_raw / horizon),
            "point_contact": float(point_contact_raw / horizon),
            "point_pen": float(point_pen_raw / horizon),
            "clearance": float(clearance_raw / horizon),
            "slip": float(slip_raw / horizon),
            "pelvis": float(pelvis_raw / horizon),
            "action": float(action_raw / horizon),
            "smooth": float(smooth_raw / horizon),
            "reach": float(reach_raw / horizon),
            "total": float(
                (surface_raw + slip_raw + pelvis_raw + action_raw + smooth_raw + reach_raw) / horizon
            ),
        }

    def replan(self, ref_pelvis, ref_lfoot, ref_rfoot, left_contact, right_contact,
               lfoot_quats, rfoot_quats):
        horizon = int(ref_pelvis.shape[0])
        if horizon <= 0:
            return

        prev_action = self._expand_action(self.current_knots, horizon)
        mean = self.current_knots.copy()

        best_knots = mean.copy()
        best_terms = self._cost_terms(
            mean, prev_action, ref_pelvis, ref_lfoot, ref_rfoot,
            left_contact, right_contact, lfoot_quats, rfoot_quats,
        )
        best_cost = best_terms["total"]

        for _ in range(self.n_iterations):
            noise = self.rng.normal(
                scale=self.noise_std,
                size=(self.n_samples, self.n_knots, self.ACTION_DIM),
            )
            noise *= self.noise_scale[np.newaxis, np.newaxis, :]
            samples = self._clip_action(mean[np.newaxis, :, :] + noise)
            samples[0] = mean

            terms = [
                self._cost_terms(
                    sample,
                    prev_action,
                    ref_pelvis,
                    ref_lfoot,
                    ref_rfoot,
                    left_contact,
                    right_contact,
                    lfoot_quats,
                    rfoot_quats,
                )
                for sample in samples
            ]
            costs = np.array([entry["total"] for entry in terms], dtype=np.float64)

            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_cost = float(costs[best_idx])
                best_knots = samples[best_idx].copy()
                best_terms = terms[best_idx]

            shifted = costs - costs[best_idx]
            weights = np.exp(-shifted / self.temperature)
            weights = np.maximum(weights, 1e-30)
            weights /= np.sum(weights)
            mean = np.sum(weights[:, np.newaxis, np.newaxis] * samples, axis=0)
            mean = self._clip_action(mean)

        mean_terms = self._cost_terms(
            mean, prev_action, ref_pelvis, ref_lfoot, ref_rfoot,
            left_contact, right_contact, lfoot_quats, rfoot_quats,
        )
        if mean_terms["total"] < best_cost:
            best_knots = mean.copy()
            best_terms = mean_terms
            best_cost = mean_terms["total"]

        best_residual = self._expand_action(best_knots, horizon)
        self.current_residual = best_residual
        self.current_action = self.current_residual[0].copy()
        if horizon > 1:
            shifted_residual = np.vstack([best_residual[1:], best_residual[-1:]])
        else:
            shifted_residual = best_residual.copy()
        self.current_knots = self._compress_action(shifted_residual)
        self.last_cost = best_cost
        self.last_cost_terms = best_terms.copy()

    def apply(self, ref_pelvis, ref_lfoot, ref_rfoot):
        horizon = int(ref_pelvis.shape[0])
        action = self.current_residual
        if action.shape[0] != horizon:
            action = self._expand_action(self.current_knots, horizon)
        return self._apply_action(action, ref_pelvis, ref_lfoot, ref_rfoot)


def main():
    parser = argparse.ArgumentParser(description="Minimal stair MPPI demo")
    parser.add_argument("--motion", type=str, default="assets/motions/walk1_subject1.npz")
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--start_frame", type=int, default=700)
    parser.add_argument("--n_frames", type=int, default=1000)
    parser.add_argument("--future_horizon", type=int, default=40)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--contact_z_threshold", type=float, default=CONTACT_Z_THRESHOLD)
    parser.add_argument("--contact_speed_threshold", type=float, default=CONTACT_SPEED_THRESHOLD)
    parser.add_argument("--lookahead", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--mppi_samples", type=int, default=64)
    parser.add_argument("--mppi_iterations", type=int, default=1)
    parser.add_argument("--mppi_knots", type=int, default=4)
    parser.add_argument("--mppi_temperature", type=float, default=0.5)
    parser.add_argument("--mppi_limit", type=float, default=0.12)
    parser.add_argument("--mppi_limit_xy", type=float, default=0.06)
    parser.add_argument("--mppi_noise_xy_scale", type=float, default=0.35)
    parser.add_argument("--swing_clearance", type=float, default=0.02)
    parser.add_argument("--footstep_margin", type=float, default=0.10)
    parser.add_argument("--toe_offset_x", type=float, default=TOE_OFFSET_X)
    parser.add_argument("--heel_offset_x", type=float, default=-0.04)
    parser.add_argument("--mid_offset_x", type=float, default=0.10)
    parser.add_argument("--toe_margin", type=float, default=0.005)
    parser.add_argument("--action_w_pelvis_z", type=float, default=2.0)
    parser.add_argument("--action_w_foot_x", type=float, default=150.0)
    parser.add_argument("--action_w_foot_y", type=float, default=500.0)
    parser.add_argument("--action_w_foot_z", type=float, default=2.0)
    parser.add_argument("--contact_point_weight", type=float, default=1000.0)
    parser.add_argument("--contact_point_pen_weight", type=float, default=3000.0)
    parser.add_argument("--stance_slip_weight", type=float, default=2000.0)
    parser.add_argument("--swing_floor_margin", type=float, default=0.003)
    parser.add_argument("--no_ghost_ref", action="store_true")
    parser.add_argument("--ghost_alpha", type=float, default=0.35)
    parser.add_argument("--ghost_color", type=str, default="0.15,0.65,1.0")
    parser.add_argument("--ik_max_iters", type=int, default=30)
    parser.add_argument("--ik_tol", type=float, default=1e-3)
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument("--ik_step_size", type=float, default=0.5)
    parser.add_argument("--ik_orientation_weight", type=float, default=0.05)
    parser.add_argument("--ik_log_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--export_npz", type=str, default=None,
                        help="Path to save per-frame xz data for plotting (e.g. out.npz)")
    parser.add_argument("--export_motion_npz", type=str, default=None,
                        help="Save IK-resolved full-body motion in input npz format")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.future_horizon <= 0:
        raise ValueError(f"--future_horizon must be > 0, got {args.future_horizon}")
    if args.speed <= 0.0:
        raise ValueError(f"--speed must be > 0, got {args.speed}")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")
    if args.mppi_knots < 2:
        raise ValueError(f"--mppi_knots must be >= 2, got {args.mppi_knots}")
    if args.mppi_limit_xy <= 0.0:
        raise ValueError(f"--mppi_limit_xy must be > 0, got {args.mppi_limit_xy}")
    if args.mppi_noise_xy_scale < 0.0:
        raise ValueError(
            f"--mppi_noise_xy_scale must be >= 0, got {args.mppi_noise_xy_scale}"
        )
    if args.footstep_margin < 0.0:
        raise ValueError(f"--footstep_margin must be >= 0, got {args.footstep_margin}")
    if args.toe_offset_x < 0.0:
        raise ValueError(f"--toe_offset_x must be >= 0, got {args.toe_offset_x}")
    if not (args.heel_offset_x < args.mid_offset_x < args.toe_offset_x):
        raise ValueError(
            "--heel_offset_x < --mid_offset_x < --toe_offset_x must hold"
        )
    if args.toe_margin < 0.0:
        raise ValueError(f"--toe_margin must be >= 0, got {args.toe_margin}")
    if args.action_w_pelvis_z < 0.0:
        raise ValueError(f"--action_w_pelvis_z must be >= 0, got {args.action_w_pelvis_z}")
    if args.action_w_foot_x < 0.0:
        raise ValueError(f"--action_w_foot_x must be >= 0, got {args.action_w_foot_x}")
    if args.action_w_foot_y < 0.0:
        raise ValueError(f"--action_w_foot_y must be >= 0, got {args.action_w_foot_y}")
    if args.action_w_foot_z < 0.0:
        raise ValueError(f"--action_w_foot_z must be >= 0, got {args.action_w_foot_z}")
    if args.contact_point_weight < 0.0:
        raise ValueError(
            f"--contact_point_weight must be >= 0, got {args.contact_point_weight}"
        )
    if args.contact_point_pen_weight < 0.0:
        raise ValueError(
            f"--contact_point_pen_weight must be >= 0, got {args.contact_point_pen_weight}"
        )
    if args.stance_slip_weight < 0.0:
        raise ValueError(
            f"--stance_slip_weight must be >= 0, got {args.stance_slip_weight}"
        )
    if args.swing_floor_margin < 0.0:
        raise ValueError(
            f"--swing_floor_margin must be >= 0, got {args.swing_floor_margin}"
        )
    if not (0.0 <= args.ghost_alpha <= 1.0):
        raise ValueError(f"--ghost_alpha must be in [0, 1], got {args.ghost_alpha}")
    if args.ik_max_iters <= 0:
        raise ValueError(f"--ik_max_iters must be > 0, got {args.ik_max_iters}")
    if args.ik_tol <= 0.0:
        raise ValueError(f"--ik_tol must be > 0, got {args.ik_tol}")
    if args.ik_damping < 0.0:
        raise ValueError(f"--ik_damping must be >= 0, got {args.ik_damping}")
    if args.ik_step_size <= 0.0:
        raise ValueError(f"--ik_step_size must be > 0, got {args.ik_step_size}")
    if args.ik_orientation_weight < 0.0:
        raise ValueError(
            f"--ik_orientation_weight must be >= 0, got {args.ik_orientation_weight}"
        )

    if args.xml is None:
        args.xml = os.path.join(
            os.path.dirname(__file__),
            "..",
            "assets",
            "g1",
            "g1_29dof_scene_rubble.xml",
        )
    xml_path = os.path.abspath(args.xml)

    print("=" * 60)
    print("MINIMAL MPPI DEMO")
    print("=" * 60)

    print("[1] Loading terrain...")
    terrain = RaycastTerrain.from_xml_path(xml_path)
    terrain.print_info()

    print(f"\n[2] Loading motion (frames {args.start_frame}-{args.start_frame + args.n_frames})...")
    motion = MotionClip(
        args.motion,
        args.start_frame,
        args.n_frames,
        contact_z_threshold=args.contact_z_threshold,
        contact_speed_threshold=args.contact_speed_threshold,
    )
    if motion.fps <= 0.0:
        raise ValueError(f"Motion fps must be > 0, got {motion.fps}")
    if motion.n_frames <= 0:
        raise ValueError("Motion segment is empty after loading")
    print(f"  {motion.n_frames} frames @ {motion.fps}Hz")
    print(f"  Estimated nominal foot z = {motion.foot_nominal_z:.3f}m")
    print(f"  Estimated pelvis height above foot = {motion.pelvis_height_above_foot:.3f}m")

    print("\n[3] Building reference + tiny MPPI...")
    ref_builder = TerrainReference(
        terrain=terrain,
        motion=motion,
        lookahead=args.lookahead,
        smoothing_alpha=args.alpha,
        footstep_margin=args.footstep_margin,
        toe_offset_x=args.toe_offset_x,
        heel_offset_x=args.heel_offset_x,
        mid_offset_x=args.mid_offset_x,
        toe_margin=args.toe_margin,
        swing_floor_margin=args.swing_floor_margin,
    )
    planner = TinyMPPI(
        terrain=terrain,
        foot_nominal_z=motion.foot_nominal_z,
        pelvis_height_above_foot=motion.pelvis_height_above_foot,
        n_samples=args.mppi_samples,
        n_iterations=args.mppi_iterations,
        noise_std=args.noise,
        noise_xy_scale=args.mppi_noise_xy_scale,
        temperature=args.mppi_temperature,
        residual_limit=args.mppi_limit,
        residual_limit_xy=args.mppi_limit_xy,
        n_knots=args.mppi_knots,
        swing_clearance=args.swing_clearance,
        toe_offset_x=args.toe_offset_x,
        heel_offset_x=args.heel_offset_x,
        mid_offset_x=args.mid_offset_x,
        contact_point_weight=args.contact_point_weight,
        contact_point_pen_weight=args.contact_point_pen_weight,
        stance_slip_weight=args.stance_slip_weight,
        action_w_pelvis_z=args.action_w_pelvis_z,
        action_w_foot_x=args.action_w_foot_x,
        action_w_foot_y=args.action_w_foot_y,
        action_w_foot_z=args.action_w_foot_z,
        seed=args.seed,
    )

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joint_mapping = build_joint_mapping(model)
    ik_solver = G1GhostJacobianIK(
        model,
        max_iters=args.ik_max_iters,
        tol=args.ik_tol,
        damping=args.ik_damping,
        step_size=args.ik_step_size,
        orientation_weight=args.ik_orientation_weight,
    )

    ghost_model = None
    ghost_data = None
    ghost_vopt = None
    ghost_pert = None
    if not args.no_ghost_ref:
        ghost_rgb = parse_rgb(args.ghost_color)
        ghost_model = mujoco.MjModel.from_xml_path(xml_path)
        ghost_data = mujoco.MjData(ghost_model)
        for i in range(ghost_model.ngeom):
            ghost_model.geom_rgba[i, :3] = ghost_rgb
            ghost_model.geom_rgba[i, 3] = float(args.ghost_alpha)
        ghost_vopt = mujoco.MjvOption()
        ghost_pert = mujoco.MjvPerturb()
        print(
            f"[Ghost] reference ghost enabled (alpha={args.ghost_alpha:.2f}, "
            f"color={args.ghost_color})"
        )
    else:
        print("[Ghost] reference ghost disabled (--no_ghost_ref)")

    print("\n" + "=" * 60)
    print("PIPELINE")
    print("  1. Extract a future motion segment")
    print("  2. Build terrain-lifted pelvis + stance-lock/smooth-swing foot reference")
    print("  3. Solve a tiny MPPI over a time-varying residual profile")
    print("  4. Solve main-robot IK to track current MPPI pelvis/feet")
    if ghost_model is not None:
        print("  5. Render terrain-lifted reference as a ghost robot")
    print("=" * 60)

    frame = 0
    horizon = args.future_horizon
    ghost_prev_leg_qpos = None
    ik_pos_err_hist = []
    ik_rot_err_hist = []

    # Export recording: exactly motion.n_frames data points, saved once.
    N = motion.n_frames
    export_path = args.export_npz
    export_done = False
    if export_path is not None:
        rec_ref_pelvis = np.zeros((N, 3), dtype=np.float64)
        rec_ref_lfoot = np.zeros((N, 3), dtype=np.float64)
        rec_ref_rfoot = np.zeros((N, 3), dtype=np.float64)
        rec_mppi_pelvis = np.zeros((N, 3), dtype=np.float64)
        rec_mppi_lfoot = np.zeros((N, 3), dtype=np.float64)
        rec_mppi_rfoot = np.zeros((N, 3), dtype=np.float64)
        rec_sim_pelvis = np.zeros((N, 3), dtype=np.float64)
        rec_sim_lfoot = np.zeros((N, 3), dtype=np.float64)
        rec_sim_rfoot = np.zeros((N, 3), dtype=np.float64)
        rec_ground_at_lfoot = np.zeros(N, dtype=np.float64)
        rec_ground_at_rfoot = np.zeros(N, dtype=np.float64)
        rec_ground_at_pelvis = np.zeros(N, dtype=np.float64)
        rec_left_contact = np.zeros(N, dtype=bool)
        rec_right_contact = np.zeros(N, dtype=bool)
        rec_raw_pelvis = np.zeros((N, 3), dtype=np.float64)
        rec_raw_lfoot = np.zeros((N, 3), dtype=np.float64)
        rec_raw_rfoot = np.zeros((N, 3), dtype=np.float64)

    motion_export_path = args.export_motion_npz
    motion_export_done = False
    if motion_export_path is not None:
        # Build mapping: npz body index -> MuJoCo body id.
        # MuJoCo body order follows kinematic chains; npz uses interleaved L/R.
        body_mapping = build_body_mapping(model)
        nbody_export = len(NPZ_BODY_NAMES)  # 30
        rec_m_joint_pos = np.zeros((N, model.nq - 7), dtype=np.float64)
        rec_m_body_pos = np.zeros((N, nbody_export, 3), dtype=np.float64)
        rec_m_body_quat = np.zeros((N, nbody_export, 4), dtype=np.float64)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [motion.get_pelvis_pos(0)[0] + 0.5, 0.0, 0.5]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        while viewer.is_running():
            frame_start = time.time()
            t = frame % motion.n_frames

            seg_idx = motion.get_future_indices(t, horizon)
            raw_pelvis, raw_lfoot, raw_rfoot = motion.get_future_segment(t, horizon)
            seg_lcontact, seg_rcontact = motion.get_future_contacts(t, horizon)

            current_pelvis = motion.get_pelvis_pos(t)
            ref_builder.update(current_pelvis[0], current_pelvis[1])
            ref_pelvis, ref_lfoot, ref_rfoot, ref_lf_quats, ref_rf_quats = (
                ref_builder.build_segment(
                    raw_pelvis, raw_lfoot, raw_rfoot, seg_lcontact, seg_rcontact, seg_idx
                )
            )

            planner.replan(
                ref_pelvis, ref_lfoot, ref_rfoot, seg_lcontact, seg_rcontact,
                ref_lf_quats, ref_rf_quats,
            )
            mppi_pelvis, mppi_lfoot, mppi_rfoot = planner.apply(
                ref_pelvis, ref_lfoot, ref_rfoot
            )

            raw_joint_qpos = reorder_to_mujoco(motion.joint_pos[t], joint_mapping)
            raw_leg_qpos = raw_joint_qpos[ik_solver.leg_qpos_adr - 7]

            ik_result = ik_solver.solve(
                data,
                target_root_pos=mppi_pelvis[0],
                target_root_quat=motion.body_quat[t, NPZ_PELVIS],
                target_lf_pos=mppi_lfoot[0],
                target_lf_quat=ref_lf_quats[0],
                target_rf_pos=mppi_rfoot[0],
                target_rf_quat=ref_rf_quats[0],
                fixed_upper_body_qpos=raw_joint_qpos,
                initial_leg_qpos=raw_leg_qpos,
            )

            if ghost_prev_leg_qpos is not None:
                ik_prev_seed = ik_solver.solve(
                    data,
                    target_root_pos=mppi_pelvis[0],
                    target_root_quat=motion.body_quat[t, NPZ_PELVIS],
                    target_lf_pos=mppi_lfoot[0],
                    target_lf_quat=ref_lf_quats[0],
                    target_rf_pos=mppi_rfoot[0],
                    target_rf_quat=ref_rf_quats[0],
                    fixed_upper_body_qpos=raw_joint_qpos,
                    initial_leg_qpos=ghost_prev_leg_qpos,
                )
                if (ik_prev_seed.left_pos_err + ik_prev_seed.right_pos_err) < (
                    ik_result.left_pos_err + ik_result.right_pos_err
                ):
                    ik_result = ik_prev_seed

            # Layer 3: IK fail-safe — if IK diverged badly, hold previous
            # pose to avoid visual glitches.  Only update ghost_prev_leg_qpos
            # when the solution is reasonable so subsequent frames get a good seed.
            ik_pos_err = max(ik_result.left_pos_err, ik_result.right_pos_err)
            ik_rot_err = max(ik_result.left_rot_err, ik_result.right_rot_err)
            ik_ok = ik_pos_err < 0.05 and ik_rot_err < 1.5

            if ik_ok or ghost_prev_leg_qpos is None:
                data.qpos[:] = ik_result.qpos
                data.qvel[:] = 0.0
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                ghost_prev_leg_qpos = ik_result.leg_qpos.copy()
            else:
                # Hold previous pose — only update pelvis position so the
                # robot at least translates with the motion.
                data.qpos[:3] = mppi_pelvis[0]
                data.qpos[3:7] = motion.body_quat[t, NPZ_PELVIS]
                mujoco.mj_forward(model, data)

            ik_pos_err_hist.append(ik_pos_err)
            ik_rot_err_hist.append(ik_rot_err)

            # Record export data for the first full pass only.
            if export_path is not None and not export_done:
                rec_ref_pelvis[t] = ref_pelvis[0]
                rec_ref_lfoot[t] = ref_lfoot[0]
                rec_ref_rfoot[t] = ref_rfoot[0]
                rec_mppi_pelvis[t] = mppi_pelvis[0]
                rec_mppi_lfoot[t] = mppi_lfoot[0]
                rec_mppi_rfoot[t] = mppi_rfoot[0]
                rec_sim_pelvis[t] = data.qpos[:3].copy()
                rec_sim_lfoot[t] = data.xpos[ik_solver.left_foot_body_id].copy()
                rec_sim_rfoot[t] = data.xpos[ik_solver.right_foot_body_id].copy()
                rec_ground_at_lfoot[t] = terrain.height_at(
                    float(mppi_lfoot[0, 0]), float(mppi_lfoot[0, 1]))
                rec_ground_at_rfoot[t] = terrain.height_at(
                    float(mppi_rfoot[0, 0]), float(mppi_rfoot[0, 1]))
                rec_ground_at_pelvis[t] = terrain.height_at(
                    float(mppi_pelvis[0, 0]), float(mppi_pelvis[0, 1]))
                rec_left_contact[t] = motion.left_contact[t]
                rec_right_contact[t] = motion.right_contact[t]
                rec_raw_pelvis[t] = raw_pelvis[0]
                rec_raw_lfoot[t] = raw_lfoot[0]
                rec_raw_rfoot[t] = raw_rfoot[0]
                if t == N - 1:
                    np.savez_compressed(
                        export_path,
                        ref_pelvis=rec_ref_pelvis,
                        ref_lfoot=rec_ref_lfoot,
                        ref_rfoot=rec_ref_rfoot,
                        mppi_pelvis=rec_mppi_pelvis,
                        mppi_lfoot=rec_mppi_lfoot,
                        mppi_rfoot=rec_mppi_rfoot,
                        sim_pelvis=rec_sim_pelvis,
                        sim_lfoot=rec_sim_lfoot,
                        sim_rfoot=rec_sim_rfoot,
                        ground_at_lfoot=rec_ground_at_lfoot,
                        ground_at_rfoot=rec_ground_at_rfoot,
                        ground_at_pelvis=rec_ground_at_pelvis,
                        left_contact=rec_left_contact,
                        right_contact=rec_right_contact,
                        raw_pelvis=rec_raw_pelvis,
                        raw_lfoot=rec_raw_lfoot,
                        raw_rfoot=rec_raw_rfoot,
                    )
                    export_done = True
                    print(f"\n[Export] Saved {N} frames to {export_path}")

            if motion_export_path is not None and not motion_export_done:
                rec_m_joint_pos[t] = reorder_from_mujoco(data.qpos[7:], joint_mapping)
                for npz_i, mj_bid in enumerate(body_mapping):
                    rec_m_body_pos[t, npz_i] = data.xpos[mj_bid]
                    rec_m_body_quat[t, npz_i] = data.xquat[mj_bid]
                if t == N - 1:
                    dt_inv = float(motion.fps)
                    m_joint_vel = np.gradient(rec_m_joint_pos, axis=0) * dt_inv
                    m_body_lin_vel = np.gradient(rec_m_body_pos, axis=0) * dt_inv
                    m_body_ang_vel = np.zeros((N, nbody_export, 3), dtype=np.float64)
                    for bi in range(nbody_export):
                        m_body_ang_vel[:, bi] = quat_ang_vel(rec_m_body_quat[:, bi], dt_inv)
                    np.savez_compressed(
                        motion_export_path,
                        fps=np.array([motion.fps], dtype=np.float32),
                        joint_pos=rec_m_joint_pos.astype(np.float32),
                        joint_vel=m_joint_vel.astype(np.float32),
                        body_pos_w=rec_m_body_pos.astype(np.float32),
                        body_quat_w=rec_m_body_quat.astype(np.float32),
                        body_lin_vel_w=m_body_lin_vel.astype(np.float32),
                        body_ang_vel_w=m_body_ang_vel.astype(np.float32),
                    )
                    motion_export_done = True
                    print(f"\n[MotionExport] Saved {N} frames to {motion_export_path}")

            if ghost_data is not None:
                update_robot_pose(
                    ghost_data,
                    root_pos=ref_pelvis[0],
                    root_quat=motion.body_quat[t, NPZ_PELVIS],
                    joint_qpos=raw_joint_qpos,
                )
                mujoco.mj_forward(ghost_model, ghost_data)

            if args.log_every > 0 and (frame % args.log_every == 0):
                action = planner.current_action
                terms = planner.last_cost_terms
                print(
                    f"[MPPI] frame={t:03d} "
                    f"res=[pdz={action[0]:+.3f}, "
                    f"l=({action[1]:+.3f},{action[2]:+.3f},{action[3]:+.3f}), "
                    f"r=({action[4]:+.3f},{action[5]:+.3f},{action[6]:+.3f})] "
                    f"cost={planner.last_cost:.6f} "
                    f"(stance={terms['stance']:.6f}, pen={terms['penetration']:.6f}, "
                    f"pcontact={terms['point_contact']:.6f}, ppen={terms['point_pen']:.6f}, "
                    f"clearance={terms['clearance']:.6f}, slip={terms['slip']:.6f}, "
                    f"pelvis={terms['pelvis']:.6f}, "
                    f"action={terms['action']:.6f}, smooth={terms['smooth']:.6f})"
                )
            if args.ik_log_every > 0 and (frame % args.ik_log_every == 0):
                print(
                    f"[IK] frame={t:03d} "
                    f"Lpos={ik_result.left_pos_err*1000.0:.2f}mm "
                    f"Rpos={ik_result.right_pos_err*1000.0:.2f}mm "
                    f"Lrot={ik_result.left_rot_err:.4f} "
                    f"Rrot={ik_result.right_rot_err:.4f} "
                    f"conv={int(ik_result.converged)} iter={ik_result.iterations}"
                )

            viewer.cam.lookat[0] = data.qpos[0]
            viewer.cam.lookat[1] = data.qpos[1]

            with viewer.lock():
                viewer.user_scn.ngeom = 0

                if ghost_model is not None:
                    mujoco.mjv_addGeoms(
                        ghost_model,
                        ghost_data,
                        ghost_vopt,
                        ghost_pert,
                        mujoco.mjtCatBit.mjCAT_ALL.value,
                        viewer.user_scn,
                    )

                add_trail(viewer.user_scn, ref_lfoot, [0.0, 0.9, 0.0, 0.55], size=0.010, stride=2)
                add_trail(viewer.user_scn, ref_rfoot, [0.9, 0.0, 0.0, 0.55], size=0.010, stride=2)
                add_trail(viewer.user_scn, ref_pelvis, [1.0, 0.9, 0.0, 0.35], size=0.008, stride=2)

                add_trail(viewer.user_scn, mppi_lfoot, [0.3, 0.5, 1.0, 0.70], size=0.010, stride=2)
                add_trail(viewer.user_scn, mppi_rfoot, [0.0, 0.2, 0.8, 0.70], size=0.010, stride=2)
                add_trail(viewer.user_scn, mppi_pelvis, [0.0, 0.8, 1.0, 0.45], size=0.008, stride=2)
                add_current_markers(viewer.user_scn, mppi_pelvis[0], mppi_lfoot[0], mppi_rfoot[0])

            viewer.sync()
            frame += 1

            dt = (1.0 / motion.fps) / args.speed
            elapsed = time.time() - frame_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    if ik_pos_err_hist:
        pos_mm = 1000.0 * np.asarray(ik_pos_err_hist, dtype=np.float64)
        rot = np.asarray(ik_rot_err_hist, dtype=np.float64)
        print("\nIK summary:")
        print(
            f"  position error: p95={np.percentile(pos_mm, 95.0):.2f}mm, "
            f"max={np.max(pos_mm):.2f}mm"
        )
        print(
            f"  orientation error: p95={np.percentile(rot, 95.0):.4f}rad, "
            f"max={np.max(rot):.4f}rad"
        )


if __name__ == "__main__":
    main()
