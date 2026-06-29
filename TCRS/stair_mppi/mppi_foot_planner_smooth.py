"""
MPPI 足部轨迹规划器 — 楼梯攀爬场景的 MuJoCo 可视化。

采样式 MPPI (cubic-spline 控制点 + softmax 加权) 优化摆动足轨迹。
保留 IK 模块用于渲染全身关节机器人。

完整管线 (Pipeline):
  1. 从 .npz 加载 MotionClip（动作捕捉数据）
  2. 构建 TerrainReference + FootstepResolver（复用 minimal_mppi_demo 中的类）
  3. 对所有摆动阶段预计算 MPPI 优化的足部轨迹
  4. 使用 SupportAwareRootZFilter 滤波骨盆 z 坐标
  5. 逐帧 IK 求解 → MuJoCo 可视化渲染

用法:
    conda run -n env_isaaclab python -m stair_mppi.mppi_foot_planner_smooth
    conda run -n env_isaaclab python -m stair_mppi.mppi_foot_planner_smooth --xml assets/g1/g1_29dof_scene_stairs_ud_mesh.xml
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from stair_mppi.ghost_ik import (
    G1GhostJacobianIK,
    GhostIKResult,
    FrameIKTargets,
    TOE_OFFSET,
    HEEL_OFFSET,
)

# 中足虚拟中心点：脚尖偏移和脚跟偏移的平均值，定义在 ankle_roll_link 局部坐标系下。
# MPPI 在这个虚拟点上规划（而非在脚踝原点），使脚尖和脚跟到轨迹的距离相等，
# 减少台阶边缘处的穿透现象。
MID_FOOT_OFFSET = 0.5 * (TOE_OFFSET + HEEL_OFFSET)  # [0.035, 0, -0.03]
IK_EDGE_HEIGHT_TOL = 0.02
IK_UNSUPPORTED_POINT_SCALE = 0.20
from stair_mppi.terrain import RaycastTerrain  # 光线投射地形查询工具
from stair_mppi.minimal_mppi_demo import (
    MotionClip,           # 动作捕捉数据加载器 (npz 格式)
    TerrainReference,     # 地形参考轨迹构建器 (将原始动捕脚位适配到目标地形)
    JOINT_NAMES_POLICY_ORDER,
    build_joint_mapping,  # 构建策略关节顺序 → MuJoCo 执行器索引的映射
    build_body_mapping,   # 构建 npz 刚体索引 → MuJoCo 刚体 ID 的映射
    reorder_to_mujoco,    # 将策略关节顺序重排为 MuJoCo 顺序
    reorder_from_mujoco,  # 将 MuJoCo 关节顺序重排为策略顺序
    update_robot_pose,    # 更新 MuJoCo data 的根位姿和关节角
    add_sphere,           # 在场景中添加球体标记
    add_trail,            # 在场景中添加轨迹线 (连续小球)
    add_current_markers,  # 添加当前帧的目标标记 (骨盆+左右脚)
    add_box,              # 在场景中添加方块标记
    parse_rgb,            # 解析 "r,g,b" 字符串为 RGB 浮点数组
    quat_ang_vel,         # 从四元数序列计算角速度
    NPZ_PELVIS,           # 骨盆在 npz 刚体数组中的索引 = 0
    NPZ_LFOOT,            # 左脚在 npz 刚体数组中的索引 = 18
    NPZ_RFOOT,            # 右脚在 npz 刚体数组中的索引 = 19
    NPZ_BODY_NAMES,       # npz 格式中刚体名称列表
    CONTACT_Z_THRESHOLD,  # 触地检测：脚踝 z 高度阈值
    CONTACT_SPEED_THRESHOLD,  # 触地检测：脚踝速度阈值
    TOE_OFFSET_X,         # 脚尖相对脚踝的 x 方向偏移
)
from stair_mppi.mppi_foot import (
    MppiFootOptimizer,    # MPPI 采样式轨迹优化器 (cubic spline 控制点)
    MppiFootParams,       # MPPI 参数数据类
    extract_yaw_from_quat,  # 从四元数中提取 yaw 角
    slerp_quaternion,     # 四元数球面线性插值 (SLERP)
)
from stair_mppi.terrain_warp import (
    SupportAwareRootZFilter,  # 支撑感知骨盆 z 滤波器
    HIP_OFFSET_Z,
    L_LEG_MAX,
    PELVIS_ANKLE_REACH,
)
from stair_mppi.ghost_ik import LEFT_LEG_JOINT_NAMES, RIGHT_LEG_JOINT_NAMES

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

NPZ_LHIP = NPZ_BODY_NAMES.index("left_hip_pitch_link")
NPZ_RHIP = NPZ_BODY_NAMES.index("right_hip_pitch_link")
NPZ_LKNEE = NPZ_BODY_NAMES.index("left_knee_link")
NPZ_RKNEE = NPZ_BODY_NAMES.index("right_knee_link")


# ---------------------------------------------------------------------------
# MPPI 摆动规划器：预计算所有摆动阶段的足部轨迹
# ---------------------------------------------------------------------------

def _push_side(terrain, foot_pos, fwd_x, fwd_y):
    """Check side-wall penetration for one foot and return push vector."""
    return terrain.foot_side_penetration(
        float(foot_pos[0]), float(foot_pos[1]), float(foot_pos[2]),
        fwd_x, fwd_y, foot_half_width=0.05, foot_half_length=0.10,
    )


def _find_swing_phases(contact_mask):
    """查找所有摆动阶段的 (start, end) 索引对。

    摆动阶段 = contact_mask 中连续 False 的区间。
    start = 第一个摆动帧, end = 第一个重新触地帧。
    抬脚帧 (liftoff) = start - 1 (摆动前最后一个触地帧)。
    落脚帧 (landing) = end (摆动后第一个触地帧)。
    """
    phases = []
    n = len(contact_mask)
    i = 0
    while i < n:
        if not contact_mask[i]:          # 找到一个非触地帧，开始记录摆动阶段
            j = i
            while j < n and not contact_mask[j]:  # 向后扫描直到触地或到达末尾
                j += 1
            phases.append((i, j))        # 记录 [i, j) 为一个完整摆动阶段
            i = j                         # 跳到摆动结束处继续扫描
        else:
            i += 1                        # 当前帧是触地，跳过
    return phases


def _find_stance_phases(contact_mask):
    """查找所有支撑阶段的 (start, end) 索引对。

    支撑阶段 = contact_mask 中连续 True 的区间。
    返回 (start, end) 列表，contact_mask[start:end] 全为 True。
    """
    phases = []
    n = len(contact_mask)
    i = 0
    while i < n:
        if contact_mask[i]:              # 找到一个触地帧，开始记录支撑阶段
            j = i
            while j < n and contact_mask[j]:  # 向后扫描直到非触地或到达末尾
                j += 1
            phases.append((i, j))        # 记录 [i, j) 为一个完整支撑阶段
            i = j
        else:
            i += 1
    return phases


def _yaw_only_quat(yaw):
    """构造仅含 yaw 角的四元数 [w, x, y, z] — 脚面平放，无 pitch/roll。

    对于楼梯等离散平面地形，脚应始终保持水平。
    有限差分计算地形法向量时可能击中台阶立面 (riser)，产生错误的 45° pitch，
    使用纯 yaw 四元数可以彻底避免这个问题。
    """
    half = yaw * 0.5  # 半角公式: quat = [cos(θ/2), 0, 0, sin(θ/2)] 表示绕 z 轴旋转 θ
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def _interpolate_yaw_quats(yaw0, yaw1, n):
    """在两个 yaw 角之间线性插值，返回纯 yaw 四元数。

    使用最短弧插值 (shortest-arc)：将角度差包裹到 [-π, π] 范围内，
    避免跨越 ±π 边界时走远路。
    返回 (n, 4) 形状数组，每行是 [w,x,y,z] 四元数，pitch=roll=0。
    """
    # 最短弧角度差：包裹到 [-π, π]
    delta = yaw1 - yaw0
    delta = (delta + np.pi) % (2 * np.pi) - np.pi  # 确保插值走最短路径
    t_frac = np.linspace(0.0, 1.0, n)  # 0→1 的 n 个等距插值参数
    yaws = yaw0 + delta * t_frac  # 线性插值 yaw 角
    out = np.zeros((n, 4), dtype=np.float64)
    for i, yaw in enumerate(yaws):
        out[i] = _yaw_only_quat(yaw)  # 每个 yaw 角转换为纯 yaw 四元数
    return out


def _quat_to_rotmat(q):
    """将四元数 [w,x,y,z] 转换为 3×3 旋转矩阵。

    使用标准四元数→旋转矩阵公式。用于将局部坐标系下的偏移向量
    (如中足偏移 MID_FOOT_OFFSET) 旋转到世界坐标系。
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _ankle_to_midfoot(ankle_pos, quat):
    """将脚踝位置转换为中足虚拟中心位置。

    公式: midfoot_world = ankle_world + R(quat) @ MID_FOOT_OFFSET
    R 将局部坐标系下的中足偏移旋转到世界坐标系后叠加到脚踝位置。
    """
    R = _quat_to_rotmat(quat)  # 脚踝朝向 → 旋转矩阵
    return ankle_pos + R @ MID_FOOT_OFFSET  # 脚踝 + 旋转后的局部偏移 = 中足世界坐标


def _midfoot_to_ankle(midfoot_pos, quat):
    """将中足虚拟中心位置反变换回脚踝位置。

    公式: ankle_world = midfoot_world - R(quat) @ MID_FOOT_OFFSET
    即 _ankle_to_midfoot 的逆运算。
    """
    R = _quat_to_rotmat(quat)
    return midfoot_pos - R @ MID_FOOT_OFFSET  # 中足 - 旋转后的偏移 = 脚踝世界坐标


def _contact_point_scale_for_pose(
    foot_pos,
    foot_quat,
    terrain,
    is_contact,
    support_height_tol=IK_EDGE_HEIGHT_TOL,
    unsupported_scale=IK_UNSUPPORTED_POINT_SCALE,
):
    """Return [ankle, toe, heel] scales for support-aware foot IK."""
    scales = np.ones(3, dtype=np.float64)
    if not is_contact:
        return scales

    R = _quat_to_rotmat(foot_quat)
    ankle_surface = float(terrain.height_at(float(foot_pos[0]), float(foot_pos[1])))
    toe_world = foot_pos + R @ TOE_OFFSET
    heel_world = foot_pos + R @ HEEL_OFFSET
    toe_surface = float(terrain.height_at(float(toe_world[0]), float(toe_world[1])))
    heel_surface = float(terrain.height_at(float(heel_world[0]), float(heel_world[1])))

    if abs(toe_surface - ankle_surface) > support_height_tol:
        scales[1] = unsupported_scale
    if abs(heel_surface - ankle_surface) > support_height_tol:
        scales[2] = unsupported_scale
    return scales


def _precompute_contact_point_scales(
    foot_positions,
    foot_quats,
    contact_mask,
    terrain,
    support_height_tol=IK_EDGE_HEIGHT_TOL,
    unsupported_scale=IK_UNSUPPORTED_POINT_SCALE,
):
    """Precompute per-frame [ankle, toe, heel] IK scales."""
    scales = np.ones((len(contact_mask), 3), dtype=np.float64)
    for i in np.flatnonzero(contact_mask):
        scales[i] = _contact_point_scale_for_pose(
            foot_positions[i],
            foot_quats[i],
            terrain,
            is_contact=True,
            support_height_tol=support_height_tol,
            unsupported_scale=unsupported_scale,
        )
    return scales


def _safe_normalize(vec, fallback):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64).copy()
    return vec / norm


# ---------------------------------------------------------------------------
# Continuous optimisation modules (no discrete branching)
# ---------------------------------------------------------------------------

_MAX_REACH = PELVIS_ANKLE_REACH - 0.01


def _should_trust_phase_clock(phase_params, fit_quality_threshold=0.72):
    """Only use phase-clock outputs when the periodic fit is clearly reliable."""
    if phase_params is None:
        return False
    gait_type = getattr(phase_params, "gait_type", "")
    fit_quality = float(getattr(phase_params, "fit_quality", 0.0))
    if gait_type not in {"walk", "run", "hop", "gallop", "stand"}:
        return False
    return fit_quality >= fit_quality_threshold


def _support_state(motion, frame_idx, phase_clock_trusted):
    """Return (left_weight, right_weight, is_flight) for one frame."""
    if phase_clock_trusted and motion.per_frame_phase is not None:
        return (
            float(motion.per_frame_phase.support_weight_left[frame_idx]),
            float(motion.per_frame_phase.support_weight_right[frame_idx]),
            bool(motion.per_frame_phase.flight_mask[frame_idx]),
        )
    return (
        float(motion.left_contact[frame_idx]),
        float(motion.right_contact[frame_idx]),
        False,
    )


def _pelvis_reach_z_upper_bound(pelvis_xy, foot_pos, support_weight, max_reach):
    """Maximum pelvis z allowed by one pelvis-ankle pair at the current xy."""
    pelvis_xy = np.asarray(pelvis_xy, dtype=np.float64)
    foot_pos = np.asarray(foot_pos, dtype=np.float64)
    support = float(np.clip(support_weight, 0.0, 1.0))
    xy_dist = float(np.linalg.norm(foot_pos[:2] - pelvis_xy[:2]))
    xy_dist = min(xy_dist, float(max_reach))
    vertical_allow = np.sqrt(max(float(max_reach) ** 2 - xy_dist ** 2, 0.0))
    return float(foot_pos[2] + vertical_allow + (1.0 - support) * 0.04)


def _continuous_ik_weights(
    base_root_xy_weight,
    base_root_z_weight,
    left_point_scale,
    right_point_scale,
    support_weight_left,
    support_weight_right,
):
    """Derive IK weights by smooth interpolation from continuous support weights.

    No if/else state machine — every output is a smooth function of the two
    scalar support weights in [0,1].

    When total support is high (double stance), weights stay at nominal.
    When total support drops (swing / flight), root xy loosens and foot
    point scales are attenuated proportionally.
    """
    lw = float(support_weight_left)
    rw = float(support_weight_right)
    total = lw + rw  # 0 (flight) … 2 (double stance)

    # confidence ∈ [0, 1]: 0 = flight, 1 = solid double stance
    confidence = float(np.clip(total / 2.0, 0.0, 1.0))

    # root xy: full weight at confidence=1, 55% at confidence=0
    xy_scale = 0.55 + 0.45 * confidence
    # root z: full weight at confidence=1, 65% at confidence=0
    z_scale = 0.65 + 0.35 * confidence

    root_xy_weight = float(base_root_xy_weight) * xy_scale
    root_z_weight = float(base_root_z_weight) * z_scale

    # foot point scales: attenuate toe/heel when that foot's own weight is low
    left_scale = np.asarray(left_point_scale, dtype=np.float64).copy()
    right_scale = np.asarray(right_point_scale, dtype=np.float64).copy()
    # ankle always keeps its scale; toe/heel soften with (1 - foot_support_weight)
    l_attenuation = 0.45 + 0.55 * lw  # 0.45 when fully swinging, 1.0 when stance
    r_attenuation = 0.45 + 0.55 * rw
    left_scale[1:] *= l_attenuation
    right_scale[1:] *= r_attenuation

    return root_xy_weight, root_z_weight, left_scale, right_scale


def _is_locked_double_stance(
    support_weight_left,
    support_weight_right,
    cur_pelvis,
    cur_lfoot,
    cur_rfoot,
    prev_pelvis=None,
    prev_lfoot=None,
    prev_rfoot=None,
    foot_tol=5e-4,
    pelvis_tol=1.5e-2,
):
    """Detect steady double-stance frames where multiseed branch-hopping hurts continuity.

    When both feet are in solid support and the latched foot targets are
    effectively unchanged from the previous frame, the best behavior is to stay
    on the previous IK branch. Re-running the multiseed bank there can flip
    between nearly equivalent local minima and create visible lower-body sway.
    """
    if (
        prev_pelvis is None
        or prev_lfoot is None
        or prev_rfoot is None
    ):
        return False
    if float(support_weight_left) < 0.95 or float(support_weight_right) < 0.95:
        return False

    foot_delta_l = float(np.linalg.norm(np.asarray(cur_lfoot) - np.asarray(prev_lfoot)))
    foot_delta_r = float(np.linalg.norm(np.asarray(cur_rfoot) - np.asarray(prev_rfoot)))
    pelvis_delta = float(np.linalg.norm(np.asarray(cur_pelvis) - np.asarray(prev_pelvis)))
    return (
        foot_delta_l <= float(foot_tol)
        and foot_delta_r <= float(foot_tol)
        and pelvis_delta <= float(pelvis_tol)
    )


def _project_pelvis_foot_to_reach(pelvis, foot, support_weight, max_reach):
    """Continuously pull one pelvis-foot pair back inside the reach envelope.

    The IK solver becomes poorly conditioned near full leg extension.  A soft
    least-squares penalty is not enough there: it can still hand IK targets
    that require the leg to be longer than the model permits.  This projection
    is intentionally one-dimensional along the pelvis-foot line, so it does not
    introduce contact-state jumps.

    Stance feet move very little; swing feet can absorb more of the correction.
    """
    pelvis = np.asarray(pelvis, dtype=np.float64).copy()
    foot = np.asarray(foot, dtype=np.float64).copy()
    rel = foot - pelvis
    dist = float(np.linalg.norm(rel))
    if dist <= max_reach or dist < 1e-8:
        return pelvis, foot

    direction = rel / dist
    excess = dist - float(max_reach)
    support = float(np.clip(support_weight, 0.0, 1.0))

    # support=1: keep the foot almost fixed and move pelvis toward it.
    # support=0: let swing foot absorb most of the correction.
    foot_fraction = 0.03 + 0.77 * (1.0 - support)
    pelvis_fraction = 1.0 - foot_fraction

    pelvis += pelvis_fraction * excess * direction
    foot -= foot_fraction * excess * direction
    return pelvis, foot


def _smooth_target_shaper(
    cur_pelvis,
    cur_lfoot,
    cur_rfoot,
    support_weight_left,
    support_weight_right,
    prev_shaped_pelvis=None,
    prev_shaped_lfoot=None,
    prev_shaped_rfoot=None,
    temporal_weight=2.0,
    support_pull_weight=3.0,
    reach_penalty_weight=20.0,
    max_reach=None,
    fps=50.0,
):
    """Conservative target shaper: only fix genuine reach violations.

    The previous least-squares version actively pulled pelvis/feet even in
    ordinary double-stance frames. That changed valid startup targets and
    created bad IK seeds. Here we keep the original targets unless the
    pelvis-foot distance actually exceeds the reach envelope.
    """
    if max_reach is None:
        max_reach = _MAX_REACH

    p0 = np.asarray(cur_pelvis, dtype=np.float64)
    l0 = np.asarray(cur_lfoot, dtype=np.float64)
    r0 = np.asarray(cur_rfoot, dtype=np.float64)
    lw = float(support_weight_left)
    rw = float(support_weight_right)
    reach_limit = max(float(max_reach), 0.1)
    over_l = float(np.linalg.norm(l0 - p0) - reach_limit)
    over_r = float(np.linalg.norm(r0 - p0) - reach_limit)
    if max(over_l, over_r) <= 1e-6:
        return p0, l0, r0

    shaped_p = p0.copy()
    shaped_l = l0.copy()
    shaped_r = r0.copy()
    for _ in range(3):
        shaped_p, shaped_l = _project_pelvis_foot_to_reach(shaped_p, shaped_l, lw, reach_limit)
        shaped_p, shaped_r = _project_pelvis_foot_to_reach(shaped_p, shaped_r, rw, reach_limit)

    return shaped_p, shaped_l, shaped_r


def _trajectory_ik_config_from_args(args):
    from stair_mppi.trajectory_ik import TrajectoryIKConfig

    return TrajectoryIKConfig(
        window_size=args.ik_window,
        commit_size=args.ik_commit,
        max_nfev=args.traj_ik_max_nfev,
        w_vel=args.traj_ik_w_vel,
        w_acc=args.traj_ik_w_acc,
        w_root_vel=args.traj_ik_w_root_vel,
        w_root_acc=args.traj_ik_w_root_acc,
        w_stance_lock=args.traj_ik_w_stance_lock,
        skip_init_max_err=args.traj_ik_skip_init_err_mm / 1000.0,
    )


def _build_ik_solver(model, args, log_config=False):
    ik_backend = getattr(args, "ik_backend", "jacobian")
    root_xy_weight = float(args.root_xy_weight)
    root_z_weight = float(args.root_z_weight)

    if ik_backend in ("curobo", "drake"):
        ik_config_path = os.path.abspath(args.ik_config) if args.ik_config else None
        if ik_backend == "curobo":
            from stair_mppi.ghost_ik_curobo import G1CuroboIK
            ik_solver = G1CuroboIK(model, ik_config_path=ik_config_path)
        else:
            from stair_mppi.ghost_ik_drake import G1DrakeIK
            ik_solver = G1DrakeIK(model, ik_config_path=ik_config_path)
        if args.ik_config:
            import json as _json
            with open(ik_config_path) as _f:
                _ik_cfg = _json.load(_f)
            _solver_cfg = _ik_cfg.get("solver", {})
            root_xy_weight = float(_solver_cfg.get("root_xy_weight", root_xy_weight))
            root_z_weight = float(_solver_cfg.get("root_z_weight", root_z_weight))
    elif args.ik_config:
        import json as _json
        ik_config_path = os.path.abspath(args.ik_config)
        if log_config:
            print(f"  Loading IK config from {ik_config_path}")
        ik_solver = G1GhostJacobianIK.from_config(model, ik_config_path)
        with open(ik_config_path) as _f:
            _ik_cfg = _json.load(_f)
        _solver_cfg = _ik_cfg.get("solver", {})
        root_xy_weight = float(_solver_cfg.get("root_xy_weight", root_xy_weight))
        root_z_weight = float(_solver_cfg.get("root_z_weight", root_z_weight))
        if log_config:
            print(
                f"  IK weights: ankle={ik_solver.point_weights[0]:.2f} "
                f"toe={ik_solver.point_weights[3]:.2f} "
                f"heel={ik_solver.point_weights[6]:.2f} "
                f"penetration={ik_solver.penetration_weight:.1f} "
                f"root_xy={root_xy_weight:.1f} root_z={root_z_weight:.1f}"
            )
    else:
        ik_solver = G1GhostJacobianIK(
            model,
            max_iters=args.ik_max_iters,
            tol=args.ik_tol,
            damping=args.ik_damping,
            step_size=args.ik_step_size,
            orientation_weight=args.ik_orientation_weight,
            posture_weight=args.ik_posture_weight,
            penetration_weight=args.ik_penetration_weight,
        )

    return ik_backend, ik_solver, root_xy_weight, root_z_weight


def _build_ghost_fk_model(xml_path):
    ghost_model = mujoco.MjModel.from_xml_path(xml_path)
    ghost_data = mujoco.MjData(ghost_model)
    return ghost_model, ghost_data


@dataclass
class LowerBodyStabilizerState:
    prev_leg_qpos: np.ndarray = None
    prev_root_pos: np.ndarray = None
    prev_left_contact: bool = False
    prev_right_contact: bool = False
    left_lock_pos: np.ndarray = None
    left_lock_quat: np.ndarray = None
    right_lock_pos: np.ndarray = None
    right_lock_quat: np.ndarray = None


def _scale_point_scale(point_scale, factor):
    scale = np.asarray(point_scale, dtype=np.float64).copy()
    return scale * float(max(factor, 0.0))


def _lower_body_stabilizer_enabled(args, ik_backend):
    return (
        ik_backend == "jacobian"
        and not getattr(args, "no_ik_lower_body_stabilizer", False)
    )


def _stabilizer_target_from_base(
    base_target,
    left_contact,
    right_contact,
    stab_state,
    args,
):
    left_pos = base_target.left_pos.copy()
    left_quat = base_target.left_quat.copy()
    right_pos = base_target.right_pos.copy()
    right_quat = base_target.right_quat.copy()

    if left_contact and stab_state.left_lock_pos is not None:
        left_pos = stab_state.left_lock_pos.copy()
        left_quat = stab_state.left_lock_quat.copy()
    if right_contact and stab_state.right_lock_pos is not None:
        right_pos = stab_state.right_lock_pos.copy()
        right_quat = stab_state.right_lock_quat.copy()

    left_scale = _scale_point_scale(
        base_target.left_point_scale,
        args.ik_stabilizer_stance_scale if left_contact else args.ik_stabilizer_swing_scale,
    )
    right_scale = _scale_point_scale(
        base_target.right_point_scale,
        args.ik_stabilizer_stance_scale if right_contact else args.ik_stabilizer_swing_scale,
    )

    root_xy_scale = 1.0
    root_z_scale = 1.0
    if left_contact or right_contact:
        root_xy_scale = float(args.ik_stabilizer_root_xy_scale)
        root_z_scale = float(args.ik_stabilizer_root_z_scale)
        if left_contact and right_contact:
            root_xy_scale *= 0.85
            root_z_scale *= 0.90

    return FrameIKTargets(
        root_pos=base_target.root_pos.copy(),
        root_quat=base_target.root_quat.copy(),
        left_pos=left_pos,
        left_quat=left_quat,
        right_pos=right_pos,
        right_quat=right_quat,
        fixed_upper_body_qpos=base_target.fixed_upper_body_qpos.copy(),
        ref_leg_qpos=base_target.ref_leg_qpos.copy() if base_target.ref_leg_qpos is not None else None,
        left_point_scale=left_scale,
        right_point_scale=right_scale,
        root_xy_weight=float(base_target.root_xy_weight) * root_xy_scale,
        root_z_weight=float(base_target.root_z_weight) * root_z_scale,
    )


def _run_lower_body_stabilizer_frame(
    data,
    ik_solver,
    base_target,
    base_state,
    left_contact,
    right_contact,
    stab_state,
    args,
):
    left_contact = bool(left_contact)
    right_contact = bool(right_contact)

    if not left_contact:
        stab_state.left_lock_pos = None
        stab_state.left_lock_quat = None
    if not right_contact:
        stab_state.right_lock_pos = None
        stab_state.right_lock_quat = None

    target = _stabilizer_target_from_base(
        base_target, left_contact, right_contact, stab_state, args
    )

    base_root, base_leg = ik_solver.unpack_state(base_state)
    if not left_contact and not right_contact:
        stab_state.prev_leg_qpos = base_leg.copy()
        stab_state.prev_root_pos = base_root.copy()
        stab_state.prev_left_contact = False
        stab_state.prev_right_contact = False
        return np.asarray(base_state, dtype=np.float64).copy()

    seed_leg = base_leg.copy()
    seed_root = base_root.copy()
    if stab_state.prev_leg_qpos is not None:
        seed_leg = 0.65 * base_leg + 0.35 * stab_state.prev_leg_qpos
    if stab_state.prev_root_pos is not None:
        seed_root = 0.65 * base_root + 0.35 * stab_state.prev_root_pos

    root_continuity_weight = float(args.ik_stabilizer_root_continuity_weight)
    if left_contact and right_contact:
        root_continuity_weight *= 1.5

    ik_result = ik_solver.solve_with_root(
        data,
        target_root_pos=target.root_pos,
        target_root_quat=target.root_quat,
        target_lf_pos=target.left_pos,
        target_lf_quat=target.left_quat,
        target_rf_pos=target.right_pos,
        target_rf_quat=target.right_quat,
        fixed_upper_body_qpos=target.fixed_upper_body_qpos,
        initial_leg_qpos=seed_leg,
        ref_leg_qpos=base_leg,
        root_xy_weight=target.root_xy_weight,
        root_z_weight=target.root_z_weight,
        left_point_scale=target.left_point_scale,
        right_point_scale=target.right_point_scale,
        allow_multiseed=False,
        force_multiseed=False,
        initial_root_pos=seed_root,
        root_continuity_weight=root_continuity_weight,
    )

    data.qpos[:] = ik_result.qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(ik_solver.model, data)

    entering_left = left_contact and not stab_state.prev_left_contact
    entering_right = right_contact and not stab_state.prev_right_contact
    if entering_left:
        stab_state.left_lock_pos = data.xpos[ik_solver.left_foot_body_id].copy()
        stab_state.left_lock_quat = data.xquat[ik_solver.left_foot_body_id].copy()
    if entering_right:
        stab_state.right_lock_pos = data.xpos[ik_solver.right_foot_body_id].copy()
        stab_state.right_lock_quat = data.xquat[ik_solver.right_foot_body_id].copy()

    stab_state.prev_leg_qpos = ik_result.leg_qpos.copy()
    stab_state.prev_root_pos = data.qpos[:3].copy()
    stab_state.prev_left_contact = left_contact
    stab_state.prev_right_contact = right_contact
    return ik_solver.pack_state(data.qpos[:3], ik_result.leg_qpos)


def _update_lower_body_stabilizer_memory(
    data,
    ik_solver,
    target,
    state_vec,
    left_contact,
    right_contact,
    stab_state,
):
    left_contact = bool(left_contact)
    right_contact = bool(right_contact)
    _apply_state_to_model(data, ik_solver, target, state_vec)

    if not left_contact:
        stab_state.left_lock_pos = None
        stab_state.left_lock_quat = None
    elif not stab_state.prev_left_contact:
        stab_state.left_lock_pos = data.xpos[ik_solver.left_foot_body_id].copy()
        stab_state.left_lock_quat = data.xquat[ik_solver.left_foot_body_id].copy()

    if not right_contact:
        stab_state.right_lock_pos = None
        stab_state.right_lock_quat = None
    elif not stab_state.prev_right_contact:
        stab_state.right_lock_pos = data.xpos[ik_solver.right_foot_body_id].copy()
        stab_state.right_lock_quat = data.xquat[ik_solver.right_foot_body_id].copy()

    _, leg_q = ik_solver.unpack_state(state_vec)
    stab_state.prev_leg_qpos = leg_q.copy()
    stab_state.prev_root_pos = data.qpos[:3].copy()
    stab_state.prev_left_contact = left_contact
    stab_state.prev_right_contact = right_contact


def _run_lower_body_stabilizer_sequence(
    data,
    ik_solver,
    frame_targets,
    left_contact,
    right_contact,
    base_states,
    args,
    base_frame_errors=None,
    progress_prefix=None,
    progress_every=200,
):
    stabilized = np.asarray(base_states, dtype=np.float64).copy()
    base_frame_errors = None if base_frame_errors is None else np.asarray(base_frame_errors, dtype=np.float64)
    stab_state = LowerBodyStabilizerState()
    n_frames = stabilized.shape[0]
    active_until = -1
    trigger_err = float(args.ik_stabilizer_trigger_mm) / 1000.0
    cooldown = max(int(args.ik_stabilizer_cooldown_frames), 0)

    for t in range(n_frames):
        need_stabilize = bool(left_contact[t] or right_contact[t])
        if base_frame_errors is not None:
            if base_frame_errors[t] > trigger_err:
                active_until = max(active_until, t + cooldown)
            need_stabilize = need_stabilize and (base_frame_errors[t] > trigger_err or t <= active_until)

        if need_stabilize:
            stabilized[t] = _run_lower_body_stabilizer_frame(
                data,
                ik_solver,
                frame_targets[t],
                stabilized[t],
                left_contact[t],
                right_contact[t],
                stab_state,
                args,
            )
        else:
            _update_lower_body_stabilizer_memory(
                data,
                ik_solver,
                frame_targets[t],
                stabilized[t],
                left_contact[t],
                right_contact[t],
                stab_state,
            )
        if progress_prefix and (t % progress_every == 0 or t == n_frames - 1):
            print(f"{progress_prefix} frame {t}/{n_frames}", flush=True)

    return stabilized


def _build_frame_ik_target_at(
    motion,
    pelvis_positions,
    mppi_lfoot,
    mppi_rfoot,
    mppi_lfoot_quats,
    mppi_rfoot_quats,
    joint_mapping,
    ik_solver,
    left_point_scales,
    right_point_scales,
    root_xy_weight,
    root_z_weight,
    phase_clock_trusted,
    t,
    prev_shaped_pelvis=None,
    prev_shaped_lfoot=None,
    prev_shaped_rfoot=None,
):
    cur_pelvis = pelvis_positions[t]
    cur_lfoot = mppi_lfoot[t]
    cur_rfoot = mppi_rfoot[t]
    cur_root_quat = motion.body_quat[t, NPZ_PELVIS]
    cur_lf_quat = mppi_lfoot_quats[t]
    cur_rf_quat = mppi_rfoot_quats[t]

    lw, rw, _ = _support_state(motion, t, phase_clock_trusted)
    cur_pelvis, cur_lfoot, cur_rfoot = _smooth_target_shaper(
        cur_pelvis,
        cur_lfoot,
        cur_rfoot,
        lw,
        rw,
        prev_shaped_pelvis,
        prev_shaped_lfoot,
        prev_shaped_rfoot,
        fps=float(motion.fps),
    )
    locked_double_stance = _is_locked_double_stance(
        lw,
        rw,
        cur_pelvis,
        cur_lfoot,
        cur_rfoot,
        prev_shaped_pelvis,
        prev_shaped_lfoot,
        prev_shaped_rfoot,
    )
    root_xy_w, root_z_w, left_scale, right_scale = _continuous_ik_weights(
        root_xy_weight,
        root_z_weight,
        left_point_scales[t],
        right_point_scales[t],
        lw,
        rw,
    )

    raw_joint_qpos = reorder_to_mujoco(motion.joint_pos[t], joint_mapping)
    ref_leg = ik_solver.extract_leg_qpos(
        np.concatenate([np.zeros(7, dtype=np.float64), raw_joint_qpos])
    )
    target = FrameIKTargets(
        root_pos=cur_pelvis.copy(),
        root_quat=cur_root_quat.copy(),
        left_pos=cur_lfoot.copy(),
        left_quat=cur_lf_quat.copy(),
        right_pos=cur_rfoot.copy(),
        right_quat=cur_rf_quat.copy(),
        fixed_upper_body_qpos=raw_joint_qpos.copy(),
        ref_leg_qpos=ref_leg.copy(),
        left_point_scale=np.asarray(left_scale, dtype=np.float64).copy(),
        right_point_scale=np.asarray(right_scale, dtype=np.float64).copy(),
        root_xy_weight=float(root_xy_w),
        root_z_weight=float(root_z_w),
    )
    return target, locked_double_stance, lw, rw


def _build_frame_ik_targets(
    motion,
    pelvis_positions,
    mppi_lfoot,
    mppi_rfoot,
    mppi_lfoot_quats,
    mppi_rfoot_quats,
    joint_mapping,
    ik_solver,
    left_point_scales,
    right_point_scales,
    root_xy_weight,
    root_z_weight,
    phase_clock_trusted,
):
    """Build the exact per-frame IK targets used by both frame and window IK."""
    N = motion.n_frames
    frame_targets = []
    locked_double_stance = np.zeros(N, dtype=bool)

    prev_shaped_pelvis = None
    prev_shaped_lfoot = None
    prev_shaped_rfoot = None
    for t in range(N):
        target, locked_double_stance[t], _, _ = _build_frame_ik_target_at(
            motion,
            pelvis_positions,
            mppi_lfoot,
            mppi_rfoot,
            mppi_lfoot_quats,
            mppi_rfoot_quats,
            joint_mapping,
            ik_solver,
            left_point_scales,
            right_point_scales,
            root_xy_weight,
            root_z_weight,
            phase_clock_trusted,
            t,
            prev_shaped_pelvis=prev_shaped_pelvis,
            prev_shaped_lfoot=prev_shaped_lfoot,
            prev_shaped_rfoot=prev_shaped_rfoot,
        )
        prev_shaped_pelvis = target.root_pos.copy()
        prev_shaped_lfoot = target.left_pos.copy()
        prev_shaped_rfoot = target.right_pos.copy()
        frame_targets.append(target)

    return frame_targets, locked_double_stance


def _solve_single_frame_ik(
    data,
    ik_solver,
    target,
    prev_leg_qpos=None,
    prev_root_pos=None,
    locked_double_stance=False,
    default_initial_leg_qpos=None,
):
    force_multiseed = bool(locked_double_stance and prev_root_pos is not None)
    root_continuity_w = 120.0 if locked_double_stance else 0.0
    ik_result = ik_solver.solve_with_root(
        data,
        target_root_pos=target.root_pos,
        target_root_quat=target.root_quat,
        target_lf_pos=target.left_pos,
        target_lf_quat=target.left_quat,
        target_rf_pos=target.right_pos,
        target_rf_quat=target.right_quat,
        fixed_upper_body_qpos=target.fixed_upper_body_qpos,
        initial_leg_qpos=prev_leg_qpos if prev_leg_qpos is not None else default_initial_leg_qpos,
        ref_leg_qpos=target.ref_leg_qpos,
        root_xy_weight=target.root_xy_weight,
        root_z_weight=target.root_z_weight,
        left_point_scale=target.left_point_scale,
        right_point_scale=target.right_point_scale,
        force_multiseed=force_multiseed,
        initial_root_pos=prev_root_pos,
        root_continuity_weight=root_continuity_w,
    )
    state_vec = ik_solver.pack_state(ik_result.qpos[:3], ik_result.leg_qpos)
    root_err = 0.0 if ik_result.root_pos_offset is None else float(np.linalg.norm(ik_result.root_pos_offset))
    frame_err = max(root_err, float(ik_result.left_pos_err), float(ik_result.right_pos_err))
    return ik_result, state_vec, frame_err


def _compute_single_frame_init_states(
    data,
    ik_solver,
    frame_targets,
    locked_double_stance,
    progress_prefix=None,
    progress_every=100,
):
    """Run the existing single-frame IK once to seed trajectory IK."""
    N = len(frame_targets)
    init_states = np.zeros((N, 15), dtype=np.float64)
    init_frame_errors = np.zeros(N, dtype=np.float64)
    prev_leg_qpos = None
    prev_root_pos = None

    for t, target in enumerate(frame_targets):
        ik_result, init_states[t], init_frame_errors[t] = _solve_single_frame_ik(
            data,
            ik_solver,
            target,
            prev_leg_qpos=prev_leg_qpos,
            prev_root_pos=prev_root_pos,
            locked_double_stance=bool(locked_double_stance[t]),
            default_initial_leg_qpos=target.ref_leg_qpos,
        )
        prev_leg_qpos = ik_result.leg_qpos.copy()
        prev_root_pos = ik_result.qpos[:3].copy()
        if progress_prefix and (t % progress_every == 0 or t == N - 1):
            print(f"{progress_prefix} frame {t}/{N}", flush=True)

    return init_states, init_frame_errors


def _maybe_stabilize_live_frame(
    data,
    ik_solver,
    target,
    base_state,
    motion,
    t,
    base_pos_err,
    stab_state,
    active_until,
    args,
):
    trigger_err = float(args.ik_stabilizer_trigger_mm) / 1000.0
    cooldown = max(int(args.ik_stabilizer_cooldown_frames), 0)
    if base_pos_err > trigger_err:
        active_until = max(active_until, t + cooldown)

    left_contact = motion.left_contact[t]
    right_contact = motion.right_contact[t]
    need_lb_stab = bool(left_contact or right_contact) and (
        base_pos_err > trigger_err or t <= active_until
    )
    if need_lb_stab:
        stabilized_state = _run_lower_body_stabilizer_frame(
            data,
            ik_solver,
            target,
            base_state,
            left_contact,
            right_contact,
            stab_state,
            args,
        )
        _apply_state_to_model(data, ik_solver, target, stabilized_state)
        final_pos_err = max(
            np.linalg.norm(data.xpos[ik_solver.left_foot_body_id] - target.left_pos),
            np.linalg.norm(data.xpos[ik_solver.right_foot_body_id] - target.right_pos),
        )
        return stabilized_state, final_pos_err, active_until

    _update_lower_body_stabilizer_memory(
        data,
        ik_solver,
        target,
        base_state,
        left_contact,
        right_contact,
        stab_state,
    )
    return base_state, base_pos_err, active_until


def _apply_state_to_model(data, ik_solver, target, state):
    """Write one trajectory-IK state back into MuJoCo qpos."""
    root_pos, leg_q = ik_solver.unpack_state(state)
    data.qpos[:3] = root_pos
    data.qpos[3:7] = target.root_quat
    data.qpos[7:] = target.fixed_upper_body_qpos
    data.qpos[ik_solver.leg_qpos_adr] = leg_q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(ik_solver.model, data)
    return root_pos, leg_q


def _solve_ik_state_sequence(
    args,
    model,
    data,
    ik_solver,
    frame_targets,
    locked_double_stance,
    left_contact,
    right_contact,
    ik_backend,
    init_progress_prefix=None,
    init_progress_every=200,
    lb_progress_prefix=None,
    lb_progress_every=200,
):
    base_states, init_frame_errors = _compute_single_frame_init_states(
        data,
        ik_solver,
        frame_targets,
        locked_double_stance,
        progress_prefix=init_progress_prefix,
        progress_every=init_progress_every,
    )

    solved_states = base_states.copy()
    if getattr(args, "ik_mode", "frame") == "window":
        if ik_backend != "jacobian":
            raise ValueError("window IK currently supports only --ik_backend jacobian")
        from stair_mppi.trajectory_ik import G1TrajectoryIK

        print(f"  Window IK enabled: window={args.ik_window} commit={args.ik_commit}")
        traj_ik = G1TrajectoryIK(model, ik_solver, _trajectory_ik_config_from_args(args))
        solved_states = traj_ik.solve_sequence(
            data,
            frame_targets,
            left_contact,
            right_contact,
            base_states,
            init_frame_errors=init_frame_errors,
        )

    if _lower_body_stabilizer_enabled(args, ik_backend):
        print("  Lower-body stabilizer enabled")
        solved_states = _run_lower_body_stabilizer_sequence(
            data,
            ik_solver,
            frame_targets,
            left_contact,
            right_contact,
            solved_states,
            args,
            base_frame_errors=init_frame_errors,
            progress_prefix=lb_progress_prefix,
            progress_every=lb_progress_every,
        )

    return solved_states, init_frame_errors


def _record_solved_sequence(
    data,
    ghost_model,
    ghost_data,
    ik_solver,
    frame_targets,
    solved_states,
    body_mapping,
    joint_mapping,
    pelvis_positions,
    opt_joint_pos,
    opt_body_pos,
    opt_body_quat,
    ghost_joint_pos,
    ghost_body_pos,
    ghost_body_quat,
    motion,
):
    ik_pos_errors = []
    for t, target in enumerate(frame_targets):
        _apply_state_to_model(data, ik_solver, target, solved_states[t])

        lf_bid = body_mapping[NPZ_LFOOT]
        rf_bid = body_mapping[NPZ_RFOOT]
        perr = np.linalg.norm(data.qpos[:3] - target.root_pos) * 1000.0
        lferr = np.linalg.norm(data.xpos[lf_bid] - target.left_pos) * 1000.0
        rferr = np.linalg.norm(data.xpos[rf_bid] - target.right_pos) * 1000.0
        max_err = max(perr, lferr, rferr)
        ik_pos_errors.append(max_err)
        if max_err > 200.0:
            lc = "S" if motion.left_contact[t] else "W"
            rc = "S" if motion.right_contact[t] else "W"
            print(
                f"  [IK-ERR] f={t:4d} L:{lc} R:{rc}  perr={perr:.0f}mm "
                f"lferr={lferr:.0f}mm rferr={rferr:.0f}mm  "
                f"pelvis_z={target.root_pos[2]:.3f} "
                f"lf_z={target.left_pos[2]:.3f} rf_z={target.right_pos[2]:.3f}"
            )

        opt_joint_pos[t] = reorder_from_mujoco(data.qpos[7:], joint_mapping)
        for npz_i, mj_bid in enumerate(body_mapping):
            opt_body_pos[t, npz_i] = data.xpos[mj_bid]
            opt_body_quat[t, npz_i] = data.xquat[mj_bid]

        update_robot_pose(
            ghost_data,
            root_pos=pelvis_positions[t],
            root_quat=target.root_quat,
            joint_qpos=target.fixed_upper_body_qpos,
        )
        mujoco.mj_forward(ghost_model, ghost_data)
        ghost_joint_pos[t] = reorder_from_mujoco(ghost_data.qpos[7:], joint_mapping)
        for npz_i, mj_bid in enumerate(body_mapping):
            ghost_body_pos[t, npz_i] = ghost_data.xpos[mj_bid]
            ghost_body_quat[t, npz_i] = ghost_data.xquat[mj_bid]

    return np.asarray(ik_pos_errors, dtype=np.float64)


def _segment_segment_closest_points(p0, p1, q0, q1):
    """Closest points between two 3D line segments."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))
    D = a * c - b * b
    eps = 1e-8

    if a < eps and c < eps:
        return p0, q0
    if a < eps:
        t = np.clip(e / max(c, eps), 0.0, 1.0)
        return p0, q0 + t * v
    if c < eps:
        s = np.clip(-d / max(a, eps), 0.0, 1.0)
        return p0 + s * u, q0

    if D < eps:
        s = 0.0
        t = np.clip(e / c, 0.0, 1.0)
    else:
        s = np.clip((b * e - c * d) / D, 0.0, 1.0)
        t = (a * e - b * d) / D
        if t < 0.0:
            t = 0.0
            s = np.clip(-d / a, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = np.clip((b - d) / a, 0.0, 1.0)

    # Re-project once after clamping to keep both points mutually consistent.
    if D >= eps:
        s = np.clip((b * t - d) / a, 0.0, 1.0)
        t = np.clip((a * e - b * d) / D if D >= eps else t, 0.0, 1.0)
        if t < 0.0:
            t = 0.0
            s = np.clip(-d / a, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = np.clip((b - d) / a, 0.0, 1.0)

    return p0 + s * u, q0 + t * v


def _capsule_clearance(seg0_a, seg0_b, radius0, seg1_a, seg1_b, radius1):
    """Signed capsule clearance: positive = separated, negative = overlapping."""
    cp0, cp1 = _segment_segment_closest_points(seg0_a, seg0_b, seg1_a, seg1_b)
    dist_vec = cp0 - cp1
    dist = float(np.linalg.norm(dist_vec))
    return dist - (float(radius0) + float(radius1)), cp0, cp1


def _solve_knee_position(hip_pos, ankle_pos, ref_knee_pos, thigh_len, shin_len):
    """Closed-form 2-link knee reconstruction using a reference bend direction."""
    hip_pos = np.asarray(hip_pos, dtype=np.float64)
    ankle_pos = np.asarray(ankle_pos, dtype=np.float64)
    ref_knee_pos = np.asarray(ref_knee_pos, dtype=np.float64)

    ha = ankle_pos - hip_pos
    d = float(np.linalg.norm(ha))
    if d < 1e-8:
        return ref_knee_pos.copy()

    d_dir = ha / d
    pole = ref_knee_pos - hip_pos
    pole -= d_dir * np.dot(pole, d_dir)
    if np.linalg.norm(pole) < 1e-8:
        pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        pole -= d_dir * np.dot(pole, d_dir)
    if np.linalg.norm(pole) < 1e-8:
        pole = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        pole -= d_dir * np.dot(pole, d_dir)
    bend_dir = _safe_normalize(pole, [0.0, 1.0, 0.0])

    d_clamped = float(np.clip(d, abs(thigh_len - shin_len) + 1e-6, thigh_len + shin_len - 1e-6))
    x = (thigh_len ** 2 - shin_len ** 2 + d_clamped ** 2) / (2.0 * d_clamped)
    y_sq = max(thigh_len ** 2 - x ** 2, 0.0)
    y = np.sqrt(y_sq)

    knee_base = hip_pos + d_dir * x
    side = 1.0 if np.dot(ref_knee_pos - knee_base, bend_dir) >= 0.0 else -1.0
    return knee_base + side * y * bend_dir


def _foot_capsule_endpoints(ankle_pos, foot_quat):
    R = _quat_to_rotmat(foot_quat)
    heel = ankle_pos + R @ HEEL_OFFSET
    toe = ankle_pos + R @ TOE_OFFSET
    return heel, toe


def _estimate_leg_capsules(
    motion,
    pelvis_positions,
    left_foot_pos,
    right_foot_pos,
    left_foot_quats,
    right_foot_quats,
):
    """Estimate hip/knee/ankle and foot capsules from planned pelvis + feet."""
    pelvis_delta = pelvis_positions - motion.body_pos[:, NPZ_PELVIS]

    left_hip = motion.body_pos[:, NPZ_LHIP] + pelvis_delta
    right_hip = motion.body_pos[:, NPZ_RHIP] + pelvis_delta
    left_ref_knee = motion.body_pos[:, NPZ_LKNEE] + pelvis_delta
    right_ref_knee = motion.body_pos[:, NPZ_RKNEE] + pelvis_delta

    left_thigh_len = float(np.median(np.linalg.norm(
        motion.body_pos[:, NPZ_LKNEE] - motion.body_pos[:, NPZ_LHIP], axis=1
    )))
    left_shin_len = float(np.median(np.linalg.norm(
        motion.body_pos[:, NPZ_LFOOT] - motion.body_pos[:, NPZ_LKNEE], axis=1
    )))
    right_thigh_len = float(np.median(np.linalg.norm(
        motion.body_pos[:, NPZ_RKNEE] - motion.body_pos[:, NPZ_RHIP], axis=1
    )))
    right_shin_len = float(np.median(np.linalg.norm(
        motion.body_pos[:, NPZ_RFOOT] - motion.body_pos[:, NPZ_RKNEE], axis=1
    )))

    n = left_foot_pos.shape[0]
    left_knee = np.zeros((n, 3), dtype=np.float64)
    right_knee = np.zeros((n, 3), dtype=np.float64)
    left_heel = np.zeros((n, 3), dtype=np.float64)
    left_toe = np.zeros((n, 3), dtype=np.float64)
    right_heel = np.zeros((n, 3), dtype=np.float64)
    right_toe = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        left_knee[i] = _solve_knee_position(
            left_hip[i], left_foot_pos[i], left_ref_knee[i], left_thigh_len, left_shin_len,
        )
        right_knee[i] = _solve_knee_position(
            right_hip[i], right_foot_pos[i], right_ref_knee[i], right_thigh_len, right_shin_len,
        )
        left_heel[i], left_toe[i] = _foot_capsule_endpoints(left_foot_pos[i], left_foot_quats[i])
        right_heel[i], right_toe[i] = _foot_capsule_endpoints(right_foot_pos[i], right_foot_quats[i])

    return {
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_heel": left_heel,
        "left_toe": left_toe,
        "right_heel": right_heel,
        "right_toe": right_toe,
    }


def _latch_stance_foot_targets(
    foot_pos,
    contact_mask,
    terrain=None,
    foot_nominal_z=0.0,
    ground_tol=0.05,
):
    """Fix each stance run to the first *grounded* stance-frame xyz.

    Phase-derived contact can start a few frames before the foot is actually
    close to the terrain.  Latching to that transition frame can create an
    unreachable IK target.  We therefore anchor only from the first stance frame
    whose target z is close to the terrain contact height; if a run has no such
    frame, it is left unchanged.  Swing frames are untouched.
    """
    out = np.asarray(foot_pos, dtype=np.float64).copy()
    contact = np.asarray(contact_mask, dtype=bool)
    n = contact.shape[0]
    i = 0
    while i < n:
        if not contact[i]:
            i += 1
            continue
        st = i
        while i < n and contact[i]:
            i += 1
        en = i

        anchor_idx = st
        if terrain is not None:
            run = out[st:en]
            floor_z = terrain.height_batch(run[:, 0], run[:, 1]) + float(foot_nominal_z)
            grounded = np.abs(run[:, 2] - floor_z) <= float(ground_tol)
            if not np.any(grounded):
                continue
            anchor_idx = st + int(np.flatnonzero(grounded)[0])

        anchor = out[anchor_idx].copy()
        if terrain is not None:
            anchor[2] = float(terrain.height_at(float(anchor[0]), float(anchor[1]))) + float(foot_nominal_z)
        out[anchor_idx:en] = anchor[None, :]
    return out


def _collision_frame_count(
    motion,
    pelvis_positions,
    left_foot_pos,
    right_foot_pos,
    left_foot_quats,
    right_foot_quats,
    shin_radius,
    foot_radius,
    collision_margin,
):
    """Count frames with any lower-leg/foot capsule interference."""
    capsules = _estimate_leg_capsules(
        motion, pelvis_positions, left_foot_pos, right_foot_pos, left_foot_quats, right_foot_quats,
    )
    n = left_foot_pos.shape[0]
    count = 0
    for i in range(n):
        pairs = (
            (capsules["left_knee"][i], left_foot_pos[i], shin_radius,
             capsules["right_knee"][i], right_foot_pos[i], shin_radius),
            (capsules["left_knee"][i], left_foot_pos[i], shin_radius,
             capsules["right_heel"][i], capsules["right_toe"][i], foot_radius),
            (capsules["left_heel"][i], capsules["left_toe"][i], foot_radius,
             capsules["right_knee"][i], right_foot_pos[i], shin_radius),
            (capsules["left_heel"][i], capsules["left_toe"][i], foot_radius,
             capsules["right_heel"][i], capsules["right_toe"][i], foot_radius),
        )
        frame_collide = False
        for seg0_a, seg0_b, r0, seg1_a, seg1_b, r1 in pairs:
            clearance, _, _ = _capsule_clearance(seg0_a, seg0_b, r0, seg1_a, seg1_b, r1)
            if clearance < collision_margin:
                frame_collide = True
                break
        count += int(frame_collide)
    return count


def _resolve_lower_leg_foot_collisions(
    motion,
    pelvis_positions,
    left_foot_pos,
    right_foot_pos,
    left_foot_quats,
    right_foot_quats,
    left_contact,
    right_contact,
    terrain,
    foot_nominal_z,
    n_iters=2,
    shin_radius=0.045,
    foot_radius=0.03,
    collision_margin=0.015,
    push_gain=0.9,
    max_push_per_frame=0.04,
    pelvis_side_margin=0.025,
    pelvis_side_gain=0.5,
):
    """Collision-aware post-pass using shin/foot capsules and swing-foot pushes."""
    if n_iters <= 0:
        return left_foot_pos, right_foot_pos

    left_foot_pos = np.asarray(left_foot_pos, dtype=np.float64).copy()
    right_foot_pos = np.asarray(right_foot_pos, dtype=np.float64).copy()
    left_contact = np.asarray(left_contact, dtype=bool)
    right_contact = np.asarray(right_contact, dtype=bool)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)

    def smooth_delta(delta):
        out = delta.copy()
        for axis in range(2):
            for _ in range(2):
                out[:, axis] = np.convolve(out[:, axis], kernel, mode="same")
        return out

    for _ in range(n_iters):
        capsules = _estimate_leg_capsules(
            motion, pelvis_positions, left_foot_pos, right_foot_pos, left_foot_quats, right_foot_quats,
        )
        left_delta = np.zeros((left_foot_pos.shape[0], 2), dtype=np.float64)
        right_delta = np.zeros((right_foot_pos.shape[0], 2), dtype=np.float64)

        for i in range(left_foot_pos.shape[0]):
            left_movable = not left_contact[i]
            right_movable = not right_contact[i]
            if not left_movable and not right_movable:
                continue

            lateral = capsules["left_hip"][i] - capsules["right_hip"][i]
            lateral[2] = 0.0
            left_out = _safe_normalize(lateral, [0.0, 1.0, 0.0])[:2]
            right_out = -left_out

            pairs = (
                (capsules["left_knee"][i], left_foot_pos[i], shin_radius,
                 capsules["right_knee"][i], right_foot_pos[i], shin_radius),
                (capsules["left_knee"][i], left_foot_pos[i], shin_radius,
                 capsules["right_heel"][i], capsules["right_toe"][i], foot_radius),
                (capsules["left_heel"][i], capsules["left_toe"][i], foot_radius,
                 capsules["right_knee"][i], right_foot_pos[i], shin_radius),
                (capsules["left_heel"][i], capsules["left_toe"][i], foot_radius,
                 capsules["right_heel"][i], capsules["right_toe"][i], foot_radius),
            )

            total_push = 0.0
            for seg0_a, seg0_b, r0, seg1_a, seg1_b, r1 in pairs:
                clearance, _, _ = _capsule_clearance(seg0_a, seg0_b, r0, seg1_a, seg1_b, r1)
                if clearance < collision_margin:
                    total_push += (collision_margin - clearance)

            if total_push > 0.0:
                push = min(push_gain * total_push, max_push_per_frame)
                if left_movable and right_movable:
                    left_delta[i] += 0.5 * push * left_out
                    right_delta[i] += 0.5 * push * right_out
                elif left_movable:
                    left_delta[i] += push * left_out
                elif right_movable:
                    right_delta[i] += push * right_out

            yaw = extract_yaw_from_quat(motion.body_quat[i, NPZ_PELVIS])
            left_axis = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
            pelvis_xy = pelvis_positions[i, :2]
            side_push_gain = pelvis_side_gain * push_gain

            if left_movable:
                left_lat = float(np.dot(left_foot_pos[i, :2] - pelvis_xy, left_axis))
                if left_lat < pelvis_side_margin:
                    left_delta[i] += min(
                        side_push_gain * (pelvis_side_margin - left_lat),
                        max_push_per_frame,
                    ) * left_axis

            if right_movable:
                right_lat = float(np.dot(right_foot_pos[i, :2] - pelvis_xy, left_axis))
                if right_lat > -pelvis_side_margin:
                    right_delta[i] -= min(
                        side_push_gain * (pelvis_side_margin + right_lat),
                        max_push_per_frame,
                    ) * left_axis

        left_delta = smooth_delta(left_delta)
        right_delta = smooth_delta(right_delta)

        left_mag = np.linalg.norm(left_delta, axis=1)
        right_mag = np.linalg.norm(right_delta, axis=1)
        left_scale = np.minimum(1.0, max_push_per_frame / np.maximum(left_mag, 1e-8))
        right_scale = np.minimum(1.0, max_push_per_frame / np.maximum(right_mag, 1e-8))
        left_delta *= left_scale[:, None]
        right_delta *= right_scale[:, None]

        # Apply deltas to swing frames only (stance frames remain locked)
        left_swing = ~left_contact
        right_swing = ~right_contact
        left_foot_pos[left_swing, :2] += left_delta[left_swing]
        right_foot_pos[right_swing, :2] += right_delta[right_swing]

        # Floor clamp (only for swing frames)
        left_floor = terrain.height_batch(left_foot_pos[:, 0], left_foot_pos[:, 1]) + foot_nominal_z
        right_floor = terrain.height_batch(right_foot_pos[:, 0], right_foot_pos[:, 1]) + foot_nominal_z
        left_foot_pos[left_swing, 2] = np.maximum(left_foot_pos[left_swing, 2], left_floor[left_swing])
        right_foot_pos[right_swing, 2] = np.maximum(right_foot_pos[right_swing, 2], right_floor[right_swing])

    return left_foot_pos, right_foot_pos


def precompute_mppi_foot(
    precomputed_ref,     # (N, 3) TerrainReference 输出的脚踝参考位置
    precomputed_quats,   # (N, 4) TerrainReference 输出的脚踝参考四元数 [w,x,y,z]
    contact_mask,        # (N,) 布尔触地掩码 (True=支撑, False=摆动)
    foot_nominal_z,      # 标量：脚踝在平地上的标称 z 高度
    terrain,             # 地形对象 (需要 height_batch 方法)
    mppi_params,         # MppiFootParams 或 PlumLandingParams：优化器参数
    fps,                 # 动作帧率 (Hz)
    pelvis_quats,        # (N, 4) 兼容保留参数；当前实现不再使用
    planner="mppi",      # "mppi" / "terrain_ref" / "plum"
):
    """对单只脚的所有摆动阶段执行 MPPI 轨迹优化。

    核心思想：在中足虚拟中心 (脚尖与脚跟的中点) 而非脚踝原点上规划。
    这样规划点更靠近地面接触面，减少台阶边缘处脚尖/脚跟的穿透。
    优化完成后将中足结果反变换回脚踝坐标供 IK 使用。

    参数:
        precomputed_ref: (N, 3) 来自 TerrainReference 的脚踝参考位置。
        precomputed_quats: (N, 4) 来自 TerrainReference 的脚踝四元数 [w,x,y,z]。
        contact_mask: (N,) 布尔触地掩码。
        foot_nominal_z: 脚踝标称高度。
        terrain: 地形对象 (需要 height_batch 方法)。
        mppi_params: MppiFootParams (mppi/terrain_ref) 或 PlumLandingParams (plum)。
        fps: 动作帧率。
        pelvis_quats: 兼容保留参数；当前实现不再使用。
        planner: "mppi" / "terrain_ref" / "plum"。

    返回:
        optimized_pos: (N, 3) 脚踝位置 (支撑帧保持不变，摆动帧被优化器覆盖)。
        optimized_quats: (N, 4) 纯 yaw 四元数 (脚面保持水平)。
    """
    # ===== Plum blossom mode: XY-only landing optimizer =====
    if planner == "plum":
        from stair_mppi.mppi_foot_plum import PlumLandingOptimizer, PlumLandingParams
        plum_params = mppi_params if isinstance(mppi_params, PlumLandingParams) else PlumLandingParams()
        plum_opt = PlumLandingOptimizer(plum_params, terrain)

        n = len(contact_mask)
        out_pos = precomputed_ref.copy()
        out_quats = precomputed_quats.copy()

        # Lock stance quaternions to pure yaw
        stance_phases = _find_stance_phases(contact_mask)
        for (st, en) in stance_phases:
            yaw = extract_yaw_from_quat(precomputed_quats[st])
            locked_quat = _yaw_only_quat(yaw)
            for i in range(st, en):
                out_quats[i] = locked_quat

        phases = _find_swing_phases(contact_mask)
        print(f"    Plum landing optimizer: {len(phases)} swing phases")

        for phase_idx, (swing_start, swing_end) in enumerate(phases):
            liftoff_idx = swing_start - 1
            landing_idx = swing_end
            if liftoff_idx < 0 or landing_idx >= n:
                continue

            n_swing = swing_end - swing_start

            # Find best landing xy on a pole/platform top
            raw_landing_xy = out_pos[landing_idx, :2].copy()
            best_xy = plum_opt.find_landing(raw_landing_xy)

            # Update landing position (xy only, z unchanged)
            out_pos[landing_idx, :2] = best_xy

            # Lock subsequent stance to this landing xy
            if landing_idx < n - 1:
                next_stance_end = swing_end
                while next_stance_end < n and contact_mask[next_stance_end]:
                    next_stance_end += 1
                for k in range(landing_idx, next_stance_end):
                    out_pos[k, :2] = best_xy

            # Interpolate swing xy with smoothstep
            liftoff_xy = out_pos[liftoff_idx, :2]
            t = np.linspace(0, 1, n_swing)
            blend = 3 * t**2 - 2 * t**3  # C1 smoothstep
            for k in range(n_swing):
                out_pos[swing_start + k, 0] = (1 - blend[k]) * liftoff_xy[0] + blend[k] * best_xy[0]
                out_pos[swing_start + k, 1] = (1 - blend[k]) * liftoff_xy[1] + blend[k] * best_xy[1]
                # z: keep original (already at pole height from z-offset)

            if phase_idx < 3 or (phase_idx + 1) % 5 == 0:
                dist = np.linalg.norm(best_xy - raw_landing_xy) * 100
                print(f"      phase {phase_idx+1}: landing ({raw_landing_xy[0]:.3f},{raw_landing_xy[1]:.3f})"
                      f" → ({best_xy[0]:.3f},{best_xy[1]:.3f}) shift={dist:.1f}cm")

        return out_pos, out_quats

    if planner == "terrain_ref":
        out_pos = precomputed_ref.copy()
        out_quats = precomputed_quats.copy()
        stance_phases = _find_stance_phases(contact_mask)
        for (st, en) in stance_phases:
            yaw = extract_yaw_from_quat(precomputed_quats[st])
            locked_quat = _yaw_only_quat(yaw)
            for i in range(st, en):
                out_quats[i] = locked_quat
        return out_pos, out_quats

    # 仅支持 MPPI swing optimizer
    opt = MppiFootOptimizer(mppi_params)
    n = len(contact_mask)
    out_pos = precomputed_ref.copy()    # 复制脚踝位置数组（摆动帧将被覆盖）
    out_quats = precomputed_quats.copy()  # 复制四元数数组

    # ===== 第一步：锁定支撑阶段的四元数为纯 yaw =====
    # 对离散台阶地形，支撑脚默认保持水平，仅保留 yaw。
    # 这样能避免在台阶边缘用有限差分法向量估计时出现的假 pitch/roll。
    stance_phases = _find_stance_phases(contact_mask)
    for (st, en) in stance_phases:
        yaw = extract_yaw_from_quat(precomputed_quats[st])
        locked_quat = _yaw_only_quat(yaw)
        for i in range(st, en):
            out_quats[i] = locked_quat

    # ===== 第二步：对每个摆动阶段执行 MPPI 优化 =====
    phases = _find_swing_phases(contact_mask)
    _n_skipped = 0

    if len(phases) > 0:
        print(f"    MPPI optimizing {len(phases)} swing phases...", flush=True)

    for phase_idx, (swing_start, swing_end) in enumerate(phases):
        if phase_idx == 0 or (phase_idx + 1) % 5 == 0 or phase_idx == len(phases) - 1:
            print(f"      phase {phase_idx + 1}/{len(phases)} frames=[{swing_start},{swing_end})", flush=True)
        liftoff_idx = swing_start - 1  # 抬脚帧 = 摆动开始前一帧
        landing_idx = swing_end        # 落脚帧 = 摆动结束后第一帧

        # 边界检查：如果抬脚/落脚帧越界，跳过优化
        # 摆动帧保留原始动捕四元数 (precomputed_quats)，不覆盖
        if liftoff_idx < 0 or landing_idx >= n:
            continue

        # ===== Conditional skip: preserve original motion when possible =====
        # Check height difference between consecutive stances
        liftoff_z = float(out_pos[liftoff_idx, 2])
        landing_z = float(out_pos[landing_idx, 2])
        height_diff = abs(landing_z - liftoff_z)

        if height_diff < 0.015:  # Same level (< 1.5cm height change)
            # Check: does the original swing trajectory collide with terrain?
            _has_collision = False
            _clearance_margin = mppi_params.h_clearance if hasattr(mppi_params, 'h_clearance') else 0.02
            for _k in range(swing_start, swing_end):
                _fx, _fy, _fz = precomputed_ref[_k]
                _tz = float(terrain.height_at(_fx, _fy))
                if _fz < _tz + foot_nominal_z + _clearance_margin:
                    _has_collision = True
                    break

            if not _has_collision:
                # SKIP: no height change, no collision → keep original trajectory
                _n_skipped += 1
                continue

        n_swing = swing_end - swing_start  # 摆动阶段的帧数

        # --- 摆动四元数：保留原始动捕四元数，跟踪参考动作的 pitch/roll ---
        # stance 阶段已在第一步中锁定为纯 yaw (脚面水平)，
        # swing 阶段保留 precomputed_quats 中的原始动捕值。
        # 在 swing 首尾各 blend_frames 帧内做 slerp 过渡，
        # 避免 stance(纯yaw) → swing(动捕) 边界处四元数跳变。
        blend_frames = min(10, n_swing // 2)  # 过渡帧数，最多占半个 swing
        if blend_frames > 0:
            q_liftoff = out_quats[liftoff_idx]  # 纯 yaw (stance 锁定的)
            q_landing = out_quats[landing_idx]  # 纯 yaw (stance 锁定的)
            # swing 开头: 从 liftoff 纯 yaw 过渡到动捕
            for b in range(blend_frames):
                alpha = (b + 1) / (blend_frames + 1)  # 0→1 不含端点
                idx_b = swing_start + b
                q_mocap = precomputed_quats[idx_b]
                q_blend = slerp_quaternion(q_liftoff, q_mocap, np.array([alpha]))[0]
                out_quats[idx_b] = q_blend
            # swing 结尾: 从动捕过渡到 landing 纯 yaw
            for b in range(blend_frames):
                alpha = (b + 1) / (blend_frames + 1)
                idx_b = swing_end - blend_frames + b
                q_mocap = precomputed_quats[idx_b]
                q_blend = slerp_quaternion(q_mocap, q_landing, np.array([alpha]))[0]
                out_quats[idx_b] = q_blend

        # --- 将脚踝坐标转换为中足坐标，用于 MPPI 规划 ---
        liftoff_midfoot = _ankle_to_midfoot(out_pos[liftoff_idx], out_quats[liftoff_idx])
        landing_midfoot = _ankle_to_midfoot(out_pos[landing_idx], out_quats[landing_idx])

        # 特殊情况：摆动帧数 < 2 时，MPPI 无法优化（至少需要 2 帧才能形成有效轨迹）
        # 直接用线性插值替代
        if n_swing < 2:
            t_frac = np.linspace(0, 1, n_swing + 2)[1:-1]  # 去掉首尾端点
            for k, frac in enumerate(t_frac):
                midfoot_interp = (1 - frac) * liftoff_midfoot + frac * landing_midfoot
                out_pos[swing_start + k] = _midfoot_to_ankle(
                    midfoot_interp, out_quats[swing_start + k]
                )
            continue

        swing_duration = n_swing / fps  # 摆动持续时间 (秒)
        ref_indices = np.arange(swing_start, swing_end)  # 摆动帧的索引数组

        # --- 构建 MPPI 跟踪目标：将参考脚踝位置转换为中足坐标 ---
        ref_traj_midfoot = np.zeros((n_swing, 3), dtype=np.float64)
        for k, idx in enumerate(ref_indices):
            ref_traj_midfoot[k] = _ankle_to_midfoot(precomputed_ref[idx], out_quats[idx])
        ref_times = np.linspace(0, swing_duration, n_swing)  # 时间采样点

        # --- 调用 MPPI 优化器 ---
        # optimize_swing 内部分别优化 x/y (纯跟踪) 和 z (跟踪 + 地形穿透惩罚)
        try:
            traj_x, traj_y, traj_z = opt.optimize_swing(
                liftoff_midfoot, landing_midfoot, swing_duration,
                ref_traj_midfoot, ref_times, terrain,
            )
        except Exception as e:
            print(f"  [MPPI] Warning: swing [{swing_start}:{swing_end}] failed: {e}")
            continue  # 优化失败时保留原始 TerrainReference 轨迹

        # --- 评估优化后的中足轨迹，转换回脚踝坐标 ---
        t_eval = np.linspace(0, swing_duration, n_swing)
        midfoot_x = traj_x.eval(t_eval)  # 在采样时间点上评估 x 轨迹
        midfoot_y = traj_y.eval(t_eval)  # 评估 y 轨迹
        midfoot_z = traj_z.eval(t_eval)  # 评估 z 轨迹

        for k, idx in enumerate(ref_indices):
            midfoot_pos = np.array([midfoot_x[k], midfoot_y[k], midfoot_z[k]])
            out_pos[idx] = _midfoot_to_ankle(midfoot_pos, out_quats[idx])  # 中足→脚踝

        # --- Blend back toward original ref if landing at same height ---
        # When MPPI was needed (collision on flat ground), blend the end of
        # the swing back toward the precomputed_ref to restore original motion shape.
        if height_diff < 0.015 and n_swing > 6:
            _blend_start = swing_start + int(0.7 * n_swing)
            for _k in range(_blend_start, swing_end):
                _alpha = (_k - _blend_start) / max(swing_end - _blend_start, 1)
                out_pos[_k] = (1.0 - _alpha) * out_pos[_k] + _alpha * precomputed_ref[_k]

    if _n_skipped > 0:
        print(f"    Skipped {_n_skipped}/{len(phases)} swing phases (no height change, no collision)")

    return out_pos, out_quats


# ---------------------------------------------------------------------------
# Trio NPZ export helper
# ---------------------------------------------------------------------------

def apply_zero_phase_lowpass(traj, fps, cutoff_hz=10.0, order=4):
    """Zero-phase Butterworth lowpass on a (N,D) or (N,) array.

    Uses scipy.signal.filtfilt for zero phase shift.
    Gracefully handles short sequences: if the signal is too short for
    the requested filter order, the order is reduced until filtfilt can
    run, or the original is returned unfiltered.
    """
    from scipy.signal import butter, filtfilt, sosfiltfilt, butter as _butter

    if cutoff_hz <= 0:
        return traj.copy()
    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        return traj.copy()

    traj = np.asarray(traj, dtype=np.float64)
    N = traj.shape[0]

    # filtfilt requires padlen <= N - 1; default padlen = 3 * max(len(a), len(b))
    # For butter order k: len(b) = len(a) = k + 1, so padlen = 3 * (k + 1).
    # If N is too short, reduce order until it fits, or skip.
    eff_order = int(order)
    while eff_order >= 1:
        padlen = 3 * (eff_order + 1)
        if padlen < N:
            break
        eff_order -= 1
    if eff_order < 1:
        return traj.copy()  # too short to filter at all

    b, a = butter(eff_order, cutoff_hz / nyquist, btype="low")

    def _filt1d(x):
        return filtfilt(b, a, x)

    if traj.ndim == 1:
        return _filt1d(traj)
    out = np.empty_like(traj)
    if traj.ndim == 2:
        for j in range(traj.shape[1]):
            out[:, j] = _filt1d(traj[:, j])
    elif traj.ndim == 3:
        for b_idx in range(traj.shape[1]):
            for d in range(traj.shape[2]):
                out[:, b_idx, d] = _filt1d(traj[:, b_idx, d])
    else:
        out = traj.copy()
    return out


def _smooth_quaternions_logmap(quats, fps, cutoff_hz, order):
    """Smooth quaternion sequences via log-map + zero-phase lowpass + exp-map.

    For each body:
      1. Ensure quaternion sign continuity (flip to shortest arc).
      2. Compute incremental rotations relative to the first frame.
      3. Map to axis-angle (log map) — 3D signal per frame.
      4. Apply zero-phase lowpass on the 3D log signal.
      5. Map back to quaternions (exp map).

    This avoids component-wise filtering artefacts (non-unit quaternions,
    wrong interpolation paths).
    """
    from scipy.spatial.transform import Rotation

    quats = np.asarray(quats, dtype=np.float64).copy()  # (N, n_bodies, 4) wxyz
    n_bodies = quats.shape[1]

    for bi in range(n_bodies):
        q = quats[:, bi]  # (N, 4) wxyz

        # Sign continuity: flip if dot with previous < 0
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] *= -1

        # Convert wxyz → xyzw for scipy
        q_xyzw = q[:, [1, 2, 3, 0]]
        R0_inv = Rotation.from_quat(q_xyzw[0]).inv()
        # Incremental rotations relative to frame 0
        R_all = Rotation.from_quat(q_xyzw)
        R_inc = R0_inv * R_all  # relative rotation
        rotvec = R_inc.as_rotvec()  # (N, 3) axis-angle

        # Lowpass the 3D rotation vector
        rotvec_filt = apply_zero_phase_lowpass(rotvec, fps, cutoff_hz, order)

        # Map back: R0 * exp(filtered rotvec)
        R_filt = Rotation.from_rotvec(rotvec_filt)
        R_out = Rotation.from_quat(q_xyzw[0]) * R_filt
        q_out_xyzw = R_out.as_quat()  # (N, 4) xyzw
        quats[:, bi] = q_out_xyzw[:, [3, 0, 1, 2]]  # back to wxyz

    return quats


def filter_motion_for_export(
    joint_pos, body_pos, body_quat, fps,
    cutoff_hz=10.0, order=4,
):
    """Apply zero-phase lowpass to all exported channels consistently.

    - joint_pos, body_pos: Butterworth filtfilt.
    - body_quat: log-map → filtfilt → exp-map (proper SO(3) smoothing).
    - All velocities (joint_vel, body_lin_vel, body_ang_vel) are recomputed
      from the filtered positions/quaternions via finite differences.

    This ensures pos/vel/quat/ang_vel are mutually consistent.
    """
    if cutoff_hz <= 0:
        jv, blv, bav = _compute_velocities(joint_pos, body_pos, body_quat, fps)
        return joint_pos.copy(), body_pos.copy(), body_quat.copy(), jv, blv, bav

    filt_joint_pos = apply_zero_phase_lowpass(joint_pos, fps, cutoff_hz, order)
    filt_body_pos = apply_zero_phase_lowpass(body_pos, fps, cutoff_hz, order)
    filt_body_quat = _smooth_quaternions_logmap(body_quat, fps, cutoff_hz, order)
    # Recompute all velocities from filtered data
    jv, blv, bav = _compute_velocities(filt_joint_pos, filt_body_pos, filt_body_quat, fps)
    return filt_joint_pos, filt_body_pos, filt_body_quat, jv, blv, bav


def filter_lower_body_motion_for_export(
    xml_path,
    joint_pos,
    body_pos,
    body_quat,
    fps,
    cutoff_hz=5.0,
    order=4,
):
    """Smooth root position + waist + leg joints, then rebuild full-body poses by FK.

    This preserves shoulder/elbow/wrist motion while making the support chain
    smoother and keeping exported body poses consistent with the exported
    joint commands.

    Root orientation is intentionally left unchanged. Smoothing pelvis
    quaternions over long clips can create catastrophic whole-body heading
    artefacts even when IK position error stays small, because the batch
    error metric does not constrain root orientation.
    """
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_quat = np.asarray(body_quat, dtype=np.float64)
    if cutoff_hz <= 0:
        jv, blv, bav = _compute_velocities(joint_pos, body_pos, body_quat, fps)
        return joint_pos.copy(), body_pos.copy(), body_quat.copy(), jv, blv, bav

    smooth_names = (
        list(LEFT_LEG_JOINT_NAMES)
        + list(RIGHT_LEG_JOINT_NAMES)
        + list(WAIST_JOINT_NAMES)
    )
    smooth_policy_idx = np.array(
        [JOINT_NAMES_POLICY_ORDER.index(name) for name in smooth_names],
        dtype=np.int64,
    )

    filt_joint_pos = joint_pos.copy()
    filt_joint_pos[:, smooth_policy_idx] = apply_zero_phase_lowpass(
        joint_pos[:, smooth_policy_idx], fps, cutoff_hz, order,
    )

    filt_root_pos = apply_zero_phase_lowpass(
        body_pos[:, NPZ_PELVIS, :], fps, cutoff_hz, order,
    )
    filt_root_quat = body_quat[:, NPZ_PELVIS].copy()

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joint_mapping = build_joint_mapping(model)
    body_mapping = build_body_mapping(model)

    filt_body_pos = np.zeros_like(body_pos, dtype=np.float64)
    filt_body_quat = np.zeros_like(body_quat, dtype=np.float64)
    for t in range(joint_pos.shape[0]):
        joint_qpos = reorder_to_mujoco(filt_joint_pos[t], joint_mapping)
        update_robot_pose(
            data,
            root_pos=filt_root_pos[t],
            root_quat=filt_root_quat[t],
            joint_qpos=joint_qpos,
        )
        mujoco.mj_forward(model, data)
        for npz_i, mj_bid in enumerate(body_mapping):
            filt_body_pos[t, npz_i] = data.xpos[mj_bid]
            filt_body_quat[t, npz_i] = data.xquat[mj_bid]

    jv, blv, bav = _compute_velocities(filt_joint_pos, filt_body_pos, filt_body_quat, fps)
    return filt_joint_pos, filt_body_pos, filt_body_quat, jv, blv, bav


def _compute_velocities(joint_pos, body_pos, body_quat, fps):
    """Compute joint/body velocities from position/quaternion sequences.

    Args:
        joint_pos: (N, n_joints) joint angles in policy order.
        body_pos: (N, n_bodies, 3) body positions.
        body_quat: (N, n_bodies, 4) body quaternions [w,x,y,z].
        fps: frame rate (Hz).

    Returns:
        joint_vel, body_lin_vel, body_ang_vel — all same shapes as inputs.
    """
    joint_vel = np.gradient(joint_pos, axis=0) * fps
    body_lin_vel = np.gradient(body_pos, axis=0) * fps
    n_bodies = body_pos.shape[1]
    body_ang_vel = np.zeros_like(body_pos)
    for bi in range(n_bodies):
        body_ang_vel[:, bi] = quat_ang_vel(body_quat[:, bi], fps)
    return joint_vel, body_lin_vel, body_ang_vel


def _fix_zip64_inplace(path):
    """Rewrite a ZIP64 NPZ as plain ZIP so cnpy can read it."""
    import struct, tempfile
    with open(path, "rb") as f:
        head = f.read(30)
    if len(head) < 30 or head[:4] != b"PK\x03\x04":
        return
    compr, uncompr = struct.unpack("<II", head[18:26])
    if compr != 0xFFFFFFFF and uncompr != 0xFFFFFFFF:
        return  # already plain ZIP
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: np.asarray(d[k]) for k in d.files}
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(path))
    os.close(tmp_fd)
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _save_one_npz(path, fps, joint_pos, body_pos, body_quat,
                  joint_vel, body_lin_vel, body_ang_vel,
                  transform_dx=None, transform_dy=None, transform_dyaw=None,
                  extra_fields=None):
    """Save a single motion NPZ in the standard BeyondMimic format."""
    kw = dict(
        fps=np.array([fps], dtype=np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
    )
    if transform_dx is not None:
        kw["transform_dx"] = np.float32(transform_dx)
        kw["transform_dy"] = np.float32(transform_dy)
        kw["transform_dyaw"] = np.float32(transform_dyaw)
    if extra_fields:
        for key, value in extra_fields.items():
            kw[key] = np.asarray(value)
    np.savez_compressed(path, **kw)
    _fix_zip64_inplace(path if isinstance(path, str) else str(path))


def _save_trio_npz(
    prefix, motion, joint_mapping, body_mapping,
    offset_joint_pos, offset_body_pos, offset_body_quat,
    mppi_joint_pos, mppi_body_pos, mppi_body_quat,
    export_lowpass_cutoff=0.0, export_lowpass_order=4,
):
    """Save raw / offset / mppi motion data as three NPZ files.

    All arrays follow the policy joint order (JOINT_NAMES_POLICY_ORDER)
    and npz body order (NPZ_BODY_NAMES / BODY_NAMES_POLICY_ORDER).

    Files produced:
        {prefix}_raw.npz   — original flat-ground motion capture
        {prefix}_offset.npz — raw joints + terrain-adjusted pelvis z (ghost robot FK)
        {prefix}_mppi.npz   — full IK-solved motion on terrain
    """
    fps = float(motion.fps)
    N = motion.n_frames

    # --- raw: directly from MotionClip (already in policy order) ---
    raw_joint_pos = motion.joint_pos[:N].copy()
    raw_body_pos = motion.body_pos[:N].copy()
    raw_body_quat = motion.body_quat[:N].copy()
    raw_jv, raw_blv, raw_bav = _compute_velocities(
        raw_joint_pos, raw_body_pos, raw_body_quat, fps
    )
    raw_path = f"{prefix}_raw.npz"
    _save_one_npz(raw_path, fps, raw_joint_pos, raw_body_pos, raw_body_quat,
                  raw_jv, raw_blv, raw_bav)
    print(f"\n[TrioExport] Saved {N} frames to {raw_path}")

    # --- offset: ghost robot (raw joints + terrain z) ---
    off_jv, off_blv, off_bav = _compute_velocities(
        offset_joint_pos, offset_body_pos, offset_body_quat, fps
    )
    off_path = f"{prefix}_offset.npz"
    _save_one_npz(off_path, fps, offset_joint_pos, offset_body_pos, offset_body_quat,
                  off_jv, off_blv, off_bav)
    print(f"[TrioExport] Saved {N} frames to {off_path}")

    # --- mppi: IK-solved on terrain (with optional export lowpass) ---
    if export_lowpass_cutoff > 0:
        mppi_joint_pos, mppi_body_pos, mppi_body_quat, mppi_jv, mppi_blv, mppi_bav = (
            filter_motion_for_export(
                mppi_joint_pos, mppi_body_pos, mppi_body_quat, fps,
                cutoff_hz=export_lowpass_cutoff, order=export_lowpass_order,
            )
        )
    else:
        mppi_jv, mppi_blv, mppi_bav = _compute_velocities(
            mppi_joint_pos, mppi_body_pos, mppi_body_quat, fps
        )
    mppi_path = f"{prefix}_mppi.npz"
    _save_one_npz(mppi_path, fps, mppi_joint_pos, mppi_body_pos, mppi_body_quat,
                  mppi_jv, mppi_blv, mppi_bav)
    print(f"[TrioExport] Saved {N} frames to {mppi_path}")


# ---------------------------------------------------------------------------
# Headless single-round pipeline (no viewer) for batch mode
# ---------------------------------------------------------------------------

def run_one_round(args, terrain, xml_path, dx, dy, dyaw):
    """Run the full pipeline for one round with given SE(2) transform.

    Returns (raw, ghost, optimized) data dicts, each containing:
        fps, joint_pos, body_pos, body_quat (all in policy/npz order).
    Ghost = raw joints + terrain-adjusted pelvis z (FK only, no IK).
    Optimized = full IK-solved motion on terrain.
    """
    # ===== Load motion =====
    motion = MotionClip(
        args.motion, args.start_frame, args.n_frames,
        contact_z_threshold=args.contact_z_threshold,
        contact_speed_threshold=args.contact_speed_threshold,
    )
    phase_clock_trusted = _should_trust_phase_clock(motion.phase_params)
    if motion.phase_params is not None:
        pp = motion.phase_params
        trust_str = "trusted" if phase_clock_trusted else "disabled"
        print(f"  Phase clock: type={pp.gait_type} fit={pp.fit_quality:.2f} [{trust_str}]")

    # ===== Apply SE(2) transform (dx, dy, dyaw) =====
    # 1) Yaw rotation around initial pelvis xy
    if abs(dyaw) > 1e-8:
        cos_y, sin_y = np.cos(dyaw), np.sin(dyaw)
        R2 = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
        q_yaw = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)], dtype=np.float64)
        pivot_xy = motion.body_pos[0, NPZ_PELVIS, :2].copy()
        xy_all = motion.body_pos[:, :, :2] - pivot_xy[None, None, :]
        motion.body_pos[:, :, :2] = np.einsum('ij,...j->...i', R2, xy_all) + pivot_xy[None, None, :]
        w0, x0, y0, z0 = q_yaw
        q = motion.body_quat
        w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        motion.body_quat = np.stack([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
        ], axis=-1).astype(motion.body_quat.dtype)
        if hasattr(motion, 'body_lin_vel'):
            n_bodies = motion.body_pos.shape[1]
            for bi in range(n_bodies):
                xy_v = motion.body_lin_vel[:, bi, :2].copy()
                motion.body_lin_vel[:, bi, :2] = (R2 @ xy_v.T).T

    # 2) XY translation
    if abs(dx) > 1e-8 or abs(dy) > 1e-8:
        motion.body_pos[:, :, 0] += dx
        motion.body_pos[:, :, 1] += dy

    # 3) Z offset (plum blossom mode: lift entire motion to pole-top height)
    if hasattr(args, 'plum_z_offset') and args.plum_z_offset > 0:
        motion.body_pos[:, :, 2] += args.plum_z_offset
        print(f"  Applied plum_z_offset: +{args.plum_z_offset:.3f}m")

    N = motion.n_frames
    fps = float(motion.fps)

    # ===== Plum blossom mode: skip TerrainReference, use raw motion + landing search =====
    if args.planner == "plum":
        from stair_mppi.mppi_foot_plum import PlumLandingOptimizer, PlumLandingParams
        pole_top_z = args.plum_z_offset if args.plum_z_offset > 0 else 0.2
        plum_params = PlumLandingParams(pole_top_z=pole_top_z)
        plum_opt = PlumLandingOptimizer(plum_params, terrain)

        # Use raw motion foot positions directly (already z-offset applied)
        lfoot_pos = motion.body_pos[:, NPZ_LFOOT, :].copy()  # (N, 3)
        rfoot_pos = motion.body_pos[:, NPZ_RFOOT, :].copy()
        lfoot_quats = motion.body_quat[:, NPZ_LFOOT, :].copy()
        rfoot_quats = motion.body_quat[:, NPZ_RFOOT, :].copy()

        # For each stance phase, find best landing on pole and lock position
        l_phases = _find_swing_phases(motion.left_contact)
        r_phases = _find_swing_phases(motion.right_contact)

        print(f"  Plum mode: L={len(l_phases)} swings, R={len(r_phases)} swings")

        # Lock stance z to pole_top + foot_nominal_z
        stance_z = pole_top_z + motion.foot_nominal_z

        def _plum_optimize_foot(foot_pos, contact_mask, phases, label):
            """Optimize landing xy for one foot, lock stance z."""
            for pi, (sw_start, sw_end) in enumerate(phases):
                landing_idx = sw_end
                if landing_idx >= N:
                    continue
                # Find best landing xy
                raw_xy = foot_pos[landing_idx, :2].copy()
                best_xy = plum_opt.find_landing(raw_xy)
                shift = np.linalg.norm(best_xy - raw_xy) * 100

                # Lock landing and subsequent stance to best_xy, z = stance_z
                next_end = sw_end
                while next_end < N and contact_mask[next_end]:
                    next_end += 1
                for k in range(sw_end, next_end):
                    foot_pos[k, :2] = best_xy
                    foot_pos[k, 2] = stance_z

                # Also lock liftoff stance (frames before this swing)
                liftoff_idx = sw_start - 1
                if liftoff_idx >= 0 and pi == 0:
                    # First swing: lock initial stance
                    init_xy = plum_opt.find_landing(foot_pos[0, :2])
                    for k in range(0, sw_start):
                        foot_pos[k, :2] = init_xy
                        foot_pos[k, 2] = stance_z

                # Swing: interpolate xy, keep original z shape but offset
                liftoff_xy = foot_pos[sw_start - 1, :2] if sw_start > 0 else best_xy
                n_swing = sw_end - sw_start
                t = np.linspace(0, 1, n_swing)
                blend = 3 * t**2 - 2 * t**3  # smoothstep
                for k in range(n_swing):
                    foot_pos[sw_start + k, :2] = (1 - blend[k]) * liftoff_xy + blend[k] * best_xy
                    # z: keep raw swing shape (already has natural arc from mocap + offset)

                if pi < 3 or (pi + 1) % 5 == 0:
                    print(f"    {label} phase {pi+1}: ({raw_xy[0]:.3f},{raw_xy[1]:.3f})"
                          f" → ({best_xy[0]:.3f},{best_xy[1]:.3f}) shift={shift:.1f}cm")

        _plum_optimize_foot(lfoot_pos, motion.left_contact, l_phases, "L")
        _plum_optimize_foot(rfoot_pos, motion.right_contact, r_phases, "R")

        # Use these as the "MPPI" output (variable names kept for downstream compatibility)
        mppi_lfoot = lfoot_pos
        mppi_rfoot = rfoot_pos
        # Quaternions: just use pure yaw from raw motion
        mppi_lfoot_quats = np.zeros((N, 4), dtype=np.float64)
        mppi_rfoot_quats = np.zeros((N, 4), dtype=np.float64)
        for i in range(N):
            yaw_l = extract_yaw_from_quat(lfoot_quats[i])
            mppi_lfoot_quats[i] = _yaw_only_quat(yaw_l)
            yaw_r = extract_yaw_from_quat(rfoot_quats[i])
            mppi_rfoot_quats[i] = _yaw_only_quat(yaw_r)

        pelvis_quats = motion.body_quat[:, NPZ_PELVIS]
        # Skip to pelvis z computation (jump past normal TerrainReference path)
        # We need ref_builder for downstream, create a minimal one
        ref_builder = TerrainReference(
            terrain=terrain, motion=motion,
            lookahead=args.lookahead,
            smoothing_alpha=args.alpha,
            footstep_margin=args.footstep_margin,
            toe_offset_x=args.toe_offset_x,
            heel_offset_x=args.heel_offset_x,
            mid_offset_x=args.mid_offset_x,
            toe_margin=args.toe_margin,
            swing_floor_margin=args.swing_floor_margin,
        )
    else:
        # ===== Standard path: Terrain reference =====
        ref_builder = TerrainReference(
            terrain=terrain, motion=motion,
            lookahead=args.lookahead,
            smoothing_alpha=args.alpha,
            footstep_margin=args.footstep_margin,
            toe_offset_x=args.toe_offset_x,
            heel_offset_x=args.heel_offset_x,
            mid_offset_x=args.mid_offset_x,
            toe_margin=args.toe_margin,
            swing_floor_margin=args.swing_floor_margin,
        )

        # ===== Swing trajectory optimization =====
        midfoot_contact_z = motion.foot_nominal_z + float(MID_FOOT_OFFSET[2])
        opt_params = MppiFootParams(
            n_knots=args.mppi_n_knots,
            n_samples=args.mppi_n_samples,
            n_iterations=args.mppi_n_iterations,
            temperature=args.mppi_temperature,
            w_track=args.w_track,
            w_terrain=args.w_terrain,
            h_clearance=args.h_clearance,
            h_contact_z=midfoot_contact_z,
        )

        pelvis_quats = motion.body_quat[:, NPZ_PELVIS]

        # ===== Enforce lateral separation: prevent L/R foot xy crossover =====
        _min_lateral_sep = 0.02
        _pelvis_xy = motion.body_pos[:, NPZ_PELVIS, :2]
        for i in range(len(pelvis_quats)):
            yaw = extract_yaw_from_quat(pelvis_quats[i])
            lat = np.array([-np.sin(yaw), np.cos(yaw)])
            lf_lat = np.dot(ref_builder.left_precomputed[i, :2] - _pelvis_xy[i], lat)
            rf_lat = np.dot(ref_builder.right_precomputed[i, :2] - _pelvis_xy[i], lat)
            if lf_lat - rf_lat < _min_lateral_sep:
                deficit = _min_lateral_sep - (lf_lat - rf_lat)
                ref_builder.left_precomputed[i, :2] += 0.5 * deficit * lat
                ref_builder.right_precomputed[i, :2] -= 0.5 * deficit * lat

        mppi_lfoot, mppi_lfoot_quats = precompute_mppi_foot(
            ref_builder.left_precomputed, ref_builder.left_precomputed_quats,
            motion.left_contact, motion.foot_nominal_z,
            terrain, opt_params, motion.fps, pelvis_quats=pelvis_quats, planner=args.planner,
        )
        mppi_rfoot, mppi_rfoot_quats = precompute_mppi_foot(
            ref_builder.right_precomputed, ref_builder.right_precomputed_quats,
            motion.right_contact, motion.foot_nominal_z,
            terrain, opt_params, motion.fps, pelvis_quats=pelvis_quats, planner=args.planner,
        )

    # Post-MPPI lateral clamp: MPPI may re-introduce crossover via terrain avoidance.
    # Apply same lateral separation as TerrainReference but on MPPI output.
    _min_lateral_sep = 0.02
    _pelvis_xy = motion.body_pos[:, NPZ_PELVIS, :2]
    _n_post_fixed = 0
    for i in range(N):
        yaw = extract_yaw_from_quat(pelvis_quats[i])
        lat = np.array([-np.sin(yaw), np.cos(yaw)])
        lf_lat = np.dot(mppi_lfoot[i, :2] - _pelvis_xy[i], lat)
        rf_lat = np.dot(mppi_rfoot[i, :2] - _pelvis_xy[i], lat)
        if lf_lat - rf_lat < _min_lateral_sep:
            deficit = _min_lateral_sep - (lf_lat - rf_lat)
            mppi_lfoot[i, :2] += 0.5 * deficit * lat
            mppi_rfoot[i, :2] -= 0.5 * deficit * lat
            _n_post_fixed += 1
    if _n_post_fixed > 0:
        print(f"  Post-MPPI lateral clamp: {_n_post_fixed}/{N} frames")

    # ===== Post-MPPI side-wall collision fix =====
    # Use horizontal raycasting to detect and push feet away from terrain side walls.
    if hasattr(terrain, 'foot_side_penetration'):
        _n_side_fixed = 0
        for i in range(N):
            yaw = extract_yaw_from_quat(pelvis_quats[i])
            fx, fy = np.cos(yaw), np.sin(yaw)
            # Left foot
            px_l, py_l = _push_side(terrain, mppi_lfoot[i], fx, fy)
            if abs(px_l) > 1e-4 or abs(py_l) > 1e-4:
                mppi_lfoot[i, 0] += px_l
                mppi_lfoot[i, 1] += py_l
                _n_side_fixed += 1
            # Right foot
            px_r, py_r = _push_side(terrain, mppi_rfoot[i], fx, fy)
            if abs(px_r) > 1e-4 or abs(py_r) > 1e-4:
                mppi_rfoot[i, 0] += px_r
                mppi_rfoot[i, 1] += py_r
                _n_side_fixed += 1
        if _n_side_fixed > 0:
            print(f"  Post-MPPI side-wall fix: {_n_side_fixed} foot-frames pushed")

    # ===== Pelvis z: construct target externally, filter only smooths =====
    #   support_z   — terrain-driven weighted foot height
    #   z_anchor    — support_z + nominal clearance
    #   z_style     — zero-mean relative pelvis variation from raw motion
    #   z_target    — z_anchor + z_style, clamped by reachability
    root_z_filter = SupportAwareRootZFilter(
        alpha_up=0.5, alpha_down=0.35, max_delta=0.035,
    )
    raw_pelvis = motion.body_pos[:, NPZ_PELVIS].copy()
    nominal_root_clearance = float(motion.pelvis_height_above_foot)
    raw_pelvis_z0 = float(raw_pelvis[0, 2])
    raw_style_z = raw_pelvis[:, 2] - raw_pelvis_z0
    max_reach = _MAX_REACH

    pelvis_z_out = np.zeros(N, dtype=np.float64)
    root_z_filter.reset()
    for i in range(N):
        lw, rw, is_flight = _support_state(motion, i, phase_clock_trusted)

        w_sum = lw + rw
        if w_sum > 1e-6:
            wn = np.array([lw, rw]) / w_sum
            support_z = float(wn[0] * mppi_lfoot[i, 2] + wn[1] * mppi_rfoot[i, 2])
        else:
            support_z = float(terrain.height_at(
                float(raw_pelvis[i, 0]), float(raw_pelvis[i, 1])
            ))

        z_anchor = support_z + nominal_root_clearance
        z_target = z_anchor + float(raw_style_z[i])

        z_upper_candidates = []
        if lw > 1e-3:
            z_upper_candidates.append(
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_lfoot[i], lw, max_reach)
            )
        if rw > 1e-3:
            z_upper_candidates.append(
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_rfoot[i], rw, max_reach)
            )
        if not z_upper_candidates:
            z_upper_candidates.extend([
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_lfoot[i], 0.0, max_reach),
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_rfoot[i], 0.0, max_reach),
            ])
        z_target = min(z_target, min(z_upper_candidates))

        pelvis_z_out[i] = root_z_filter.step(z_target, is_flight=is_flight)

    pelvis_positions = raw_pelvis.copy()
    pelvis_positions[:, 2] = pelvis_z_out

    # ===== Collision resolution =====
    if args.leg_collision_iters > 0:
        mppi_lfoot, mppi_rfoot = _resolve_lower_leg_foot_collisions(
            motion, pelvis_positions, mppi_lfoot, mppi_rfoot,
            mppi_lfoot_quats, mppi_rfoot_quats,
            motion.left_contact, motion.right_contact, terrain, motion.foot_nominal_z,
            n_iters=args.leg_collision_iters,
            shin_radius=args.shin_collision_radius,
            foot_radius=args.foot_collision_radius,
            collision_margin=args.leg_collision_margin,
            push_gain=args.collision_push_gain,
            max_push_per_frame=args.collision_max_push,
            pelvis_side_margin=args.pelvis_side_margin,
            pelvis_side_gain=args.pelvis_side_gain,
        )

    # Remove stance-foot target drift: each stance run keeps the first grounded stance xyz.
    if not args.no_stance_latch:
        mppi_lfoot = _latch_stance_foot_targets(
            mppi_lfoot, motion.left_contact, terrain, motion.foot_nominal_z,
            ground_tol=args.stance_latch_ground_tol,
        )
        mppi_rfoot = _latch_stance_foot_targets(
            mppi_rfoot, motion.right_contact, terrain, motion.foot_nominal_z,
            ground_tol=args.stance_latch_ground_tol,
        )

    left_point_scales = _precompute_contact_point_scales(
        mppi_lfoot, mppi_lfoot_quats, motion.left_contact, terrain,
    )
    right_point_scales = _precompute_contact_point_scales(
        mppi_rfoot, mppi_rfoot_quats, motion.right_contact, terrain,
    )

    # ===== MuJoCo model + IK setup =====
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joint_mapping = build_joint_mapping(model)
    body_mapping = build_body_mapping(model)

    _ik_backend, ik_solver, root_xy_weight, root_z_weight = _build_ik_solver(
        model, args, log_config=False
    )

    # Ghost FK model (for offset/ghost recording)
    ghost_model, ghost_data = _build_ghost_fk_model(xml_path)

    nbody_export = len(NPZ_BODY_NAMES)
    n_joints = model.nq - 7  # 29 for G1

    # Buffers: raw is read from motion directly
    # Ghost (offset): raw joints + terrain-adjusted pelvis z
    ghost_joint_pos = np.zeros((N, n_joints), dtype=np.float64)
    ghost_body_pos = np.zeros((N, nbody_export, 3), dtype=np.float64)
    ghost_body_quat = np.zeros((N, nbody_export, 4), dtype=np.float64)
    # Optimized (MPPI + IK)
    opt_joint_pos = np.zeros((N, n_joints), dtype=np.float64)
    opt_body_pos = np.zeros((N, nbody_export, 3), dtype=np.float64)
    opt_body_quat = np.zeros((N, nbody_export, 4), dtype=np.float64)

    frame_targets, locked_double_stance = _build_frame_ik_targets(
        motion,
        pelvis_positions,
        mppi_lfoot,
        mppi_rfoot,
        mppi_lfoot_quats,
        mppi_rfoot_quats,
        joint_mapping,
        ik_solver,
        left_point_scales,
        right_point_scales,
        root_xy_weight,
        root_z_weight,
        phase_clock_trusted,
    )

    solved_states, _ = _solve_ik_state_sequence(
        args,
        model,
        data,
        ik_solver,
        frame_targets,
        locked_double_stance,
        motion.left_contact[:N],
        motion.right_contact[:N],
        _ik_backend,
        init_progress_prefix="    [window-init]" if getattr(args, "ik_mode", "frame") == "window" else None,
        init_progress_every=200,
        lb_progress_prefix="    [lb-stab]",
        lb_progress_every=200,
    )
    ik_pos_errors = _record_solved_sequence(
        data,
        ghost_model,
        ghost_data,
        ik_solver,
        frame_targets,
        solved_states,
        body_mapping,
        joint_mapping,
        pelvis_positions,
        opt_joint_pos,
        opt_body_pos,
        opt_body_quat,
        ghost_joint_pos,
        ghost_body_pos,
        ghost_body_quat,
        motion,
    )

    # IK error summary
    ik_pos_errors = np.array(ik_pos_errors)
    print(f"  IK summary: p50={np.percentile(ik_pos_errors,50):.0f}mm "
          f"p95={np.percentile(ik_pos_errors,95):.0f}mm "
          f"max={ik_pos_errors.max():.0f}mm  "
          f"bad(>200mm)={np.sum(ik_pos_errors>200)}/{N}")

    # Compute velocities
    raw_joint = motion.joint_pos[:N].copy()
    raw_body = motion.body_pos[:N].copy()
    raw_bquat = motion.body_quat[:N].copy()
    raw_jv, raw_blv, raw_bav = _compute_velocities(raw_joint, raw_body, raw_bquat, fps)
    ghost_jv, ghost_blv, ghost_bav = _compute_velocities(ghost_joint_pos, ghost_body_pos, ghost_body_quat, fps)
    opt_jv, opt_blv, opt_bav = _compute_velocities(opt_joint_pos, opt_body_pos, opt_body_quat, fps)

    return {
        "fps": fps,
        "ik_pos_errors": ik_pos_errors.copy(),
        "ik_bad_frame_count": int(np.sum(ik_pos_errors > float(args.batch_reject_ik_mm))),
        "raw": (raw_joint, raw_jv, raw_body, raw_bquat, raw_blv, raw_bav),
        "ghost": (ghost_joint_pos, ghost_jv, ghost_body_pos, ghost_body_quat, ghost_blv, ghost_bav),
        "optimized": (opt_joint_pos, opt_jv, opt_body_pos, opt_body_quat, opt_blv, opt_bav),
    }


# ---------------------------------------------------------------------------
# Batch mode: run multiple rounds with randomized SE(2) transforms
# ---------------------------------------------------------------------------

def _motion_tag(motion_path, start_frame):
    """Build a compact tag from the motion filename and start frame."""
    stem = os.path.splitext(os.path.basename(motion_path))[0]
    return f"{stem}_f{start_frame}"


def _optimized_variant_name(planner: str) -> str:
    if planner == "terrain_ref":
        return "ik_only"
    if planner == "mppi":
        return "mppi_ik"
    if planner == "plum":
        return "plum_ik"
    return f"{planner}_ik"


def _run_batch(args, terrain, xml_path):
    """Run batch sampling: multiple rounds with randomized x/y/yaw.

    Each round outputs 3 NPZ files:
        raw/{motion_tag}_round_XXXX_dx...npz
        optimized/{motion_tag}_round_XXXX_dx...npz
        ghost/{motion_tag}_round_XXXX_dx...npz

    All body_pos_w and body_quat_w are in **global** (world) coordinates.
    """
    out_dir = args.batch_output_dir
    raw_dir = os.path.join(out_dir, "raw")
    opt_dir = os.path.join(out_dir, "optimized")
    ghost_dir = os.path.join(out_dir, "ghost")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(ghost_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_rounds = args.n_rounds
    round_start = args.round_start
    accepted_rounds = 0
    rejected_rounds = 0
    attempted_rounds = 0

    # Compute effective lowpass cutoff for batch optimized export.
    # Priority:
    #   1. batch_export_lowpass_cutoff > 0: explicit batch cutoff
    #   2. batch_export_lowpass_cutoff == 0: fall back to generic export_lowpass_cutoff
    #   3. batch_export_lowpass_cutoff < 0: disable batch lowpass
    if args.batch_export_lowpass_cutoff < 0:
        effective_cutoff = 0.0
    elif args.batch_export_lowpass_cutoff > 0:
        effective_cutoff = float(args.batch_export_lowpass_cutoff)
    else:
        effective_cutoff = float(args.export_lowpass_cutoff)
    if args.hardware_bandwidth_hz > 0:
        hw_cutoff = 0.8 * args.hardware_bandwidth_hz
        effective_cutoff = hw_cutoff if effective_cutoff <= 0 else min(effective_cutoff, hw_cutoff)

    print("=" * 60)
    print(f"BATCH MODE: target_valid_rounds={n_rounds}, round_start={round_start}")
    print(f"  dx range: {args.dx_range}")
    print(f"  dy range: {args.dy_range}")
    print(f"  dyaw range: {args.dyaw_range}")
    print(f"  output: {out_dir}")
    print(f"  reject if ik_bad_frames(>{args.batch_reject_ik_mm:.0f}mm) > {args.batch_reject_ik_bad_frames}")
    if effective_cutoff > 0:
        print(f"  export lowpass: {effective_cutoff:.1f} Hz (order={args.export_lowpass_order})")
    print("=" * 60)

    # Skip transforms for attempts before round_start to keep reproducibility.
    for _ in range(round_start):
        rng.uniform(args.dx_range[0], args.dx_range[1])
        rng.uniform(args.dy_range[0], args.dy_range[1])
        rng.uniform(args.dyaw_range[0], args.dyaw_range[1])

    t_total_start = time.time()
    round_idx = round_start
    while accepted_rounds < n_rounds:
        dx = float(rng.uniform(args.dx_range[0], args.dx_range[1]))
        dy = float(rng.uniform(args.dy_range[0], args.dy_range[1]))
        dyaw = float(rng.uniform(args.dyaw_range[0], args.dyaw_range[1]))
        attempted_rounds += 1

        # Format filename: round_XXXX_dx+1.234_dy-5.678_dyaw+0.1234
        dx_s = f"{dx:+.3f}"
        dy_s = f"{dy:+.3f}"
        dyaw_s = f"{dyaw:+.4f}"
        mtag = _motion_tag(args.motion, args.start_frame)
        tag = f"{mtag}_round_{round_idx:04d}_dx{dx_s}_dy{dy_s}_dyaw{dyaw_s}"

        print(f"\n{'='*60}")
        print(
            f"[Attempt {attempted_rounds}] valid={accepted_rounds}/{n_rounds} "
            f"idx={round_idx} dx={dx:.3f} dy={dy:.3f} dyaw={dyaw:.4f}"
        )
        print(f"{'='*60}")
        t0 = time.time()

        try:
            result = run_one_round(args, terrain, xml_path, dx, dy, dyaw)
        except Exception as e:
            print(f"  [ERROR] Round {round_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            rejected_rounds += 1
            round_idx += 1
            continue

        bad_frames = int(result.get("ik_bad_frame_count", 0))
        if args.batch_reject_ik_bad_frames >= 0 and bad_frames > int(args.batch_reject_ik_bad_frames):
            dt_round = time.time() - t0
            print(
                f"  [REJECT] ik_bad_frames={bad_frames} > {args.batch_reject_ik_bad_frames} "
                f"(threshold={args.batch_reject_ik_mm:.0f}mm)  ({dt_round:.1f}s)"
            )
            rejected_rounds += 1
            round_idx += 1
            continue

        fps = result["fps"]
        # Save raw
        jp, jv, bp, bq, blv, bav = result["raw"]
        raw_path = os.path.join(raw_dir, f"{tag}_raw.npz")
        _save_one_npz(
            raw_path, fps, jp, bp, bq, jv, blv, bav,
            transform_dx=dx, transform_dy=dy, transform_dyaw=dyaw,
            extra_fields={
                "planner": args.planner,
                "adapter_variant": "raw",
            },
        )
        # Save optimized (with optional export lowpass)
        jp, jv, bp, bq, blv, bav = result["optimized"]
        if effective_cutoff > 0:
            jp, bp, bq, jv, blv, bav = filter_lower_body_motion_for_export(
                xml_path, jp, bp, bq, fps,
                cutoff_hz=effective_cutoff,
                order=args.export_lowpass_order,
            )
        opt_path = os.path.join(opt_dir, f"{tag}_optimized.npz")
        _save_one_npz(
            opt_path, fps, jp, bp, bq, jv, blv, bav,
            transform_dx=dx, transform_dy=dy, transform_dyaw=dyaw,
            extra_fields={
                "planner": args.planner,
                "adapter_variant": _optimized_variant_name(args.planner),
                "ik_pos_error_mm": result["ik_pos_errors"].astype(np.float32),
                "ik_bad_frame_count": np.int32(result["ik_bad_frame_count"]),
            },
        )
        # Save ghost
        jp, jv, bp, bq, blv, bav = result["ghost"]
        ghost_path = os.path.join(ghost_dir, f"{tag}_ghost.npz")
        _save_one_npz(
            ghost_path, fps, jp, bp, bq, jv, blv, bav,
            transform_dx=dx, transform_dy=dy, transform_dyaw=dyaw,
            extra_fields={
                "planner": args.planner,
                "adapter_variant": "z_only" if args.planner == "terrain_ref" else "ghost",
            },
        )

        dt_round = time.time() - t0
        print(f"  Saved: {tag}_{{raw,optimized,ghost}}.npz  ({dt_round:.1f}s)")
        accepted_rounds += 1
        round_idx += 1

    dt_total = time.time() - t_total_start
    summary_path = os.path.join(out_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "planner": args.planner,
                "motion": args.motion,
                "xml": xml_path,
                "seed": int(args.seed),
                "accepted_rounds": int(accepted_rounds),
                "attempted_rounds": int(attempted_rounds),
                "rejected_rounds": int(rejected_rounds),
                "total_time_sec": float(dt_total),
                "valid_clip_per_sec": float(accepted_rounds / dt_total) if dt_total > 0 else None,
            },
            f,
            indent=2,
        )
    print(f"\n{'='*60}")
    print(
        f"BATCH COMPLETE: valid={accepted_rounds}/{n_rounds} "
        f"attempts={attempted_rounds} rejected={rejected_rounds} "
        f"in {dt_total:.1f}s ({dt_total/max(accepted_rounds,1):.1f}s/valid_round)"
    )
    print(f"{'='*60}")


def _run_batch_plum(args, terrain, xml_path):
    """Batch mode for plum-blossom-pole terrain: iterate over units.

    Parses the terrain XML to find all unit bodies (named 'unit_XXXX'),
    computes the SE(2) transform to place the robot on platform_1 of each unit
    facing toward platform_2, and runs the full pipeline.
    """
    import mujoco as _mj

    out_dir = args.batch_output_dir
    raw_dir = os.path.join(out_dir, "raw")
    opt_dir = os.path.join(out_dir, "optimized")
    ghost_dir = os.path.join(out_dir, "ghost")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(ghost_dir, exist_ok=True)

    # Parse unit positions from MuJoCo model.
    # Units are identified by platform geom names: u{XXXX}_plat1 / u{XXXX}_plat2.
    # Unit center = midpoint of plat1 and plat2 positions.
    model = _mj.MjModel.from_xml_path(xml_path)
    plat1_map = {}  # unit_idx -> (x, y)
    plat2_map = {}
    for gid in range(model.ngeom):
        name = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, gid) or ""
        if "_plat1" in name:
            uid = name.split("_plat1")[0]  # e.g. "u0000"
            plat1_map[uid] = (float(model.geom_pos[gid, 0]), float(model.geom_pos[gid, 1]))
        elif "_plat2" in name:
            uid = name.split("_plat2")[0]
            plat2_map[uid] = (float(model.geom_pos[gid, 0]), float(model.geom_pos[gid, 1]))
    units = []  # (name, cx, cy)
    for uid in sorted(plat1_map.keys()):
        if uid in plat2_map:
            p1 = plat1_map[uid]
            p2 = plat2_map[uid]
            cx = 0.5 * (p1[0] + p2[0])
            cy = 0.5 * (p1[1] + p2[1])
            units.append((uid, cx, cy))
    print(f"Found {len(units)} terrain units")

    if not units:
        print("ERROR: No 'unit_XXXX' bodies found in XML. Aborting.")
        return

    # Motion initial pelvis position (needed to compute dx/dy)
    motion_tmp = MotionClip(
        args.motion, args.start_frame, args.n_frames,
        contact_z_threshold=args.contact_z_threshold,
        contact_speed_threshold=args.contact_speed_threshold,
    )
    pelvis_start_xy = motion_tmp.body_pos[0, NPZ_PELVIS, :2].copy()
    del motion_tmp

    # Compute platform_1 local y offset from actual data.
    # plat1 is the platform with smaller y relative to unit center.
    uid0 = sorted(plat1_map.keys())[0]
    p1_0 = plat1_map[uid0]
    p2_0 = plat2_map[uid0]
    unit_cy_0 = 0.5 * (p1_0[1] + p2_0[1])
    plat1_local_y = p1_0[1] - unit_cy_0  # negative value (e.g. -1.9)
    print(f"  platform_1 local y offset: {plat1_local_y:.3f}m")

    # Walking direction of raw motion at start
    # For walk1_subject1 frame 1100: walks in -Y direction
    # We want robot to walk in +Y (from plat1 toward plat2)
    # So dyaw = pi (flip 180 degrees)
    dyaw = np.pi

    # Compute effective lowpass cutoff
    if args.batch_export_lowpass_cutoff < 0:
        effective_cutoff = 0.0
    elif args.batch_export_lowpass_cutoff > 0:
        effective_cutoff = float(args.batch_export_lowpass_cutoff)
    else:
        effective_cutoff = float(args.export_lowpass_cutoff)
    if args.hardware_bandwidth_hz > 0:
        hw_cutoff = 0.8 * args.hardware_bandwidth_hz
        effective_cutoff = hw_cutoff if effective_cutoff <= 0 else min(effective_cutoff, hw_cutoff)

    # Subset of units to process
    start_idx = args.round_start
    n_total = min(args.n_rounds, len(units)) if args.n_rounds > 0 else len(units)
    end_idx = min(start_idx + n_total, len(units))

    print("=" * 60)
    print(f"BATCH PLUM BLOSSOM: units [{start_idx}, {end_idx}) of {len(units)}")
    print(f"  output: {out_dir}")
    if effective_cutoff > 0:
        print(f"  export lowpass: {effective_cutoff:.1f} Hz")
    print("=" * 60)

    accepted = 0
    rejected = 0
    t_total_start = time.time()

    for ui in range(start_idx, end_idx):
        unit_name, unit_cx, unit_cy = units[ui]

        # Compute transform: after dyaw=pi rotation around pelvis_start,
        # pelvis is still at pelvis_start_xy. Then translate to platform_1 center.
        # Target position = (unit_cx, unit_cy + plat1_local_y)
        dx = unit_cx - pelvis_start_xy[0]
        dy = (unit_cy + plat1_local_y) - pelvis_start_xy[1]

        mtag = _motion_tag(args.motion, args.start_frame)
        tag = f"{mtag}_round_{ui:04d}_dx{dx:+.3f}_dy{dy:+.3f}_dyaw{dyaw:+.4f}"

        print(f"\n{'='*60}")
        print(f"[{ui - start_idx + 1}/{end_idx - start_idx}] {unit_name} "
              f"pos=({unit_cx:.1f}, {unit_cy:.1f}) dx={dx:.3f} dy={dy:.3f}")
        print(f"{'='*60}")
        t0 = time.time()

        try:
            result = run_one_round(args, terrain, xml_path, dx, dy, dyaw)
        except Exception as e:
            print(f"  [ERROR] {unit_name}: {e}")
            import traceback
            traceback.print_exc()
            rejected += 1
            continue

        bad_frames = int(result.get("ik_bad_frame_count", 0))
        if args.batch_reject_ik_bad_frames >= 0 and bad_frames > int(args.batch_reject_ik_bad_frames):
            dt_r = time.time() - t0
            print(f"  [REJECT] ik_bad_frames={bad_frames} > {args.batch_reject_ik_bad_frames} ({dt_r:.1f}s)")
            rejected += 1
            continue

        fps = result["fps"]
        # Save raw
        jp, jv, bp, bq, blv, bav = result["raw"]
        _save_one_npz(os.path.join(raw_dir, f"{tag}_raw.npz"),
                      fps, jp, bp, bq, jv, blv, bav,
                      transform_dx=dx, transform_dy=dy, transform_dyaw=dyaw,
                      extra_fields={
                          "planner": args.planner,
                          "adapter_variant": "raw",
                      })
        # Save optimized
        jp, jv, bp, bq, blv, bav = result["optimized"]
        if effective_cutoff > 0:
            jp, bp, bq, jv, blv, bav = filter_lower_body_motion_for_export(
                xml_path, jp, bp, bq, fps,
                cutoff_hz=effective_cutoff, order=args.export_lowpass_order)
        _save_one_npz(os.path.join(opt_dir, f"{tag}_optimized.npz"),
                      fps, jp, bp, bq, jv, blv, bav,
                      transform_dx=dx, transform_dy=dy, transform_dyaw=dyaw,
                      extra_fields={
                          "planner": args.planner,
                          "adapter_variant": _optimized_variant_name(args.planner),
                          "ik_pos_error_mm": result["ik_pos_errors"].astype(np.float32),
                          "ik_bad_frame_count": np.int32(result["ik_bad_frame_count"]),
                      })
        # Save ghost
        jp, jv, bp, bq, blv, bav = result["ghost"]
        _save_one_npz(os.path.join(ghost_dir, f"{tag}_ghost.npz"),
                      fps, jp, bp, bq, jv, blv, bav,
                      transform_dx=dx, transform_dy=dy, transform_dyaw=dyaw,
                      extra_fields={
                          "planner": args.planner,
                          "adapter_variant": "z_only" if args.planner == "terrain_ref" else "ghost",
                      })

        dt_r = time.time() - t0
        print(f"  Saved: {tag}  ({dt_r:.1f}s)")
        accepted += 1

    dt_total = time.time() - t_total_start
    summary_path = os.path.join(out_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "planner": args.planner,
                "motion": args.motion,
                "xml": xml_path,
                "seed": int(args.seed),
                "accepted_rounds": int(accepted),
                "attempted_rounds": int(end_idx - start_idx),
                "rejected_rounds": int(rejected),
                "total_time_sec": float(dt_total),
                "valid_clip_per_sec": float(accepted / dt_total) if dt_total > 0 else None,
            },
            f,
            indent=2,
        )
    print(f"\n{'='*60}")
    print(f"BATCH PLUM COMPLETE: accepted={accepted} rejected={rejected} "
          f"total={end_idx - start_idx} in {dt_total:.1f}s")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 主函数：完整管线 = 加载数据 → 构建地形参考 → MPPI 优化 → 骨盆滤波 → IK → MuJoCo 渲染
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MPPI foot trajectory planner")
    # ---- 基础参数 ----
    parser.add_argument("--motion", type=str, default="assets/motions/walk1_subject1.npz")  # 动捕 npz 文件路径
    parser.add_argument("--xml", type=str, default="assets/g1/g1_29dof_scene_stairs_ud.xml")  # MuJoCo 场景 XML 路径
    parser.add_argument("--start_frame", type=int, default=2600)  # 起始帧号 #2400
    parser.add_argument("--n_frames", type=int, default=1000)  # 使用的总帧数 #400
    parser.add_argument("--speed", type=float, default=1.0)  # 回放速度倍率
    # ---- 触地检测参数 ----
    parser.add_argument("--contact_z_threshold", type=float, default=CONTACT_Z_THRESHOLD)  # 脚踝 z 阈值
    parser.add_argument("--contact_speed_threshold", type=float, default=CONTACT_SPEED_THRESHOLD)  # 脚踝速度阈值
    # ---- TerrainReference 参数 ----
    parser.add_argument("--lookahead", type=float, default=0.0)  # 地形前瞻距离 (米)
    parser.add_argument("--alpha", type=float, default=0.08)  # 平滑系数 (EMA alpha)
    parser.add_argument("--footstep_margin", type=float, default=0.10)  # 落脚点裕量 (米)
    parser.add_argument("--toe_offset_x", type=float, default=TOE_OFFSET_X)  # 脚尖 x 偏移
    parser.add_argument("--heel_offset_x", type=float, default=-0.06)  # 脚跟 x 偏移
    parser.add_argument("--mid_offset_x", type=float, default=0.15)  # 中足 x 偏移
    parser.add_argument("--toe_margin", type=float, default=0.005)  # 脚尖安全裕量
    parser.add_argument("--swing_floor_margin", type=float, default=0.003)  # 摆动时地面间距裕量
    # ---- 轨迹规划器选择 ----
    parser.add_argument("--planner", type=str, default="mppi",
                        choices=["mppi", "terrain_ref", "plum"],
                        help="摆动轨迹优化器: mppi (采样优化，默认), "
                             "terrain_ref (不优化摆腿，仅 TerrainReference + IK), "
                             "plum (梅花桩 xy 着地)")
    parser.add_argument("--plum_z_offset", type=float, default=0.0,
                        help="Z offset to lift entire motion for plum blossom terrain (m). "
                             "Set to pole height (e.g. 0.2) when using --planner plum.")
    # ---- MPPI cost 权重 (也用于 terrain_ref skip-collision 检查) ----
    parser.add_argument("--w_track", type=float, default=10.0)  # 跟踪项权重 (跟踪参考轨迹)
    parser.add_argument("--w_terrain", type=float, default=1000.0)  # 地形穿透惩罚权重
    parser.add_argument("--h_clearance", type=float, default=0.03)  # 地形间距裕量 (米)
    # ---- MPPI 采样参数 ----
    parser.add_argument("--mppi_n_knots", type=int, default=5)      # MPPI 内部控制点数
    parser.add_argument("--mppi_n_samples", type=int, default=128)  # MPPI 采样数
    parser.add_argument("--mppi_n_iterations", type=int, default=10) # MPPI 迭代次数
    parser.add_argument("--mppi_temperature", type=float, default=0.1) # MPPI softmax 温度
    # ---- Lower-leg / foot collision post-pass ----
    parser.add_argument("--leg_collision_iters", type=int, default=4,
                        help="Lower-leg/foot collision resolution passes after swing planning")
    parser.add_argument("--leg_collision_margin", type=float, default=0.025,
                        help="Desired capsule clearance margin (m)")
    parser.add_argument("--shin_collision_radius", type=float, default=0.06,
                        help="Capsule radius for shin collision checks (m)")
    parser.add_argument("--foot_collision_radius", type=float, default=0.04,
                        help="Capsule radius for foot collision checks (m)")
    parser.add_argument("--collision_push_gain", type=float, default=0.9,
                        help="Swing-foot outward push gain for collision resolution")
    parser.add_argument("--collision_max_push", type=float, default=0.06,
                        help="Maximum per-frame collision push applied to a swing foot (m)")
    parser.add_argument("--pelvis_side_margin", type=float, default=0.04,
                        help="Minimum pelvis-local lateral margin kept for each foot during swing (m)")
    parser.add_argument("--pelvis_side_gain", type=float, default=0.5,
                        help="Relative gain for the pelvis-side barrier inside collision resolution")
    parser.add_argument("--no_stance_latch", action="store_true",
                        help="Disable stance-foot target latching after collision post-pass")
    parser.add_argument("--stance_latch_ground_tol", type=float, default=0.05,
                        help="Max |foot_z - terrain_contact_z| for selecting a stance latch anchor (m)")
    # ---- IK 求解器参数 ----
    parser.add_argument("--ik_max_iters", type=int, default=60)  # IK 最大迭代次数
    parser.add_argument("--ik_tol", type=float, default=1e-3)  # IK 收敛容差
    parser.add_argument("--ik_damping", type=float, default=0.08)  # IK 阻尼系数 (防止奇异)
    parser.add_argument("--ik_step_size", type=float, default=0.5)  # IK 步长 (0-1)
    parser.add_argument("--ik_orientation_weight", type=float, default=0.3)  # IK 朝向权重
    parser.add_argument("--ik_posture_weight", type=float, default=0.08,
                        help="姿态正则化 (保持关节角接近动捕数据)")
    parser.add_argument("--root_xy_weight", type=float, default=2.0,
                        help="根节点 x,y 跟踪权重 (中等=跟随行走方向)")
    parser.add_argument("--root_z_weight", type=float, default=10.0,
                        help="根节点 z 跟踪权重 (高=防止下沉)")
    parser.add_argument("--ik_penetration_weight", type=float, default=10.0,
                        help="脚部穿透惩罚权重 (不对称：只惩罚低于目标 z 的情况)")
    parser.add_argument("--ik_config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "ik_config.json"),
                        help="IK 权重 JSON 配置文件 (覆盖其他 --ik_* 参数)")
    parser.add_argument("--ik_backend", type=str, choices=["jacobian", "curobo", "drake"],
                        default="jacobian",
                        help="IK backend: jacobian (CPU, original), curobo (GPU, cuRobo L-BFGS), or drake (SNOPT constrained opt)")
    parser.add_argument("--ik_mode", type=str, choices=["frame", "window"], default="frame",
                        help="IK mode: frame (per-frame solve) or window (multi-frame trajectory optimization)")
    parser.add_argument("--ik_window", type=int, default=9, help="Window size for trajectory IK")
    parser.add_argument("--ik_commit", type=int, default=3, help="Commit size for trajectory IK")
    parser.add_argument("--traj_ik_max_nfev", type=int, default=8)
    parser.add_argument("--traj_ik_w_vel", type=float, default=2.0)
    parser.add_argument("--traj_ik_w_acc", type=float, default=10.0)
    parser.add_argument("--traj_ik_w_root_vel", type=float, default=5.0)
    parser.add_argument("--traj_ik_w_root_acc", type=float, default=20.0)
    parser.add_argument("--traj_ik_w_stance_lock", type=float, default=50.0)
    parser.add_argument("--traj_ik_skip_init_err_mm", type=float, default=80.0,
                        help="Skip solving windows whose single-frame init max error stays below this threshold (mm).")
    parser.add_argument("--no_ik_lower_body_stabilizer", action="store_true",
                        help="Disable the lower-body stance stabilizer post-pass for Jacobian IK.")
    parser.add_argument("--ik_stabilizer_stance_scale", type=float, default=2.2,
                        help="Multiplier on foot point scales for stance feet during lower-body stabilization.")
    parser.add_argument("--ik_stabilizer_swing_scale", type=float, default=0.60,
                        help="Multiplier on foot point scales for swing feet during lower-body stabilization.")
    parser.add_argument("--ik_stabilizer_root_xy_scale", type=float, default=0.60,
                        help="Scale root xy tracking weight during lower-body stabilization when any foot is in stance.")
    parser.add_argument("--ik_stabilizer_root_z_scale", type=float, default=0.75,
                        help="Scale root z tracking weight during lower-body stabilization when any foot is in stance.")
    parser.add_argument("--ik_stabilizer_root_continuity_weight", type=float, default=80.0,
                        help="Root continuity weight for the lower-body stabilizer post-pass.")
    parser.add_argument("--ik_stabilizer_trigger_mm", type=float, default=60.0,
                        help="Only run the lower-body stabilizer on frames whose base IK error exceeds this threshold (mm).")
    parser.add_argument("--ik_stabilizer_cooldown_frames", type=int, default=6,
                        help="After a triggered frame, keep the lower-body stabilizer active for this many subsequent frames.")
    parser.add_argument("--rolling_interval", type=int, default=0,
                        help="Rolling MPPI recompute interval (frames). 0=disabled (use precomputed). "
                             "e.g. 100 = recompute swing trajectories every 100 frames from current position")
    # ---- Ghost 参考机器人 (半透明幽灵，显示 TerrainReference 而非 MPPI 的结果) ----
    parser.add_argument("--no_ghost_ref", action="store_true")  # 禁用幽灵显示
    parser.add_argument("--ghost_alpha", type=float, default=0.35)  # 幽灵透明度
    parser.add_argument("--ghost_color", type=str, default="0.15,0.65,1.0")  # 幽灵 RGB 颜色
    parser.add_argument("--no_debug_overlay", action="store_true",
                        help="Disable trail lines and target markers; show only the robot body (and optional ghost).")
    # ---- 日志 / 导出 ----
    parser.add_argument("--log_every", type=int, default=50)  # 每 N 帧打印状态日志
    parser.add_argument("--ik_log_every", type=int, default=50)  # IK 日志间隔
    parser.add_argument("--export_npz", type=str, default=None)  # 导出轨迹 npz 路径
    parser.add_argument("--export_motion_npz", type=str, default=None)  # 导出完整动作 npz 路径
    parser.add_argument("--export_trio_npz", type=str, default=None,
                        help="Export three NPZ files: <prefix>_raw.npz, <prefix>_offset.npz, "
                             "<prefix>_mppi.npz. Each has the same fields as the original motion NPZ.")
    parser.add_argument("--seed", type=int, default=42)  # 随机种子
    # ---- 导出侧零相位低通 ----
    parser.add_argument("--export_lowpass_cutoff", type=float, default=0.0,
                        help="Zero-phase lowpass cutoff (Hz) for exported npz. 0=disabled.")
    parser.add_argument("--export_lowpass_order", type=int, default=4,
                        help="Butterworth filter order for export lowpass.")
    parser.add_argument("--batch_export_lowpass_cutoff", type=float, default=5.0,
                        help="Lowpass cutoff (Hz) for batch optimized exports. >0=force, 0=use export_lowpass_cutoff, <0=disable.")
    parser.add_argument("--batch_reject_ik_mm", type=float, default=200.0,
                        help="Batch round rejection threshold in mm for per-frame IK max error.")
    parser.add_argument("--batch_reject_ik_bad_frames", type=int, default=8,
                        help="Reject a batch round when frames with IK error > batch_reject_ik_mm exceed this count. <0=disable rejection.")
    parser.add_argument("--hardware_bandwidth_hz", type=float, default=0.0,
                        help="Robot hardware bandwidth (Hz). If >0, effective cutoff = "
                             "min(export_lowpass_cutoff, 0.8*hardware_bandwidth_hz).")
    # ---- 批量采样模式 (headless, 无 viewer) ----
    parser.add_argument("--n_rounds", type=int, default=0,
                        help="批量采样轮数。>0 时启用 headless 批量模式，"
                             "每轮随机化 x/y/yaw，输出 raw/optimized/ghost 三个 npz。")
    parser.add_argument("--round_start", type=int, default=0,
                        help="批量采样起始轮编号 (用于断点续采)。")
    parser.add_argument("--dx_range", type=float, nargs=2, default=[-30.0, 30.0],
                        metavar=("MIN", "MAX"),
                        help="随机 x 偏移范围 (米)。")
    parser.add_argument("--dy_range", type=float, nargs=2, default=[-30.0, 30.0],
                        metavar=("MIN", "MAX"),
                        help="随机 y 偏移范围 (米)。")
    parser.add_argument("--dyaw_range", type=float, nargs=2, default=[-3.14, 3.14],
                        metavar=("MIN", "MAX"),
                        help="随机 yaw 偏移范围 (弧度)。")
    parser.add_argument("--batch_output_dir", type=str,
                        default="outputs/terrain",
                        help="批量模式输出根目录，下设 raw/ optimized/ ghost/ 子目录。")
    parser.add_argument("--batch_plum", action="store_true",
                        help="Plum-blossom-pole batch mode: iterate over all 'unit_XXXX' bodies "
                             "in the terrain XML, placing robot on platform_1 facing platform_2.")
    # 随机起始偏移
    parser.add_argument("--xy_offset_range", type=float, default=1,
                        help="Random uniform x,y offset range (metres). "
                             "E.g. 0.5 → x,y each sampled from [-0.5, 0.5].")
    parser.add_argument("--xy_offset", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="Exact x,y offset in metres. E.g. --xy_offset 0.3 -0.1")
    parser.add_argument("--yaw_offset", type=float, default=0.0,
                        help="Yaw rotation applied to the raw motion (degrees). "
                             "Rotates all body positions and orientations around the "
                             "initial pelvis position. Joint angles are unchanged.")
    args = parser.parse_args()

    # 如果未指定 XML，使用默认的上下楼梯场景
    if args.xml is None:
        args.xml = os.path.join(
            os.path.dirname(__file__), "..", "assets", "g1",
            "g1_29dof_scene_stairs_ud.xml",
        )
    xml_path = os.path.abspath(args.xml)

    print("=" * 60)
    print("MPPI FOOT PLANNER")
    print("=" * 60)

    # ===== 第 1 步：加载地形 =====
    # 从 MuJoCo XML 构建光线投射地形查询对象
    print("[1] Loading terrain...")
    terrain = RaycastTerrain.from_xml_path(xml_path)  # 加载 MuJoCo 场景并初始化光线投射
    terrain.print_info()  # 打印地形信息 (台阶数量、高度等)

    # ===== 批量采样模式 =====
    if args.batch_plum:
        _run_batch_plum(args, terrain, xml_path)
        return
    if args.n_rounds > 0:
        _run_batch(args, terrain, xml_path)
        return

    # ===== 第 2 步：加载动捕数据 =====
    print(f"\n[2] Loading motion (frames {args.start_frame}-{args.start_frame + args.n_frames})...")
    motion = MotionClip(
        args.motion, args.start_frame, args.n_frames,
        contact_z_threshold=args.contact_z_threshold,  # 脚踝 z < 阈值 → 判定触地
        contact_speed_threshold=args.contact_speed_threshold,  # 脚踝速度 < 阈值 → 确认触地
    )
    phase_clock_trusted = _should_trust_phase_clock(motion.phase_params)
    if motion.phase_params is not None:
        pp = motion.phase_params
        trust_str = "trusted" if phase_clock_trusted else "disabled"
        print(f"  Phase clock: type={pp.gait_type}, fit={pp.fit_quality:.2f} [{trust_str}]")

    # 施加随机 x,y 起始偏移 (平移所有刚体位置；z 不变)
    rng = np.random.default_rng(args.seed)
    if args.xy_offset_range > 0.0:
        dx = rng.uniform(-args.xy_offset_range, args.xy_offset_range)
        dy = rng.uniform(-args.xy_offset_range, args.xy_offset_range)
        motion.body_pos[:, :, 0] += dx
        motion.body_pos[:, :, 1] += dy
        print(f"  Applied random start offset: dx={dx:.3f}m, dy={dy:.3f}m")

    # 施加精确 x,y 偏移 (与 xy_offset_range 可叠加使用)
    if args.xy_offset is not None:
        dx, dy = args.xy_offset
        motion.body_pos[:, :, 0] += dx
        motion.body_pos[:, :, 1] += dy
        print(f"  Applied exact xy offset: dx={dx:.3f}m, dy={dy:.3f}m")

    # 施加 yaw 旋转 (绕初始骨盆位置的 z 轴旋转)
    if abs(args.yaw_offset) > 1e-6:
        yaw_rad = np.deg2rad(args.yaw_offset)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        # 2D rotation matrix for xy
        R2 = np.array([[cos_y, -sin_y],
                        [sin_y,  cos_y]], dtype=np.float64)
        # Yaw quaternion [w, x, y, z]
        q_yaw = np.array([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)],
                         dtype=np.float64)
        # Rotate around the initial pelvis xy position
        pivot_xy = motion.body_pos[0, NPZ_PELVIS, :2].copy()
        N_frames = motion.body_pos.shape[0]
        n_bodies = motion.body_pos.shape[1]
        # Rotate body positions: translate to pivot, rotate xy, translate back
        # Vectorized over all frames and bodies at once
        xy_all = motion.body_pos[:, :, :2] - pivot_xy[None, None, :]  # (N, B, 2)
        motion.body_pos[:, :, :2] = np.einsum('ij,...j->...i', R2, xy_all) + pivot_xy[None, None, :]
        # Rotate body quaternions: q_new = q_yaw * q_old (pre-multiply)
        # Vectorized Hamilton product over all frames and bodies
        w0, x0, y0, z0 = q_yaw
        q = motion.body_quat  # (N, n_bodies, 4) — [w, x, y, z]
        w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        motion.body_quat = np.stack([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
        ], axis=-1).astype(motion.body_quat.dtype)
        # Rotate body linear velocities (xy components)
        if hasattr(motion, 'body_lin_vel'):
            for bi in range(n_bodies):
                xy_v = motion.body_lin_vel[:, bi, :2].copy()
                motion.body_lin_vel[:, bi, :2] = (R2 @ xy_v.T).T
        print(f"  Applied yaw offset: {args.yaw_offset:.1f}° ({yaw_rad:.4f} rad)")

    # Z offset (plum blossom mode)
    if args.plum_z_offset > 0:
        motion.body_pos[:, :, 2] += args.plum_z_offset
        print(f"  Applied plum_z_offset: +{args.plum_z_offset:.3f}m")

    print(f"  {motion.n_frames} frames @ {motion.fps}Hz")
    print(f"  foot_nominal_z = {motion.foot_nominal_z:.3f}m")  # 脚踝在平地上的标称 z 高度
    print(f"  pelvis_height_above_foot = {motion.pelvis_height_above_foot:.3f}m")  # 骨盆到脚的高度差

    # ===== 第 3 步：构建地形参考轨迹 =====
    # TerrainReference 将原始动捕脚位适配到目标地形:
    #   - 支撑脚锚定到台阶表面 (通过 FootstepResolver 计算安全落脚位置)
    #   - 摆动脚在抬脚点和落脚点之间插值 (三次 Hermite xy + 五次多项式 z + 形状渐变)
    print("\n[3] Building terrain reference...")
    ref_builder = TerrainReference(
        terrain=terrain, motion=motion,
        lookahead=args.lookahead,             # 地形前瞻距离
        smoothing_alpha=args.alpha,            # EMA 平滑系数
        footstep_margin=args.footstep_margin,  # 落脚点安全裕量
        toe_offset_x=args.toe_offset_x,       # 脚尖 x 偏移
        heel_offset_x=args.heel_offset_x,     # 脚跟 x 偏移
        mid_offset_x=args.mid_offset_x,       # 中足 x 偏移
        toe_margin=args.toe_margin,            # 脚尖安全裕量
        swing_floor_margin=args.swing_floor_margin,  # 摆动地面间距裕量
    )

    # ===== 第 4 步：预计算摆动轨迹 =====
    planner_name = args.planner.upper()
    print(f"\n[4] Pre-computing {planner_name} swing trajectories...")
    # 中足接触 z 高度：中足虚拟中心在支撑期间位于
    # foot_nominal_z + MID_FOOT_OFFSET[2] 处 (比脚踝低 3cm，因为中足偏移的 z 分量为负)
    midfoot_contact_z = motion.foot_nominal_z + float(MID_FOOT_OFFSET[2])
    print(f"  Mid-foot contact z: {midfoot_contact_z:.4f}m (ankle={motion.foot_nominal_z:.4f}m, offset={MID_FOOT_OFFSET[2]:.4f}m)")

    # 构造 MPPI 优化器参数
    opt_params = MppiFootParams(
        n_knots=args.mppi_n_knots,
        n_samples=args.mppi_n_samples,
        n_iterations=args.mppi_n_iterations,
        temperature=args.mppi_temperature,
        w_track=args.w_track,
        w_terrain=args.w_terrain,
        h_clearance=args.h_clearance,
        h_contact_z=midfoot_contact_z,
    )

    # 统计左右脚的摆动阶段数量
    left_phases = _find_swing_phases(motion.left_contact)
    right_phases = _find_swing_phases(motion.right_contact)
    print(f"  Left foot:  {len(left_phases)} swing phases")
    print(f"  Right foot: {len(right_phases)} swing phases")

    pelvis_quats = motion.body_quat[:, NPZ_PELVIS]  # (N, 4) wxyz — 骨盆四元数作为 yaw 来源

    # Enforce lateral separation (same as pre-computation path)
    _min_lateral_sep = 0.05
    _pelvis_xy2 = motion.body_pos[:, NPZ_PELVIS, :2]
    for i in range(len(pelvis_quats)):
        yaw = extract_yaw_from_quat(pelvis_quats[i])
        lat = np.array([-np.sin(yaw), np.cos(yaw)])
        lf_lat = np.dot(ref_builder.left_precomputed[i, :2] - _pelvis_xy2[i], lat)
        rf_lat = np.dot(ref_builder.right_precomputed[i, :2] - _pelvis_xy2[i], lat)
        if lf_lat - rf_lat < _min_lateral_sep:
            deficit = _min_lateral_sep - (lf_lat - rf_lat)
            ref_builder.left_precomputed[i, :2] += 0.5 * deficit * lat
            ref_builder.right_precomputed[i, :2] -= 0.5 * deficit * lat

    # 分别对左右脚执行优化
    t0 = time.time()
    mppi_lfoot, mppi_lfoot_quats = precompute_mppi_foot(
        ref_builder.left_precomputed,        # TerrainReference 输出的左脚参考位置
        ref_builder.left_precomputed_quats,  # TerrainReference 输出的左脚参考四元数
        motion.left_contact,                  # 左脚触地掩码
        motion.foot_nominal_z,                # 脚踝标称高度
        terrain, opt_params, motion.fps,
        pelvis_quats=pelvis_quats,
        planner=args.planner,
    )
    mppi_rfoot, mppi_rfoot_quats = precompute_mppi_foot(
        ref_builder.right_precomputed,       # 右脚参考位置
        ref_builder.right_precomputed_quats, # 右脚参考四元数
        motion.right_contact,                 # 右脚触地掩码
        motion.foot_nominal_z,
        terrain, opt_params, motion.fps,
        pelvis_quats=pelvis_quats,
        planner=args.planner,
    )
    dt_opt = time.time() - t0
    print(f"  {planner_name} optimization took {dt_opt:.2f}s")

    # ===== 第 5 步：骨盆 z 坐标滤波 =====
    # 新架构: 外部构造 z_target, filter 只负责平滑
    #   support_z   = weighted warped foot height (terrain-driven)
    #   z_anchor    = support_z + nominal_root_clearance
    #   z_style     = raw pelvis z variation relative to frame 0
    #   z_target    = z_anchor + z_style, clamped by reachability
    print("\n[5] Computing pelvis z (target construction + EMA smoothing)...")
    root_z_filter = SupportAwareRootZFilter(
        alpha_up=0.5, alpha_down=0.35, max_delta=0.035,
    )
    N = motion.n_frames
    raw_pelvis = motion.body_pos[:, NPZ_PELVIS].copy()
    nominal_root_clearance = float(motion.pelvis_height_above_foot)
    raw_pelvis_z0 = float(raw_pelvis[0, 2])
    raw_style_z = raw_pelvis[:, 2] - raw_pelvis_z0
    max_reach = _MAX_REACH

    pelvis_z_out = np.zeros(N, dtype=np.float64)
    root_z_filter.reset()
    for i in range(N):
        lw, rw, is_flight = _support_state(motion, i, phase_clock_trusted)

        w_sum = lw + rw
        if w_sum > 1e-6:
            wn = np.array([lw, rw]) / w_sum
            support_z = float(wn[0] * mppi_lfoot[i, 2] + wn[1] * mppi_rfoot[i, 2])
        else:
            support_z = float(terrain.height_at(
                float(raw_pelvis[i, 0]), float(raw_pelvis[i, 1])
            ))

        z_anchor = support_z + nominal_root_clearance
        z_target = z_anchor + float(raw_style_z[i])

        z_upper_candidates = []
        if lw > 1e-3:
            z_upper_candidates.append(
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_lfoot[i], lw, max_reach)
            )
        if rw > 1e-3:
            z_upper_candidates.append(
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_rfoot[i], rw, max_reach)
            )
        if not z_upper_candidates:
            z_upper_candidates.extend([
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_lfoot[i], 0.0, max_reach),
                _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_rfoot[i], 0.0, max_reach),
            ])
        z_target = min(z_target, min(z_upper_candidates))

        pelvis_z_out[i] = root_z_filter.step(z_target, is_flight=is_flight)

    pelvis_positions = raw_pelvis.copy()
    pelvis_positions[:, 2] = pelvis_z_out
    print(f"  Pelvis z: min={pelvis_z_out.min():.3f} max={pelvis_z_out.max():.3f}")

    if args.leg_collision_iters > 0:
        print("\n[5b] Resolving lower-leg / foot collisions...")
        coll_before = _collision_frame_count(
            motion,
            pelvis_positions,
            mppi_lfoot,
            mppi_rfoot,
            mppi_lfoot_quats,
            mppi_rfoot_quats,
            shin_radius=args.shin_collision_radius,
            foot_radius=args.foot_collision_radius,
            collision_margin=args.leg_collision_margin,
        )
        mppi_lfoot, mppi_rfoot = _resolve_lower_leg_foot_collisions(
            motion,
            pelvis_positions,
            mppi_lfoot,
            mppi_rfoot,
            mppi_lfoot_quats,
            mppi_rfoot_quats,
            motion.left_contact,
            motion.right_contact,
            terrain,
            motion.foot_nominal_z,
            n_iters=args.leg_collision_iters,
            shin_radius=args.shin_collision_radius,
            foot_radius=args.foot_collision_radius,
            collision_margin=args.leg_collision_margin,
            push_gain=args.collision_push_gain,
            max_push_per_frame=args.collision_max_push,
            pelvis_side_margin=args.pelvis_side_margin,
            pelvis_side_gain=args.pelvis_side_gain,
        )
        coll_after = _collision_frame_count(
            motion,
            pelvis_positions,
            mppi_lfoot,
            mppi_rfoot,
            mppi_lfoot_quats,
            mppi_rfoot_quats,
            shin_radius=args.shin_collision_radius,
            foot_radius=args.foot_collision_radius,
            collision_margin=args.leg_collision_margin,
        )
        print(f"  Capsule collision frames: before={coll_before} after={coll_after}")

    # Remove stance-foot target drift: each stance run keeps the first grounded stance xyz.
    if not args.no_stance_latch:
        mppi_lfoot = _latch_stance_foot_targets(
            mppi_lfoot, motion.left_contact, terrain, motion.foot_nominal_z,
            ground_tol=args.stance_latch_ground_tol,
        )
        mppi_rfoot = _latch_stance_foot_targets(
            mppi_rfoot, motion.right_contact, terrain, motion.foot_nominal_z,
            ground_tol=args.stance_latch_ground_tol,
        )

    left_point_scales = _precompute_contact_point_scales(
        mppi_lfoot, mppi_lfoot_quats, motion.left_contact, terrain,
    )
    right_point_scales = _precompute_contact_point_scales(
        mppi_rfoot, mppi_rfoot_quats, motion.right_contact, terrain,
    )
    left_edge_frames = int(np.sum(np.any(left_point_scales[:, 1:] < 0.999, axis=1)))
    right_edge_frames = int(np.sum(np.any(right_point_scales[:, 1:] < 0.999, axis=1)))
    print(f"  Support-aware IK downweights: left={left_edge_frames} frames right={right_edge_frames} frames")

    # 穿透检查：计算优化后脚踝轨迹在摆动阶段的最大地形穿透深度
    for label, pos, contact in [
        ("Left", mppi_lfoot, motion.left_contact),
        ("Right", mppi_rfoot, motion.right_contact),
    ]:
        terr_z = terrain.height_batch(pos[:, 0], pos[:, 1])  # 查询每帧脚位下方的地形高度
        # 穿透量 = (地形高度 + 脚踝标称高度) - 脚踝实际 z，正值表示穿透
        pen = np.maximum(terr_z + motion.foot_nominal_z - pos[:, 2], 0)
        swing_mask = ~contact  # 只检查摆动帧 (支撑帧贴地是正常的)
        if np.any(swing_mask):
            max_pen = np.max(pen[swing_mask])
            print(f"  {label} swing max penetration: {max_pen*1000:.2f}mm")

    # ===== 第 6 步：设置 MuJoCo 渲染和 IK 求解器 =====
    print("\n[6] Setting up MuJoCo viewer...")
    model = mujoco.MjModel.from_xml_path(xml_path)  # 加载 MuJoCo 模型
    data = mujoco.MjData(model)  # 创建仿真数据对象
    joint_mapping = build_joint_mapping(model)  # 构建关节映射 (策略顺序 ↔ MuJoCo 顺序)

    # --- 初始化 IK 求解器 ---
    _ik_backend, ik_solver, root_xy_weight, root_z_weight = _build_ik_solver(
        model, args, log_config=True
    )

    # --- Ghost 幽灵机器人 (可选) ---
    # 显示 TerrainReference (MPPI 之前) 的参考姿态，半透明蓝色
    # 用于对比 MPPI 优化前后的效果
    ghost_model = ghost_data = ghost_vopt = ghost_pert = None
    if not args.no_ghost_ref:
        ghost_rgb = parse_rgb(args.ghost_color)  # 解析颜色字符串
        ghost_model = mujoco.MjModel.from_xml_path(xml_path)  # 第二个 MuJoCo 模型实例
        ghost_data = mujoco.MjData(ghost_model)
        # 将所有几何体设为半透明蓝色
        for i in range(ghost_model.ngeom):
            ghost_model.geom_rgba[i, :3] = ghost_rgb    # RGB 颜色
            ghost_model.geom_rgba[i, 3] = float(args.ghost_alpha)  # 透明度
        ghost_vopt = mujoco.MjvOption()   # 可视化选项
        ghost_pert = mujoco.MjvPerturb()  # 扰动对象 (不使用，但 API 需要)
        print(f"  Ghost enabled (alpha={args.ghost_alpha:.2f})")

    # --- 导出数据缓冲区 ---
    export_done = False
    if args.export_npz:
        # 记录轨迹数据：目标位置 + IK 求解后的实际仿真位置
        rec_pelvis = np.zeros((N, 3))      # 目标骨盆位置
        rec_lfoot = np.zeros((N, 3))       # 目标左脚位置
        rec_rfoot = np.zeros((N, 3))       # 目标右脚位置
        rec_sim_pelvis = np.zeros((N, 3))  # IK 后实际骨盆位置
        rec_sim_lfoot = np.zeros((N, 3))   # IK 后实际左脚位置
        rec_sim_rfoot = np.zeros((N, 3))   # IK 后实际右脚位置

    motion_export_done = False
    body_mapping = None
    if args.export_motion_npz:
        # 导出完整动作 npz (可供后续重放或作为新动捕数据使用)
        body_mapping = build_body_mapping(model)  # npz刚体索引→MuJoCo刚体ID
        nbody_export = len(NPZ_BODY_NAMES)
        rec_m_joint_pos = np.zeros((N, model.nq - 7))       # 关节角记录
        rec_m_body_pos = np.zeros((N, nbody_export, 3))      # 刚体位置记录
        rec_m_body_quat = np.zeros((N, nbody_export, 4))     # 刚体四元数记录

    # --- Trio export: raw / raw+offset / mppi ---
    trio_export_done = False
    if args.export_trio_npz:
        if body_mapping is None:
            body_mapping = build_body_mapping(model)
        nbody_export = len(NPZ_BODY_NAMES)
        n_joints = model.nq - 7  # 29 joints for G1
        # raw: original motion data (already in policy order in MotionClip)
        # — no per-frame recording needed, read directly from motion.*
        # offset: ghost robot (raw joints + terrain-adjusted pelvis z)
        trio_offset_joint_pos = np.zeros((N, n_joints), dtype=np.float64)
        trio_offset_body_pos = np.zeros((N, nbody_export, 3), dtype=np.float64)
        trio_offset_body_quat = np.zeros((N, nbody_export, 4), dtype=np.float64)
        # mppi: main robot (IK-solved)
        trio_mppi_joint_pos = np.zeros((N, n_joints), dtype=np.float64)
        trio_mppi_body_pos = np.zeros((N, nbody_export, 3), dtype=np.float64)
        trio_mppi_body_quat = np.zeros((N, nbody_export, 4), dtype=np.float64)
        print(f"  Trio export enabled: {args.export_trio_npz}_{{raw,offset,mppi}}.npz")
        # Dedicated FK model+data for offset recording when ghost is disabled.
        if ghost_data is None:
            _trio_fk_model = mujoco.MjModel.from_xml_path(xml_path)
            _trio_fk_data = mujoco.MjData(_trio_fk_model)
        else:
            _trio_fk_model = ghost_model
            _trio_fk_data = ghost_data

    # ===== 第 7 步：主渲染循环 =====
    # 管线: MotionClip → TerrainRef → MPPI → RootZFilter → IK → MuJoCo
    print("\n" + "=" * 60)
    print(f"PIPELINE: MotionClip → TerrainRef → {planner_name} → RootZFilter → IK → MuJoCo")
    print("=" * 60)

    frame = 0                # 全局帧计数器 (可超过 N，循环播放)
    prev_leg_qpos = None     # 上一帧的腿部关节角 (用于 IK 热启动)
    prev_root_pos = None
    prev_shaped_pelvis = None
    prev_shaped_lfoot = None
    prev_shaped_rfoot = None
    # Cache: store computed qpos from first pass for smooth replay
    _qpos_cache = np.zeros((N, model.nq), dtype=np.float64)
    _first_pass_done = False
    ik_pos_err_hist = []     # 记录每帧的 IK 位置误差
    ik_rot_err_hist = []     # 记录每帧的 IK 朝向误差
    viewer_lb_stab_state = LowerBodyStabilizerState()
    viewer_lb_active_until = -1

    # ===== Window IK: pre-compute all frames before viewer starts =====
    if getattr(args, "ik_mode", "frame") == "window":
        if _ik_backend != "jacobian":
            raise ValueError("window IK currently supports only --ik_backend jacobian")

        print(f"\n[Window IK] Pre-computing {N} frames (window={args.ik_window}, commit={args.ik_commit})...")

        _frame_targets, _locked_double_stance = _build_frame_ik_targets(
            motion,
            pelvis_positions,
            mppi_lfoot,
            mppi_rfoot,
            mppi_lfoot_quats,
            mppi_rfoot_quats,
            joint_mapping,
            ik_solver,
            left_point_scales,
            right_point_scales,
            root_xy_weight,
            root_z_weight,
            phase_clock_trusted,
        )
        _solved_states, _ = _solve_ik_state_sequence(
            args,
            model,
            data,
            ik_solver,
            _frame_targets,
            _locked_double_stance,
            motion.left_contact[:N],
            motion.right_contact[:N],
            _ik_backend,
            init_progress_prefix="    [init]",
            init_progress_every=100,
            lb_progress_prefix="    [lb-stab]",
            lb_progress_every=100,
        )

        # Step 3: Write solved states into qpos_cache
        for _t in range(N):
            _apply_state_to_model(data, ik_solver, _frame_targets[_t], _solved_states[_t])
            _qpos_cache[_t] = data.qpos.copy()

        _first_pass_done = True
        print(f"  [Window IK] Done. All {N} frames pre-computed.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置初始相机位置
        viewer.cam.lookat[:] = [pelvis_positions[0, 0] + 0.5, 0.0, 0.5]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        cam_follow_xy = pelvis_positions[0, :2].astype(np.float64).copy()
        cam_follow_deadband = 0.02
        cam_follow_alpha_idle = 0.08
        cam_follow_alpha_move = 0.22

        while viewer.is_running():
            frame_start = time.time()
            t = frame % N  # 循环播放：帧索引对总帧数取模

            # 循环重置时清除所有帧间状态
            if t == 0 and frame > 0:
                _first_pass_done = True  # Mark first pass complete
                prev_shaped_pelvis = None
                prev_shaped_lfoot = None
                prev_shaped_rfoot = None
                prev_leg_qpos = None
                prev_root_pos = None
                viewer_lb_stab_state = LowerBodyStabilizerState()
                viewer_lb_active_until = -1

            # After first pass: replay from cache at constant fps (no recomputation)
            if _first_pass_done:
                data.qpos[:] = _qpos_cache[t]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                target_xy = np.asarray(data.qpos[:2], dtype=np.float64)
                delta_xy = target_xy - cam_follow_xy
                dist_xy = float(np.linalg.norm(delta_xy))
                if dist_xy > cam_follow_deadband:
                    alpha = cam_follow_alpha_move if dist_xy > 0.10 else cam_follow_alpha_idle
                    cam_follow_xy = cam_follow_xy + alpha * delta_xy
                viewer.cam.lookat[0] = cam_follow_xy[0]
                viewer.cam.lookat[1] = cam_follow_xy[1]
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if ghost_model is not None:
                        mujoco.mjv_addGeoms(
                            ghost_model, ghost_data, ghost_vopt, ghost_pert,
                            mujoco.mjtCatBit.mjCAT_ALL.value, viewer.user_scn)
                viewer.sync()
                frame += 1
                dt = (1.0 / motion.fps) / args.speed
                elapsed = time.time() - frame_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            # --- Rolling MPPI recompute (every rolling_interval frames) ---
            _rolling_interval = getattr(args, "rolling_interval", 0)
            if _rolling_interval > 0 and t > 0 and t % _rolling_interval == 0:
                # Recompute MPPI from frame t onward (only future swing phases)
                _horizon = min(N - t, _rolling_interval * 3)  # look 3x ahead
                _sl = slice(t, t + _horizon)
                _lref = ref_builder.left_precomputed[_sl].copy()
                _rref = ref_builder.right_precomputed[_sl].copy()
                _lq = ref_builder.left_precomputed_quats[_sl].copy()
                _rq = ref_builder.right_precomputed_quats[_sl].copy()
                _lc = motion.left_contact[_sl]
                _rc = motion.right_contact[_sl]
                _pq = motion.body_quat[_sl, NPZ_PELVIS]
                _lout, _lqout = precompute_mppi_foot(
                    _lref, _lq, _lc, motion.foot_nominal_z,
                    terrain, opt_params, motion.fps, pelvis_quats=_pq, planner=args.planner,
                )
                _rout, _rqout = precompute_mppi_foot(
                    _rref, _rq, _rc, motion.foot_nominal_z,
                    terrain, opt_params, motion.fps, pelvis_quats=_pq, planner=args.planner,
                )
                mppi_lfoot[_sl] = _lout
                mppi_rfoot[_sl] = _rout
                mppi_lfoot_quats[_sl] = _lqout
                mppi_rfoot_quats[_sl] = _rqout

            base_target, locked_double_stance, _, _ = _build_frame_ik_target_at(
                motion,
                pelvis_positions,
                mppi_lfoot,
                mppi_rfoot,
                mppi_lfoot_quats,
                mppi_rfoot_quats,
                joint_mapping,
                ik_solver,
                left_point_scales,
                right_point_scales,
                root_xy_weight,
                root_z_weight,
                phase_clock_trusted,
                t,
                prev_shaped_pelvis=prev_shaped_pelvis,
                prev_shaped_lfoot=prev_shaped_lfoot,
                prev_shaped_rfoot=prev_shaped_rfoot,
            )
            prev_shaped_pelvis = base_target.root_pos.copy()
            prev_shaped_lfoot = base_target.left_pos.copy()
            prev_shaped_rfoot = base_target.right_pos.copy()

            ik_result, base_state, _ = _solve_single_frame_ik(
                data,
                ik_solver,
                base_target,
                prev_leg_qpos=prev_leg_qpos,
                prev_root_pos=prev_root_pos,
                locked_double_stance=locked_double_stance,
            )

            # 记录 IK 误差
            ik_pos_err = max(ik_result.left_pos_err, ik_result.right_pos_err)
            ik_rot_err = max(ik_result.left_rot_err, ik_result.right_rot_err)

            if _lower_body_stabilizer_enabled(args, _ik_backend):
                base_state, ik_pos_err, viewer_lb_active_until = _maybe_stabilize_live_frame(
                    data,
                    ik_solver,
                    base_target,
                    base_state,
                    motion,
                    t,
                    ik_pos_err,
                    viewer_lb_stab_state,
                    viewer_lb_active_until,
                    args,
                )
            else:
                _apply_state_to_model(data, ik_solver, base_target, base_state)

            final_qpos = data.qpos.copy()
            final_leg_qpos = ik_solver.extract_leg_qpos(data.qpos)
            final_root_pos = data.qpos[:3].copy()

            # 大误差暂停：IK ERROR > 500mm 时暂停程序，方便观察
            if ik_pos_err * 1000 > 500:
                print(f"\n{'!'*60}")
                print(f"IK ERROR > 500mm at frame t={t}")
                print(f"  left_pos_err  = {ik_result.left_pos_err*1000:.1f}mm")
                print(f"  right_pos_err = {ik_result.right_pos_err*1000:.1f}mm")
                print(f"  left_rot_err  = {ik_result.left_rot_err*1000:.1f}mm")
                print(f"  right_rot_err = {ik_result.right_rot_err*1000:.1f}mm")
                print(f"  root_offset   = {ik_result.root_pos_offset}")
                print(f"  pelvis target = {base_target.root_pos}")
                print(f"  lfoot target  = {base_target.left_pos}")
                print(f"  rfoot target  = {base_target.right_pos}")
                print(f"  pelvis actual = {data.qpos[:3]}")
                print(f"  lfoot actual  = {data.xpos[ik_solver.left_foot_body_id]}")
                print(f"  rfoot actual  = {data.xpos[ik_solver.right_foot_body_id]}")
                print(f"  leg_qpos      = {ik_result.leg_qpos}")
                if prev_leg_qpos is not None:
                    print(f"  prev_leg_qpos = {prev_leg_qpos}")
                    print(f"  max_dq        = {np.max(np.abs(ik_result.leg_qpos - prev_leg_qpos)):.3f}rad")
                print(f"{'!'*60}")
                # input("Press Enter to continue...")

            # 始终使用 IK 结果
            data.qpos[:] = final_qpos
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            prev_root_pos = final_root_pos.copy()

            # --- 关节跳变检测诊断 ---
            if prev_leg_qpos is not None:
                dq = final_leg_qpos - prev_leg_qpos  # 帧间关节角变化
                max_dq = float(np.max(np.abs(dq)))
                if max_dq > 0.3:  # 超过 0.3 rad ≈ 17° 视为跳变
                    # 计算骨盆到各脚目标的 3D 距离 (用于诊断是否超过腿部可达距离)
                    pelvis_3d = final_root_pos
                    d_lf = np.linalg.norm(base_target.left_pos - pelvis_3d)
                    d_rf = np.linalg.norm(base_target.right_pos - pelvis_3d)
                    lc = "stance" if motion.left_contact[t] else "SWING"
                    rc = "stance" if motion.right_contact[t] else "SWING"
                    ms = " MS" if getattr(ik_result, "used_multiseed", False) else ""
                    print(f"  [JUMP] f{t}: dq={max_dq:.2f} perr={ik_pos_err:.3f} "
                          f"d_lf={d_lf:.3f} d_rf={d_rf:.3f} "
                          f"L:{lc} R:{rc} "
                          f"pelvis_z={pelvis_3d[2]:.3f} "
                          f"lf_z={base_target.left_pos[2]:.3f} rf_z={base_target.right_pos[2]:.3f}{ms}")

            # 始终从上一帧的解热启动，这是防止帧间关节跳变的最关键因素
            # 相邻帧的构型非常相似，上一帧的解是最好的初始种子
            prev_leg_qpos = final_leg_qpos.copy()

            # Cache qpos for smooth replay on subsequent loops
            if not _first_pass_done:
                _qpos_cache[t] = data.qpos.copy()

            ik_pos_err_hist.append(ik_pos_err)
            ik_rot_err_hist.append(ik_rot_err)

            # --- 导出轨迹数据 (可选) ---
            if args.export_npz and not export_done:
                rec_pelvis[t] = base_target.root_pos       # 记录目标骨盆位置
                rec_lfoot[t] = base_target.left_pos         # 记录目标左脚位置
                rec_rfoot[t] = base_target.right_pos         # 记录目标右脚位置
                rec_sim_pelvis[t] = data.qpos[:3]  # IK 后实际骨盆位置 (qpos 前 3 维)
                rec_sim_lfoot[t] = data.xpos[ik_solver.left_foot_body_id]   # IK 后左脚实际位置
                rec_sim_rfoot[t] = data.xpos[ik_solver.right_foot_body_id]  # IK 后右脚实际位置
                if t == N - 1:  # 第一轮播放结束时保存
                    np.savez_compressed(
                        args.export_npz,
                        pelvis=rec_pelvis, lfoot=rec_lfoot, rfoot=rec_rfoot,
                        sim_pelvis=rec_sim_pelvis,
                        sim_lfoot=rec_sim_lfoot, sim_rfoot=rec_sim_rfoot,
                        left_contact=motion.left_contact,
                        right_contact=motion.right_contact,
                    )
                    export_done = True
                    print(f"\n[Export] Saved {N} frames to {args.export_npz}")

            # --- 导出完整动作 npz (可选) ---
            if args.export_motion_npz and not motion_export_done:
                # 将 MuJoCo 关节角重排回策略顺序 (npz 格式)
                rec_m_joint_pos[t] = reorder_from_mujoco(data.qpos[7:], joint_mapping)
                # 记录所有刚体的位置和四元数 (按 npz 刚体索引排列)
                for npz_i, mj_bid in enumerate(body_mapping):
                    rec_m_body_pos[t, npz_i] = data.xpos[mj_bid]
                    rec_m_body_quat[t, npz_i] = data.xquat[mj_bid]
                if t == N - 1:  # 第一轮播放结束时保存
                    _eff_cutoff = float(args.export_lowpass_cutoff)
                    if args.hardware_bandwidth_hz > 0:
                        _hw = 0.8 * args.hardware_bandwidth_hz
                        _eff_cutoff = _hw if _eff_cutoff <= 0 else min(_eff_cutoff, _hw)
                    fps_f = float(motion.fps)
                    if _eff_cutoff > 0:
                        (rec_m_joint_pos, rec_m_body_pos, rec_m_body_quat,
                         m_joint_vel, m_body_lin_vel, m_body_ang_vel) = (
                            filter_lower_body_motion_for_export(
                                xml_path,
                                rec_m_joint_pos, rec_m_body_pos, rec_m_body_quat,
                                fps_f, cutoff_hz=_eff_cutoff,
                                order=args.export_lowpass_order,
                            )
                        )
                    else:
                        m_joint_vel, m_body_lin_vel, m_body_ang_vel = (
                            _compute_velocities(rec_m_joint_pos, rec_m_body_pos,
                                                rec_m_body_quat, fps_f)
                        )
                    np.savez_compressed(
                        args.export_motion_npz,
                        fps=np.array([motion.fps], dtype=np.float32),
                        joint_pos=rec_m_joint_pos.astype(np.float32),
                        joint_vel=m_joint_vel.astype(np.float32),
                        body_pos_w=rec_m_body_pos.astype(np.float32),
                        body_quat_w=rec_m_body_quat.astype(np.float32),
                        body_lin_vel_w=m_body_lin_vel.astype(np.float32),
                        body_ang_vel_w=m_body_ang_vel.astype(np.float32),
                    )
                    motion_export_done = True
                    print(f"\n[MotionExport] Saved {N} frames to {args.export_motion_npz}")

            # --- Ghost 幽灵渲染：显示 MPPI 之前的 TerrainReference 参考姿态 ---
            if ghost_data is not None:
                update_robot_pose(
                    ghost_data,
                    root_pos=pelvis_positions[t],    # 使用相同的骨盆位置
                    root_quat=base_target.root_quat,          # 使用相同的骨盆朝向
                    joint_qpos=base_target.fixed_upper_body_qpos,        # 使用原始动捕关节角 (无 IK)
                )
                mujoco.mj_forward(ghost_model, ghost_data)  # 更新幽灵的前向运动学

            # --- Trio export: record offset (ghost) and mppi (main robot) ---
            if args.export_trio_npz and not trio_export_done:
                # offset: ghost robot = raw joints + terrain-adjusted pelvis z
                # If ghost_data was not created for display, do FK on dedicated data
                if ghost_data is None:
                    update_robot_pose(
                        _trio_fk_data,
                        root_pos=pelvis_positions[t],
                        root_quat=base_target.root_quat,
                        joint_qpos=base_target.fixed_upper_body_qpos,
                    )
                    mujoco.mj_forward(_trio_fk_model, _trio_fk_data)
                    fk_src = _trio_fk_data
                else:
                    fk_src = ghost_data  # already FK'd above
                trio_offset_joint_pos[t] = reorder_from_mujoco(
                    fk_src.qpos[7:], joint_mapping
                )
                for npz_i, mj_bid in enumerate(body_mapping):
                    trio_offset_body_pos[t, npz_i] = fk_src.xpos[mj_bid]
                    trio_offset_body_quat[t, npz_i] = fk_src.xquat[mj_bid]

                # mppi: main robot after IK solve
                trio_mppi_joint_pos[t] = reorder_from_mujoco(
                    data.qpos[7:], joint_mapping
                )
                for npz_i, mj_bid in enumerate(body_mapping):
                    trio_mppi_body_pos[t, npz_i] = data.xpos[mj_bid]
                    trio_mppi_body_quat[t, npz_i] = data.xquat[mj_bid]

                if t == N - 1:
                    _eff_cutoff = float(args.export_lowpass_cutoff)
                    if args.hardware_bandwidth_hz > 0:
                        _hw = 0.8 * args.hardware_bandwidth_hz
                        _eff_cutoff = _hw if _eff_cutoff <= 0 else min(_eff_cutoff, _hw)
                    _save_trio_npz(
                        args.export_trio_npz, motion, joint_mapping, body_mapping,
                        trio_offset_joint_pos, trio_offset_body_pos, trio_offset_body_quat,
                        trio_mppi_joint_pos, trio_mppi_body_pos, trio_mppi_body_quat,
                        export_lowpass_cutoff=_eff_cutoff,
                        export_lowpass_order=args.export_lowpass_order,
                    )
                    trio_export_done = True

            # --- 定期日志输出 ---
            if args.log_every > 0 and frame % args.log_every == 0:
                # L/R 表示左右脚触地状态 (L=左触地, .=左摆动)
                contact_str = (
                    ("L" if motion.left_contact[t] else ".") +
                    ("R" if motion.right_contact[t] else ".")
                )
                # 根节点位移偏移量 (IK 求解器可能微调根位置)
                root_off = ik_result.root_pos_offset
                root_off_str = (
                    f"root_off=({root_off[0]*1000:.1f},{root_off[1]*1000:.1f},{root_off[2]*1000:.1f})mm"
                    if root_off is not None else ""
                )
                ms_str = (
                    f" seed={ik_result.seed_index} cost={ik_result.cost:.4g}"
                    if getattr(ik_result, "used_multiseed", False) else ""
                )
                print(
                    f"[{frame:05d}] t={t:03d} {contact_str} "
                    f"pelvis=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) "
                    f"IK pos_err={ik_pos_err*1000:.1f}mm rot_err={ik_rot_err:.3f} "
                    f"{root_off_str}{ms_str}"
                )

            # --- 可视化：轨迹线 + 目标标记 ---
            # 构建未来 40 帧的预览窗口用于绘制轨迹线
            horizon = min(40, N - t)
            trail_indices = np.arange(t, t + horizon)
            trail_lfoot = mppi_lfoot[trail_indices]       # MPPI 优化后的左脚轨迹
            trail_rfoot = mppi_rfoot[trail_indices]       # MPPI 优化后的右脚轨迹
            trail_pelvis = pelvis_positions[trail_indices]  # 骨盆轨迹

            # TerrainReference (MPPI 之前) 的参考轨迹，用于对比
            ref_lfoot = ref_builder.left_precomputed[trail_indices]
            ref_rfoot = ref_builder.right_precomputed[trail_indices]

            # 相机跟随骨盆 xy 位置
            target_xy = np.asarray(data.qpos[:2], dtype=np.float64)
            delta_xy = target_xy - cam_follow_xy
            dist_xy = float(np.linalg.norm(delta_xy))
            if dist_xy > cam_follow_deadband:
                alpha = cam_follow_alpha_move if dist_xy > 0.10 else cam_follow_alpha_idle
                cam_follow_xy = cam_follow_xy + alpha * delta_xy
            viewer.cam.lookat[0] = cam_follow_xy[0]
            viewer.cam.lookat[1] = cam_follow_xy[1]

            with viewer.lock():
                viewer.user_scn.ngeom = 0  # 清除上一帧的自定义几何体

                # 渲染幽灵机器人的几何体
                if ghost_model is not None:
                    mujoco.mjv_addGeoms(
                        ghost_model, ghost_data, ghost_vopt, ghost_pert,
                        mujoco.mjtCatBit.mjCAT_ALL.value, viewer.user_scn,
                    )

                if not args.no_debug_overlay:
                    # TerrainReference 参考轨迹线 (绿色=左脚, 红色=右脚, 半透明)
                    add_trail(viewer.user_scn, ref_lfoot, [0.0, 0.9, 0.0, 0.35], size=0.008, stride=2)
                    add_trail(viewer.user_scn, ref_rfoot, [0.9, 0.0, 0.0, 0.35], size=0.008, stride=2)

                    # MPPI 优化后轨迹线 (蓝色, 不透明, 更粗)
                    add_trail(viewer.user_scn, trail_lfoot, [0.3, 0.5, 1.0, 0.80], size=0.012, stride=2)
                    add_trail(viewer.user_scn, trail_rfoot, [0.0, 0.2, 0.8, 0.80], size=0.012, stride=2)
                    # 骨盆轨迹线 (青色, 半透明)
                    add_trail(viewer.user_scn, trail_pelvis, [0.0, 0.8, 1.0, 0.45], size=0.008, stride=2)

                    # 当前帧目标位置标记
                    add_current_markers(
                        viewer.user_scn,
                        base_target.root_pos,
                        base_target.left_pos,
                        base_target.right_pos,
                    )

            viewer.sync()  # 同步渲染
            frame += 1

            # 帧率控制：按动捕帧率回放，speed 倍率可调
            dt = (1.0 / motion.fps) / args.speed
            elapsed = time.time() - frame_start
            if elapsed < dt:
                time.sleep(dt - elapsed)  # 如果处理太快则等待

    # --- 结束后打印 IK 误差统计 ---
    if ik_pos_err_hist:
        pos_mm = 1000.0 * np.asarray(ik_pos_err_hist)  # 转换为毫米
        rot = np.asarray(ik_rot_err_hist)
        print("\nIK summary:")
        print(f"  position error: p95={np.percentile(pos_mm, 95):.2f}mm, max={np.max(pos_mm):.2f}mm")
        print(f"  orientation error: p95={np.percentile(rot, 95):.4f}rad, max={np.max(rot):.4f}rad")


if __name__ == "__main__":
    main()
