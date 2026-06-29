from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms

from pmt_tasks.mdp.commands import MotionCommand, MultiMotionCommand
from pmt_tasks.mdp.commands import MultiMotionCommandV2
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def tracking_obs(env: ManagerBasedEnv, command_name: str, obs_type: str) -> torch.Tensor:
    """Parametrized motion-tracking observation term (§9b consolidation).

    Collapses the four near-identical tracking terms that differ only in the
    frame-pair transformed:

    - ``"motion_anchor_pos"`` / ``"motion_anchor_ori"``: reference (motion) anchor
      expressed in the robot anchor frame.
    - ``"robot_body_pos"`` / ``"robot_body_ori"``: robot bodies expressed in the
      robot anchor frame.

    The ``*_pos`` variants return the position vector flattened; the ``*_ori``
    variants return the first two columns of the rotation matrix (tan-norm style).
    The four original functions below are thin wrappers, so existing env cfgs keep
    importing them unchanged. Behavior-preserving.
    """
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)

    if obs_type in ("motion_anchor_pos", "motion_anchor_ori"):
        pos, ori = subtract_frame_transforms(
            command.robot_anchor_pos_w,
            command.robot_anchor_quat_w,
            command.anchor_pos_w,
            command.anchor_quat_w,
        )
        if obs_type == "motion_anchor_pos":
            return pos.view(env.num_envs, -1)
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(mat.shape[0], -1)

    if obs_type in ("robot_body_pos", "robot_body_ori"):
        num_bodies = len(command.cfg.body_names)
        pos_b, ori_b = subtract_frame_transforms(
            command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
            command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
            command.robot_body_pos_w,
            command.robot_body_quat_w,
        )
        if obs_type == "robot_body_pos":
            return pos_b.view(env.num_envs, -1)
        mat = matrix_from_quat(ori_b)
        return mat[..., :2].reshape(mat.shape[0], -1)

    raise ValueError(
        f"Unknown obs_type '{obs_type}' (expected one of "
        "'motion_anchor_pos'|'motion_anchor_ori'|'robot_body_pos'|'robot_body_ori')."
    )


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Thin wrapper over :func:`tracking_obs` (obs_type='robot_body_pos')."""
    return tracking_obs(env, command_name, "robot_body_pos")


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Thin wrapper over :func:`tracking_obs` (obs_type='robot_body_ori')."""
    return tracking_obs(env, command_name, "robot_body_ori")


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Thin wrapper over :func:`tracking_obs` (obs_type='motion_anchor_pos')."""
    return tracking_obs(env, command_name, "motion_anchor_pos")


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Thin wrapper over :func:`tracking_obs` (obs_type='motion_anchor_ori')."""
    return tracking_obs(env, command_name, "motion_anchor_ori")


def raw_motion_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "raw_joint_pos") or not hasattr(command, "raw_joint_vel"):
        raise TypeError(
            f"Command '{command_name}' ({type(command)}) does not have raw_joint_pos/raw_joint_vel."
        )
    return torch.cat([command.raw_joint_pos, command.raw_joint_vel], dim=1)


def _joint_command_window(env: ManagerBasedEnv, command_name: str, *, source: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "get_joint_command_window"):
        raise TypeError(
            f"Command '{command_name}' ({type(command)}) does not expose get_joint_command_window()."
        )
    return command.get_joint_command_window(source=source, flatten=True)


def minco_adapted_joint_command_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Sampled past/current/future adapted MINCO joint window [N, W*(q+qd)]."""
    return _joint_command_window(env, command_name, source="adapted")


def minco_raw_joint_command_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Sampled past/current/future raw joint window [N, W*(q+qd)]."""
    return _joint_command_window(env, command_name, source="raw")


def mppi_adapted_joint_command_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Sampled past/current/future adapted MPPI joint window [N, W*(q+qd)]."""
    return _joint_command_window(env, command_name, source="adapted")


def mppi_raw_joint_command_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Sampled past/current/future raw MPPI joint window [N, W*(q+qd)]."""
    return _joint_command_window(env, command_name, source="raw")


# =============================================================================
# ADD Observations
# =============================================================================


def _matrix_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) to tangent-normal 6D representation."""
    mat = matrix_from_quat(q)
    tan = mat[..., :, 0]
    norm = mat[..., :, 2]
    return torch.cat([tan, norm], dim=-1)


_ADD_G1_DEFAULT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_ADD_G1_JOINT_AXIS_SUFFIXES = (
    ("_pitch_joint", (0.0, 1.0, 0.0)),
    ("_knee_joint", (0.0, 1.0, 0.0)),
    ("_elbow_joint", (0.0, 1.0, 0.0)),
    ("_roll_joint", (1.0, 0.0, 0.0)),
    ("_yaw_joint", (0.0, 0.0, 1.0)),
)

_ADD_AGENT_JOINT_QUAT_ATTRS = (
    "robot_joint_quat",
    "robot_joint_quat_w",
    "robot_joint_rot",
    "robot_joint_rot_w",
)
_ADD_DEMO_JOINT_QUAT_ATTRS = ("joint_quat", "joint_quat_w", "joint_rot", "joint_rot_w")

# Number of MimicKit ADD key bodies (left/right ankle, head, left/right wrist). The
# command's body_names is [anchor_body, *key_bodies], so when more than this many body
# columns are present the leading anchor column is sliced out of the disc body-pos block.
_ADD_NUM_KEY_BODIES = 4


def _get_multimotion_command_v2(env: ManagerBasedEnv, command_name: str) -> MultiMotionCommandV2:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiMotionCommandV2):
        raise TypeError(
            f"ADD discriminator observations require MultiMotionCommandV2 under command '{command_name}'. "
            f"Received: {type(command)}"
        )
    return command


def _get_add_joint_names(command: MultiMotionCommandV2, num_joints: int) -> tuple[str, ...]:
    for source in (getattr(command, "robot", None), getattr(getattr(command, "robot", None), "data", None), command):
        joint_names = getattr(source, "joint_names", None)
        if joint_names is not None and len(joint_names) == num_joints:
            return tuple(joint_names)

    if num_joints == len(_ADD_G1_DEFAULT_JOINT_NAMES):
        return _ADD_G1_DEFAULT_JOINT_NAMES

    raise RuntimeError(
        f"Cannot derive ADD joint rotations for {num_joints} joints without command joint quaternions or names."
    )


def _add_joint_axis_from_name(joint_name: str) -> tuple[float, float, float]:
    for suffix, axis in _ADD_G1_JOINT_AXIS_SUFFIXES:
        if joint_name.endswith(suffix):
            return axis
    raise RuntimeError(f"Cannot infer ADD joint axis for joint '{joint_name}'.")


def _add_joint_axes(command: MultiMotionCommandV2, joint_pos: torch.Tensor) -> torch.Tensor:
    joint_names = _get_add_joint_names(command, joint_pos.shape[-1])
    axes = [_add_joint_axis_from_name(joint_name) for joint_name in joint_names]
    return torch.tensor(axes, device=joint_pos.device, dtype=joint_pos.dtype)


def _axis_angle_to_quat(angles: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    half_angle = 0.5 * angles.unsqueeze(-1)
    quat_w = torch.cos(half_angle)
    quat_xyz = axes.unsqueeze(0) * torch.sin(half_angle)
    return torch.cat([quat_w, quat_xyz], dim=-1)


def _get_add_joint_quat(
    command: MultiMotionCommandV2,
    joint_pos: torch.Tensor,
    quat_attr_names: tuple[str, ...],
) -> torch.Tensor:
    for attr_name in quat_attr_names:
        joint_quat = getattr(command, attr_name, None)
        if (
            isinstance(joint_quat, torch.Tensor)
            and joint_quat.ndim == 3
            and joint_quat.shape[:2] == joint_pos.shape[:2]
            and joint_quat.shape[-1] == 4
        ):
            return joint_quat

    return _axis_angle_to_quat(joint_pos, _add_joint_axes(command, joint_pos))


def _add_joint_rot_tan_norm(
    command: MultiMotionCommandV2,
    joint_pos: torch.Tensor,
    quat_attr_names: tuple[str, ...],
) -> torch.Tensor:
    joint_quat = _get_add_joint_quat(command, joint_pos, quat_attr_names)
    return _matrix_to_tan_norm(joint_quat).reshape(joint_pos.shape[0], -1)


def _add_disc_obs(
    env: ManagerBasedEnv,
    command: MultiMotionCommandV2,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    body_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_quat_attrs: tuple[str, ...],
) -> torch.Tensor:
    root_rot_tn = _matrix_to_tan_norm(root_quat)
    joint_rot_tn = _add_joint_rot_tan_norm(command, joint_pos, joint_quat_attrs)
    # body_pos columns follow the command's body_names order: index 0 is the anchor
    # body (torso_link), indices 1: are the 4 ADD key bodies (feet + hands; the G1 has
    # no head_link). Drop the anchor row so the body-pos block is exactly the 4 key
    # bodies relative to root (4*3 = 12D).
    key_body_pos = body_pos[:, 1:] if body_pos.shape[1] > _ADD_NUM_KEY_BODIES else body_pos
    body_pos_rel = (key_body_pos - root_pos.unsqueeze(1)).reshape(env.num_envs, -1)
    return torch.cat(
        [root_pos, root_rot_tn, joint_rot_tn, body_pos_rel, root_lin_vel, root_ang_vel, joint_vel],
        dim=-1,
    )


def add_disc_obs_agent(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Discriminator observations for the current robot state (global frame).

    MimicKit global_obs=True layout for G1 ADD:
        root_pos(3) + root_rot_tn(6) + joint_rot_tn(29*6) + body_pos_rel(4*3)
        + root_lin_vel(3) + root_ang_vel(3) + joint_vel(29) = 230 dims.
    """
    command = _get_multimotion_command_v2(env, command_name)
    return _add_disc_obs(
        env,
        command,
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.robot_joint_pos,
        command.robot_body_pos_w,
        command.robot_anchor_lin_vel_w,
        command.robot_anchor_ang_vel_w,
        command.robot_joint_vel,
        _ADD_AGENT_JOINT_QUAT_ATTRS,
    )


def add_disc_obs_demo(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Discriminator observations for the reference motion state (global frame).

    Same 230D layout as :func:`add_disc_obs_agent`.
    """
    command = _get_multimotion_command_v2(env, command_name)
    return _add_disc_obs(
        env,
        command,
        command.anchor_pos_w,
        command.anchor_quat_w,
        command.joint_pos,
        command.body_pos_w,
        command.anchor_lin_vel_w,
        command.anchor_ang_vel_w,
        command.joint_vel,
        _ADD_DEMO_JOINT_QUAT_ATTRS,
    )


# =============================================================================
# Transformer Observations
# =============================================================================

def projected_gravity_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projected gravity in the robot root frame.

    Shape: [num_envs, 3]
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def proprio(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str | None = None,
) -> torch.Tensor:
    """proprioception vector (paper: o_t in R^93).

    Order: [g_proj(3), base_ang_vel(3), joint_pos_rel(29), joint_vel_rel(29), last_action(29)]
    Shape: [num_envs, 93]
    """
    asset = env.scene[asset_cfg.name]

    g_proj = asset.data.projected_gravity_b
    omega = asset.data.root_ang_vel_b
    q_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dq_rel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]

    if action_name is None:
        a_prev = env.action_manager.action
    else:
        a_prev = env.action_manager.get_term(action_name).raw_actions

    return torch.cat([g_proj, omega, q_rel, dq_rel, a_prev], dim=-1)


def command_window(
    env: ManagerBasedEnv,
    command_name: str,
    half_window: int = 10,
    stride: int = 1,
    flatten: bool = False,
) -> torch.Tensor:
    """Centered reference command window (paper: g_{t-L:t+L}).

    Each step in the window is a 38D vector:
      [v_ref_b(3), w_ref_b(3), g_ref_b(3), q_ref(29)]

    Returns:
      - If flatten=False: [num_envs, 2*half_window+1, 38]
      - If flatten=True : [num_envs, (2*half_window+1)*38]
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiMotionCommandV2):
        raise TypeError(
            f"command_window expects MultiMotionCommandV2 under command '{command_name}'. "
            f"Received: {type(command)}"
        )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return command.get_command_window(env_ids, half_window=half_window, stride=stride, flatten=flatten)


def reference_base_height(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference base height used as critic privilege."""
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiMotionCommandV2):
        raise TypeError(
            f"reference_base_height expects MultiMotionCommandV2 for command '{command_name}'. "
            f"Received: {type(command)}"
        )

    body_names = list(getattr(command.cfg, "body_names", []) or [])
    body_index = body_names.index("pelvis") if "pelvis" in body_names else 0
    return command.body_pos_w[:, body_index, 2:3]


def motion_anchor_delta_window(
    env: ManagerBasedEnv,
    command_name: str,
    half_window: int = 10,
    stride: int = 1,
    flatten: bool = False,
) -> torch.Tensor:
    """Centered motion-anchor displacement window in the current motion-anchor frame.

    Each step is a 3D vector:
      delta_p_ref = R_anchor(t)^-1 * (p_anchor(t+k) - p_anchor(t))

    Returns:
      - If flatten=False: [num_envs, 2*half_window+1, 3]
      - If flatten=True : [num_envs, (2*half_window+1)*3]
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiMotionCommandV2):
        raise TypeError(
            f"motion_anchor_delta_window expects MultiMotionCommandV2 under command '{command_name}'. "
            f"Received: {type(command)}"
        )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return command.get_motion_anchor_delta_window(
        env_ids,
        half_window=half_window,
        stride=stride,
        flatten=flatten,
    )


def foot_traj_delta_target(
    env: ManagerBasedEnv,
    teacher_command_name: str,
    student_command_name: str,
    body_names: list[str] | tuple[str, ...],
    window_size: int = 5,
    stride: int = 1,
    flatten: bool = True,
) -> torch.Tensor:
    """Future foot-trajectory residual between teacher and student transformer commands.

    The target is expressed in each command's anchor frame to stay invariant to
    global placement. For each future step, we gather the requested body
    positions, transform them into the command anchor frame, and then compute:

      teacher_foot_pos_anchor - student_foot_pos_anchor

    Returns:
      - If flatten=True : [num_envs, window_size * len(body_names) * 3]
      - If flatten=False: [num_envs, window_size, len(body_names) * 3]
    """
    teacher_command = _get_multimotion_command_v2(env, teacher_command_name)
    student_command = _get_multimotion_command_v2(env, student_command_name)

    if not body_names:
        out_shape = (env.num_envs, 0) if flatten else (env.num_envs, window_size, 0)
        return torch.zeros(*out_shape, device=env.device, dtype=torch.float32)

    teacher_body_ids = torch.tensor(
        [teacher_command.cfg.body_names.index(name) for name in body_names],
        device=env.device,
        dtype=torch.long,
    )
    student_body_ids = torch.tensor(
        [student_command.cfg.body_names.index(name) for name in body_names],
        device=env.device,
        dtype=torch.long,
    )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    _, _, teacher_body_pos_w, teacher_body_quat_w, _, _ = teacher_command.data_store.get_motion_window_full(
        teacher_command.motion_ids[env_ids],
        teacher_command.frame_ids[env_ids],
        window_size=window_size,
        stride=stride,
    )
    _, _, student_body_pos_w, student_body_quat_w, _, _ = student_command.data_store.get_motion_window_full(
        student_command.motion_ids[env_ids],
        student_command.frame_ids[env_ids],
        window_size=window_size,
        stride=stride,
    )

    teacher_anchor_pos_w = teacher_body_pos_w[:, :, teacher_command.motion_anchor_body_index]
    teacher_anchor_quat_w = teacher_body_quat_w[:, :, teacher_command.motion_anchor_body_index]
    student_anchor_pos_w = student_body_pos_w[:, :, student_command.motion_anchor_body_index]
    student_anchor_quat_w = student_body_quat_w[:, :, student_command.motion_anchor_body_index]

    teacher_foot_pos_w = teacher_body_pos_w.index_select(dim=2, index=teacher_body_ids)
    student_foot_pos_w = student_body_pos_w.index_select(dim=2, index=student_body_ids)

    num_envs = env.num_envs
    num_bodies = len(body_names)

    teacher_rel = teacher_foot_pos_w - teacher_anchor_pos_w.unsqueeze(2)
    teacher_rel = quat_apply(
        quat_inv(teacher_anchor_quat_w).unsqueeze(2).expand(-1, -1, num_bodies, -1).reshape(-1, 4),
        teacher_rel.reshape(-1, 3),
    ).reshape(num_envs, window_size, num_bodies, 3)

    student_rel = student_foot_pos_w - student_anchor_pos_w.unsqueeze(2)
    student_rel = quat_apply(
        quat_inv(student_anchor_quat_w).unsqueeze(2).expand(-1, -1, num_bodies, -1).reshape(-1, 4),
        student_rel.reshape(-1, 3),
    ).reshape(num_envs, window_size, num_bodies, 3)

    delta = teacher_rel - student_rel
    if flatten:
        return delta.reshape(num_envs, -1).float()
    return delta.reshape(num_envs, window_size, -1).float()


def vel_gt(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ground-truth base linear velocity in body frame (supervision only).

    Shape: [num_envs, 3] = [v_x, v_y, v_z]
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def vel_yaw_gt(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ground-truth base linear velocity + yaw-rate in body frame (supervision only).

    Shape: [num_envs, 4] = [v_x, v_y, v_z, yaw_rate]
    """
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    yaw_rate = asset.data.root_ang_vel_b[:, 2:3]
    return torch.cat([lin_vel, yaw_rate], dim=-1)

# SONIC Observations
def sonic_robot_motion(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get SONIC robot motion input (future 10 frames)."""
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    if hasattr(command, "get_sonic_robot_window"):
        # Use env.scene.env_ids if available (InteractiveScene usually doesn't have env_ids directly accessible as a property in recent versions,
        # but ManagerBasedEnv should have access to them or just pass indices).
        # Actually, in IsaacLab, the environment usually has `env.scene.env_ids` if we are resetting,
        # but for observations we typically process all environments.
        # command.get_sonic_robot_window expects indices.
        # We should pass torch.arange(env.num_envs, device=env.device) if we want all.
        
        # env.scene might not expose env_ids. The command methods likely expect a tensor of indices.
        # Let's pass all env indices.
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        return command.get_sonic_robot_window(all_env_ids)
    # return torch.zeros(env.num_envs, 580, device=env.device) # Fallback

def sonic_human_motion(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get SONIC human motion input (future 10 frames)."""
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    if hasattr(command, "get_sonic_human_window"):
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        return command.get_sonic_human_window(all_env_ids)
    # return torch.zeros(env.num_envs, 660, device=env.device) # Fallback


def sonic_robot_motion_delta(
    env: ManagerBasedEnv,
    teacher_command_name: str,
    student_command_name: str,
) -> torch.Tensor:
    """Future 10-frame motion residual between optimized and raw commands."""

    return sonic_robot_motion(env, teacher_command_name) - sonic_robot_motion(env, student_command_name)



def motion_anchor_delta_b(
    env: ManagerBasedEnv,
    teacher_command_name: str,
    student_command_name: str,
) -> torch.Tensor:
    """Current anchor residual in the robot frame between optimized and raw commands."""

    pos_delta = motion_anchor_pos_b(env, teacher_command_name) - motion_anchor_pos_b(env, student_command_name)
    ori_delta = motion_anchor_ori_b(env, teacher_command_name) - motion_anchor_ori_b(env, student_command_name)
    return torch.cat([pos_delta, ori_delta], dim=-1)


def sonic_decoder_step_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str | None = None,
) -> torch.Tensor:
    """Single-step SONIC decoder proprio observation (93D).

    Order per step:
        [base_ang_vel(3), joint_pos(29), joint_vel(29), last_action(29), gravity_dir(3)].

    This is intended to be used with an observation-group history length of 10, yielding:
        10 * 93 = 930 dims.
    """
    asset = env.scene[asset_cfg.name]
    omega = asset.data.root_ang_vel_b
    # Official SONIC deploy logs body_q as (joint_pos - default_joint_pos).
    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dq = asset.data.joint_vel[:, asset_cfg.joint_ids]
    g_proj = asset.data.projected_gravity_b

    if action_name is None:
        a_prev = env.action_manager.action
    else:
        a_prev = env.action_manager.get_term(action_name).raw_actions

    return torch.cat([omega, q, dq, a_prev, g_proj], dim=-1)


def sonic_g1_encoder_branch(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """G1 encoder branch input for SONIC deploy contract (640D).

    Layout:
        motion_joint_positions_and_velocities_10frame_step5 (580)
        motion_anchor_orientation_10frame_step5 (60)

    The first 580 dims follow the released deploy wrapper contract:
        [q_0..q_9 (290), dq_0..dq_9 (290)]

    ``SonicActorCritic._prepare_robot_encoder_input(..., g1_onnx_repack)``
    then applies the same [10, 58] reshape used by the exported ONNX wrapper.
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiMotionCommandV2):
        raise TypeError(
            f"sonic_g1_encoder_branch expects MultiMotionCommandV2 under command '{command_name}'. "
            f"Received: {type(command)}"
        )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Future joint windows: 10 frames sampled every 5 ticks.
    joint_pos, joint_vel = command.data_store.get_motion_window(
        command.motion_ids[env_ids],
        command.frame_ids[env_ids],
        window_size=10,
        stride=5,
    )

    # Future anchor orientation window (10 x 6D).
    if command.data_store._stacked_body_quat_w is None or command.data_store.motion_lengths is None:
        raise RuntimeError("Motion data buffers are not initialized for SONIC encoder observations.")

    storage_device = command.data_store.storage_device
    motion_ids = command.motion_ids[env_ids].to(storage_device)
    current_steps = command.frame_ids[env_ids].to(storage_device)
    offsets = torch.arange(10, device=storage_device, dtype=current_steps.dtype) * 5

    frame_indices = current_steps.unsqueeze(1) + offsets.unsqueeze(0)
    motion_lengths = command.data_store.motion_lengths[motion_ids].unsqueeze(1).to(storage_device)
    frame_indices = torch.clamp(frame_indices, min=0)
    frame_indices = torch.minimum(frame_indices, motion_lengths - 1).long()

    batch_motion_ids = motion_ids.unsqueeze(1).expand(-1, 10)
    anchor_quat = command.data_store._stacked_body_quat_w[
        batch_motion_ids, frame_indices, command.motion_anchor_body_index
    ]

    if joint_pos.device != env.device:
        joint_pos = joint_pos.to(env.device)
    if joint_vel.device != env.device:
        joint_vel = joint_vel.to(env.device)
    if anchor_quat.device != env.device:
        anchor_quat = anchor_quat.to(env.device)

    # Match deploy gatherer semantics: use base->reference orientation, not raw reference world orientation.
    base_quat = command.robot_anchor_quat_w[:, None, :].expand(-1, 10, -1)
    anchor_rel_quat = quat_mul(quat_inv(base_quat), anchor_quat)
    anchor_ori_6d = matrix_from_quat(anchor_rel_quat)[..., :2].reshape(env.num_envs, -1)

    g1_encoder = torch.cat(
        [joint_pos.reshape(env.num_envs, -1), joint_vel.reshape(env.num_envs, -1), anchor_ori_6d],
        dim=-1,
    ).float()

    if g1_encoder.shape[-1] != 640:
        raise RuntimeError(f"Expected g1 encoder dim 640, got {g1_encoder.shape[-1]}.")
    return g1_encoder


def sonic_teleop_placeholder(env: ManagerBasedEnv) -> torch.Tensor:
    """Placeholder for teleop branch input (267D)."""
    return torch.zeros(env.num_envs, 267, device=env.device, dtype=torch.float)


def sonic_smpl_placeholder(env: ManagerBasedEnv) -> torch.Tensor:
    """Placeholder for SMPL branch input (840D)."""
    return torch.zeros(env.num_envs, 840, device=env.device, dtype=torch.float)


def sonic_encoder_mode_4(env: ManagerBasedEnv, mode_id: int = 0) -> torch.Tensor:
    """Encoder mode one-hot-like payload used by deploy SONIC encoder.

    Layout matches deploy's `encoder_mode_4`: [mode_id, 0, 0, 0].
    """
    mode = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float)
    mode[:, 0] = float(mode_id)
    return mode

def _apply_hit_z_dropout(hit_z: torch.Tensor, nan_dropout_prob: float) -> torch.Tensor:
    if nan_dropout_prob <= 0.0:
        return hit_z
    if not 0.0 <= nan_dropout_prob <= 1.0:
        raise ValueError(f"nan_dropout_prob must be in [0, 1], got {nan_dropout_prob}")
    dropout_mask = torch.rand_like(hit_z) < nan_dropout_prob
    return hit_z.masked_fill(dropout_mask, torch.nan)


def _apply_uniform_noise(values: torch.Tensor, noise_range: tuple[float, float] | None) -> torch.Tensor:
    if noise_range is None:
        return values
    noise_min, noise_max = float(noise_range[0]), float(noise_range[1])
    if noise_max < noise_min:
        raise ValueError(f"noise_range must satisfy max >= min, got {noise_range}")
    if noise_min == 0.0 and noise_max == 0.0:
        return values
    return values + torch.empty_like(values).uniform_(noise_min, noise_max)


def _apply_clip(values: torch.Tensor, clip_range: tuple[float, float] | None) -> torch.Tensor:
    if clip_range is None:
        return values
    clip_min, clip_max = float(clip_range[0]), float(clip_range[1])
    if clip_max < clip_min:
        raise ValueError(f"clip_range must satisfy max >= min, got {clip_range}")
    return values.clamp(min=clip_min, max=clip_max)


def height_scan_fill(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.0,
    nan_dropout_prob: float = 0.0,
    invalid_fill_value: float = 30.0,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    ``nan_dropout_prob`` simulates missing depth returns by randomly dropping a
    fraction of ``hit_z`` values before the existing NaN/Inf fill path.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]
    hit_z = _apply_hit_z_dropout(hit_z, nan_dropout_prob)
    hit_z = torch.nan_to_num(hit_z, nan=invalid_fill_value, posinf=invalid_fill_value, neginf=invalid_fill_value)
    sensor_height = sensor.data.pos_w[:, 2].unsqueeze(1)
    sensor_height = torch.nan_to_num(sensor_height, nan=1.0, posinf=1.0, neginf=1.0)
    return sensor_height - hit_z - offset


def height_scan_for_vision(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.,
    nan_dropout_prob: float = 0.0,
    invalid_fill_value: float = 30.0,
    append_validity_mask: bool = False,
    noise_range: tuple[float, float] | None = None,
    clip_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Height scan for vision transformer input.
    
    Returns flat height values suitable for VisionSonicActorCritic's process_height_scan().
    The VisionSonicActorCritic will internally convert this to (x, y, z) format.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the height scanner sensor.
        offset: Height offset to subtract (default 0.5m).
        nan_dropout_prob: Probability of dropping a ray hit before filling invalids.
        invalid_fill_value: Replacement z-value for NaN/Inf hits after dropout.
        append_validity_mask: If True, append a flat 0/1 validity mask after the
            height values. This is intended for stepping-stone vision only.
        noise_range: Optional uniform noise range applied to the height values only.
        clip_range: Optional clip range applied to the height values only after noise.
        
    Returns:
        Tensor of shape (num_envs, H*W) containing relative heights by default,
        or (num_envs, 2*H*W) when ``append_validity_mask`` is True.
        The grid dimensions are determined by the RayCaster's pattern config.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # Get hit z-coordinates and handle invalid values
    hit_z = sensor.data.ray_hits_w[..., 2]
    hit_z = _apply_hit_z_dropout(hit_z, nan_dropout_prob)
    validity_mask = torch.isfinite(hit_z).to(dtype=torch.float32)
    hit_z = torch.nan_to_num(hit_z, nan=invalid_fill_value, posinf=invalid_fill_value, neginf=invalid_fill_value)
    
    # Compute relative height (sensor height - hit point - offset)
    sensor_height = sensor.data.pos_w[:, 2:3]  # [B, 1]
    sensor_height = torch.nan_to_num(sensor_height, nan=0.78, posinf=0.78, neginf=0.78)
    
    # Relative height: how far below the sensor each point is
    relative_height = sensor_height - hit_z - offset  # [B, num_rays]
    relative_height = _apply_uniform_noise(relative_height, noise_range)
    relative_height = _apply_clip(relative_height, clip_range)
    if append_validity_mask:
        relative_height = torch.cat([relative_height, validity_mask], dim=-1)
    return relative_height


def zero_height_scan_for_vision(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    append_validity_mask: bool = False,
) -> torch.Tensor:
    """Height-scan-shaped zero observation for blind policies.

    This preserves the checkpoint/schema contract for vision-capable policies
    while removing terrain height information from the actual input.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    num_rays = int(sensor.data.ray_hits_w.shape[1])
    height = torch.zeros(env.num_envs, num_rays, device=env.device, dtype=torch.float32)
    if append_validity_mask:
        validity = torch.ones_like(height)
        return torch.cat([height, validity], dim=-1)
    return height


def depth_image_for_vision(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    depth_clip: tuple[float, float] = (0.15, 3.0),
    normalize: bool = True,
    nan_dropout_prob: float = 0.0,
    invalid_fill_value: float | None = None,
    flatten: bool = False,
    append_validity_mask: bool = False,
) -> torch.Tensor:
    """Return a normalized depth image for future vision-CNN/SRU policies.

    This keeps the depth path separate from the existing height-scan path.
    The output is channel-first ``[B, 1, H, W]`` unless ``flatten=True``.
    When ``append_validity_mask=True``, a second channel stores the 0/1 validity mask.
    """
    sensor: Camera | RayCasterCamera | TiledCamera = env.scene.sensors[sensor_cfg.name]
    image = sensor.data.output[data_type]

    if image.ndim == 3:
        image = image.unsqueeze(-1)
    if image.ndim != 4:
        raise ValueError(
            f"Depth image observation '{sensor_cfg.name}:{data_type}' must be 3D or 4D. Got shape: {tuple(image.shape)}"
        )

    if image.shape[-1] != 1:
        image = image[..., :1]

    if nan_dropout_prob < 0.0 or nan_dropout_prob > 1.0:
        raise ValueError(f"nan_dropout_prob must be in [0, 1], got {nan_dropout_prob}")
    if nan_dropout_prob > 0.0:
        dropout_mask = (torch.rand_like(image[..., 0]) < nan_dropout_prob).unsqueeze(-1)
        image = image.masked_fill(dropout_mask, torch.nan)

    validity_mask = torch.isfinite(image).to(dtype=torch.float32)
    min_depth, max_depth = float(depth_clip[0]), float(depth_clip[1])
    if max_depth <= min_depth:
        raise ValueError(f"depth_clip must satisfy max > min, got {depth_clip}")

    fill_value = max_depth if invalid_fill_value is None else float(invalid_fill_value)
    image = torch.nan_to_num(image, nan=fill_value, posinf=fill_value, neginf=fill_value)
    image = image.clamp(min=min_depth, max=max_depth)
    if normalize:
        image = (image - min_depth) / (max_depth - min_depth + 1.0e-8)

    if append_validity_mask:
        image = torch.cat([image, validity_mask], dim=-1)

    image = image.permute(0, 3, 1, 2).contiguous().float()
    if flatten:
        return image.flatten(start_dim=1)
    return image
