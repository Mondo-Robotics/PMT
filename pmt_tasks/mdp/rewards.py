from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, yaw_quat
from isaaclab.assets import Articulation, RigidObject
from pmt_tasks.mdp.commands import MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand | MultiMotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def apply_reward_weight_set(rewards_cfg, weight_set) -> None:
    """Apply a data-driven reward weight-set onto a ``RewardsCfg`` instance (§9b).

    This realizes the ``configs/reward/*.yaml`` "reward-as-data" axis: instead of a
    bespoke ``RewardsCfg`` subclass per task differing only by weights, a task ships
    a ``{term_name: value}`` dict and this helper mutates the (already-instantiated)
    rewards cfg in place. Meant to be called from ``__post_init__`` and from the
    builder's reward-weight injection path so both share one code path.

    ``value`` may be either:
      - a scalar (``float``/``int``): sets ``term.weight``;
      - a mapping: ``{"weight": <float>, **param_overrides}`` -- sets ``term.weight``
        (if present) and each remaining key onto ``term.params``.

    Terms named in ``weight_set`` that are absent on ``rewards_cfg`` (e.g. ``None``)
    are silently skipped, matching the prior inline behavior. Behavior-preserving
    for the scalar case that env cfgs use today.
    """
    if not weight_set:
        return
    items = weight_set.items() if hasattr(weight_set, "items") else dict(weight_set).items()
    for term_name, value in items:
        term = getattr(rewards_cfg, term_name, None)
        if term is None:
            continue
        if isinstance(value, (int, float)):
            term.weight = float(value)
            continue
        # mapping: weight + optional param overrides
        mapping = dict(value)
        if "weight" in mapping:
            term.weight = float(mapping.pop("weight"))
        if mapping:
            params = getattr(term, "params", None)
            if params is None:
                term.params = dict(mapping)
            else:
                params.update(mapping)


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def negative_joint_power_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
    deadband: float = 150.0,
    power_norm: float = 500.0,
) -> torch.Tensor:
    """Penalize excessive negative joint power after a deadband."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    negative_power = torch.clamp(-(applied_torque * joint_vel) - deadband, min=0.0)
    return torch.sum(torch.square(negative_power / power_norm), dim=-1)


def raw_terrain_anchor_position_z_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    raw_terrain_pos = command.raw_body_pos_w_with_terrain_height
    raw_anchor_z = raw_terrain_pos[:, command.motion_anchor_body_index, 2]
    robot_anchor_z = command.robot_anchor_pos_w[:, 2]
    error = torch.square(raw_anchor_z - robot_anchor_z)
    return torch.exp(-error / std**2)


def raw_terrain_anchor_position_xy_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    raw_terrain_pos = command.raw_body_pos_w_with_terrain_height
    raw_anchor_xy = raw_terrain_pos[:, command.motion_anchor_body_index, :2]
    robot_anchor_xy = command.robot_anchor_pos_w[:, :2]
    error = torch.sum(torch.square(raw_anchor_xy - robot_anchor_xy), dim=-1)
    return torch.exp(-error / std**2)


def raw_terrain_anchor_position_xy_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    raw_terrain_pos = command.raw_body_pos_w_with_terrain_height
    raw_anchor_xy = raw_terrain_pos[:, command.motion_anchor_body_index, :2]
    robot_anchor_xy = command.robot_anchor_pos_w[:, :2]
    return torch.sum(torch.square(raw_anchor_xy - robot_anchor_xy), dim=-1)


def raw_terrain_anchor_yaw_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    raw_anchor_quat = command.raw_body_quat_w[:, command.motion_anchor_body_index]
    robot_anchor_quat = command.robot_anchor_quat_w
    error = quat_error_magnitude(yaw_quat(raw_anchor_quat), yaw_quat(robot_anchor_quat)) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def tracking_local_vr_5point_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Track five sparse body points in the local anchor frame.

    This mirrors the SONIC release reward term while using the body buffers
    exposed by WBT's motion commands.  If ``body_names`` is omitted, the term
    uses pelvis, wrists, and ankles when those bodies are present in the command.
    """
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        body_names = [
            "pelvis",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
    body_indexes = _get_body_indexes(command, body_names)
    if len(body_indexes) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    ref_diff = command.body_pos_w[:, body_indexes] - command.anchor_pos_w[:, None, :]
    robot_diff = command.robot_body_pos_w[:, body_indexes] - command.robot_anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w[:, None, :].expand(-1, len(body_indexes), -1)
    robot_root_quat = command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indexes), -1)

    ref_local = quat_apply(quat_inv(ref_root_quat), ref_diff)
    robot_local = quat_apply(quat_inv(robot_root_quat), robot_diff)
    error = torch.sum(torch.square(robot_local - ref_local), dim=-1)
    return torch.exp(-error.mean(dim=-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def raw_terrain_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.raw_terrain_body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def raw_terrain_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.raw_terrain_body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize high angular velocity jitter on selected links with a deadzone."""
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    body_indexes = _get_body_indexes(command, body_names)
    if len(body_indexes) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    speed = torch.linalg.norm(command.robot_body_ang_vel_w[:, body_indexes], dim=-1)
    excess = torch.relu(speed - threshold)
    return torch.mean(torch.square(excess), dim=-1)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def feet_lateral_contact_force_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.0
) -> torch.Tensor:
    """Penalty on the horizontal (XY) component of net contact force on selected bodies.

    Foot-on-flat-ground reactions are ~vertical; lateral force appears when a link collides
    with a vertical obstacle face (e.g. the side of a stepping stone). Summing the XY-norm
    over the selected bodies yields a scalar penalty per env.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_xy = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2]
    lateral_norm = torch.linalg.norm(forces_xy, dim=-1)
    if threshold > 0.0:
        lateral_norm = torch.clamp(lateral_norm - threshold, min=0.0)
    return torch.sum(lateral_norm, dim=-1)
