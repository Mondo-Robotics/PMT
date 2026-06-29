from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from pmt_tasks.mdp.commands import MotionCommand, MultiMotionCommand
from pmt_tasks.mdp.rewards import _get_body_indexes


def exceeded_tracking_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    dimensions: str = "xyz",
    use_terrain: bool = False,
) -> torch.Tensor:
    """Parametrized anchor-position tracking-error termination (§9b consolidation).

    Collapses ``bad_anchor_pos`` (xyz norm), ``bad_anchor_pos_z_only`` (|z|), and
    ``bad_raw_terrain_anchor_pos_xy`` (xy norm against the terrain-height anchor)
    into one function. The three original names are kept as thin wrappers below so
    existing env cfgs keep importing them; this is behavior-preserving.

    Args:
        dimensions: ``"xyz"`` -> 3D norm, ``"z"`` -> absolute z error, ``"xy"`` ->
            2D (horizontal) norm.
        use_terrain: when True, the reference anchor xy/z is taken from
            ``raw_body_pos_w_with_terrain_height`` at the anchor body index
            (matches the old ``bad_raw_terrain_anchor_pos_xy``). Only meaningful
            for ``dimensions="xy"`` today (the only old terrain wrapper).
    """
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)

    if use_terrain:
        raw_terrain_pos = command.raw_body_pos_w_with_terrain_height
        ref_anchor = raw_terrain_pos[:, command.motion_anchor_body_index, :]
    else:
        ref_anchor = command.anchor_pos_w
    robot_anchor = command.robot_anchor_pos_w

    if dimensions == "z":
        return torch.abs(ref_anchor[:, -1] - robot_anchor[:, -1]) > threshold
    if dimensions == "xy":
        return torch.norm(ref_anchor[:, :2] - robot_anchor[:, :2], dim=1) > threshold
    if dimensions == "xyz":
        return torch.norm(ref_anchor[:, :3] - robot_anchor[:, :3], dim=1) > threshold
    raise ValueError(f"Unknown dimensions '{dimensions}' (expected 'xyz'|'z'|'xy').")


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Thin wrapper over :func:`exceeded_tracking_error` (dimensions='xyz')."""
    return exceeded_tracking_error(env, command_name, threshold, dimensions="xyz")


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Thin wrapper over :func:`exceeded_tracking_error` (dimensions='z')."""
    return exceeded_tracking_error(env, command_name, threshold, dimensions="z")


def bad_raw_terrain_anchor_pos_xy(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Thin wrapper over :func:`exceeded_tracking_error` (dimensions='xy', terrain)."""
    return exceeded_tracking_error(env, command_name, threshold, dimensions="xy", use_terrain=True)


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def exceeded_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Terminate when any selected body exceeds a full XYZ position threshold.

    Canonical impl for the (verified identical) ``bad_motion_body_pos`` /
    ``exceeded_body_pos`` pair (§9b). ``bad_motion_body_pos`` below is a thin alias.
    """
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Thin alias of :func:`exceeded_body_pos` (verified identical, §9b)."""
    return exceeded_body_pos(env, command_name, threshold, body_names=body_names)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Thin wrapper over :func:`exceeded_body_height` (threshold_adaptive=False, §9b).

    Behavior-preserving: the old impl took ``[..., -1]`` (the z column of a 3-vector,
    i.e. index 2) and ``torch.any(error > threshold, dim=-1)`` -- exactly the
    non-adaptive branch of ``exceeded_body_height`` (which uses index ``2``).
    """
    return exceeded_body_height(
        env, command_name, threshold, threshold_adaptive=False, body_names=body_names
    )


def exceeded_anchor_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    root_height_threshold: float = 0.5,
) -> torch.Tensor:
    """SONIC-style anchor height termination."""
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    height_error = torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    if not threshold_adaptive:
        return height_error > threshold

    ref_height = command.anchor_pos_w[:, 2]
    adaptive_threshold = torch.full_like(height_error, threshold)
    adaptive_threshold[ref_height < root_height_threshold] = down_threshold
    return height_error > adaptive_threshold


def exceeded_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """SONIC-style full anchor orientation termination."""
    del asset_cfg
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    return torch.square(quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)) > threshold


def exceeded_body_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    body_names: list[str] | None = None,
    root_height_threshold: float = 0.5,
) -> torch.Tensor:
    """SONIC-style end-effector height termination."""
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    height_error = torch.abs(
        command.body_pos_relative_w[:, body_indexes, 2] - command.robot_body_pos_w[:, body_indexes, 2]
    )
    if not threshold_adaptive:
        return torch.any(height_error > threshold, dim=-1)

    ref_height = command.anchor_pos_w[:, 2]
    adaptive_threshold = torch.full_like(height_error, threshold)
    adaptive_threshold[ref_height < root_height_threshold] = down_threshold
    return torch.any(height_error > adaptive_threshold, dim=-1)


def tracking_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the sampled motion clip has been consumed."""
    command: MotionCommand | MultiMotionCommand = env.command_manager.get_term(command_name)
    if hasattr(command, "data_store") and getattr(command.data_store, "motion_lengths", None) is not None:
        lengths = command.data_store.motion_lengths[command.motion_ids.to(command.data_store.motion_lengths.device)]
        frame_ids = command.frame_ids.to(lengths.device)
        return (frame_ids + 1 >= lengths).to(env.device)
    return env.episode_length_buf >= env.max_episode_length - 1
