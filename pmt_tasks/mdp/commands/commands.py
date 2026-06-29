from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any
import math
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int] | torch.Tensor,
        device: str | torch.device = "cpu",
        use_mmap: bool = False,
        use_fp16: bool = False,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        mmap_mode = "r" if use_mmap else None
        data = np.load(motion_file, mmap_mode=mmap_mode)
        self.fps = data["fps"]
        tensor_device = torch.device(device)
        # Use float16 for memory optimization if enabled (50% memory reduction)
        dtype = torch.float16 if use_fp16 else torch.float32
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=dtype, device=tensor_device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=dtype, device=tensor_device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=dtype, device=tensor_device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=dtype, device=tensor_device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=dtype, device=tensor_device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=dtype, device=tensor_device)
        self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=tensor_device)
        self.time_step_total = self.joint_pos.shape[0]
        self._use_fp16 = use_fp16
        self._motion_file = motion_file  # Store for debug
        
        # Validate body indexes against data shape
        num_bodies_in_file = self._body_pos_w.shape[1]
        max_body_idx = self._body_indexes.max().item() if len(self._body_indexes) > 0 else 0
        if max_body_idx >= num_bodies_in_file:
            raise ValueError(
                f"[MotionLoader] Body index {max_body_idx} out of range for motion file {motion_file}. "
                f"File has {num_bodies_in_file} bodies, but body_indexes requires up to index {max_body_idx}. "
                f"body_pos_w shape: {self._body_pos_w.shape}, body_indexes: {self._body_indexes.tolist()}"
            )

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)

        self.motion_library_device = torch.device(self.cfg.motion_library_device)
        self.motion_library_mmap = self.cfg.motion_library_mmap

        body_indices = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long
        )
        self.body_indexes = body_indices.to(self.device)
        # store a copy on the motion-library device (may be same as env device)
        self.motion_body_indexes = body_indices.to(self.motion_library_device)

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()
        self._update_motion_buffers_for_envs(env_ids)
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = anchor_pos_w_repeat - robot_anchor_pos_w_repeat
        delta_pos_w[..., :2] = 0.0
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = (
            robot_anchor_pos_w_repeat + delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING  # type: ignore[assignment]

    motion_file: str = MISSING  # type: ignore[assignment]
    anchor_body: str = MISSING  # type: ignore[assignment]
    body_names: list[str] = MISSING  # type: ignore[assignment]

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Memory optimization options for motion library
    motion_library_device: str = "cuda"  # "cuda" for GPU (fast), "cpu" for CPU (saves GPU VRAM)
    motion_library_mmap: bool = False  # True to use memory-mapped files (reduces RAM usage)
    motion_library_fp16: bool = False  # True to use float16 (50% memory reduction)


class MultiMotionCommand(CommandTerm):
    """Command term that supports multiple motion files.

    Each environment can sample from different motion files.
    """
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.motion_library_device = torch.device(getattr(self.cfg, "motion_library_device", "cuda"))
        self.motion_library_mmap = getattr(self.cfg, "motion_library_mmap", False)
        self.motion_library_fp16 = getattr(self.cfg, "motion_library_fp16", False)
        # Determine dtype for motion data (float16 saves 50% GPU memory)
        self._motion_dtype = torch.float16 if self.motion_library_fp16 else torch.float32

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        body_indices = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long
        )
        self.body_indexes = body_indices.to(self.device)
        self.motion_body_indexes = body_indices.to(self.motion_library_device)

        # Load all motion files
        self.motions = []
        self.human_motions = [] # SONIC: Human motion data
        for motion_file in self.cfg.motion_files:
            motion = MotionLoader(
                motion_file,
                self.motion_body_indexes,
                device=self.motion_library_device,
                use_mmap=self.motion_library_mmap,
                use_fp16=self.motion_library_fp16,
            )
            self.motions.append(motion)
            
            # SONIC: Load paired human motion
            # Assume corresponding file is in ../human_lafan1/.. instead of ../g1_lafan1/..
            # and has the same filename.
            human_file = motion_file.replace("robot_", "human_")
            # We assume human data is also converted to npz with "positions" key [T, J, 3]
            # AND we need to apply the same rotation (-90 yaw) and translation (x+1.0, y+1.5) as in replay script
            if os.path.exists(human_file):
                try:
                    h_data = np.load(human_file, mmap_mode="r" if self.motion_library_mmap else None)
                    # positions: [T, J, 3]
                    raw_positions = torch.tensor(h_data["positions"], dtype=torch.float32, device=self.motion_library_device)
                    
                    # --- Apply Alignment Transforms ---
                    # 1. Rotate -90 deg around Z (Yaw)
                    rot_quat = quat_from_euler_xyz(
                        torch.tensor(0.0, device=self.motion_library_device), 
                        torch.tensor(0.0, device=self.motion_library_device), 
                        torch.tensor(np.pi/2, device=self.motion_library_device)
                    )
                    # Expand for batch: [T*J, 3]
                    num_frames, num_joints, _ = raw_positions.shape
                    flat_pos = raw_positions.view(-1, 3)
                    batch_quat = rot_quat.unsqueeze(0).expand(flat_pos.shape[0], -1)
                    
                    rotated_pos = quat_apply(batch_quat, flat_pos).view(num_frames, num_joints, 3)
                    
                    # 2. Translate X+1.0, Y+1.5
                    rotated_pos[:, :, 0] += 0.0
                    rotated_pos[:, :, 1] += 0.0
                    
                    # Store processed data
                    self.human_motions.append({
                        "joint_pos": rotated_pos, # Now aligned with robot world frame
                        "fps": h_data.get("fps", 50.0),
                        "time_step_total": num_frames,
                    })
                except Exception as e:
                    print(f"[MultiMotionCommand] Failed to load human motion {human_file}: {e}")
                    self.human_motions.append(None)
            else:
                 print(f"[MultiMotionCommand] Human motion file not found: {human_file}")
                 self.human_motions.append(None)

        # Chunk long motions if configured (avoids multinomial 2^24 limit)
        if getattr(self.cfg, 'motion_chunk_length', 0) > 0:
            self.motions, self.human_motions = self._chunk_all_motions(
                self.motions, self.human_motions, self.cfg.motion_chunk_length
            )

        self.num_motions = len(self.motions)
        
        # Pre-compute motion lengths for efficient vectorized operations
        self.motion_lengths = torch.tensor(
            [m.time_step_total for m in self.motions], dtype=torch.long, device=self.device
        )
        self.human_motion_lengths = torch.tensor(
            [n['time_step_total'] for n in self.human_motions], dtype=torch.long, device=self.device
        )        
        # Pre-stack motion data for vectorized access (pad to max length)
        # This enables fully vectorized updates without for loops
        self.max_motion_length = max(m.time_step_total for m in self.motions)
        self.max_human_motion_length = max(n['time_step_total'] for n in self.human_motions)
        num_joints = self.motions[0].joint_pos.shape[1]
        num_bodies = len(cfg.body_names)
        
        # Stack and pad all motion data [num_motions, max_length, ...]
        # Use configured dtype (float16 for memory optimization)
        self._stacked_joint_pos = torch.zeros(
            self.num_motions, self.max_motion_length, num_joints, dtype=self._motion_dtype, device=self.motion_library_device
        )
        self._stacked_joint_vel = torch.zeros(
            self.num_motions, self.max_motion_length, num_joints, dtype=self._motion_dtype, device=self.motion_library_device
        )
        self._stacked_body_pos_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 3, dtype=self._motion_dtype, device=self.motion_library_device
        )
        self._stacked_body_quat_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 4, dtype=self._motion_dtype, device=self.motion_library_device
        )
        self._stacked_body_lin_vel_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 3, dtype=self._motion_dtype, device=self.motion_library_device
        )
        self._stacked_body_ang_vel_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 3, dtype=self._motion_dtype, device=self.motion_library_device
        )
        
        for i, motion in enumerate(self.motions):
            length = motion.time_step_total
            self._stacked_joint_pos[i, :length] = motion.joint_pos
            self._stacked_joint_vel[i, :length] = motion.joint_vel
            self._stacked_body_pos_w[i, :length] = motion.body_pos_w
            self._stacked_body_quat_w[i, :length] = motion.body_quat_w
            self._stacked_body_lin_vel_w[i, :length] = motion.body_lin_vel_w
            self._stacked_body_ang_vel_w[i, :length] = motion.body_ang_vel_w

        # SONIC: Stack human data
        self._stacked_human_joint_pos = None
        if any(h is not None for h in self.human_motions):
            valid_h = next(h for h in self.human_motions if h is not None)
            # Assuming [T, J, 3] structure
            if len(valid_h["joint_pos"].shape) == 3:
                h_num_joints = valid_h["joint_pos"].shape[1]
                h_joint_dim = valid_h["joint_pos"].shape[2]
                self._stacked_human_joint_pos = torch.zeros(
                    self.num_motions, self.max_human_motion_length, h_num_joints, h_joint_dim,
                    device=self.motion_library_device
                )
                for i, h_motion in enumerate(self.human_motions):
                    if h_motion is not None:
                        length = min(h_motion["joint_pos"].shape[0], self.max_human_motion_length)
                        self._stacked_human_joint_pos[i, :length] = h_motion["joint_pos"][:length]
            else:
                print("[MultiMotionCommand] Human joint_pos has unexpected shape, skipping stack.")

        # Track which motion each environment is using
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Track the STARTING motion and frame for each episode (for adaptive sampling)
        # These are set when an episode begins and used to update failure counts when it ends
        self.start_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Cache buffers for properties to avoid memory leaks from repeated allocation
        # These are updated in _update_command() and reused in properties
        self._joint_pos_buf = torch.zeros(self.num_envs, self.motions[0].joint_pos.shape[1], device=self.device)
        self._joint_vel_buf = torch.zeros(self.num_envs, self.motions[0].joint_vel.shape[1], device=self.device)
        self._body_pos_w_buf = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self._body_quat_w_buf = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self._body_quat_w_buf[:, :, 0] = 1.0
        self._body_lin_vel_w_buf = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self._body_ang_vel_w_buf = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)

        # Initialize metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        
        # Adaptive sampling metrics and tracking
        if self.cfg.enable_adaptive_sampling:
            # Track failure and success counts per motion (with small initial count to avoid division by zero)
            self.failed_motion_count = torch.ones(self.num_motions, device=self.device)
            self.success_motion_count = torch.ones(self.num_motions, device=self.device)
            self.current_failed_motion_count = torch.zeros(self.num_motions, device=self.device)
            self.current_success_motion_count = torch.zeros(self.num_motions, device=self.device)
            
            # Metrics for monitoring adaptive sampling
            self.metrics["adaptive_sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["adaptive_pfail_entropy"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["adaptive_pfail_mean"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["adaptive_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["total_failed_count"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["total_success_count"] = torch.zeros(self.num_envs, device=self.device)
            # Optional: per-frame adaptive sampling structures (opt-in)
            if getattr(self.cfg, "enable_frame_adaptive_sampling", False):
                # Per-frame counts: [num_motions, max_motion_length]
                self.failed_frame_count = torch.ones(self.num_motions, self.max_motion_length, device=self.device)
                self.success_frame_count = torch.ones(self.num_motions, self.max_motion_length, device=self.device)
                self.current_failed_frame_count = torch.zeros(self.num_motions, self.max_motion_length, device=self.device)
                self.current_success_frame_count = torch.zeros(self.num_motions, self.max_motion_length, device=self.device)
                # total flattened frames for efficient bincount operations
                self._total_frames = int(self.num_motions * self.max_motion_length)
                
                # Kernel for conv1d smoothing of failure probabilities across adjacent frames
                if self.cfg.adaptive_kernel_size > 1:
                    kernel = torch.tensor(
                        [self.cfg.adaptive_lambda ** i for i in range(self.cfg.adaptive_kernel_size)],
                        device=self.device, dtype=torch.float32
                    )
                    self._adaptive_kernel = (kernel / kernel.sum()).view(1, 1, -1)
                else:
                    self._adaptive_kernel = None
            
            # Counter for interval-based EMA updates (fast: updates sampling probabilities)
            self._steps_since_last_adaptive_update = 0
            # Counter for count reset (slow: resets accumulated fail/success counts)
            self._steps_since_last_count_reset = 0

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Get joint positions for each environment from their respective motions."""
        return self._joint_pos_buf

    @property
    def joint_vel(self) -> torch.Tensor:
        """Get joint velocities for each environment from their respective motions."""
        return self._joint_vel_buf

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Get body positions for each environment from their respective motions."""
        return self._body_pos_w_buf + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Get body quaternions for each environment from their respective motions."""
        return self._body_quat_w_buf

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Get body linear velocities for each environment from their respective motions."""
        return self._body_lin_vel_w_buf

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Get body angular velocities for each environment from their respective motions."""
        return self._body_ang_vel_w_buf

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        # Select motion for each environment (adaptive or uniform/weighted)
        sampled_time_steps = None
        if self.cfg.enable_adaptive_sampling:
            # Adaptive sampling: Sample based on failure rates (curriculum learning)
            sampled = self._adaptive_sample_motions(env_ids)
            # _adaptive_sample_motions returns (motion_ids, time_steps) when frame-level sampling
            if isinstance(sampled, tuple) and len(sampled) == 2:
                sampled_motions, sampled_time_steps = sampled
            else:
                sampled_motions = sampled
            self.motion_ids[env_ids] = torch.as_tensor(sampled_motions, dtype=torch.long, device=self.device)
        elif self.cfg.sample_uniform:
            # Uniform sampling: All motions have equal probability
            self.motion_ids[env_ids] = torch.randint(0, self.num_motions, (len(env_ids),), device=self.device)
        else:
            # Weighted sampling: Use provided probabilities
            probs = torch.tensor(self.cfg.motion_probabilities, device=self.device)
            self.motion_ids[env_ids] = torch.multinomial(probs, len(env_ids), replacement=True)

        # If adaptive sampling provided explicit frame indices, use them. Otherwise sample random phase.
        selected_lengths = self.motion_lengths[self.motion_ids[env_ids]]
        if sampled_time_steps is None:
            # Sample random phase for each environment
            phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
            # Set time steps based on phase and motion length
            self.time_steps[env_ids] = (phase * (selected_lengths - 1)).long()
        else:
            # Elementwise clamp provided frame indices to the selected motion lengths
            sampled_ts = sampled_time_steps.to(self.device)
            sampled_ts = torch.where(sampled_ts < 0, torch.zeros_like(sampled_ts), sampled_ts)
            max_vals = (selected_lengths - 1)
            sampled_ts = torch.minimum(sampled_ts, max_vals)
            self.time_steps[env_ids] = sampled_ts.long()
        
        # Record the STARTING motion and frame for adaptive sampling tracking
        # These will be used to update failure/success counts when the episode ends
        self.start_motion_ids[env_ids] = self.motion_ids[env_ids].clone()
        self.start_time_steps[env_ids] = self.time_steps[env_ids].clone()

        # CRITICAL FIX: Update motion buffers BEFORE reading from them
        # This ensures robot reset uses correct motion data for newly sampled motion/phase
        self._update_motion_buffers_for_envs(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1

        # Interval-based adaptive sampling statistics update
        if self.cfg.enable_adaptive_sampling:
            self._steps_since_last_adaptive_update += 1
            self._steps_since_last_count_reset += 1
            
            # Fast update: apply EMA to update sampling probabilities
            if self._steps_since_last_adaptive_update >= self.cfg.adaptive_update_interval:
                self._update_adaptive_sampling_stats(reset_counts=False)
                self._steps_since_last_adaptive_update = 0
            
            # Slow reset: reset accumulated counts to prevent unbounded growth
            # and allow the system to "forget" old statistics
            if self._steps_since_last_count_reset >= self.cfg.adaptive_count_reset_interval:
                self._reset_adaptive_counts()
                self._steps_since_last_count_reset = 0

        # Vectorized reset logic: Check which environments need to reset based on their motion's length
        # Get the length for each environment's current motion
        current_lengths = self.motion_lengths[self.motion_ids]

        # Find which environments have exceeded their motion length
        reset_mask = self.time_steps >= current_lengths

        if reset_mask.any():
            env_ids = torch.where(reset_mask)[0]
            self._resample_command(env_ids)

        # Update cached buffers (CRITICAL: prevents memory leaks from repeated tensor creation)
        self._update_motion_buffers()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = anchor_pos_w_repeat - robot_anchor_pos_w_repeat
        delta_pos_w[..., :2] = 0.0
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = (
            robot_anchor_pos_w_repeat + delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
        )

    def _update_motion_buffers(self):
        """Update cached motion buffers using fully vectorized operations (no loops!)."""
        clamped_time_steps = torch.clamp(self.time_steps, 0, self.max_motion_length - 1)

        (
            joint_pos,
            joint_vel,
            body_pos,
            body_quat,
            body_lin_vel,
            body_ang_vel,
        ) = self._gather_motion_slices(self.motion_ids, clamped_time_steps)

        self._joint_pos_buf.copy_(joint_pos)
        self._joint_vel_buf.copy_(joint_vel)
        self._body_pos_w_buf.copy_(body_pos)
        self._body_quat_w_buf.copy_(body_quat)
        self._body_lin_vel_w_buf.copy_(body_lin_vel)
        self._body_ang_vel_w_buf.copy_(body_ang_vel)

    def _update_motion_buffers_for_envs(self, env_ids: torch.Tensor | Sequence[int]):
        """Update motion buffers for specific environments only (fully vectorized)."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return

        motion_ids_subset = self.motion_ids[env_ids]
        clamped_time_steps = torch.clamp(self.time_steps[env_ids], 0, self.max_motion_length - 1)

        (
            joint_pos,
            joint_vel,
            body_pos,
            body_quat,
            body_lin_vel,
            body_ang_vel,
        ) = self._gather_motion_slices(motion_ids_subset, clamped_time_steps)

        self._joint_pos_buf[env_ids] = joint_pos
        self._joint_vel_buf[env_ids] = joint_vel
        self._body_pos_w_buf[env_ids] = body_pos
        self._body_quat_w_buf[env_ids] = body_quat
        self._body_lin_vel_w_buf[env_ids] = body_lin_vel
        self._body_ang_vel_w_buf[env_ids] = body_ang_vel

    def _gather_motion_slices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor):
        """Fetch motion data slices from the library device and return them on env device."""
        if self.motion_library_device != self.device:
            motion_ids = motion_ids.to(self.motion_library_device, non_blocking=True)
            time_steps = time_steps.to(self.motion_library_device, non_blocking=True)

        joint_pos = self._stacked_joint_pos[motion_ids, time_steps]
        joint_vel = self._stacked_joint_vel[motion_ids, time_steps]
        body_pos = self._stacked_body_pos_w[motion_ids, time_steps]
        body_quat = self._stacked_body_quat_w[motion_ids, time_steps]
        body_lin_vel = self._stacked_body_lin_vel_w[motion_ids, time_steps]
        body_ang_vel = self._stacked_body_ang_vel_w[motion_ids, time_steps]

        if self.motion_library_device != self.device:
            joint_pos = joint_pos.to(self.device, non_blocking=True)
            joint_vel = joint_vel.to(self.device, non_blocking=True)
            body_pos = body_pos.to(self.device, non_blocking=True)
            body_quat = body_quat.to(self.device, non_blocking=True)
            body_lin_vel = body_lin_vel.to(self.device, non_blocking=True)
            body_ang_vel = body_ang_vel.to(self.device, non_blocking=True)

        return joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel

    def _adaptive_sample_motions(self, env_ids: torch.Tensor | Sequence[int]) -> Any:
        """Sample motions based on failure rates for curriculum learning."""
        import math
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        
        # Track which environments failed vs succeeded
        # CRITICAL: Use START motion/frame (not current) to track which starting conditions led to failure
        if hasattr(self._env, 'termination_manager'):
            episode_failed = self._env.termination_manager.terminated[env_ids]
            
            if episode_failed.any():
                # Increment failure counts for the STARTING motion/frame (not current frame!)
                # This tracks "which starting frames lead to failures"
                failed_start_motion_ids = self.start_motion_ids[env_ids][episode_failed]
                failed_counts = torch.bincount(failed_start_motion_ids, minlength=self.num_motions)
                self.current_failed_motion_count += failed_counts
                # If per-frame adaptive sampling is enabled, also increment per-frame counters
                if getattr(self.cfg, "enable_frame_adaptive_sampling", False):
                    failed_start_time_steps = self.start_time_steps[env_ids][episode_failed]
                    linear_idx = failed_start_motion_ids * self.max_motion_length + failed_start_time_steps
                    flat_counts = torch.bincount(linear_idx, minlength=self._total_frames)
                    self.current_failed_frame_count += flat_counts.view(self.num_motions, self.max_motion_length)
            
            # Success = not failed (includes timeouts and natural motion completion)
            episode_success = ~episode_failed
            if episode_success.any():
                success_start_motion_ids = self.start_motion_ids[env_ids][episode_success]
                success_counts = torch.bincount(success_start_motion_ids, minlength=self.num_motions)
                self.current_success_motion_count += success_counts
                if getattr(self.cfg, "enable_frame_adaptive_sampling", False):
                    success_start_time_steps = self.start_time_steps[env_ids][episode_success]
                    linear_idx = success_start_motion_ids * self.max_motion_length + success_start_time_steps
                    flat_counts = torch.bincount(linear_idx, minlength=self._total_frames)
                    self.current_success_frame_count += flat_counts.view(self.num_motions, self.max_motion_length)
        
        # Compute failure probability for each motion (motion-level)
        total_attempts = self.failed_motion_count + self.success_motion_count
        p_fail = self.failed_motion_count / (total_attempts + 1e-8)

        # If frame-level adaptive sampling is enabled, compute per-frame probabilities
        if getattr(self.cfg, "enable_frame_adaptive_sampling", False):
            # Create a mask for valid frames across all motions to avoid padded frames.
            motion_lengths_device = self.motion_lengths.to(self.device)
            valid_frames_mask = (
                torch.arange(self.max_motion_length, device=self.device).unsqueeze(0)
                < motion_lengths_device.unsqueeze(1)
            )
            valid_frames_mask_flat = valid_frames_mask.view(-1)

            # Compute per-frame failure probability: shape [num_motions, max_motion_length]
            pf = self.failed_frame_count / (self.failed_frame_count + self.success_frame_count + 1e-8)
            
            # Apply kernel smoothing if kernel_size > 1 (smooths failure probabilities across adjacent frames)
            if self._adaptive_kernel is not None:
                # Zero-out invalid frames before smoothing to prevent leakage from padded regions
                pf = pf.clone()
                pf[~valid_frames_mask] = 0.0
                
                # Conv1d expects [batch, channels, length] -> treat each motion as a batch item
                # pf shape: [num_motions, max_motion_length] -> [num_motions, 1, max_motion_length]
                pf_3d = pf.unsqueeze(1)
                padding = self.cfg.adaptive_kernel_size - 1
                # Use CONSTANT padding (0) to avoid smearing last-frame failures into void
                pf_padded = torch.nn.functional.pad(pf_3d, (0, padding), mode='constant', value=0.0)
                pf_smoothed = torch.nn.functional.conv1d(pf_padded, self._adaptive_kernel)
                pf = pf_smoothed.squeeze(1)  # Back to [num_motions, max_motion_length]
            
            # Flatten [num_motions * max_motion_length]
            pf_flat = pf.view(-1)

            # Zero-out invalid frames before weighting
            pf_flat = pf_flat.clone()
            pf_flat[~valid_frames_mask_flat] = 0.0

            # Exponential weighting
            pf_weighted = torch.pow(pf_flat, self.cfg.adaptive_beta)
            pf_weighted[~valid_frames_mask_flat] = 0.0
            
            # CRITICAL: Normalize pf_weighted to sum to 1.0 BEFORE mixing with uniform
            # This ensures the uniform_ratio actually controls the exploration percentage
            pf_weighted = pf_weighted / (pf_weighted.sum() + 1e-8)

            # Mix with uniform over valid frames (both now sum to 1.0)
            valid_frame_count = valid_frames_mask_flat.sum().clamp(min=1)
            total_frames = float(valid_frame_count)
            uniform_frame_prob = 1.0 / total_frames
            sampling_probs = (
                pf_weighted * (1.0 - self.cfg.adaptive_uniform_ratio)
                + uniform_frame_prob * self.cfg.adaptive_uniform_ratio
            )

            # Ensure invalid frames remain zero probability and renormalize (safety check)
            sampling_probs[~valid_frames_mask_flat] = 0.0
            sampling_probs = sampling_probs / (sampling_probs.sum() + 1e-8)

            # Sample flattened frame indices according to computed probabilities
            sampled_flat = torch.multinomial(sampling_probs, len(env_ids), replacement=True)

            # Convert to motion id and frame index
            sampled_motion_ids = (sampled_flat // self.max_motion_length).to(torch.long)
            sampled_time_steps = (sampled_flat % self.max_motion_length).to(torch.long)

            # Note: EMA update and counter reset moved to _update_adaptive_sampling_stats()
            # which is called at fixed intervals from _update_command()

            return sampled_motion_ids.to(self.device), sampled_time_steps.to(self.device)

        # Motion-level sampling (original behavior)
        # Apply exponential weighting (beta controls focus on hard motions)
        p_fail_weighted = torch.pow(p_fail, self.cfg.adaptive_beta)
        p_fail_weighted = p_fail_weighted / (p_fail_weighted.sum() + 1e-8)

        # Mix with uniform distribution to ensure all motions get some samples
        uniform_prob = 1.0 / self.num_motions
        sampling_probs = (
            p_fail_weighted * (1.0 - self.cfg.adaptive_uniform_ratio) +
            uniform_prob * self.cfg.adaptive_uniform_ratio
        )

        # Sample motions according to computed probabilities
        sampled_motion_ids = torch.multinomial(sampling_probs, len(env_ids), replacement=True)

        # Note: EMA update and counter reset moved to _update_adaptive_sampling_stats()
        # which is called at fixed intervals from _update_command()

        return sampled_motion_ids

    def _update_adaptive_sampling_stats(self, reset_counts: bool = False):
        """Update adaptive sampling statistics using EMA.
        
        This method is called at fixed intervals (adaptive_update_interval steps)
        to aggregate episode outcomes and update sampling probabilities.
        
        Args:
            reset_counts: If True, also reset the current_*_count accumulators.
                         Typically False, as count reset happens at a longer interval.
        """
        import math
        
        alpha = self.cfg.adaptive_alpha
        
        # EMA update of motion-level counts
        # This blends current batch statistics with historical statistics
        self.failed_motion_count = (
            alpha * self.current_failed_motion_count + 
            (1.0 - alpha) * self.failed_motion_count
        )
        self.success_motion_count = (
            alpha * self.current_success_motion_count + 
            (1.0 - alpha) * self.success_motion_count
        )
        
        # Update metrics for monitoring
        total_attempts = self.failed_motion_count + self.success_motion_count
        p_fail = self.failed_motion_count / (total_attempts + 1e-8)
        
        # Apply exponential weighting for sampling probability calculation
        p_fail_weighted = torch.pow(p_fail, self.cfg.adaptive_beta)
        p_fail_weighted = p_fail_weighted / (p_fail_weighted.sum() + 1e-8)
        
        uniform_prob = 1.0 / self.num_motions
        sampling_probs = (
            p_fail_weighted * (1.0 - self.cfg.adaptive_uniform_ratio) +
            uniform_prob * self.cfg.adaptive_uniform_ratio
        )
        
        H_samp = -(sampling_probs * torch.log(sampling_probs + 1e-12)).sum()
        H_samp_norm = H_samp / math.log(self.num_motions)
        
        p_fail_norm = p_fail / (p_fail.sum() + 1e-8)
        H_pfail = -(p_fail_norm * torch.log(p_fail_norm + 1e-12)).sum()
        H_pfail_norm = H_pfail / math.log(self.num_motions)
        
        pmax = sampling_probs.max()
        
        self.metrics["adaptive_sampling_entropy"][:] = H_samp_norm
        self.metrics["adaptive_pfail_entropy"][:] = H_pfail_norm
        self.metrics["adaptive_pfail_mean"][:] = p_fail.mean()
        self.metrics["adaptive_top1_prob"][:] = pmax
        self.metrics["total_failed_count"][:] = self.failed_motion_count.sum()
        self.metrics["total_success_count"][:] = self.success_motion_count.sum()
        
        # If frame-level adaptive sampling is enabled, also update per-frame counts
        if getattr(self.cfg, "enable_frame_adaptive_sampling", False):
            self.failed_frame_count = (
                alpha * self.current_failed_frame_count + 
                (1.0 - alpha) * self.failed_frame_count
            )
            self.success_frame_count = (
                alpha * self.current_success_frame_count + 
                (1.0 - alpha) * self.success_frame_count
            )
        
        # Only reset accumulators if explicitly requested (typically at longer intervals)
        if reset_counts:
            self._reset_adaptive_counts()

    def _reset_adaptive_counts(self):
        """Reset the current episode count accumulators.
        
        This is called at longer intervals (adaptive_count_reset_interval) to:
        1. Prevent unbounded accumulation of counts
        2. Allow the system to gradually "forget" old statistics
        3. Adapt to changing difficulty as training progresses
        """
        self.current_failed_motion_count.zero_()
        self.current_success_motion_count.zero_()
        
        if getattr(self.cfg, "enable_frame_adaptive_sampling", False):
            self.current_failed_frame_count.zero_()
            self.current_success_frame_count.zero_()

    def _chunk_all_motions(self, motions: list, human_motions: list, max_length: int):
        """Chunk motions longer than max_length into smaller clips."""
        import types
        chunked_motions = []
        chunked_humans = []
        
        for motion, human in zip(motions, human_motions):
            if motion.time_step_total <= max_length:
                chunked_motions.append(motion)
                chunked_humans.append(human)
            else:
                num_chunks = (motion.time_step_total + max_length - 1) // max_length
                for i in range(num_chunks):
                    start = i * max_length
                    end = min((i + 1) * max_length, motion.time_step_total)
                    
                    chunk = types.SimpleNamespace(
                        fps=motion.fps,
                        joint_pos=motion.joint_pos[start:end],
                        joint_vel=motion.joint_vel[start:end],
                        body_pos_w=motion.body_pos_w[start:end],
                        body_quat_w=motion.body_quat_w[start:end],
                        body_lin_vel_w=motion.body_lin_vel_w[start:end],
                        body_ang_vel_w=motion.body_ang_vel_w[start:end],
                        time_step_total=end - start,
                    )
                    chunked_motions.append(chunk)
                    
                    if human is not None:
                        h_end = min(end, human["time_step_total"])
                        h_start = min(start, human["time_step_total"])
                        chunked_humans.append({
                            "joint_pos": human["joint_pos"][h_start:h_end],
                            "fps": human["fps"],
                            "time_step_total": max(1, h_end - h_start),
                        })
                    else:
                        chunked_humans.append(None)
        
        return chunked_motions, chunked_humans

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )
                
                if self.cfg.human_visualizer_cfg:
                    self.human_visualizer = VisualizationMarkers(self.cfg.human_visualizer_cfg)

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)
            
            if hasattr(self, "human_visualizer"):
                self.human_visualizer.set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
            
            if hasattr(self, "human_visualizer"):
                self.human_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])
        
        if hasattr(self, "human_visualizer") and self._stacked_human_joint_pos is not None:
            motion_ids = self.motion_ids.to(self.motion_library_device)
            human_time_steps = (self.time_steps * 0.6).long()
            clamped_time_steps = torch.clamp(
                human_time_steps.to(self.motion_library_device), 0, self.max_human_motion_length - 1
            )
            human_pos = self._stacked_human_joint_pos[motion_ids, clamped_time_steps]
            human_pos = human_pos.to(self.device)
            human_pos_w = human_pos + self._env.scene.env_origins[:, None, :]
            flat_pos = human_pos_w.view(-1, 3)
            flat_quat = torch.zeros(flat_pos.shape[0], 4, device=self.device)
            flat_quat[:, 0] = 1.0
            self.human_visualizer.visualize(flat_pos, flat_quat)

    def get_sonic_robot_window(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get future robot motion window (10 frames, dt=0.1s) for SONIC."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        motion_ids = self.motion_ids[env_ids].to(self.motion_library_device)
        current_steps = self.time_steps[env_ids].to(self.motion_library_device)
        
        # Params
        dt = 0.1
        fps = 50.0 # Assume 50Hz or retrieve from metadata
        stride = int(dt * fps) # 5 frames
        window_size = 10
        
        offsets = torch.arange(window_size, device=self.motion_library_device) * stride
        indices = current_steps.unsqueeze(1) + offsets.unsqueeze(0) # [B, 10]
        
        lengths = self.motion_lengths[motion_ids].unsqueeze(1)
        indices = torch.clamp(indices, max=lengths - 1)
        
        batch_motion_ids = motion_ids.unsqueeze(1).expand(-1, window_size)
        
        j_pos = self._stacked_joint_pos[batch_motion_ids, indices] 
        j_vel = self._stacked_joint_vel[batch_motion_ids, indices]
        
        res = torch.cat([j_pos, j_vel], dim=-1)
        res = res.view(env_ids.shape[0], -1)
        
        return res.to(self.device)

    def get_sonic_human_window(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get future human motion window (10 frames, dt=0.02s) for SONIC."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        dt = 0.02
        fps = 50.0 
        motion_ids = self.motion_ids[env_ids].to(self.motion_library_device)
        current_steps = self.time_steps[env_ids].to(self.motion_library_device)
        human_motion_current_steps = current_steps*0.6 # Assume synchronized for now

        stride = max(1, int(dt * fps)) # 1 frame
        window_size = 10
        
        offsets = torch.arange(window_size, device=self.motion_library_device) * stride
        indices = human_motion_current_steps.unsqueeze(1) + offsets.unsqueeze(0)
        
        lengths = self.human_motion_lengths[motion_ids].unsqueeze(1)
        indices = torch.clamp(indices, max=lengths - 1).int()
        
        batch_motion_ids = motion_ids.unsqueeze(1).expand(-1, window_size)
        
        # if self._stacked_human_joint_pos is None:
        #     return torch.zeros(env_ids.shape[0], 10 * 24 * 3, device=self.device)
             
        h_pos = self._stacked_human_joint_pos[batch_motion_ids, indices]
        res = h_pos.view(env_ids.shape[0], -1)
        return res.to(self.device)

@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the multi-motion command that supports multiple motion files."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING  # type: ignore[assignment]

    motion_files: list[str] = MISSING  # type: ignore[assignment]
    anchor_body: str = MISSING  # type: ignore[assignment]
    body_names: list[str] = MISSING  # type: ignore[assignment]

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # Sampling strategy configuration
    sample_uniform: bool = True
    # Probabilities for each motion (only used if sample_uniform is False and adaptive sampling is disabled)
    motion_probabilities: list[float] = []
    
    # Adaptive sampling configuration (curriculum learning)
    enable_adaptive_sampling: bool = False
    # If True, adaptively sample specific frames (not just whole motions)
    # so the system will prefer starting frames that have higher failure rates.
    enable_frame_adaptive_sampling: bool = False
    adaptive_beta: float = 0.5  # Exponent for failure probability (0=uniform, 1=proportional, >1=focus on hard)
    adaptive_alpha: float = 0.001  # EMA smoothing factor for counts (lower=more stable, higher=more responsive)
    adaptive_uniform_ratio: float = 0.1  # Minimum uniform sampling probability (prevents zero probability)
    adaptive_update_interval: int = 240  # How often (in steps) to update sampling probabilities via EMA
    adaptive_count_reset_interval: int = 24000  # How often (in steps) to reset accumulated counts (longer = more memory)
    # Kernel smoothing for frame-level adaptive sampling (smooths failure probabilities across adjacent frames)
    adaptive_kernel_size: int = 50  # Kernel size for conv1d smoothing (1=no smoothing, >1=applies exponential kernel)
    adaptive_lambda: float = 0.8  # Decay factor for exponential kernel (higher=more smoothing across frames)

    human_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/human_pose",
        markers={
            "joint": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Memory optimization options for motion library
    motion_library_device: str = "cuda"  # "cuda" for GPU (fast), "cpu" for CPU (saves GPU VRAM)
    motion_library_mmap: bool = False  # True to use memory-mapped files (reduces RAM usage)
    motion_library_fp16: bool = False  # True to use float16 (50% memory reduction)
    
    # Chunk long motions to avoid multinomial 2^24 limit
    motion_chunk_length: int = 0  # Max frames per motion chunk (0 = no chunking)


class SingleMotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class SingleMotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = SingleMotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class SingleMotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = SingleMotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class SyncedStudentMotionCommand(SingleMotionCommand):
    """A motion command that syncs its time_steps from a primary motion command.

    This is used for distillation where the teacher sees one motion clip (e.g., stair climbing)
    and the student sees a different clip (e.g., flat walking), but both must stay in phase.
    The primary command handles resampling and robot state resets; this command only copies
    time_steps and recomputes its own motion buffers.
    """

    cfg: SyncedStudentMotionCommandCfg

    def _get_primary_command(self) -> SingleMotionCommand:
        return self._env.command_manager.get_term(self.cfg.sync_command_name)

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        if len(env_ids) == 0:
            return
        # Copy time_steps from primary command (it already resampled and reset robot state)
        primary = self._get_primary_command()
        self.time_steps[:] = primary.time_steps

    def _update_command(self):
        # Copy time_steps from primary command (it already incremented and resampled)
        primary = self._get_primary_command()
        self.time_steps[:] = primary.time_steps

        # Recompute relative body poses for this clip's motion data
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _update_metrics(self):
        # No-op: metrics are tracked by the primary motion command
        pass


@configclass
class SyncedStudentMotionCommandCfg(SingleMotionCommandCfg):
    """Configuration for a synced student motion command.

    This command syncs its time_steps from a primary motion command specified by
    ``sync_command_name``. Used for distillation with two different motion clips.
    """

    class_type: type = SyncedStudentMotionCommand

    sync_command_name: str = "motion"
    """Name of the primary motion command to sync time_steps from."""