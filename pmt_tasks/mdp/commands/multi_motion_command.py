"""Clean multi-motion command implementation with separated concerns.

This module provides a refactored motion tracking system with:
- MotionDataStore: Handles loading, stacking, and device management
- MotionSampler: Strategy pattern for motion/frame sampling
- MultiMotionCommandV2: Clean orchestration class

Design Principles:
1. Separation of concerns - data loading, sampling, and command logic are independent
2. Composition over inheritance - samplers and data stores are injected
3. Vectorized operations - all batch operations are fully vectorized
4. Memory-efficient - supports CPU storage with GPU transfer on-demand
[ Configuration (Cfg) ]
                                            |
                                            v
+=======================================================================================+
|                        MultiMotionCommandV2 (The Conductor)                           |
|  Inherits: CommandTerm                                                                |
|                                                                                       |
|   +-----------------------------+               +---------------------------------+   |
|   |      MotionDataStore        |               |          MotionSampler          |   |
|   |      (The Warehouse)        |               |           (The Brain)           |   |
|   |                             |               |                                 |   |
|   |  [1. Initialization]        |               |  [Strategy]                     |   |
|   |   - Loads .npz Files        |               |   - Uniform / Adaptive          |   |
|   |   - Chunks Long Motions     |               |   - Kernel Smoothing (Lookback) |   |
|   |   - Stacks Tensors (CPU/GPU)|               |                                 |   |
|   |   - Releases Raw Memory     |               |  [State Tracking]               |   |
|   |                             |               |   - Failure Heatmap (History)   |   |
|   |  [2. Vectorized Access]     |               |   - Success/Fail Counters       |   |
|   |   -> get_motion_data()      |               |                                 |   |
|   |   -> get_sonic_window()     |               |   -> sample(num_envs)           |   |
|   +--------------+--------------+               |   -> update(terminated_mask)    |   |
|                  |                              +---------------+-----------------+   |
|                  |                                 ^            |                     |
|                  | (3) Request Data                | (2) Update | (1) Request IDs     |
|                  |     for IDs                     |     Stats  |     (Motion/Frame)  |
|                  v                                 |            v                     |
|   +-----------------------------+                  |    +----------------+            |
|   |     Orchestration Logic     |------------------+----| Internal State |            |
|   |                             |                       +----------------+            |
|   |  [Buffers]                  |                        - motion_ids                 |
|   |   - joint_pos_buf           |                        - frame_ids                  |
|   |   - body_pos_relative_w     |                        - start_ids (for update)     |
|   |                             |                                                     |
|   |  [Critical Flows]           |                                                     |
|   |   A. reset(env_ids)  -----> Trigger Sampler -> Get Data -> **Teleport Robot** |
|   |   B. compute(dt)     -----> Step Frame -> Get Data -> **Update Buffers** |
|   +--------------+--------------+                                                     |
+==================+====================================================================+
                   |
                   | (4) Output: Target References & Debug Viz
                   v
+------------------+------------------+
|      Isaac Lab Environment          |
|                                     |
|  - Robot (Articulation)             |
|  - Termination Manager  ------------+ (5) Feedback: Terminated / Failed?
+-------------------------------------+


"""



from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import Tensor

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from .commands import MotionCommand, MotionCommandCfg
from isaaclab.utils import configclass
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


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MotionData:
    """Container for a single motion's data.
    
    Attributes:
        fps: Frame rate of the motion
        joint_pos: Joint positions [T, num_joints] (None after stacking to save memory)
        joint_vel: Joint velocities [T, num_joints] (None after stacking)
        body_pos_w: Body positions in world frame [T, num_bodies, 3] (None after stacking)
        body_quat_w: Body quaternions in world frame [T, num_bodies, 4] (None after stacking)
        body_lin_vel_w: Body linear velocities [T, num_bodies, 3] (None after stacking)
        body_ang_vel_w: Body angular velocities [T, num_bodies, 3] (None after stacking)
        num_frames: Total number of frames
        metadata: Optional metadata (cube configs, terrain info, etc.)
        source_file: Path to source file for debugging
        
    Note:
        After MotionDataStore._stack_motions() is called, individual tensor fields
        are set to None to prevent memory doubling. Only metadata is preserved.
    """
    fps: float
    joint_pos: Tensor | None
    joint_vel: Tensor | None
    body_pos_w: Tensor | None
    body_quat_w: Tensor | None
    body_lin_vel_w: Tensor | None
    body_ang_vel_w: Tensor | None
    num_frames: int
    metadata: dict = field(default_factory=dict)
    source_file: str = ""
    
    # Optional paired human motion data (for SONIC)
    human_joint_pos: Tensor | None = None
    human_num_frames: int = 0
    human_fps: float = 50.0  # Human motion frame rate (typically 30Hz)


@dataclass
class SamplingResult:
    """Result of a sampling operation.
    
    Attributes:
        motion_ids: Selected motion indices [num_samples]
        frame_ids: Selected frame indices [num_samples]
    """
    motion_ids: Tensor
    frame_ids: Tensor


# =============================================================================
# Motion Data Store
# =============================================================================


class MotionDataStore:
    """Manages loading, stacking, and retrieval of motion data.
    
    Features:
    - Supports loading from npz files with optional metadata
    - Stacks data into contiguous tensors for vectorized access
    - Supports CPU storage with on-demand GPU transfer
    - Supports optional chunking of long motions
    - Memory-efficient fp16 option
    
    Example:
        >>> store = MotionDataStore(device="cuda", storage_device="cpu")
        >>> store.load_files(file_paths, body_indices)
        >>> joint_pos, joint_vel = store.get_joint_data(motion_ids, frame_ids)
    """
    
    def __init__(
        self,
        device: str | torch.device = "cuda",
        storage_device: str | torch.device = "cuda",
        use_fp16: bool = False,
        chunk_length: int = 0,
    ):
        """Initialize the data store.
        
        Args:
            device: Device for computation (typically GPU)
            storage_device: Device for storing stacked data (CPU for large datasets)
            use_fp16: If True, store data in float16 for 50% memory reduction
            chunk_length: If > 0, split motions longer than this into chunks
        """
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.chunk_length = chunk_length
        
        # Motion data (populated by load_files)
        self.motions: list[MotionData] = []
        self.num_motions: int = 0
        self.motion_lengths: Tensor | None = None
        self.max_motion_length: int = 0
        
        # Dimensions (set during stacking)
        self.num_joints: int = 0
        self.num_bodies: int = 0
        
        # Stacked tensors for vectorized access
        self._stacked_joint_pos: Tensor | None = None
        self._stacked_joint_vel: Tensor | None = None
        self._stacked_body_pos_w: Tensor | None = None
        self._stacked_body_quat_w: Tensor | None = None
        self._stacked_body_lin_vel_w: Tensor | None = None
        self._stacked_body_ang_vel_w: Tensor | None = None
        
        # Human motion data (optional)
        self._stacked_human_joint_pos: Tensor | None = None
        self.human_motion_lengths: Tensor | None = None
        self.max_human_motion_length: int = 0
    
    def load_files(
        self,
        file_paths: list[str],
        body_indices: Tensor | list[int],
        load_human: bool = False,
        human_path_transform: callable = None,
    ) -> None:
        """Load motion files and stack into tensors.
        
        Args:
            file_paths: List of paths to npz motion files
            body_indices: Indices of bodies to extract from motion data
            load_human: If True, also load paired human motion data
            human_path_transform: Function to transform robot path to human path
                                  Default: replace 'robot_' with 'human_'
        """
        if not file_paths:
            print("[MotionDataStore] No files to load")
            return
        
        body_indices = torch.as_tensor(body_indices, dtype=torch.long, device=self.storage_device)
        
        if human_path_transform is None:
            human_path_transform = lambda p: p.replace("robot_", "human_")
        
        # Load all motions
        self.motions = []
        for file_path in file_paths:
            motion = self._load_single_motion(file_path, body_indices, load_human, human_path_transform)
            if motion is not None:
                # Allow subclasses to post-process motion data (e.g., extract cube configs)
                motion = self.post_process_motion_data(motion)
                if motion is not None:
                    self.motions.append(motion)
        
        # Apply chunking if configured
        if self.chunk_length > 0:
            self.motions = self._chunk_motions(self.motions)
        
        self.num_motions = len(self.motions)
        if self.num_motions == 0:
            print("[MotionDataStore] Warning: No motions loaded")
            return
        
        # Compute lengths
        self.motion_lengths = torch.tensor(
            [m.num_frames for m in self.motions], dtype=torch.long, device=self.device
        )
        self.max_motion_length = max(m.num_frames for m in self.motions)
        
        # Human motion lengths
        if load_human:
            self.human_motion_lengths = torch.tensor(
                [m.human_num_frames if m.human_joint_pos is not None else 1 for m in self.motions],
                dtype=torch.long, device=self.device
            )
            # Use actual tensor sizes (more reliable than stored human_num_frames)
            valid_human = [m for m in self.motions if m.human_joint_pos is not None and m.human_joint_pos.shape[0] > 0]
            self.max_human_motion_length = max((m.human_joint_pos.shape[0] for m in valid_human), default=0)
        
        # Stack into contiguous tensors
        self._stack_motions()
        
        print(f"[MotionDataStore] Loaded {self.num_motions} motions, max_length={self.max_motion_length}")
    
    def post_process_motion_data(self, motion: MotionData) -> MotionData | None:
        """Hook for subclasses to post-process loaded motion data.
        
        Override this method to extract additional metadata, validate data,
        or filter out invalid motions.
        
        Args:
            motion: The loaded motion data
            
        Returns:
            The (possibly modified) motion data, or None to skip this motion
            
        Example:
            def post_process_motion_data(self, motion):
                # Extract cube configurations from metadata
                if "cube1_size" in motion.metadata:
                    motion.metadata["cube_configs"] = self._parse_cube_configs(motion.metadata)
                return motion
        """
        return motion
    
    def _load_single_motion(
        self,
        file_path: str,
        body_indices: Tensor,
        load_human: bool,
        human_path_transform: callable,
    ) -> MotionData | None:
        """Load a single motion file."""
        if not os.path.isfile(file_path):
            print(f"[MotionDataStore] File not found: {file_path}")
            return None
        
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"[MotionDataStore] Failed to load {file_path}: {e}")
            return None
        
        # Core motion data
        fps = float(data["fps"])

        joint_pos = torch.tensor(data["joint_pos"], dtype=self.dtype, device=self.storage_device)
        joint_vel = torch.tensor(data["joint_vel"], dtype=self.dtype, device=self.storage_device)
        
        # Body data with index selection
        body_pos_w = torch.tensor(data["body_pos_w"], dtype=self.dtype, device=self.storage_device)
        body_quat_w = torch.tensor(data["body_quat_w"], dtype=self.dtype, device=self.storage_device)
        body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=self.dtype, device=self.storage_device)
        body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=self.dtype, device=self.storage_device)
        
        # Select only the bodies we need
        body_indices_storage = body_indices.to(self.storage_device)
        body_pos_w = body_pos_w[:, body_indices_storage]
        body_quat_w = body_quat_w[:, body_indices_storage]
        body_lin_vel_w = body_lin_vel_w[:, body_indices_storage]
        body_ang_vel_w = body_ang_vel_w[:, body_indices_storage]
        
        num_frames = joint_pos.shape[0]
        
        # Extract metadata (any key not in the core set)
        core_keys = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", 
                     "body_lin_vel_w", "body_ang_vel_w", "positions"}
        metadata = {}
        for key in data.keys():
            if key not in core_keys:
                metadata[key] = data[key]
        
        # Load human motion if requested
        human_joint_pos = None
        human_num_frames = 0
        human_fps = 30.0  # Default human fps
        if load_human:
            human_path = human_path_transform(file_path)
            if os.path.isfile(human_path):
                try:
                    h_data = np.load(human_path)
                    if "positions" in h_data:
                        human_joint_pos = torch.tensor(
                            h_data["positions"], dtype=torch.float32, device=self.storage_device
                        )
                        human_num_frames = human_joint_pos.shape[0]
                        # Extract human fps if available, otherwise default to 30Hz
                        human_fps = float(h_data.get("fps", 30.0))
                except Exception as e:
                    print(f"[MotionDataStore] Failed to load human motion {human_path}: {e}")
        
        return MotionData(
            fps=fps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            num_frames=num_frames,
            metadata=metadata,
            source_file=file_path,
            human_joint_pos=human_joint_pos,
            human_num_frames=human_num_frames,
            human_fps=human_fps,
        )
    
    def _chunk_motions(self, motions: list[MotionData]) -> list[MotionData]:
        """Split long motions into smaller chunks."""
        chunked = []
        for motion in motions:
            if motion.num_frames <= self.chunk_length:
                chunked.append(motion)
            else:
                # Chunking happens BEFORE stacking, so tensors must exist
                assert motion.joint_pos is not None, "Cannot chunk motion with released tensors"
                assert motion.joint_vel is not None, "Cannot chunk motion with released tensors"
                assert motion.body_pos_w is not None, "Cannot chunk motion with released tensors"
                assert motion.body_quat_w is not None, "Cannot chunk motion with released tensors"
                assert motion.body_lin_vel_w is not None, "Cannot chunk motion with released tensors"
                assert motion.body_ang_vel_w is not None, "Cannot chunk motion with released tensors"
                
                num_chunks = (motion.num_frames + self.chunk_length - 1) // self.chunk_length
                for i in range(num_chunks):
                    start = i * self.chunk_length
                    end = min((i + 1) * self.chunk_length, motion.num_frames)
                    
                    chunk = MotionData(
                        fps=motion.fps,
                        joint_pos=motion.joint_pos[start:end],
                        joint_vel=motion.joint_vel[start:end],
                        body_pos_w=motion.body_pos_w[start:end],
                        body_quat_w=motion.body_quat_w[start:end],
                        body_lin_vel_w=motion.body_lin_vel_w[start:end],
                        body_ang_vel_w=motion.body_ang_vel_w[start:end],
                        num_frames=end - start,
                        metadata=motion.metadata,  # Shared metadata
                        source_file=f"{motion.source_file}[chunk_{i}]",
                    )
                    
                    # Chunk human motion if present
                    if motion.human_joint_pos is not None and motion.human_num_frames > 0:
                        # Calculate human frame indices using fps ratio
                        ratio = motion.human_fps / motion.fps if motion.fps > 0 else 0.6
                        h_start = int(start * ratio)
                        h_end = int(end * ratio)
                        h_start = min(h_start, motion.human_num_frames)
                        h_end = min(h_end, motion.human_num_frames)
                        h_length = h_end - h_start
                        if h_length > 0:
                            chunk.human_joint_pos = motion.human_joint_pos[h_start:h_end]
                            chunk.human_num_frames = h_length
                            chunk.human_fps = motion.human_fps
                        else:
                            # No human frames in this chunk
                            chunk.human_joint_pos = None
                            chunk.human_num_frames = 0
                    
                    chunked.append(chunk)
        
        return chunked
    
    def _stack_motions(self) -> None:
        """Stack all motions into contiguous tensors for vectorized access.
        
        IMPORTANT: After stacking, individual motion tensors are released to
        prevent memory doubling. Only lightweight metadata is kept in self.motions.
        """
        if not self.motions:
            return
        
        # Get dimensions from first motion (tensors exist before stacking)
        first_motion = self.motions[0]
        assert first_motion.joint_pos is not None, "Motion data already released"
        assert first_motion.body_pos_w is not None, "Motion data already released"
        self.num_joints = first_motion.joint_pos.shape[1]
        self.num_bodies = first_motion.body_pos_w.shape[1]
        num_joints = self.num_joints
        num_bodies = self.num_bodies
        
        # Allocate stacked tensors
        self._stacked_joint_pos = torch.zeros(
            self.num_motions, self.max_motion_length, num_joints,
            dtype=self.dtype, device=self.storage_device
        )
        self._stacked_joint_vel = torch.zeros(
            self.num_motions, self.max_motion_length, num_joints,
            dtype=self.dtype, device=self.storage_device
        )
        self._stacked_body_pos_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 3,
            dtype=self.dtype, device=self.storage_device
        )
        self._stacked_body_quat_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 4,
            dtype=self.dtype, device=self.storage_device
        )
        self._stacked_body_lin_vel_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 3,
            dtype=self.dtype, device=self.storage_device
        )
        self._stacked_body_ang_vel_w = torch.zeros(
            self.num_motions, self.max_motion_length, num_bodies, 3,
            dtype=self.dtype, device=self.storage_device
        )
        
        # Fill in data and RELEASE individual tensors to prevent memory doubling
        for i, motion in enumerate(self.motions):
            # Tensors must exist before stacking
            assert motion.joint_pos is not None, f"Motion {i} joint_pos is None"
            assert motion.joint_vel is not None, f"Motion {i} joint_vel is None"
            assert motion.body_pos_w is not None, f"Motion {i} body_pos_w is None"
            assert motion.body_quat_w is not None, f"Motion {i} body_quat_w is None"
            assert motion.body_lin_vel_w is not None, f"Motion {i} body_lin_vel_w is None"
            assert motion.body_ang_vel_w is not None, f"Motion {i} body_ang_vel_w is None"
            
            length = motion.num_frames
            self._stacked_joint_pos[i, :length] = motion.joint_pos
            self._stacked_joint_vel[i, :length] = motion.joint_vel
            self._stacked_body_pos_w[i, :length] = motion.body_pos_w
            self._stacked_body_quat_w[i, :length] = motion.body_quat_w
            self._stacked_body_lin_vel_w[i, :length] = motion.body_lin_vel_w
            self._stacked_body_ang_vel_w[i, :length] = motion.body_ang_vel_w
            
            # Release individual tensors after copying to stacked storage
            # Keep only metadata (fps, num_frames, source_file, metadata dict)
            motion.joint_pos = None
            motion.joint_vel = None
            motion.body_pos_w = None
            motion.body_quat_w = None
            motion.body_lin_vel_w = None
            motion.body_ang_vel_w = None
        
        # Stack human motion if present
        if self.max_human_motion_length > 0:
            valid_human = [m for m in self.motions if m.human_joint_pos is not None]
            if valid_human and valid_human[0].human_joint_pos is not None:
                h_pos = valid_human[0].human_joint_pos
                h_num_joints = h_pos.shape[1]
                h_dim = h_pos.shape[2] if len(h_pos.shape) > 2 else 3
                
                self._stacked_human_joint_pos = torch.zeros(
                    self.num_motions, self.max_human_motion_length, h_num_joints, h_dim,
                    dtype=torch.float32, device=self.storage_device
                )
                
                for i, motion in enumerate(self.motions):
                    if motion.human_joint_pos is not None and motion.human_num_frames > 0:
                        # Get actual tensor size (more reliable than stored num_frames)
                        actual_frames = motion.human_joint_pos.shape[0]
                        if actual_frames > 0:
                            length = min(actual_frames, self.max_human_motion_length)
                            self._stacked_human_joint_pos[i, :length] = motion.human_joint_pos[:length]
                        # Release human motion tensor after copying
                        motion.human_joint_pos = None
        
        # Store FPS ratio for human/robot motion alignment (human_fps / robot_fps)
        # This allows correct temporal alignment regardless of dataset-specific frame rates
        self._stacked_fps_ratio = torch.zeros(self.num_motions, device=self.storage_device)
        for i, motion in enumerate(self.motions):
            # Default ratio 0.6 = 30Hz human / 50Hz robot
            ratio = motion.human_fps / motion.fps if motion.fps > 0 else 0.6
            self._stacked_fps_ratio[i] = ratio
        
        # Force garbage collection to reclaim memory immediately
        import gc
        gc.collect()
        if self.storage_device.type == "cuda":
            torch.cuda.empty_cache()
        
        print(f"[MotionDataStore] Stacked data and released individual tensors to save memory")
    
    def get_motion_data(
        self,
        motion_ids: Tensor,
        frame_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get motion data for given motion and frame indices (vectorized).
        
        Args:
            motion_ids: Motion indices [batch_size]
            frame_ids: Frame indices [batch_size]
        
        Returns:
            Tuple of (joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel)
            Each tensor has shape [batch_size, ...]
        """
        # Move indices to storage device if needed
        if motion_ids.device != self.storage_device:
            motion_ids = motion_ids.to(self.storage_device, non_blocking=True)
            frame_ids = frame_ids.to(self.storage_device, non_blocking=True)
        
        # Clamp frame indices
        frame_ids = torch.clamp(frame_ids, 0, self.max_motion_length - 1)
        
        # Vectorized gather (stacked tensors are guaranteed non-None after load_files)
        assert self._stacked_joint_pos is not None, "Motion data not loaded"
        assert self._stacked_joint_vel is not None, "Motion data not loaded"
        assert self._stacked_body_pos_w is not None, "Motion data not loaded"
        assert self._stacked_body_quat_w is not None, "Motion data not loaded"
        assert self._stacked_body_lin_vel_w is not None, "Motion data not loaded"
        assert self._stacked_body_ang_vel_w is not None, "Motion data not loaded"
        
        joint_pos = self._stacked_joint_pos[motion_ids, frame_ids]
        joint_vel = self._stacked_joint_vel[motion_ids, frame_ids]
        body_pos = self._stacked_body_pos_w[motion_ids, frame_ids]
        body_quat = self._stacked_body_quat_w[motion_ids, frame_ids]
        body_lin_vel = self._stacked_body_lin_vel_w[motion_ids, frame_ids]
        body_ang_vel = self._stacked_body_ang_vel_w[motion_ids, frame_ids]
        
        # Move to compute device if needed
        if self.storage_device != self.device:
            joint_pos = joint_pos.to(self.device, non_blocking=True)
            joint_vel = joint_vel.to(self.device, non_blocking=True)
            body_pos = body_pos.to(self.device, non_blocking=True)
            body_quat = body_quat.to(self.device, non_blocking=True)
            body_lin_vel = body_lin_vel.to(self.device, non_blocking=True)
            body_ang_vel = body_ang_vel.to(self.device, non_blocking=True)
        
        return joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel
    
    def get_human_motion_data(
        self,
        motion_ids: Tensor,
        frame_ids: Tensor,
    ) -> Tensor | None:
        """Get human motion data for given motion and frame indices."""
        if self._stacked_human_joint_pos is None or self.max_human_motion_length == 0:
            return None
        
        if motion_ids.device != self.storage_device:
            motion_ids = motion_ids.to(self.storage_device, non_blocking=True)
            frame_ids = frame_ids.to(self.storage_device, non_blocking=True)
        
        frame_ids = torch.clamp(frame_ids, 0, max(0, self.max_human_motion_length - 1))
        human_pos = self._stacked_human_joint_pos[motion_ids, frame_ids]
        
        if self.storage_device != self.device:
            human_pos = human_pos.to(self.device, non_blocking=True)
        
        return human_pos
    
    def get_motion_window(
        self,
        motion_ids: Tensor,
        start_frames: Tensor,
        window_size: int,
        stride: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Get a window of future motion data (for SONIC encoder).
        
        Args:
            motion_ids: Motion indices [batch_size]
            start_frames: Starting frame indices [batch_size]
            window_size: Number of frames in the window
            stride: Frame stride (e.g., 5 for 0.1s at 50Hz)
        
        Returns:
            Tuple of (joint_pos, joint_vel) each with shape [batch_size, window_size, ...]
        """
        if motion_ids.device != self.storage_device:
            motion_ids = motion_ids.to(self.storage_device, non_blocking=True)
            start_frames = start_frames.to(self.storage_device, non_blocking=True)
        
        # Create frame offsets: [0, stride, 2*stride, ...]
        offsets = torch.arange(window_size, device=self.storage_device) * stride
        
        # Expand to [batch_size, window_size]
        frame_indices = start_frames.unsqueeze(1) + offsets.unsqueeze(0)
        
        # Clamp to motion lengths
        assert self.motion_lengths is not None, "Motion data not loaded"
        lengths = self.motion_lengths[motion_ids].unsqueeze(1).to(self.storage_device)
        frame_indices = torch.clamp(frame_indices, min=0)
        frame_indices = torch.minimum(frame_indices, lengths - 1)
        
        # Expand motion_ids to [batch_size, window_size]
        batch_motion_ids = motion_ids.unsqueeze(1).expand(-1, window_size)
        
        # Gather
        assert self._stacked_joint_pos is not None, "Motion data not loaded"
        assert self._stacked_joint_vel is not None, "Motion data not loaded"
        joint_pos = self._stacked_joint_pos[batch_motion_ids, frame_indices]
        joint_vel = self._stacked_joint_vel[batch_motion_ids, frame_indices]
        
        if self.storage_device != self.device:
            joint_pos = joint_pos.to(self.device, non_blocking=True)
            joint_vel = joint_vel.to(self.device, non_blocking=True)
        
        return joint_pos, joint_vel

    def get_motion_window_full(
        self,
        motion_ids: Tensor,
        start_frames: Tensor,
        window_size: int,
        stride: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get a window of motion data (joints + bodies) starting from start_frames.

        Args:
            motion_ids: Motion indices [batch_size]
            start_frames: Starting frame indices [batch_size]
            window_size: Number of frames in the window
            stride: Frame stride (e.g., 5 for 0.1s at 50Hz)

        Returns:
            Tuple of (joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w)
            Each tensor has shape [batch_size, window_size, ...]
        """
        if motion_ids.device != self.storage_device:
            motion_ids = motion_ids.to(self.storage_device, non_blocking=True)
            start_frames = start_frames.to(self.storage_device, non_blocking=True)

        offsets = torch.arange(window_size, device=self.storage_device) * stride
        frame_indices = start_frames.unsqueeze(1) + offsets.unsqueeze(0)

        assert self.motion_lengths is not None, "Motion data not loaded"
        lengths = self.motion_lengths[motion_ids].unsqueeze(1).to(self.storage_device)
        frame_indices = torch.clamp(frame_indices, min=0)
        frame_indices = torch.minimum(frame_indices, lengths - 1)

        batch_motion_ids = motion_ids.unsqueeze(1).expand(-1, window_size)

        assert self._stacked_joint_pos is not None, "Motion data not loaded"
        assert self._stacked_joint_vel is not None, "Motion data not loaded"
        assert self._stacked_body_pos_w is not None, "Motion data not loaded"
        assert self._stacked_body_quat_w is not None, "Motion data not loaded"
        assert self._stacked_body_lin_vel_w is not None, "Motion data not loaded"
        assert self._stacked_body_ang_vel_w is not None, "Motion data not loaded"

        joint_pos = self._stacked_joint_pos[batch_motion_ids, frame_indices]
        joint_vel = self._stacked_joint_vel[batch_motion_ids, frame_indices]
        body_pos_w = self._stacked_body_pos_w[batch_motion_ids, frame_indices]
        body_quat_w = self._stacked_body_quat_w[batch_motion_ids, frame_indices]
        body_lin_vel_w = self._stacked_body_lin_vel_w[batch_motion_ids, frame_indices]
        body_ang_vel_w = self._stacked_body_ang_vel_w[batch_motion_ids, frame_indices]

        if self.storage_device != self.device:
            joint_pos = joint_pos.to(self.device, non_blocking=True)
            joint_vel = joint_vel.to(self.device, non_blocking=True)
            body_pos_w = body_pos_w.to(self.device, non_blocking=True)
            body_quat_w = body_quat_w.to(self.device, non_blocking=True)
            body_lin_vel_w = body_lin_vel_w.to(self.device, non_blocking=True)
            body_ang_vel_w = body_ang_vel_w.to(self.device, non_blocking=True)

        return joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    def gather_centered_window(
        self,
        motion_ids: Tensor,
        center_frames: Tensor,
        half_window: int,
        stride: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Centered window of all 6 fields, each [B, 2*half_window+1, ...].

        Mirrors FlatMotionStore.gather_centered_window so command-level windowing code
        can call the same API on the dense or the flat store. A centered window of
        half-width L is a forward window of size 2L+1 starting at center - L*stride.
        """
        window_size = 2 * half_window + 1
        start = center_frames - half_window * stride
        return self.get_motion_window_full(motion_ids, start, window_size, stride)


# =============================================================================
# Sampling Strategies
# =============================================================================


class MotionSampler(ABC):
    """Abstract base class for motion sampling strategies.
    
    Samplers are responsible for:
    1. Selecting which motion to use for each environment
    2. Selecting which frame to start from
    3. (Optionally) Updating internal state based on episode outcomes
    """
    
    def __init__(self, num_motions: int, motion_lengths: Tensor, device: torch.device):
        """Initialize the sampler.
        
        Args:
            num_motions: Total number of available motions
            motion_lengths: Length of each motion [num_motions]
            device: Computation device
        """
        self.num_motions = num_motions
        self.motion_lengths = motion_lengths.to(device)
        self.device = device
    
    @abstractmethod
    def sample(self, num_samples: int) -> SamplingResult:
        """Sample motion and frame indices.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            SamplingResult with motion_ids and frame_ids
        """
        pass
    
    def update(
        self,
        motion_ids: Tensor,
        frame_ids: Tensor,
        terminated: Tensor,
    ) -> None:
        """Update sampler state based on episode outcomes.
        
        Override in subclasses that need to track success/failure.
        
        Args:
            motion_ids: Starting motion IDs for completed episodes
            frame_ids: Starting frame IDs for completed episodes
            terminated: Whether each episode was terminated (failed)
        """
        pass

    def step(self) -> None:
        """Optional per-step hook for samplers (e.g., EMA updates).

        Most samplers are stateless and can leave this as a no-op.
        """
        pass
    
    def get_metrics(self) -> dict[str, float]:
        """Get sampling metrics for logging.
        
        Returns:
            Dictionary of metric names to values
        """
        return {}


class UniformSampler(MotionSampler):
    """Uniform random sampling across all motions and frames."""
    
    def sample(self, num_samples: int) -> SamplingResult:
        """Sample uniformly from all motions and frames."""
        # Sample motion uniformly
        motion_ids = torch.randint(0, self.num_motions, (num_samples,), device=self.device)
        
        # Sample frame uniformly within each motion's valid range
        # Use randint-like behavior: sample from [0, length) uniformly
        selected_lengths = self.motion_lengths[motion_ids]
        # Generate uniform random in [0, 1), multiply by length, floor to get [0, length-1]
        # But torch.rand() is [0, 1), so we need to ensure last frame is reachable
        # Solution: use length directly and clamp the result
        phase = torch.rand(num_samples, device=self.device)
        frame_ids = (phase * selected_lengths.float()).long()
        # Clamp to valid range (handles edge case where phase ≈ 1.0)
        frame_ids = torch.clamp(frame_ids, max=selected_lengths - 1)
        
        return SamplingResult(motion_ids=motion_ids, frame_ids=frame_ids)


class AdaptiveSampler(MotionSampler):
    """Adaptive sampling based on failure rates (curriculum learning).
    
    Features:
    - Tracks success/failure counts per motion and optionally per frame
    - Uses EMA to smooth failure statistics
    - Applies exponential weighting (beta) to focus on hard motions/frames
    - Mixes with uniform sampling to ensure exploration
    
    The probability of sampling motion/frame is proportional to:
        p_fail^beta * (1 - uniform_ratio) + uniform_ratio / N
    """
    
    def __init__(
        self,
        num_motions: int,
        motion_lengths: Tensor,
        device: torch.device,
        # Adaptive sampling parameters
        enable_frame_sampling: bool = True,
        beta: float = 0.5,
        alpha: float = 0.001,
        uniform_ratio: float = 0.1,
        update_interval: int = 240,
        count_reset_interval: int = 24000,
        # Kernel smoothing for frame-level
        kernel_size: int = 50,
        kernel_lambda: float = 0.8,
    ):
        """Initialize adaptive sampler.
        
        Args:
            num_motions: Number of motions
            motion_lengths: Length of each motion
            device: Computation device
            enable_frame_sampling: If True, sample at frame level; else motion level only
            beta: Exponent for failure probability (0=uniform, 1=proportional, >1=focus on hard)
            alpha: EMA smoothing factor (lower=more stable)
            uniform_ratio: Minimum uniform sampling probability
            update_interval: Steps between EMA updates
            count_reset_interval: Steps between count resets
            kernel_size: Size of smoothing kernel for frame probabilities
            kernel_lambda: Decay factor for exponential kernel
        """
        super().__init__(num_motions, motion_lengths, device)
        
        self.enable_frame_sampling = enable_frame_sampling
        self.beta = beta
        self.alpha = alpha
        self.uniform_ratio = uniform_ratio
        self.update_interval = update_interval
        self.count_reset_interval = count_reset_interval
        
        max_length = int(motion_lengths.max().item())
        
        # Motion-level counts (always tracked)
        self.failed_motion_count = torch.ones(num_motions, device=device)
        self.success_motion_count = torch.ones(num_motions, device=device)
        self.current_failed_motion_count = torch.zeros(num_motions, device=device)
        self.current_success_motion_count = torch.zeros(num_motions, device=device)
        
        # Frame-level counts (optional)
        if enable_frame_sampling:
            self.max_motion_length = max_length
            self.failed_frame_count = torch.ones(num_motions, max_length, device=device)
            self.success_frame_count = torch.ones(num_motions, max_length, device=device)
            self.current_failed_frame_count = torch.zeros(num_motions, max_length, device=device)
            self.current_success_frame_count = torch.zeros(num_motions, max_length, device=device)
            
            # Valid frames mask
            frame_indices = torch.arange(max_length, device=device).unsqueeze(0)
            self.valid_frames_mask = frame_indices < motion_lengths.unsqueeze(1)
            self.total_valid_frames = self.valid_frames_mask.sum().item()
            
            # Smoothing kernel
            # Kernel is [λ^(k-1), λ^(k-2), ..., λ, 1] so that:
            # - kernel[-1] = 1 (highest weight for current frame)
            # - kernel[-2] = λ (weight for previous frame)
            # - kernel[0] = λ^(k-1) (smallest weight for oldest frame in window)
            # This ensures failures spread FORWARD: failure at frame i → elevated prob at i, i+1, ...
            if kernel_size > 1:
                # Create [λ^(k-1), λ^(k-2), ..., λ^1, λ^0] = [λ^(k-1), ..., λ, 1]
                kernel = torch.tensor(
                    [kernel_lambda ** (kernel_size - 1 - i) for i in range(kernel_size)],
                    device=device, dtype=torch.float32
                )
                self._kernel = (kernel / kernel.sum()).view(1, 1, -1)
                self._kernel_size = kernel_size
            else:
                self._kernel = None
                self._kernel_size = 0
        
        # Step counters
        self._steps_since_update = 0
        
        # Cached metrics
        self._last_entropy = 0.0
        self._last_pfail_mean = 0.0
        self._last_top1_prob = 0.0
    
    def sample(self, num_samples: int) -> SamplingResult:
        """Sample based on failure probabilities."""
        if self.enable_frame_sampling:
            return self._sample_frame_level(num_samples)
        else:
            return self._sample_motion_level(num_samples)
    
    def _sample_motion_level(self, num_samples: int) -> SamplingResult:
        """Sample at motion level only."""
        # Compute failure probability
        total = self.failed_motion_count + self.success_motion_count
        p_fail = self.failed_motion_count / (total + 1e-8)
        
        # Add small floor to prevent numerical issues with pow(tiny, high_beta)
        p_fail = torch.clamp(p_fail, min=1e-6)
        
        # Apply beta weighting
        p_weighted = torch.pow(p_fail, self.beta)
        p_weighted = p_weighted / (p_weighted.sum() + 1e-8)
        
        # Mix with uniform
        uniform = 1.0 / self.num_motions
        probs = p_weighted * (1.0 - self.uniform_ratio) + uniform * self.uniform_ratio
        
        # Final renormalization for numerical stability
        probs = probs / probs.sum()
        
        # Sample motions
        motion_ids = torch.multinomial(probs, num_samples, replacement=True)
        
        # Sample frames uniformly within each motion
        selected_lengths = self.motion_lengths[motion_ids]
        phase = torch.rand(num_samples, device=self.device)
        frame_ids = (phase * selected_lengths.float()).long()
        frame_ids = torch.clamp(frame_ids, max=selected_lengths - 1)
        
        return SamplingResult(motion_ids=motion_ids, frame_ids=frame_ids)
    
    def _sample_frame_level(self, num_samples: int) -> SamplingResult:
        """Sample at frame level with kernel smoothing."""
        # Compute per-frame failure probability
        total = self.failed_frame_count + self.success_frame_count
        pf = self.failed_frame_count / (total + 1e-8)
        
        # Apply kernel smoothing
        # A failure at frame i should spread to frames i, i+1, i+2, ... (forward spread)
        # With flipped kernel [λ^(k-1), ..., λ, 1] and LEFT padding:
        # - output[i] = sum_j input[i-k+1+j] * kernel[j]
        # - The current frame (input[i]) gets the highest weight (kernel[-1] = 1)
        # - Earlier frames get decaying weights
        if self._kernel is not None:
            pf = pf.clone()
            pf[~self.valid_frames_mask] = 0.0
            
            # Conv1d: [num_motions, 1, max_length]
            pf_3d = pf.unsqueeze(1)
            padding = self._kernel_size - 1
            # Pad on LEFT side for causal kernel (looks back at previous frames),
            # matching BinBasedAdaptiveSampler. With flipped kernel [λ^(k-1),...,1]
            # left padding makes output[i] = sum_j input[i-k+1+j]*kernel[j], so a
            # failure at frame i raises probability at i, i+1, ... (forward spread).
            pf_padded = torch.nn.functional.pad(pf_3d, (padding, 0), mode='constant', value=0.0)
            pf_smoothed = torch.nn.functional.conv1d(pf_padded, self._kernel)
            pf = pf_smoothed.squeeze(1)
            
            # CRITICAL: Re-zero invalid frames after smoothing to prevent leakage
            # from valid frames into padded regions
            pf[~self.valid_frames_mask] = 0.0
        
        # Flatten and apply weighting
        pf_flat = pf.view(-1).clone()
        valid_flat = self.valid_frames_mask.view(-1)
        pf_flat[~valid_flat] = 0.0
        
        # Add small floor to prevent numerical issues with pow(tiny, high_beta)
        pf_flat = torch.clamp(pf_flat, min=1e-6)
        pf_flat[~valid_flat] = 0.0  # Re-zero invalid after clamping
        
        # Beta weighting
        pf_weighted = torch.pow(pf_flat, self.beta)
        pf_weighted[~valid_flat] = 0.0
        pf_weighted = pf_weighted / (pf_weighted.sum() + 1e-8)
        
        # Mix with uniform
        uniform = 1.0 / max(1, self.total_valid_frames)
        probs = pf_weighted * (1.0 - self.uniform_ratio) + uniform * self.uniform_ratio
        probs[~valid_flat] = 0.0
        
        # Final renormalization for numerical stability
        probs = probs / probs.sum()
        
        # Sample
        sampled_flat = torch.multinomial(probs, num_samples, replacement=True)
        
        # Convert to motion/frame indices
        motion_ids = (sampled_flat // self.max_motion_length).long()
        frame_ids = (sampled_flat % self.max_motion_length).long()
        
        return SamplingResult(motion_ids=motion_ids, frame_ids=frame_ids)
    
    def update(
        self,
        motion_ids: Tensor,
        frame_ids: Tensor,
        terminated: Tensor,
    ) -> None:
        """Update failure/success counts based on episode outcomes."""
        if motion_ids.numel() == 0:
            return
        
        # Move ALL inputs to the sampler's device up front (mirrors
        # BinBasedAdaptiveSampler.update) so every downstream boolean index — both
        # motion-level (motion_ids[mask]) and frame-level (flat_index[mask]) — is
        # device-consistent even when the caller's tensors live on CPU while the
        # counters live on CUDA. Indexing a CPU tensor with a CUDA mask (or vice
        # versa) raises, so the moves MUST precede any masking.
        motion_ids = motion_ids.to(self.device, dtype=torch.long)
        frame_ids = frame_ids.to(self.device, dtype=torch.long)
        terminated = terminated.to(self.device)
        failed_mask = terminated
        success_mask = ~terminated

        # Update motion-level counts
        if failed_mask.any():
            failed_ids = motion_ids[failed_mask]
            counts = torch.bincount(failed_ids, minlength=self.num_motions)
            self.current_failed_motion_count += counts.to(self.device)

        if success_mask.any():
            success_ids = motion_ids[success_mask]
            counts = torch.bincount(success_ids, minlength=self.num_motions)
            self.current_success_motion_count += counts.to(self.device)

        # Update frame-level counts (vectorized scatter_add_ — no Python per-env loop).
        # Phase 0 hygiene (adaptive_sampling_discussion.md §2.B.3): the old per-env
        # for-loop with .item() forced a CPU sync on every reset. Flatten (motion, frame)
        # to a 1-D index and scatter-add the failed/success increments in one shot.
        #
        # Frame clamping: the old loop used `min(fid, max_len-1)` (UPPER clamp only),
        # so a negative fid would 2-D-index `[mid, -1]` -> the motion's LAST frame. We
        # reproduce that exact semantics for the real input domain (frame_ids >= 0 —
        # start frames are always non-negative) while ALSO being safe for the
        # degenerate negative case: emulate the old per-row wrap with a modulo into
        # [0, max_motion_length) BEFORE flattening, which equals `[mid, fid % L]` and
        # matches `[mid, -1] == [mid, L-1]` without spilling into another motion's row.
        if self.enable_frame_sampling:
            L = self.max_motion_length
            fids = torch.clamp(frame_ids, max=L - 1)   # old loop's upper clamp
            fids = torch.remainder(fids, L)            # per-row wrap == old 2-D [mid, fid]
            flat_index = motion_ids * L + fids
            flat_failed = self.current_failed_frame_count.view(-1)
            flat_success = self.current_success_frame_count.view(-1)
            if failed_mask.any():
                idx = flat_index[failed_mask]
                flat_failed.scatter_add_(0, idx, torch.ones_like(idx, dtype=flat_failed.dtype))
            if success_mask.any():
                idx = flat_index[success_mask]
                flat_success.scatter_add_(0, idx, torch.ones_like(idx, dtype=flat_success.dtype))
    
    def step(self) -> None:
        """Called each simulation step to update internal state."""
        self._steps_since_update += 1
        
        # EMA update
        if self._steps_since_update >= self.update_interval:
            self._ema_update()
            self._reset_counts()
            self._steps_since_update = 0
    
    def _ema_update(self) -> None:
        """Apply EMA to blend current counts with historical."""
        alpha = self.alpha
        
        # Motion-level
        self.failed_motion_count = (
            alpha * self.current_failed_motion_count +
            (1.0 - alpha) * self.failed_motion_count
        )
        self.success_motion_count = (
            alpha * self.current_success_motion_count +
            (1.0 - alpha) * self.success_motion_count
        )
        
        # Frame-level
        if self.enable_frame_sampling:
            self.failed_frame_count = (
                alpha * self.current_failed_frame_count +
                (1.0 - alpha) * self.failed_frame_count
            )
            self.success_frame_count = (
                alpha * self.current_success_frame_count +
                (1.0 - alpha) * self.success_frame_count
            )
        
        # Update cached metrics
        self._update_metrics()
    
    def _reset_counts(self) -> None:
        """Reset current count accumulators."""
        self.current_failed_motion_count.zero_()
        self.current_success_motion_count.zero_()
        
        if self.enable_frame_sampling:
            self.current_failed_frame_count.zero_()
            self.current_success_frame_count.zero_()
    
    def _update_metrics(self) -> None:
        """Update cached metrics for logging."""
        import math
        
        total = self.failed_motion_count + self.success_motion_count
        p_fail = self.failed_motion_count / (total + 1e-8)
        
        # Compute sampling probabilities
        p_weighted = torch.pow(p_fail, self.beta)
        p_weighted = p_weighted / (p_weighted.sum() + 1e-8)
        uniform = 1.0 / self.num_motions
        probs = p_weighted * (1.0 - self.uniform_ratio) + uniform * self.uniform_ratio
        
        # Entropy (normalized)
        H = -(probs * torch.log(probs + 1e-12)).sum()
        self._last_entropy = (H / math.log(self.num_motions)).item()
        
        # Mean failure probability
        self._last_pfail_mean = p_fail.mean().item()
        
        # Top-1 probability
        self._last_top1_prob = probs.max().item()
    
    def get_metrics(self) -> dict[str, float]:
        """Get sampling metrics for logging."""
        return {
            "adaptive_entropy": self._last_entropy,
            "adaptive_pfail_mean": self._last_pfail_mean,
            "adaptive_top1_prob": self._last_top1_prob,
            "total_failed": self.failed_motion_count.sum().item(),
            "total_success": self.success_motion_count.sum().item(),
        }
    
    def get_failure_probabilities(self) -> Tensor:
        """Get current per-motion failure probabilities (for testing/debugging)."""
        total = self.failed_motion_count + self.success_motion_count
        return self.failed_motion_count / (total + 1e-8)
    
    def get_sampling_probabilities(self) -> Tensor:
        """Get current sampling probabilities (for testing/debugging)."""
        p_fail = self.get_failure_probabilities()
        
        # Add small floor to prevent numerical issues with pow(tiny, high_beta)
        p_fail_floored = torch.clamp(p_fail, min=1e-6)
        
        p_weighted = torch.pow(p_fail_floored, self.beta)
        p_weighted = p_weighted / (p_weighted.sum() + 1e-8)
        
        uniform = 1.0 / self.num_motions
        probs = p_weighted * (1.0 - self.uniform_ratio) + uniform * self.uniform_ratio
        
        # Final renormalization to ensure sum == 1.0 (handles floating point drift)
        probs = probs / probs.sum()
        
        return probs


# =============================================================================
# Multi-Motion Command V2
# =============================================================================


class MultiMotionCommandV2(CommandTerm):
    """Clean multi-motion command using separated concerns.
    
    This class orchestrates:
    - MotionDataStore: For loading and retrieving motion data
    - MotionSampler: For selecting motions and frames
    
    Key improvements over the original:
    - Clear separation between data management and sampling
    - Easy to swap samplers (uniform, adaptive, custom)
    - Cleaner codebase with less duplication
    """
    
    cfg: "MultiMotionCommandV2Cfg"
    
    def __init__(self, cfg: "MultiMotionCommandV2Cfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body)
        
        # Get body indices
        body_indices = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=self.device
        )
        self.body_indices = body_indices
        
        # Initialize data store (overridable so the streaming subclass can swap in
        # an indexed, working-set store without duplicating __init__).
        self._init_data_store()

        # Initialize sampler
        self.sampler = self._create_sampler()
        
        # State tracking
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_frame_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._default_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._default_reset_lerp = torch.zeros(self.num_envs, 1, device=self.device)
        
        # Motion data buffers (updated each step)
        num_bodies = len(cfg.body_names)
        # Get num_joints from data store (set during stacking)
        num_joints = self.data_store.num_joints if self.data_store.num_joints > 0 else 29
        
        self._joint_pos_buf = torch.zeros(self.num_envs, num_joints, device=self.device)
        self._joint_vel_buf = torch.zeros(self.num_envs, num_joints, device=self.device)
        self._body_pos_w_buf = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self._body_quat_w_buf = torch.zeros(self.num_envs, num_bodies, 4, device=self.device)
        self._body_quat_w_buf[:, :, 0] = 1.0
        self._body_lin_vel_w_buf = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self._body_ang_vel_w_buf = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        
        # Relative pose buffers
        self.body_pos_relative_w = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, num_bodies, 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        
        # Initialize metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        
        # Flag to skip sampler updates during the first reset
        # On initial reset(), terminated is False for all envs, which would incorrectly
        # count as "success" and bias the adaptive sampling statistics
        self._is_first_reset = True
    
    def _init_data_store(self) -> None:
        """Create the data store and load motions. Overridable by subclasses.

        Base behavior: the original eager, load-everything dense store.
        """
        self.data_store = MotionDataStore(
            device=self.device,
            storage_device=self.cfg.storage_device,
            use_fp16=self.cfg.use_fp16,
            chunk_length=self.cfg.chunk_length,
        )
        self.data_store.load_files(
            file_paths=self.cfg.motion_files,
            body_indices=self.body_indices,
            load_human=self.cfg.load_human_motion,
        )

    def _create_sampler(self) -> MotionSampler:
        """Create the appropriate sampler based on config."""
        if self.data_store.num_motions == 0 or self.data_store.motion_lengths is None:
            # Fallback for empty data store
            dummy_lengths = torch.tensor([1], device=self.device)
            return UniformSampler(1, dummy_lengths, self.device)
        
        motion_lengths = self.data_store.motion_lengths
        
        if self.cfg.sampler_type == "uniform":
            return UniformSampler(
                self.data_store.num_motions,
                motion_lengths,
                self.device,
            )
        elif self.cfg.sampler_type == "adaptive":
            return AdaptiveSampler(
                num_motions=self.data_store.num_motions,
                motion_lengths=motion_lengths,
                device=self.device,
                enable_frame_sampling=self.cfg.enable_frame_sampling,
                beta=self.cfg.adaptive_beta,
                alpha=self.cfg.adaptive_alpha,
                uniform_ratio=self.cfg.adaptive_uniform_ratio,
                update_interval=self.cfg.adaptive_update_interval,
                kernel_size=self.cfg.adaptive_kernel_size,
                kernel_lambda=self.cfg.adaptive_kernel_lambda,
            )
        elif self.cfg.sampler_type == "bin_adaptive":
            from .bin_based_sampler import BinBasedAdaptiveSampler

            return BinBasedAdaptiveSampler(
                num_motions=self.data_store.num_motions,
                motion_lengths=motion_lengths,
                device=self.device,
                motion_fps=self.cfg.motion_fps,
                bin_duration=self.cfg.bin_duration,
                beta=self.cfg.adaptive_beta,
                alpha=self.cfg.adaptive_alpha,
                uniform_ratio=self.cfg.adaptive_uniform_ratio,
                update_interval=self.cfg.adaptive_update_interval,
                kernel_size=self.cfg.adaptive_kernel_size,
                kernel_lambda=self.cfg.adaptive_kernel_lambda,
            )
        else:
            raise ValueError(f"Unknown sampler type: {self.cfg.sampler_type}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def command(self) -> Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def joint_pos(self) -> Tensor:
        return self._joint_pos_buf
    
    @property
    def joint_vel(self) -> Tensor:
        return self._joint_vel_buf
    
    @property
    def body_pos_w(self) -> Tensor:
        return self._body_pos_w_buf + self._env.scene.env_origins[:, None, :]
    
    @property
    def body_quat_w(self) -> Tensor:
        return self._body_quat_w_buf
    
    @property
    def body_lin_vel_w(self) -> Tensor:
        return self._body_lin_vel_w_buf
    
    @property
    def body_ang_vel_w(self) -> Tensor:
        return self._body_ang_vel_w_buf
    
    @property
    def anchor_pos_w(self) -> Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]
    
    @property
    def anchor_quat_w(self) -> Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]
    
    @property
    def anchor_lin_vel_w(self) -> Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]
    
    @property
    def anchor_ang_vel_w(self) -> Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]
    
    @property
    def robot_joint_pos(self) -> Tensor:
        return self.robot.data.joint_pos
    
    @property
    def robot_joint_vel(self) -> Tensor:
        return self.robot.data.joint_vel
    
    @property
    def robot_body_pos_w(self) -> Tensor:
        return self.robot.data.body_pos_w[:, self.body_indices]
    
    @property
    def robot_body_quat_w(self) -> Tensor:
        return self.robot.data.body_quat_w[:, self.body_indices]
    
    @property
    def robot_anchor_pos_w(self) -> Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
    
    @property
    def robot_anchor_quat_w(self) -> Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
    
    @property
    def robot_body_lin_vel_w(self) -> Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indices]
    
    @property
    def robot_body_ang_vel_w(self) -> Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indices]
    
    @property
    def robot_anchor_lin_vel_w(self) -> Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]
    
    @property
    def robot_anchor_ang_vel_w(self) -> Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def _resample_command(self, env_ids: Sequence[int] | Tensor) -> None:
        """Resample motion and frame for given environments."""
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        if env_ids.numel() == 0:
            return
        
        # Update sampler with episode outcomes
        # SKIP on first reset to avoid biasing statistics with false "successes"
        if self._is_first_reset:
            # Check if this is the initial full reset (all environments)
            if len(env_ids) == self.num_envs:
                self._is_first_reset = False
                # Don't update sampler on first reset - no valid episode outcomes yet
        elif hasattr(self._env, 'termination_manager'):
            terminated = self._env.termination_manager.terminated[env_ids]
            # Routed through a hook so group-aware subclasses can map global motion
            # ids to per-group local ids before updating their per-group samplers.
            self._record_outcomes(env_ids, terminated)
            # Hook for subclasses (e.g. streaming) to also record outcomes against
            # a global, swap-persistent curriculum. No-op in the base class.
            self._on_episode_outcomes(env_ids, terminated)

        # Sample new motion/frame (routed through a hook so group-aware subclasses
        # can draw per-env-group samples and remap local ids back to global).
        result = self._draw_samples(env_ids)
        self._default_reset_mask[env_ids] = False
        self._default_reset_lerp[env_ids] = 0.0

        # Optionally force every reset to start from frame 0 of the sampled motion.
        # The sampler still chooses which motion (preserving adaptive curriculum over
        # clips); we only override the frame index so envs always begin from the
        # motion's natural starting pose rather than a random/airborne phase.
        if getattr(self.cfg, "force_start_frame_zero", False):
            result = SamplingResult(
                motion_ids=result.motion_ids,
                frame_ids=torch.zeros_like(result.frame_ids),
            )

        # Optionally start a subset of resets from an early frame and blend their
        # joint configuration toward the robot's default pose. This improves reset
        # robustness for stepping-stone distillation / finetuning by preventing all
        # envs from spawning directly into potentially aggressive motion poses.
        if self.cfg.default_reset_joint_prob > 0.0:
            default_mask_local = torch.rand(len(env_ids), device=self.device) < float(self.cfg.default_reset_joint_prob)
            if default_mask_local.any():
                default_env_ids = env_ids[default_mask_local]
                low, high = self.cfg.default_reset_frame_range
                low = int(low)
                high = int(high)
                if high < low:
                    low, high = high, low

                frame_ids = result.frame_ids.clone()
                lengths = self.data_store.motion_lengths[result.motion_ids[default_mask_local]]
                sampled = torch.randint(low, high + 1, (default_env_ids.numel(),), device=self.device)
                sampled = torch.minimum(sampled, lengths - 1)
                sampled = torch.clamp(sampled, min=0)
                frame_ids[default_mask_local] = sampled
                result = SamplingResult(motion_ids=result.motion_ids, frame_ids=frame_ids)

                lerp_min, lerp_max = self.cfg.default_reset_joint_lerp_range
                lerp_min = float(lerp_min)
                lerp_max = float(lerp_max)
                if lerp_max < lerp_min:
                    lerp_min, lerp_max = lerp_max, lerp_min
                self._default_reset_mask[default_env_ids] = True
                self._default_reset_lerp[default_env_ids] = sample_uniform(
                    lerp_min, lerp_max, (default_env_ids.numel(), 1), self.device
                )

        self.motion_ids[env_ids] = result.motion_ids
        self.frame_ids[env_ids] = result.frame_ids
        
        # Track starting state for next update
        self.start_motion_ids[env_ids] = result.motion_ids.clone()
        self.start_frame_ids[env_ids] = result.frame_ids.clone()
        
        # Update motion buffers for these envs
        self._update_motion_buffers_for_envs(env_ids)
        
        # Apply noise and reset robot state
        self._reset_robot_state(env_ids)

        # Keep joint-position action offset aligned with the new reference (optional).
        self._maybe_update_action_offset(env_ids)

        # Terminations are evaluated before ``_update_command`` on the next RL
        # step, so reset must leave the relative reference buffers coherent.
        # Otherwise strict body/foot terms can fire from stale zeros/defaults on
        # the first post-reset step even when the robot was reset onto the
        # sampled reference.
        self._compute_relative_poses()
        self._update_metrics()
    
    def _on_episode_outcomes(self, env_ids: Tensor, terminated: Tensor) -> None:
        """Hook: record episode outcomes for the resetting envs.

        No-op in the base command. The streaming subclass overrides this to feed
        a global, swap-persistent curriculum (translating resident-local motion
        ids to global ids first).
        """
        pass

    def _record_outcomes(self, env_ids: Tensor, terminated: Tensor) -> None:
        """Feed episode outcomes to the sampler.

        Default: update the single sampler with the global start ids. Group-aware
        subclasses override this to route outcomes to per-group samplers (mapping
        global motion ids to per-group local ids first).
        """
        self.sampler.update(
            self.start_motion_ids[env_ids],
            self.start_frame_ids[env_ids],
            terminated,
        )

    def _draw_samples(self, env_ids: Tensor) -> SamplingResult:
        """Draw a SamplingResult (global motion ids + frame ids) for ``env_ids``.

        Default: draw ``len(env_ids)`` samples from the single sampler. Group-aware
        subclasses override this to draw per-env-group samples (preserving the
        env_ids -> group assignment) and remap per-group local ids back to global.
        """
        return self.sampler.sample(len(env_ids))

    def _pose_velocity_noise(self, env_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Per-env pose/velocity reset noise, each shaped [len(env_ids), 6].

        Default behavior uses the single cfg ranges for every env. Group-aware
        subclasses override this to apply different ranges per env group (e.g.
        zero positional noise for terrain-anchored envs).
        """
        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        pose_ranges = [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in pose_keys]
        ranges = torch.tensor(pose_ranges, device=self.device)
        pose_rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        vel_ranges = [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in pose_keys]
        ranges = torch.tensor(vel_ranges, device=self.device)
        vel_rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        return pose_rand, vel_rand

    def _joint_position_noise(self, joint_pos_shape: torch.Size) -> Tensor:
        """Joint-position reset noise for ALL envs, shaped like joint_pos.

        Default uses the single cfg range. Group-aware subclasses override this to
        zero the noise for terrain-anchored envs.
        """
        return sample_uniform(*self.cfg.joint_position_range, joint_pos_shape, self.device)

    def _reset_robot_state(self, env_ids: Tensor) -> None:
        """Reset robot state with noise for given environments."""
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # Pose + velocity noise (per-env, overridable for group-aware resets).
        pose_rand, vel_rand = self._pose_velocity_noise(env_ids)

        root_pos[env_ids] += pose_rand[:, 0:3]
        delta_ori = quat_from_euler_xyz(pose_rand[:, 3], pose_rand[:, 4], pose_rand[:, 5])
        root_ori[env_ids] = quat_mul(delta_ori, root_ori[env_ids])

        root_lin_vel[env_ids] += vel_rand[:, :3]
        root_ang_vel[env_ids] += vel_rand[:, 3:]

        # Joint state with noise
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += self._joint_position_noise(joint_pos.shape)
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]

        # For selected environments, blend the reference pose back toward the
        # articulation default pose and optionally zero velocities.
        default_reset_mask = self._default_reset_mask[env_ids]
        if default_reset_mask.any():
            default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
            lerp = self._default_reset_lerp[env_ids][default_reset_mask]
            joint_pos_env = joint_pos[env_ids]
            joint_pos_env[default_reset_mask] = torch.lerp(
                joint_pos_env[default_reset_mask], default_joint_pos[default_reset_mask], lerp
            )
            joint_pos[env_ids] = joint_pos_env
            joint_vel_env = joint_vel[env_ids]
            joint_vel_env[default_reset_mask] = default_joint_vel[default_reset_mask]
            root_lin_vel_env = root_lin_vel[env_ids]
            root_ang_vel_env = root_ang_vel[env_ids]
            if self.cfg.default_reset_zero_vel:
                joint_vel_env[default_reset_mask] = 0.0
                root_lin_vel_env[default_reset_mask] = 0.0
                root_ang_vel_env[default_reset_mask] = 0.0
            joint_vel[env_ids] = joint_vel_env
            root_lin_vel[env_ids] = root_lin_vel_env
            root_ang_vel[env_ids] = root_ang_vel_env

        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], soft_limits[:, :, 0], soft_limits[:, :, 1])

        # Write to simulation
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
    
    def _update_command(self) -> None:
        """Update command - called every step."""
        # Advance frame counter
        self.frame_ids += 1
        
        # Update sampler internal state (if any).
        self.sampler.step()
        
        # Check for motions that have ended
        assert self.data_store.motion_lengths is not None, "Motion data not loaded"
        motion_lengths = self.data_store.motion_lengths[self.motion_ids]
        reset_mask = self.frame_ids >= motion_lengths
        
        if reset_mask.any():
            self._resample_command(torch.where(reset_mask)[0])
        
        # Update all motion buffers
        self._update_motion_buffers()
        
        # Compute relative body poses
        self._compute_relative_poses()

        # Optionally keep the joint-position action offset aligned with the reference.
        # This enables residual targets: q_target = q_ref + scale * action.
        self._maybe_update_action_offset(slice(None))
    
    def _update_motion_buffers(self) -> None:
        """Update motion buffers for all environments."""
        (
            joint_pos, joint_vel,
            body_pos, body_quat,
            body_lin_vel, body_ang_vel
        ) = self.data_store.get_motion_data(self.motion_ids, self.frame_ids)
        
        self._joint_pos_buf.copy_(joint_pos.float())
        self._joint_vel_buf.copy_(joint_vel.float())
        self._body_pos_w_buf.copy_(body_pos.float())
        self._body_quat_w_buf.copy_(body_quat.float())
        self._body_lin_vel_w_buf.copy_(body_lin_vel.float())
        self._body_ang_vel_w_buf.copy_(body_ang_vel.float())
    
    def _update_motion_buffers_for_envs(self, env_ids: Tensor) -> None:
        """Update motion buffers for specific environments."""
        (
            joint_pos, joint_vel,
            body_pos, body_quat,
            body_lin_vel, body_ang_vel
        ) = self.data_store.get_motion_data(
            self.motion_ids[env_ids],
            self.frame_ids[env_ids]
        )
        
        self._joint_pos_buf[env_ids] = joint_pos.float()
        self._joint_vel_buf[env_ids] = joint_vel.float()
        self._body_pos_w_buf[env_ids] = body_pos.float()
        self._body_quat_w_buf[env_ids] = body_quat.float()
        self._body_lin_vel_w_buf[env_ids] = body_lin_vel.float()
        self._body_ang_vel_w_buf[env_ids] = body_ang_vel.float()
    
    def _compute_relative_poses(self) -> None:
        """Compute body poses relative to robot anchor."""
        num_bodies = len(self.cfg.body_names)
        
        anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        
        delta_pos_w = anchor_pos_w - robot_anchor_pos_w
        delta_pos_w[..., :2] = 0.0  # Zero out x,y delta
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))
        
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = (
            robot_anchor_pos_w + delta_pos_w +
            quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w)
        )
    
    def _update_metrics(self) -> None:
        """Update tracking error metrics."""
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        #sampler_stats = self.sampler.get_metrics() # You defined this earlier
        # self.metrics["sampler/entropy"] = sampler_stats["adaptive_entropy"]
        # self.metrics["sampler/top1_prob"] = sampler_stats["adaptive_top1_prob"]
        # Add sampler metrics
        sampler_metrics = self.sampler.get_metrics()
        for key, value in sampler_metrics.items():
            self.metrics[key] = torch.full((self.num_envs,), value, device=self.device)
    
    # =========================================================================
    # SONIC Methods
    # =========================================================================
    
    def get_sonic_robot_window(self, env_ids: Tensor) -> Tensor:
        """Get future robot motion window for SONIC encoder."""
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        
        joint_pos, joint_vel = self.data_store.get_motion_window(
            self.motion_ids[env_ids],
            self.frame_ids[env_ids],
            window_size=10,
            stride=5,  # 0.1s at 50Hz
        )
        
        # Concatenate and flatten
        result = torch.cat([joint_pos, joint_vel], dim=-1)
        return result.view(len(env_ids), -1).float()
    
    def get_sonic_human_window(self, env_ids: Tensor) -> Tensor:
        """Get future human motion window (10 frames, dt=0.02s) for SONIC encoder.
        
        Returns shape: [num_envs, 10 * num_human_joints * 3]
        """
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        # Check if human data is available
        if self.data_store._stacked_human_joint_pos is None or self.data_store.max_human_motion_length == 0:
            return torch.zeros(len(env_ids), 10 * 22 * 3, device=self.device)
        
        storage_device = self.data_store.storage_device
        
        # Human motion parameters
        dt = 0.02
        fps = 50.0
        stride = max(1, int(dt * fps))  # 1 frame
        window_size = 10
        
        motion_ids = self.motion_ids[env_ids].to(storage_device)
        current_steps = self.frame_ids[env_ids].to(storage_device)
        
        # Get per-motion fps ratio for proper temporal alignment
        fps_ratios = self.data_store._stacked_fps_ratio[motion_ids]  # [batch_size]
        human_current_steps = (current_steps.float() * fps_ratios).long()
        
        # Create window offsets: [0, 1, 2, ..., 9] * stride
        offsets = torch.arange(window_size, device=storage_device) * stride
        
        # Expand to [batch_size, window_size]
        indices = human_current_steps.unsqueeze(1) + offsets.unsqueeze(0)
        
        # Clamp to valid range
        max_len = self.data_store.max_human_motion_length
        indices = torch.clamp(indices, 0, max(0, max_len - 1)).long()
        
        # Expand motion_ids to [batch_size, window_size]
        batch_motion_ids = motion_ids.unsqueeze(1).expand(-1, window_size)
        
        # Gather human positions from stacked data
        h_pos = self.data_store._stacked_human_joint_pos[batch_motion_ids, indices]
        
        # Flatten to [batch_size, 10 * num_joints * 3]
        result = h_pos.view(len(env_ids), -1).float()
        
        if storage_device != self.device:
            result = result.to(self.device)
        
        return result


    # =========================================================================
    # Command-window Methods
    # =========================================================================

    def get_command_window(
        self,
        env_ids: Tensor | Sequence[int],
        *,
        half_window: int = 10,
        stride: int = 1,
        flatten: bool = False,
    ) -> Tensor:
        """Get a centered reference command window for transformer policies.

        Each step in the window is a 38D vector:
          [v_ref_b(3), w_ref_b(3), g_ref_b(3), q_ref(29)]

        Args:
            env_ids: Environment indices.
            half_window: L in the paper (window size = 2L+1).
            stride: Frame stride in the motion library.
            flatten: If True, returns [B, (2L+1)*38] instead of [B, 2L+1, 38].

        Returns:
            Command window tensor on compute device.
        """
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            window_size = 2 * half_window + 1
            out_shape = (0, window_size * 38) if flatten else (0, window_size, 38)
            return torch.zeros(*out_shape, device=self.device)

        window_size = 2 * half_window + 1

        # Store-agnostic: gather the centered window via the data-store API so this
        # works on both the dense MotionDataStore and the streaming FlatMotionStore
        # (which has no _stacked_* fields). Returns tensors on the compute device.
        motion_ids = self.motion_ids[env_ids]
        center_frames = self.frame_ids[env_ids]
        (
            q_ref,
            _joint_vel,
            _body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = self.data_store.gather_centered_window(
            motion_ids, center_frames, half_window=half_window, stride=stride
        )

        # Slice the anchor body and compute in float32 for stability.
        q_ref = q_ref.float()
        anchor_quat_w = body_quat_w[:, :, self.motion_anchor_body_index].float()
        anchor_lin_vel_w = body_lin_vel_w[:, :, self.motion_anchor_body_index].float()
        anchor_ang_vel_w = body_ang_vel_w[:, :, self.motion_anchor_body_index].float()

        inv_quat = quat_inv(anchor_quat_w)
        v_ref_b = quat_apply(inv_quat, anchor_lin_vel_w)
        w_ref_b = quat_apply(inv_quat, anchor_ang_vel_w)

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=v_ref_b.dtype).view(1, 1, 3)
        g_ref_b = quat_apply(inv_quat, gravity_w.expand(v_ref_b.shape[0], window_size, 3))

        cmd = torch.cat([v_ref_b, w_ref_b, g_ref_b, q_ref], dim=-1)
        if flatten:
            return cmd.reshape(cmd.shape[0], -1)
        return cmd

    def get_motion_anchor_delta_window(
        self,
        env_ids: Tensor | Sequence[int],
        *,
        half_window: int = 10,
        stride: int = 1,
        flatten: bool = False,
    ) -> Tensor:
        """Get a centered motion-anchor displacement window in the current motion-anchor frame.

        Each step in the window is a 3D vector:
          delta_p_ref = R_anchor(t)^-1 * (p_anchor(t+k) - p_anchor(t))

        Args:
            env_ids: Environment indices.
            half_window: L in the paper (window size = 2L+1).
            stride: Frame stride in the motion library.
            flatten: If True, returns [B, (2L+1)*3] instead of [B, 2L+1, 3].

        Returns:
            Motion-anchor displacement window tensor on compute device.
        """
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            window_size = 2 * half_window + 1
            out_shape = (0, window_size * 3) if flatten else (0, window_size, 3)
            return torch.zeros(*out_shape, device=self.device)

        window_size = 2 * half_window + 1

        # Store-agnostic gather (works on dense + flat stores; tensors on device).
        motion_ids = self.motion_ids[env_ids]
        center_frames = self.frame_ids[env_ids]
        (
            _q_ref,
            _joint_vel,
            body_pos_w,
            body_quat_w,
            _body_lin_vel_w,
            _body_ang_vel_w,
        ) = self.data_store.gather_centered_window(
            motion_ids, center_frames, half_window=half_window, stride=stride
        )

        anchor_pos_w = body_pos_w[:, :, self.motion_anchor_body_index].float()
        anchor_quat_w = body_quat_w[:, :, self.motion_anchor_body_index].float()

        center_pos_w = anchor_pos_w[:, half_window : half_window + 1]
        center_quat_w = anchor_quat_w[:, half_window : half_window + 1]
        delta_pos_w = anchor_pos_w - center_pos_w
        delta_pos_ref = quat_apply(quat_inv(center_quat_w).expand(-1, window_size, -1), delta_pos_w)

        if flatten:
            return delta_pos_ref.reshape(delta_pos_ref.shape[0], -1)
        return delta_pos_ref

    # =========================================================================
    # Residual Action Offset Support
    # =========================================================================

    def _maybe_update_action_offset(self, env_ids: Tensor | slice) -> None:
        """Optionally set JointPositionAction offset to current q_ref.

        When enabled, the joint position action target becomes:
          q_target = q_ref + scale * action
        """
        if not getattr(self.cfg, "update_action_offset_with_ref", False):
            return

        action_term_name = getattr(self.cfg, "action_term_name", "joint_pos")
        try:
            action_term = self._env.action_manager.get_term(action_term_name)
        except Exception:
            return

        offset = getattr(action_term, "_offset", None)
        if not isinstance(offset, torch.Tensor):
            return

        q_ref = self.joint_pos if isinstance(env_ids, slice) else self.joint_pos[env_ids]
        if offset.shape[-1] != q_ref.shape[-1]:
            raise ValueError(
                f"Action term '{action_term_name}' offset dim {offset.shape[-1]} does not match "
                f"reference joint dim {q_ref.shape[-1]}."
            )

        if isinstance(env_ids, slice):
            offset.copy_(q_ref.to(offset.device, non_blocking=True).float())
        else:
            # Indexed assignment (NOT offset[env_ids].copy_, which writes to a temporary
            # copy and leaves the real offset stale for freshly-reset envs).
            offset[env_ids] = q_ref.to(offset.device, non_blocking=True).float()


# =============================================================================
# Command Configuration
# =============================================================================


class SyncedStudentMultiMotionCommandV2(MultiMotionCommandV2):
    """Secondary MultiMotion command that mirrors a primary command's motion/frame state."""

    cfg: "SyncedStudentMultiMotionCommandV2Cfg"

    def _create_sampler(self) -> MotionSampler:
        """Use a lightweight sampler because synchronized student commands never sample independently."""
        if self.data_store.num_motions == 0 or self.data_store.motion_lengths is None:
            dummy_lengths = torch.tensor([1], device=self.device)
            return UniformSampler(1, dummy_lengths, self.device)
        return UniformSampler(
            self.data_store.num_motions,
            self.data_store.motion_lengths,
            self.device,
        )

    def _get_primary_command(self) -> MultiMotionCommandV2:
        primary = self._env.command_manager.get_term(self.cfg.sync_command_name)
        if not isinstance(primary, MultiMotionCommandV2):
            raise TypeError(
                f"Synced student command requires MultiMotionCommandV2 under '{self.cfg.sync_command_name}', "
                f"received {type(primary)}."
            )
        return primary

    def _resample_command(self, env_ids: Sequence[int] | Tensor) -> None:
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return

        primary = self._get_primary_command()
        self.motion_ids[env_ids] = primary.motion_ids[env_ids]
        self.frame_ids[env_ids] = primary.frame_ids[env_ids]
        self.start_motion_ids[env_ids] = primary.start_motion_ids[env_ids]
        self.start_frame_ids[env_ids] = primary.start_frame_ids[env_ids]
        self._update_motion_buffers_for_envs(env_ids)
        self._compute_relative_poses()
        self._maybe_update_action_offset(env_ids)

    def _update_command(self) -> None:
        primary = self._get_primary_command()
        self.motion_ids.copy_(primary.motion_ids)
        self.frame_ids.copy_(primary.frame_ids)
        self.start_motion_ids.copy_(primary.start_motion_ids)
        self.start_frame_ids.copy_(primary.start_frame_ids)
        self._update_motion_buffers()
        self._compute_relative_poses()
        self._maybe_update_action_offset(slice(None))

    def _update_metrics(self) -> None:
        pass


@configclass
class MultiMotionCommandV2Cfg(CommandTermCfg):
    """Configuration for the refactored multi-motion command.
    
    Defines configuration for MultiMotionCommandV2 which orchestrates
    motion data loading and adaptive sampling.
    """
    
    class_type: type = MultiMotionCommandV2
    
    # Robot configuration
    asset_name: str = "robot"
    anchor_body: str = "torso_link"
    body_names: list[str] = []
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    # Motion files
    motion_files: list[str] = []
    
    # Data store configuration
    storage_device: str = "cuda"  # "cpu" for large datasets
    use_fp16: bool = False
    chunk_length: int = 0  # 0 = no chunking
    load_human_motion: bool = True
    
    # Sampler configuration
    sampler_type: str = "adaptive"  # "uniform", "adaptive", or "bin_adaptive"
    
    # Adaptive sampler parameters
    enable_frame_sampling: bool = True
    adaptive_beta: float = 2
    adaptive_alpha: float = 0.001
    adaptive_uniform_ratio: float = 0.1
    adaptive_update_interval: int = 240
    adaptive_kernel_size: int = 50
    adaptive_kernel_lambda: float = 0.8

    # Bin-based sampler parameters (used when sampler_type == "bin_adaptive")
    motion_fps: float = 50.0
    bin_duration: float = 1.0  # seconds per bin
    
    # Reset noise ranges
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    default_reset_joint_prob: float = 0.0
    default_reset_joint_lerp_range: tuple[float, float] = (1.0, 1.0)
    default_reset_frame_range: tuple[int, int] = (0, 0)
    default_reset_zero_vel: bool = False

    # When True, every reset starts from frame 0 of the sampled motion (the
    # motion's natural starting pose) instead of a sampler-chosen phase/frame.
    # Useful for warm-starting acrobatic clips so envs never spawn mid-air in an
    # aggressive (e.g. inverted cartwheel) pose that no policy can recover from.
    # The sampler still picks WHICH motion; only the frame index is overridden.
    force_start_frame_zero: bool = False

    # Optional: enable residual joint targets.
    # When True, the command term will keep the specified joint-position action term's
    # offset aligned to the current reference joint positions (q_ref).
    update_action_offset_with_ref: bool = False
    action_term_name: str = "joint_pos"

@configclass
class SyncedStudentMultiMotionCommandV2Cfg(MultiMotionCommandV2Cfg):
    """Configuration for a MultiMotion command synchronized to a primary command."""

    class_type: type = SyncedStudentMultiMotionCommandV2
    sync_command_name: str = "motion"
