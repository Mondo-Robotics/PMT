"""TerrainFlatMix env cfg — ONE store, ONE sampler, per-clip origin+noise.

PMT plan §9b Phase 2.3b. ONE shared transformer policy is trained on a mixture of:
- terrain-anchored clips (big-map world frame; zero reset noise, zero origin offset), and
- flat clips (arbitrary plane motions; shifted onto the (90,0,0) flat patch baked into the
  combined mesh, with reset pose/velocity/joint noise).

Unlike the old ``G1TerrainFlatMultiMotionEnvCfg`` (which used the hard-env-partition
``GroupedMultiMotionCommandV2``: first K envs pinned to terrain @ origin, rest pinned to
flat @ (90,0,0)), this env uses the FLEXIBLE ``UnifiedMultiMotionCommand`` with per-clip
``{origin offset + noise}``: any env can play any clip, and its world origin + reset noise
are set per-reset from the clip's ``is_terrain`` flag (``mdp.per_env_origin`` /
``mdp.per_env_pose_velocity_noise``). This subsumes the grouped use-case WITHOUT a hard
env partition (plan §9b plan-of-record).

No-collision invariant: ``inject_env_origins=True`` keys each env's origin on the SAME
per-clip ``is_terrain`` flag that drives reset noise, and the injection happens in
``UnifiedMultiMotionCommand._reset_robot_state`` (after refreshing the flag, BEFORE the
root state is written from ``body_pos_w = _body_pos_w_buf + env_origins``). So a flat clip's
robot root is written at the flat patch (origin (90,0,0)) and never on the terrain mesh,
while a terrain clip's root is written at its baked mesh location (origin 0). ``env_spacing``
is 0 so the terrain importer adds no origin of its own.

The obs/reward/termination/scene structure is reused from the stepping-stone env cfg
(same transformer teacher stack), with the combined big-map+flat mesh and the unified
command swapped in.
"""
from __future__ import annotations

import os
from typing import List, Optional

from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.robots.g1 import G1_CYLINDER_CFG
from pmt_tasks.tracking_env_cfg import TrackingEnvCfg
from pmt_tasks.path_defaults import sonic_path, terrain_asset_path, terrain_motion_path
from pmt_tasks.utils.motion_paths import find_motion_files
from pmt_tasks.env_cfgs.pmt.stepping_stone import (
    END_EFFECTOR_BODY_NAMES,
    TransformerSteppingStoneObservationsCfg,
    SteppingStoneRewardsCfg,
    SteppingStoneSceneCfg,
    SteppingStoneTerminationsCfg,
    TRACKED_BODY_NAMES,
    _DEFAULT_MESH,
)

# Flat patch world origin baked by add_flat_patch_to_stl.py (--center-xy 90 0, top-z 0).
FLAT_ORIGIN: List[float] = [90.0, 0.0, 0.0]

# Combined big-map + flat-patch mesh placeholder for standalone construction.
_DEFAULT_TERRAIN_FLAT_MESH = terrain_asset_path("g1_29dof_big_map_with_flat.stl")
# Generic default clip dirs so the env cfg constructs standalone (the builder overrides
# these from the resolved config). Terrain clips are anchored to the big-map mesh; flat
# clips are lafan plane clips.
_DEFAULT_TERRAIN_MOTION_PATHS = [
    terrain_motion_path("terrain_mocaphouse", "walk_dance1sub1start", "optimized"),
]
_DEFAULT_FLAT_MOTION_PATHS = [
    sonic_path("lafan1", "robot_lafan1"),
]

# Flat-group reset noise (matches the base flat MultiMotionV2 config; terrain clips
# always get zero noise so they stay aligned to the mesh). source:
# terrain_flat_mix_env_cfg.py FLAT_POSE_RANGE / FLAT_VELOCITY_RANGE / FLAT_JOINT_*.
FLAT_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}
FLAT_VELOCITY_RANGE = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.2, 0.2),
    "roll": (-0.22, 0.22),
    "pitch": (-0.22, 0.22),
    "yaw": (-0.3, 0.3),
}
FLAT_JOINT_POSITION_RANGE = (-0.1, 0.1)


@configclass
class TerrainFlatUnifiedCommandsCfg:
    """Unified multi-motion command: terrain + flat clips in ONE store, ONE sampler.

    Per-clip {origin offset + noise}: terrain clips -> origin 0 + zero noise; flat clips
    -> origin flat_origin + cfg reset noise. ``motion_files`` (the full mixed list) and
    ``terrain_motion_files`` (the terrain subset) are filled in __post_init__.
    """

    motion = mdp.UnifiedMultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        anchor_body="torso_link",
        body_names=TRACKED_BODY_NAMES,
        storage_device="cuda",
        use_fp16=False,
        chunk_length=4000,
        load_human_motion=False,
        sampler_type="bin_adaptive",
        enable_frame_sampling=True,
        adaptive_beta=2,
        adaptive_alpha=0.001,
        adaptive_uniform_ratio=0.1,
        adaptive_update_interval=240,
        adaptive_kernel_size=50,
        adaptive_kernel_lambda=0.8,
        motion_files=[],            # full mixed list, filled in __post_init__
        terrain_motion_files=[],    # terrain subset, filled in __post_init__
        # PLANE reset noise (terrain rows always get zero — keyed on the per-clip flag).
        pose_range=FLAT_POSE_RANGE,
        velocity_range=FLAT_VELOCITY_RANGE,
        joint_position_range=FLAT_JOINT_POSITION_RANGE,
        # Per-clip env-origin injection (the §9b plan-of-record mechanism).
        inject_env_origins=True,
        flat_origin=FLAT_ORIGIN,
    )


@configclass
class PMTTerrainFlatMixEnvCfg(TrackingEnvCfg):
    """Mixed terrain+flat transformer training, ONE shared policy (unified command).

    The builder sets these instance attributes before __post_init__ runs:
      - ``pmt_mesh_path``: combined big-map + flat-patch mesh (from ${paths.TERRAIN_ROOT}/...)
      - ``pmt_terrain_motion_paths``: terrain-anchored clip dirs (zero noise/origin)
      - ``pmt_flat_motion_paths``: flat plane clip dirs (noise + flat_origin shift)
      - ``pmt_decimation`` / ``pmt_sim_dt``: per-motion control rate (§3a)
      - ``pmt_flat_origin``: flat-patch world origin (default (90,0,0))
    """

    pmt_mesh_path: str = _DEFAULT_TERRAIN_FLAT_MESH
    pmt_terrain_motion_paths: Optional[List[str]] = None
    pmt_flat_motion_paths: Optional[List[str]] = None
    pmt_decimation: int = 4
    pmt_sim_dt: float = 0.005
    pmt_flat_origin: List[float] = FLAT_ORIGIN

    scene: SteppingStoneSceneCfg = SteppingStoneSceneCfg(num_envs=64, env_spacing=0.0)
    observations: TransformerSteppingStoneObservationsCfg = TransformerSteppingStoneObservationsCfg()
    commands: TerrainFlatUnifiedCommandsCfg = TerrainFlatUnifiedCommandsCfg()
    rewards: SteppingStoneRewardsCfg = SteppingStoneRewardsCfg()
    terminations: SteppingStoneTerminationsCfg = SteppingStoneTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # config-driven decimation/sim.dt (§3a dt-from-motion).
        self.decimation = int(self.pmt_decimation)
        self.sim.dt = float(self.pmt_sim_dt)
        self.sim.render_interval = self.decimation

        # combined mesh from config (§5 paths).
        self.scene.terrain.mesh_path = self.pmt_mesh_path

        # robot + residual action setup (residual action: q_target = q_ref + a, scale 1.0).
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # env_spacing MUST be 0: per-clip origin injection sets the only origin offset;
        # the importer must add none of its own (terrain clips are world-placed on mesh).
        self.scene.env_spacing = 0.0
        self.actions.joint_pos.scale = 1.0
        self.commands.motion.update_action_offset_with_ref = True

        # command wiring.
        self.commands.motion.anchor_body = "torso_link"
        self.commands.motion.body_names = TRACKED_BODY_NAMES
        self.commands.motion.sampler_type = "bin_adaptive"
        self.commands.motion.flat_origin = list(self.pmt_flat_origin)

        # match old teacher: drop these domain-randomization events.
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.base_com = None

        # discover terrain + flat motion files (paths may be roots, not .npz files) —
        # DEFERRED until BOTH pmt_*_motion_paths are set. The builder constructs with
        # both None (first pass), then sets the real paths and re-runs. On the None pass
        # we SKIP discovery so we never fail-loud on missing default dirs (fatal on the
        # cluster). Discovery + fail-loud runs only when the paths are explicitly set.
        # The _DEFAULT_* constants are kept for reference but are NOT auto-fallbacks.
        if not self.pmt_terrain_motion_paths and not self.pmt_flat_motion_paths:
            print("[PMTTerrainFlatMix] deferred motion discovery (no pmt_*_motion_paths yet)")
        else:
            terrain_files = (
                find_motion_files(motion_paths=self.pmt_terrain_motion_paths, strict=False).files
                if self.pmt_terrain_motion_paths else []
            )
            flat_files = (
                find_motion_files(motion_paths=self.pmt_flat_motion_paths, strict=False).files
                if self.pmt_flat_motion_paths else []
            )
            if not terrain_files:
                raise ValueError(f"No terrain motion files found in {self.pmt_terrain_motion_paths}")
            if not flat_files:
                raise ValueError(f"No flat motion files found in {self.pmt_flat_motion_paths}")
            print(
                f"[PMTTerrainFlatMix] terrain={len(terrain_files)} flat={len(flat_files)} "
                f"flat_origin={self.pmt_flat_origin} mesh={self.pmt_mesh_path}"
            )

            # ONE store: the full mixed list. terrain_motion_files is the terrain subset
            # the command tags is_terrain=True (zero noise/origin); the rest are flat.
            self.commands.motion.motion_files = list(terrain_files) + list(flat_files)
            self.commands.motion.terrain_motion_files = list(terrain_files)


@configclass
class PMTTerrainFlatMixEnvCfg_PLAY(PMTTerrainFlatMixEnvCfg):
    """Small play/debug variant."""

    scene: SteppingStoneSceneCfg = SteppingStoneSceneCfg(num_envs=8, env_spacing=0.0)

    def __post_init__(self):
        super().__post_init__()
        self.terminations.ee_body_pos = None
