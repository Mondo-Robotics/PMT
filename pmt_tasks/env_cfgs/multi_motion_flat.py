"""MultiMotion V2 flat env cfg — ONE class for the whole MultiMotion/Flat family.

PMT plan §6 Phase 2.1 + §10 PART A. Reproduces the STRUCTURE of the old
``G1MultiMotionV2EnvCfg`` (multi_motion_env_cfg.py:167 -> TrackingEnvCfg) by composing
the ported Isaac Lab manager cfgs from ``pmt_tasks.mdp`` and ``pmt_tasks.tracking_env_cfg``
on a FLAT plane with the V2 multi-motion command and the base policy/critic obs groups.

This single class covers ALL five family members WITHOUT env-class proliferation —
the variation is config-driven (plan §3/§9b):
  - sampler (uniform / adaptive / bin_adaptive)  -> ``pmt_sampler_type`` (from motion/*.yaml)
  - storage (eager / streaming)                  -> ``pmt_storage_mode``  (from motion/*.yaml)
  - algorithm (ppo / bpo)                         -> agent-cfg swap only (same env)

When ``pmt_storage_mode == "streaming"`` the command is swapped to the streaming
variant (``StreamingMultiMotionCommandV2Cfg``), realizing plan §9b's "TWO command
classes" via a flag rather than a bespoke env subclass.

The data-driven values (motion clip dir, decimation/sim.dt, sampler, storage mode) are
injected by ``pmt_tasks.builder.build_env_cfg`` from the resolved OmegaConf config via
the ``pmt_*`` class attributes that ``__post_init__`` reads.
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from pmt_tasks.tracking_env_cfg import MySceneCfg, ObservationsCfg, TrackingEnvCfg
from pmt_tasks.path_defaults import motion_path
from pmt_tasks.utils.motion_paths import find_motion_files

# 14 tracked bodies (multi_motion_env_cfg.py:227, same as the V2 base + ADD).
TRACKED_BODY_NAMES: List[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# Generic default clip dir so the env cfg constructs standalone; builder overrides.
_DEFAULT_MOTION_PATHS = [motion_path("lafan_walk")]


@configclass
class MultiMotionFlatSceneCfg(MySceneCfg):
    """Flat plane scene + contact sensor (no height scanner; multi-motion flat is blind)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


@configclass
class MultiMotionCommandsCfg:
    """V2 multi-motion command on the flat plane (mirror of G1MultiMotionV2 base)."""

    motion = mdp.MultiMotionCommandV2Cfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        motion_files=[],
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
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
            "roll": (-0.22, 0.22),
            "pitch": (-0.22, 0.22),
            "yaw": (-0.3, 0.3),
        },
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class PMTMultiMotionFlatEnvCfg(TrackingEnvCfg):
    """MultiMotion V2 flat env, composed from ported mdp cfgs.

    Structurally equivalent to the old G1MultiMotionV2EnvCfg (flat plane + V2
    multi-motion command + base policy/critic obs groups). The sampler and storage
    mode are config flags, NOT subclasses (the old repo's
    Uniform/Adaptive/Streaming env subclasses collapse into this single class).

    Builder-injected attributes (set before __post_init__ re-runs):
      - ``pmt_motion_paths``: list of motion dirs/files (from ${paths.MOTION_ROOT}/...)
      - ``pmt_decimation`` / ``pmt_sim_dt``: control rate (§3a; flat -> 4/0.005)
      - ``pmt_sampler_type``: "uniform" | "adaptive" | "bin_adaptive"
      - ``pmt_storage_mode``: "eager" | "streaming"
      - ``pmt_max_working_set`` / ``pmt_num_load_workers`` / ``pmt_use_process_pool``:
        streaming-only knobs (used iff pmt_storage_mode == "streaming").
    """

    pmt_motion_paths: Optional[Union[str, List[str]]] = None
    pmt_decimation: int = 4
    pmt_sim_dt: float = 0.005
    pmt_sampler_type: str = "bin_adaptive"
    pmt_storage_mode: str = "eager"
    pmt_max_working_set: int = 0
    pmt_num_load_workers: int = 16
    pmt_use_process_pool: bool = False

    scene: MultiMotionFlatSceneCfg = MultiMotionFlatSceneCfg(num_envs=64, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        # config-driven decimation/sim.dt (§3a). flat = 4 / 0.005.
        self.decimation = int(self.pmt_decimation)
        self.sim.dt = float(self.pmt_sim_dt)
        self.sim.render_interval = self.decimation

        # robot + action scale (standard multi-motion V2 base, NOT residual).
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 0.0
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # command wiring + config-driven sampler.
        self.commands.motion.anchor_body = "torso_link"
        self.commands.motion.body_names = TRACKED_BODY_NAMES
        self.commands.motion.sampler_type = str(self.pmt_sampler_type)

        # tightened thresholds (mirror the ADD env / base multi-motion termination).
        if getattr(self.terminations, "ee_body_pos", None) is not None:
            self.terminations.ee_body_pos.params["threshold"] = 0.4
        if getattr(self.terminations, "anchor_pos", None) is not None:
            self.terminations.anchor_pos.params["threshold"] = 0.4

        # resolve motion files (fail-loud) — DEFERRED until pmt_motion_paths is set.
        # The builder constructs with pmt_motion_paths=None (first __post_init__ pass),
        # then sets the real path and re-runs. On the None pass we SKIP discovery so we
        # never fail-loud on a missing default dir (fatal on the cluster). Discovery +
        # fail-loud runs only when pmt_motion_paths is explicitly set.
        # _DEFAULT_MOTION_PATHS is kept for reference but is NOT an auto-fallback.
        if not self.pmt_motion_paths:
            print("[PMTMultiMotionFlat] deferred motion discovery (no pmt_motion_paths yet)")
        else:
            discovery = find_motion_files(motion_paths=self.pmt_motion_paths, strict=False)
            motion_files = discovery.files
            if not motion_files:
                raise ValueError(
                    "No multi-motion files found. "
                    f"Searched paths: {discovery.searched_paths or self.pmt_motion_paths}"
                )
            print(f"[PMTMultiMotionFlat] Discovered {len(motion_files)} motion files")
            for index, motion_file in enumerate(motion_files[:5]):
                print(f"  [{index}] {os.path.basename(motion_file)}")
            if len(motion_files) > 5:
                print(f"  ... and {len(motion_files) - 5} more")
            self.commands.motion.motion_files = motion_files

        # storage_mode=streaming -> swap to the memory-bounded streaming command
        # (plan §9b "TWO command classes" via a flag). Copies the eager command's
        # settings, mirroring G1MultiMotionV2StreamingEnvCfg.__post_init__:381-422.
        if str(self.pmt_storage_mode) == "streaming":
            base = self.commands.motion
            self.commands.motion = mdp.StreamingMultiMotionCommandV2Cfg(
                asset_name=base.asset_name,
                resampling_time_range=base.resampling_time_range,
                debug_vis=base.debug_vis,
                motion_files=base.motion_files,
                anchor_body=base.anchor_body,
                body_names=base.body_names,
                storage_device=base.storage_device,
                use_fp16=True,           # streaming forces fp16
                chunk_length=0,          # ragged storage -> no chunking
                load_human_motion=False,
                sampler_type=base.sampler_type,
                enable_frame_sampling=base.enable_frame_sampling,
                adaptive_beta=base.adaptive_beta,
                adaptive_alpha=base.adaptive_alpha,
                adaptive_uniform_ratio=base.adaptive_uniform_ratio,
                adaptive_update_interval=base.adaptive_update_interval,
                adaptive_kernel_size=base.adaptive_kernel_size,
                adaptive_kernel_lambda=base.adaptive_kernel_lambda,
                motion_fps=base.motion_fps,
                bin_duration=base.bin_duration,
                pose_range=base.pose_range,
                velocity_range=base.velocity_range,
                joint_position_range=base.joint_position_range,
                default_reset_joint_prob=base.default_reset_joint_prob,
                default_reset_joint_lerp_range=base.default_reset_joint_lerp_range,
                default_reset_frame_range=base.default_reset_frame_range,
                default_reset_zero_vel=base.default_reset_zero_vel,
                update_action_offset_with_ref=base.update_action_offset_with_ref,
                action_term_name=base.action_term_name,
                max_working_set=int(self.pmt_max_working_set),
                num_load_workers=int(self.pmt_num_load_workers),
                use_process_pool=bool(self.pmt_use_process_pool),
            )
