"""Stepping-stone teacher/student distillation env cfg.

PMT plan §6 Phase 2.2 + §10 PART A. Reproduces the STRUCTURE of the old
``G1SteppingStoneDistillEnvCfg`` (distill_stepping_stone_env_cfg.py:489):
  - teacher obs group (cmd "motion", privileged/optimized, no corruption);
  - student policy obs group (cmd "student_motion", raw, with noise + height_scan);
  - critic obs group (cmd "student_motion" + height_scan);
  - auxiliary motion_target / anchor_target groups (teacher-vs-student deltas);
  - a PAIRED command: ``motion`` (MultiMotionCommandV2, optimized clips) +
    ``student_motion`` (SyncedStudentMultiMotionCommandV2, raw clips synced to motion).

It REUSES the stepping-stone scene (mesh + height scanner + contact sensor), rewards,
and terminations from ``pmt_tasks.env_cfgs.pmt.stepping_stone``. The DistillationRunner
routes the ``teacher`` obs set to the teacher MLP and ``policy`` to the student MLP.

The builder injects the terrain mesh + paired optimized/raw clip dirs + decimation/dt
via the ``pmt_*`` attributes that ``__post_init__`` reads.
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import pmt_tasks.mdp as mdp
from pmt_tasks.env_cfgs.pmt.stepping_stone import (
    END_EFFECTOR_BODY_NAMES,
    TRACKED_BODY_NAMES,
    SteppingStoneRewardsCfg,
    SteppingStoneSceneCfg,
    SteppingStoneTerminationsCfg,
)
from pmt_tasks.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from pmt_tasks.tracking_env_cfg import TrackingEnvCfg
from pmt_tasks.path_defaults import terrain_asset_path, terrain_motion_path
from pmt_tasks.utils.motion_paths import pair_motion_files

# Generic default paired clip dirs (the builder overrides from the resolved config).
_DEFAULT_OPTIMIZED = [
    terrain_motion_path(
        "terrain_positive_stepping_stone_with_stairs",
        "walk1_subject1_stair",
        "optimized",
    ),
]
_DEFAULT_RAW = [
    terrain_motion_path(
        "terrain_positive_stepping_stone_with_stairs",
        "walk1_subject1_stair",
        "raw",
    ),
]
_DEFAULT_MESH = terrain_asset_path("positive_stepping_with_stairs.stl")


@configclass
class DistillSteppingStoneCommandsCfg:
    """Paired optimized/raw multi-motion commands (mirror of old class:219)."""

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
        pose_range={},
        velocity_range={},
        joint_position_range=(-0.1, 0.1),
        default_reset_joint_prob=0.2,
        default_reset_joint_lerp_range=(0.85, 1.0),
        default_reset_frame_range=(0, 20),
        default_reset_zero_vel=True,
    )

    student_motion = mdp.SyncedStudentMultiMotionCommandV2Cfg(
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
        sampler_type="uniform",
        enable_frame_sampling=False,
        adaptive_beta=2,
        adaptive_alpha=0.001,
        adaptive_uniform_ratio=0.1,
        adaptive_update_interval=240,
        adaptive_kernel_size=50,
        adaptive_kernel_lambda=0.8,
        pose_range={},
        velocity_range={},
        joint_position_range=(-0.1, 0.1),
        sync_command_name="motion",
    )


@configclass
class DistillSteppingStoneObservationsCfg:
    """Teacher/student/critic + auxiliary delta obs (mirror of old class:357)."""

    @configclass
    class TeacherCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}, history_length=10, noise=Unoise(n_min=-0.15, n_max=0.15))
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "motion"}, history_length=1, noise=Unoise(n_min=-0.15, n_max=0.15))
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, history_length=10, noise=Unoise(n_min=-0.05, n_max=0.15))
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, history_length=10, noise=Unoise(n_min=-0.15, n_max=0.15))
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"}, history_length=10, noise=Unoise(n_min=-0.06, n_max=0.06))
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"}, history_length=10, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10, noise=Unoise(n_min=-0.3, n_max=0.3))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10, noise=Unoise(n_min=-0.12, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action, history_length=10)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "student_motion"}, history_length=10, noise=Unoise(n_min=-0.15, n_max=0.15))
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "student_motion"}, history_length=1, noise=Unoise(n_min=-0.15, n_max=0.15))
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "student_motion"}, history_length=10, noise=Unoise(n_min=-0.05, n_max=0.15))
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "student_motion"}, history_length=10, noise=Unoise(n_min=-0.15, n_max=0.15))
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "student_motion"}, history_length=10, noise=Unoise(n_min=-0.06, n_max=0.06))
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "student_motion"}, history_length=10, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10, noise=Unoise(n_min=-0.3, n_max=0.3))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10, noise=Unoise(n_min=-0.12, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action, history_length=10)
        height_scan = ObsTerm(
            func=mdp.height_scan_fill,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "nan_dropout_prob": 0.05},
            clip=(-3.0, 3.0),
            noise=Unoise(n_min=-0.06, n_max=0.06),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "student_motion"}, history_length=10)
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "student_motion"}, history_length=1)
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "student_motion"}, history_length=10)
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "student_motion"}, history_length=10)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "student_motion"}, history_length=10)
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "student_motion"}, history_length=10)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        actions = ObsTerm(func=mdp.last_action, history_length=10)
        height_scan = ObsTerm(
            func=mdp.height_scan_fill,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "nan_dropout_prob": 0.05},
            clip=(-3.0, 3.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MotionTargetCfg(ObsGroup):
        motion_delta = ObsTerm(
            func=mdp.sonic_robot_motion_delta,
            params={"teacher_command_name": "motion", "student_command_name": "student_motion"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AnchorTargetCfg(ObsGroup):
        anchor_delta = ObsTerm(
            func=mdp.motion_anchor_delta_b,
            params={"teacher_command_name": "motion", "student_command_name": "student_motion"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    motion_target: MotionTargetCfg = MotionTargetCfg()
    anchor_target: AnchorTargetCfg = AnchorTargetCfg()


@configclass
class PMTSteppingStoneDistillEnvCfg(TrackingEnvCfg):
    """Teacher/student distillation env on the stepping-stone terrain.

    Structurally equivalent to the old G1SteppingStoneDistillEnvCfg. The builder sets
    these instance attributes before __post_init__ runs:
      - ``pmt_mesh_path``: terrain mesh .stl
      - ``pmt_optimized_motion_paths`` / ``pmt_raw_motion_paths``: paired clip dirs
      - ``pmt_decimation`` / ``pmt_sim_dt``: per-motion control rate (§3a)
    """

    pmt_mesh_path: str = _DEFAULT_MESH
    pmt_optimized_motion_paths: Optional[Union[str, List[str]]] = None
    pmt_raw_motion_paths: Optional[Union[str, List[str]]] = None
    pmt_decimation: int = 4
    pmt_sim_dt: float = 0.005
    pmt_sampler_type: str = "bin_adaptive"

    scene: SteppingStoneSceneCfg = SteppingStoneSceneCfg(num_envs=64, env_spacing=0.0)
    observations: DistillSteppingStoneObservationsCfg = DistillSteppingStoneObservationsCfg()
    commands: DistillSteppingStoneCommandsCfg = DistillSteppingStoneCommandsCfg()
    rewards: SteppingStoneRewardsCfg = SteppingStoneRewardsCfg()
    terminations: SteppingStoneTerminationsCfg = SteppingStoneTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.decimation = int(self.pmt_decimation)
        self.sim.dt = float(self.pmt_sim_dt)
        self.sim.render_interval = self.decimation

        self.scene.terrain.mesh_path = self.pmt_mesh_path
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 0.0
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        self.commands.motion.anchor_body = "torso_link"
        self.commands.motion.body_names = TRACKED_BODY_NAMES
        self.commands.motion.sampler_type = str(self.pmt_sampler_type)
        self.commands.student_motion.anchor_body = "torso_link"
        self.commands.student_motion.body_names = TRACKED_BODY_NAMES

        # match old teacher: drop these domain-randomization events.
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.base_com = None

        # tighten terminations to the stepping-stone end-effector set.
        if getattr(self.terminations, "ee_body_pos", None) is not None:
            self.terminations.ee_body_pos.params["body_names"] = END_EFFECTOR_BODY_NAMES

        # resolve paired optimized/raw motions (fail-loud) — DEFERRED until BOTH paired
        # path attrs are set. The builder constructs with both None (first pass), then
        # sets the real paths and re-runs. On the None pass we SKIP discovery so we never
        # fail-loud on missing default dirs (fatal on the cluster). Pairing + fail-loud
        # runs only when the paths are explicitly set. The _DEFAULT_* constants are kept
        # for reference but are NOT auto-fallbacks.
        if not self.pmt_optimized_motion_paths and not self.pmt_raw_motion_paths:
            print("[PMTSteppingStoneDistill] deferred motion discovery (no pmt_*_motion_paths yet)")
        else:
            paired = pair_motion_files(
                self.pmt_optimized_motion_paths, self.pmt_raw_motion_paths, strict=True
            )
            if len(paired.optimized_files) == 0:
                raise ValueError(
                    "No paired stepping-stone motions found for distillation. "
                    f"optimized: {paired.optimized_searched_paths}, raw: {paired.raw_searched_paths}"
                )
            self.commands.motion.motion_files = paired.optimized_files
            self.commands.student_motion.motion_files = paired.raw_files
            print(f"[PMTSteppingStoneDistill] Paired {len(paired.optimized_files)} optimized/raw motions")
            for index, motion_file in enumerate(paired.optimized_files[:5]):
                print(f"  [{index}] {os.path.basename(motion_file)}")
            if len(paired.optimized_files) > 5:
                print(f"  ... and {len(paired.optimized_files) - 5} more")


# =====================================================================================
# vision-transformer student / blind-transformer teacher latent-anchor distillation
# =====================================================================================
#
# PMT plan §6 / §10 PART A. Faithful port of the original
#   G1SteppingStoneStudentLatentAnchorDistillEnvCfg
# (distill_stepping_stone_env_cfg.py:935), whose parents are G1SteppingStoneDistillEnvCfg
# (:909) and G1SteppingStoneDistillEnvCfg (:489). The student is a vision-augmented transformer
# (VisionTransformerActorCritic) and the teacher is the BLIND TransformerActorCritic whose
# trained ckpt is loaded by the VisionStudentTeacher wrapper. The obs structure exposes
# BOTH the student transformer groups (proprio[_history], student command/anchor windows,
# vel_gt, anchor_gt, vision, foot_traj_target) and the matching teacher groups
# (teacher, teacher_command_window, teacher_motion_anchor_delta_window, teacher_anchor_body_pose)
# so the runner's obs_groups map (see G1SteppingStoneVisionLatentAnchorDistillRunnerCfg)
# can route the teacher set to the frozen TransformerActorCritic and the student set to the
# VisionTransformerActorCritic.

FOOT_TRAJ_BODY_NAMES: List[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


@configclass
class TransformerDistillSteppingStoneObservationsCfg:
    """Teacher/student transformer observation groups for vision-conditioned distillation.

    Faithful port of the original TransformerDistillSteppingStoneObservationsCfg
    (distill_stepping_stone_env_cfg.py:546). The teacher group (history_length=1,
    command_name="motion") matches the blind transformer teacher checkpoint; the student
    groups read the synced "student_motion" command.
    """

    @configclass
    class TeacherCfg(ObsGroup):
        # IMPORTANT: this teacher group MUST match the obs the trained blind transformer teacher
        # ckpt (pmt_stepping_stone, G1SteppingStonePPORunnerCfg) was trained on:
        # its actor "policy" set = [policy, proprio] where the "policy" group is the
        # stepping-stone PolicyCfg (history_length=10, command_name="motion"). The teacher
        # ckpt signature has actor_obs_dim=3533, so the teacher group here reproduces that
        # exact PolicyCfg (history_length=10) — NOT a history_length=1 collapse (which gave
        # actor_obs_dim=959 and the schema-mismatch load failure).
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}, history_length=10)
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "motion"}, history_length=1)
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, history_length=10)
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, history_length=10)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"}, history_length=10)
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"}, history_length=10)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        actions = ObsTerm(func=mdp.last_action, history_length=10)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "student_motion"}, history_length=10, noise=Unoise(n_min=-0.15, n_max=0.15))
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "student_motion"}, history_length=1, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10, noise=Unoise(n_min=-0.12, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action, history_length=10)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "student_motion"}, history_length=1)
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "student_motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "student_motion"}, history_length=1)
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "student_motion"}, history_length=1)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "student_motion"}, history_length=1)
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "student_motion"}, history_length=1)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=1)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=1)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=1)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=1)
        actions = ObsTerm(func=mdp.last_action, history_length=1)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class StudentCommandWindowCfg(ObsGroup):
        command_window = ObsTerm(
            func=mdp.command_window,
            params={"command_name": "student_motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentMotionAnchorDeltaWindowCfg(ObsGroup):
        motion_anchor_delta_window = ObsTerm(
            func=mdp.motion_anchor_delta_window,
            params={"command_name": "student_motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AnchorBodyPoseCfg(ObsGroup):
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "student_motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "student_motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherAnchorBodyPoseCfg(ObsGroup):
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherCommandWindowCfg(ObsGroup):
        command_window = ObsTerm(
            func=mdp.command_window,
            params={"command_name": "motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherMotionAnchorDeltaWindowCfg(ObsGroup):
        motion_anchor_delta_window = ObsTerm(
            func=mdp.motion_anchor_delta_window,
            params={"command_name": "motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelGtXYZCfg(ObsGroup):
        vel_gt = ObsTerm(func=mdp.vel_gt)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelGtXYZYawCfg(ObsGroup):
        vel_gt = ObsTerm(func=mdp.vel_yaw_gt)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AnchorGtCfg(ObsGroup):
        anchor_gt = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "student_motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class FootTrajTargetCfg(ObsGroup):
        foot_traj = ObsTerm(
            func=mdp.foot_traj_delta_target,
            params={
                "teacher_command_name": "motion",
                "student_command_name": "student_motion",
                "body_names": FOOT_TRAJ_BODY_NAMES,
                "window_size": 5,
                "stride": 1,
                "flatten": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VisionCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan_for_vision,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "nan_dropout_prob": 0.1,
                "append_validity_mask": True,
                "noise_range": (-0.06, 0.06),
                "clip_range": (-3.0, 3.0),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    proprio: ProprioCfg = ProprioCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    command_window: StudentCommandWindowCfg = StudentCommandWindowCfg()
    motion_anchor_delta_window: StudentMotionAnchorDeltaWindowCfg = StudentMotionAnchorDeltaWindowCfg()
    anchor_body_pose: AnchorBodyPoseCfg = AnchorBodyPoseCfg()
    teacher_anchor_body_pose: TeacherAnchorBodyPoseCfg = TeacherAnchorBodyPoseCfg()
    teacher_command_window: TeacherCommandWindowCfg = TeacherCommandWindowCfg()
    teacher_motion_anchor_delta_window: TeacherMotionAnchorDeltaWindowCfg = TeacherMotionAnchorDeltaWindowCfg()
    vel_gt_xyz: VelGtXYZCfg = VelGtXYZCfg()
    vel_gt_xyz_yaw: VelGtXYZYawCfg = VelGtXYZYawCfg()
    anchor_gt: AnchorGtCfg = AnchorGtCfg()
    foot_traj_target: FootTrajTargetCfg = FootTrajTargetCfg()
    vision: VisionCfg = VisionCfg()


@configclass
class PMTSteppingStoneVisionLatentAnchorDistillEnvCfg(PMTSteppingStoneDistillEnvCfg):
    """vision-transformer student / blind-transformer teacher latent-anchor distillation env.

    Faithful port of G1SteppingStoneStudentLatentAnchorDistillEnvCfg: swaps in the
    transformer teacher/student/vision observation groups, makes the student act around the raw
    reference trajectory (student_motion.update_action_offset_with_ref=True, action
    scale 1.0), and removes the student anchor_body_pose group (latent-anchor variant:
    the student's anchor estimator is supervised in latent space, not from a privileged
    anchor-body-pose obs).
    """

    observations: TransformerDistillSteppingStoneObservationsCfg = TransformerDistillSteppingStoneObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The student acts around the raw reference trajectory; teacher actions are
        # converted onto this reference frame inside the student-teacher wrapper.
        self.actions.joint_pos.scale = 1.0
        self.commands.motion.update_action_offset_with_ref = False
        self.commands.student_motion.update_action_offset_with_ref = True

        # LatentAnchor variant: drop only the student anchor-body-pose group.
        self.observations.anchor_body_pose = None
