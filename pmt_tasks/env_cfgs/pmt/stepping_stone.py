"""stepping-stone env cfg, assembled from ported pmt_tasks.mdp cfgs.

PMT plan §10 PART A (hybrid-reuse-mdp-cfgs). This module reproduces the STRUCTURE
of the old ``G1SteppingStoneMultiMotionEnvCfg``
(source: .../config/g1/stepping_stone_env_cfg.py:710) by composing the ported
Isaac Lab manager cfgs from ``pmt_tasks.mdp`` and ``pmt_tasks.tracking_env_cfg``.

The VALUES that vary per machine/task — the terrain mesh path and the motion
clip directory — are NOT hard-coded here. They are injected by the builder
(``pmt_tasks.builder.build_env_cfg``) from the resolved OmegaConf config, via the
``pmt_mesh_path`` / ``pmt_motion_paths`` class attributes that ``__post_init__``
reads. Everything else (scene sensors, obs group structure, reward terms,
termination thresholds, decimation/dt) is structurally identical to the old
class so the produced env is observation/action equivalent (and resume-compatible)
with the old ``SteppingStone-G1-v0`` task.

Fields still defaulted from the old class (Phase-2 parametrization TODOs — see
report): the exact obs term list, reward weights, termination thresholds, and
sensor geometry are baked here rather than driven from YAML. The builder DOES
override the data-driven values the spec calls out: mesh path, motion files,
decimation, sim.dt, reward weights, obs history_length.
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from pmt_tasks.tracking_env_cfg import (
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    TrackingEnvCfg,
)
from pmt_tasks.path_defaults import terrain_asset_path, terrain_motion_path
from pmt_tasks.utils.motion_paths import find_motion_files
from pmt_tasks.utils.terrain_importer_compat import MeshCompatibleTerrainImporterCfg

# 14 tracked bodies (spec / old stepping_stone_env_cfg.py:201).
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
END_EFFECTOR_BODY_NAMES: List[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# Generic defaults so the env cfg constructs standalone (the builder overrides
# these from the resolved config; see build_env_cfg).
_DEFAULT_MESH = terrain_asset_path("positive_stepping_with_stairs.stl")
_DEFAULT_MOTION_PATHS = [
    terrain_motion_path(
        "terrain_positive_stepping_stone_with_stairs",
        "walk1_subject1_stair",
        "optimized",
    ),
]


@configclass
class SteppingStoneSceneCfg(MySceneCfg):
    """Scene with the stepping-stone mesh + height scanner + contact sensor.

    Mirror of old SteppingStoneSceneCfg (stepping_stone_env_cfg.py:224). The mesh
    path is a placeholder; the builder rewrites ``scene.terrain.mesh_path`` from the
    resolved config before the scene is built.
    """

    terrain = MeshCompatibleTerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="mesh",
        mesh_path=_DEFAULT_MESH,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/terrain"],
        max_distance=30.0,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


@configclass
class SteppingStoneCommandsCfg:
    """Multi-motion command with zero reset offsets for terrain-anchored clips.

    Mirror of old SteppingStoneCommandsCfg (stepping_stone_env_cfg.py:381). The
    ``motion_files`` placeholder is rewritten by the builder/__post_init__ via the
    discovered clip list.
    """

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
        joint_position_range=(0.0, 0.0),
    )


# --- observations: the transformer obs stack (structurally identical) -----
#
# Reproduces the old TransformerSteppingStoneObservationsCfg
# (stepping_stone_env_cfg.py:504), which inherits the SteppingStone policy/critic
# groups (history_length 10 / 1) and adds the transformer obs groups that the old
# MultiMotionV2ObservationsCfg defined (multi_motion_env_cfg.py:479).


@configclass
class SteppingStoneObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
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
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}, history_length=1)
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, history_length=1)
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, history_length=1)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"}, history_length=1)
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"}, history_length=1)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=1)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=1)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=1)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=1)
        actions = ObsTerm(func=mdp.last_action, history_length=1)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TransformerSteppingStoneObservationsCfg(SteppingStoneObservationsCfg):
    """SteppingStone obs + transformer obs groups (resume-faithful structure)."""

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
    class CommandWindowCfg(ObsGroup):
        command_window = ObsTerm(
            func=mdp.command_window,
            params={"command_name": "motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MotionAnchorDeltaWindowCfg(ObsGroup):
        motion_anchor_delta_window = ObsTerm(
            func=mdp.motion_anchor_delta_window,
            params={"command_name": "motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AnchorBodyPoseCfg(ObsGroup):
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})

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
        anchor_gt = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    command_window: CommandWindowCfg = CommandWindowCfg()
    motion_anchor_delta_window: MotionAnchorDeltaWindowCfg = MotionAnchorDeltaWindowCfg()
    anchor_body_pose: AnchorBodyPoseCfg = AnchorBodyPoseCfg()
    vel_gt_xyz: VelGtXYZCfg = VelGtXYZCfg()
    vel_gt_xyz_yaw: VelGtXYZYawCfg = VelGtXYZYawCfg()
    anchor_gt: AnchorGtCfg = AnchorGtCfg()


@configclass
class SteppingStoneRewardsCfg(RewardsCfg):
    """Tighter tracking rewards (mirror of old SteppingStoneRewardsCfg:586)."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1,
        params={"command_name": "motion", "std": 0.2},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.35},
    )
    motion_foot_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.1,
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        },
    )
    motion_foot_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 1.0,
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        },
    )
    feet_lateral_contact = RewTerm(
        func=mdp.feet_lateral_contact_force_l2,
        weight=-0.03,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_knee_link",
                    "right_knee_link",
                ],
            ),
            "threshold": 5.0,
        },
    )


@configclass
class SteppingStoneTerminationsCfg(TerminationsCfg):
    """Mirror of old SteppingStoneTerminationsCfg:644."""

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos,
        params={"command_name": "motion", "threshold": 0.35},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos,
        params={
            "command_name": "motion",
            "threshold": 0.35,
            "body_names": END_EFFECTOR_BODY_NAMES,
        },
    )


@configclass
class PMTSteppingStoneEnvCfg(TrackingEnvCfg):
    """stepping-stone env, composed from ported mdp cfgs.

    Structurally equivalent to the old G1SteppingStoneMultiMotionEnvCfg
    (scene + transformer obs + tighter rewards + anchor/ee terminations + residual action).

    The builder sets these instance attributes before __post_init__ runs:
      - ``pmt_mesh_path``: terrain mesh .stl (from ${paths.TERRAIN_ROOT}/...)
      - ``pmt_motion_paths``: list of motion dirs/files (from ${paths.MOTION_ROOT}/...)
      - ``pmt_decimation`` / ``pmt_sim_dt``: per-motion control rate (§3a)
      - ``pmt_reward_weights``: {term_name: weight} data overrides (§9b)
      - ``pmt_history_length``: actor obs history window (left at 10 by default)
    """

    # builder-injected config values (defaults are the spec's verified values, so
    # the env still builds if the builder does not override them).
    pmt_mesh_path: str = _DEFAULT_MESH
    pmt_motion_paths: Optional[Union[str, List[str]]] = None
    pmt_decimation: int = 4
    pmt_sim_dt: float = 0.005
    pmt_reward_weights: Optional[dict] = None
    pmt_history_length: int = 10

    scene: SteppingStoneSceneCfg = SteppingStoneSceneCfg(num_envs=64, env_spacing=0.0)
    observations: TransformerSteppingStoneObservationsCfg = TransformerSteppingStoneObservationsCfg()
    commands: SteppingStoneCommandsCfg = SteppingStoneCommandsCfg()
    rewards: SteppingStoneRewardsCfg = SteppingStoneRewardsCfg()
    terminations: SteppingStoneTerminationsCfg = SteppingStoneTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # config-driven decimation/sim.dt (§3a dt-from-motion).
        self.decimation = int(self.pmt_decimation)
        self.sim.dt = float(self.pmt_sim_dt)
        self.sim.render_interval = self.decimation

        # terrain mesh from config (§5 paths).
        self.scene.terrain.mesh_path = self.pmt_mesh_path

        # robot + residual action setup (residual action: q_target = q_ref + a, scale 1.0).
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 0.0
        self.actions.joint_pos.scale = 1.0
        self.commands.motion.update_action_offset_with_ref = True

        # command wiring.
        self.commands.motion.anchor_body = "torso_link"
        self.commands.motion.body_names = TRACKED_BODY_NAMES
        self.commands.motion.sampler_type = "bin_adaptive"

        # match old teacher: drop these domain-randomization events.
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.base_com = None

        # actor obs history window (§9b: 1-vs-10 is a param). Default 10 keeps the
        # resume-faithful structure; builder may override pmt_history_length.
        if int(self.pmt_history_length) != 10:
            for term_name in (
                "command", "motion_anchor_pos_b", "motion_anchor_ori_b",
                "body_pos", "body_ori", "base_lin_vel", "base_ang_vel",
                "joint_pos", "joint_vel", "actions",
            ):
                term = getattr(self.observations.policy, term_name, None)
                if term is not None and getattr(term, "history_length", None) not in (None, 1):
                    term.history_length = int(self.pmt_history_length)

        # config-driven reward weights (§9b reward-as-data, shared helper).
        if self.pmt_reward_weights:
            mdp.apply_reward_weight_set(self.rewards, self.pmt_reward_weights)

        # resolve motion files (fail-loud) — DEFERRED until pmt_motion_paths is set.
        # The builder constructs the env cfg with pmt_motion_paths=None (running
        # __post_init__ once), then sets the real path and re-runs __post_init__.
        # On the construction pass (None) we SKIP discovery so we never fail-loud on a
        # missing default dir (which is fatal on the cluster). Discovery + fail-loud
        # runs only when pmt_motion_paths is explicitly set. _DEFAULT_MOTION_PATHS is
        # kept for reference but is NOT an auto-fallback.
        if not self.pmt_motion_paths:
            print("[PMTSteppingStone] deferred motion discovery (no pmt_motion_paths yet)")
        else:
            discovery = find_motion_files(motion_paths=self.pmt_motion_paths, strict=False)
            motion_files = discovery.files
            if not motion_files:
                raise ValueError(
                    "No stepping-stone motion files found. "
                    f"Searched paths: {discovery.searched_paths or self.pmt_motion_paths}"
                )
            print(f"[PMTSteppingStone] Discovered {len(motion_files)} motion files")
            for index, motion_file in enumerate(motion_files[:5]):
                print(f"  [{index}] {os.path.basename(motion_file)}")
            if len(motion_files) > 5:
                print(f"  ... and {len(motion_files) - 5} more")
            self.commands.motion.motion_files = motion_files
