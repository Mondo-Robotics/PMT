"""ADD (discriminator) multi-motion flat env cfg, assembled from ported mdp cfgs.

PMT plan §10 PART A (hybrid-reuse-mdp-cfgs) + §6 Phase-1 widened slice. This module
reproduces the STRUCTURE of the old ``ADDG1MultiMotionV2EnvCfg``
(source: .../config/g1/multi_motion_env_cfg.py:455 -> G1MultiMotionV2EnvCfg:167 ->
TrackingEnvCfg) by composing the ported Isaac Lab manager cfgs from ``pmt_tasks.mdp``
and ``pmt_tasks.tracking_env_cfg``.

It adds the two ADD discriminator obs GROUPS (``add_disc_obs`` =
``mdp.add_disc_obs_agent`` 230D, ``add_disc_demo`` = ``mdp.add_disc_obs_demo`` 230D)
on top of the standard policy/critic groups, on a FLAT plane terrain, with the V2
multi-motion command.

The data-driven values (motion clip dir, decimation/sim.dt) are injected by the
builder (``pmt_tasks.builder.build_env_cfg``) from the resolved OmegaConf config via
the ``pmt_*`` class attributes that ``__post_init__`` reads.
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from pmt_tasks.tracking_env_cfg import (
    MySceneCfg,
    ObservationsCfg,
    TrackingEnvCfg,
)
from pmt_tasks.path_defaults import motion_path
from pmt_tasks.utils.motion_paths import find_motion_files

# MimicKit ADD G1 discriminator key bodies (the 4 key bodies whose positions enter
# the discriminator body-pos block). The anchor body ("torso_link") is prepended to
# the command's ``body_names`` (see ADD_BODY_NAMES) so the MultiMotionCommandV2 can
# resolve ``body_names.index(anchor_body)``; the ADD disc obs slices the anchor row
# back out so only these key-body positions are used (MimicKit add_g1_env key_bodies).
# NB: the Unitree G1 29-DOF description has NO ``head_link`` body, so the key set is the
# 4 limb-end bodies (feet + hands); torso_link already serves as the upper-body anchor.
ADD_KEY_BODY_NAMES: List[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# The motion command's tracked-body set = anchor body (index 0) + the 4 ADD key bodies.
ADD_ANCHOR_BODY: str = "torso_link"
TRACKED_BODY_NAMES: List[str] = [ADD_ANCHOR_BODY, *ADD_KEY_BODY_NAMES]

# Generic default clip dir so the env cfg constructs standalone; builder overrides.
_DEFAULT_MOTION_PATHS = [motion_path("debug", "robot_lafan1")]


@configclass
class ADDFlatSceneCfg(MySceneCfg):
    """Flat plane scene + contact sensor (no height scanner; ADD is blind)."""

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
class ADDMultiMotionCommandsCfg:
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
class ADDObservationsCfg(ObservationsCfg):
    """Standard policy/critic groups + the two ADD discriminator groups.

    Mirror of the old ADDObservationsCfg (multi_motion_env_cfg.py:431). Both
    discriminator groups carry a single 230D term each (add_disc_obs_agent /
    add_disc_obs_demo).
    """

    @configclass
    class AddDiscObsCfg(ObsGroup):
        add_disc_obs_agent = ObsTerm(func=mdp.add_disc_obs_agent, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AddDiscDemoCfg(ObsGroup):
        add_disc_obs_demo = ObsTerm(func=mdp.add_disc_obs_demo, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    add_disc_obs: AddDiscObsCfg = AddDiscObsCfg()
    add_disc_demo: AddDiscDemoCfg = AddDiscDemoCfg()


@configclass
class PMTADDMultiMotionEnvCfg(TrackingEnvCfg):
    """ADD multi-motion flat env, composed from ported mdp cfgs.

    Structurally equivalent to the old ADDG1MultiMotionV2EnvCfg (flat plane + V2
    multi-motion command + standard policy/critic + the two 230D discriminator
    groups + tightened anchor/ee termination thresholds at 0.4).

    Builder-injected attributes (set before __post_init__ re-runs):
      - ``pmt_motion_paths``: list of motion dirs/files (from ${paths.MOTION_ROOT}/...)
      - ``pmt_decimation`` / ``pmt_sim_dt``: control rate (§3a; ADD flat -> 4/0.005)
    """

    pmt_motion_paths: Optional[Union[str, List[str]]] = None
    pmt_decimation: int = 4
    pmt_sim_dt: float = 0.005

    scene: ADDFlatSceneCfg = ADDFlatSceneCfg(num_envs=64, env_spacing=0.0)
    observations: ADDObservationsCfg = ADDObservationsCfg()
    commands: ADDMultiMotionCommandsCfg = ADDMultiMotionCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        # config-driven decimation/sim.dt (§3a). ADD flat = 4 / 0.005.
        self.decimation = int(self.pmt_decimation)
        self.sim.dt = float(self.pmt_sim_dt)
        self.sim.render_interval = self.decimation

        # robot + action scale (standard multi-motion V2 base, NOT residual).
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 0.0
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # command wiring.
        self.commands.motion.anchor_body = "torso_link"
        self.commands.motion.body_names = TRACKED_BODY_NAMES

        # tightened thresholds (old ADDG1MultiMotionV2EnvCfg.__post_init__:462-463).
        if getattr(self.terminations, "ee_body_pos", None) is not None:
            self.terminations.ee_body_pos.params["threshold"] = 0.4
        if getattr(self.terminations, "anchor_pos", None) is not None:
            self.terminations.anchor_pos.params["threshold"] = 0.4

        # resolve motion files (fail-loud) — DEFERRED until pmt_motion_paths is set.
        # The builder constructs with pmt_motion_paths=None (first pass), then sets the
        # real path and re-runs. On the None pass we SKIP discovery so we never fail-loud
        # on a missing default dir (fatal on the cluster). Discovery + fail-loud runs only
        # when pmt_motion_paths is explicitly set. _DEFAULT_MOTION_PATHS is kept for
        # reference but is NOT an auto-fallback.
        if not self.pmt_motion_paths:
            print("[PMTADDMultiMotion] deferred motion discovery (no pmt_motion_paths yet)")
        else:
            discovery = find_motion_files(motion_paths=self.pmt_motion_paths, strict=False)
            motion_files = discovery.files
            if not motion_files:
                raise ValueError(
                    "No ADD multi-motion files found. "
                    f"Searched paths: {discovery.searched_paths or self.pmt_motion_paths}"
                )
            print(f"[PMTADDMultiMotion] Discovered {len(motion_files)} motion files")
            for index, motion_file in enumerate(motion_files[:5]):
                print(f"  [{index}] {os.path.basename(motion_file)}")
            if len(motion_files) > 5:
                print(f"  ... and {len(motion_files) - 5} more")
            self.commands.motion.motion_files = motion_files
