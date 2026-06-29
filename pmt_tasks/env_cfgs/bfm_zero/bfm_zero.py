"""BFM-Zero (faithful FB-CPR-Aux) env cfgs for PMT G1 tasks.

These cfgs turn existing PMT motion-tracking envs into physics + mocap sources for the
off-policy, latent-conditioned BFM-Zero algorithm:

- Actions are direct default-offset joint targets, not residual actions.
- Env reward is zero; FB-CPR-Aux owns the learning signal.
- Terminations are time-out only.
- The motion command keeps the full 30-body BFM privileged body set so online and expert
  observations use the same body ordering.
"""

from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from motion_tracking_rl.bfm_zero.body_names import PRIVILEGED_BODY_NAMES
from pmt_tasks.env_cfgs.multi_motion_flat import MultiMotionFlatSceneCfg, PMTMultiMotionFlatEnvCfg
from pmt_tasks.env_cfgs.pmt.terrain_flat_mix import PMTTerrainFlatMixEnvCfg
from pmt_tasks.robots.g1 import G1_ACTION_SCALE
from pmt_tasks.path_defaults import motion_path, sonic_path


def _split_env_paths(value: str) -> list[str]:
    return [os.path.expanduser(path.strip()) for path in value.split(os.pathsep) if path.strip()]


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_bool(*names: str, default: bool = False) -> bool:
    value = _first_env(*names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bfm_zero_flat_motion_paths() -> list[str]:
    """Resolve flat robot-motion roots for the pure-plane BFM-Zero task."""

    env_paths = _first_env(
        "PMT_BFM_ZERO_FLAT_MOTION_PATHS",
        "WBT_BFM_ZERO_FLAT_MOTION_PATHS",
        "PMT_TERRAIN_FLAT_STREAM_FLAT_PATHS",
        "WBT_TERRAIN_FLAT_STREAM_FLAT_PATHS",
    )
    if env_paths:
        return _split_env_paths(env_paths)

    full_lafan = sonic_path("lafan1", "robot_lafan1")
    if os.path.exists(full_lafan):
        return [full_lafan]
    return [motion_path("debug", "robot_lafan1")]


@configclass
class BFMZeroZeroRewardsCfg:
    """Zero learning reward: FB-CPR owns the objective; aux penalties are adapter-side."""

    alive = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class BFMZeroTimeoutTerminationsCfg:
    """Time-out-only termination; no tracking-error termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BFMZeroG1TerrainFlatStreamingEnvCfg(PMTTerrainFlatMixEnvCfg):
    """BFM-Zero variant over PMT's terrain+flat mocap source."""

    rewards: BFMZeroZeroRewardsCfg = BFMZeroZeroRewardsCfg()
    terminations: BFMZeroTimeoutTerminationsCfg = BFMZeroTimeoutTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.update_action_offset_with_ref = False
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.body_names = list(PRIVILEGED_BODY_NAMES)
        assert self.commands.motion is not None


@configclass
class BFMZeroG1FlatMultiMotionV2EnvCfg(PMTMultiMotionFlatEnvCfg):
    """Pure-plane BFM-Zero task using PMT's eager MultiMotionCommandV2 loader."""

    pmt_motion_paths: list[str] = _bfm_zero_flat_motion_paths()
    pmt_sampler_type: str = "uniform"
    pmt_storage_mode: str = "eager"
    scene: MultiMotionFlatSceneCfg = MultiMotionFlatSceneCfg(num_envs=4096, env_spacing=6.0)
    rewards: BFMZeroZeroRewardsCfg = BFMZeroZeroRewardsCfg()
    terminations: BFMZeroTimeoutTerminationsCfg = BFMZeroTimeoutTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 6.0
        self.commands.motion.update_action_offset_with_ref = False
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body = "torso_link"
        self.commands.motion.body_names = list(PRIVILEGED_BODY_NAMES)
        self.commands.motion.load_human_motion = False
        self.commands.motion.sampler_type = _first_env(
            "PMT_BFM_ZERO_MOTION_SAMPLER", "WBT_BFM_ZERO_MOTION_SAMPLER"
        ) or self.pmt_sampler_type
        self.commands.motion.storage_device = _first_env(
            "PMT_BFM_ZERO_MOTION_STORAGE_DEVICE", "WBT_BFM_ZERO_MOTION_STORAGE_DEVICE"
        ) or self.commands.motion.storage_device
        self.commands.motion.use_fp16 = _env_bool(
            "PMT_BFM_ZERO_MOTION_USE_FP16",
            "WBT_BFM_ZERO_MOTION_USE_FP16",
            default=bool(self.commands.motion.use_fp16),
        )

        assert self.scene.terrain.terrain_type == "plane"


@configclass
class BFMZeroG1FlatMultiMotionV2EnvCfg_PLAY(BFMZeroG1FlatMultiMotionV2EnvCfg):
    """Small play/debug variant."""

    scene: MultiMotionFlatSceneCfg = MultiMotionFlatSceneCfg(num_envs=8, env_spacing=6.0)
