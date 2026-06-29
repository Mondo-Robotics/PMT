"""Back-flip env: stepping-stone stack on the big_map mesh, per-task dt (§3a).

PMT plan §10 PART A + §6 Phase-1 slice. The back_flip_merged clips are TERRAIN-ANCHORED
(their body_pos_w is placed in big_map world coords), so the env MUST run on the big_map
STL mesh — exactly like the walk_dance / cartwheel teachers. Backflip therefore subclasses
``PMTSteppingStoneEnvCfg`` directly and reuses its mesh-terrain scene + transformer
obs stack; it only diverges in three task-specific ways:

  1. per-task control rate: ``decimation=10`` / ``sim.dt=0.002`` (§3a dt-from-motion, driven
     from configs/motion/backflip.yaml — NOT hard-coded here).
  2. rewards: SteppingStone reward set + a knee negative-power safety term
     (BackFlipMergedRewardsCfg, distill_stepping_stone_env_cfg.py:988).
  3. termination: the default ee_body_pos uses the FULL-XYZ body-position error
     (bad_motion_body_pos), which fires the instant the flip swings the end-effectors
     horizontally -> ~90% of episodes terminated immediately and the policy never learned
     the flip (reward plateaued ~8 w/ joint_pos error ~1.5). The original backflip teacher
     overrides it to the Z-ONLY check (bad_motion_body_pos_z_only) so the big horizontal
     swing is tolerated while vertical drift still terminates a fall.

The parent ``__post_init__`` already applies the mesh (``pmt_mesh_path``), dec/dt, residual
action, command wiring, DR-event drops, and motion discovery. This subclass just swaps the
reward cfg and re-points the ee termination after delegating to it.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.env_cfgs.pmt.stepping_stone import (
    PMTSteppingStoneEnvCfg,
    SteppingStoneRewardsCfg,
)


@configclass
class BackFlipRewardsCfg(SteppingStoneRewardsCfg):
    """SteppingStone tracking rewards + knee negative-power safety.

    Mirror of BackFlipMergedRewardsCfg (distill_stepping_stone_env_cfg.py:988):
    adds ``knee_negative_power`` = negative_joint_power_l2 (weight -10, knee joints,
    deadband 150, power_norm 500) on top of the SteppingStone reward set.
    """

    knee_negative_power = RewTerm(
        func=mdp.negative_joint_power_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
            "deadband": 150.0,
            "power_norm": 500.0,
        },
    )


@configclass
class PMTBackFlipEnvCfg(PMTSteppingStoneEnvCfg):
    """Back-flip env on the big_map mesh terrain (terrain-anchored clips).

    Inherits the mesh-terrain scene, transformer obs, command wiring, residual
    action, DR-event drops, and §3a dt-from-motion from PMTSteppingStoneEnvCfg.
    Builder injects ``pmt_mesh_path`` (big_map STL), ``pmt_motion_paths`` (back_flip_merged),
    and ``pmt_decimation`` / ``pmt_sim_dt`` (10 / 0.002).
    """

    # §3a defaults so the cfg constructs standalone; the builder OVERRIDES from
    # configs/motion/backflip.yaml.
    pmt_decimation: int = 10
    pmt_sim_dt: float = 0.002

    rewards: BackFlipRewardsCfg = BackFlipRewardsCfg()

    def __post_init__(self):
        # Parent applies mesh + dec/dt + residual action + command wiring + DR drops +
        # motion discovery (all the stepping-stone teacher wiring backflip shares).
        super().__post_init__()

        # CRITICAL backflip fix: re-point the ee_body_pos termination to the Z-ONLY
        # check so the flip's large horizontal end-effector swing is tolerated while
        # vertical drift (an actual fall) still terminates. The full-XYZ default fires
        # on ~90% of backflip episodes and the policy never learns the flip.
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )
