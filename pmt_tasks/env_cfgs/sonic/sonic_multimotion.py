"""SONIC MultiMotion V2 flat env cfg.

The existing ``sonic_multimotion_flat`` task supports four modes through
``pmt_sonic_mode``:

* ``scratch`` keeps the original PMT observation contract and paired robot/SMPL
  motion loading for training the raw 580/660-dim encoders from scratch.
* ``finetune_all``, ``finetune_decoder``, and ``play`` switch the same task to
  the release/deploy G1 observation contract used by the pretrained ONNX.
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.env_cfgs.multi_motion_flat import PMTMultiMotionFlatEnvCfg
from pmt_tasks.tracking_env_cfg import SonicObservationsCfg


SONIC_MODES = ("scratch", "finetune_all", "finetune_decoder", "play")


def _normalize_sonic_mode(mode: str | None) -> str:
    normalized = (mode or "scratch").strip().lower()
    if normalized not in SONIC_MODES:
        raise ValueError(
            f"Unsupported sonic_mode={mode!r}; expected one of {', '.join(SONIC_MODES)}."
        )
    return normalized


@configclass
class SonicMultiMotionV2ObservationsCfg(SonicObservationsCfg):
    """SONIC obs: deploy-aligned policy/critic + robot/human encoder groups.

    Mirror of the old SonicMultiMotionV2ObservationsCfg (multi_motion_env_cfg.py:682):
    inherits the SONIC policy/critic groups and adds robot_encoder / human_encoder.
    """

    @configclass
    class RobotEncoderCfg(ObsGroup):
        """Robot motion encoder input (580D = 10 frames x (29 pos + 29 vel))."""

        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "motion"})

    @configclass
    class HumanEncoderCfg(ObsGroup):
        """Human motion encoder input (660D = 10 frames x 22 joints x 3)."""

        human_motion = ObsTerm(func=mdp.sonic_human_motion, params={"command_name": "motion"})

    robot_encoder: RobotEncoderCfg = RobotEncoderCfg()
    human_encoder: HumanEncoderCfg = HumanEncoderCfg()


@configclass
class SonicDeployObservationsCfg(SonicObservationsCfg):
    """Release/deploy G1 observation contract for ONNX-backed SONIC modes."""

    @configclass
    class PolicyCfg(ObsGroup):
        decoder_step_obs = ObsTerm(func=mdp.sonic_decoder_step_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        )
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RobotEncoderCfg(ObsGroup):
        g1_encoder_obs = ObsTerm(
            func=mdp.sonic_g1_encoder_branch,
            params={"command_name": "motion"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EncoderModeCfg(ObsGroup):
        encoder_mode = ObsTerm(func=mdp.sonic_encoder_mode_4, params={"mode_id": 0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    robot_encoder: RobotEncoderCfg = RobotEncoderCfg()
    encoder_mode_4: EncoderModeCfg = EncoderModeCfg()


@configclass
class PMTSonicMultiMotionFlatEnvCfg(PMTMultiMotionFlatEnvCfg):
    """SONIC env: scratch encoder obs or release/deploy ONNX obs by mode."""

    pmt_sonic_mode: str = "scratch"
    observations: SonicMultiMotionV2ObservationsCfg = SonicMultiMotionV2ObservationsCfg()

    def __post_init__(self):
        sonic_mode = _normalize_sonic_mode(getattr(self, "pmt_sonic_mode", "scratch"))
        self.pmt_sonic_mode = sonic_mode
        self.observations = (
            SonicMultiMotionV2ObservationsCfg()
            if sonic_mode == "scratch"
            else SonicDeployObservationsCfg()
        )

        super().__post_init__()

        self.commands.motion.load_human_motion = sonic_mode == "scratch"

        policy_obs = getattr(self.observations, "policy", None)
        if policy_obs is not None and hasattr(policy_obs, "motion_anchor_pos_b"):
            policy_obs.motion_anchor_pos_b = None
        if policy_obs is not None and hasattr(policy_obs, "base_lin_vel"):
            policy_obs.base_lin_vel = None
