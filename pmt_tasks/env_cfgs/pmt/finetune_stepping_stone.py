"""PPO-finetune env cfg for the distilled vision-transformer student (PMT plan P3).

This is the THIRD stage of the teacher -> distill -> finetune pipeline. Unlike the
distillation env (paired ``motion``/``student_motion`` commands routed to a frozen
teacher + student), the finetune env runs a SINGLE ``motion`` command and a single
``VisionTransformerActorCritic`` policy trained with PPO (initialized from the
distilled student checkpoint via ``base_policy_ckpt``).

Structure = single-command stepping-stone transformer env (PMTSteppingStoneEnvCfg /
TransformerSteppingStoneObservationsCfg) PLUS the height-scan ``vision`` obs group the
VisionTransformerActorCritic requires. The builder injects the big_map terrain mesh +
the single optimized walk_dance clip dir + decimation/sim_dt.

The agent cfg's obs_groups (G1SteppingStoneVisionTeacherFinetuneRunnerCfg) route these
single-command groups to the vision student; there are NO teacher groups (single-policy
PPO).
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.env_cfgs.pmt.stepping_stone import (
    PMTSteppingStoneEnvCfg,
    TransformerSteppingStoneObservationsCfg,
)


@configclass
class FinetuneSteppingStoneVisionObservationsCfg(TransformerSteppingStoneObservationsCfg):
    """Single-command transformer obs groups + height-scan ``vision`` group.

    Inherits the single ``motion``-command transformer groups from
    TransformerSteppingStoneObservationsCfg and adds the ``vision`` height-scan group
    that VisionTransformerActorCritic requires.

    CRITICAL: the distilled student's actor was trained on the REDUCED student PolicyCfg
    (6 terms: command, robot_motion, base_ang_vel, joint_pos, joint_vel, actions — it
    EXCLUDES the privileged anchor/body-pose/base_lin_vel terms that the full
    TransformerSteppingStone PolicyCfg carries). That gives actor_obs_dim=2153. We MUST
    reproduce that exact policy group here (on the single ``motion`` command) or the
    base_policy_ckpt load fails with an actor_obs_dim mismatch (2153 vs the full 3533).
    Mirror of distill_stepping_stone.TransformerDistillSteppingStoneObservationsCfg.PolicyCfg.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}, history_length=10)
        robot_motion = ObsTerm(func=mdp.sonic_robot_motion, params={"command_name": "motion"}, history_length=1)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        actions = ObsTerm(func=mdp.last_action, history_length=10)

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

    # override the inherited full PolicyCfg with the reduced student policy group.
    policy: PolicyCfg = PolicyCfg()
    vision: VisionCfg = VisionCfg()


@configclass
class PMTSteppingStoneVisionTeacherFinetuneEnvCfg(PMTSteppingStoneEnvCfg):
    """PPO-finetune env for the distilled vision-transformer student.

    Single-command stepping-stone transformer env + height-scan ``vision`` group. The
    builder sets pmt_mesh_path (big_map), pmt_motion_paths (single optimized walk_dance
    clip dir), pmt_decimation/pmt_sim_dt before __post_init__.

    Matches the distilled student's residual-action contract: the student acts around
    the raw reference trajectory (action scale 1.0, update_action_offset_with_ref=True),
    same as PMTSteppingStoneEnvCfg.__post_init__ already sets.
    """

    observations: FinetuneSteppingStoneVisionObservationsCfg = (
        FinetuneSteppingStoneVisionObservationsCfg()
    )
