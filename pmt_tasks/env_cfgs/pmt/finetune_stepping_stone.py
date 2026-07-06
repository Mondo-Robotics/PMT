"""PPO-finetune env cfg for the distilled vision-transformer student (PMT plan P3).

This is the THIRD stage of the teacher -> distill -> finetune pipeline. The finetune
env uses PAIRED motion commands (same as distillation): reward is computed from the
optimized trajectory (``motion`` command) while the policy observes the raw trajectory
(``student_motion`` command). This matches the whole_body_tracking finetune logic
(RGMTG1SteppingStoneLatentAnchorFinetuneEnvCfg).

Inherits the full distill env (which already has all necessary obs groups: policy,
proprio, proprio_history, command_window, motion_anchor_delta_window, critic,
vel_gt_xyz, anchor_gt, vision, foot_traj_target) and removes only the teacher-specific
groups that require the DistillationRunner (teacher, teacher_anchor_body_pose,
teacher_command_window, teacher_motion_anchor_delta_window).

The builder injects pmt_mesh_path, pmt_optimized_motion_paths, pmt_raw_motion_paths,
pmt_decimation, pmt_sim_dt before __post_init__.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.env_cfgs.pmt.distill_stepping_stone import (
    PMTSteppingStoneVisionLatentAnchorDistillEnvCfg,
)


@configclass
class PMTSteppingStoneVisionTeacherFinetuneEnvCfg(PMTSteppingStoneVisionLatentAnchorDistillEnvCfg):
    """PPO-finetune env: paired motion (reward=optimized, obs=raw), no teacher groups.

    Inherits the full distill env (paired commands, stepping-stone terrain, all student
    obs groups including vision height-scan, latent-anchor action contract), then removes
    teacher-only obs groups so PPO runner works (no teacher supervision).
    """

    def __post_init__(self):
        super().__post_init__()

        # Remove teacher-only observation groups (not needed for single-policy PPO)
        self.observations.teacher = None
        self.observations.teacher_anchor_body_pose = None
        self.observations.teacher_command_window = None
        self.observations.teacher_motion_anchor_delta_window = None

        # Also remove anchor_body_pose (latent-anchor variant: supervised in latent space)
        self.observations.anchor_body_pose = None
