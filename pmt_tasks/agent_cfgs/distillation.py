"""Stepping-stone distillation agent cfg (ported from rsl_rl_distill_cfg
G1SteppingStoneDistillRunnerCfg).

PMT plan §6 Phase 2.2 / §10 PART D. ``build_agent_cfg`` returns a fresh instance per
call (§10/D). The policy is ``StudentTeacher`` (compat network "student_teacher") on
the ``DistillationRunner`` (NOT the on_policy runner); the algorithm is
``Distillation``.

TEACHER-CKPT FALLBACK (gate, not result): ``teacher_ckpt_path=None``. The source uses
``STEPPING_STONE_TEACHER_CKPT`` which does NOT exist locally and would be shape-
incompatible with the [1024,512,256,128] student/teacher arch anyway. With None,
``StudentTeacher.__init__`` builds the teacher as a RANDOM-init MLP (it only loads a
ckpt when a path is given). The distillation loss is therefore MEANINGLESS, but this
validates the RUNNER PATH end-to-end: DistillationRunner constructs, dispatches via
registry.get_runner, runs teacher/student rollout, and takes gradient steps. A real
trained teacher is a Phase-2-later TODO.

Values are the verified ground-truth from G1SteppingStoneDistillRunnerCfg
(rsl_rl_distill_cfg.py:241).
"""
from __future__ import annotations

import os

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)
from isaaclab.utils import configclass


@configclass
class G1SteppingStoneDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Stepping-stone distillation with DAgger-style mixed rollout."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_stepping_stone_distill"

    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
    }

    debug_use_teacher_actions_for_env_step = False
    debug_rollout_action_stats = True

    # DAgger-style mixed rollout: teacher_mix 1.0 -> 0.0 over 3000 iters.
    student_mean_for_env_step = True
    teacher_mix_start = 1.0
    teacher_mix_end = 0.0
    teacher_mix_anneal_iters = 3000

    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        init_noise_std=0.01,
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        # FALLBACK: None by default. NOTE the DistillationRunner.learn() guards on
        # ``policy.loaded_teacher`` and RAISES if no teacher ckpt was loaded — so a
        # truly-None teacher CANNOT run (the doc-comment's "random-init proves the
        # path" was aspirational; the runner guard blocks it). To smoke the runner
        # path on the cluster without a real trained teacher, point
        # $PMT_STEPPING_STONE_TEACHER_CKPT at a shape-matching stub ckpt (actor.* keys).
        # Real STEPPING_STONE_TEACHER_CKPT is a Phase-2-later TODO.
        teacher_ckpt_path=os.environ.get("PMT_STEPPING_STONE_TEACHER_CKPT") or None,
        motion_target_obs_key="motion_target",
        anchor_target_obs_key="anchor_target",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=4,
        learning_rate=1e-4,
        gradient_length=1,
        loss_type="mse",
        motion_loss_coef=0.05,
        anchor_loss_coef=0.0,
    )


# =====================================================================================
# vision-transformer student / blind-transformer teacher latent-anchor distillation
# =====================================================================================
#
# Faithful port of the original G1SteppingStoneVisionLatentAnchorDistillRunnerCfg
# (rsl_rl_distill_cfg.py:518) and its parent G1SteppingStoneVisionDistillRunnerCfg
# (:457). The policy is the VisionStudentTeacher wrapper: a vision-augmented transformer
# student (VisionTransformerActorCritic) distilled from the BLIND TransformerActorCritic
# teacher whose trained ckpt is loaded via ``teacher_ckpt_path``. The builder resolves the
# named ckpt (${checkpoints.ss_teacher} -> .../model_9999.pt) and injects it into both
# ``policy.teacher_ckpt_path`` (frozen distillation target) and the student warm-start
# ``base_policy_ckpt`` (partial transfer of the overlapping blind transformer backbone).

# Verified blind-transformer student/teacher hyperparams (rsl_rl_distill_cfg.py:81/127).
VISION_STUDENT_CFG = {
    "actor_obs_normalization": True,
    "history_obs_normalization": True,
    "command_obs_normalization": True,
    "critic_obs_normalization": True,
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
    "n_embd": 128,
    "n_heads": 4,
    "history_len": 10,
    "cmd_len": 21,
    "mlp_ratio": 4,
    "state_dependent_std": False,
    "log_std_bounds": (-5.0, 2.0),
    "min_std": 1.0e-6,
    "validate_args": True,
    "use_vel_estimator": True,
    "vel_estimator_detach": False,
    "vel_estimator_hidden_dims": (256, 128),
    "vel_estimator_output_dim": 3,
    "vel_gt_normalization": True,
    "use_anchor_estimator": True,
    "anchor_estimator_detach": True,
    "anchor_estimator_hidden_dims": (256, 128),
    "anchor_estimator_output_dim": 3,
    "anchor_gt_normalization": True,
    "anchor_estimator_latent_inputs": ("h_last", "u_t"),
    "use_foot_traj_head": True,
    "foot_traj_hidden_dims": (256, 128),
    "foot_traj_use_vision": True,
    "map_height": 17,
    "map_width": 11,
    "map_resolution": 0.1,
    "dim_map_embed": 64,
    "num_attn_heads": 4,
    "z_clip": 3.0,
    "normalize_height": True,
    "critic_use_vision": True,
    # Warm-start overlapping blind transformer tensors from the teacher checkpoint; the builder
    # injects ``base_policy_ckpt`` (resolved ${checkpoints.ss_teacher}).
    "training_stage": "finetune_all",
    "base_policy_ckpt": None,
    "allow_partial_base_policy_transfer": True,
}

VISION_TEACHER_CFG = {
    "actor_obs_normalization": True,
    "history_obs_normalization": True,
    "command_obs_normalization": True,
    "critic_obs_normalization": True,
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
    "n_embd": 128,
    "n_heads": 4,
    "history_len": 10,
    "cmd_len": 21,
    "mlp_ratio": 4,
    "state_dependent_std": False,
    "log_std_bounds": (-5.0, 2.0),
    "min_std": 1.0e-6,
    "validate_args": True,
    # Must match the full blind transformer teacher checkpoint used by teacher_ckpt_path.
    "use_vel_estimator": True,
    "vel_estimator_detach": True,
    "vel_estimator_hidden_dims": (256, 128),
    "vel_estimator_output_dim": 3,
    "vel_gt_normalization": True,
    "use_anchor_estimator": True,
    "anchor_estimator_detach": True,
    "anchor_estimator_hidden_dims": (256, 128),
    "anchor_estimator_output_dim": 3,
    "anchor_gt_normalization": True,
    "anchor_estimator_latent_inputs": ("h_last", "u_t"),
}


@configclass
class VisionStudentTeacherCfg:
    """vision-transformer student distilled from a blind transformer teacher (port of rsl_rl_distill_cfg:310)."""

    class_name: str = "VisionStudentTeacher"
    align_teacher_to_student_reference: bool = True
    foot_traj_target_obs_key: str | None = "foot_traj_target"
    student_cfg: dict = None
    teacher_cfg: dict = None
    teacher_class_name: str = "TransformerActorCritic"
    teacher_ckpt_path: str | None = None
    teacher_load_strict: bool = True

    def __post_init__(self):
        # Fresh dict copies per instance (avoid shared-mutable class attrs).
        if self.student_cfg is None:
            self.student_cfg = dict(VISION_STUDENT_CFG)
        if self.teacher_cfg is None:
            self.teacher_cfg = dict(VISION_TEACHER_CFG)


@configclass
class G1SteppingStoneVisionLatentAnchorDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """vision distillation with latent-only student anchor estimator.

    Faithful port of G1SteppingStoneVisionLatentAnchorDistillRunnerCfg
    (rsl_rl_distill_cfg.py:518). obs_groups route the student transformer groups to the
    VisionTransformerActorCritic student and the teacher transformer groups to the frozen
    blind TransformerActorCritic teacher (which loads ``policy.teacher_ckpt_path``).
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_stepping_stone_vision_latent_anchor_distill"
    resume = False

    obs_groups = {
        "policy": ["policy", "proprio"],
        "policy_history": ["proprio_history"],
        "command_window": ["command_window", "motion_anchor_delta_window"],
        "critic": ["critic"],
        "vel_gt": ["vel_gt_xyz"],
        "anchor_gt": ["anchor_gt"],
        "teacher": ["teacher", "proprio"],
        "teacher_policy_history": ["proprio_history"],
        "teacher_command_window": ["teacher_command_window", "teacher_motion_anchor_delta_window"],
        "teacher_anchor_estimator": ["teacher_anchor_body_pose"],
    }

    debug_use_teacher_actions_for_env_step = False
    student_mean_for_env_step = True
    teacher_mix_start = 1.0
    teacher_mix_end = 0.0
    teacher_mix_anneal_iters = 100

    debug_rollout_action_stats = False
    debug_rollout_action_steps = 2
    debug_rollout_action_print_freq = 1

    policy = VisionStudentTeacherCfg()
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=4,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
        vel_loss_coef=0.0,
        vel_loss_type="huber",
        vel_loss_delta=1.0,
        anchor_est_loss_coef=0.0,
        anchor_est_loss_type="huber",
        anchor_est_loss_delta=1.0,
        foot_traj_loss_coef=0.0,
        foot_traj_loss_type="huber",
        foot_traj_loss_delta=0.05,
    )
