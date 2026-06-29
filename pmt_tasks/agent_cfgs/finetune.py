"""PPO-finetune agent cfg for the distilled vision-transformer student (PMT plan P3).

Third stage of the teacher -> distill -> finetune pipeline. The policy is a standalone
``VisionTransformerActorCritic`` (the SAME network the distilled student wraps) trained
with on-policy PPO and initialized from the distilled-student checkpoint via
``base_policy_ckpt``.

The network's ``_smart_load_checkpoint`` already strips the ``student.`` prefix from a
distillation checkpoint's ``model_state_dict`` and loads the overlapping tensors (the
distilled student carries the vision encoder, so a FULL transfer happens). The builder
resolves the distilled-student ckpt (``${checkpoints.distilled_student}``) and injects it
into ``policy.base_policy_ckpt``.

obs_groups mirror the on-policy transformer PPO runner (G1SteppingStonePPORunnerCfg) +
the height-scan ``vision`` set. NO teacher groups (single-policy PPO). Network hyperparams
are the distilled student's (distillation.VISION_STUDENT_CFG) so the checkpoint loads.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from pmt_tasks.agent_cfgs.distillation import VISION_STUDENT_CFG


@configclass
class RslRlPpoVisionTransformerActorCriticCfg(RslRlPpoActorCriticCfg):
    """Policy configuration for ``VisionTransformerActorCritic`` (PPO finetune).

    Declares every network field the distilled student uses so the runner can pass them
    straight into the network constructor (on_policy_runner builds the policy with
    ``actor_critic_class(obs, obs_groups, num_actions, **policy_cfg)``). ``base_policy_ckpt``
    points at the distilled-student checkpoint; the network strips its ``student.`` prefix.
    """

    class_name: str = "VisionTransformerActorCritic"

    # transformer backbone (matches VISION_STUDENT_CFG)
    n_embd: int = 128
    n_heads: int = 4
    history_len: int = 10
    cmd_len: int = 21
    mlp_ratio: int = 4

    state_dependent_std: bool = False
    log_std_bounds: tuple[float, float] = (-5.0, 2.0)
    min_std: float = 1e-6
    validate_args: bool = True

    history_obs_normalization: bool = True
    command_obs_normalization: bool = True

    # velocity / anchor / foot-traj heads (latent-anchor student)
    use_vel_estimator: bool = True
    vel_estimator_detach: bool = False
    vel_estimator_hidden_dims: tuple[int, ...] = (256, 128)
    vel_estimator_output_dim: int = 3
    vel_gt_normalization: bool = True
    use_anchor_estimator: bool = True
    anchor_estimator_detach: bool = True
    anchor_estimator_hidden_dims: tuple[int, ...] = (256, 128)
    anchor_estimator_output_dim: int = 3
    anchor_gt_normalization: bool = True
    anchor_estimator_latent_inputs: tuple[str, ...] = ("h_last", "u_t")

    use_foot_traj_head: bool = True
    # Single-command finetune has no teacher/student foot_traj DELTA target group, so
    # supervision is off (algorithm.foot_traj_loss_coef=0). We still BUILD the head with
    # an explicit output dim so the distilled student's foot_traj_head tensors transfer
    # cleanly (window_size=5 * 2 foot bodies * 3 xyz = 30; see FootTrajTargetCfg).
    foot_traj_output_dim: int = 30
    foot_traj_target_obs_key: str | None = None
    foot_traj_hidden_dims: tuple[int, ...] = (256, 128)
    foot_traj_use_vision: bool = True

    # vision (height-map) encoder
    map_height: int = 17
    map_width: int = 11
    map_resolution: float = 0.1
    dim_map_embed: int = 64
    num_attn_heads: int = 4
    z_clip: float = 3.0
    normalize_height: bool = True
    critic_use_vision: bool = True

    # latent-anchor / residual toggles (network ablation knobs)
    use_action_residual: bool = True
    use_identity_gates: bool = True

    # warm-start from the distilled student checkpoint. The builder resolves the
    # concrete model_*.pt path (${checkpoints.distilled_student}) and overwrites this.
    training_stage: str = "finetune_all"
    base_policy_ckpt: str | None = None
    allow_partial_base_policy_transfer: bool = True


@configclass
class G1SteppingStoneVisionTeacherFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """On-policy PPO finetune runner for the distilled vision-transformer student.

    obs_groups = single-command transformer PPO sets (G1SteppingStonePPORunnerCfg) + the
    height-scan ``vision`` set. All groups read the single ``motion`` command (no teacher).
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_ppoft_vt_ss_latent_anchor"

    obs_groups = {
        "policy": ["policy", "proprio"],
        "policy_history": ["proprio_history"],
        "command_window": ["command_window", "motion_anchor_delta_window"],
        "critic": ["critic"],
        "vel_gt": ["vel_gt_xyz"],
        "anchor_gt": ["anchor_gt"],
        "vision": ["vision"],
    }

    policy = RslRlPpoVisionTransformerActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        history_obs_normalization=True,
        command_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=list(VISION_STUDENT_CFG["actor_hidden_dims"]),
        critic_hidden_dims=list(VISION_STUDENT_CFG["critic_hidden_dims"]),
        activation=VISION_STUDENT_CFG["activation"],
        n_embd=VISION_STUDENT_CFG["n_embd"],
        n_heads=VISION_STUDENT_CFG["n_heads"],
        history_len=VISION_STUDENT_CFG["history_len"],
        cmd_len=VISION_STUDENT_CFG["cmd_len"],
        mlp_ratio=VISION_STUDENT_CFG["mlp_ratio"],
        state_dependent_std=VISION_STUDENT_CFG["state_dependent_std"],
        log_std_bounds=VISION_STUDENT_CFG["log_std_bounds"],
        min_std=VISION_STUDENT_CFG["min_std"],
        validate_args=VISION_STUDENT_CFG["validate_args"],
        use_vel_estimator=VISION_STUDENT_CFG["use_vel_estimator"],
        vel_estimator_detach=VISION_STUDENT_CFG["vel_estimator_detach"],
        vel_estimator_hidden_dims=VISION_STUDENT_CFG["vel_estimator_hidden_dims"],
        vel_estimator_output_dim=VISION_STUDENT_CFG["vel_estimator_output_dim"],
        vel_gt_normalization=VISION_STUDENT_CFG["vel_gt_normalization"],
        use_anchor_estimator=VISION_STUDENT_CFG["use_anchor_estimator"],
        anchor_estimator_detach=VISION_STUDENT_CFG["anchor_estimator_detach"],
        anchor_estimator_hidden_dims=VISION_STUDENT_CFG["anchor_estimator_hidden_dims"],
        anchor_estimator_output_dim=VISION_STUDENT_CFG["anchor_estimator_output_dim"],
        anchor_gt_normalization=VISION_STUDENT_CFG["anchor_gt_normalization"],
        anchor_estimator_latent_inputs=VISION_STUDENT_CFG["anchor_estimator_latent_inputs"],
        use_foot_traj_head=VISION_STUDENT_CFG["use_foot_traj_head"],
        foot_traj_output_dim=30,
        foot_traj_target_obs_key=None,
        foot_traj_hidden_dims=VISION_STUDENT_CFG["foot_traj_hidden_dims"],
        foot_traj_use_vision=VISION_STUDENT_CFG["foot_traj_use_vision"],
        map_height=VISION_STUDENT_CFG["map_height"],
        map_width=VISION_STUDENT_CFG["map_width"],
        map_resolution=VISION_STUDENT_CFG["map_resolution"],
        dim_map_embed=VISION_STUDENT_CFG["dim_map_embed"],
        num_attn_heads=VISION_STUDENT_CFG["num_attn_heads"],
        z_clip=VISION_STUDENT_CFG["z_clip"],
        normalize_height=VISION_STUDENT_CFG["normalize_height"],
        critic_use_vision=VISION_STUDENT_CFG["critic_use_vision"],
        training_stage="finetune_all",
        base_policy_ckpt=None,  # builder injects the resolved distilled-student ckpt
        allow_partial_base_policy_transfer=True,
    )

    # PPO finetune hyperparams: lower LR than scratch (5e-4 -> 1e-4), modest entropy.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        vel_loss_coef=1.0,
        vel_loss_type="huber",
        vel_loss_delta=1.0,
        anchor_est_loss_coef=1.0,
        anchor_est_loss_type="huber",
        anchor_est_loss_delta=1.0,
    )
