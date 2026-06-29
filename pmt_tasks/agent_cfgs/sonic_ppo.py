"""SONIC PPO agent cfg for the existing ``sonic_multimotion_flat`` task.

``sonic_mode=scratch`` preserves the original scratch policy exactly: raw
580/660 encoder inputs, no pretrained ONNX, and ``SonicActorCritic``.

``sonic_mode`` values ``finetune_all``, ``finetune_decoder``, and ``play`` use
the release G1 ONNX deploy contract through ``OfficialSonicActorCritic``.  The
ONNX directory can be overridden with ``PMT_SONIC_ONNX_DIR``.
"""
from __future__ import annotations

import os
from dataclasses import field
from pathlib import Path

from isaaclab.utils import configclass

from pmt_tasks.isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from pmt_tasks.path_defaults import repo_path

SONIC_MODES = ("scratch", "finetune_all", "finetune_decoder", "play")
SONIC_RELEASE_ONNX_ENV = "PMT_SONIC_ONNX_DIR"
DEFAULT_SONIC_RELEASE_ONNX_DIR = Path(repo_path("third_party", "sonic_release"))

SONIC_AUX_LOSS_COEF = {
    "g1_recon": 0.01,
    "g1_smpl_latent": 1.0,
    "g1_teleop_latent": 1.0,
    "teleop_smpl_latent": 1.0,
    "reencoded_smpl_g1_latent": 1.0,
}


def normalize_sonic_mode(mode: str | None = None) -> str:
    """Normalize and validate the task-level SONIC mode switch."""

    normalized = (mode or "scratch").strip().lower()
    if normalized not in SONIC_MODES:
        raise ValueError(
            f"Unsupported sonic_mode={mode!r}; expected one of {', '.join(SONIC_MODES)}."
        )
    return normalized


def _sonic_release_dir() -> Path:
    override = os.environ.get(SONIC_RELEASE_ONNX_ENV)
    return Path(override).expanduser() if override else DEFAULT_SONIC_RELEASE_ONNX_DIR


def _sonic_release_onnx(filename: str, *, required: bool = False) -> str | None:
    path = _sonic_release_dir() / filename
    if path.is_file():
        return str(path)
    if required:
        raise FileNotFoundError(
            f"SONIC release ONNX file not found: {path}. Set {SONIC_RELEASE_ONNX_ENV} "
            "to the directory containing model_encoder.onnx and model_decoder.onnx."
        )
    return None


def _sonic_release_onnx_pair(*, required: bool) -> tuple[str | None, str | None]:
    return (
        _sonic_release_onnx("model_encoder.onnx", required=required),
        _sonic_release_onnx("model_decoder.onnx", required=required),
    )


@configclass
class SonicActorCriticCfg:
    """SONIC actor-critic policy cfg (kwargs flow to SonicActorCritic.__init__).

    The on_policy runner pops ``class_name`` and forwards the remaining fields as
    kwargs to the SonicActorCritic constructor (which ignores any it does not use).
    """

    class_name: str = "SonicActorCritic"
    init_noise_std: float = 0.05
    min_action_std: float = 0.001
    max_action_std: float = 0.5
    action_encoder_source: str = "robot"  # robot | human | hybrid | auto | mode
    detach_action_token: bool = True
    decoder_proprio_layout: str = "none"
    control_decoder_input_order: str = "proprio_token"
    robot_encoder_layout: str = "raw"  # raw | g1_onnx_repack
    train_robot_encoder: bool = True
    train_human_encoder: bool = True
    train_hybrid_encoder: bool = False
    hybrid_motion_dim: int | None = None
    # GATE: scratch — no external ONNX; encoders/decoder random-init.
    pretrained_encoder_onnx_path: str | None = None
    pretrained_decoder_onnx_path: str | None = None
    load_pretrained_robot_encoder: bool = False
    load_pretrained_human_encoder: bool = False
    load_pretrained_hybrid_encoder: bool = False
    load_pretrained_control_decoder: bool = False
    strict_pretrained_shapes: bool = False
    actor_hidden_dims: list[int] = field(
        default_factory=lambda: [4096, 4096, 2048, 2048, 1024, 1024, 512, 512]
    )
    critic_hidden_dims: list[int] = field(
        default_factory=lambda: [4096, 4096, 2048, 2048, 1024, 1024, 512, 512]
    )
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 512])
    motion_decoder_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 512])
    latent_dim: int = 64
    num_fsq_levels: int = 32
    fsq_level_list: int | list[int] = 32
    max_num_tokens: int = 2
    reencode_smpl_g1_recon: bool = True
    aux_loss_coef: dict[str, float] = field(default_factory=lambda: dict(SONIC_AUX_LOSS_COEF))
    robot_motion_dim: int = 580
    human_motion_dim: int = 660  # 22 joints * 3 * 10 frames
    activation: str = "silu"


@configclass
class OfficialSonicG1DeployPolicyCfg:
    """ONNX-backed G1 deploy policy cfg for finetuning and direct play."""

    class_name: str = "OfficialSonicActorCritic"
    training_mode: str = "finetune_all"

    init_noise_std: float = 0.05
    min_action_std: float = 0.001
    max_action_std: float = 0.5

    action_encoder_source: str = "mode"
    encoder_mode_key: str = "encoder_mode_4"
    detach_action_token: bool = False
    decoder_proprio_layout: str = "interleaved_step_history"
    control_decoder_input_order: str = "token_proprio"
    robot_encoder_layout: str = "g1_onnx_repack"

    train_robot_encoder: bool = True
    train_human_encoder: bool = False
    train_hybrid_encoder: bool = False
    hybrid_motion_dim: int = 840
    robot_motion_dim: int = 640
    human_motion_dim: int = 267

    pretrained_encoder_onnx_path: str | None = None
    pretrained_decoder_onnx_path: str | None = None
    load_pretrained_robot_encoder: bool = True
    load_pretrained_human_encoder: bool = False
    load_pretrained_hybrid_encoder: bool = False
    load_pretrained_control_decoder: bool = True
    strict_pretrained_shapes: bool = True

    freeze_robot_encoder: bool = False
    freeze_human_encoder: bool = False
    freeze_hybrid_encoder: bool = False
    freeze_control_decoder: bool = False
    freeze_motion_decoder: bool = False
    freeze_critic: bool = False
    freeze_action_std: bool = False

    # NOTE: these MUST match OfficialSonicActorCritic.__init__ kwarg names
    # (aux_train_*_encoder); the base actor silently ignores unknown kwargs, so a
    # misnamed field would be dead. The aux flags gate which encoder branches the
    # SONIC reconstruction/latent aux losses train.
    aux_train_robot_encoder: bool = True
    aux_train_human_encoder: bool = False
    aux_train_hybrid_encoder: bool = False

    actor_hidden_dims: list[int] = field(
        default_factory=lambda: [2048, 2048, 1024, 1024, 512, 512]
    )
    critic_hidden_dims: list[int] = field(
        default_factory=lambda: [2048, 2048, 1024, 1024, 512, 512]
    )
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 512])
    motion_decoder_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 512])

    latent_dim: int = 64
    num_fsq_levels: int = 32
    fsq_level_list: int | list[int] = 32
    max_num_tokens: int = 2
    reencode_smpl_g1_recon: bool = True
    aux_loss_coef: dict[str, float] = field(default_factory=lambda: dict(SONIC_AUX_LOSS_COEF))
    activation: str = "silu"


@configclass
class G1SonicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """SONIC PPO runner for the MultiMotion V2 flat env (scratch encoders)."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_sonic_flat"
    empirical_normalization = False
    resume = False

    # SonicActorCritic gets robot_encoder/human_encoder from the full obs dict;
    # obs_groups only carries policy/critic.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = SonicActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        aux_loss_scale=1.0,
        aux_loss_coef=dict(SONIC_AUX_LOSS_COEF),
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1SonicDeployPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """SONIC PPO runner using the G1 release/deploy ONNX observation contract."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_sonic_flat"
    empirical_normalization = False
    resume = False

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "robot_encoder": ["robot_encoder"],
        "encoder_mode_4": ["encoder_mode_4"],
    }

    policy = OfficialSonicG1DeployPolicyCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        aux_loss_scale=1.0,
        aux_loss_coef=dict(SONIC_AUX_LOSS_COEF),
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.1,
    )


def make_official_sonic_g1_deploy_policy_cfg(mode: str) -> OfficialSonicG1DeployPolicyCfg:
    """Build a release-ONNX policy cfg for a non-scratch SONIC mode."""

    sonic_mode = normalize_sonic_mode(mode)
    if sonic_mode == "scratch":
        raise ValueError("scratch mode uses SonicActorCriticCfg, not the deploy policy cfg.")

    encoder_onnx, decoder_onnx = _sonic_release_onnx_pair(required=True)
    policy = OfficialSonicG1DeployPolicyCfg()
    policy.training_mode = sonic_mode
    policy.pretrained_encoder_onnx_path = encoder_onnx
    policy.pretrained_decoder_onnx_path = decoder_onnx

    # Inactive branches (human/hybrid) are kept FROZEN to match the reference
    # OfficialSonic presets (train_human/hybrid_encoder=False there); only the active
    # robot encoder + control/motion decoders + critic are trained per mode.
    if sonic_mode == "finetune_all":
        policy.detach_action_token = False
        policy.freeze_robot_encoder = False
        policy.freeze_human_encoder = True
        policy.freeze_hybrid_encoder = True
        policy.freeze_control_decoder = False
        policy.freeze_motion_decoder = False
        policy.freeze_critic = False
        policy.freeze_action_std = False
        policy.aux_train_robot_encoder = True
    elif sonic_mode == "finetune_decoder":
        policy.detach_action_token = True
        policy.freeze_robot_encoder = True
        policy.freeze_human_encoder = True
        policy.freeze_hybrid_encoder = True
        policy.freeze_control_decoder = False
        policy.freeze_motion_decoder = False
        policy.freeze_critic = False
        policy.freeze_action_std = False
        policy.aux_train_robot_encoder = False
    elif sonic_mode == "play":
        policy.detach_action_token = True
        policy.freeze_robot_encoder = True
        policy.freeze_human_encoder = True
        policy.freeze_hybrid_encoder = True
        policy.freeze_control_decoder = True
        policy.freeze_motion_decoder = True
        policy.freeze_critic = True
        policy.freeze_action_std = True
        policy.aux_train_robot_encoder = False

    return policy


def make_g1_sonic_runner_cfg(mode: str | None = None) -> RslRlOnPolicyRunnerCfg:
    """Return the scratch or release/deploy SONIC runner cfg for ``sonic_mode``."""

    sonic_mode = normalize_sonic_mode(mode)
    if sonic_mode == "scratch":
        return G1SonicPPORunnerCfg()

    runner_cfg = G1SonicDeployPPORunnerCfg()
    runner_cfg.policy = make_official_sonic_g1_deploy_policy_cfg(sonic_mode)
    return runner_cfg
