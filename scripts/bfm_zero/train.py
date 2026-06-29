"""Launch BFM-Zero (FB-CPR-Aux) training on an IsaacLab G1 motion task.

Canonical local recipe (single GPU):
    conda activate <env>
    export OMNI_KIT_ACCEPT_EULA=YES
    # The BFM-Zero (FB-CPR-Aux) networks/agent are vendored inside PMT
    # (motion_tracking_rl/bfm_zero/_vendor), so no external BFM-Zero repo is required.
    # (optional) point the flat MultiMotionV2 store at local clip dirs:
    # export PMT_BFM_ZERO_FLAT_MOTION_PATHS=<motion-dir>
    cd <repo>
    python scripts/bfm_zero/train.py --task BFM-Zero-Flat-MultiMotionV2-G1-v0 \
        --agent_preset smoke --num_envs 8 --headless --total_env_steps 4096 --num_seed_steps 64 \
        --update_agent_every 128 --num_agent_updates 1 --batch_size 64 \
        --work_dir logs/bfm_zero_smoke

Unlike the transformer PPO launcher, this does NOT require a wandb motion registry: the motion command
loads clips directly from disk.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train BFM-Zero (FB-CPR-Aux) on an IsaacLab task.")
parser.add_argument("--task", type=str, default="BFM-Zero-Flat-MultiMotionV2-G1-v0")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel envs.")
parser.add_argument("--seed", type=int, default=4728)
# NOTE: ``--device`` is provided by AppLauncher.add_app_launcher_args; do not redefine it here.
parser.add_argument("--total_env_steps", type=int, default=384_000_000)
parser.add_argument("--num_seed_steps", type=int, default=10_240)
parser.add_argument("--update_agent_every", type=int, default=1024)
parser.add_argument("--num_agent_updates", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--buffer_size", type=int, default=5_120_000)
parser.add_argument(
    "--buffer_device",
    type=str,
    default="cpu",
    help="Device for the TRAIN replay buffer (cpu keeps GPU memory for the agent).",
)
parser.add_argument(
    "--expert_buffer_device",
    type=str,
    default=None,
    help="Device for the EXPERT buffer (defaults to the agent device; it is sampled by "
    "_sample_tracking_z without a device move, so it must live on the agent device).",
)
parser.add_argument(
    "--agent_preset",
    type=str,
    default="full",
    choices=["full", "smoke"],
    help="'full' = paper-size nets (2048x6); 'smoke' = small nets (512x2) for pipeline validation. "
    "num_parallel stays 2 in both (ensemble math requires it).",
)
parser.add_argument("--expert_seq_length", type=int, default=8)
parser.add_argument("--motion_resample_frequency", type=int, default=0)
parser.add_argument("--log_every", type=int, default=2048, help="Log metrics every N env transitions.")
parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on the agent.")
parser.add_argument(
    "--checkpoint_every",
    type=int,
    default=0,
    help="Checkpoint every N env transitions (0 = only final save).",
)
parser.add_argument(
    "--eval_every",
    type=int,
    default=0,
    help="Run zero-shot tracking eval every N env transitions (0 = disabled).",
)
parser.add_argument("--eval_horizon", type=int, default=250, help="Tracking-eval rollout length (frames).")
parser.add_argument("--work_dir", type=str, default="logs/bfm_zero/g1_flat_multimotion")
parser.add_argument(
    "--bfm_zero_repo",
    type=str,
    default=None,
    help="Deprecated/no-op: the FB-CPR-Aux code is now vendored under "
    "motion_tracking_rl/bfm_zero/_vendor. Kept only for backward-compatible CLI invocations.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after the sim app is up."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from pmt_tasks.registry_gym import register_pmt_tasks  # noqa: E402

# The BFM-Zero (FB-CPR-Aux) networks/agent are vendored inside PMT
# (motion_tracking_rl/bfm_zero/_vendor) and imported directly — no sys.path injection needed.
from motion_tracking_rl.bfm_zero import config as bfm_config  # noqa: E402
from motion_tracking_rl.bfm_zero.expert_streaming import build_static_expert_buffer_from_store  # noqa: E402
from motion_tracking_rl.bfm_zero.runner import BFMZeroRunner  # noqa: E402
from motion_tracking_rl.bfm_zero.vec_env import BFMZeroVecEnv  # noqa: E402
from pmt_tasks.env_cfgs.multi_motion_flat import TRACKED_BODY_NAMES  # noqa: E402


def _build_expert_buffer(vec_env, seq_length, expert_device):
    """Materialize a StaticExpertBuffer from the env motion command's loaded store.

    Works with both the old grouped streaming command and the pure-flat MultiMotionCommandV2 task.
    We read the command's loaded clip frames to build expert state/privileged observations.
    """
    cmd = vec_env.motion_command
    store = getattr(cmd, "data_store", None)
    if store is None:
        raise RuntimeError("Motion command does not expose ``data_store`` for expert sampling.")
    robot = vec_env.robot
    default_joint_pos = robot.data.default_joint_pos[0].detach()
    num = int(store.num_motions)
    # Grouped streaming store layout: [0, terrain_n) = terrain, [terrain_n, num) = flat.
    # Plain MultiMotionV2 is flat-only here, so mark every clip flat.
    terrain_n = getattr(store, "terrain_n", None)
    if terrain_n is None:
        is_flat_flags = [True] * num
    else:
        terrain_n = int(terrain_n)
        is_flat_flags = [i >= terrain_n for i in range(num)]
    return build_static_expert_buffer_from_store(
        store,
        seq_length=seq_length,
        default_joint_pos=default_joint_pos,
        is_flat_flags=is_flat_flags,
        device=expert_device,
    )


def main():
    from isaaclab_tasks.utils import parse_env_cfg

    register_pmt_tasks()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    print("[bfm_zero.train] building BFMZeroVecEnv ...", flush=True)
    vec_env = BFMZeroVecEnv(env, tracked_body_names=TRACKED_BODY_NAMES, device=args_cli.device)
    print("[bfm_zero.train] vec env ready; resetting once ...", flush=True)
    vec_env.reset()

    # The expert buffer is sampled by _sample_tracking_z WITHOUT a device move, so it must live on
    # the agent device. Default expert_buffer_device to the agent device.
    expert_device = args_cli.expert_buffer_device or args_cli.device

    # Closure to (re)build the expert buffer from the CURRENT resident streaming working set;
    # the runner calls this again after each working-set swap so expert sampling stays in sync.
    def expert_builder():
        return _build_expert_buffer(vec_env, args_cli.expert_seq_length, expert_device)

    print("[bfm_zero.train] building expert buffer ...", flush=True)
    expert_buffer = expert_builder()
    print(f"[bfm_zero.train] expert buffer ready: {len(expert_buffer)} windows", flush=True)

    # Clamp the expert rollout-trajectory length to what the resident clips can serve.
    max_len = getattr(expert_buffer, "_max_len", None)
    rollout_len = 250 if max_len is None else max(1, min(250, int(max_len) - 1))

    # Agent architecture preset: 'full' = paper-size; 'smoke' = small nets for pipeline validation.
    if args_cli.agent_preset == "smoke":
        # embedding_layers must be >= 2 (residual_embedding assertion).
        arch_kwargs = dict(hidden_dim=512, hidden_layers=2, embedding_layers=2,
                           disc_hidden_dim=256, disc_hidden_layers=2)
    else:
        arch_kwargs = {}

    agent_cfg = bfm_config.build_agent_config(
        device=args_cli.device,
        seq_length=args_cli.expert_seq_length,
        compile_agent=not args_cli.no_compile,
        bfm_zero_repo=args_cli.bfm_zero_repo,
        batch_size=args_cli.batch_size,
        rollout_expert_trajectories_length=rollout_len,
        **arch_kwargs,
    )

    print(f"[bfm_zero.train] building agent (preset={args_cli.agent_preset}) ...", flush=True)
    runner = BFMZeroRunner(
        vec_env=vec_env,
        expert_buffer=expert_buffer,
        device=args_cli.device,
        seed=args_cli.seed,
        num_seed_steps=args_cli.num_seed_steps,
        update_agent_every=args_cli.update_agent_every,
        num_agent_updates=args_cli.num_agent_updates,
        buffer_size=args_cli.buffer_size,
        batch_size=args_cli.batch_size,
        log_every=args_cli.log_every,
        checkpoint_every=args_cli.checkpoint_every,
        eval_every=args_cli.eval_every,
        eval_horizon=args_cli.eval_horizon,
        work_dir=args_cli.work_dir,
        agent_cfg=agent_cfg,
        motion_resample_frequency=args_cli.motion_resample_frequency,
        expert_builder=expert_builder,
        buffer_device=args_cli.buffer_device,
    )

    print("[bfm_zero.train] starting training loop ...", flush=True)
    runner.train(total_env_steps=args_cli.total_env_steps)
    runner.save("checkpoint")
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
