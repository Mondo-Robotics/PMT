"""PMT train.py — ONE script, runner dispatched from the derived runner axis.

Mirrors the old scripts/rsl_rl/train.py loop but (1) dispatches the runner CLASS
from the agent cfg's class_name via motion_tracking_rl.registry.get_runner (instead
of hard-importing OnPolicyRunner), so distillation tasks route to DistillationRunner;
(2) resolves env/agent cfgs through PMT's factory closures registered on the gym id;
(3) gets motion files from the resolved config (no wandb).

App-launch-first ordering (required): AppLauncher().app BEFORE importing isaaclab.envs.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a PMT RL agent.")
parser.add_argument("--task", type=str, required=True, help="PMT gym task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of envs.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--profile", type=str, default=None, help="PMT paths profile (local|cluster).")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
parser.add_argument("--resume_path", type=str, default=None, help="Explicit checkpoint .pt to resume.")
AppLauncher.add_app_launcher_args(parser)
args_cli, task_overrides = parser.parse_known_args()

if args_cli.profile is not None:
    os.environ["PMT_PROFILE"] = args_cli.profile

# launch the app FIRST.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from motion_tracking_rl import registry  # noqa: E402
from pmt_tasks.builder import build_agent_cfg, build_env_cfg  # noqa: E402
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from pmt_tasks.registry_gym import gym_id_for_stem, register_pmt_tasks  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _parse_task_overrides(raw_args: list[str]) -> list[str]:
    overrides = [arg for arg in raw_args if "=" in arg and not arg.startswith("-")]
    unsupported = [arg for arg in raw_args if arg not in overrides]
    if unsupported:
        raise ValueError(
            "Unsupported extra arguments for scripts/train.py: "
            f"{unsupported}. Use key=value task overrides only where supported."
        )
    return overrides


def main():
    registry.autoload()
    register_pmt_tasks()

    # resolve env + agent cfgs through the registered factory closures (§10).
    overrides = _parse_task_overrides(task_overrides)
    if overrides:
        sonic_task_id = gym_id_for_stem("sonic_multimotion_flat")
        if args_cli.task != sonic_task_id:
            raise ValueError(
                "Task config overrides in scripts/train.py are currently supported "
                f"only for {sonic_task_id}; got task {args_cli.task} with "
                f"overrides {overrides}."
            )
        env_cfg = build_env_cfg("sonic_multimotion_flat", profile=args_cli.profile, overrides=overrides)
        agent_cfg = build_agent_cfg("sonic_multimotion_flat", profile=args_cli.profile, overrides=overrides)
    else:
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # non-hydra CLI overrides.
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume:
        agent_cfg.resume = True
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # logging dir.
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # build env.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # DISPATCH the runner class from the agent cfg's class_name (registry, §3b/§10).
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # resume.
    if agent_cfg.resume:
        resume_path = args_cli.resume_path or get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO] resuming from checkpoint: {resume_path}")
        runner.load(resume_path)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
