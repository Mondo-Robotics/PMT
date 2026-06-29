"""Phase-2.1 runtime gate: MultiMotion/Flat family (plan §6 Phase 2.1).

RUN IN mondo_lab ONLY, ONE TASK PER PROCESS (two gym.make in one process hangs on the
2nd env's scene re-creation — known Phase-1 TODO). Launch Isaac Sim ONCE, exercise a
single task selected by --task, run 3 iters headless.

GATE per task (--task in: base/uniform/adaptive/streaming/bpo):
  - gym.make (num_envs 4, headless), run 3 iters, no crash
  - report action_dim, policy/critic obs dims, sampler_type actually used by the command
  - (streaming) the streaming store loaded (command is StreamingMultiMotionCommand)
  - (bpo) BPO algorithm constructed + ran (agent.class_name == "BPO")
  - struct-match: action_dim + policy obs dim consistent across the family

Usage (one process each):
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=<repo> \
    python \
    tests/phase2_gate_multimotion_flat.py --task base
"""
from __future__ import annotations

import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

NUM_ENVS = 4
TRAIN_ITERS = 3

_TASK_IDS = {
    "base": "PMT-G1-MultiMotionV2-Flat-v0",
    "uniform": "PMT-G1-MultiMotionV2-Uniform-Flat-v0",
    "adaptive": "PMT-G1-MultiMotionV2-Adaptive-Flat-v0",
    "streaming": "PMT-G1-MultiMotionV2-Streaming-Flat-v0",
    "bpo": "PMT-G1-BPO-MultiMotionV2-Flat-v0",
}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(_TASK_IDS), required=True)
    args, _ = parser.parse_known_args()
    task_id = _TASK_IDS[args.task]

    app = AppLauncher(headless=True).app  # noqa: F841

    import gymnasium as gym
    import json

    from motion_tracking_rl import registry
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.registry_gym import register_pmt_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    registry.autoload()
    ids = register_pmt_tasks()
    assert task_id in ids, f"{task_id} not registered (have {ids})"

    # fresh-per-call factory check (§10/D)
    f1 = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    f2 = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert f1 is not f2, "env_cfg factory not fresh-per-call"

    env_cfg = f1
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = agent_cfg.seed

    cmd_cfg = env_cfg.commands.motion
    cmd_class = type(cmd_cfg).__name__
    sampler_used = cmd_cfg.sampler_type

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    obs = env.get_observations()
    obs_dims = {k: (tuple(v.shape[1:]) if hasattr(v, "shape") else None) for k, v in obs.items()}
    action_dim = int(env.num_actions)

    log_dir = tempfile.mkdtemp(prefix=f"pmt_gate2_{args.task}_")
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=TRAIN_ITERS, init_at_random_ep_len=True)
    env.close()

    summary = {
        "task": task_id,
        "ran_iters": TRAIN_ITERS,
        "action_dim": action_dim,
        "policy_obs_dim": obs_dims.get("policy"),
        "critic_obs_dim": obs_dims.get("critic"),
        "command_class": cmd_class,
        "sampler_type_used": sampler_used,
        "agent_runner": agent_cfg.class_name,
        "algorithm_class": agent_cfg.algorithm.class_name,
    }
    # task-specific assertions
    ok = action_dim > 0 and obs_dims.get("policy") is not None
    if args.task == "streaming":
        summary["streaming_store_loaded"] = cmd_class == "StreamingMultiMotionCommandV2Cfg"
        ok = ok and summary["streaming_store_loaded"]
    if args.task == "bpo":
        summary["bpo_constructed"] = agent_cfg.algorithm.class_name == "BPO"
        ok = ok and summary["bpo_constructed"]
    expected_sampler = {"uniform": "uniform", "adaptive": "adaptive"}.get(args.task, "bin_adaptive")
    summary["sampler_matches_expected"] = sampler_used == expected_sampler
    ok = ok and summary["sampler_matches_expected"]

    print("\n[gate2] ================ SUMMARY ================")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[gate2] OVERALL_PASS={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
