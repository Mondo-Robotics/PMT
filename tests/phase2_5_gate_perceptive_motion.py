"""Phase 2.5 runtime gate — PMT-PerceptiveMotionTokenTracker-G1-v0. RUN IN mondo_lab.

Launches Isaac Sim, builds the PMT token-tracker env via gym.make, constructs the
PerceptiveMotionTokenTracker network FROM SCRATCH (pmt_only_mode, no pretrained PMT
checkpoint), and runs a few PPO iterations. ONE gym.make per process.

Usage:
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=<repo> \
    python tests/phase2_5_gate_perceptive_motion.py
"""
from __future__ import annotations

import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

NUM_ENVS = 4
TRAIN_ITERS = 2
TASK_ID = "PMT-PerceptiveMotionTokenTracker-G1-v0"
EXPECTED_ACTION_DIM = 29


def main() -> int:
    app = AppLauncher(headless=True).app  # noqa: F841

    import gymnasium as gym

    from motion_tracking_rl import registry
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.registry_gym import register_pmt_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    registry.autoload()
    ids = register_pmt_tasks()
    assert TASK_ID in ids, f"{TASK_ID} not registered (got {ids})"
    print(f"[gate] {TASK_ID} registered")

    env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = agent_cfg.seed

    # ckpt-required-disabled confirmation (the un-gated gate target).
    print(f"[gate] policy.class_name = {agent_cfg.policy.class_name}")
    print(f"[gate] require_pmt_checkpoint = {agent_cfg.policy.require_pmt_checkpoint}")
    print(f"[gate] pmt_only_mode = {agent_cfg.policy.pmt_only_mode}")
    print(f"[gate] freeze_pmt = {agent_cfg.policy.freeze_pmt}")
    print(f"[gate] obs_groups = {agent_cfg.obs_groups}")
    assert agent_cfg.policy.class_name == "PerceptiveMotionTokenTracker"
    assert agent_cfg.policy.require_pmt_checkpoint is False

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    obs = env.get_observations()
    obs_dims = {k: (tuple(v.shape[1:]) if hasattr(v, "shape") else None) for k, v in obs.items()}
    print(f"[gate] env built. num_envs={env.num_envs} action_dim={env.num_actions}")
    print(f"[gate] obs keys present: {sorted(obs.keys())}")
    print(f"[gate] obs group dims: {obs_dims}")
    assert env.num_actions == EXPECTED_ACTION_DIM

    log_dir = tempfile.mkdtemp(prefix="pmt_phase2_5_gate_")
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # network-constructed confirmation
    policy = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    print(f"[gate] network constructed: {type(policy).__name__}")
    print(f"[gate] resolved obs_groups (runner): {dict(runner.cfg['obs_groups'])}")
    assert "future_motion_window" in runner.cfg["obs_groups"], "future_motion_window missing"
    assert "height_scan" in runner.cfg["obs_groups"], "height_scan missing"

    runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=TRAIN_ITERS, init_at_random_ep_len=True)
    print(f"[gate] ran {TRAIN_ITERS} PPO iters from scratch OK")
    print("[gate] PHASE 2.5 RUNTIME GATE: GO")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
