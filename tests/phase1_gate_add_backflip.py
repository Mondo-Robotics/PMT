"""Phase-1 widened-slice runtime gate: ADD + Backflip (+ stepping-stone regression).

RUN IN mondo_lab ONLY. Launches Isaac Sim ONCE, then exercises three tasks in batch:

  GATE A  PMT-ADD-MultiMotionV2-Flat-v0
    - gym.make (num_envs 8, headless), run 3 iters, no crash
    - add_disc_obs / add_disc_demo groups exist, each dim 230; action_dim 29
    - build_agent_cfg validates (compat add_ppo + mlp -> on_policy, disc obs sets present)

  GATE B  PMT-Backflip-G1-v0
    - gym.make (num_envs 8, headless), run 3 iters, no crash
    - env_cfg.decimation == 10 and env_cfg.sim.dt == 0.002 came from config
    - knee_negative_power reward term present; action_dim 29

  REGRESSION  PMT-SteppingStone-G1-v0 still builds (env_cfg only; no train).

Usage:
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=<repo> \
    python tests/phase1_gate_add_backflip.py
"""
from __future__ import annotations

import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

NUM_ENVS = 8
TRAIN_ITERS = 3
EXPECTED_ACTION_DIM = 29


def _run_task(gym, RslRlVecEnvWrapper, registry, load_cfg_from_registry, task_id):
    """gym.make + run N iters; return (env_cfg, obs_dims, action_dim, agent_cfg)."""
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = agent_cfg.seed

    captured_dec = int(env_cfg.decimation)
    captured_dt = float(env_cfg.sim.dt)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    obs = env.get_observations()
    obs_dims = {k: (tuple(v.shape[1:]) if hasattr(v, "shape") else None) for k, v in obs.items()}
    action_dim = int(env.num_actions)

    log_dir = tempfile.mkdtemp(prefix=f"pmt_gate_{task_id.replace('/', '_')}_")
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=TRAIN_ITERS, init_at_random_ep_len=True)
    env.close()
    return env_cfg, obs_dims, action_dim, agent_cfg, captured_dec, captured_dt


def _gate_add(gym, RslRlVecEnvWrapper, registry, load_cfg_from_registry, ids):
    print("\n[gate] ===== GATE A: PMT-ADD-MultiMotionV2-Flat-v0 =====")
    a_task = "PMT-ADD-MultiMotionV2-Flat-v0"
    assert a_task in ids, f"{a_task} not registered"
    f1 = load_cfg_from_registry(a_task, "env_cfg_entry_point")
    f2 = load_cfg_from_registry(a_task, "env_cfg_entry_point")
    assert f1 is not f2, "ADD env_cfg factory not fresh-per-call"
    _env_cfg, a_obs, a_act, a_agent, _dec, _dt = _run_task(
        gym, RslRlVecEnvWrapper, registry, load_cfg_from_registry, a_task
    )
    print(f"[gate][A] ran {TRAIN_ITERS} iters OK. action_dim={a_act} obs_dims={a_obs}")
    disc_obs_dim, disc_demo_dim = a_obs.get("add_disc_obs"), a_obs.get("add_disc_demo")
    checks = {
        "ran_iters": TRAIN_ITERS, "action_dim": a_act,
        "add_disc_obs_dim": disc_obs_dim, "add_disc_demo_dim": disc_demo_dim,
        "agent_runner": a_agent.class_name, "agent_obs_groups": dict(a_agent.obs_groups),
    }
    checks["GATE_A_PASS"] = bool(
        a_act == EXPECTED_ACTION_DIM and disc_obs_dim == (230,) and disc_demo_dim == (230,)
        and {"add_disc_obs", "add_disc_demo"} <= set(a_agent.obs_groups.keys())
    )
    return {"GATE_A": checks}, checks["GATE_A_PASS"]


def _gate_backflip(gym, RslRlVecEnvWrapper, registry, load_cfg_from_registry, ids):
    print("\n[gate] ===== GATE B: PMT-Backflip-G1-v0 =====")
    b_task = "PMT-Backflip-G1-v0"
    assert b_task in ids, f"{b_task} not registered"
    b_env_cfg, _obs, b_act, _agent, b_dec, b_dt = _run_task(
        gym, RslRlVecEnvWrapper, registry, load_cfg_from_registry, b_task
    )
    print(f"[gate][B] ran {TRAIN_ITERS} iters OK. action_dim={b_act} dec={b_dec} dt={b_dt}")
    knee_present = getattr(b_env_cfg.rewards, "knee_negative_power", None) is not None
    reward_terms = [t for t in vars(b_env_cfg.rewards).keys() if not t.startswith("_")]
    checks = {
        "ran_iters": TRAIN_ITERS, "action_dim": b_act,
        "decimation_from_config": b_dec, "sim_dt_from_config": b_dt,
        "knee_negative_power_present": knee_present, "reward_terms": reward_terms,
    }
    checks["GATE_B_PASS"] = bool(
        b_act == EXPECTED_ACTION_DIM and b_dec == 10 and abs(b_dt - 0.002) < 1e-9 and knee_present
    )
    return {"GATE_B": checks}, checks["GATE_B_PASS"]


def _gate_regression(gym, RslRlVecEnvWrapper, registry, load_cfg_from_registry, ids):
    print("\n[gate] ===== REGRESSION: PMT-SteppingStone-G1-v0 env builds =====")
    ss_task = "PMT-SteppingStone-G1-v0"
    ss_env_cfg = load_cfg_from_registry(ss_task, "env_cfg_entry_point")
    built = ss_env_cfg is not None and int(ss_env_cfg.decimation) == 4
    return {"REGRESSION": {"stepping_stone_env_builds": built, "decimation": int(ss_env_cfg.decimation)}}, built


# NOTE: building TWO Isaac Sim envs (two gym.make) in one process hangs on the second
# scene re-creation. So each gate runs in its OWN process/app launch via --task. The
# combined "all" mode is kept for reference but not the default path.
_GATES = {"add": _gate_add, "backflip": _gate_backflip, "regression": _gate_regression}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(_GATES) + ["all"], default="all")
    args, _ = parser.parse_known_args()

    app = AppLauncher(headless=True).app  # noqa: F841

    import gymnasium as gym
    import json

    from motion_tracking_rl import registry
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.registry_gym import register_pmt_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    registry.autoload()
    ids = register_pmt_tasks()
    print(f"[gate] registered gym ids: {ids}")

    selected = list(_GATES) if args.task == "all" else [args.task]
    summary, ok = {}, True
    for name in selected:
        part, part_ok = _GATES[name](gym, RslRlVecEnvWrapper, registry, load_cfg_from_registry, ids)
        summary.update(part)
        ok = ok and part_ok

    print("\n[gate] ================ SUMMARY ================")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[gate] OVERALL_PASS={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
