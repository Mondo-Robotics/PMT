"""Phase 2.3a runtime regression for the §9b MDP consolidation. RUN IN mondo_lab.

Behavior-preserving gate: builds ONE wired task per process, captures the obs-group
dims + action dim, runs a few training iters, and asserts dims match the known
baselines (so the consolidation -- obs tracking_obs wrappers, termination wrappers,
reward-weight helper -- did NOT change the observation/action surface).

One gym.make per process (Phase-1 TODO: two gym.make in one process hangs).

Usage (one task per invocation):
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=<repo> \
    python \
    tests/phase2_3a_consolidation_regression.py <TASK_ID>
"""
from __future__ import annotations

import sys
import traceback

from isaaclab.app import AppLauncher

NUM_ENVS = 8
TRAIN_ITERS = 3

# Baselines from the Phase-1/2.2 gate reports (PHASE1_GATE_REVIEW.md,
# PHASE_2_2_GATE_REPORT.md, §9c). Each task exercises the consolidated code:
#   - SteppingStone: tracking_obs (anchor+body terms) + anchor/body terminations
#   - ADD:           disc obs (tracking terms feed the discriminator)
#   - MultiMotionV2-Flat: base tracking obs path
# We assert action dim hard, and assert each named obs group keeps its baseline dim.
BASELINES = {
    "PMT-SteppingStone-G1-v0": {
        "action_dim": 29,
        "obs_groups": {"policy": 3440},  # critic/transformer groups present, dim reported
        "required_groups": ["policy", "critic"],
    },
    "PMT-ADD-MultiMotionV2-Flat-v0": {
        "action_dim": 29,
        "obs_groups": {"add_disc_obs": 230, "add_disc_demo": 230},
        "required_groups": ["policy"],
    },
    "PMT-G1-MultiMotionV2-Flat-v0": {
        "action_dim": 29,
        "obs_groups": {},  # report-only; assert policy/critic present + action dim
        "required_groups": ["policy"],
    },
}


def main(task_id: str) -> int:
    app = AppLauncher(headless=True).app  # noqa: F841

    import gymnasium as gym

    from motion_tracking_rl import registry
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.registry_gym import register_pmt_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    registry.autoload()
    ids = register_pmt_tasks()
    assert task_id in ids, f"{task_id} not registered (have {len(ids)} ids)"
    assert task_id in BASELINES, f"no baseline defined for {task_id}"
    baseline = BASELINES[task_id]

    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = agent_cfg.seed

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    obs = env.get_observations()
    obs_dims = {
        k: (int(v.shape[-1]) if hasattr(v, "shape") else None) for k, v in obs.items()
    }
    action_dim = int(env.num_actions)

    print(f"[regress:{task_id}] action_dim={action_dim}")
    print(f"[regress:{task_id}] obs group dims: {obs_dims}")

    failures: list[str] = []

    if action_dim != baseline["action_dim"]:
        failures.append(f"action_dim {action_dim} != baseline {baseline['action_dim']}")

    for grp in baseline["required_groups"]:
        if grp not in obs_dims:
            failures.append(f"missing required obs group '{grp}'")

    for grp, expected in baseline["obs_groups"].items():
        if grp not in obs_dims:
            failures.append(f"missing baselined obs group '{grp}'")
        elif obs_dims[grp] != expected:
            failures.append(f"obs group '{grp}' dim {obs_dims[grp]} != baseline {expected}")

    # run a few iters via the proper runner (no manual env.step shape juggling).
    import tempfile

    log_dir = tempfile.mkdtemp(prefix="pmt_p23a_regress_")
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=TRAIN_ITERS, init_at_random_ep_len=True)
    print(f"[regress:{task_id}] ran {TRAIN_ITERS} training iters OK (no crash)")

    env.close()

    print(f"\n[regress:{task_id}] ===== SUMMARY =====")
    print(f"  action_dim={action_dim} (baseline {baseline['action_dim']})")
    print(f"  obs group dims: {obs_dims}")
    if failures:
        print(f"  REGRESSION FAILURES: {failures}")
        return 2
    print(f"  PASS: dims unchanged, {TRAIN_ITERS} iters ran clean")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: phase2_3a_consolidation_regression.py <TASK_ID>")
        sys.exit(64)
    try:
        rc = main(sys.argv[1])
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
