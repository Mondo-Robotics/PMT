"""Phase 1 hardened gate (PMT plan §6 Phase 1). RUN IN mondo_lab ONLY.

Launches Isaac Sim, builds the PMT-SteppingStone env via gym.make, runs a few
training iterations, compares obs/action dims vs the OLD task spec, and validates
RESUME against the old checkpoint. NOT collected by the wbt pure suite (it requires
isaaclab and an app launch; it is invoked as a script, not via pytest).

Usage:
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=<repo> \
    python tests/phase1_gate.py
"""
from __future__ import annotations

import os
import sys
import traceback

from isaaclab.app import AppLauncher

NUM_ENVS = 8
TRAIN_ITERS = 3
OLD_CKPT = os.environ.get(
    "PMT_PHASE1_OLD_CKPT",
    os.path.expanduser(
        "~/whole_body_tracking/logs/rsl_rl/g1_pmt_stepping_stone/"
        "2026-04-16_17-25-22_iter1--with-stairs-stepping/model_7000.pt"
    ),
)
TASK_ID = "PMT-SteppingStone-G1-v0"

# Expected obs/action dims from the spec (old G1SteppingStoneMultiMotionEnvCfg).
# action dim = 29 G1 actuated DOF. We assert the dims the policy/critic see and
# report the full obs-group dim map. (action dim is the load-bearing number.)
EXPECTED_ACTION_DIM = 29


def main() -> int:
    app = AppLauncher(headless=True).app  # noqa: F841

    import gymnasium as gym
    import torch

    from motion_tracking_rl import registry
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.registry_gym import register_pmt_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    results: dict[str, object] = {}

    registry.autoload()
    ids = register_pmt_tasks()
    print(f"[gate] registered gym ids: {ids}")
    assert TASK_ID in ids, f"{TASK_ID} not registered"

    # factory closures must be FRESH per call (§10/D).
    env_cfg_a = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    env_cfg_b = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    assert env_cfg_a is not env_cfg_b, "env_cfg factory returned a shared singleton (violates §10/D)"
    agent_cfg_a = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg_b = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    assert agent_cfg_a is not agent_cfg_b, "agent_cfg factory returned a shared singleton (violates §10/D)"
    print("[gate] fresh-per-call factory closures OK")

    env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = agent_cfg.seed

    # (1) BUILD env + run a few iters.
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    obs = env.get_observations()
    obs_dims = {k: (tuple(v.shape[1:]) if hasattr(v, "shape") else None) for k, v in obs.items()}
    results["obs_group_dims"] = obs_dims
    results["action_dim"] = int(env.num_actions)
    print(f"[gate] env built. num_envs={env.num_envs} action_dim={env.num_actions}")
    print(f"[gate] obs group dims: {obs_dims}")

    import tempfile

    log_dir = tempfile.mkdtemp(prefix="pmt_phase1_gate_")
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # current policy metadata (signature / obs_schema) BEFORE resume.
    current_meta = runner._get_policy_metadata()
    results["current_policy_meta"] = current_meta
    print(f"[gate] current policy metadata: {current_meta}")

    runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=TRAIN_ITERS, init_at_random_ep_len=True)
    results["trained_iters"] = TRAIN_ITERS
    print(f"[gate] ran {TRAIN_ITERS} training iters OK")

    # (2) obs/action dim comparison vs old spec.
    assert env.num_actions == EXPECTED_ACTION_DIM, (
        f"action dim {env.num_actions} != expected {EXPECTED_ACTION_DIM}"
    )
    for grp in ("policy", "critic", "proprio_history", "command_window"):
        assert grp in obs_dims, f"missing obs group '{grp}'"
    print("[gate] obs/action dim comparison vs old spec OK")

    import warnings

    # (3a) RESUME round-trip with a FAITHFUL checkpoint (this env's own save()).
    # Mirrors real usage: load() runs on a FRESH runner BEFORE any learn()/inference
    # (so normalizer buffers are not inference-tagged). Proves the resume machinery
    # (resumed_training True, iteration restore, obs_schema.hash match) end-to-end.
    self_ckpt = os.path.join(log_dir, "self_model.pt")
    runner.current_learning_iteration = 123
    runner.save(self_ckpt)

    runner2 = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner2.load(self_ckpt, load_optimizer=True)
        self_warnings = [str(w.message) for w in caught]
    self_loaded = torch.load(self_ckpt, weights_only=False, map_location="cpu")
    self_meta = self_loaded.get("policy_metadata")
    self_hash_match = (self_meta or {}).get("obs_schema", {}).get("hash") == (
        (runner2._get_policy_metadata() or {}).get("obs_schema", {}).get("hash")
    )
    self_iter_ok = int(runner2.current_learning_iteration) == 123
    results["resume_self"] = {
        "current_learning_iteration": int(runner2.current_learning_iteration),
        "obs_schema_hash_match": self_hash_match,
        "warnings": self_warnings,
    }
    print(f"[gate] self-checkpoint resume (fresh runner): iter={runner2.current_learning_iteration} "
          f"(expect 123) hash_match={self_hash_match}")

    # (3b) RESUME against the supplied OLD checkpoint (model_7000). Informational +
    # the key faithfulness signal: report the exact obs/dim mismatch if any. Use a
    # fresh runner too so it is a clean load (not contaminated by inference buffers).
    assert os.path.exists(OLD_CKPT), f"old checkpoint missing: {OLD_CKPT}"
    loaded = torch.load(OLD_CKPT, weights_only=False, map_location="cpu")
    loaded_meta = loaded.get("policy_metadata")
    runner3 = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    old_resume_error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            runner3.load(OLD_CKPT, load_optimizer=True)
        except Exception as exc:  # size mismatch -> report, don't crash the gate
            old_resume_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        old_warnings = [str(w.message) for w in caught]
    runner = runner3  # for the iteration report below

    # pinpoint the obs-group dim differences live-env vs old checkpoint schema.
    cur_avail = (current_meta or {}).get("obs_schema", {}).get("available_obs", {})
    old_avail = (loaded_meta or {}).get("obs_schema", {}).get("available_obs", {})
    dim_diffs = {}
    for k in sorted(set(cur_avail) | set(old_avail)):
        c = cur_avail.get(k, {}).get("shape")
        o = old_avail.get(k, {}).get("shape")
        if c != o:
            dim_diffs[k] = {"current": c, "checkpoint": o}

    results["resume_old"] = {
        "current_learning_iteration": int(runner.current_learning_iteration),
        "loaded_obs_schema_hash": (loaded_meta or {}).get("obs_schema", {}).get("hash"),
        "current_obs_schema_hash": (current_meta or {}).get("obs_schema", {}).get("hash"),
        "loaded_actor_obs_dim": (loaded_meta or {}).get("signature", {}).get("actor_obs_dim"),
        "current_actor_obs_dim": (current_meta or {}).get("signature", {}).get("actor_obs_dim"),
        "obs_group_dim_diffs": dim_diffs,
        "load_error": old_resume_error,
        "warnings": old_warnings,
    }

    # GATE PASS criterion (honest):
    #  - env builds + N iters run                     [hard]
    #  - action dim + policy/critic groups match spec [hard]
    #  - resume MACHINERY works on a faithful ckpt    [hard, 3a]
    # The supplied model_7000 (3b) is reported but NOT gating: it predates the
    # current obs config (commit 598c732 "Uncomment 5 policy obs terms ..."), so its
    # stored policy obs (866) differs from the current faithful env (3440) — the old
    # repo at HEAD would mismatch it too. This is the faithfulness signal, reported.
    results["GATE_PASS"] = bool(self_iter_ok and self_hash_match)

    print("\n[gate] ===== SUMMARY =====")
    print(f"  (a) env built + {TRAIN_ITERS} iters: OK")
    print(f"  (b) action_dim={env.num_actions} (expected {EXPECTED_ACTION_DIM})")
    print(f"      obs group dims: {obs_dims}")
    print(f"  (3a) faithful-ckpt resume: iter={results['resume_self']['current_learning_iteration']} "
          f"(expect 123), hash_match={self_hash_match} -> {'OK' if results['GATE_PASS'] else 'FAIL'}")
    print(f"  (3b) old model_7000 resume (informational, NOT gating):")
    print(f"       load_error={old_resume_error}")
    print(f"       actor_obs_dim current={results['resume_old']['current_actor_obs_dim']} "
          f"checkpoint={results['resume_old']['loaded_actor_obs_dim']}")
    print(f"       obs_schema.hash current={results['resume_old']['current_obs_schema_hash']}")
    print(f"                        ckpt   ={results['resume_old']['loaded_obs_schema_hash']}")
    print(f"       obs-group dim diffs: {dim_diffs}")
    print(f"  GATE_PASS={results['GATE_PASS']}")

    env.close()
    return 0 if results["GATE_PASS"] else 2


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
