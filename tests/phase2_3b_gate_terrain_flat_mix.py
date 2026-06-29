"""Phase-2.3b runtime gate: TerrainFlatMix via the FLEXIBLE UnifiedMotionCommandV2.

RUN IN mondo_lab ONLY, ONE TASK PER PROCESS. Launch Isaac Sim ONCE, make
PMT-TerrainFlatMix-G1-v0 (num_envs 8, headless), run 3 iters headless.

GATE:
  - gym.make + 3 iters, no crash; report action_dim, policy/critic obs dims
  - ASSERT the command is UnifiedMultiMotionCommand (flexible), NOT GroupedMultiMotionCommandV2
  - ASSERT both terrain AND flat clips loaded into ONE store (report counts)
  - ASSERT per-clip env-origin split after a reset: envs playing flat clips have
    origin ~= flat_origin (90,0,0); envs playing terrain clips have origin ~= 0
  - report a sample of (env, is_terrain, origin) rows showing the split

Usage:
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=<repo> \
    python \
    tests/phase2_3b_gate_terrain_flat_mix.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "PMT-TerrainFlatMix-G1-v0"
NUM_ENVS = 8
TRAIN_ITERS = 3


def main() -> int:
    app = AppLauncher(headless=True).app  # noqa: F841

    import torch
    import gymnasium as gym

    from motion_tracking_rl import registry
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.registry_gym import register_pmt_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    registry.autoload()
    ids = register_pmt_tasks()
    assert TASK_ID in ids, f"{TASK_ID} not registered (have {ids})"

    f1 = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    f2 = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    assert f1 is not f2, "env_cfg factory not fresh-per-call"

    env_cfg = f1
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = agent_cfg.seed

    cmd_cfg = env_cfg.commands.motion
    cmd_cfg_class = type(cmd_cfg).__name__

    # GATE SPEEDUP ONLY: the production task discovers 500 terrain + 40 flat clips; an
    # eager CUDA load of all 540 at chunk_length=4000 is far too heavy for a 3-iter smoke
    # gate (>10 min just to load). Trim to a small mixed subset so the gate exercises the
    # SAME code path (one store, both groups, per-clip origin split) quickly. The full
    # clip set is the production config and is left untouched.
    MAX_TERRAIN, MAX_FLAT = 4, 4
    terr = [p for p in cmd_cfg.motion_files if p in set(cmd_cfg.terrain_motion_files)]
    flat = [p for p in cmd_cfg.motion_files if p not in set(cmd_cfg.terrain_motion_files)]
    terr, flat = terr[:MAX_TERRAIN], flat[:MAX_FLAT]
    cmd_cfg.terrain_motion_files = list(terr)
    cmd_cfg.motion_files = list(terr) + list(flat)
    cmd_cfg.chunk_length = 0  # ragged: no padding to 4000 frames for the tiny gate set
    print(f"[gate2.3b] trimmed to terrain={len(terr)} flat={len(flat)} for the smoke gate")

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    obs_dims = {
        k: (tuple(v.shape[1:]) if hasattr(v, "shape") else None) for k, v in obs.items()
    }
    action_dim = int(wrapped.num_actions)

    # the live command term
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("motion")
    cmd_class = type(command).__name__

    # clip counts in the ONE store
    is_terrain_clip = command._is_terrain_clip  # [num_clips] bool
    n_terrain = int(is_terrain_clip.sum().item())
    n_flat = int((~is_terrain_clip).sum().item())
    n_clips = int(is_terrain_clip.numel())

    # The env's initial full reset (during gym.make) already ran _resample_command for
    # all envs, which refreshed _is_terrain_env and injected per-clip env origins. Inspect
    # that state directly (no manual resample needed).
    is_terrain_env = command._is_terrain_env.clone()  # [num_envs] bool
    origins = base_env.scene.terrain.env_origins.clone()  # [num_envs, 3]
    flat_origin = torch.tensor(
        list(cmd_cfg.flat_origin), device=origins.device, dtype=origins.dtype
    )

    # per-env origin split assertion
    flat_mask = ~is_terrain_env
    terr_mask = is_terrain_env
    flat_ok = (
        bool((origins[flat_mask] - flat_origin).abs().max() < 1e-3)
        if bool(flat_mask.any()) else True
    )
    terr_ok = (
        bool(origins[terr_mask].abs().max() < 1e-3)
        if bool(terr_mask.any()) else True
    )

    sample = [
        {
            "env": i,
            "is_terrain": bool(is_terrain_env[i].item()),
            "origin": [round(float(x), 3) for x in origins[i].tolist()],
        }
        for i in range(NUM_ENVS)
    ]

    # run iters (no crash)
    log_dir = tempfile.mkdtemp(prefix="pmt_gate2_3b_")
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=TRAIN_ITERS, init_at_random_ep_len=True)
    wrapped.close()

    is_unified = cmd_class == "UnifiedMultiMotionCommand"
    not_grouped = "Grouped" not in cmd_class
    one_store_both = n_terrain > 0 and n_flat > 0

    summary = {
        "task": TASK_ID,
        "ran_iters": TRAIN_ITERS,
        "action_dim": action_dim,
        "policy_obs_dim": obs_dims.get("policy"),
        "critic_obs_dim": obs_dims.get("critic"),
        "cmd_cfg_class": cmd_cfg_class,
        "command_class": cmd_class,
        "is_unified_not_grouped": is_unified and not_grouped,
        "num_clips_total": n_clips,
        "num_terrain_clips": n_terrain,
        "num_flat_clips": n_flat,
        "one_store_both_groups": one_store_both,
        "flat_origin_cfg": [float(x) for x in cmd_cfg.flat_origin],
        "flat_envs_at_flat_origin": flat_ok,
        "terrain_envs_at_zero_origin": terr_ok,
        "per_env_origin_sample": sample,
    }
    ok = (
        action_dim > 0
        and obs_dims.get("policy") is not None
        and is_unified
        and not_grouped
        and one_store_both
        and flat_ok
        and terr_ok
    )

    print("\n[gate2.3b] ================ SUMMARY ================")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[gate2.3b] OVERALL_PASS={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
