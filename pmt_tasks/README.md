# pmt_tasks — the PMT task layer

`pmt_tasks/` is the **task composition + registration layer** of this config-driven Isaac Lab
motion-tracking RL repo. The `configs/` tree is the source of truth (independent axes:
`robot` / `terrain` / `motion` / `scene` / `sensor` / `obs` / `reward` / `network` /
`algorithm` / `stage` / `runner`). `pmt_tasks/builder.py` composes a chosen
`configs/task/<stem>.yaml` into a concrete Isaac Lab env cfg + agent cfg, `registry_gym.py`
registers one gym id per task, and `scripts/train.py` runs it through Isaac Lab's normal
`gym.make` / `load_cfg_from_registry` chain.

```
configs/task/<stem>.yaml          source of truth: a defaults: axis list + overrides
        │
        ▼
build_task_config()               SELECT (compose defaults) → DERIVE (obs_groups,
 (builder.py)                      reward_weights) → VALIDATE (compat) → resolved OmegaConf
        │
        ├── build_env_cfg(stem)    → _ENV_BUILDERS[stem] → @configclass TrackingEnvCfg
        └── build_agent_cfg(stem)  → per-stem RslRlOnPolicyRunnerCfg
        │
        ▼
register_pmt_tasks()              gym.register(_TASK_ID_MAP[stem]) with fresh-per-call
 (registry_gym.py)                factory closures as entry points
        │
        ▼
scripts/train.py                  load_cfg_from_registry(task_id, …) → ManagerBasedRLEnv
```

## Directory map

| Path | What it is |
| --- | --- |
| `builder.py` | The build pass. `build_task_config()` does SELECT→DERIVE→VALIDATE→emit (pure OmegaConf, no isaaclab); `build_env_cfg()` / `build_agent_cfg()` lazily emit fresh `@configclass` instances via the per-stem `_ENV_BUILDERS` dispatch and the agent-cfg dispatch. `_resolve_backend()` picks `isaaclab` (default) vs `mjlab`. |
| `derive.py` | Pure derivation rules (no isaaclab). `derive_obs_groups()` computes the obs-set→obs-group map; `derive_reward_weights()` surfaces reward weights as data; `anchor_in_actor_obs()` flags the privileged anchor term. |
| `registry_gym.py` | `_TASK_ID_MAP` (task stem → gym id) + `register_pmt_tasks()`: one `gym.register` per `configs/task/*.yaml` using factory closures, plus the separate `BFM-Zero-Flat-MultiMotionV2-G1-v0` env-only registration. |
| `env_cfgs/` | Per-family environment `@configclass` cfgs (scene + obs groups + commands + rewards + terminations). Has its own [README.md](env_cfgs/README.md). |
| `agent_cfgs/` | Per-family agent / PPO runner cfgs (`RslRlOnPolicyRunnerCfg` etc.): `transformer.py`, `multi_motion_ppo.py`, `add_ppo.py`, `sonic_ppo.py`, `distillation.py`, `finetune.py`, `rgmt.py`, `pmt_transformer.py`, `perceptive_motion_token.py`, `perceptive_residual_behavior_token.py`. |
| `mdp/` | MDP terms: `commands/` (motion commands + samplers/storage), `rewards.py`, `observations.py`, `terminations.py`, `events.py`. Degrades gracefully (USD-free) when isaaclab is absent. |
| `robots/` | Robot asset cfgs (`g1.py`, `actuator.py`, …). |
| `backends/` | Backend dispatch. `mjlab.py` provides `_MJLAB_ENV_BUILDERS` for the optional mjlab backend; default stays isaaclab. |
| `isaaclab_rl/` | rsl_rl integration glue. |
| `utils/` | Path/terrain helpers (`motion_paths.py`, `terrain*.py`). |
| `tracking_env_cfg.py` | Base `TrackingEnvCfg` reused by the `env_cfgs/` families. |
| `asset_config.py`, `path_defaults.py` | Robot-binary resolver and default path roots. |
| `__init__.py` | Re-exports the pure builder/derive API (`build_task_config`, `load_paths`, `derive_*`). |

## Task stem → gym id

From `registry_gym._TASK_ID_MAP` (any unmapped stem falls back to `PMT-<stem>-v0`):

| Task stem (`configs/task/<stem>.yaml`) | Gym id |
| --- | --- |
| `multimotionv2_flat` | `PMT-G1-MultiMotionV2-Flat-v0` |
| `multimotionv2_uniform_flat` | `PMT-G1-MultiMotionV2-Uniform-Flat-v0` |
| `multimotionv2_adaptive_flat` | `PMT-G1-MultiMotionV2-Adaptive-Flat-v0` |
| `multimotionv2_streaming_flat` | `PMT-G1-MultiMotionV2-Streaming-Flat-v0` |
| `multimotionv2_streaming_100style` | `PMT-G1-MultiMotionV2-Streaming-100style-Flat-v0` |
| `multimotionv2_100style_flat` | `PMT-G1-MultiMotionV2-100style-Flat-v0` |
| `bpo_multimotionv2_flat` | `PMT-G1-BPO-MultiMotionV2-Flat-v0` |
| `fpo_plus_flat` | `PMT-G1-FPOPlus-SingleClip-Flat-v0` |
| `add_multimotion_flat` | `PMT-ADD-MultiMotionV2-Flat-v0` |
| `sonic_multimotion_flat` | `PMT-SONIC-G1-MultiMotionV2-Flat-v0` |
| `rgmt` | `RGMT-G1-v0` |
| `pmt_stepping_stone` | `PMT-SteppingStone-G1-v0` |
| `backflip` | `PMT-Backflip-G1-v0` |
| `terrain_flat_mix` | `PMT-TerrainFlatMix-G1-v0` |
| `walk_dance_bigmap` | `PMT-WalkDanceBigMap-G1-v0` |
| `cartwheel_bigmap` | `PMT-CartwheelBigMap-G1-v0` |
| `pmt_adaptive_sampling` | `PMT-AdaptiveSampling-G1-v0` |
| `pmt_adaptive_sampling_baseline` | `PMT-AdaptiveSampling-Baseline-G1-v0` |
| `distill_stepping_stone` | `PMT-Distill-SteppingStone-G1-v0` |
| `distill_stepping_stone_latent_anchor` | `PMT-Distill-SteppingStone-LatentAnchor-G1-v0` |
| `ppofinetune_vision_teacher_stepping_stone_latent_anchor` | `PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0` |
| `perceptive_motion_token_tracker` | `PMT-PerceptiveMotionTokenTracker-G1-v0` |
| `pmt_pcrbt` | `PMT-PCaRBT-G1-v0` |
| `pmt_pcrbt_100style` | `PMT-PCaRBT-100style-G1-v0` |

`register_pmt_tasks()` additionally registers `BFM-Zero-Flat-MultiMotionV2-G1-v0` (env cfg only —
it uses its own separate runner via `scripts/bfm_zero/train.py`, not an rsl_rl agent cfg).

Train any task: `scripts/train.py --task <gym-id> --num_envs <n> --headless`. Replay a checkpoint:
`scripts/play.py --task <gym-id> --resume_path <ckpt.pt>`. (`register_pmt_tasks()` must run after
the Isaac Lab app launches; `scripts/train.py` and `scripts/play.py` already call it.)

## How to add a new task

Grounded in how existing tasks are wired (e.g. `pmt_stepping_stone`, `terrain_flat_mix`):

1. **Add the task yaml** `configs/task/<stem>.yaml` with a `defaults:` list selecting one choice
   per axis (`robot` / `terrain` / `motion` / `scene` / `sensor` / `obs` / `reward` / `network` /
   `algorithm` / `stage`), plus any task-local overrides (e.g. `experiment_name`, a
   `motion: { motion_files: ... }` block). Add new axis-choice yamls under `configs/<axis>/` if
   your combo needs them.

2. **Make sure the chosen `network`/`algorithm` resolve a compat axis.** `build_task_config()`
   resolves names against `registry.NETWORK_COMPAT` / `ALGORITHM_COMPAT`; either decorate the class
   with `@register_*("<name>", compat_name=...)` or set an explicit `compat_name:` in its config
   yaml, or the build fails loud. The compat check then derives the runner (`on_policy` /
   `distillation`).

3. **Wire an env builder** in `builder.py`: add `"<stem>": _build_<family>_env` to `_ENV_BUILDERS`.
   Reuse an existing `_build_*` helper if the env structure matches (many stems share one helper —
   e.g. all `multimotionv2_*` use the flat builder), or write a new helper that instantiates the
   family's `@configclass` from `env_cfgs/` and injects the data-driven values (mesh path, motion
   paths, decimation/sim_dt from the motion axis) then calls `env_cfg.__post_init__()`.

4. **Wire an agent cfg** in `build_agent_cfg()`: add a branch that imports and instantiates the
   matching `agent_cfgs/*` runner cfg and sets `expected_runner` to the runner the derivation will
   produce (the builder asserts they agree, fail-loud).

5. **Add the gym id** to `registry_gym._TASK_ID_MAP` (`"<stem>": "PMT-<...>-v0"`). Optional —
   without an entry the task still registers under the `PMT-<stem>-v0` fallback. No codegen is
   needed; `register_pmt_tasks()` auto-discovers every `configs/task/*.yaml`.

6. **Run it**: `scripts/train.py --task <gym-id> --num_envs <n> --headless`.

## See also

- [`env_cfgs/README.md`](env_cfgs/README.md) — the env-cfg families, network classes, and scenes.
- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — the config-axis architecture and the
  SELECT→DERIVE→VALIDATE→emit design.
