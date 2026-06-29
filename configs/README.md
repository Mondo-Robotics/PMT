# `configs/` — the PMT source of truth

Every PMT task is a composition of **axis group YAMLs** plus a small **task YAML** that selects
one choice per axis and adds overrides. The builder (`pmt_tasks/builder.py`) composes these,
derives the coupled fields, validates the combination, and emits Isaac Lab `@configclass`
instances. This is the config-author guide.

```
configs/
├── paths.yaml + paths/      # machine roots, profile-selected (local|cluster)
├── checkpoints/             # named prior runs (teacher/base ckpts)
├── robot/ terrain/ motion/ scene/ sensor/   # SELECTED axes (one choice per task)
├── obs/ reward/ network/ algorithm/ stage/   #   "
├── runner/                  # DERIVED from algorithm (but selectable; must agree)
└── task/                    # one composition file per real experiment
```

## How to add a new task

Write **one** `task/*.yaml`: a `defaults:` list (one choice per axis) plus overrides.
Worked example:

```yaml
# configs/task/pmt_stepping_stone.yaml  (was SteppingStone-G1-v0)
defaults:
  - robot: g1
  - terrain: stepping_stone
  - motion: multi
  - scene: none
  - sensor: none
  - obs: transformer_hist
  - reward: deepmimic_anchor
  - network: transformer
  - algorithm: ppo          # runner derived -> on_policy
  - stage: scratch
experiment_name: pmt_stepping_stone

# task-local overrides merge on top of the composed axes:
motion:
  motion_files: ${paths.DATASET_ROOT}/terrain/.../optimized
```

Then map the stem to a gym id in `pmt_tasks/registry_gym.py::_TASK_ID_MAP` (or rely on the
`PMT-<stem>-v0` fallback). The pure CI (`tests/test_all_tasks_resolve.py`) automatically loads
the new yaml and asserts it composes + derives + validates.

**Tasks that differ only on *derived* fields share a task YAML** — the builder derives the rest
(plan §9). You only need a new task YAML when an *independent* axis choice differs.

### Referencing a named checkpoint

```yaml
network:
  teacher_ckpt: ${checkpoints.ss_teacher}   # resolves to a run_dir/file at build time
```

The named run lives in `configs/checkpoints/ss_teacher.yaml` (`run_dir` + `checkpoint:
latest|model_X.pt`). Resolution is fail-loud at build time. See `configs/checkpoints/README.md`.

## How to add a network / algorithm

This is **~3–4 touch points**, not one (plan §4/§9 — honest count). The registry removes the
*runner-dispatch* edits, not the *schema* edits.

**Add a network:**
1. The decorated `nn.Module` in `motion_tracking_rl/networks/`:
   `@register_network("MyNetClass", compat_name="my_net")` (the `name` must equal the old/ckpt
   `class_name` for checkpoint compatibility; `compat_name` is the axis name).
2. A `configs/network/my_net.yaml` with `name: MyNetClass` + hyperparams.
3. A `@configclass` schema entry — `policy` in `pmt_tasks/isaaclab_rl/rsl_rl/rl_cfg.py` (line ~597,
   `policy: RslRlPpoActorCriticCfg | ...`) must accept its cfg type.
4. A `compat.py` line — add `"my_net"` to the `compatible_networks` of every algorithm that
   should accept it.

**Add an algorithm:**
1. The decorated class in `motion_tracking_rl/algorithms/`:
   `@register_algorithm("MyAlg", compat_name="my_alg")`; add it to `registry.autoload()`.
2. A `configs/algorithm/my_alg.yaml` with `name: MyAlg` + feature flags + hyperparams.
3. A **new `compat.SPECS` entry** (`AlgorithmSpec`) declaring its runner, compatible networks,
   feature support (rnd/symmetry/recurrent), paired-command requirement, and required obs sets.

`registry.assert_compat_consistency()` (run by the builder and CI) fails loud if the registry
compat tables drift from `compat.SPECS`.

## How to run ablation configs

Use a direct-train gym id for runtime launches. The perceptive-motion token tracker is the
launchable PMT vision-style target:

```bash
python scripts/train.py --task PMT-PerceptiveMotionTokenTracker-G1-v0 \
  --headless --num_envs <n> --max_iterations <iters>
```

The pure tests verify the mechanism without a runtime:
`tests/test_perceptive_motion_family.py::test_multirun_dotlist_flips_fields` composes
`vision_ablation_base` with `overrides=[...]` and asserts the field flips. That config remains a
composition/ablation demo and test target; it is not a single-command train task until the
vision-teacher checkpoint, height-scan data, and finetune wiring are present.

## `paths.yaml` — profiles + named checkpoints

`paths.yaml` carries a `local:` and a `cluster:` block (roots: `DATA_ROOT`, `MOTION_ROOT`,
`TERRAIN_ROOT`, `CKPT_ROOT`, `DATASET_ROOT`). Select with `--profile local|cluster` (or
`$PMT_PROFILE`, default `local`). Group YAMLs reference `${paths.*}` + **relative** subpaths only
— never absolute machine paths. `PMT_PROFILE` selects the profile block; each root in that block
also reads its own `PMT_*` env override (e.g. `PMT_DATA_ROOT`, `PMT_MOTION_ROOT`, `PMT_DATASET_ROOT`,
`PMT_SONIC_ROOT`, `PMT_CKPT_ROOT`) so individual roots can be repointed without editing the YAML.
See [`docs/USAGE.md`](../docs/USAGE.md) for the full env-var table.

`configs/checkpoints/*.yaml` are **named prior runs** (per-task data a root profile can't express:
timestamped run dir + glob-latest). Tasks reference them via `${checkpoints.<name>}`.

## Which axes are derived vs selected

Selected (one choice per task): `robot terrain motion scene sensor obs reward network algorithm
stage`. Derived by `pmt_tasks/derive.py` (do **not** put these in YAML):

- `obs_groups` — `derive_obs_groups_with_features(network, stage, algorithm_spec, obs, rnd)`.
- `reward_weights` — `derive_reward_weights(reward, obs)` (obs-coupled, e.g. the no-anchor "fair"
  weight-set).
- `runner` — from `compat.SPECS[algorithm].runner` (the `runner/` axis is selectable but must
  agree with the derived value, else the builder raises).
- per-motion `decimation`/`sim.dt`, per-clip reset noise + env origins — in the emitted
  `@configclass.__post_init__` (build time, before `gym.make`).

See `docs/ARCHITECTURE.md` for the full taxonomy + derivation table.
