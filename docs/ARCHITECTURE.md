# PMT architecture

Concise reference for the PMT design. For the migration history, rationale, and full
residual-risk list see `../PMT_MIGRATION_PLAN.md` (this is the canonical record; §-refs
below point into it).

## The core idea

A task is a **composition of independent axes**, not a Python subclass. The original
`whole_body_tracking` repo materialized each of 163 tasks as a hand-written
`(env_cfg_class, runner_cfg_class)` pair. PMT replaces the class trees with:

- **`configs/`** — axis group YAMLs + one task YAML per experiment (the source of truth).
- **`pmt_tasks/builder.py`** — SELECT → DERIVE → VALIDATE → emit `@configclass`.
- **`motion_tracking_rl/registry.py` + `compat.py`** — a decorator registry that replaces
  `eval(class_name)`, plus a compatibility matrix that rejects invalid combos at build time.

## Axis taxonomy

| Axis | Selects | Source / where it lands |
|---|---|---|
| `robot` | actuators, action scale, decimation/sim_dt | `pmt_tasks/robots/`, `configs/robot/` |
| `terrain` | terrain importer + generator | `configs/terrain/`, terrain utils in `mdp/` |
| `motion` | clip set, storage mode, sampler | `configs/motion/`, `mdp/*motion*` |
| `scene` | rigid objects (none / omniretarget cubes) | `configs/scene/`, `mdp/events.py` |
| `sensor` | scene cfg + obs shape + net input dim (none/height_scan) | `configs/sensor/` |
| `obs` | which observation *terms* to compute | `configs/obs/`, `mdp/observations.py` |
| `reward` | which reward *terms* (weights derived) | `configs/reward/`, `mdp/rewards.py` |
| `network` | actor/critic architecture | `configs/network/`, `motion_tracking_rl/networks/` |
| `algorithm` | ppo/bpo/add_ppo/fpo/fpo_plus/distillation | `configs/algorithm/`, `algorithms/` |
| `stage` | scratch/finetune/distill (lr, freeze, resume) | `configs/stage/` |
| `runner` | **DERIVED** from algorithm (`SPECS[alg].runner`) | `configs/runner/`, `runners/` |

## Derived fields (`pmt_tasks/derive.py`)

YAML provides base obs/reward **terms**; the builder derives which sets they go into and their
coupled magnitudes (plan §3a). Every old `__post_init__` side-effect is an explicit derivation
rule, not an invisible mutation.

| Derived field | Derived from | Function |
|---|---|---|
| `obs_groups` | network heads + stage + algorithm + rnd | `derive_obs_groups_with_features` |
| `reward_weights` | whether the anchor obs term is in actor obs | `derive_reward_weights` |
| `runner` | algorithm spec | `compat.SPECS[alg].runner` |
| `decimation` / `sim.dt` / term thresholds | the motion | emitted `@configclass.__post_init__` |
| per-clip reset noise + env origins | motion × terrain | `UnifiedMotionCommandV2` (per-clip is_terrain flag) |
| extra obs sets (`add_disc_*`, `teacher`) | algorithm | `SPECS[alg].required_obs_sets` |

## Compatibility matrix (`motion_tracking_rl/compat.py`)

`AlgorithmSpec` declares, per algorithm: `runner`, `compatible_networks`,
`requires_paired_command`, `supports_rnd/symmetry/recurrent`, `required_obs_sets`. The builder
calls `compat.validate(...)` and **raises a clear error at build time** for invalid combos (e.g.
`fpo + transformer`, `fpo + rnd`, `distillation` without a paired command, `add_ppo` without
discriminator obs sets). The full matrix is generated from SPECS into
[`compat_matrix.md`](compat_matrix.md) by `tests/gen_compat_matrix_doc.py`.

`registry.assert_compat_consistency()` guards table drift (every registered algorithm compat name
is a SPECS key; every registered network is referenced by some spec). The only network with a real
backing class but no `@register_network` is `diffusion` (`KNOWN_PENDING = {diffusion}` — no FPO
task is wired). Pinned by `tests/test_all_tasks_resolve.py::test_known_pending_is_exactly_diffusion`.

## Two-class command design (plan §9b)

Motion commands collapse from ~9 near-duplicate classes to **two** + orthogonal wrappers:

- `MultiMotionCommandV2` — eager core. **Base.**
- `UnifiedMotionCommandV2(MultiMotionCommandV2)` — the **flexible** class: per-clip
  `{origin offset + noise}` injection keyed on a per-clip `is_terrain` flag (terrain clips are
  world-placed on their mesh; flat clips are offset to the `(90,0,0)` flat patch), with
  flag-selected storage (`eager|streaming|packed`) and sampler (`uniform|adaptive|bin_adaptive`).
  This subsumes the old `Streaming`/`Grouped`/`Comparison` storage/sampler variants.

`GhostMotionCommand` / `AdaptiveMotionCommand` stay separate (orthogonal wrappers). The legacy
`GroupedMultiMotionCommandV2` and the single-file `MotionCommand` are **intentionally retained**
(not deleted) but are legacy — TerrainFlatMix uses the unified class. Not yet subsumed: per-group
sampler curriculum + `flat_ratio` coverage (add a group-aware sampler *flag*, not a third class —
plan §9c).

## Layering contract (plan §10)

PMT's composition is strictly upstream of and disjoint from Isaac Lab's Hydra:

1. **Build time (PMT):** compose axis YAMLs → `derive.py` → `compat.validate` → emit a concrete
   `TrackingEnvCfg` subclass + `RslRlOnPolicyRunnerCfg` instance (all derivation in
   `__post_init__`/builder, before `gym.make`).
2. **Registration:** one `gym.register` per `configs/task/*.yaml` with **fresh-per-call** factory
   closures as entry points (never a shared singleton — `from_dict` mutates in place).
3. **CLI time (Isaac Lab, unchanged):** `@hydra_task_config` resolves the closures, stores one
   node from the instance's `to_dict()`, applies `env.*=`/`agent.*=`/`--multirun` overrides.

## Legacy / optional surfaces (retained, not dead)

- `motion_tracking_rl/modules/` — import shim aliasing old `modules.*` paths to `networks.*` for
  checkpoint/qualname compatibility (plan §4). Keep.
- `GroupedMultiMotionCommandV2`, single-file `MotionCommand` — legacy commands, retained.
- `DiffusionActorCritic` — the network FPO requires; kept even though no FPO task is wired.
- `fpo_plus` — kept (owner decision). `diffusion_distillation` was dropped per plan §8.2.
- BFM-Zero is a separate integration concern (plan §8.1), not a PMT network.
