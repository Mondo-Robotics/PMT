# PMT — Perceptive Motion Tracking

**Project page:** https://acodedog.github.io/perceptive-bfm/

PMT trains humanoid **motion-tracking** RL policies on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/):
DeepMimic-style imitation, vision/terrain perception, cross-embodiment (SONIC), adversarial
imitation (ADD), and **teacher → distill → finetune** pipelines. It is a config-driven
reorganization of the original `whole_body_tracking` repo, whose 163 tasks were each a bespoke
Python class hierarchy.

> **The PMT thesis: a task is a short YAML composition of independent axes, not a subclass.**
> A builder *composes* the selected axes, *derives* the coupled fields, *validates* the
> combination against a compatibility matrix, and emits Isaac Lab `@configclass` instances.
> Adding or changing a task touches **data + builder logic**, not class trees.

---

## Table of contents

1. [Install](#1-install)
2. [Data layout & paths](#2-data-layout--paths)
3. [Repository layout](#3-repository-layout)
4. [How it works: the axis taxonomy](#4-how-it-works-the-axis-taxonomy)
5. [Quickstart: train & play](#5-quickstart-train--play)
6. [The PMT pipeline: teacher → distill → finetune](#6-the-pmt-pipeline-teacher--distill--finetune)
7. [Task catalog](#7-task-catalog)
8. [How to add a new task](#8-how-to-add-a-new-task)
9. [How to train on a new motion](#9-how-to-train-on-a-new-motion)
10. [How to add a new robot](#10-how-to-add-a-new-robot)
11. [How to add a network or algorithm](#11-how-to-add-a-network-or-algorithm)
12. [Verification](#12-verification)
13. [Further reading](#13-further-reading)

---

## 1. Install

PMT runs **inside an Isaac Lab Python environment** — it does not vendor Isaac Sim / Isaac Lab.
You need a working Isaac Lab install first (this repo is developed against the checkout at
`~/IsaacLab`).

```bash
# 1. Activate the conda env that has Isaac Lab + Isaac Sim installed.
conda activate cluster_isaaclab          # or your local Isaac Lab env name

# 2. Accept the Omniverse EULA for headless launches.
export OMNI_KIT_ACCEPT_EULA=YES

# 3. Install PMT (editable) into that env.
cd /path/to/PMT
python -m pip install -e .
```

`pip install -e .` installs the `motion_tracking_rl` RL core (see
[`pyproject.toml`](pyproject.toml)). `pmt_tasks/`, `configs/`, and `scripts/` are used in place
from the repo root, so **run all commands from the repository root**.

**Prerequisites**

| Requirement | Notes |
| --- | --- |
| Isaac Lab + Isaac Sim | An importable `isaaclab` / `isaaclab_tasks` (the train/play scripts launch the Omniverse app). |
| Python ≥ 3.10 | Matches the Isaac Lab interpreter. |
| Robot assets | The ~216 MB Unitree G1 USD/meshes are **not** copied into PMT. They are resolved via `PMT_ASSET_DIR` / `WHOLE_BODY_TRACKING_ASSET_DIR`, or fall back to a sibling `whole_body_tracking` checkout (see [`pmt_tasks/asset_config.py`](pmt_tasks/asset_config.py)). |
| Motion clips | Per-task `*.npz` clips under the data roots below. |

The **pure** test suite (builder/compat/path logic) runs without Isaac Sim; only the runtime
*gate* scripts need the live app. See [§12](#12-verification).

---

## 2. Data layout & paths

PMT never hard-codes machine paths. [`configs/paths.yaml`](configs/paths.yaml) carries a `local:`
and a `cluster:` block of **roots**; group/task YAMLs reference `${paths.*}` + a *relative*
subpath only. `PMT_PROFILE` (or `--profile`) selects the block; each root also reads its own
`PMT_*` env override.

```bash
export PMT_PROFILE=local                              # local | cluster (default: local)
export PMT_DATA_ROOT=<...>/whole_body_tracking        # meshes (*.stl), logs
export PMT_MOTION_ROOT=$PMT_DATA_ROOT/motions         # plane / single-motion clips
export PMT_DATASET_ROOT=<...>/motions                 # parent of terrain/ + sonic/
export PMT_TERRAIN_MOTION_ROOT=$PMT_DATASET_ROOT/terrain
export PMT_SONIC_ROOT=$PMT_DATASET_ROOT/sonic
```

| Variable | Points to |
| --- | --- |
| `PMT_PROFILE` | Selects the `paths.yaml` block (`local` \| `cluster`). |
| `PMT_DATA_ROOT` | Root for meshes/terrain assets and (default) logs. |
| `PMT_MOTION_ROOT` | Standalone flat motion clips (lafan_walk, back_flip, …). |
| `PMT_DATASET_ROOT` | Parent of the `terrain/` and `sonic/` clip trees. |
| `PMT_TERRAIN_MOTION_ROOT` | Terrain-anchored clip root. |
| `PMT_SONIC_ROOT` | Paired SONIC robot/human clip root. |
| `PMT_CKPT_ROOT` | Where named checkpoints (`configs/checkpoints/*`) resolve. |
| `PMT_ASSET_DIR` | Robot USD/mesh dir (falls back to a sibling `whole_body_tracking` checkout). |
| `PMT_SONIC_ONNX_DIR` | Release SONIC ONNX dir (`model_encoder.onnx` + `model_decoder.onnx`). |

The full env-var table (including per-task overrides like `PMT_BACKFLIP_MOTION`) is in
[`docs/USAGE.md`](docs/USAGE.md). Checkpoints are written under
`logs/rsl_rl/<experiment_name>/<run_name>/model_<iteration>.pt`.

---

## 3. Repository layout

```
PMT/
├── configs/                # THE SOURCE OF TRUTH (axis-group YAMLs + task compositions)
│   ├── paths.yaml          #   machine roots, profile-selected (local | cluster)
│   ├── checkpoints/        #   named prior runs (teacher/base ckpts)
│   ├── robot/ terrain/ motion/ scene/ sensor/   # SELECTED axes (one choice per task)
│   ├── obs/ reward/ network/ algorithm/ stage/  #   "
│   ├── runner/             #   DERIVED from algorithm (selectable but must agree)
│   └── task/               #   one composition file per real experiment
├── motion_tracking_rl/     # the RL core (renamed from whole_body_tracking.sonic_rsl_rl)
│   ├── registry.py         #   @register_algorithm / @register_network / @register_runner
│   ├── compat.py           #   AlgorithmSpec matrix — invalid combos fail loud at build time
│   ├── algorithms/         #   ppo, bpo, add_ppo, fpo, fpo_plus, distillation
│   ├── networks/           #   mlp, transformer, vision_transformer, diffusion, sonic, ...
│   ├── runners/ storage/ env/ utils/
│   └── bfm_zero/           #   BFM-Zero (FB-CPR-Aux) RL-core, vendored
├── pmt_tasks/              # the task layer
│   ├── builder.py          #   SELECT → DERIVE → VALIDATE → emit @configclass
│   ├── derive.py           #   the derivation rules (obs_groups, reward weights, ...)
│   ├── registry_gym.py     #   one gym.register per configs/task/*.yaml
│   ├── mdp/                #   observations / rewards / events / terminations + commands/
│   │   └── commands/       #     command stack (multi/unified/streaming + samplers + libs)
│   ├── env_cfgs/           #   per-family env cfgs: pmt/ add/ rgmt/ sonic/ pmt_token/ bfm_zero/
│   └── robots/ agent_cfgs/ isaaclab_rl/ utils/
├── scripts/                # train.py, play.py, bfm_zero/, submit/, mjlab_*, ...
├── tests/                  # pure (wbt) test suite + runtime gate scripts
├── cluster_yaml/           # md_rl cluster job templates
└── docs/                   # ARCHITECTURE.md, USAGE.md, compat_matrix.md
```

`configs/` is the source of truth — see [`configs/README.md`](configs/README.md) for the
config-author guide and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design.

---

## 4. How it works: the axis taxonomy

A task **selects** the independent axes; the builder **derives** the coupled ones.

| Axis | Selected in YAML | Notes |
|---|---|---|
| `robot` | yes | actuators, action scale, default `decimation`/`sim.dt` |
| `terrain` | yes | `flat` / `stepping_stone` / `big_map` / `terrain_flat_mix` |
| `motion` | yes | clip set + storage mode (`eager`/`streaming`) + sampler (`uniform`/`adaptive`/`bin_adaptive`) |
| `scene` | yes | `none` (default); object-interaction scenes are a future extension |
| `sensor` | yes | `none` / `height_scan` (some env families wire their own height-scan sensor in the env cfg regardless of this axis) |
| `obs` | yes (base terms) | which observation *terms* to compute |
| `reward` | yes (terms + weights) | the reward term→weight dict (the fair/no-anchor variant is chosen by selecting `deepmimic_anchor` vs `deepmimic_anchor_fair`) |
| `network` | yes | actor/critic architecture |
| `algorithm` | yes | `ppo` / `bpo` / `add_ppo` / `fpo_plus` / `distillation` (`sonic_ppo` is a PPO preset: its YAML sets `name: PPO`) |
| `stage` | yes | `scratch` / `finetune` / `distill` (lr, freeze, resume, teacher-mix) |
| `runner` | **DERIVED** | from `compat.SPECS[<resolved compat_name>].runner` (selectable, but must agree) |

**Derived / coupled (not free-selected):**
- `obs_groups` — which obs sets exist and their members — derived in
  [`pmt_tasks/derive.py`](pmt_tasks/derive.py) from (network heads, stage, algorithm); extra obs
  sets (`add_disc_obs`/`add_disc_demo`, `teacher`) are pulled in by the algorithm spec.
- `decimation` / `sim.dt` — the env builders copy these from the `motion` axis
  (`pmt_tasks/builder.py`, e.g. flat `dec=4`/`dt=0.005`, backflip `dec=10`/`dt=0.002`).
- termination thresholds — set in the per-family env cfgs (`pmt_tasks/env_cfgs/…`), e.g. backflip
  overrides the end-effector check to z-only.
- per-clip reset noise + env origins — handled in the command (`UnifiedMultiMotionCommand`) keyed
  on a per-clip `is_terrain` flag (terrain clips world-placed; flat clips offset to the flat patch).

The **build → register → CLI** contract (PMT composition is strictly upstream of Isaac Lab's
Hydra) is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 5. Quickstart: train & play

Train the standard flat multi-motion G1 task:

```bash
python scripts/train.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs <n> --headless --max_iterations <iters>
```

Checkpoints land under `logs/rsl_rl/<experiment_name>/<run_name>/model_<iteration>.pt`
(this task's `experiment_name` is `g1_multi_motion_flat`).

Play a trained checkpoint on a motion file or directory:

```bash
python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs 1 --resume_path <checkpoint.pt> \
  --motion_file <motion-file-or-dir> --headless --max_steps 300
```

`scripts/train.py` dispatches the **runner class** from the derived `runner` axis, so distillation
tasks route to `DistillationRunner` and on-policy tasks to `OnPolicyRunner` — same entrypoint.

Common flags: `--num_envs`, `--max_iterations`, `--seed`, `--profile local|cluster`,
`--resume` / `--resume_path <ckpt.pt>`, `--headless`.

---

## 5b. Pretrained models

Ready-to-use G1 policies ship under [`checkpoints/pretrained/`](checkpoints/pretrained/),
tracked with **git-lfs** (large binaries):

```bash
git lfs install          # once
git lfs pull             # fetch the .pt files after cloning
```

| File | Task / gym id | Network | Iter |
| --- | --- | --- | --- |
| `multimotionv2_flat.pt` | `PMT-G1-MultiMotionV2-Flat-v0` | MLP actor-critic | 39999 |
| `backflip_teacher.pt` | `PMT-Backflip-G1-v0` | TransformerActorCritic | 16000 |
| `cartwheel_teacher.pt` | `PMT-CartwheelBigMap-G1-v0` | TransformerActorCritic | 17400 |

```bash
# roll out the flagship flat policy
python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --resume_path checkpoints/pretrained/multimotionv2_flat.pt --num_envs 16

# or warm-start training from it
python scripts/train.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --resume --resume_path checkpoints/pretrained/multimotionv2_flat.pt --num_envs <n> --headless
```

See [`checkpoints/pretrained/README.md`](checkpoints/pretrained/README.md) for provenance and
the per-checkpoint task/data requirements.

---

## 6. The PMT pipeline: teacher → distill → finetune

PMT (Perceptive Motion Tracking) is the most involved workflow. A **blind privileged teacher**
is trained on terrain, then **distilled** into a **vision student** that replaces the teacher's
privileged terrain knowledge with proprioception + a height-scan (but not the privileged anchor),
and the student can optionally be **PPO-finetuned**. This is the stepping-stone reference pipeline.

```
 (raw + optimized clips, terrain mesh)
            │
            ▼
 ┌─────────────────────────┐   teacher ckpt (model_*.pt)
 │ 1. TEACHER  (scratch)   │ ───────────────────────────┐
 │   blind transformer     │                            │
 │   PPO on stepping-stone │                            ▼
 └─────────────────────────┘            configs/checkpoints/ss_teacher.yaml
            │                            (run_dir + checkpoint: latest|model_X.pt)
            │                                            │
            ▼                                            │ ${checkpoints.ss_teacher}
 ┌─────────────────────────┐                            │
 │ 2. DISTILL  (distill)   │ ◀──────────────────────────┘
 │   vision STUDENT learns │   teacher reads OPTIMIZED clips ("motion"),
 │   from frozen teacher   │   student reads synced RAW clips ("student_motion"),
 │   via height-scan       │   teacher-mix anneals 1.0 → 0.0
 └─────────────────────────┘
            │  student ckpt
            ▼
 ┌─────────────────────────┐
 │ 3. FINETUNE (finetune)  │   PPO-finetune the vision policy (warmup-freeze,
 │   PPO refine the policy │   reset action-std on load)
 └─────────────────────────┘
```

**Raw vs optimized clips.** Terrain pipelines use **two** versions of the same source clip:
- **optimized** — terrain-IK-adapted so feet land on stones/stairs; the *teacher* tracks these.
- **raw** — the un-adapted motion; the *student* tracks these (synced to the teacher by basename).

The paired-command env (`configs/motion/stepping_stone_paired.yaml`) loads both: optimized →
teacher command `motion`, raw → student command `student_motion`. (Optimized clips are produced
upstream by the terrain motion-adapter in the `whole_body_tracking` repo; PMT consumes the
resulting `*.npz`.)

### Stage 1 — train the teacher (`stage: scratch`)

```bash
python scripts/train.py --task PMT-SteppingStone-G1-v0 \
  --num_envs <n> --headless --max_iterations <iters>
```

- Task: [`configs/task/pmt_stepping_stone.yaml`](configs/task/pmt_stepping_stone.yaml)
  (`network: transformer`, `obs: transformer_hist`, `sensor: none`, `algorithm: ppo`).
- Output: `logs/rsl_rl/pmt_stepping_stone/<run>/model_<iter>.pt`.

### Register the teacher checkpoint

Point [`configs/checkpoints/ss_teacher.yaml`](configs/checkpoints/ss_teacher.yaml) at the run you
just trained:

```yaml
run_dir: ${paths.CKPT_ROOT}/pmt_stepping_stone/<timestamped_run>
checkpoint: model_9999.pt        # explicit file -> joined onto run_dir
                                 # (in this wave `latest` resolves to run_dir as a string;
                                 #  fail-loud glob-latest is a planned deliverable)
```

### Stage 2 — distill into a vision student (`stage: distill`)

```bash
python scripts/train.py --task PMT-Distill-SteppingStone-LatentAnchor-G1-v0 \
  --num_envs <n> --headless --max_iterations <iters>
```

- Task: [`configs/task/distill_stepping_stone_latent_anchor.yaml`](configs/task/distill_stepping_stone_latent_anchor.yaml)
  (`network: vision_student_latent_anchor`, `obs: vision_student`, `sensor: height_scan`,
  `algorithm: distillation` → `DistillationRunner`).
- The task references the teacher via `network.teacher_ckpt: ${checkpoints.ss_teacher}`; the
  builder injects it into the frozen distillation target. With `teacher_ckpt: null` the teacher is
  random-init — the **runner path** runs end-to-end but the distillation loss is meaningless (this
  is the CI gate; supply a real teacher for a real run).
- A simpler `StudentTeacher` MLP-pair variant exists for path-testing:
  `PMT-Distill-SteppingStone-G1-v0` (also `sensor: height_scan`; it differs from the latent-anchor
  task by its network — the MLP `student_teacher` pair vs `vision_student_latent_anchor`).

### Stage 3 — PPO-finetune the vision policy (`stage: finetune`)

The finetune scaffold is
[`configs/task/ppofinetune_vision_teacher_stepping_stone_latent_anchor.yaml`](configs/task/ppofinetune_vision_teacher_stepping_stone_latent_anchor.yaml)
(`stage: finetune` → `lr=5e-4`, `warmup_freeze_iters=200`, `reset_action_std_on_load=true`). It is
**not yet a one-command train target**: it requires a trained vision-teacher checkpoint and
finetune env/agent wiring before `gym.make` can build it (the builder raises a clear message
listing the wired stems). Wire those, then launch it like any other PPO task.

> The three `stage/*.yaml` files are the single source of the lr / freeze / resume / teacher-mix
> deltas; the rest of each task differs only by the independent axis it selects (sensor, obs,
> network, algorithm).

---

## 7. Task catalog

Most direct-train tasks are a `configs/task/*.yaml` composition registered by
`pmt_tasks.registry_gym.register_pmt_tasks` (BFM-Zero is the exception — it is registered
separately in the same module with its own FB-CPR-Aux runner, and has no task YAML). Train with
`scripts/train.py --task <id> …`; play with `scripts/play.py --task <id> --resume_path <ckpt.pt>
--motion_file <npz-or-dir> …`. See [`docs/USAGE.md`](docs/USAGE.md) for full per-task command forms.

**Motion-tracking PPO**
- `PMT-G1-MultiMotionV2-Flat-v0`, `-Uniform-Flat-v0`, `-Adaptive-Flat-v0`, `-Streaming-Flat-v0`
  (sampler/storage variants), `-100style-Flat-v0`, `-Streaming-100style-Flat-v0`
- `PMT-SteppingStone-G1-v0` (teacher), `PMT-Backflip-G1-v0`, `PMT-TerrainFlatMix-G1-v0`,
  `PMT-WalkDanceBigMap-G1-v0`, `PMT-CartwheelBigMap-G1-v0`

**Distillation / perceptive**
- `PMT-Distill-SteppingStone-G1-v0` (`StudentTeacher` MLP pair, path test)
- `PMT-Distill-SteppingStone-LatentAnchor-G1-v0` (vision student, latent anchor)
- `PMT-PerceptiveMotionTokenTracker-G1-v0`

**Other algorithm families**
- `PMT-G1-BPO-MultiMotionV2-Flat-v0` (bounded-ratio PPO)
- `PMT-ADD-MultiMotionV2-Flat-v0` (adversarial)
- `PMT-G1-FPOPlus-SingleClip-Flat-v0` (flow-policy FPO++, diffusion net)
- `PMT-SONIC-G1-MultiMotionV2-Flat-v0` (cross-embodiment; supports release ONNX, see USAGE)
- `RGMT-G1-v0` (paper-faithful transformer actor-critic, no vision)
- `BFM-Zero-Flat-MultiMotionV2-G1-v0` via `scripts/bfm_zero/train.py` (separate FB-CPR-Aux runner)

`vision_ablation_base` and `ppofinetune_vision_teacher_stepping_stone_latent_anchor` are reserved
composition/finetune targets, not one-command train tasks (they need extra assets / wiring first).

---

## 8. How to add a new task

Write **one** `configs/task/<stem>.yaml`: a `defaults:` list (one choice per axis) plus
task-local overrides. Tasks that differ only on *derived* fields share a YAML — you only need a
new task YAML when an *independent* axis choice differs.

```yaml
# configs/task/my_task.yaml
defaults:
  - robot: g1
  - terrain: flat
  - motion: multi
  - scene: none
  - sensor: none
  - obs: proprio
  - reward: deepmimic_anchor
  - network: mlp
  - algorithm: ppo          # runner derived -> on_policy
  - stage: scratch
experiment_name: my_task

# task-local overrides merge on top of the composed axes:
motion:
  motion_files: ${paths.MOTION_ROOT}/my_clips
```

Then map the stem to an env builder in `_ENV_BUILDERS` in
[`pmt_tasks/builder.py`](pmt_tasks/builder.py): `build_env_cfg` looks the stem up there and raises
if it is absent. If the task reuses an existing family (e.g. flat MLP, transformer terrain), this
is a **one-line** entry pointing the stem at that family's existing builder (see how the
`multimotionv2_*` stems all map to `_build_multimotion_flat_env`); only a genuinely new
scene/sensor wiring needs a new builder function. Optionally map the stem to a friendly gym id in
[`pmt_tasks/registry_gym.py`](pmt_tasks/registry_gym.py) `_TASK_ID_MAP`, or rely on the
`PMT-<stem>-v0` fallback.

The pure CI ([`tests/test_all_tasks_resolve.py`](tests/test_all_tasks_resolve.py)) automatically
loads every new YAML and asserts it composes + derives + validates.

---

## 9. How to train on a new motion

The motion you need depends on the task family:

**A. Plane / flat tasks (most non-PMT tasks) — just point at the clips.**
A flat task needs only flat-plane clips and the flat terrain. Drop your `*.npz` clips in a
directory and override `motion.motion_files`:

```yaml
# configs/task/my_flat_task.yaml  (terrain: flat, motion: multi or single_clip)
motion:
  motion_files: ${paths.MOTION_ROOT}/my_flat_clips   # a dir of NPZs, or a single .npz
```

For a single clip, select `motion: single_clip`; for many clips, `motion: multi` (eager) with a
sampler/storage choice (`sampler: uniform|adaptive|bin_adaptive`, `storage_mode:
eager|streaming`). Streaming bounds memory for large clip sets (e.g. the 100-style split).

**B. Terrain / PMT tasks — you need raw *and* optimized clips.**
Terrain tracking requires both versions of each clip ([§6](#6-the-pmt-pipeline-teacher--distill--finetune)):
- **optimized** clips (terrain-IK adapted, fitted to the terrain mesh) for the teacher,
- **raw** clips for the vision student,

plus the matching terrain mesh (`*.stl`) referenced by the `terrain` axis. Optimized clips are
produced upstream by the `whole_body_tracking` terrain motion-adapter; PMT consumes the resulting
`*.npz`. Wire them via a paired motion YAML (model it on
[`configs/motion/stepping_stone_paired.yaml`](configs/motion/stepping_stone_paired.yaml)):

```yaml
motion_files:     ${paths.TERRAIN_MOTION_ROOT}/<dataset>/<clip>/optimized   # teacher
raw_motion_files: ${paths.TERRAIN_MOTION_ROOT}/<dataset>/<clip>/raw         # student
paired: true
```

Set the per-motion control rate (`decimation` / `sim_dt`) in the motion YAML — the env builder
copies it onto the env cfg (e.g. backflip uses `dec=10`/`dt=0.002`). Termination thresholds are
**not** on the motion YAML; they live in the per-family env cfgs (`pmt_tasks/env_cfgs/…`), e.g.
backflip re-points the end-effector check to a z-only variant. Clip `*.npz` are stored in
Isaac-Lab BFS body/joint order — see [`scripts/pmt_npz_to_mjlab.py`](scripts/pmt_npz_to_mjlab.py)
for the exact array contract.

---

## 10. How to add a new robot

The `robot` axis currently ships only G1, but the pattern is:

1. **Asset config** — add a robot module under [`pmt_tasks/robots/`](pmt_tasks/robots/) (model it
   on [`g1.py`](pmt_tasks/robots/g1.py)) exposing an Isaac Lab `ArticulationCfg` (the USD path,
   actuator stiffness/damping/armature) and an action-scale constant. Robot USD/meshes are
   resolved through [`pmt_tasks/asset_config.py`](pmt_tasks/asset_config.py) (`PMT_ASSET_DIR`).
2. **Axis YAML** — add [`configs/robot/<name>.yaml`](configs/robot/g1.yaml) with `name:`,
   default `decimation`, `sim_dt`, and any robot-level fields.
3. **Wire it into the env cfg** — the env families import the robot's `ArticulationCfg`
   (e.g. `from pmt_tasks.robots.g1 import G1_CYLINDER_CFG, G1_ACTION_SCALE` in
   [`pmt_tasks/env_cfgs/multi_motion_flat.py`](pmt_tasks/env_cfgs/multi_motion_flat.py)). Add the
   analogous import/branch for your robot so the selected `robot` axis maps to its cfg.
4. **Motions for that embodiment** — motion clips are robot-specific (joint order / body set), so
   provide clips retargeted to the new robot.

Then select `robot: <name>` in any task YAML. (Full cross-embodiment training, including SMPL/human
encoders, is the SONIC path — see [`docs/USAGE.md`](docs/USAGE.md).)

---

## 11. How to add a network or algorithm

This is a small, *explicit* number of touch points (the registry removes the *runner-dispatch*
edits, not the *schema* edits). Full worked guide in [`configs/README.md`](configs/README.md).

**Add a network** (≈5 touch points):
1. Decorate the `nn.Module` in `motion_tracking_rl/networks/`:
   `@register_network("MyNetClass", compat_name="my_net")` — `name` must equal the ckpt
   `class_name` for checkpoint compatibility; `compat_name` is the axis name.
2. Add the module to the network import list in `registry.autoload()` so the decorator actually
   fires (autoload imports a fixed list; an unimported module is never registered).
3. `configs/network/my_net.yaml` with `name: MyNetClass` + hyperparams.
4. Extend the `policy` `@configclass` union in
   `pmt_tasks/isaaclab_rl/rsl_rl/rl_cfg.py` to accept its cfg type.
5. Add `"my_net"` to the `compatible_networks` of every algorithm in `compat.py` that should accept it.

**Add an algorithm** (≈3 touch points):
1. Decorate the class in `motion_tracking_rl/algorithms/` and add it to `registry.autoload()`.
2. `configs/algorithm/my_alg.yaml` with `name:` + feature flags + hyperparams.
3. A new `compat.SPECS` entry (`AlgorithmSpec`) declaring its runner, compatible networks, feature
   support (rnd/symmetry/recurrent), paired-command requirement, and required obs sets.

`registry.assert_compat_consistency()` (run by the builder and CI) fails loud if the registry
tables drift from `compat.SPECS`. The generated matrix is [`docs/compat_matrix.md`](docs/compat_matrix.md).

---

## 12. Verification

Run the **pure** suite (no Isaac Sim needed) — task registration, compat matrix, builder coverage,
path resolution, and derivation:

```bash
conda run -n cluster_isaaclab python -m pytest \
  tests/test_compat_matrix.py \
  tests/test_builder_slice.py \
  tests/test_compat_name_unification.py \
  tests/test_all_tasks_resolve.py \
  tests/test_paths.py \
  -q
```

Or run the whole pure suite at once:

```bash
conda run -n cluster_isaaclab python -m pytest tests/ -q
```

Runtime **gate** scripts under `tests/` (e.g. `phase*_gate*.py`) create Isaac Lab app/runtime
state and require an Isaac-capable environment.

---

## 13. Further reading

| Doc | Contents |
| --- | --- |
| [`docs/USAGE.md`](docs/USAGE.md) | Full env-var table, per-task command forms, SONIC ONNX, BFM-Zero. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Axis taxonomy, derivation table, compat matrix, layering contract. |
| [`configs/README.md`](configs/README.md) | Config-author guide: add task / network / algorithm, paths & profiles. |
| [`configs/checkpoints/README.md`](configs/checkpoints/README.md) | Named-checkpoint schema (teacher/base ckpts). |
| [`cluster_yaml/README.md`](cluster_yaml/README.md) | md_rl cluster job templates. |
| [`docs/compat_matrix.md`](docs/compat_matrix.md) | Generated algorithm × network compatibility matrix. |

---

## 14. License

PMT's own original code is released under the **BSD 3-Clause License** (see
[`LICENSE`](LICENSE)). PMT also incorporates, derives from, or vendors code from
several upstream projects, each governed by its own license — see
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES) for the full inventory:

| Component | License | Where |
| --- | --- | --- |
| [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) (ETH Zurich + NVIDIA) | BSD-3-Clause | `motion_tracking_rl/` RL core |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause | `pmt_tasks/isaaclab_rl/` |
| [`whole_body_tracking`](https://github.com/HybridRobotics/whole_body_tracking) | MIT | task structure, `pmt_tasks/utils/terrain.py` |
| [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) (Meta Platforms) | **CC BY-NC 4.0** | `motion_tracking_rl/bfm_zero/_vendor/`, `networks/ode_solver.py` |

> ⚠️ **Non-commercial restriction.** The bundled BFM-Zero (FB-CPR-Aux) code is
> licensed under **CC BY-NC 4.0 (NonCommercial)**. Because it ships in this
> repository, the repository as distributed is, as a combined work, usable for
> **non-commercial / research purposes only**. For commercial use you must
> remove `motion_tracking_rl/bfm_zero/_vendor/` and
> `motion_tracking_rl/networks/ode_solver.py` and avoid the BFM-Zero code paths.
> All upstream copyright and attribution notices must be retained on redistribution.
