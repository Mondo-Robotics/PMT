# PMT Usage Guide

This guide covers setup, data paths, launchable tasks, pretrained SONIC rollout, BFM-Zero, and release verification. Commands assume they are run from the repository root and use placeholders instead of machine-specific paths.

## Setup

PMT runs inside an Isaac Lab Python environment. Use the environment name that matches your local Isaac Lab installation:

```bash
conda activate <env>
export OMNI_KIT_ACCEPT_EULA=YES
python -m pip install -e .
```

Set data paths with environment variables when your data does not live in the default layout. `PMT_PROFILE` selects which block of `configs/paths.yaml` is used (`local` or `cluster`); every root in that block reads its own `PMT_*` env var and falls back to a generic default derived from `$HOME` when the var is unset. The defaults below are the **local** profile defaults (`configs/paths.yaml`); the `cluster` profile uses the same env-var names with `$HOME/pmt_cluster_data/...` fallbacks.

| Variable | Points to | Local default (when unset) |
| --- | --- | --- |
| `PMT_PROFILE` | Selects the `configs/paths.yaml` block | `local` (valid: `local`, `cluster`) |
| `PMT_DATA_ROOT` | Root for meshes/terrain assets and (default) logs | `$HOME/whole_body_tracking` |
| `PMT_MOTION_ROOT` | Standalone motion clips (lafan_walk, debug, back_flip, ...) | `$PMT_DATA_ROOT/motions` |
| `PMT_DATASET_ROOT` | Parent of the terrain/ and sonic/ clip trees | `$HOME/whole_body_tracking_motions/motions` |
| `PMT_TERRAIN_MOTION_ROOT` | Terrain-anchored clip root | `$PMT_DATASET_ROOT/terrain` |
| `PMT_SONIC_ROOT` | Paired SONIC robot/human clip root | `$PMT_DATASET_ROOT/sonic` |
| `PMT_CKPT_ROOT` | Where named checkpoints (`configs/checkpoints/*`) resolve | `$PMT_DATA_ROOT/logs/rsl_rl` |
| `PMT_MULTIMOTION_FLAT_MOTION` | Full clip dir for the MultiMotionV2-Flat target | `$PMT_MOTION_ROOT/lafan_walk` |
| `PMT_BACKFLIP_MOTION` | Backflip clip(s) | `$PMT_MOTION_ROOT/back_flip/flip_360_001__A304_wbt.npz` |
| `PMT_SONIC_ONNX_DIR` | Release SONIC ONNX dir (`model_encoder.onnx` + `model_decoder.onnx`) | `<repo>/third_party/sonic_release` |
| `BFM_ZERO_REPO` | Deprecated/no-op. The BFM-Zero (FB-CPR-Aux) code is now vendored in PMT (`motion_tracking_rl/bfm_zero/_vendor`); this env var is ignored. | — |
| `PMT_REPO_ROOT` | PMT repo root used to resolve repo-relative asset fallbacks | auto-detected from the package location |

> Note: `TERRAIN_ROOT` is derived as `DATA_ROOT` inside `configs/paths.yaml` (no separate
> env var). Per-task BFM-Zero motion overrides (e.g. `PMT_BFM_ZERO_FLAT_MOTION_PATHS`) are
> covered in the BFM-Zero section below.

Example local layout (only set the vars whose data is not already at the default):

```bash
export PMT_PROFILE=local
export PMT_DATA_ROOT=<pmt-data-root>                  # meshes, logs
export PMT_MOTION_ROOT=$PMT_DATA_ROOT/motions         # plane/single-motion clips
export PMT_DATASET_ROOT=<dataset-root>/motions        # parent of terrain/ + sonic/
export PMT_TERRAIN_MOTION_ROOT=$PMT_DATASET_ROOT/terrain
export PMT_SONIC_ROOT=$PMT_DATASET_ROOT/sonic
```

## Quickstart

Train the standard flat multi-motion G1 task:

```bash
python scripts/train.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs <n> --headless --max_iterations <iters>
```

RSL-RL checkpoints are written under:

```text
logs/rsl_rl/<experiment_name>/<run_name>/model_<iteration>.pt
```

For this task, the experiment name is `g1_multi_motion_flat`.

Play a trained checkpoint on a motion file or motion directory:

```bash
python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs 1 --resume_path <checkpoint.pt> \
  --motion_file <motion-file-or-dir> --headless --max_steps 300
```

## Task Catalog

All task YAMLs under `configs/task/` are registered to gym ids. The tasks below have direct builders and can be launched with the listed command forms. Replace `<n>`, `<iters>`, `<checkpoint.pt>`, and `<motion-file-or-dir>` with values for your run.

### Motion-Tracking PPO

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `PMT-G1-MultiMotionV2-Flat-v0` | `ppo` / `mlp` | Standard flat multi-motion G1 tracking task. | `python scripts/train.py --task PMT-G1-MultiMotionV2-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-G1-MultiMotionV2-Uniform-Flat-v0` | `ppo` / `mlp` | Uniform-sampling flat multi-motion variant. | `python scripts/train.py --task PMT-G1-MultiMotionV2-Uniform-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-G1-MultiMotionV2-Uniform-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-G1-MultiMotionV2-Adaptive-Flat-v0` | `ppo` / `mlp` | Adaptive-sampling flat multi-motion variant. | `python scripts/train.py --task PMT-G1-MultiMotionV2-Adaptive-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-G1-MultiMotionV2-Adaptive-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-G1-MultiMotionV2-Streaming-Flat-v0` | `ppo` / `mlp` | Streaming flat multi-motion variant. | `python scripts/train.py --task PMT-G1-MultiMotionV2-Streaming-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-G1-MultiMotionV2-Streaming-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-G1-MultiMotionV2-100style-Flat-v0` | `ppo` / `mlp` | Flat multi-motion variant over the 100-style split. | `python scripts/train.py --task PMT-G1-MultiMotionV2-100style-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-G1-MultiMotionV2-100style-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-G1-MultiMotionV2-Streaming-100style-Flat-v0` | `ppo` / `mlp` | Streaming flat multi-motion variant over the 100-style split. | `python scripts/train.py --task PMT-G1-MultiMotionV2-Streaming-100style-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-G1-MultiMotionV2-Streaming-100style-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-SteppingStone-G1-v0` | `ppo` / `transformer` | Transformer stepping-stone locomotion and tracking task. | `python scripts/train.py --task PMT-SteppingStone-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-SteppingStone-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-Backflip-G1-v0` | `ppo` / `transformer` | Transformer backflip tracking task. | `python scripts/train.py --task PMT-Backflip-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-Backflip-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-TerrainFlatMix-G1-v0` | `ppo` / `transformer` | Transformer mixed flat-terrain and terrain task. | `python scripts/train.py --task PMT-TerrainFlatMix-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-TerrainFlatMix-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-WalkDanceBigMap-G1-v0` | `ppo` / `transformer` | Big-map walk and dance tracking task. | `python scripts/train.py --task PMT-WalkDanceBigMap-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-WalkDanceBigMap-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-CartwheelBigMap-G1-v0` | `ppo` / `transformer` | Big-map cartwheel tracking task. | `python scripts/train.py --task PMT-CartwheelBigMap-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-CartwheelBigMap-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |

### BPO

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `PMT-G1-BPO-MultiMotionV2-Flat-v0` | `bpo` / `mlp` | BPO flat multi-motion G1 tracking task. | `python scripts/train.py --task PMT-G1-BPO-MultiMotionV2-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-G1-BPO-MultiMotionV2-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |

### ADD Adversarial

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `PMT-ADD-MultiMotionV2-Flat-v0` | `add_ppo` / `mlp` | ADD adversarial flat multi-motion task. | `python scripts/train.py --task PMT-ADD-MultiMotionV2-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-ADD-MultiMotionV2-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |

### SONIC Cross-Embodiment

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `PMT-SONIC-G1-MultiMotionV2-Flat-v0` | `sonic_ppo` / `sonic` | SONIC cross-embodiment flat multi-motion task. | `python scripts/train.py --task PMT-SONIC-G1-MultiMotionV2-Flat-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-SONIC-G1-MultiMotionV2-Flat-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |

### Distillation

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `PMT-Distill-SteppingStone-G1-v0` | `distillation` / `student_teacher` | Student-teacher distillation for the stepping-stone task. | `python scripts/train.py --task PMT-Distill-SteppingStone-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-Distill-SteppingStone-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |
| `PMT-Distill-SteppingStone-LatentAnchor-G1-v0` | `distillation` / `vision_student_latent_anchor` | Vision-student distillation with latent-anchor supervision. | `python scripts/train.py --task PMT-Distill-SteppingStone-LatentAnchor-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-Distill-SteppingStone-LatentAnchor-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |

### Perceptive Motion

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `PMT-PerceptiveMotionTokenTracker-G1-v0` | `ppo` / `perceptive_motion_token_tracker` | Perceptive motion-token tracker task with vision-aware policy inputs. | `python scripts/train.py --task PMT-PerceptiveMotionTokenTracker-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task PMT-PerceptiveMotionTokenTracker-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |

### BFM-Zero

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `BFM-Zero-Flat-MultiMotionV2-G1-v0` | `FB-CPR-Aux` | BFM-Zero runner for flat multi-motion G1 tracking (FB-CPR-Aux code vendored in PMT; no external repo needed). | `python scripts/bfm_zero/train.py --task BFM-Zero-Flat-MultiMotionV2-G1-v0 --agent_preset smoke --num_envs <n> --headless` | Use BFM-Zero runner evaluation flags, for example `python scripts/bfm_zero/train.py --task BFM-Zero-Flat-MultiMotionV2-G1-v0 --agent_preset smoke --num_envs <n> --headless --eval_every <steps> --eval_horizon <frames>` |

### RGMT Paper Task

| Gym id | Algorithm / network | Description | Train | Play |
| --- | --- | --- | --- | --- |
| `RGMT-G1-v0` | `ppo` / `transformer` | Paper-faithful RGMT deploy task. | `python scripts/train.py --task RGMT-G1-v0 --num_envs <n> --headless --max_iterations <iters>` | `python scripts/play.py --task RGMT-G1-v0 --num_envs 1 --resume_path <checkpoint.pt> --motion_file <motion-file-or-dir> --headless --max_steps 300` |

## SONIC Pretrained Model

The SONIC task can load a release encoder and decoder from ONNX files. Point `PMT_SONIC_ONNX_DIR` at a directory containing:

```text
model_encoder.onnx
model_decoder.onnx
```

```bash
export PMT_SONIC_ONNX_DIR=<release-onnx-dir>
```

`sonic_mode` controls what is initialized and trained:

| Mode | Behavior |
| --- | --- |
| `scratch` | Train the SONIC policy from scratch without loading release ONNX weights. |
| `finetune_all` | Load release ONNX weights and train the robot encoder, decoders, critic, and action standard deviation. |
| `finetune_decoder` | Load release ONNX weights, freeze the robot encoder, and train the decoders, critic, and action standard deviation. |
| `play` | Load release ONNX weights and freeze the policy for rollout; no RSL checkpoint is required. |

Roll out the pretrained SONIC policy on one motion:

```bash
export PMT_SONIC_ONNX_DIR=<release-onnx-dir>
python scripts/play.py --task PMT-SONIC-G1-MultiMotionV2-Flat-v0 \
  --sonic_mode play --num_envs 1 --motion_file <motion-file.npz> \
  --headless --max_steps 300
```

Finetune the release policy:

```bash
export PMT_SONIC_ONNX_DIR=<release-onnx-dir>
python scripts/train.py --task PMT-SONIC-G1-MultiMotionV2-Flat-v0 \
  --num_envs <n> --headless --max_iterations <iters> \
  sonic_mode=finetune_decoder
```

Use `sonic_mode=finetune_all` instead when the robot encoder should be updated too.

## BFM-Zero

BFM-Zero uses a separate runner. The FB-CPR-Aux networks/agent are **vendored inside PMT**
(`motion_tracking_rl/bfm_zero/_vendor`, a verbatim copy of the minimal `humanoidverse` import
closure), so **no external BFM-Zero checkout is required** — only the PMT repo. The legacy
`BFM_ZERO_REPO` env var and `--bfm_zero_repo` flag are deprecated no-ops kept for backward
compatibility.

```bash
python scripts/bfm_zero/train.py --task BFM-Zero-Flat-MultiMotionV2-G1-v0 \
  --agent_preset smoke --num_envs <n> --headless
```

Use `--agent_preset smoke` for a small verification-oriented agent preset and `--agent_preset full` for the full training preset:

```bash
python scripts/bfm_zero/train.py --task BFM-Zero-Flat-MultiMotionV2-G1-v0 \
  --agent_preset full --num_envs <n> --headless
```

## Testing And Verification

Run the pure compatibility and path-resolution tests before release:

```bash
conda run -n cluster_isaaclab python -m pytest \
  tests/test_compat_matrix.py \
  tests/test_builder_slice.py \
  tests/test_compat_name_unification.py \
  tests/test_all_tasks_resolve.py \
  tests/test_paths.py \
  -q
```

This suite validates task id registration, task-name compatibility, builder coverage, path default behavior, and launchability metadata without running training. Runtime gate scripts under `tests/` create Isaac Lab app/runtime state and require an Isaac-capable environment.

## Not-Yet-Direct-Train Tasks

Two task configs are intentionally not single-command train targets:

| Config stem | Why it is not direct-train |
| --- | --- |
| `vision_ablation_base` | Composition and ablation demo plus pure-test target. It needs explicit network overrides, vision-teacher assets, and runtime wiring before it can become a direct train task. |
| `ppofinetune_vision_teacher_stepping_stone_latent_anchor` | PPO finetune scaffold for a pretrained vision-teacher checkpoint. The env and agent builders are intentionally not wired until that checkpoint and launch contract are provided. |

## The PMT pipeline: teacher → distill → finetune

PMT's most involved workflow trains a **blind privileged teacher** on terrain, **distills** it into
a **vision student** that replaces the privileged terrain knowledge with proprioception + a
height-scan (but not the privileged anchor), and optionally **PPO-finetunes** the student. This is
the stepping-stone reference pipeline.

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
teacher command `motion`, raw → student command `student_motion`. Optimized clips are produced
upstream by [TCRS](../TCRS/README.md); PMT consumes the resulting `*.npz`.

### Stage 1 — train the teacher (`stage: scratch`)

```bash
python scripts/train.py --task PMT-SteppingStone-G1-v0 \
  --num_envs <n> --headless --max_iterations <iters>
```

- Task: `configs/task/pmt_stepping_stone.yaml` (`network: transformer`, `obs: transformer_hist`,
  `sensor: none`, `algorithm: ppo`).
- Output: `logs/rsl_rl/pmt_stepping_stone/<run>/model_<iter>.pt`.

### Register the teacher checkpoint

Point `configs/checkpoints/ss_teacher.yaml` at the run you just trained:

```yaml
run_dir: ${paths.CKPT_ROOT}/pmt_stepping_stone/<timestamped_run>
checkpoint: model_9999.pt        # explicit file -> joined onto run_dir
```

### Stage 2 — distill into a vision student (`stage: distill`)

```bash
python scripts/train.py --task PMT-Distill-SteppingStone-LatentAnchor-G1-v0 \
  --num_envs <n> --headless --max_iterations <iters>
```

- Task: `configs/task/distill_stepping_stone_latent_anchor.yaml`
  (`network: vision_student_latent_anchor`, `obs: vision_student`, `sensor: height_scan`,
  `algorithm: distillation` → `DistillationRunner`).
- The task references the teacher via `network.teacher_ckpt: ${checkpoints.ss_teacher}`; the
  builder injects it into the frozen distillation target. With `teacher_ckpt: null` the teacher is
  random-init — the **runner path** runs end-to-end but the distillation loss is meaningless (this
  is the CI gate; supply a real teacher for a real run).
- A simpler `StudentTeacher` MLP-pair variant exists for path-testing:
  `PMT-Distill-SteppingStone-G1-v0`.

### Stage 3 — PPO-finetune the vision policy (`stage: finetune`)

The finetune scaffold is `configs/task/ppofinetune_vision_teacher_stepping_stone_latent_anchor.yaml`
(`stage: finetune` → `lr=5e-4`, `warmup_freeze_iters=200`, `reset_action_std_on_load=true`). It is
**not yet a one-command train target**: it requires a trained vision-teacher checkpoint and finetune
env/agent wiring before `gym.make` can build it. Wire those, then launch it like any other PPO task.

> The three `stage/*.yaml` files are the single source of the lr / freeze / resume / teacher-mix
> deltas; the rest of each task differs only by the independent axis it selects.

## Extending PMT

A task **selects** the independent axes (`robot` / `terrain` / `motion` / `scene` / `sensor` /
`obs` / `reward` / `network` / `algorithm` / `stage`); the builder **derives** the coupled ones
(`runner`, `obs_groups`, `decimation` / `sim.dt`, termination thresholds). See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full axis taxonomy and derivation table.

### Add a new task

Write **one** `configs/task/<stem>.yaml`: a `defaults:` list (one choice per axis) plus task-local
overrides. Tasks that differ only on *derived* fields share a YAML — you only need a new task YAML
when an *independent* axis choice differs.

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

Then map the stem to an env builder in `_ENV_BUILDERS` in `pmt_tasks/builder.py`: `build_env_cfg`
looks the stem up there and raises if it is absent. If the task reuses an existing family, this is a
**one-line** entry pointing the stem at that family's existing builder; only a genuinely new
scene/sensor wiring needs a new builder function. Optionally map the stem to a friendly gym id in
`pmt_tasks/registry_gym.py` (`_TASK_ID_MAP`), or rely on the `PMT-<stem>-v0` fallback. The pure CI
(`tests/test_all_tasks_resolve.py`) automatically loads every new YAML and asserts it composes +
derives + validates.

### Train on a new motion

**A. Plane / flat tasks — just point at the clips.** Drop your `*.npz` clips in a directory and
override `motion.motion_files`:

```yaml
# configs/task/my_flat_task.yaml  (terrain: flat, motion: multi or single_clip)
motion:
  motion_files: ${paths.MOTION_ROOT}/my_flat_clips   # a dir of NPZs, or a single .npz
```

For a single clip, select `motion: single_clip`; for many clips, `motion: multi` (eager) with a
sampler/storage choice (`sampler: uniform|adaptive|bin_adaptive`, `storage_mode: eager|streaming`).
Streaming bounds memory for large clip sets (e.g. the 100-style split).

**B. Terrain / PMT tasks — you need raw *and* optimized clips.** Terrain tracking requires both
versions of each clip: **optimized** clips (terrain-IK adapted) for the teacher, **raw** clips for
the vision student, plus the matching terrain mesh referenced by the `terrain` axis. Optimized clips
are produced upstream by [TCRS](../TCRS/README.md); PMT consumes the resulting `*.npz`. Wire them via
a paired motion YAML (model it on `configs/motion/stepping_stone_paired.yaml`):

```yaml
motion_files:     ${paths.TERRAIN_MOTION_ROOT}/<dataset>/<clip>/optimized   # teacher
raw_motion_files: ${paths.TERRAIN_MOTION_ROOT}/<dataset>/<clip>/raw         # student
paired: true
```

Set the per-motion control rate (`decimation` / `sim_dt`) in the motion YAML — the env builder copies
it onto the env cfg (e.g. backflip uses `dec=10`/`dt=0.002`). Termination thresholds live in the
per-family env cfgs (`pmt_tasks/env_cfgs/…`), not the motion YAML. Clip `*.npz` are stored in
Isaac-Lab BFS body/joint order — see `scripts/pmt_npz_to_mjlab.py` for the exact array contract.

### Add a new robot

The `robot` axis currently ships only G1, but the pattern is:

1. **Asset config** — add a robot module under `pmt_tasks/robots/` (model it on `g1.py`) exposing an
   Isaac Lab `ArticulationCfg` (USD path, actuator stiffness/damping/armature) and an action-scale
   constant. Robot USD/meshes are resolved through `pmt_tasks/asset_config.py` (`PMT_ASSET_DIR`).
2. **Axis YAML** — add `configs/robot/<name>.yaml` with `name:`, default `decimation`, `sim_dt`, and
   any robot-level fields.
3. **Wire it into the env cfg** — the env families import the robot's `ArticulationCfg`
   (e.g. `from pmt_tasks.robots.g1 import G1_CYLINDER_CFG, G1_ACTION_SCALE` in
   `pmt_tasks/env_cfgs/multi_motion_flat.py`). Add the analogous import/branch for your robot.
4. **Motions for that embodiment** — motion clips are robot-specific (joint order / body set), so
   provide clips retargeted to the new robot.

Then select `robot: <name>` in any task YAML. (Full cross-embodiment training, including SMPL/human
encoders, is the SONIC path — see the SONIC section above.)

### Add a network or algorithm

Full worked guide in `configs/README.md`.

**Add a network** (≈5 touch points):

1. Decorate the `nn.Module` in `motion_tracking_rl/networks/`:
   `@register_network("MyNetClass", compat_name="my_net")` — `name` must equal the ckpt `class_name`
   for checkpoint compatibility; `compat_name` is the axis name.
2. Add the module to the network import list in `registry.autoload()` so the decorator fires.
3. `configs/network/my_net.yaml` with `name: MyNetClass` + hyperparams.
4. Extend the `policy` `@configclass` union in `pmt_tasks/isaaclab_rl/rsl_rl/rl_cfg.py`.
5. Add `"my_net"` to the `compatible_networks` of every algorithm in `compat.py` that should accept it.

**Add an algorithm** (≈3 touch points):

1. Decorate the class in `motion_tracking_rl/algorithms/` and add it to `registry.autoload()`.
2. `configs/algorithm/my_alg.yaml` with `name:` + feature flags + hyperparams.
3. A new `compat.SPECS` entry (`AlgorithmSpec`) declaring its runner, compatible networks, feature
   support, paired-command requirement, and required obs sets.

`registry.assert_compat_consistency()` (run by the builder and CI) fails loud if the registry tables
drift from `compat.SPECS`. The generated matrix is [compat_matrix.md](compat_matrix.md).
