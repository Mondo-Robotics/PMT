# pmt/ — the PMT transformer-teacher + distillation family

The core PMT pipeline: blind privileged **transformer teachers** trained on terrain, then
**distilled** into vision students. All env cfgs here use the Unitree G1 and the transformer
obs/action stack (`obs: transformer_hist`). Built by the `_build_*` functions in
`pmt_tasks/builder.py`.

## Tasks

| gym id | task yaml | env cfg | network | terrain |
| --- | --- | --- | --- | --- |
| `PMT-SteppingStone-G1-v0` | `pmt_stepping_stone.yaml` | `stepping_stone.py` | `TransformerActorCritic` | stepping-stone mesh |
| `PMT-WalkDanceBigMap-G1-v0` | `walk_dance_bigmap.yaml` | `stepping_stone.py` | `TransformerActorCritic` | big_map mesh |
| `PMT-CartwheelBigMap-G1-v0` | `cartwheel_bigmap.yaml` | `stepping_stone.py` | `TransformerActorCritic` | big_map mesh |
| `PMT-Backflip-G1-v0` | `backflip.yaml` | `backflip.py` | `TransformerActorCritic` | big_map mesh |
| `PMT-TerrainFlatMix-G1-v0` | `terrain_flat_mix.yaml` | `terrain_flat_mix.py` | `TransformerActorCritic` | big_map + flat patch |
| `PMT-AdaptiveSampling-G1-v0` | `pmt_adaptive_sampling.yaml` | `adaptive_sampling.py` | `TransformerActorCritic` | flat plane |
| `PMT-AdaptiveSampling-Baseline-G1-v0` | `pmt_adaptive_sampling_baseline.yaml` | `adaptive_sampling.py` | `TransformerActorCritic` | flat plane |
| `PMT-Distill-SteppingStone-G1-v0` | `distill_stepping_stone.yaml` | `distill_stepping_stone.py` | `StudentTeacher` (MLP pair) | stepping-stone + height scan |
| `PMT-Distill-SteppingStone-LatentAnchor-G1-v0` | `distill_stepping_stone_latent_anchor.yaml` | `distill_stepping_stone.py` | `VisionStudentTeacher` | big_map + height scan |
| `PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0` | `ppofinetune_vision_teacher_stepping_stone_latent_anchor.yaml` | `finetune_stepping_stone.py` | vision student | big_map + height scan |

- **Teacher tasks** (`SteppingStone`, `WalkDanceBigMap`, `CartwheelBigMap`) all share
  `stepping_stone.py` — only the terrain mesh and motion clip dir change via config (no new env
  class). These are blind: full proprio + command obs, **no vision**.
- **Backflip** exercises the per-motion control rate (`dec=10` / `dt=0.002` flows from the motion
  axis) and a z-only end-effector termination.
- **TerrainFlatMix** trains one transformer on a mixture of terrain-anchored + flat clips with
  per-clip origin/reset-noise (`UnifiedMotionCommandV2`).
- **AdaptiveSampling** is a flat deploy task with the adaptive-sampling streaming command over the
  sonic + snap_robot corpus; the `-Baseline-` variant zeroes the sampling signals for an A/B.
- **Distill tasks** distill a frozen teacher into a vision student over a **paired** command
  (optimized clips → teacher `motion`, raw clips → student `student_motion`); `algorithm:
  distillation` routes to `DistillationRunner`. The latent-anchor task reads the teacher via
  `network.teacher_ckpt: ${checkpoints.ss_teacher}` (set `configs/checkpoints/ss_teacher.yaml`).

## Train / play

```bash
# train a teacher (stepping-stone)
python scripts/train.py --task PMT-SteppingStone-G1-v0 --num_envs <n> --headless

# distill into a vision student (needs a real teacher in ss_teacher.yaml)
python scripts/train.py --task PMT-Distill-SteppingStone-LatentAnchor-G1-v0 --num_envs <n> --headless
```

## Pretrained models

Two big_map teachers ship pretrained and load directly:

```bash
python scripts/play.py --task PMT-Backflip-G1-v0 \
  --resume_path checkpoints/pretrained/backflip_teacher.pt --num_envs 16

python scripts/play.py --task PMT-CartwheelBigMap-G1-v0 \
  --resume_path checkpoints/pretrained/cartwheel_teacher.pt --num_envs 16
```

Both need the big_map mesh + the corresponding terrain-anchored clips; see
[`checkpoints/pretrained/README.md`](../../../checkpoints/pretrained/README.md). The teacher →
distill → finetune pipeline is documented in the top-level [README §6](../../../README.md#6-the-pmt-pipeline-teacher--distill--finetune).
