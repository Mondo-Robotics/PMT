# Demo motion clips

A 100-pair sample of terrain-anchored G1 motion clips, so the terrain / distill tasks are
runnable out-of-the-box. Tracked with **git-lfs** (`git lfs pull` after cloning).

```
terrain_mocaphouse/walk_dance1sub2start/
  raw/        100× *_raw.npz         flat-retargeted clips (student reference)
  optimized/  100× *_optimized.npz   terrain-optimized clips (teacher reference)
```

- **Paired by filename stem**: `<stem>_raw.npz` ↔ `<stem>_optimized.npz` (the distill pipeline
  pairs them — `optimized` → teacher command `motion`, `raw` → student command `student_motion`).
- The `optimized` clips are the TCRS terrain-adaptation output (feet re-placed onto the big_map
  mesh); the `raw` clips are the flat-retarget inputs. Schema: 50 fps, 29 G1 joints, world-frame
  body poses (`joint_pos/joint_vel/body_pos_w/body_quat_w/...`).
- This is a **100-clip subset** of the full 500-clip `walk_dance1sub2start` set (kept small for the
  public repo); enough to train/play the terrain teacher and the distill student on a demo scale.

## Point PMT at this data

The terrain tasks resolve clips under `${paths.TERRAIN_MOTION_ROOT}/terrain_mocaphouse/...`. Set
that root to this repo's `assets/motions` (local profile):

```bash
export PMT_TERRAIN_MOTION_ROOT=$(pwd)/assets/motions      # from the repo root
# WalkDanceBigMap teacher (uses optimized/ clips on the big_map mesh)
python scripts/train.py --task PMT-WalkDanceBigMap-G1-v0 --num_envs <n> --headless
# or play the shipped teacher checkpoint:
python scripts/play.py --task PMT-WalkDanceBigMap-G1-v0 \
  --resume_path checkpoints/pretrained/walkdance_bigmap_teacher.pt --num_envs 4
```

(The distill tasks pair `optimized/` + `raw/` automatically from the same parent dir.)
