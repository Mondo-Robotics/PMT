# TCRS — Terrain-adaptive motion optimizer (MPPI)

Takes a **flat-ground motion clip** (`.npz`) and a **terrain scene XML**, and generates the
**terrain-optimized** version of that motion for the Unitree G1 — feet are re-placed to land
on the terrain via an MPPI swing planner + Jacobian IK, the pelvis height is re-targeted, and
the result is exported as `.npz` clips ready to be consumed downstream (e.g. as PMT's
`optimized/` reference clips).

```
 flat motion .npz  +  terrain scene .xml
              │
              ▼
   stair_mppi.mppi_foot_planner_smooth   (MPPI swing → root-z filter → IK)
              │
              ▼
   raw/ + optimized/ + ghost/  *.npz     (PMT-compatible schema)
```

## Layout

```
stair_mppi/
  mppi_foot_planner_smooth.py   # MAIN entry point (CLI). Pipeline orchestrator.
  mppi_foot.py                  # MPPI foot-swing trajectory optimizer
  ghost_ik.py                   # Jacobian whole-body IK solver (default backend)
  ik_config.json                # default IK weights (loaded by --ik_config default)
  minimal_mppi_demo.py          # MotionClip / TerrainReference / footstep helpers
  terrain.py                    # raycast terrain-height queries
  terrain_warp.py               # support-aware root-z filter
  gait_phase.py                 # gait-phase clock (contact detection)
assets/
  g1/g1_29dof_scene_stairs_ud.xml   # terrain scene (positive stepping-stones + stairs, box geoms) — default --xml
  g1/g1_29dof_positive_stepping_with_stairs_box.xml  # identical scene, original name
  g1/g1_29dof_rev_1_0_no_plane.xml  # G1 robot model, no ground plane (included by the scene)
  g1/g1_29dof_rev_1_0.xml           # G1 robot model with a flat ground plane (alt scene include)
  g1/meshes/*.STL                   # robot collision/visual meshes
  motions/walk1_subject1.npz        # sample flat-ground motion clip
```

This module is the **core MPPI terrain-adaptation function only**. The optional IK backends
(curobo / drake), the plum-blossom and somersault planners, the window-IK mode, and all the
figure-rendering / replay utilities have been removed to keep it minimal. The default path is
`--planner mppi --ik_backend jacobian`.

## Usage

> Always run from the repo root (the dir that contains `stair_mppi/`), otherwise
> `python -m stair_mppi...` fails with `ModuleNotFoundError: No module named 'stair_mppi'`.
> Needs an env with `mujoco` (e.g. conda `my_isaaclab` / `wbt`).

```bash
cd /home/zifan_wang/opensource/PMT/TCRS
```

### Step 1 — Generate (flat motion + terrain → optimized motion)

Batch (headless) mode is the generator — it writes `raw/`, `optimized/`, `ghost/` npz dirs:

```bash
# minimal: uses the default terrain XML (assets/g1/g1_29dof_scene_stairs_ud.xml)
MUJOCO_GL=egl python -u -m stair_mppi.mppi_foot_planner_smooth \
  --motion assets/motions/walk1_subject1.npz \
  --start_frame 1100 --n_frames 500 \
  --planner mppi --ik_backend jacobian \
  --n_rounds 8 \
  --batch_output_dir outputs/my_experiment
```

This writes, per accepted round:
```
outputs/my_experiment/
  raw/        <clip>_f1100_round_0000_dx..._raw.npz        # original flat motion, terrain-placed
  optimized/  <clip>_f1100_round_0000_dx..._optimized.npz  # terrain-adapted (the one you want)
  ghost/      <clip>_f1100_round_0000_dx..._ghost.npz       # z-only ghost reference
```

### Step 2 — Visualize an output clip

Replay any `raw/`, `optimized/`, or `ghost/` npz on the terrain scene with `visualize_npz.py`.
Point `--npz` at one of the files Step 1 produced:

```bash
# headless -> video (works under MUJOCO_GL=egl; needs imageio + imageio-ffmpeg)
MUJOCO_GL=egl python -m visualize_npz \
  --npz outputs/my_experiment/optimized/walk1_subject1_f1100_round_0000_dx+16.437_dy-3.667_dyaw+2.2520_optimized.npz \
  --out clip.mp4

# or just grab the first optimized clip automatically:
CLIP=$(ls outputs/my_experiment/optimized/*.npz | head -1)
MUJOCO_GL=egl python -m visualize_npz --npz "$CLIP" --out clip.mp4

# interactive MuJoCo viewer instead of a video (needs a display; omit MUJOCO_GL=egl and --out)
python -m visualize_npz --npz "$CLIP"
```

`--out` accepts `.mp4` or `.gif`; `--xml <scene.xml>` if the clip used a non-default terrain;
`--speed`, `--fps`, `--width`, `--height` are also available. The visualizer force-enables all
geom groups, so the stepping-stone terrain (geom group 2) always shows. There is **no flat
floor plane** — the robot stands on the box stepping-stones; the blue checkerboard is just the
default ground grid.

### Key arguments (Step 1)

| Arg | Default | Meaning |
|-----|---------|---------|
| `--motion` | `assets/motions/walk1_subject1.npz` | input flat-ground clip (`.npz`) |
| `--xml` | `assets/g1/g1_29dof_scene_stairs_ud.xml` | terrain scene (must `<include>` a robot XML, e.g. `g1_29dof_rev_1_0_no_plane.xml`) |
| `--start_frame` / `--n_frames` | `2600` / `1000` | slice of the clip to optimize |
| `--planner` | `mppi` | swing planner (core path) |
| `--ik_backend` | `jacobian` | IK solver (core path) |
| `--n_rounds N` | `0` | **N>0 → headless batch mode**; samples N terrain placements, writes npz |
| `--batch_output_dir` | `outputs/terrain` | output root (`raw/`, `optimized/`, `ghost/` subdirs) |

Run headless with `MUJOCO_GL=egl`. With `--n_rounds 0` (the default) it instead opens an
interactive MuJoCo viewer.

### Output

Each accepted round writes `{raw,optimized,ghost}/<tag>.npz`, where `<tag>` encodes the
clip name, start frame, and the random terrain transform (`dx/dy/dyaw`). The npz schema
matches the input motion contract so it can be consumed directly by PMT:

`fps`, `joint_pos [T,29]`, `joint_vel [T,29]`, `body_pos_w [T,30,3]`, `body_quat_w [T,30,4]`,
`body_lin_vel_w`, `body_ang_vel_w` (plus `transform_dx/dy/dyaw` and IK-quality metadata).

## Dependencies

- `mujoco>=3.1`, `numpy`, `scipy`
- `imageio` + `imageio-ffmpeg` (only for `visualize_npz.py --out <video>`)
