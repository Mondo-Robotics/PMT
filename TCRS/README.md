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

## Quick start

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

### Key arguments

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

## Visualize a generated clip

Replay any `raw/`, `optimized/`, or `ghost/` npz on the terrain scene with `visualize_npz.py`:

```bash
# headless -> video (works under MUJOCO_GL=egl; needs imageio + imageio-ffmpeg)
MUJOCO_GL=egl python -m visualize_npz \
  --npz outputs/my_experiment/optimized/<clip>_optimized.npz \
  --out clip.mp4

# interactive MuJoCo viewer (needs a display; omit --out)
python -m visualize_npz --npz outputs/my_experiment/optimized/<clip>_optimized.npz
```

Pass `--xml <scene.xml>` if the clip was generated on a non-default terrain. `--out` accepts
`.mp4` or `.gif`; `--speed`, `--fps`, `--width`, `--height` are also available.

## Dependencies

- `mujoco>=3.1`, `numpy`, `scipy`
- `imageio` + `imageio-ffmpeg` (only for `visualize_npz.py --out <video>`)
