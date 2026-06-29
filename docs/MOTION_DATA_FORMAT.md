# Motion clip format — what to feed the motion command

PMT's motion commands (`MultiMotionCommandV2` and friends, in
`pmt_tasks/mdp/commands/`) consume **`.npz` motion clips**. This is the contract: get these
keys, shapes, dtypes, and body/joint order right and any clip will load; get them wrong and the
clip either fails to load or *tracks garbage* (it loads and steps but the order is silently
mismatched). The loader is `MotionDataStore._load_motion_file`
(`pmt_tasks/mdp/commands/multi_motion_command.py`).

## Required keys

A clip with `T` frames, `J` joints, `B` bodies is a `.npz` with **exactly these arrays**
(all `float32`):

| Key | Shape | Meaning |
| --- | --- | --- |
| `fps` | `(1,)` | Frame rate, e.g. `[50.]`. Must equal the env control rate (`1 / (decimation·sim_dt)`). |
| `joint_pos` | `(T, J)` | Joint positions (rad), articulation order — see below. |
| `joint_vel` | `(T, J)` | Joint velocities (rad/s), same order. |
| `body_pos_w` | `(T, B, 3)` | Per-body position in the **world** frame (m). |
| `body_quat_w` | `(T, B, 4)` | Per-body orientation, world frame, **quaternion `(w, x, y, z)`** (scalar-first, Isaac-Lab convention). |
| `body_lin_vel_w` | `(T, B, 3)` | Per-body linear velocity, world frame (m/s). |
| `body_ang_vel_w` | `(T, B, 3)` | Per-body angular velocity, world frame (rad/s). |

For the **G1** these are `J = 29` joints and `B = 30` bodies. A real demo clip
(`assets/motions/.../*_optimized.npz`) is exactly:

```
fps            (1,)            float32
joint_pos      (1500, 29)      float32
joint_vel      (1500, 29)      float32
body_pos_w     (1500, 30, 3)   float32
body_quat_w    (1500, 30, 4)   float32
body_lin_vel_w (1500, 30, 3)   float32
body_ang_vel_w (1500, 30, 3)   float32
```

All seven keys are required; a missing key raises a `KeyError` at load. `T` must match across
every array. The command selects the bodies it tracks from `body_*_w` by index, so the **full**
body set must be present in the file (don't pre-trim it).

## Extra keys are metadata (kept, not required)

Any key **not** in the core set
(`fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, positions`)
is stashed verbatim into the clip's `metadata` dict and ignored by the tracker. The demo clips
carry `transform_dx / transform_dy / transform_dyaw` (the TCRS terrain-placement offset) this way —
harmless to keep, safe to omit.

## Body / joint order (the silent-failure trap)

Motion arrays are indexed **positionally** — there are no per-frame name labels. PMT clips are
saved in **Isaac-Lab BFS articulation order**. If you generate clips from another tool, you must
emit them in this exact order (or remap before training). The canonical lists live in
`scripts/pmt_npz_to_mjlab.py` (`BFS_BODIES`, 30 entries; `BFS_JOINTS`, 29 entries) — that script
is also the reference for converting to a different engine's order (see
[`MJLAB_USAGE.md`](MJLAB_USAGE.md)). The order begins:

```
bodies:  pelvis, left_hip_pitch_link, right_hip_pitch_link, waist_yaw_link, ...   (30 total)
joints:  left_hip_pitch_joint, right_hip_pitch_joint, waist_yaw_joint, ...        (29 total)
```

The order is **robot-specific** — clips for a new embodiment must follow that robot's articulation
order and body set (see "Add a new robot" in [`USAGE.md`](USAGE.md)).

## Paired clips (distill / SONIC)

- **Distill / vision** tasks pair a terrain-`optimized` clip (teacher reference) with the matching
  flat-`raw` clip (student reference). Both `.npz` follow the schema above; they're paired by
  filename stem and wired via a paired motion YAML (`raw_motion_files:` + `paired: true`). See the
  Demo-data section of [`README.md`](../README.md) and `assets/motions/README.md`.
- **SONIC** (cross-embodiment) additionally loads a sibling **human** clip carrying a `positions`
  array `(T_human, …)` (its own frame rate, default 30 Hz). It's loaded only when the command asks
  for it; `positions` is treated as a core key so it is not duplicated into metadata.

## Control rate

`fps` in the clip must match the env control rate the task runs at. Set the per-motion rate in the
**motion YAML** (`decimation` / `sim_dt`); the builder copies it onto the env cfg. Flat clips use
`dec=4 / dt=0.005` → 50 Hz; the backflip family uses `dec=10 / dt=0.002`. A clip whose `fps`
disagrees with `1 / (decimation·sim_dt)` is played at the wrong speed.

## Generating clips

Terrain-anchored clips are produced by **TCRS** (raw flat clip + terrain XML → `raw/` +
`optimized/` `.npz`); see [`../TCRS/README.md`](../TCRS/README.md). Whatever the source, the output
must satisfy the schema above. To sanity-check a clip without launching a full env, use the NPZ
validator built into `scripts/mjlab_smoke_phase_a.py` (`validate_npz`) — it asserts the six core
body/joint keys are present (`fps` optional there), that `body_quat_w` is `(T, B, 4)`, and that the
joint and body frame counts agree.
