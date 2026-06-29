# Running PMT on the mjlab (MuJoCo-Warp) backend

PMT can run on **[mjlab](https://github.com/mujocolab/mjlab)** (a MuJoCo-Warp reimplementation of
Isaac Lab's manager-based API) in addition to Isaac Lab. This is a **port in progress**, not a
drop-in second engine — read the scope below before relying on it. The design rationale and full
status table are in [`MJLAB_BACKEND_PLAN.md`](MJLAB_BACKEND_PLAN.md); this doc is the *how-to*.

## What works today

- **Backend dispatch** — `pmt_tasks.builder.build_env_cfg(task, backend="mjlab")` emits an mjlab
  `ManagerBasedRlEnvCfg` from the same resolved PMT config that drives the Isaac-Lab path.
- **Flat tracking family** — these task stems have an mjlab env builder (see
  `pmt_tasks/backends/mjlab.py`, `_MJLAB_ENV_BUILDERS`):
  `multimotionv2_flat`, `multimotionv2_uniform_flat`, `multimotionv2_adaptive_flat`,
  `multimotionv2_streaming_flat`, `multimotionv2_streaming_100style`, `multimotionv2_100style_flat`,
  `sonic_multimotion_flat`.
- **Trained-ckpt rollout in mjlab** — load a PMT G1 flat checkpoint and measure / view its tracking
  in mjlab (`scripts/mjlab_eval_pmt_ckpt.py`, `scripts/mjlab_view_pmt_ckpt.py`).

## What does NOT work yet

- Terrain / vision / distill / BPO tasks have **no mjlab env builder** — `build_env_cfg(...,
  backend="mjlab")` raises `NotImplementedError` listing the wired stems. Isaac Lab stays the
  default and only backend for those.
- **Checkpoints do not transfer for free.** Two physics engines → two policies. A PMT policy
  trained in Isaac Lab is only a *partial* zero-shot in mjlab (stays upright, positive reward, body
  tracked — but not deployment-grade). Exact tracking needs a short **fine-tune / retrain pass in
  mjlab**. "Same task on two backends" = two training runs.

## Prerequisites

mjlab is a **separate environment** from your Isaac Lab env — PMT does not vendor it. Get a working
mjlab install (its own `.venv`), then run the PMT mjlab scripts with **mjlab's interpreter** so both
`mjlab` and `pmt_tasks` are importable:

```bash
# from the PMT repo root, using mjlab's venv interpreter
<mjlab-repo>/.venv/bin/python -m pip install -e .     # make pmt_tasks importable in mjlab's venv
```

The backend-neutral term math is bridged by `pmt_tasks/mdp/_backend.py` (resolves the Isaac-Lab vs
mjlab math-util and robot-data-field name differences behind one import), so PMT's MDP terms run
unchanged on mjlab.

## 1. Convert a clip to mjlab body/joint order

PMT clips are stored in **Isaac-Lab BFS** order; mjlab indexes motion arrays against its **MJCF**
order. The two differ, so a clip loads and steps but tracks garbage unless remapped (verified by
forward kinematics, max body-pos error 0.000 m). Convert once:

```bash
<mjlab-repo>/.venv/bin/python scripts/pmt_npz_to_mjlab.py in.npz out_mjlab.npz
```

(See [`MOTION_DATA_FORMAT.md`](MOTION_DATA_FORMAT.md) for the clip schema and the BFS order lists.)
The viewer script auto-remaps a raw PMT clip for you; the eval script expects an already-remapped
(`*_mjlab.npz`) clip.

## 2. Sanity-check a clip in stock mjlab

Proves a real PMT clip loads, resets, and steps with finite tracking errors in mjlab's stock G1
flat env (also runs the NPZ validator):

```bash
<mjlab-repo>/.venv/bin/python scripts/mjlab_smoke_phase_a.py --motion out_mjlab.npz
```

## 3. Evaluate a trained PMT checkpoint in mjlab

```bash
<mjlab-repo>/.venv/bin/python scripts/mjlab_eval_pmt_ckpt.py \
    --ckpt checkpoints/pretrained/multimotionv2_flat.pt \
    --motion out_mjlab.npz --steps 200 --num-envs 16
```

Reports joint / anchor / body tracking error. The script builds an mjlab flat-tracking env whose
**actor** obs group uses mjlab's critic-term layout (the 286-dim PMT flat actor obs:
`command(58) anchor_pos_b(3) anchor_ori_b(6) body_pos(42) body_ori(84) base_lin_vel(3)
base_ang_vel(3) joint_pos(29) joint_vel(29) actions(29)`), loads the actor weights + obs
normalizer, and wires the MJCF↔BFS joint-order bridge **at the policy boundary** (the policy was
trained in BFS order, so its inputs/outputs must be permuted, not just the clip).

## 4. View a checkpoint in the interactive viewer

```bash
# native window if $DISPLAY is set, else a viser web URL
<mjlab-repo>/.venv/bin/python scripts/mjlab_view_pmt_ckpt.py \
    --ckpt checkpoints/pretrained/multimotionv2_flat.pt \
    --motion assets/motions/.../some_clip.npz       # raw PMT clip; auto-remapped

# force the web viewer (good over SSH) — prints a URL
<mjlab-repo>/.venv/bin/python scripts/mjlab_view_pmt_ckpt.py --ckpt ... --motion ... --viewer viser
```

Pass `--already-mjlab` if the clip is already in MJCF order.

## Selecting the backend in code

`build_env_cfg` resolves the backend by precedence **explicit arg > `$PMT_BACKEND` > a `backend:`
config axis > `isaaclab`**:

```python
from pmt_tasks.builder import build_env_cfg
env_cfg = build_env_cfg("PMT-G1-MultiMotionV2-Flat-v0", backend="mjlab")
```

```bash
export PMT_BACKEND=mjlab     # or leave unset / =isaaclab for the default
```

Default is always `isaaclab`, so every existing Isaac-Lab task and command is unchanged.
