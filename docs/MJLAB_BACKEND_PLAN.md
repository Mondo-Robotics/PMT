# PMT → mjlab: Multi-Backend Port Plan

**Status:** Phases A–E implemented & tested on branch `mjlab-port` · **Date:** 2026-06-24
· **Author:** research round (Codex + 2× Claude planning agents, in agreement)

Goal: make PMT (Isaac Lab humanoid motion-tracking RL) run on **mjlab** (MuJoCo-Warp
reimplementation of Isaac Lab's manager-based API). User preference: **port to mjlab first,
merge to a unified backend later** — without paying merge debt twice.

## Implementation status (branch `mjlab-port`)
| Phase | What | Status |
|---|---|---|
| A | branch + stock-mjlab smoke on real PMT clip; **found+fixed** BFS→MJCF clip remap (`scripts/pmt_npz_to_mjlab.py`, FK-verified 0.000 m) | ✅ done |
| B | USD-free `import pmt_tasks.mdp` (guard fix) + `_backend.py` shim (math + RobotView aliases) | ✅ done |
| C | `build_env_cfg(backend=...)` dispatch + `backends/mjlab.py` emitter (flat family) | ✅ done |
| D | load trained `model_39999.pt` → **partial zero-shot** (upright 5 s, +reward, body tracked); exact transfer needs short mjlab fine-tune | ✅ done |
| E | support matrix: 17 configs resolve, **7 flat tasks emit mjlab envs**, 10 out-of-scope fail loud | ✅ done |
| E+ | port multi-clip `MotionDataStore`/samplers (mjlab `MotionCommand` is single-clip); terrain/vision/distill parity; mjlab fine-tune for deployment tracking | ⏳ future |

Tests: **510 passed / 13 skipped** (wbt, no regression) + shim/emitter/matrix green in the mjlab venv.
New files: `scripts/{pmt_npz_to_mjlab,mjlab_smoke_phase_a,mjlab_eval_pmt_ckpt}.py`,
`pmt_tasks/mdp/_backend.py`, `pmt_tasks/backends/mjlab.py`,
`tests/test_mjlab_{backend_shim,emitter,task_matrix}.py`.

---

## TL;DR — the agreed decision

**Hybrid, not either/or.** Three independent investigations (Codex + two Claude planning
agents) converged on the same answer:

> **Operationally:** do the port on a short-lived branch where **mjlab is the default (and
> initially only) backend** — this gives fast porting momentum.
>
> **Architecturally:** the code on that branch should already be **Approach 2** — keep PMT's
> shared config/builder spine intact, and isolate the backend behind a dispatch seam. Do **not**
> build a standalone mjlab-only architecture.

Why: a pure fork (naïve Approach 1) makes the later merge a *semantic reconciliation* problem
across configs, task names, reward terms, runner settings, and motion assumptions — Codex called
it "behavioral archaeology." Keeping the shared spine costs almost nothing now (PMT already
separates a backend-neutral config pass from lazy Isaac-importing emit) and makes the eventual
merge a near-no-op.

This is **NOT** a migration of trained policies: two physics engines → two checkpoints. "Same
task on two backends" means two training runs and two reward-tuning passes. Acceptable for a
sim2sim / port goal; not a free win.

---

## Why this is tractable (ground truth from the code)

mjlab is a deliberate reimplementation of Isaac Lab's manager API and **already ships a
motion-tracking task** at `mjlab/src/mjlab/tasks/tracking/` — itself a BeyondMimic /
`whole_body_tracking` port, the same lineage as PMT. So we port between cousins, not into a void.

| Layer | Isaac Lab (PMT) | mjlab | Divergence |
|---|---|---|---|
| Manager term cfg classes (`ObservationTermCfg`, `RewardTermCfg`, `EventTermCfg`, `CommandTermCfg`, `SceneEntityCfg`, `ManagerTermBase`) | `isaaclab.managers` | `mjlab.managers` | **name-identical** |
| **Motion NPZ schema** (`joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w`) | PMT loader | mjlab `MotionLoader` | **identical** — existing clips load unmodified* |
| Term math reading the **command** (`command.robot_body_pos_w`, `anchor_pos_w`) | PMT `rewards/terminations/observations` | mjlab exposes byte-identical attr names | **zero** |
| Raw `robot.data.*` reads | `body_pos_w`, `root_lin_vel_b`, `applied_torque`, `asset_name` | `body_link_pos_w`, `root_link_lin_vel_b`, `joint_torques`, `entity_name` | **small, mechanical** (field renames) |
| rsl-rl runner | sim-agnostic; consumes obs/reward tensors | mjlab `MotionTrackingOnPolicyRunner` (+ONNX, wandb) | thin VecEnv-wrapper normalization |
| Env-cfg **emit** (scene/asset/sensors/events) | `@configclass` tree, USD `ArticulationCfg`, USD/`pxr` terrain | `make_*()`→`ManagerBasedRlEnvCfg` dataclass, MJCF `Entity` | **large, irreducibly backend-specific** |

\* subject to body/joint **ordering** matching the mjlab MJCF — see Risk #1.

### `robot.data.*` field-rename map (mjlab) — CORRECTED after review
```
body_pos_w      → body_link_pos_w        root_lin_vel_b  → root_link_lin_vel_b
body_quat_w     → body_link_quat_w       root_ang_vel_b  → root_link_ang_vel_b
body_lin_vel_w  → body_link_lin_vel_w    applied_torque  → qfrc_actuator   (NOT joint_torques)
body_ang_vel_w  → body_link_ang_vel_w    SceneEntityCfg.asset_name → .name (NOT entity_name)
# identical already: joint_pos, joint_vel, default_joint_pos, default_joint_vel, projected_gravity_b
```
> **Review correction 1 (CRITICAL):** mjlab `EntityData.joint_torques` **raises
> `NotImplementedError`** ("ambiguous; use qfrc_actuator"). PMT `rewards.py:76,89` reads
> `applied_torque` for power → shim MUST map to `qfrc_actuator` (actuator force in joint space).
> **Review correction 2:** mjlab `SceneEntityCfg` field is **`name`** (`scene_entity_config.py:63`),
> not `entity_name` — and this is a **config-time** rename, the RobotView (data-time) shim can't
> fix it. The mjlab *command cfg* separately uses `entity_name`/`anchor_body_name` vs PMT's
> `asset_name`/`anchor_body`.
> **Review correction 3:** the "term math reading the command = zero divergence" row is too broad.
> True only for terms consuming a stable command interface. Narrow it to a **`MotionCommandView`
> contract** (anchor_pos_w, body_pos_relative_w, body_quat_relative_w, robot body targets) and add
> golden parity tests on a fixed clip+frame. Also: PMT terminations use `GRAVITY_VEC_W` /
> `raw_body_pos_w_with_terrain_height`; mjlab uses `gravity_vec_w` and has no terrain-height
> variant → those terms are NOT trivially shared.

### Verified semantic contract (checked against real code/assets — 2026-06-24)
- **mjlab G1**: nq=36 (7 free + **29 actuated**), **30 bodies**, all **14 tracked bodies +
  `torso_link` anchor present**. Trained-clip npz: **29 joints, 30 bodies, fps=50**. → counts
  match; env rate = 0.005·4 = 50 Hz = clip fps, **no resampling** for this family.
- **Residual runtime risk** (Phase A smoke catches it): the clip's 30-body *axis order* must match
  mjlab's model body order — mjlab indexes `body_pos_w[:, find_bodies(names, preserve_order=True)]`
  with no shape/order validation.
- mjlab `MotionLoader` ignores `fps` (PMT requires it) → clips at a different fps would silently
  play at wrong speed. **Add an NPZ validator** (keys, dtype, shape, fps==env-rate, quaternion
  convention, joint/body names+order) instead of trusting "loads unmodified."

### ⚠️ Phase A RESULT (executed 2026-06-24) — clips need a body/joint axis REMAP
Running mjlab's stock G1-flat task on a real cluster clip (`100style/.../Aeroplane_BR.npz`)
**loads, resets, and steps with finite errors — but tracks GARBAGE** because PMT/SONIC clips
store `body_*`/`joint_*` arrays in **Isaac-Lab BFS articulation order**
(`whole_body_tracking/g1_motion_gen/g1mg/constants.py`), while mjlab indexes them positionally
against **MJCF order**. Diagnostic giveaway: raw npz frame-0 `torso_link` z ≈ 0.06 m (floor).
This is exactly Risk #1 — a *silent* failure that a naive "it runs!" smoke would have missed.

**Fix (verified by forward kinematics, max body-pos error 0.000 m; confirmed live in mjlab:
reference anchor z 0.06 m → 0.85 m, matching robot):** a write-time converter
`scripts/pmt_npz_to_mjlab.py` permutes both axes BFS→MJCF:
```
body_perm  = [0,1,4,7,10,14,18,2,5,8,11,15,19,3,6,9,12,16,20,22,24,26,28,13,17,21,23,25,27,29]
joint_perm = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]
```
Decision for Phase C: do the remap **once at clip-ingest** (offline convert, cache mjlab-order
npz) rather than per-step in the command term — keeps the hot path clean and the mjlab
`MotionCommand` unchanged. `scripts/mjlab_smoke_phase_a.py` is the NPZ validator/smoke harness.

**Codex independently confirmed** the permutations against the SONIC source
(`gear_sonic/envs/manager_env/robots/g1.py:28`, torso at source idx 9). **Quaternion convention
checked: npz `body_quat_w` is already `wxyz`** (|w|≈0.91 dominant) = MuJoCo convention, so the
remap copies quats as-is, NO `xyzw↔wxyz` flip needed. (SONIC's own loader does the flip upstream
at retarget time; the saved npz are post-flip.)

### MVP target = a real trained checkpoint (not a fresh clip)
`G1-MultiMotionV2-Streaming-Flat-v0/2026-06-03_12-44-16/model_39999.pt` on A800
(`<cluster-logs>/`). MLP policy, flat/G1, 100style motions,
anchor `torso_link`, reward weights `0.5/0.5/1/1/1/1/-0.1/-10` — **the same reward/obs functions
mjlab's tracking template implements**. (The `PMT-*` log dirs are empty failed launches; real
ckpts are under legacy run names.) This is a *streaming multi-clip* task, so loading the ckpt for
end-to-end tracking is **Phase D/E**; the Phase A smoke uses mjlab's stock single-clip task on one
of these npz clips.

---

## Architecture: where the backend boundary lives

PMT's `pmt_tasks/builder.py` **already** splits the world at exactly the right place:
- a **pure-OmegaConf, backend-neutral** pass — `load_paths` / `load_checkpoints` /
  `_compose_defaults` / `build_task_config()` (no isaaclab/omni imports);
- **lazy `_build_*_env()` emitters** that import isaaclab and populate `@configclass` instances.

We formalize that seam into a `backends/` package and dispatch by a `backend:` config axis.

```
pmt_tasks/
  builder.py            # SHARED: config composition + backend dispatch (keep build_task_config neutral)
  backends/
    __init__.py         # _ENV_BUILDERS_BY_BACKEND, _AGENT_BUILDERS_BY_BACKEND
    common.py           # only truly backend-neutral emit helpers
    isaaclab.py         # current _build_*_env / agent emitters move here verbatim
    mjlab.py            # PMT cfg → mjlab ManagerBasedRlEnvCfg / rl_cfg (uses mjlab tracking template)
  mdp/
    _backend.py         # NEW: RobotView field-alias shim + SceneEntityCfg resolver
    events.py           # backend-neutral (math-only) DR terms; USD-free at import
    events_isaaclab.py  # NEW: USD/pxr/omni-spawning terms split out of events.py
    commands.py, rewards.py, terminations.py, observations.py   # SHARED via _backend shim
  registry_gym.py       # backend-aware entry_point factory (thin, NOT the main boundary)
configs/
  backend/{isaaclab,mjlab}.yaml   # NEW axis; default isaaclab
```

Three coordinated switch points, one source of truth:

1. **Source of truth = `backend:` axis in YAML.** Add `configs/backend/{isaaclab,mjlab}.yaml`;
   add `backend` to `_AXES` in `builder.py` defaulting to `isaaclab` when absent (so the ~30
   existing tasks are untouched). Resolved at `cfg["backend"]["name"]`.
2. **Emit dispatch = split registry, keyed by backend.** Replace the single `_ENV_BUILDERS`
   with `_ISAAC_ENV_BUILDERS` / `_MJLAB_ENV_BUILDERS`. `build_env_cfg(..., backend=...)` and
   `build_agent_cfg(..., backend=...)` pick the table; a missing entry raises a clear
   `NotImplementedError(f"task '{stem}' has no {backend} env builder")`. **This is the
   incremental seam** — mjlab arrives one task at a time; isaaclab keeps working untouched.
3. **gym register = backend-aware `entry_point`.** `registry_gym.py` peeks
   `build_task_config(stem)["backend"]["name"]` (cheap, pure OmegaConf) and registers
   `isaaclab.envs:ManagerBasedRLEnv` vs mjlab's `ManagerBasedRlEnv`. Use distinct ids / suffix
   (e.g. `PMT-...-Mjlab-v0`) so both can register in one process — never two entry_points under
   one id.

### The field-alias shim (`pmt_tasks/mdp/_backend.py`)
A `RobotView` adapter wraps the backend asset and exposes PMT's canonical names, mapping to the
underlying field. `commands.py` changes `self.robot.data.body_pos_w` →
`self._view.body_pos_w` (~8 refs in one file, plus a few in rewards/terminations). **One file
owns the mapping; every term function keeps its current name.** `SceneEntityCfg` resolves through
the same shim (`asset_name`/`entity_name`).

> Optional cleaner long-term variant: adopt mjlab's `body_link_*` as the canonical PMT names and
> alias them on the isaaclab side instead. Decide before porting many terms; either direction the
> rule is identical (terms read canonical names through one adapter).

### Keeping USD out of the mjlab path (critical)
`pmt_tasks/mdp/events.py` imports `omni.usd` / `pxr` / `isaaclab.sim` / `isaaclab.terrains`
**at module load**, and `mdp/__init__.py` re-exports it — so `import pmt_tasks.mdp` would drag
USD into the mjlab process and crash it. Fix = **split the module** (not just lazy-import):
- Move USD/pxr-dependent terms (omniretarget mesh-spawn events, trimesh plane spawn) →
  `pmt_tasks/mdp/events_isaaclab.py`, imported only by the isaaclab emit helpers.
- Backend-neutral DR/push terms stay in `events.py`, math via the `_backend` shim.
- mjlab's DR comes from `mjlab.envs.mdp.dr` (`body_com_offset`, `encoder_bias`, `geom_friction`)
  and `mjlab.envs.mdp.push_by_setting_velocity` — PMT does not reimplement these.
- **CI smoke test:** in a mjlab-only env, `python -c "import pmt_tasks.mdp"` must succeed.

---

## Shared vs forked (keeps the later merge cheap)

**Shared (single copy, both backends):**
- `configs/**` YAML axes — `backend` is one more axis. (Asset-naming *values* gain backend-tagged
  variants where they reference USD-vs-MJCF; the axis *structure* is shared.)
- `build_task_config()` + the whole SELECT→DERIVE→VALIDATE pass, `derive.py`,
  `motion_tracking_rl/compat.py`, `registry.py` — pure Python/OmegaConf.
- Motion NPZ schema + path resolution + **sampler math**. ⚠️ **Review correction:** only
  `bin_based_sampler.py` is actually import-free today; `multi_motion_command.py` (which holds
  `MotionDataStore` + `UniformSampler/AdaptiveSampler`) **inherits isaaclab `CommandTerm` and
  imports `isaaclab.assets`/`isaaclab.managers` at line 10-12**. So the merge-cheap move is to
  **split a pure-torch sampler/store core (tensors in → indices out) from the backend `CommandTerm`
  wrapper** — not to assume it's already shareable. `bin_based_sampler.py` is the reference shape.
- Term math (`rewards/terminations/observations`, tracking `commands` logic) — via `_backend` shim.
- Runner-level intent: experiment names, PPO/BPO/distill settings, max iters, teacher checkpoints.

**Forked / backend-specific (two implementations behind dispatch):**
- Env-cfg emit: `backends/isaaclab.py` (`@configclass`) vs `backends/mjlab.py`
  (`make_*()`→`ManagerBasedRlEnvCfg`, mirroring `mjlab/tasks/tracking`).
- Robot assets: `pmt_tasks/robots/g1.py` (USD/URDF `ArticulationCfg`) vs mjlab
  `get_g1_robot_cfg()` (MJCF). **Actuator models differ** (Isaac ImplicitActuator PD gains vs
  mjlab reflected-inertia electric actuator) — not numerically equal; explicit decision needed.
- Sensors (isaaclab `ContactSensorCfg`/`RayCaster`/`Camera` vs mjlab `ContactSensorCfg`/builtin IMU).
- Events/DR (USD-spawning omniretarget = isaaclab-only).
- Terrain: PMT custom USD mesh (stepping-stone/big_map) vs mjlab plane/MJCF. **Biggest gap** —
  mjlab tracking ships only plane; heightfield/mesh + raycast parity is the hard, possibly-blocking
  item for terrain & perceptive/vision tasks.
- VecEnv wrapper: normalize `motion_tracking_rl` runners to drive either
  `isaaclab.envs:ManagerBasedRLEnv` or `mjlab.envs:ManagerBasedRlEnv`.

---

## Execution plan (incremental — isaaclab stays green throughout)

> **Phase reordering (review #1):** the stock-mjlab smoke moves to the FRONT. Proving a real PMT
> clip loads/steps in mjlab must happen *before* any shared-spine refactor, so we never refactor
> against an unproven assumption. The semantic contract for G1-flat is already **verified green**
> (see above); Phase A converts that to a runtime assertion.

### Phase A — Branch + stock-mjlab smoke on a real PMT clip (PROVE FIRST)
- Branch `mjlab-port` off `master`. Consume mjlab as an **editable dependency** of
  `<mjlab-repo>` (do not vendor).
- Pull one trained-family clip (e.g. `…/100style/robot_100style/Aeroplane/Aeroplane_BR.npz`).
- Run mjlab's stock `Mjlab-Tracking-Flat-Unitree-G1` with
  `--env.commands.motion.motion-file <clip>.npz`; reset + step; **assert finite command/body
  errors** and visually confirm the reference body order isn't scrambled (Risk #1 runtime check).
- Write the **NPZ validator** here (keys/dtype/shape/fps==env-rate/quaternion/joint+body
  names+order) and a **golden parity harness** stub.

### Phase B — Import hygiene + data/command shims (no behavior change)
- Split `events.py` → `events_isaaclab.py`; make `import pmt_tasks.mdp` USD-free + CI smoke test
  (`python -c "import pmt_tasks.mdp"` in a mjlab-only env). Also neutralize the eager
  `isaaclab` imports in `commands.py:10`, `observations.py:6`, `terminations.py:12`, and the
  isaaclab-gated re-exports in `mdp/__init__.py` (review: these block the shim, not just events).
- Add `mdp/_backend.py`: `RobotView` field-alias (incl. `applied_torque→qfrc_actuator`) +
  config-time `SceneEntityCfg` `asset_name→name` adapter + a `MotionCommandView` contract. Route
  raw reads through it. Verify isaaclab parity (identical results) + golden parity tests.

### Phase C — Backend dispatch + mjlab emitter (isaaclab path identical)
- Add `backend` config axis (default `isaaclab`); move isaaclab emitters to `backends/isaaclab.py`;
  split registry tables (`_ISAAC_/_MJLAB_ENV_BUILDERS`); backend-aware `registry_gym` entry_point.
- Add `backends/mjlab.py` populating mjlab's `make_tracking_env_cfg()` from resolved OmegaConf
  (reward weights, std, body_names, anchor, decimation/dt); register `PMT-Tracking-Flat-G1-Mjlab-v0`.
  Use mjlab's own tracking task as a **correctness oracle** (same lineage).

### Phase D — Load the trained PMT ckpt, measure tracking in mjlab  ✅ EXECUTED
Loaded `G1-MultiMotionV2-Streaming-Flat …/model_39999.pt` (ActorCritic MLP 512/256/128,
actor obs 286, act 29) into mjlab via `scripts/mjlab_eval_pmt_ckpt.py`.

**Findings (rich diagnosis, Codex-verified):**
- Actor obs 286 == mjlab's **critic** layout exactly (command58 + anchor_pos3 + anchor_ori6 +
  body_pos42 + body_ori84 + base_lin3 + base_ang3 + joint_pos29 + joint_vel29 + actions29).
  Build the mjlab env with actor group = critic terms; weights map directly.
- **Joint-order bridge required at the policy boundary** (not just clips): PMT policy trained in
  Isaac BFS joint order; mjlab obs/action in MJCF. Permute the joint-indexed obs slices
  (`[0:29],[29:58],[199:228],[228:257],[257:286]`) MJCF→BFS into the net, and the action BFS→MJCF
  out. (body/anchor/vel slices are name-resolved / frame quantities → order-safe.)
- Verified IDENTICAL Isaac↔mjlab: action scale (per-joint), action offset (`use_default_offset`,
  same default pose), obs term order, `body_ori`/`anchor_ori` 6D encoding
  (`matrix_from_quat[...,:2]`), `command` = absolute ref joint_pos+joint_vel.
- **Residual sim2sim gaps** (the real, expected ones): PD stiffness for **ankle + waist_pitch/roll
  is 2× in mjlab** (28.5 vs PMT 14.25); `base_lin_vel` is a MuJoCo velocimeter at an offset IMU
  site vs Isaac root-frame velocity; MuJoCo vs PhysX contact/actuator model.

**Result: PARTIAL ZERO-SHOT TRANSFER (a genuine success for cross-sim).** The policy keeps the
robot **upright for 250 steps / 5 s without falling** (pelvis-z ~0.76–0.80 m vs 0.79 standing),
positive mean reward, body-relative pose tracked (~0.07 m re-anchored body-pos error). The global
anchor wanders (in-place dance drifts in world xy) — consistent, NOT a fall. The earlier
"joint err 1 rad" headline was misleading: 0.586 rad L2 over 29 joints ≈ 0.11 rad RMS, and mjlab's
body-pos metric is **re-anchored to the live robot** (yaw+xy), so low local error coexists with
global drift by construction.

**Conclusion (matches the plan's upfront thesis):** the port pipeline is structurally correct and
verified end-to-end; exact zero-shot transfer is bounded by irreducible two-physics gaps →
**fine-tune / retrain a short pass in mjlab** for deployment-grade tracking. Concrete cheap win
available: align ankle/waist PD gains in a PMT-specific mjlab G1 cfg.

### Phase E — Multi-clip command + samplers, then per-task tests
- mjlab's `MotionCommand` is **single-clip**. Port PMT's `MotionDataStore` + samplers by
  **splitting the pure-torch core from the `CommandTerm` wrapper** (per the shared/forked note),
  rewriting only the sim-write glue (`write_root_state_to_sim`/`write_joint_state_to_sim` —
  signatures align in mjlab's `MotionCommand`). Reconcile PMT's bin sampler vs mjlab's
  `_adaptive_sampling` to ONE (add an equivalence test on identical failure outcomes).

### Phase 6 — VecEnv hardening + harder tasks
- Normalize the VecEnv wrapper contract (obs-dict/extras/`episode_length_buf`).
- Terrain/sensor parity for stepping-stone & perceptive/vision (mjlab heightfield + raycast) —
  scope explicitly as follow-on; may be mjlab-blocked.

---

## Top risks (ranked) & mitigations

1. **Body/joint ordering mismatch** between PMT-authored clips and mjlab's G1 MJCF → motions play
   back garbled with **no error**. *Mitigate first*: load a clip, compare `joint_pos.shape[1]` /
   `body_pos_w.shape[1]` and names vs mjlab G1 `nq`/body list before any training.
2. **Actuator / physics fidelity gap** (PD ImplicitActuator vs reflected-inertia electric model).
   Rewards may need re-tuning; sim2real characteristics change. Decide explicitly: port PMT gains
   or adopt mjlab's.
3. **VecEnv contract drift** — runners assume isaaclab's env semantics; mjlab must be wrapper-
   matched (more than field renames).
4. **Backend imports leaking into the shared path** — `isaaclab`/`omni`/`pxr`/`mjlab` must stay
   lazy and backend-local; shared PMT import path must work with neither sim initialized.
5. **Duplicating YAML task defs for mjlab** → silent drift in reward weights/obs/runner. Add
   backend-specific override fields only for *simulator facts*, never whole task copies.
6. **`mjlab state freshness`** — mjlab `EntityData` needs `sim.forward()` after writes; reset/DR
   ordering can yield stale buffers.

---

## Effort tiers
- **MVP** (G1, single flat tracking task, smoke rollout): **~3–7 days** — G1 MJCF already exists;
  mostly the backend flag + mjlab emitter + field-alias shim.
- **Flat-tracking parity** (multi-motion, obs-group derivation, reward-weight plumbing, checkpoint
  compat, regressions): **~2–4 weeks.**
- **Full PMT parity** (terrain, height-scan/perception, cameras, DR, contact calibration, all task
  variants, distill/BPO/SONIC, training-behavior match): **~1–3+ months.**

---

## Key files (reference)
**PMT**
- `pmt_tasks/builder.py` — backend-neutral `build_task_config()` + emit seam (`_build_*_env`, L247–)
- `pmt_tasks/registry_gym.py` — gym entry_point factory
- `pmt_tasks/mdp/{commands,observations,rewards,terminations,events}.py` — terms (`events.py` = USD hazard)
- `pmt_tasks/mdp/{multi_motion_command,packed_motion_lib,streaming_motion_lib,bin_based_sampler}.py`
- `pmt_tasks/robots/g1.py` — USD/URDF `ArticulationCfg`
- `configs/**` — YAML axes (source of truth); add `configs/backend/`
- `motion_tracking_rl/` — rsl-rl runner / algorithms / networks (sim-agnostic)

**mjlab (porting template + oracle)**
- `src/mjlab/tasks/tracking/tracking_env_cfg.py` — `make_tracking_env_cfg()` reference
- `src/mjlab/tasks/tracking/config/g1/{env_cfgs,rl_cfg}.py`
- `src/mjlab/tasks/tracking/mdp/{commands,rewards,terminations,observations}.py`
- `src/mjlab/asset_zoo/robots/unitree_g1/{g1_constants.py,xmls/g1.xml}` — existing G1 MJCF + actuators
- `src/mjlab/{envs,entity,managers,sim,terrains,sensor}/` — API equivalents
