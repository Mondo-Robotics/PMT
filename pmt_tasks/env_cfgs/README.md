# PMT env_cfgs — environment configs per task family

This directory holds the per-task **environment configs** (scene + obs groups + commands +
rewards + terminations) for every released PMT task. Each task's env cfg pairs with an
**agent cfg** (`pmt_tasks/agent_cfgs/`) and a **policy network**
(`motion_tracking_rl/networks/`), wired together by `pmt_tasks/builder.py` and registered as a
gym id in `pmt_tasks/registry_gym.py`.

PMT composes a task from independent axes (`robot` / `terrain` / `motion` / `obs` / `reward` /
`network` / `algorithm` / `stage`); the builder selects an env cfg from this directory, an agent
cfg, and a network class. See the top-level [`README.md`](../../README.md) for the axis taxonomy.

---

## The wiring (5 touch-points)

1. `configs/task/<stem>.yaml` — the `defaults:` axis list + task-local overrides.
2. `configs/network/<name>.yaml` — selects the registered network class.
3. `pmt_tasks/env_cfgs/<family>/...` — the **env cfg** (scene + obs groups + commands), built
   via a `_build_*` function in `pmt_tasks/builder.py` (`_ENV_BUILDERS`).
4. `pmt_tasks/agent_cfgs/<name>.py` — the runner/agent cfg (obs_groups map, policy knobs, PPO +
   aux coefficients), built via `build_agent_cfg` in `pmt_tasks/builder.py`.
5. `pmt_tasks/registry_gym.py` (`_TASK_ID_MAP`) — maps the task stem to a friendly gym id
   (otherwise the `PMT-<stem>-v0` fallback id is used).

Train any task with `scripts/train.py --task <gym-id> --num_envs <n> --headless`; play a
checkpoint with `scripts/play.py --task <gym-id> --resume_path <ckpt.pt> --num_envs <n>`.

---

## Released families

| Subdir / file | Tasks (gym ids) | Network class | Scene | README |
| --- | --- | --- | --- | --- |
| `pmt/` | `PMT-SteppingStone-G1-v0`, `PMT-WalkDanceBigMap-G1-v0`, `PMT-CartwheelBigMap-G1-v0`, `PMT-Backflip-G1-v0`, `PMT-TerrainFlatMix-G1-v0`, `PMT-AdaptiveSampling-G1-v0`(+`-Baseline-`), `PMT-Distill-SteppingStone-G1-v0`, `PMT-Distill-SteppingStone-LatentAnchor-G1-v0`, `PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0` | `TransformerActorCritic` / `StudentTeacher` / `VisionStudentTeacher` | terrain mesh (stepping-stone / big_map) ± height scanner | [pmt/README.md](pmt/README.md) |
| `pmt_token/` | `PMT-PerceptiveMotionTokenTracker-G1-v0` | `PerceptiveMotionTokenTracker` | big_map mesh | [pmt_token/README.md](pmt_token/README.md) |
| `pmt_pcrbt/` | `PMT-PCaRBT-G1-v0`, `PMT-PCaRBT-100style-G1-v0` | `PerceptiveResidualBehaviorTokenTracker` | flat plane (no mesh, no vision) | [pmt_pcrbt/README.md](pmt_pcrbt/README.md) |
| `multi_motion_flat.py` | `PMT-G1-MultiMotionV2-Flat-v0` family (`-Uniform-`, `-Adaptive-`, `-Streaming-`, `-100style-`), `PMT-G1-BPO-MultiMotionV2-Flat-v0`, `PMT-G1-FPOPlus-SingleClip-Flat-v0` | `ActorCritic` (MLP) / diffusion | flat plane | (covered in top README §7) |
| `add/` | `PMT-ADD-MultiMotionV2-Flat-v0` | `ActorCritic` (MLP) + discriminator | flat plane | [add/README.md](add/README.md) |
| `rgmt/` | `RGMT-G1-v0` | `TransformerActorCritic` | flat plane (no vision) | [rgmt/README.md](rgmt/README.md) |
| `sonic/` | `PMT-SONIC-G1-MultiMotionV2-Flat-v0` | `SonicActorCritic` | flat plane | [sonic/README.md](sonic/README.md) |
| `bfm_zero/` | `BFM-Zero-Flat-MultiMotionV2-G1-v0` | FB-CPR-Aux (separate runner) | flat plane | [bfm_zero/README.md](bfm_zero/README.md) |

`multi_motion_flat.py` is the shared flat-plane MLP base — most flat tasks (and the `pmt_pcrbt/`,
`rgmt/`, `bfm_zero/` env cfgs) subclass or reuse it. The flat MLP family is documented in the
top-level [README §7](../../README.md#7-task-catalog), so it has no dedicated subdir README.

---

## The behavior-token tracker family

Two released methods condition the policy on a sequence of **behavior tokens** read from the
future-motion window (instead of a flat command vector). Both are PPO-pretrained from scratch.

| Method | Network class file | Network class |
| --- | --- | --- |
| PMT token tracker (continuous, mean-pool) | `motion_tracking_rl/networks/perceptive_motion/token_tracker.py` | `PerceptiveMotionTokenTracker` |
| **PCRBT** — P-CaRBT (discrete FSQ behavior tokens) | `motion_tracking_rl/networks/perceptive_motion/behavior_token_tracker.py` | `PerceptiveResidualBehaviorTokenTracker` |

**PCRBT (P-CaRBT, Perception-Conditioned Contact-Aware Residual Behavior Tokenizer)** is a released
method: a **residual FSQ behavior tokenizer** (`_ResidualFSQBehaviorTokenizer` in
`token_tracker.py`) discretizes the future-motion window into behavior tokens that condition the
policy. It runs on the flat plane in `pmt_only_mode` (no vision group). Task / network / env cfg:

| Layer | File |
| --- | --- |
| task yaml | `configs/task/pmt_pcrbt.yaml`, `configs/task/pmt_pcrbt_100style.yaml` |
| network axis | `configs/network/pmt_pcrbt.yaml` (`name: PerceptiveResidualBehaviorTokenTracker`) |
| env cfg | `pmt_tasks/env_cfgs/pmt_pcrbt/perceptive_residual_behavior_token.py` (`PMTPCaRBTFlatEnvCfg`) |
| gym ids | `PMT-PCaRBT-G1-v0`, `PMT-PCaRBT-100style-G1-v0` |

The token tracker's flat env exposes only the groups the tracker needs in `pmt_only_mode`:

```
policy:               [policy, proprio]
policy_history:       [proprio_history]
future_motion_window: [command_window, motion_anchor_delta_window]
critic:               [critic]
```

See [`pmt_pcrbt/README.md`](pmt_pcrbt/README.md) and [`pmt_token/README.md`](pmt_token/README.md)
for the per-task commands.

---

## Pretrained models

Three ready-to-roll G1 policies ship under [`checkpoints/pretrained/`](../../checkpoints/pretrained/)
(`multimotionv2_flat.pt`, `backflip_teacher.pt`, `cartwheel_teacher.pt`). Each maps to a gym id in
the table above; load with `scripts/play.py --resume_path <pt>`. See
[`checkpoints/pretrained/README.md`](../../checkpoints/pretrained/README.md) for provenance and
the per-checkpoint task/data requirements.
