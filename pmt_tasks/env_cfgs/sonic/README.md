# sonic/ — SONIC cross-embodiment tracker

SONIC trains a `SonicActorCritic` (dual robot/human encoder + FSQ token + control/motion decoder)
on the flat plane with the V2 multi-motion command and **paired robot/human** motion. The shipped
task config (`sonic_mode: scratch`) trains the encoders **from scratch** — no external ONNX
required — so it runs standalone on local paired robot/human lafan clips.

## Task

| gym id | task yaml | env cfg | network | algorithm |
| --- | --- | --- | --- | --- |
| `PMT-SONIC-G1-MultiMotionV2-Flat-v0` | `configs/task/sonic_multimotion_flat.yaml` | `sonic_multimotion.py` (`PMTSonicMultiMotionFlatEnvCfg`) | `SonicActorCritic` (`configs/network/sonic.yaml`) | `sonic_ppo` (PPO preset) |

- Terrain: flat plane. Motion: `sonic_robot_human` (paired robot + human lafan clips,
  `${paths.MULTIMOTION_FLAT_MOTION}`; the `human_lafan1` sibling auto-loads).
- Obs: `sonic` (SONIC encoder obs groups).

## Train / play

```bash
python scripts/train.py --task PMT-SONIC-G1-MultiMotionV2-Flat-v0 --num_envs <n> --headless

python scripts/play.py --task PMT-SONIC-G1-MultiMotionV2-Flat-v0 \
  --resume_path <ckpt.pt> --num_envs 1 --motion_file <npz-or-dir>
```

No pretrained checkpoint ships for this task. Using the **release SONIC ONNX** encoders
(`PMT_SONIC_ONNX_DIR` → `model_encoder.onnx` + `model_decoder.onnx`) instead of scratch encoders is
described in [`docs/USAGE.md`](../../../docs/USAGE.md).
