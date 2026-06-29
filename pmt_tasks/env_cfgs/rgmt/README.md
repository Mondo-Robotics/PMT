# rgmt/ — paper-faithful transformer tracker

A paper-faithful transformer actor-critic motion tracker on a **flat plane** with **no vision**.
The actor obs intentionally exclude base linear velocity (asymmetric actor/critic); the command
uses bin-adaptive sampling over flat clips.

## Task

| gym id | task yaml | env cfg | network |
| --- | --- | --- | --- |
| `RGMT-G1-v0` | `configs/task/rgmt.yaml` | `rgmt.py` (`RGMTG1EnvCfg`) | `TransformerActorCritic` (`configs/network/transformer.yaml`) |

- Terrain: flat plane. Motion: `multi_bin_adaptive` over flat lafan1
  (`${paths.MULTIMOTION_FLAT_MOTION}`).
- Obs: `paper_transformer_hist`. Algorithm: `ppo`.

## Train / play

```bash
python scripts/train.py --task RGMT-G1-v0 --num_envs <n> --headless

python scripts/play.py --task RGMT-G1-v0 \
  --resume_path <ckpt.pt> --num_envs 1 --motion_file <npz-or-dir>
```

No pretrained checkpoint ships for this task.
