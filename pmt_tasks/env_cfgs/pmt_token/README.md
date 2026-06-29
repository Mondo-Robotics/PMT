# pmt_token/ — perceptive motion-token tracker

The continuous (mean-pool) **behavior-token** tracker: a `_MotionTokenizer` turns the
future-motion window into continuous behavior tokens that condition the policy decoder. This task
PPO-pretrains the tokenizer + decoder **from scratch** (`pmt_only_mode`,
`require_pmt_checkpoint=false`) on the big_map terrain.

## Task

| gym id | task yaml | env cfg | network |
| --- | --- | --- | --- |
| `PMT-PerceptiveMotionTokenTracker-G1-v0` | `configs/task/perceptive_motion_token_tracker.yaml` | `perceptive_motion_token.py` (`PMTPerceptiveMotionTokenTrackerEnvCfg`) | `PerceptiveMotionTokenTracker` (`configs/network/perceptive_motion_token_tracker.yaml`) |

- Terrain: big_map mesh, trained on terrain-anchored walk_dance optimized clips.
- Obs: `perceptive_motion_token` (token/window groups + a flat `vision` height-scan group).
- Network class lives in `motion_tracking_rl/networks/perceptive_motion/token_tracker.py`.

## Train / play

```bash
python scripts/train.py --task PMT-PerceptiveMotionTokenTracker-G1-v0 --num_envs <n> --headless

python scripts/play.py --task PMT-PerceptiveMotionTokenTracker-G1-v0 \
  --resume_path <ckpt.pt> --num_envs 1 --motion_file <npz-or-dir>
```

No pretrained checkpoint ships for this task. The discrete-FSQ variant (P-CaRBT) is in
[`../pmt_pcrbt/`](../pmt_pcrbt/README.md).
