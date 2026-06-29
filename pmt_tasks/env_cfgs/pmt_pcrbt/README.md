# pmt_pcrbt/ — P-CaRBT (FSQ behavior-token tracker)

**P-CaRBT** (Perception-Conditioned Contact-Aware Residual Behavior Tokenizer) is the released
**discrete FSQ behavior-token** tracker. A residual FSQ tokenizer
(`_ResidualFSQBehaviorTokenizer`) quantizes the future-motion window into discrete behavior tokens
that condition the policy. This is a single-stage PPO pretrain from scratch (`pmt_only_mode`,
`require_pmt_checkpoint=false`) on the **flat plane** — reduced OBS, **no vision group**.

The env cfg is built on the flat multimotion env (`PMTMultiMotionFlatEnvCfg`) and exposes only the
groups the tracker needs:

```
policy:               [policy, proprio]
policy_history:       [proprio_history]   (len-10, unflattened)
future_motion_window: [command_window, motion_anchor_delta_window]
critic:               [critic]
```

## Tasks

| gym id | task yaml | motion dataset |
| --- | --- | --- |
| `PMT-PCaRBT-G1-v0` | `configs/task/pmt_pcrbt.yaml` | flat lafan1 (`${paths.MULTIMOTION_FLAT_MOTION}`) |
| `PMT-PCaRBT-100style-G1-v0` | `configs/task/pmt_pcrbt_100style.yaml` | 1620-clip 100style flat set |

Both share `perceptive_residual_behavior_token.py` (`PMTPCaRBTFlatEnvCfg`), the agent cfg, and
`configs/network/pmt_pcrbt.yaml` (`name: PerceptiveResidualBehaviorTokenTracker`); only the motion
dataset differs. The clip count does not change obs dims, so the 100style task is resumable from a
lafan1 checkpoint. Network class:
`motion_tracking_rl/networks/perceptive_motion/behavior_token_tracker.py`.

## Train / play

```bash
python scripts/train.py --task PMT-PCaRBT-G1-v0 --num_envs <n> --headless

python scripts/play.py --task PMT-PCaRBT-G1-v0 \
  --resume_path <ckpt.pt> --num_envs 1 --motion_file <npz-or-dir>
```

No pretrained checkpoint ships for this task.
