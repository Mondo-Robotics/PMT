# add/ — ADD (adversarial motion imitation)

ADD trains a flat-plane G1 motion-tracking policy with an **adversarial discriminator** instead of
(or alongside) a dense tracking reward. The algorithm is `add_ppo`, which pulls in the
discriminator obs sets (`add_disc_obs` / `add_disc_demo`) and a null task reward (`reward:
add_null`); the policy network is a plain MLP `ActorCritic`.

## Task

| gym id | task yaml | env cfg | network | algorithm |
| --- | --- | --- | --- | --- |
| `PMT-ADD-MultiMotionV2-Flat-v0` | `configs/task/add_multimotion_flat.yaml` | `add_multimotion.py` (`PMTADDMultiMotionEnvCfg`) | `ActorCritic` (MLP) + discriminator | `add_ppo` |

- Terrain: flat plane. Motion: `add_multi` over flat lafan1 clips
  (`${paths.MULTIMOTION_FLAT_MOTION}`).
- Obs: `add_discriminator`. The `add_ppo` algorithm spec adds the discriminator obs/demo groups.

## Train / play

```bash
python scripts/train.py --task PMT-ADD-MultiMotionV2-Flat-v0 --num_envs <n> --headless

python scripts/play.py --task PMT-ADD-MultiMotionV2-Flat-v0 \
  --resume_path <ckpt.pt> --num_envs 1 --motion_file <npz-or-dir>
```

No pretrained checkpoint ships for this task.
