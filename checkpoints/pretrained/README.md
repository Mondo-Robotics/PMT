# Pretrained PMT checkpoints

Pretrained G1 humanoid motion-tracking policies, ready to load and roll out. The `.pt` and
`.onnx` files are tracked with **git-lfs**:

```bash
git lfs install      # once
git lfs pull         # fetch the binaries after cloning
```

These are the validated post-refactor runs (trained 2026-06-25…28). `reward` = max
`Train/mean_reward`; all load into the current `motion_tracking_rl` network classes.

| File | Task / gym id | Network | iter | reward |
| --- | --- | --- | --- | --- |
| `multimotionv2_flat.pt` | `PMT-G1-MultiMotionV2-Flat-v0` | ActorCritic (MLP) | 20k | 40.9 |
| `multimotionv2_streaming_flat.pt` | `PMT-G1-MultiMotionV2-Streaming-Flat-v0` | ActorCritic (MLP) | 20k | 39.5 |
| `multimotionv2_adaptive_flat.pt` | `PMT-G1-MultiMotionV2-Adaptive-Flat-v0` | ActorCritic (MLP) | 20k | 46.4 |
| `multimotionv2_uniform_flat.pt` | `PMT-G1-MultiMotionV2-Uniform-Flat-v0` | ActorCritic (MLP) | 20k | 46.4 |
| `bpo_multimotionv2_flat.pt` | `PMT-G1-BPO-MultiMotionV2-Flat-v0` | ActorCritic (MLP) | 20k | 33.7 |
| `fpo_plus_singleclip_flat.pt` | `PMT-G1-FPOPlus-SingleClip-Flat-v0` | DiffusionActorCritic | 3.5k | 39.9 |
| `add_multimotion_flat.pt` | `PMT-ADD-MultiMotionV2-Flat-v0` | ActorCritic + discriminator | 20k | 18.0¹ |
| `rgmt_flat.pt` | `RGMT-G1-v0` | TransformerActorCritic | 20k | 33.5 |
| `perceptive_motion_token_tracker.pt` | `PMT-PerceptiveMotionTokenTracker-G1-v0` | PerceptiveMotionTokenTracker | 20k | 80.9 |
| `pcrbt_100style.pt` | `PMT-PCaRBT-100style-G1-v0` | PerceptiveResidualBehaviorTokenTracker | 30k | 62.4 |
| `walkdance_bigmap_teacher.pt` | `PMT-WalkDanceBigMap-G1-v0` | TransformerActorCritic | 20k | 86.6 |
| `ppoft_vision_bigmap_walkdance.pt` | `PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0` | VisionTransformerActorCritic | 3.2k | 71.9 |
| `sonic_onnx/` | `PMT-SONIC-G1-MultiMotionV2-Flat-v0` | SONIC (official ONNX) | — | — |

¹ ADD's `Train/mean_reward` is intentionally small (adversarial imitation); the run is full-length and healthy.

`ppoft_vision_bigmap_walkdance.pt` is the **stage-3 PPO-finetuned vision student** of the
teacher → distill → finetune pipeline on the big_map walk_dance clips (warm-started from the
distilled vision student, then PPO-sharpened). It pairs with the `walkdance_bigmap_teacher.pt`
teacher: the teacher is the blind transformer, this is the deployable vision-conditioned policy.

**SONIC** ships as the official released ONNX (`sonic_onnx/model_encoder.onnx` +
`model_decoder.onnx` + `observation_config_sonic_release.yaml`) — the canonical deploy artifact,
loadable into `SonicActorCritic` via the ONNX repack contract (see `scripts/sonic_test/`).

> **BFM-Zero:** no pretrained checkpoint is shipped — the FB-CPR runner did not produce a
> released policy checkpoint. Train it via `scripts/bfm_zero/train.py`.

## Load + roll out

```bash
# rollout any policy on its task (play.py uses --resume_path)
python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --resume_path checkpoints/pretrained/multimotionv2_flat.pt --num_envs 16

python scripts/play.py --task PMT-G1-MultiMotionV2-Streaming-Flat-v0 \
  --resume_path checkpoints/pretrained/multimotionv2_streaming_flat.pt --num_envs 16
python scripts/play.py --task PMT-G1-MultiMotionV2-Adaptive-Flat-v0 \
  --resume_path checkpoints/pretrained/multimotionv2_adaptive_flat.pt --num_envs 16
python scripts/play.py --task PMT-G1-MultiMotionV2-Uniform-Flat-v0 \
  --resume_path checkpoints/pretrained/multimotionv2_uniform_flat.pt --num_envs 16

# warm-start training from a pretrained policy
python scripts/train.py --task PMT-WalkDanceBigMap-G1-v0 \
  --resume --resume_path checkpoints/pretrained/walkdance_bigmap_teacher.pt --num_envs <n> --headless
```

## Provenance
Trained on the internal A800 cluster (Isaac Lab / rsl_rl), validation runs of 2026-06-25…28,
final-iteration checkpoints. All verified to load into the open-source `motion_tracking_rl`
classes (the rgmt→pmt + core rename preserved state_dict module names). Exact closed-loop
transfer to a different physics engine may need a short fine-tune (see `docs/MJLAB_BACKEND_PLAN.md`).
