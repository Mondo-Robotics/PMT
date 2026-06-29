# Pretrained PMT checkpoints

Pretrained G1 humanoid motion-tracking policies, ready to load and roll out. These `.pt`
files are tracked with **git-lfs** (run `git lfs install` once, then `git lfs pull`).

| File | Task / gym id | Network | Iter | Notes |
| --- | --- | --- | --- | --- |
| `multimotionv2_flat.pt` | `PMT-G1-MultiMotionV2-Flat-v0` | MLP actor-critic | 39999 | Flagship flat motion-tracking policy (lafan1). ~8 MB. |
| `backflip_teacher.pt` | `PMT-Backflip-G1-v0` (big_map) | TransformerActorCritic | 16000 | Backflip teacher on the big_map mesh. |
| `cartwheel_teacher.pt` | `PMT-CartwheelBigMap-G1-v0` | TransformerActorCritic | 17400 | Cartwheel/kungfu teacher on the big_map mesh. |

Each checkpoint is an rsl_rl-style dict with `model_state_dict`, `optimizer_state_dict`,
`iter`, `infos`, and `policy_metadata`.

## Load + roll out

```bash
# flat flagship (MLP) — rollout / play
python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --resume_path checkpoints/pretrained/multimotionv2_flat.pt --num_envs 16

# resume training from a pretrained policy
python scripts/train.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --resume --resume_path checkpoints/pretrained/multimotionv2_flat.pt --num_envs <n> --headless
```

(The backflip/cartwheel teachers load the same way with their gym ids; they need the big_map
mesh + the corresponding clips — see the task READMEs.)

## Provenance
Trained on the internal A800 cluster (Isaac Lab / rsl_rl). The flat policy is the
`G1-MultiMotionV2-Streaming-Flat-v0` run (2026-06-03), 40k iters; the teachers are the
`g1_rgmt_*_teacher` big_map runs. Loading is engine-agnostic at the weights level; exact
closed-loop transfer to a different physics engine may need a short fine-tune (see
`docs/MJLAB_BACKEND_PLAN.md`).
