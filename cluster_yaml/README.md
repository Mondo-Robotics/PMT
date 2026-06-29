# PMT cluster job templates

These YAML files are `md_rl_kit` job-submission templates for launching PMT
training on a cluster. They preserve the expected md_rl structure: each job
defines the shared workspace, Python paths, environment variables, and one or
more `md_rl.train` tasks that launch `scripts/submit/train_pmt.py`.

The templates intentionally use placeholders for cluster-specific paths. Replace
these before submitting a job:

| Placeholder | Meaning |
|-------------|---------|
| `<CLUSTER_WORKSPACE>` | Cluster directory that contains the PMT checkout. The YAMLs append `/PMT`. |
| `<WHOLE_BODY_TRACKING_REPO>` | Cluster checkout of the `whole_body_tracking` repo, used for IsaacLab assets. |
| `<MD_RL_LOG_ROOT>` | Root directory for md_rl training logs and reference checkpoints. |
| `<CLUSTER_HOST>` | SSH host or scheduler login node used in example commands. |

All YAMLs set `PMT_PROFILE: "cluster"`. The cluster profile in
`configs/paths.yaml` is controlled by the same PMT data environment variables
documented in `docs/USAGE.md`; set these in your shell, scheduler environment,
or YAML `os_env_vars` as needed for your data layout:

`PMT_DATA_ROOT`, `PMT_MOTION_ROOT`, `PMT_DATASET_ROOT`,
`PMT_TERRAIN_MOTION_ROOT`, `PMT_SONIC_ROOT`, `PMT_CKPT_ROOT`,
`PMT_MULTIMOTION_FLAT_MOTION`, `PMT_BACKFLIP_MOTION`,
`PMT_SONIC_ONNX_DIR`, and `PMT_REPO_ROOT`.

## GPU plan

`--num_envs` is the per-GPU environment count for these md_rl jobs.
`--distributed true` is set whenever `num_gpus > 1`. Jobs are normally tested
sequentially, so per-job GPU counts do not need to sum to the full cluster size.

| YAML | gym task | gpus | num_envs (per-gpu) | total envs | data (cluster-resolved) | status |
|------|----------|-----:|-------------------:|-----------:|-------------------------|--------|
| `smoke_all.yaml` | PMT-WalkDanceBigMap-G1-v0 | 1 | 8 | 8 | big_map.stl + `motions/terrain_mocaphouse/walk_dance1sub2start/optimized` | smoke (max_iter=2) |
| `walk_dance_bigmap.yaml` | PMT-WalkDanceBigMap-G1-v0 | 8 | 4096 | 32768 | big_map.stl + `motions/terrain_mocaphouse/walk_dance1sub2start/optimized` | REAL (class-A) |
| `terrain_flat_mix.yaml` | PMT-TerrainFlatMix-G1-v0 | 8 | 4096 | 32768 | `g1_29dof_big_map_with_flat.stl` + `motions/terrain_mocaphouse/walk_dance1sub1start/optimized` (terrain) + `sonic/lafan1/robot_lafan1` (flat) | REAL (class-A+B) |
| `multimotionv2_flat.yaml` | PMT-G1-MultiMotionV2-Flat-v0 | 4 | 4096 | 16384 | `sonic/lafan1/robot_lafan1` (40 paired clips) | REAL (class-B) |
| `pmt_stepping_stone.yaml` | PMT-SteppingStone-G1-v0 | 8 | 4096 | 32768 | `positive_stepping_with_stairs.stl` + `motions/terrain_positive_stepping_stone_with_stairs/walk1_subject1_stair/optimized` | DATA-GATED (class-A) |
| `perceptive_motion_token_tracker.yaml` | PMT-PerceptiveMotionTokenTracker-G1-v0 | 4 | 4096 | 16384 | `positive_stepping_with_stairs.stl` + `motions/terrain_positive_stepping_stone_with_stairs/walk1_subject1_stair/optimized` | DATA-GATED (class-A) |

## Job classes

- REAL training (`walk_dance_bigmap`, `terrain_flat_mix`, `multimotionv2_flat`):
  both mesh and motion directories must resolve to paths that exist on your
  cluster. Submit after a smoke job passes.
- Smoke-only (`smoke_all`): proves the PMT entry point, cluster path profile,
  and Ray plumbing on 1 GPU with `--num_envs 8 --max_iterations 2`. Run this
  first.
- DATA-GATED (`pmt_stepping_stone`, `perceptive_motion_token_tracker`): the env
  constructs and paths resolve cleanly, but the stepping-stone optimized clip
  directory must be present before submission. Upload or repoint the clips if
  the job fails at motion discovery.

## Data-gated / construct-only tasks without a job YAML

These compose under `PMT_PROFILE=cluster`, but their cluster motion directories
may need to be repointed to real data or uploaded before they can train:

- `distill_stepping_stone`, `distill_stepping_stone_latent_anchor`, and
  `ppofinetune_*`: require stepping-stone clips and, for distill/finetune
  variants, a teacher checkpoint under `PMT_CKPT_ROOT`.
- `sonic_multimotion_flat`: should point its motion YAML to the cluster SONIC
  paired robot/human clip directories.
- `backflip` and `add_multimotion_flat`: require the relevant single-motion or
  debug motion clips under the configured PMT motion roots.

## Submit

After replacing placeholders and activating the md_rl environment on your
cluster:

```bash
ssh <CLUSTER_HOST> 'cd <CLUSTER_WORKSPACE>/PMT && md_rl_kit submit cluster_yaml/smoke_all.yaml'
```

Run a smoke YAML first; once it passes, submit the REAL-training YAMLs.
