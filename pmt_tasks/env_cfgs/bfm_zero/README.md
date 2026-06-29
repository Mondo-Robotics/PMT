# bfm_zero/ — BFM-Zero (FB-CPR-Aux)

BFM-Zero is a behavior-foundation-model RL core (FB-CPR-Aux) vendored inside PMT
(`motion_tracking_rl/bfm_zero/`). It does **not** use the rsl_rl agent cfg or `scripts/train.py`;
instead it has its **own runner** and a dedicated launcher at `scripts/bfm_zero/train.py`.

The env cfg is registered with **only** an `env_cfg_entry_point` (no rsl_rl agent cfg), so the
BFM-Zero launcher can build it through Isaac Lab's normal `gym.make` path.

## Task

| gym id | env cfg | launcher |
| --- | --- | --- |
| `BFM-Zero-Flat-MultiMotionV2-G1-v0` | `bfm_zero.py` (`BFMZeroG1FlatMultiMotionV2EnvCfg`, a flat-plane subclass of `PMTMultiMotionFlatEnvCfg`) | `scripts/bfm_zero/train.py` |

There is **no `configs/task/*.yaml`** for BFM-Zero — it is registered directly in
`pmt_tasks/registry_gym.py`.

## Train

```bash
python scripts/bfm_zero/train.py --task BFM-Zero-Flat-MultiMotionV2-G1-v0 \
  --agent_preset smoke --num_envs 8 --headless --total_env_steps 4096 --num_seed_steps 64
```

(Drop `--agent_preset smoke` and raise `--total_env_steps` / `--num_envs` for a real run.)

## License note

The vendored BFM-Zero code (`motion_tracking_rl/bfm_zero/_vendor/`,
`motion_tracking_rl/networks/ode_solver.py`) is **CC BY-NC 4.0 (NonCommercial)**. See the
top-level [README §14](../../../README.md#14-license).
