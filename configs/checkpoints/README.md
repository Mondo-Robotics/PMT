# Named checkpoints (PMT plan §5)

The current repo selects teacher/base checkpoints by a **timestamped run dir +
glob-latest mtime** (`rsl_rl_transformer_ppo_cfg.py:100-121`). A path *profile*
(local/cluster roots) cannot express this — it is per-task data. So named prior
runs live here, one YAML each.

## Schema

```yaml
# configs/checkpoints/<name>.yaml
run_dir:    ${paths.CKPT_ROOT}/<experiment>/<timestamped_run>
checkpoint: latest          # or an explicit file like model_10800.pt
```

- `run_dir` uses `${paths.CKPT_ROOT}` + a **relative** subpath (profile-portable).
- Current behavior (`load_checkpoints()` in `pmt_tasks/builder.py`): an explicit
  `checkpoint: model_X.pt` is joined onto `run_dir`; `checkpoint: latest` resolves
  to `run_dir` **as a string only** — there is no filesystem glob/mtime scan yet.
- PLANNED (plan §5, not yet implemented): `latest` should glob-latest `model_*.pt`
  by mtime, and resolution should be **fail-loud** (raise at build time if
  `run_dir`/the resolved file is missing) so a clone is not stranded with a
  cryptic file-not-found.

## Referencing from a task

```yaml
# in a task/*.yaml
network:
  teacher_ckpt: ${checkpoints.ss_teacher}
```

The builder injects the `checkpoints` namespace (loaded from this dir) so the
interpolation resolves to the named run's resolved path.
