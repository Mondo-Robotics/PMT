"""Phase E (MJLAB_BACKEND_PLAN.md): support matrix across ALL PMT task families.

For every task config, assert the backend-neutral `build_task_config` resolves (this works
in any env), and classify mjlab-backend support:

  * MJLAB_SUPPORTED   — G1 flat tracking family; mjlab emitter wired + tested.
  * MJLAB_OUT_OF_SCOPE — needs mjlab features not yet present (custom USD terrain, height-scan
    / vision, distillation runner). Documented, asserted to RAISE a clear
    NotImplementedError on the mjlab path rather than silently mis-emit.

The actual env build for supported tasks runs only when mjlab is installed (mjlab venv).
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_MJLAB = importlib.util.find_spec("mjlab") is not None

ALL_TASKS = [
    "add_multimotion_flat",
    "backflip",
    "bpo_multimotionv2_flat",
    "cartwheel_bigmap",
    "distill_stepping_stone_latent_anchor",
    "distill_stepping_stone",
    "multimotionv2_100style_flat",
    "multimotionv2_adaptive_flat",
    "multimotionv2_flat",
    "multimotionv2_streaming_100style",
    "multimotionv2_streaming_flat",
    "multimotionv2_uniform_flat",
    "perceptive_motion_token_tracker",
    "pmt_stepping_stone",
    "sonic_multimotion_flat",
    "terrain_flat_mix",
    "walk_dance_bigmap",
]

# mjlab-supported today (flat G1 tracking family — see backends/mjlab._MJLAB_ENV_BUILDERS).
MJLAB_SUPPORTED = {
    "multimotionv2_flat",
    "multimotionv2_uniform_flat",
    "multimotionv2_adaptive_flat",
    "multimotionv2_streaming_flat",
    "multimotionv2_streaming_100style",
    "multimotionv2_100style_flat",
    "sonic_multimotion_flat",
}


@pytest.mark.parametrize("task", ALL_TASKS)
def test_task_config_resolves(task):
    """Backend-neutral config must resolve for every task (no sim needed)."""
    from pmt_tasks.builder import build_task_config

    cfg = build_task_config(task)
    assert "motion" in cfg


@pytest.mark.parametrize("task", sorted(set(ALL_TASKS) - MJLAB_SUPPORTED))
def test_unsupported_tasks_raise_on_mjlab(task):
    """Out-of-scope tasks must FAIL LOUD on the mjlab backend, not silently mis-emit."""
    from pmt_tasks.builder import build_env_cfg

    with pytest.raises(NotImplementedError):
        build_env_cfg(task, backend="mjlab")


@pytest.mark.skipif(not _HAS_MJLAB, reason="needs mjlab")
@pytest.mark.parametrize("task", sorted(MJLAB_SUPPORTED))
def test_supported_tasks_emit_mjlab_env(task):
    """Every supported flat task must emit a valid mjlab env cfg (build, no step)."""
    from pmt_tasks.builder import build_env_cfg

    # NOTE: relies on the configured motion dir being present; if the dataset path is
    # absent locally the emitter raises FileNotFoundError, which we surface clearly.
    try:
        env_cfg = build_env_cfg(task, backend="mjlab")
    except FileNotFoundError as e:
        pytest.skip(f"motion data not present locally: {e}")
    # mjlab ManagerBasedRlEnvCfg shape checks
    assert hasattr(env_cfg, "decimation")
    assert "motion" in env_cfg.commands
    assert "motion_global_root_pos" in env_cfg.rewards
