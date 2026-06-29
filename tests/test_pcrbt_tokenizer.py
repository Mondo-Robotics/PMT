"""Phase-1 unit tests for the P-CaRBT behavior tokenizer + network.

These run WITHOUT Isaac Lab: they build synthetic TensorDict observations that
mirror the pmt-pretrain obs_groups dims and exercise the new
``PerceptiveResidualBehaviorTokenTracker`` end-to-end (act, evaluate, aux losses).
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from motion_tracking_rl.networks.perceptive_motion.token_tracker import (
    _ResidualFSQBehaviorTokenizer,
)
from motion_tracking_rl.networks.perceptive_motion.behavior_token_tracker import (
    PerceptiveResidualBehaviorTokenTracker,
)


FUTURE_MOTION_LEN = 21
# command_window per-frame = 38 (v3+w3+g3+q29); anchor delta per-frame = 3.
FRAME_DIM = 38 + 3
POLICY_DIM = 64
HISTORY_DIM = 128
CRITIC_DIM = 100


def _make_obs(batch: int = 8) -> tuple[TensorDict, dict[str, list[str]]]:
    future_dim = FUTURE_MOTION_LEN * FRAME_DIM
    data = {
        "policy": torch.randn(batch, POLICY_DIM),
        "proprio_history": torch.randn(batch, HISTORY_DIM),
        "command_window": torch.randn(batch, FUTURE_MOTION_LEN, 38),
        "motion_anchor_delta_window": torch.randn(batch, FUTURE_MOTION_LEN, 3),
        "critic": torch.randn(batch, CRITIC_DIM),
    }
    obs = TensorDict(data, batch_size=[batch])
    obs_groups = {
        "policy": ["policy"],
        "policy_history": ["proprio_history"],
        "future_motion_window": ["command_window", "motion_anchor_delta_window"],
        "critic": ["critic"],
    }
    assert future_dim == 38 * FUTURE_MOTION_LEN + 3 * FUTURE_MOTION_LEN
    return obs, obs_groups


def test_residual_fsq_tokenizer_shapes_and_ste():
    tok = _ResidualFSQBehaviorTokenizer(
        future_motion_dim=FUTURE_MOTION_LEN * FRAME_DIM,
        future_motion_len=FUTURE_MOTION_LEN,
        model_dim=128,
        token_dim=64,
        num_tokens=4,
        activation="elu",
        num_heads=4,
        fsq_levels=[8, 8, 8, 5, 5],
        num_residual_levels=2,
    )
    flat = torch.randn(6, FUTURE_MOTION_LEN * FRAME_DIM, requires_grad=True)
    z_e, z_q = tok(flat)
    assert z_e.shape == (6, 4, 64)
    assert z_q.shape == (6, 4, 64)
    # STE: gradient flows back to the input through the quantizer.
    z_q.pow(2).mean().backward()
    assert flat.grad is not None and float(flat.grad.abs().sum()) > 0.0
    # Per-level FSQ indices recorded, within per-dim level bounds.
    levels = tok.last_code_indices()
    assert len(levels) == 2
    bounds = torch.tensor([8, 8, 8, 5, 5])
    for idx in levels:
        assert idx.shape == (6, 4, 5)
        assert int(idx.min()) >= 0
        assert bool((idx.max(dim=0).values.max(dim=0).values < bounds).all())


def test_plain_fsq_single_level():
    tok = _ResidualFSQBehaviorTokenizer(
        future_motion_dim=FUTURE_MOTION_LEN * FRAME_DIM,
        future_motion_len=FUTURE_MOTION_LEN,
        model_dim=128,
        token_dim=64,
        num_tokens=4,
        activation="elu",
        num_heads=4,
        fsq_levels=[8, 8, 8, 5, 5],
        num_residual_levels=1,
    )
    _, z_q = tok(torch.randn(3, FUTURE_MOTION_LEN * FRAME_DIM))
    assert z_q.shape == (3, 4, 64)
    assert len(tok.last_code_indices()) == 1


def _build_tracker(use_phase_head: bool = False, aux_coef: dict | None = None):
    obs, obs_groups = _make_obs()
    tracker = PerceptiveResidualBehaviorTokenTracker(
        obs,
        obs_groups,
        num_actions=29,
        fsq_levels=[8, 8, 8, 5, 5],
        num_residual_levels=1,
        use_phase_head=use_phase_head,
        aux_loss_coef=aux_coef or {"fsq_usage_entropy": 0.01, "motion_recon": 0.1},
        # pmt-pretrain from-scratch contract:
        policy_set_name="policy",
        history_set_name="policy_history",
        future_motion_set_name="future_motion_window",
        critic_set_name="critic",
        future_motion_len=FUTURE_MOTION_LEN,
        num_motion_tokens=4,
        motion_token_dim=64,
        adapter_mode="no_adapter",
        pmt_only_mode=True,
        freeze_pmt=False,
        require_height_scan=False,
        require_teacher_motion_target=False,
        require_pmt_checkpoint=False,
        use_foot_event_posterior=False,
    )
    return tracker, obs


def test_tracker_act_and_evaluate():
    tracker, obs = _build_tracker()
    action = tracker.act_inference(obs)
    assert action.shape == (8, 29)
    assert torch.isfinite(action).all()
    value = tracker.evaluate(obs)
    assert value.shape == (8, 1)
    # The tokenizer is the FSQ one.
    assert isinstance(tracker.motion_tokenizer, _ResidualFSQBehaviorTokenizer)


def test_tracker_aux_losses():
    tracker, obs = _build_tracker()
    result = tracker.compute_sonic_aux_losses(obs)
    assert "aux_losses" in result and "aux_loss_coef" in result
    losses = result["aux_losses"]
    # On flat (no contact/phase labels): only usage-entropy + motion_recon are active.
    assert "fsq_usage_entropy" in losses
    assert "motion_recon" in losses
    assert "contact_bce" not in losses
    for name, value in losses.items():
        assert torch.isfinite(value), name
    # motion_recon must carry gradient into the tokenizer (STE path).
    losses["motion_recon"].backward()
    grads = [p.grad for p in tracker.motion_tokenizer.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_fsq_usage_entropy_is_differentiable():
    # The soft usage-entropy surrogate must carry gradient into fsq_in (anti-collapse).
    tracker, obs = _build_tracker(aux_coef={"fsq_usage_entropy": 0.01})
    result = tracker.compute_sonic_aux_losses(obs)
    ent = result["aux_losses"]["fsq_usage_entropy"]
    assert ent.requires_grad
    ent.backward()
    fsq_in_grads = [
        p.grad for p in tracker.motion_tokenizer.fsq_in.parameters() if p.grad is not None
    ]
    assert len(fsq_in_grads) > 0 and any(float(g.abs().sum()) > 0 for g in fsq_in_grads)
    # Monitor (hard, detached) entropy is a finite float in [0, 1].
    mon = tracker.fsq_usage_entropy_monitor()
    assert mon is not None and 0.0 <= mon <= 1.0 + 1e-6


def test_tracker_rejects_pmt_checkpoint():
    # P-CaRBT trains from scratch; a PMT ckpt path must be refused loudly.
    obs, obs_groups = _make_obs()
    import pytest

    with pytest.raises(ValueError):
        PerceptiveResidualBehaviorTokenTracker(
            obs,
            obs_groups,
            num_actions=29,
            pmt_ckpt_path="/nonexistent/model.pt",
            future_motion_len=FUTURE_MOTION_LEN,
            num_motion_tokens=4,
            motion_token_dim=64,
            adapter_mode="no_adapter",
            pmt_only_mode=False,
            freeze_pmt=True,
            require_height_scan=False,
            require_pmt_checkpoint=False,
        )


def test_tracker_with_phase_head():
    tracker, obs = _build_tracker(use_phase_head=True)
    out = tracker.infer_student_outputs(obs)
    assert "phase_hat" in out
    assert out["phase_hat"].shape == (8, 2)
    # Normalized to unit circle.
    norms = out["phase_hat"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
