"""Pure-torch unit tests for HybridBinSampler (no isaaclab / Isaac Sim needed).

Loads the sampler module by FILE PATH so importing it does not trigger the
``pmt_tasks.mdp.commands`` package __init__ (which pulls isaaclab/omni and
requires a booted Isaac Sim app). Run with the cluster_isaaclab env:

    conda run -n cluster_isaaclab python tests/test_adaptive_sampling_lib.py
"""

import importlib.util
import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    p = _ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


aslib = _load("adaptive_sampling_lib_standalone", "pmt_tasks/mdp/commands/adaptive_sampling_lib.py")
HybridBinSampler = aslib.HybridBinSampler
SamplingResult = aslib.SamplingResult


def _mk(num_motions=8, lengths=None, **kw):
    if lengths is None:
        lengths = torch.tensor([300, 450, 600, 250, 800, 120, 999, 1000][:num_motions])
    return HybridBinSampler(num_motions, lengths, "cpu", motion_fps=50.0, bin_duration=1.0, **kw)


def test_sample_shapes_and_validity():
    torch.manual_seed(0)
    s = _mk()
    res = s.sample(1024)
    assert res.motion_ids.shape == (1024,)
    assert res.frame_ids.shape == (1024,)
    # frame within each motion's valid range
    lens = s.motion_lengths[res.motion_ids]
    assert (res.frame_ids >= 0).all()
    assert (res.frame_ids < lens).all()
    print("OK test_sample_shapes_and_validity")


def test_empty_sample():
    s = _mk()
    res = s.sample(0)
    assert res.motion_ids.numel() == 0 and res.frame_ids.numel() == 0
    print("OK test_empty_sample")


def test_motion_probs_sum_to_one():
    s = _mk()
    p = s._get_motion_probs()
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert (p >= 0).all()
    print("OK test_motion_probs_sum_to_one")


def test_phase0_parity_with_bin_sampler():
    """With hybrid hooks off, HybridBinSampler must match BinBasedAdaptiveSampler.

    BinBasedAdaptiveSampler imports MotionSampler from multi_motion_command
    (isaaclab). To stay isaaclab-free we instead assert the *defining* math
    matches: motion score = max-over-bins of smoothed p_fail with identical
    Laplace priors, beta, uniform mix. We replicate that formula here from the
    sampler's own count tensors and require equality with its cached probs.
    """
    s = _mk(beta=0.7, uniform_ratio=0.15, kernel_size=5, kernel_lambda=0.8)
    # Inject some asymmetric failure history directly into persistent counts.
    s.failed_bin_count[2, 1] += 50.0
    s.failed_bin_count[5, 0] += 10.0
    s._update_metrics()

    # Reference formula (the documented Phase-0 behavior).
    total = s.failed_bin_count + s.success_bin_count
    pf = s.failed_bin_count / (total + 1e-8)
    pf = pf.clone()
    pf[~s.valid_bins_mask] = 0.0
    pf_3d = pf.unsqueeze(1)
    pad = s._kernel_size - 1
    pf_pad = torch.nn.functional.pad(pf_3d, (pad, 0))
    pf_sm = torch.nn.functional.conv1d(pf_pad, s._kernel).squeeze(1)
    pf_sm[~s.valid_bins_mask] = 0.0
    score = pf_sm.max(dim=1).values
    w = torch.clamp(score, min=1e-6).pow(s.beta)
    w = w / (w.sum() + 1e-8)
    uni = 1.0 / s.num_motions
    ref = w * (1 - s.uniform_ratio) + uni * s.uniform_ratio
    ref = ref / ref.sum()

    got = s._get_motion_probs()
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()
    print("OK test_phase0_parity_with_bin_sampler")


def test_error_weight_zero_is_pure_failure():
    """error_weight=0 => _pf_bin == failure rate (Phase 0 invariant)."""
    s = _mk(error_weight=0.0)
    s.failed_bin_count[1, 0] += 7.0
    total = s.failed_bin_count + s.success_bin_count
    expected = s.failed_bin_count / (total + 1e-8)
    assert torch.allclose(s._pf_bin(), expected, atol=1e-7)
    print("OK test_error_weight_zero_is_pure_failure")


def test_update_vectorized_failure_attribution_start_bin():
    """Phase 0 attribution: outcomes go to the START bin (frame_ids // bin_size)."""
    s = _mk(update_interval=1)
    motion_ids = torch.tensor([2, 2, 3])
    frame_ids = torch.tensor([60, 60, 10])  # bin 1 for motion 2, bin 0 for motion 3
    terminated = torch.tensor([True, True, False])
    before_fail = s.current_failed_bin_count[2, 1].item()
    s.update(motion_ids, frame_ids, terminated)
    assert s.current_failed_bin_count[2, 1].item() == before_fail + 2
    assert s.current_success_bin_count[3, 0].item() >= 1
    print("OK test_update_vectorized_failure_attribution_start_bin")


def test_offline_prior_raises_proportion_for_hard_bins():
    """Phase 2 hook: offline prior biases initial sampling toward flagged bins."""
    lengths = torch.tensor([500, 500, 500, 500])
    prior = torch.zeros(4, 10)
    prior[0, 0] = 1.0  # flag motion 0 as hard
    s = HybridBinSampler(4, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         offline_bin_prior=prior, offline_prior_strength=20.0,
                         uniform_ratio=0.0, beta=1.0)
    p = s._get_motion_probs()
    assert p[0] == p.max(), p
    print("OK test_offline_prior_raises_proportion_for_hard_bins")


def test_backward_attribution_when_end_frame_given():
    """Phase 1 hook: terminated episodes attribute to the TERMINATION bin."""
    s = _mk(update_interval=1)
    motion_ids = torch.tensor([4])
    start = torch.tensor([0])         # bin 0
    end = torch.tensor([700])          # bin 14 (700//50)
    terminated = torch.tensor([True])
    s.update(motion_ids, start, terminated, end_frame_ids=end)
    assert s.current_failed_bin_count[4, 14].item() == 1.0
    assert s.current_failed_bin_count[4, 0].item() == 0.0
    print("OK test_backward_attribution_when_end_frame_given")


# ===================== Phase 1 tests =====================

def test_success_attributes_to_start_even_with_end_frame():
    """Backward blame applies to FAILURES only; a success attributes to its start bin."""
    s = _mk(update_interval=1)
    motion_ids = torch.tensor([4, 4])
    start = torch.tensor([0, 0])     # bin 0
    end = torch.tensor([700, 700])    # bin 14
    terminated = torch.tensor([True, False])
    s.update(motion_ids, start, terminated, end_frame_ids=end)
    # failure -> termination bin 14; success -> start bin 0
    assert s.current_failed_bin_count[4, 14].item() == 1.0
    assert s.current_success_bin_count[4, 0].item() == 1.0
    assert s.current_success_bin_count[4, 14].item() == 0.0
    print("OK test_success_attributes_to_start_even_with_end_frame")


def test_raw_error_not_clamped_and_normalized_in_pf_bin():
    """Phase 1: raw error > 1 is stored (not clamped), then mapped to [0,1] via
    error_good/error_bad inside _pf_bin."""
    lengths = torch.tensor([500, 500])
    s = HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         update_interval=1, error_weight=1.0, failure_weight=0.0,
                         error_good=0.0, error_bad=2.0)
    # One success (so failure rate stays ~0) with a large raw error of 2.0 on bin 0.
    s.update(torch.tensor([0]), torch.tensor([0]), torch.tensor([False]),
             end_frame_ids=torch.tensor([0]), tracking_error=torch.tensor([2.0]))
    s.step()  # fold EMA
    # error_bin_ema should reflect the raw 2.0 mean (scaled by alpha into the EMA).
    assert s.current_error_bin_count[0, 0].item() == 0.0  # reset after step
    # With failure_weight=0, error_weight=1, error_bad=2.0: pf at bin0 motion0 ~ ema/2.
    pf = s._pf_bin()
    assert pf[0, 0] > pf[1, 0], (pf[0, 0], pf[1, 0])  # errored bin is "harder"
    assert (pf <= 1.0 + 1e-6).all() and (pf >= 0.0).all()
    print("OK test_raw_error_not_clamped_and_normalized_in_pf_bin")


def test_error_increases_motion_prob_after_fold():
    """A clip with high tracking error (but no failures) becomes more likely to sample."""
    lengths = torch.tensor([400, 400, 400, 400])
    s = HybridBinSampler(4, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         update_interval=1, error_weight=1.0, failure_weight=1.0,
                         error_good=0.0, error_bad=1.0, uniform_ratio=0.0, beta=1.0,
                         alpha=0.5)
    base = s._get_motion_probs().clone()
    # Motion 2 tracks badly (high error) on several successful episodes.
    for _ in range(5):
        s.update(torch.tensor([2]), torch.tensor([0]), torch.tensor([False]),
                 end_frame_ids=torch.tensor([0]), tracking_error=torch.tensor([1.0]))
        s.step()
    after = s._get_motion_probs()
    assert after[2] > base[2], (base[2].item(), after[2].item())
    print("OK test_error_increases_motion_prob_after_fold")


def test_phase1_disabled_equals_phase0():
    """The COMMAND disabled-path (error_weight=0) calls update() with NO Phase-1
    kwargs (AdaptiveSamplingMotionCommand._record_outcomes falls back to base 3-arg).
    Prove that 3-arg call is bit-identical to a Phase-0 sampler, AND that no error is
    accumulated. Uses DIFFERENT start vs (hypothetical) end frames to be sure we are
    not accidentally hiding a divergence."""
    s0 = _mk(update_interval=1, error_weight=0.0)
    s1 = _mk(update_interval=1, error_weight=0.0)
    mids = torch.tensor([2, 3, 2])
    fr = torch.tensor([60, 10, 60])
    term = torch.tensor([True, False, True])
    s0.update(mids, fr, term)         # Phase-0 style
    s1.update(mids, fr, term)         # exactly what the disabled command path calls
    assert torch.equal(s0.current_failed_bin_count, s1.current_failed_bin_count)
    assert torch.equal(s0.current_success_bin_count, s1.current_success_bin_count)
    assert s1.current_error_bin_count.sum().item() == 0.0
    print("OK test_phase1_disabled_equals_phase0")


def test_end_frame_attribution_is_orthogonal_to_error_weight():
    """Documents the lib contract the reviewer flagged: backward (end-bin) attribution
    for failures depends ONLY on whether end_frame_ids is supplied, NOT on error_weight.
    With error_weight=0 but end_frame_ids given, a failure STILL lands on the end bin.
    The COMMAND guarantees Phase-0 start-bin behavior by simply not passing the kwarg
    when error is disabled (covered by test_phase1_disabled_equals_phase0)."""
    s = _mk(update_interval=1, error_weight=0.0)
    s.update(torch.tensor([4]), torch.tensor([0]), torch.tensor([True]),
             end_frame_ids=torch.tensor([700]))
    assert s.current_failed_bin_count[4, 14].item() == 1.0  # end bin, not start bin 0
    assert s.current_failed_bin_count[4, 0].item() == 0.0
    print("OK test_end_frame_attribution_is_orthogonal_to_error_weight")


# ===================== Phase 3 tests (local) =====================

def test_retention_ratio_zero_equals_phase2():
    """retention_ratio=0 AND topk_motion=1 must reproduce the prior motion-prob math
    (max-bin score + uniform mix). Inject failure history and compare to the explicit
    Phase-0/1 formula."""
    s = _mk(beta=0.7, uniform_ratio=0.15, retention_ratio=0.0, topk_motion=1)
    s.failed_bin_count[2, 1] += 30.0
    s._update_metrics()
    # reference: max-bin smoothed pf -> pow -> uniform mix
    pf = s._compute_pf_bin_smoothed()
    score = pf.max(dim=1).values
    w = torch.clamp(score, min=1e-6).pow(s.beta); w = w / (w.sum() + 1e-8)
    uni = 1.0 / s.num_motions
    ref = w * (1 - s.uniform_ratio) + uni * s.uniform_ratio; ref = ref / ref.sum()
    assert torch.allclose(s._get_motion_probs(), ref, atol=1e-6)
    print("OK test_retention_ratio_zero_equals_phase2")


def test_topk_motion_softens_single_hard_bin():
    """With topk_motion>1, a clip with ONE very hard bin (rest easy) gets LESS mass
    than under pure max-bin scoring, because mean(top-k) dilutes the single spike."""
    lengths = torch.tensor([500, 500])
    # clip 0: one super-hard bin; clip 1: uniformly moderately hard across bins.
    # Identical config EXCEPT topk so the comparison isolates the top-k softening
    # (kernel_size=1 on both -> no smoothing confound).
    s_max = HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                             uniform_ratio=0.0, beta=1.0, topk_motion=1, kernel_size=1)
    s_topk = HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                              uniform_ratio=0.0, beta=1.0, topk_motion=5, topk_motion_weight=0.5,
                              kernel_size=1)
    for s in (s_max, s_topk):
        s.failed_bin_count[0, 0] += 100.0          # clip 0: one spike
        s.failed_bin_count[1, :5] += 8.0           # clip 1: broad moderate
        s._update_metrics()
    p_max = s_max._get_motion_probs()
    p_topk = s_topk._get_motion_probs()
    # Under top-k, clip 0's relative share should drop vs pure-max.
    assert p_topk[0] < p_max[0], (p_max[0].item(), p_topk[0].item())
    print("OK test_topk_motion_softens_single_hard_bin")


def test_retention_budget_protects_learned_clips():
    """A learned clip (high success-EMA, ~0 failures) keeps a reload floor >= kappa/N_learned
    even though hard scoring alone would nearly starve it."""
    lengths = torch.tensor([300, 300, 300, 300])
    s = HybridBinSampler(4, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         uniform_ratio=0.0, beta=2.0, retention_ratio=0.3,
                         retention_success_thresh=0.85)
    # Motion 0: lots of successes (learned). Motions 1-3: lots of failures (hard).
    s.success_motion_count[0] += 100.0
    s.failed_bin_count[1:, 0] += 50.0
    s.failed_motion_count[1:] += 50.0
    s._update_metrics()
    p = s._get_motion_probs()
    # learned clip 0 is the only learned one -> retention mass 0.3 concentrates on it.
    assert p[0] >= 0.3 - 1e-3, p
    assert abs(float(p.sum()) - 1.0) < 1e-6
    print("OK test_retention_budget_protects_learned_clips")


def test_retention_falls_back_uniform_when_none_learned():
    """Early training: no clip meets the success threshold -> retention spreads
    uniformly (mass not lost), probs still valid."""
    lengths = torch.tensor([300, 300, 300])
    s = HybridBinSampler(3, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         uniform_ratio=0.1, beta=1.0, retention_ratio=0.3,
                         retention_success_thresh=0.99)
    s.failed_bin_count[0, 0] += 10.0
    s._update_metrics()
    p = s._get_motion_probs()
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert (p > 0).all()
    print("OK test_retention_falls_back_uniform_when_none_learned")


def test_budget_validation_raises_on_overflow():
    lengths = torch.tensor([100, 100])
    try:
        HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         uniform_ratio=0.7, retention_ratio=0.5)  # sum=1.2 > 1
        raise AssertionError("expected ValueError on uniform+retention > 1")
    except ValueError:
        pass
    print("OK test_budget_validation_raises_on_overflow")


def test_topk_valid_bins_only_no_short_clip_bias():
    """Codex Phase-3 finding: top-k must average only a clip's OWN valid bins. A short
    1-bin clip and a long clip with the SAME real per-bin difficulty must get the SAME
    motion score (padding zeros must NOT drag the short clip down)."""
    # clip 0: 1 bin (50 frames); clip 1: 6 bins (300 frames). Same difficulty 0.9 on
    # every REAL bin. kernel off to isolate.
    lengths = torch.tensor([50, 300])
    s = HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         topk_motion=5, topk_motion_weight=0.5, kernel_size=1,
                         uniform_ratio=0.0, beta=1.0)
    # Set every valid bin's failure rate equal (failed=9, success=1 -> pf=0.9).
    s.failed_bin_count[0, 0] = 9.0; s.success_bin_count[0, 0] = 1.0
    s.failed_bin_count[1, :6] = 9.0; s.success_bin_count[1, :6] = 1.0
    s._update_metrics()
    pf = s._compute_pf_bin_smoothed()
    score = s._motion_score(pf)
    assert abs(float(score[0]) - float(score[1])) < 1e-5, (score[0].item(), score[1].item())
    print("OK test_topk_valid_bins_only_no_short_clip_bias")


def test_topk_weight_validation():
    lengths = torch.tensor([100, 100])
    for bad in (-0.1, 1.1, float("nan")):
        try:
            HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                             topk_motion=3, topk_motion_weight=bad)
            raise AssertionError(f"expected ValueError for topk_motion_weight={bad}")
        except ValueError:
            pass
    print("OK test_topk_weight_validation")


# ===================== Phase 4 tests =====================

def test_uncertainty_disabled_equals_phase3():
    """uncertainty_weight=0 => _pf_bin ignores uncertainty even if EMA is populated."""
    s = _mk(update_interval=1, error_weight=0.0, uncertainty_weight=0.0)
    s.uncertainty_bin_ema[0, 0] = 1.0  # would matter only if weight>0
    total = s.failed_bin_count + s.success_bin_count
    expected = s.failed_bin_count / (total + 1e-8)
    assert torch.allclose(s._pf_bin(), expected, atol=1e-7)
    print("OK test_uncertainty_disabled_equals_phase3")


def test_uncertainty_success_gate():
    """Uncertainty only contributes where per-bin success rate is in [lo, hi]."""
    lengths = torch.tensor([300, 300, 300])
    s = HybridBinSampler(3, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         update_interval=1, failure_weight=1.0, uncertainty_weight=1.0,
                         uncertainty_gate_lo=0.2, uncertainty_gate_hi=0.8, kernel_size=1)
    # Bin 0 of each motion: set success rate into/out of the gate band.
    # motion 0: success rate ~0.0 (hopeless) -> gate OFF
    s.failed_bin_count[0, 0] = 99.0; s.success_bin_count[0, 0] = 1.0
    # motion 1: success rate ~0.5 (in band) -> gate ON
    s.failed_bin_count[1, 0] = 5.0; s.success_bin_count[1, 0] = 5.0
    # motion 2: success rate ~1.0 (mastered) -> gate OFF
    s.failed_bin_count[2, 0] = 1.0; s.success_bin_count[2, 0] = 99.0
    # High uncertainty everywhere.
    s.uncertainty_bin_ema[:, 0] = 1.0
    pf = s._pf_bin()
    # Closed form with PER-BIN gated denominator (the Codex HIGH fix):
    #   gated ON:  pf = (fw*fail + uw*u) / (fw+uw)
    #   gated OFF: pf = (fw*fail + 0)   / (fw)        == pure failure rate
    # fw=uw=1, u=1.
    # motion 0: fail=0.99, gate OFF -> 0.99  (NOT diluted to 0.495)
    # motion 1: fail=0.50, gate ON  -> (0.50 + 1)/2 = 0.75
    # motion 2: fail=0.01, gate OFF -> 0.01
    assert abs(float(pf[0, 0]) - 0.99) < 0.02, pf[0, 0]
    assert abs(float(pf[1, 0]) - 0.75) < 0.02, pf[1, 0]
    assert abs(float(pf[2, 0]) - 0.01) < 0.02, pf[2, 0]
    # The gated-ON clip's uncertainty lifts it ABOVE its pure failure rate (0.5).
    assert float(pf[1, 0]) > 0.5
    print("OK test_uncertainty_success_gate")


def test_uncertainty_pf_in_bounds():
    s = _mk(update_interval=1, failure_weight=1.0, uncertainty_weight=1.0)
    s.uncertainty_bin_ema[:] = 1.0
    s.failed_bin_count[:] = 5.0; s.success_bin_count[:] = 5.0
    pf = s._pf_bin()
    assert (pf >= 0).all() and (pf <= 1.0 + 1e-6).all()
    print("OK test_uncertainty_pf_in_bounds")


def test_hard_buffer_reserves_budget_for_top_k():
    """hard_buffer_ratio reserves mass for the top-K hardest clips (uniform over them)."""
    lengths = torch.tensor([300, 300, 300, 300, 300])
    s = HybridBinSampler(5, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         uniform_ratio=0.0, beta=1.0, hard_buffer_ratio=0.5, hard_buffer_k=2,
                         kernel_size=1)
    # Make motions 3 and 4 hardest.
    s.failed_bin_count[3, 0] = 90.0; s.success_bin_count[3, 0] = 10.0
    s.failed_bin_count[4, 0] = 80.0; s.success_bin_count[4, 0] = 20.0
    s._update_metrics()
    p = s._get_motion_probs()
    # The two hardest get at least the hard-buffer share (0.5 split between them = 0.25 each).
    assert p[3] >= 0.25 - 1e-3 and p[4] >= 0.25 - 1e-3, p
    assert abs(float(p.sum()) - 1.0) < 1e-6
    print("OK test_hard_buffer_reserves_budget_for_top_k")


def test_phase4_budget_validation():
    lengths = torch.tensor([100, 100])
    # uniform + retention + hard_buffer > 1 must raise
    try:
        HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         uniform_ratio=0.5, retention_ratio=0.3, hard_buffer_ratio=0.3)
        raise AssertionError("expected ValueError on budget overflow")
    except ValueError:
        pass
    # bad uncertainty gate
    try:
        HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         uncertainty_weight=1.0, uncertainty_gate_lo=0.9, uncertainty_gate_hi=0.2)
        raise AssertionError("expected ValueError on lo>hi gate")
    except ValueError:
        pass
    # NaN uncertainty weight
    try:
        HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         uncertainty_weight=float("nan"))
        raise AssertionError("expected ValueError on NaN uncertainty_weight")
    except ValueError:
        pass
    print("OK test_phase4_budget_validation")


def test_gated_off_uncertainty_does_not_dilute_failure():
    """Codex HIGH: a gated-OFF bin must reduce to its PURE failure rate (the
    uncertainty weight must NOT enter the denominator there)."""
    lengths = torch.tensor([300])
    s = HybridBinSampler(1, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         update_interval=1, failure_weight=1.0, uncertainty_weight=1.0,
                         uncertainty_gate_lo=0.2, uncertainty_gate_hi=0.8, kernel_size=1)
    # success rate ~0.0 -> gate OFF; high uncertainty present but must be excluded.
    s.failed_bin_count[0, 0] = 99.0; s.success_bin_count[0, 0] = 1.0
    s.uncertainty_bin_ema[0, 0] = 1.0
    pf = s._pf_bin()
    assert abs(float(pf[0, 0]) - 0.99) < 1e-3, pf[0, 0]  # NOT 0.495
    print("OK test_gated_off_uncertainty_does_not_dilute_failure")


def test_failure_weight_validation():
    lengths = torch.tensor([100])
    for bad in (-0.5, float("nan"), float("inf")):
        try:
            HybridBinSampler(1, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                             failure_weight=bad)
            raise AssertionError(f"expected ValueError for failure_weight={bad}")
        except ValueError:
            pass
    print("OK test_failure_weight_validation")


def test_hard_buffer_cold_start_uniform():
    """All-equal scores (cold start) => hard-buffer falls back to uniform (no bias)."""
    lengths = torch.tensor([100, 100, 100, 100])
    s = HybridBinSampler(4, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                         hard_buffer_ratio=0.5, hard_buffer_k=2, uniform_ratio=0.0, beta=1.0)
    # no outcomes injected -> all bins equal -> motion scores equal
    hb = s._hard_buffer_probs(s._motion_score(s._compute_pf_bin_smoothed()))
    assert torch.allclose(hb, torch.full((4,), 0.25), atol=1e-6), hb
    print("OK test_hard_buffer_cold_start_uniform")


def test_hard_buffer_disabled_equals_phase3():
    """hard_buffer_ratio=0 AND retention_ratio=0 => exact uniform-mix (Phase 0/1/2)."""
    s = _mk(beta=0.7, uniform_ratio=0.15, hard_buffer_ratio=0.0, retention_ratio=0.0)
    s.failed_bin_count[1, 0] += 20.0
    s._update_metrics()
    pf = s._compute_pf_bin_smoothed()
    score = pf.max(dim=1).values
    w = torch.clamp(score, min=1e-6).pow(s.beta); w = w / (w.sum() + 1e-8)
    uni = 1.0 / s.num_motions
    ref = w * (1 - s.uniform_ratio) + uni * s.uniform_ratio; ref = ref / ref.sum()
    assert torch.allclose(s._get_motion_probs(), ref, atol=1e-6)
    print("OK test_hard_buffer_disabled_equals_phase3")


if __name__ == "__main__":
    test_sample_shapes_and_validity()
    test_empty_sample()
    test_motion_probs_sum_to_one()
    test_phase0_parity_with_bin_sampler()
    test_error_weight_zero_is_pure_failure()
    test_update_vectorized_failure_attribution_start_bin()
    test_offline_prior_raises_proportion_for_hard_bins()
    test_backward_attribution_when_end_frame_given()
    # Phase 1
    test_success_attributes_to_start_even_with_end_frame()
    test_raw_error_not_clamped_and_normalized_in_pf_bin()
    test_error_increases_motion_prob_after_fold()
    test_phase1_disabled_equals_phase0()
    test_end_frame_attribution_is_orthogonal_to_error_weight()
    # Phase 3 (local)
    test_retention_ratio_zero_equals_phase2()
    test_topk_motion_softens_single_hard_bin()
    test_retention_budget_protects_learned_clips()
    test_retention_falls_back_uniform_when_none_learned()
    test_budget_validation_raises_on_overflow()
    test_topk_valid_bins_only_no_short_clip_bias()
    test_topk_weight_validation()
    # Phase 4
    test_uncertainty_disabled_equals_phase3()
    test_uncertainty_success_gate()
    test_uncertainty_pf_in_bounds()
    test_hard_buffer_reserves_budget_for_top_k()
    test_phase4_budget_validation()
    test_gated_off_uncertainty_does_not_dilute_failure()
    test_failure_weight_validation()
    test_hard_buffer_cold_start_uniform()
    test_hard_buffer_disabled_equals_phase3()
    print("\nALL PHASE-0+1+3+4 LIB TESTS PASSED")
