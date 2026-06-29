"""Pure-torch tests for the Phase-2 offline difficulty prior (no Isaac Sim).

Loads the prior module by FILE PATH (avoids the package __init__ -> isaaclab).
Run: TMPDIR=/dev/shm conda run -n cluster_isaaclab python tests/test_adaptive_sampling_prior.py
"""
import importlib.util
import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    p = _ROOT / rel
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


pm = _load("prior_test", "pmt_tasks/mdp/commands/adaptive_sampling_prior.py")


def test_bin_slices_cover_all_frames():
    sl = pm._bin_slices(125, 50)  # 3 bins: [0,50),[50,100),[100,125)
    assert sl == [(0, 50), (50, 100), (100, 125)], sl
    sl2 = pm._bin_slices(40, 50)  # short clip -> 1 bin
    assert sl2 == [(0, 40)], sl2
    print("OK test_bin_slices_cover_all_frames")


def test_high_freq_ratio_higher_for_fast_signal():
    T = 200
    t = torch.arange(T).float()
    slow = torch.sin(2 * 3.14159 * t / 100).unsqueeze(1)   # 2 cycles over T
    fast = torch.sin(2 * 3.14159 * t / 4).unsqueeze(1)      # 50 cycles over T
    r_slow = pm._high_freq_ratio(slow)
    r_fast = pm._high_freq_ratio(fast)
    assert r_fast > r_slow, (r_slow, r_fast)
    assert 0.0 <= r_slow <= 1.0 and 0.0 <= r_fast <= 1.0
    # constant signal -> 0 (no dynamics)
    assert pm._high_freq_ratio(torch.ones(T, 3)) == 0.0
    # too-short signal -> 0
    assert pm._high_freq_ratio(torch.randn(2, 3)) == 0.0
    print("OK test_high_freq_ratio_higher_for_fast_signal")


def test_compute_clip_bin_features_shapes_and_jerk():
    T, J, B = 130, 5, 4
    fps, bin_size = 50.0, 50
    # Build a clip whose SECOND bin has much larger joint velocity changes (jerk).
    jv = torch.zeros(T, J)
    jv[50:100] = torch.randn(50, J) * 10.0   # high jerk bin 1
    blv = torch.zeros(T, B, 3)
    feats = pm.compute_clip_bin_features(jv, blv, fps, bin_size)
    assert feats.num_bins == 3
    assert feats.rms_jerk.shape == (3,)
    assert feats.rms_jerk[1] > feats.rms_jerk[0], feats.rms_jerk
    assert feats.rms_jerk[1] > feats.rms_jerk[2], feats.rms_jerk
    print("OK test_compute_clip_bin_features_shapes_and_jerk")


def test_combine_features_normalizes_and_pads():
    # Two clips, different bin counts; ensure padding bins are 0 and range in [0,1].
    f0 = pm.ClipBinFeatures(hf_ratio=torch.tensor([0.1, 0.9, 0.5]),
                            rms_jerk=torch.tensor([1.0, 5.0, 2.0]),
                            body_accel=torch.tensor([0.0, 2.0, 1.0]), num_bins=3)
    f1 = pm.ClipBinFeatures(hf_ratio=torch.tensor([0.2, 0.3]),
                            rms_jerk=torch.tensor([0.5, 0.6]),
                            body_accel=torch.tensor([0.1, 0.2]), num_bins=2)
    prior = pm.combine_features_to_prior([f0, f1], max_bins=3)
    assert prior.shape == (2, 3)
    assert (prior >= 0).all() and (prior <= 1).all()
    assert prior[1, 2].item() == 0.0  # padding bin of clip 1
    # clip 0 bin 1 has the highest features -> should be the max entry
    assert prior[0, 1] == prior.max(), prior
    print("OK test_combine_features_normalizes_and_pads")


def test_slice_prior_to_working_set():
    # Global prior for 5 clips, 4 bins each.
    gp = torch.arange(20).float().reshape(5, 4) / 20.0
    w2g = torch.tensor([3, 0, 4])           # resident local 0->global3, 1->0, 2->4
    out = pm.slice_prior_to_working_set(gp, w2g, local_max_bins=3)
    assert out.shape == (3, 3)
    assert torch.allclose(out[0], gp[3, :3])
    assert torch.allclose(out[1], gp[0, :3])
    assert torch.allclose(out[2], gp[4, :3])
    # local_max_bins larger than global -> zero-pad
    out2 = pm.slice_prior_to_working_set(gp, w2g, local_max_bins=6)
    assert out2.shape == (3, 6)
    assert (out2[:, 4:] == 0).all()
    print("OK test_slice_prior_to_working_set")


def test_injection_into_sampler_matches_failed_count_formula():
    """The sampler seeds failed_bin_count = 1 + strength*prior on valid bins."""
    aslib = _load("aslib_for_prior", "pmt_tasks/mdp/commands/adaptive_sampling_lib.py")
    lengths = torch.tensor([200, 200])
    prior = torch.zeros(2, 4)
    prior[0, 1] = 0.5
    s = aslib.HybridBinSampler(2, lengths, "cpu", motion_fps=50.0, bin_duration=1.0,
                               offline_bin_prior=prior, offline_prior_strength=10.0)
    # bin1 of motion0 should be 1 + 10*0.5 = 6.0; others stay 1.0 (valid bins).
    assert abs(s.failed_bin_count[0, 1].item() - 6.0) < 1e-6, s.failed_bin_count[0, 1]
    assert abs(s.failed_bin_count[0, 0].item() - 1.0) < 1e-6
    print("OK test_injection_into_sampler_matches_failed_count_formula")


def test_boundary_diff_not_dropped():
    """Codex finding: the jerk diff at a bin boundary (frame e-1 -> e) must be counted
    in exactly one bin, not dropped. The only velocity change here is at the 49->50
    transition; bin 0 (frames 0..49, owning diff indices [0,50)) must capture it."""
    T, J = 120, 3
    jv = torch.zeros(T, J)
    jv[50:] = 5.0  # single step change: diff lives at jacc index 49 (frame 49->50)
    feats = pm.compute_clip_bin_features(jv, None, fps=50.0, bin_size=50)
    assert feats.rms_jerk[0] > 0.0, feats.rms_jerk      # boundary diff captured in bin 0
    assert feats.rms_jerk[1] == 0.0, feats.rms_jerk     # post-step velocity constant
    print("OK test_boundary_diff_not_dropped")


def test_non_finite_inputs_are_sanitized():
    """NaN/Inf in a clip must not propagate into features."""
    T, J, B = 100, 4, 3
    jv = torch.randn(T, J)
    jv[10, 0] = float("nan")
    jv[20, 1] = float("inf")
    blv = torch.randn(T, B, 3)
    blv[30, 0, 0] = float("-inf")
    feats = pm.compute_clip_bin_features(jv, blv, fps=50.0, bin_size=50)
    assert torch.isfinite(feats.hf_ratio).all()
    assert torch.isfinite(feats.rms_jerk).all()
    assert torch.isfinite(feats.body_accel).all()
    prior = pm.combine_features_to_prior([feats], max_bins=feats.num_bins)
    assert torch.isfinite(prior).all() and (prior >= 0).all() and (prior <= 1).all()
    print("OK test_non_finite_inputs_are_sanitized")


def test_robust_norm_handles_nan_and_degenerate():
    out = pm._robust_unit_norm(torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
                               torch.ones(2, 2, dtype=torch.bool))
    assert torch.isfinite(out).all()
    eq = pm._robust_unit_norm(torch.full((3, 2), 5.0), torch.ones(3, 2, dtype=torch.bool))
    assert (eq == 0).all()
    print("OK test_robust_norm_handles_nan_and_degenerate")


if __name__ == "__main__":
    test_bin_slices_cover_all_frames()
    test_high_freq_ratio_higher_for_fast_signal()
    test_compute_clip_bin_features_shapes_and_jerk()
    test_combine_features_normalizes_and_pads()
    test_slice_prior_to_working_set()
    test_injection_into_sampler_matches_failed_count_formula()
    test_boundary_diff_not_dropped()
    test_non_finite_inputs_are_sanitized()
    test_robust_norm_handles_nan_and_degenerate()
    print("\nALL PHASE-2 PRIOR TESTS PASSED")
