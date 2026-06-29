"""Phase-3 GLOBAL anti-forgetting (age term) tests for GlobalCurriculum.

Pure-torch (streaming_motion_lib has no isaaclab import), loaded by file path.
Run: TMPDIR=/dev/shm conda run -n cluster_isaaclab python tests/test_global_curriculum_age.py
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


slib = _load("slib_age_test", "pmt_tasks/mdp/commands/streaming_motion_lib.py")
GlobalCurriculum = slib.GlobalCurriculum


def test_age_ratio_zero_is_unchanged():
    """age_ratio=0 must reproduce the original failure+uniform probabilities exactly."""
    a = GlobalCurriculum(10, beta=1.0, uniform_ratio=0.1, age_ratio=0.0)
    b = GlobalCurriculum(10, beta=1.0, uniform_ratio=0.1)  # original signature
    a.failed[3] += 20.0; b.failed[3] += 20.0
    # advance ages on a (should NOT matter when age_ratio=0)
    a._folds_since_seen[:] = 50.0
    assert torch.allclose(a.probabilities(), b.probabilities(), atol=1e-7)
    print("OK test_age_ratio_zero_is_unchanged")


def test_fold_increments_age_for_untouched_resets_touched():
    gc = GlobalCurriculum(5, alpha=0.5)
    # Touch clips 0 and 2 this round.
    gc.update(torch.tensor([0, 2]), torch.tensor([True, False]))
    gc.fold()
    age = gc._folds_since_seen
    assert age[0] == 0.0 and age[2] == 0.0, age      # touched -> reset
    assert age[1] == 1.0 and age[3] == 1.0 and age[4] == 1.0, age  # untouched -> +1
    # Next round touch only clip 1.
    gc.update(torch.tensor([1]), torch.tensor([False]))
    gc.fold()
    assert gc._folds_since_seen[1] == 0.0
    assert gc._folds_since_seen[3] == 2.0  # untouched twice
    print("OK test_fold_increments_age_for_untouched_resets_touched")


def test_age_term_lifts_stale_clip_probability():
    """With age_ratio>0, a long-unsampled clip gets MORE reload probability than with
    age_ratio=0, even if its failure stats are unremarkable."""
    n = 8
    base = GlobalCurriculum(n, beta=1.0, uniform_ratio=0.1, age_ratio=0.0)
    aged = GlobalCurriculum(n, beta=1.0, uniform_ratio=0.1, age_ratio=0.4, age_tau=5.0)
    # identical failure stats
    for gc in (base, aged):
        gc.failed[:] = 1.0; gc.success[:] = 1.0
    # clip 7 is very stale on the aged curriculum
    aged._folds_since_seen[7] = 100.0
    p_base = base.probabilities()
    p_aged = aged.probabilities()
    assert p_aged[7] > p_base[7], (p_base[7].item(), p_aged[7].item())
    assert abs(float(p_aged.sum()) - 1.0) < 1e-6
    print("OK test_age_term_lifts_stale_clip_probability")


def test_age_probs_finite_when_all_zero_age():
    gc = GlobalCurriculum(6, age_ratio=0.3, age_tau=10.0)
    # all ages 0 -> age weights all 0 -> fallback uniform; probs must be valid
    p = gc.probabilities()
    assert torch.isfinite(p).all() and abs(float(p.sum()) - 1.0) < 1e-6
    print("OK test_age_probs_finite_when_all_zero_age")


def test_budget_overflow_raises():
    try:
        GlobalCurriculum(5, uniform_ratio=0.8, age_ratio=0.5)  # 1.3 > 1
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("OK test_budget_overflow_raises")


def test_each_component_validated():
    """Codex finding: negative components must be rejected, not just a bad sum."""
    for u, a in ((-0.1, 0.5), (0.6, -0.2), (1.2, 0.0), (0.0, 1.3)):
        try:
            GlobalCurriculum(5, uniform_ratio=u, age_ratio=a)
            raise AssertionError(f"expected ValueError for uniform={u}, age={a}")
        except ValueError:
            pass
    print("OK test_each_component_validated")


def test_state_dict_roundtrip_includes_age():
    gc = GlobalCurriculum(5, age_ratio=0.2)
    gc._folds_since_seen[:] = torch.tensor([1., 2., 3., 4., 5.])
    st = gc.state_dict()
    assert "folds_since_seen" in st
    gc2 = GlobalCurriculum(5, age_ratio=0.2)
    gc2.load_state_dict(st)
    assert torch.equal(gc2._folds_since_seen.cpu(), torch.tensor([1., 2., 3., 4., 5.]))
    # backward-compat: a pre-Phase-3 state without the key must not crash
    old = {"failed": gc.failed.cpu(), "success": gc.success.cpu()}
    gc3 = GlobalCurriculum(5, age_ratio=0.2)
    gc3.load_state_dict(old)  # should not raise
    print("OK test_state_dict_roundtrip_includes_age")


if __name__ == "__main__":
    test_age_ratio_zero_is_unchanged()
    test_fold_increments_age_for_untouched_resets_touched()
    test_age_term_lifts_stale_clip_probability()
    test_age_probs_finite_when_all_zero_age()
    test_budget_overflow_raises()
    test_each_component_validated()
    test_state_dict_roundtrip_includes_age()
    print("\nALL PHASE-3 GLOBAL AGE TESTS PASSED")
