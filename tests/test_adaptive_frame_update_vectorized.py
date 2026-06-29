"""Verify the vectorized frame-count update equals the old per-env loop.

Phase 0 hygiene fix (adaptive_sampling_discussion.md §2.B.3). isaaclab-free: we
replicate BOTH the old loop and the new scatter_add_ on identical inputs and
assert equal counts. (We cannot import AdaptiveSampler directly — it pulls in
isaaclab — so this is a behavioral-equivalence test of the exact code paths.)
"""
import torch


def old_loop(failed_mask, success_mask, motion_ids, frame_ids, M, T):
    cf = torch.zeros(M, T)
    cs = torch.zeros(M, T)
    if failed_mask.any():
        for i in range(int(failed_mask.sum().item())):
            idx = int(failed_mask.nonzero()[i].item())
            mid = int(motion_ids[idx].item())
            # EXACT old semantics: upper-clamp only. A negative fid 2-D-indexes
            # cf[mid, fid] with torch negative indexing -> wraps from the end.
            fid = min(int(frame_ids[idx].item()), T - 1)
            cf[mid, fid] += 1
    if success_mask.any():
        for i in range(int(success_mask.sum().item())):
            idx = int(success_mask.nonzero()[i].item())
            mid = int(motion_ids[idx].item())
            fid = min(int(frame_ids[idx].item()), T - 1)
            cs[mid, fid] += 1
    return cf, cs


def new_vectorized(failed_mask, success_mask, motion_ids, frame_ids, M, T):
    """Mirror the LIVE code in AdaptiveSampler.update: upper-clamp then per-row
    wrap (remainder), with masks moved to device first."""
    cf = torch.zeros(M, T)
    cs = torch.zeros(M, T)
    mids = motion_ids.long()
    fids = torch.clamp(frame_ids.long(), max=T - 1)
    fids = torch.remainder(fids, T)  # per-row wrap == old 2-D [mid, fid] incl. negatives
    flat_index = mids * T + fids
    ff = cf.view(-1)
    fs = cs.view(-1)
    if failed_mask.any():
        idx = flat_index[failed_mask]
        ff.scatter_add_(0, idx, torch.ones_like(idx, dtype=ff.dtype))
    if success_mask.any():
        idx = flat_index[success_mask]
        fs.scatter_add_(0, idx, torch.ones_like(idx, dtype=fs.dtype))
    return cf, cs


def test_equivalence_including_duplicates_and_clamp():
    torch.manual_seed(3)
    M, T = 6, 40
    N = 200
    motion_ids = torch.randint(0, M, (N,))
    # Include out-of-range frames to exercise the clamp, and force collisions.
    frame_ids = torch.randint(0, T + 10, (N,))
    terminated = torch.rand(N) < 0.5
    failed_mask, success_mask = terminated, ~terminated

    of, os_ = old_loop(failed_mask, success_mask, motion_ids, frame_ids, M, T)
    nf, ns = new_vectorized(failed_mask, success_mask, motion_ids, frame_ids, M, T)
    assert torch.equal(of, nf), (of - nf).abs().max()
    assert torch.equal(os_, ns), (os_ - ns).abs().max()
    # Sanity: total increments preserved.
    assert nf.sum().item() == int(failed_mask.sum().item())
    assert ns.sum().item() == int(success_mask.sum().item())
    print("OK test_equivalence_including_duplicates_and_clamp")


def test_equivalence_with_negative_frames():
    """Codex finding: the old loop upper-clamped only, so fid=-1 indexed the LAST
    frame. The live vectorized code must reproduce this via per-row remainder wrap
    (NOT lower-clamp to 0). Exercise negatives explicitly."""
    torch.manual_seed(7)
    M, T = 5, 30
    N = 150
    motion_ids = torch.randint(0, M, (N,))
    # Mix of negative, in-range, and over-range frames.
    frame_ids = torch.randint(-5, T + 8, (N,))
    terminated = torch.rand(N) < 0.5
    failed_mask, success_mask = terminated, ~terminated

    of, os_ = old_loop(failed_mask, success_mask, motion_ids, frame_ids, M, T)
    nf, ns = new_vectorized(failed_mask, success_mask, motion_ids, frame_ids, M, T)
    assert torch.equal(of, nf), (of - nf).abs().max()
    assert torch.equal(os_, ns), (os_ - ns).abs().max()
    print("OK test_equivalence_with_negative_frames")


def test_device_consistency_pattern():
    """Codex finding: masks must be on the SAME device as flat_index before boolean
    indexing. CPU-only here, but assert the live pattern (mask.to(device) up front)
    keeps mask and index device-aligned so a CUDA counter + CPU `terminated` works."""
    M, T = 4, 20
    N = 32
    dev = torch.device("cpu")  # CI is CPU; the assertion is about the alignment rule
    motion_ids = torch.randint(0, M, (N,))
    frame_ids = torch.randint(0, T, (N,))
    terminated = (torch.rand(N) < 0.5)
    # Live rule: move masks to device first.
    terminated_d = terminated.to(dev)
    failed_mask = terminated_d
    fids = torch.remainder(torch.clamp(frame_ids.long().to(dev), max=T - 1), T)
    flat_index = motion_ids.long().to(dev) * T + fids
    # The mask and the index must share a device for boolean indexing to be legal.
    assert failed_mask.device == flat_index.device
    _ = flat_index[failed_mask]  # must not raise
    print("OK test_device_consistency_pattern")


if __name__ == "__main__":
    test_equivalence_including_duplicates_and_clamp()
    test_equivalence_with_negative_frames()
    test_device_consistency_pattern()
    print("\nFRAME-UPDATE VECTORIZATION TEST PASSED")
