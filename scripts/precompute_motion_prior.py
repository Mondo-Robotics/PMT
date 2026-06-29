#!/usr/bin/env python3
"""Offline precompute of the per-bin difficulty prior (Phase 2).

Reads every robot motion clip under the given paths, computes cheap per-bin
kinematic features (high-frequency energy ratio of joint velocity, RMS jerk, body
acceleration proxy), normalizes them across the whole dataset, and writes a
PATH-KEYED cache to disk. Training then loads this cache and injects the prior into
the hybrid sampler's failed_bin_count so high-dynamic segments are sampled more even
before any online failure statistics accumulate.

This script does NOT import isaaclab/Isaac Sim — it uses the pure-torch streaming
loader (read_motion_npz) and the pure-torch prior module, so it runs standalone:

    conda run -n cluster_isaaclab python scripts/precompute_motion_prior.py \
        --motion_paths /data/.../sonic /home/.../snap_robot \
        --out /data/.../adaptive_prior.pt --bin_duration 1.0

Cache layout (torch.save dict): {version, bin_size, max_bins, w_*, paths:[abs...],
prior: FloatTensor[N, max_bins], num_bins: LongTensor[N]}. Training aligns rows to
its motion index by absolute path.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module(name: str, relpath: str):
    """Import a pmt_tasks module by FILE PATH (avoids the package __init__ that pulls
    isaaclab). Registers in sys.modules so dataclasses resolve."""
    p = _ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# Pure-torch deps loaded by path (no isaaclab).
slib = _load_module("slib_precompute", "pmt_tasks/mdp/commands/streaming_motion_lib.py")
prior_mod = _load_module("prior_precompute", "pmt_tasks/mdp/commands/adaptive_sampling_prior.py")


def _discover(motion_paths):
    """Discover robot clips. Reuse find_motion_files if importable, else glob."""
    try:
        from pmt_tasks.utils.motion_paths import find_motion_files  # may pull light deps only
        res = find_motion_files(motion_paths=motion_paths, strict=False)
        return [os.path.abspath(f) for f in res.files]
    except Exception:
        import glob
        files = []
        for root in motion_paths:
            root = os.path.abspath(os.path.expanduser(root))
            if os.path.isfile(root) and root.endswith(".npz"):
                files.append(root)
            else:
                for f in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
                    parts = pathlib.Path(f).parts
                    if not any(p.lower().startswith("human") for p in parts):
                        files.append(os.path.abspath(f))
        return files


def main():
    ap = argparse.ArgumentParser(description="Precompute per-bin difficulty prior (Phase 2).")
    ap.add_argument("--motion_paths", nargs="+", required=True, help="Dirs/files of robot npz clips.")
    ap.add_argument("--out", required=True, help="Output cache .pt path.")
    ap.add_argument("--bin_duration", type=float, default=1.0, help="Seconds per bin.")
    ap.add_argument("--default_fps", type=float, default=50.0)
    ap.add_argument(
        "--bin_size_frames", type=int, default=0,
        help="Force a FIXED frame-count bin size for ALL clips (overrides fps*bin_duration). "
             "MUST equal the trainer's round(cfg.motion_fps * cfg.bin_duration) — the sampler "
             "bins by FRAME INDEX, so a fixed frame grid is the correct alignment and makes a "
             "mixed-fps corpus representable in a single-grid cache. The cache stores this as "
             "bin_size. Recommended for datasets with mixed fps (e.g. 30 + 50).",
    )
    ap.add_argument("--w_hf", type=float, default=0.4)
    ap.add_argument("--w_jerk", type=float, default=0.3)
    ap.add_argument("--w_accel", type=float, default=0.3)
    ap.add_argument("--num_body", type=int, default=30, help="Body count to read (indices 0..num_body-1).")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only process first N clips (0=all).")
    args = ap.parse_args()

    files = _discover(args.motion_paths)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"[precompute] no robot clips found under {args.motion_paths}")
    print(f"[precompute] {len(files)} clips")

    body_indices = torch.arange(args.num_body, dtype=torch.long)
    cpu = torch.device("cpu")

    per_clip = []
    kept_paths = []
    bin_size_ref = None
    mixed_bin_sizes = set()
    for i, f in enumerate(files):
        m = slib.read_motion_npz(f, body_indices, torch.float32, cpu)
        if m is None:
            print(f"[precompute] skip unreadable {f}")
            continue
        fps = float(m.fps) if m.fps and m.fps > 0 else args.default_fps
        # Bin size in FRAMES. With --bin_size_frames we force the sampler's fixed frame
        # grid for every clip (correct: the sampler bins by frame index, not seconds),
        # which makes a mixed-fps corpus representable. Otherwise derive from this clip's
        # fps (which requires a single-fps dataset, enforced below).
        if args.bin_size_frames > 0:
            bin_size = int(args.bin_size_frames)
        else:
            bin_size = max(1, int(round(fps * args.bin_duration)))
        # The cache stores ONE dataset-wide bin_size; the trainer validates it against
        # round(cfg.motion_fps * cfg.bin_duration). A mixed-fps dataset would make the
        # per-bin column grid ambiguous, so we fail loud rather than emit a cache that
        # silently misaligns prior columns to sampler bins at train time.
        mixed_bin_sizes.add(bin_size)
        if bin_size_ref is None:
            bin_size_ref = bin_size
        feats = prior_mod.compute_clip_bin_features(
            joint_vel=m.joint_vel, body_lin_vel_w=m.body_lin_vel_w, fps=fps, bin_size=bin_size
        )
        per_clip.append(feats)
        kept_paths.append(f)
        if (i + 1) % 200 == 0:
            print(f"[precompute] {i + 1}/{len(files)} ...")

    if len(mixed_bin_sizes) > 1:
        raise SystemExit(
            f"[precompute] clips have MIXED bin sizes {sorted(mixed_bin_sizes)} "
            f"(differing fps at bin_duration={args.bin_duration}). A single-grid cache "
            f"cannot represent them. Re-run on a single-fps subset, or extend the cache "
            f"format to per-clip bin grids."
        )

    if not per_clip:
        raise SystemExit("[precompute] no clips produced features")

    max_bins = max(f.num_bins for f in per_clip)
    prior = prior_mod.combine_features_to_prior(
        per_clip, max_bins, w_hf=args.w_hf, w_jerk=args.w_jerk, w_accel=args.w_accel
    )
    num_bins = torch.tensor([f.num_bins for f in per_clip], dtype=torch.long)

    out = {
        "version": prior_mod.PRIOR_CACHE_VERSION,
        "bin_size": int(bin_size_ref),
        "bin_duration": float(args.bin_duration),
        "max_bins": int(max_bins),
        "w_hf": args.w_hf, "w_jerk": args.w_jerk, "w_accel": args.w_accel,
        "paths": kept_paths,
        "prior": prior,            # [N, max_bins] in [0,1]
        "num_bins": num_bins,      # [N]
    }
    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(out, out_path)
    nz = (prior.sum(dim=1) > 0).sum().item()
    print(f"[precompute] wrote {out_path}: prior {tuple(prior.shape)}, "
          f"{nz}/{len(kept_paths)} clips with nonzero prior, "
          f"mean={prior[prior>0].mean().item():.3f}" if (prior > 0).any() else
          f"[precompute] wrote {out_path}: prior {tuple(prior.shape)} (all zero)")


if __name__ == "__main__":
    main()
