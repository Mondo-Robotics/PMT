"""Packed / memory-mapped streaming motion library (Phases 1–3 of the plan).

This module is **purely additive**: it imports from ``streaming_motion_lib`` and
subclasses its building blocks so the original file is untouched. It is, like its
parent, free of any ``isaaclab`` / ``omni`` import so it can be unit-tested with a
plain torch + numpy interpreter.

What it adds (see docs/streaming_motion_loader_plan.md):

* **Phase 1 — offline pre-pack + mmap.** ``prepack_motion_dataset`` flattens a set
  of ``.npz`` clips into one uncompressed memory-mappable binary per field plus a
  JSON sidecar index. ``PackedMotionStore`` opens those binaries with ``np.memmap``
  and serves a working-set swap as an mmap row-range read instead of a zlib decode.
  The OS page cache then reuses host RAM as a hot cache across swaps for free.
* **Phase 2 (global) — failure-cap.** ``CoverageCurriculum`` adds a
  ``max_prob_per_motion`` water-filling cap so one pathological clip cannot starve
  the rest of the dataset.
* **Phase 3 — coverage guarantee.** ``CoverageCurriculum`` blends a seeded,
  reshuffling epoch sweep (``_EpochIterator``) with the failure-weighted multinomial
  so every clip is visited within a bounded number of swaps, even zero-failure ones.

``PackedMotionStore`` overrides only the *read* path (``_read_many``); every other
behavior — atomic ``prepare_working_set`` / ``commit_prepared``, ragged flat layout,
per-motion frame clamp, getters — is inherited unchanged from ``FlatMotionStore``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# Relative import when loaded as a package module (the cluster/training path);
# fall back to a bare import when run standalone with the mdp dir on sys.path
# (the unit-test / benchmark path).
try:
    from .streaming_motion_lib import (
        DEFAULT_FPS,
        REQUIRED_ARRAY_KEYS,
        FlatMotionStore,
        GlobalCurriculum,
        LoadedMotion,
        MotionEntry,
        read_motion_npz,
    )
except ImportError:  # pragma: no cover - standalone fallback
    from streaming_motion_lib import (
        DEFAULT_FPS,
        REQUIRED_ARRAY_KEYS,
        FlatMotionStore,
        GlobalCurriculum,
        LoadedMotion,
        MotionEntry,
        read_motion_npz,
    )

# Bump when the on-disk layout changes in a backward-incompatible way. A pack
# written with a different version is rejected at open time (no silent OOB).
PACK_SCHEMA_VERSION = 1

# The flat binary file written per field. body_* keep the RAW (pre-body-selection)
# layout so a single pack is reusable across different body_indices selections,
# exactly matching read_motion_npz which selects bodies at read time.
_FIELD_NDIM = {
    "joint_pos": 2,
    "joint_vel": 2,
    "body_pos_w": 3,
    "body_quat_w": 3,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
}
INDEX_FILENAME = "index.json"


# ====================================================================== #
# Phase 1 — offline packer
# ====================================================================== #


def _np_dtype_name(use_fp16: bool) -> str:
    return "float16" if use_fp16 else "float32"


def _json_safe_metadata(metadata: dict) -> dict:
    """Extract per-clip metadata that survives a JSON sidecar round-trip.

    Terrain clips carry small scalar anchor fields (``transform_dx`` /
    ``transform_dy`` / ``transform_dyaw``) and string provenance (``source_*``).
    These must be preserved so a packed terrain dataset stays functionally usable
    (the command reads ``transform_*`` for mesh anchoring). Large arrays are NOT
    kept here — only scalars, short vectors, and strings — so the sidecar stays
    tiny. Anything else is dropped (and counted by the caller).
    """
    out: dict = {}
    dropped: list[str] = []
    for key, value in (metadata or {}).items():
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S") or arr.dtype == object:
            # String / object scalar (e.g. provenance paths).
            flat = arr.reshape(-1)
            out[key] = str(flat[0]) if flat.size == 1 else [str(x) for x in flat.tolist()]
        elif arr.dtype.kind in ("f", "i", "u", "b") and arr.size <= 4:
            flat = arr.reshape(-1)
            out[key] = float(flat[0]) if flat.size == 1 else [float(x) for x in flat.tolist()]
        else:
            dropped.append(key)
    if dropped:
        out["__dropped_keys__"] = dropped
    return out


def prepack_motion_dataset(
    file_paths: Sequence[str],
    out_dir: str,
    *,
    use_fp16: bool = False,
    num_workers: int = 16,
) -> dict:
    """Flatten ``file_paths`` (``.npz`` clips) into a packed mmap dataset in ``out_dir``.

    Writes one uncompressed ``<field>.bin`` per core field (RAW body layout, no body
    selection) plus ``index.json`` describing dtype, per-clip frame counts / fps /
    row offset, and field shapes. Returns the parsed index dict.

    Clips that fail to read are skipped (mirroring the loader's tolerance) and noted
    in the printed summary; their ids never enter the index.
    """
    os.makedirs(out_dir, exist_ok=True)
    dtype = np.float16 if use_fp16 else np.float32

    # First pass: read headers/arrays clip-by-clip on CPU, discover shapes, and
    # stream rows straight into the open binary files (so we never hold the whole
    # dataset in RAM during packing).
    paths = list(file_paths)
    if not paths:
        raise ValueError("prepack_motion_dataset: no input files")

    # Probe the first readable clip to fix per-field tail shapes (num_joints, bodies).
    cpu = torch.device("cpu")
    probe: LoadedMotion | None = None
    probe_idx = 0
    body_indices_all: Tensor | None = None
    for i, p in enumerate(paths):
        # Read with a full-body identity selection so RAW bodies are preserved.
        with np.load(p) as data:
            if not all(k in set(data.files) for k in REQUIRED_ARRAY_KEYS):
                continue
            n_body_raw = int(np.asarray(data["body_pos_w"]).shape[1])
        body_indices_all = torch.arange(n_body_raw, dtype=torch.long)
        probe = read_motion_npz(p, body_indices_all, torch.float32, cpu)
        if probe is not None:
            probe_idx = i
            break
    if probe is None or body_indices_all is None:
        raise RuntimeError("prepack_motion_dataset: no readable clips found")

    num_joints = int(probe.joint_pos.shape[1])
    num_bodies_raw = int(probe.body_pos_w.shape[1])
    tail_shapes = {
        "joint_pos": (num_joints,),
        "joint_vel": (num_joints,),
        "body_pos_w": (num_bodies_raw, 3),
        "body_quat_w": (num_bodies_raw, 4),
        "body_lin_vel_w": (num_bodies_raw, 3),
        "body_ang_vel_w": (num_bodies_raw, 3),
    }

    handles = {k: open(os.path.join(out_dir, f"{k}.bin"), "wb") for k in tail_shapes}
    entries: list[dict] = []
    offset = 0
    n_skipped = 0
    try:
        for i, p in enumerate(paths):
            m = probe if i == probe_idx else read_motion_npz(p, body_indices_all, torch.float32, cpu)
            if m is None:
                n_skipped += 1
                continue
            t = int(m.num_frames)
            fields = {
                "joint_pos": m.joint_pos,
                "joint_vel": m.joint_vel,
                "body_pos_w": m.body_pos_w,
                "body_quat_w": m.body_quat_w,
                "body_lin_vel_w": m.body_lin_vel_w,
                "body_ang_vel_w": m.body_ang_vel_w,
            }
            for k, tensor in fields.items():
                arr = np.ascontiguousarray(tensor.numpy().astype(dtype))
                if arr.shape != (t, *tail_shapes[k]):
                    raise ValueError(
                        f"{p}: field {k} shape {arr.shape} != {(t, *tail_shapes[k])}"
                    )
                handles[k].write(arr.tobytes())
            entries.append(
                {
                    "path": p,
                    "num_frames": t,
                    "fps": float(m.fps),
                    "offset": offset,
                    "metadata": _json_safe_metadata(m.metadata),
                }
            )
            offset += t
    finally:
        for h in handles.values():
            h.close()

    index = {
        "schema_version": PACK_SCHEMA_VERSION,
        "dtype": _np_dtype_name(use_fp16),
        "num_joints": num_joints,
        "num_bodies_raw": num_bodies_raw,
        "total_frames": offset,
        "tail_shapes": {k: list(v) for k, v in tail_shapes.items()},
        "entries": entries,
    }
    with open(os.path.join(out_dir, INDEX_FILENAME), "w") as f:
        json.dump(index, f)
    print(
        f"[prepack] wrote {len(entries)} clips ({offset} frames) to {out_dir}; "
        f"skipped {n_skipped}; dtype={index['dtype']}"
    )
    del num_workers  # reserved for a future parallel packer; single-stream for now.
    return index


def load_pack_index(pack_dir: str) -> dict:
    """Read + validate the sidecar index without opening the big binaries (Phase 1/2).

    This is the O(1) replacement for the per-file npz-header scan: one JSON load
    yields fps / length / offset for the whole dataset.
    """
    index_path = os.path.join(pack_dir, INDEX_FILENAME)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"pack index not found: {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    version = int(index.get("schema_version", -1))
    if version != PACK_SCHEMA_VERSION:
        raise ValueError(
            f"pack schema_version {version} != supported {PACK_SCHEMA_VERSION}; "
            f"re-run prepack_motion_dataset on {pack_dir}"
        )
    if index.get("dtype") not in ("float16", "float32"):
        raise ValueError(f"pack index has unsupported dtype {index.get('dtype')!r}")
    return index


# ====================================================================== #
# Phase 1 — memory-mapped store
# ====================================================================== #


@dataclass
class _PackHandles:
    mmaps: dict
    offsets: np.ndarray  # [N] row offset of each global clip
    lengths: np.ndarray  # [N] frame count of each global clip
    dtype: np.dtype
    num_bodies_raw: int


class PackedMotionStore(FlatMotionStore):
    """``FlatMotionStore`` whose working-set reads come from an mmap pack, not zlib.

    Only the read path is overridden. ``set_packed_index`` opens the binaries once
    (cheap, lazy: pages are faulted in on access), builds a ``path -> (offset,len)``
    map, and installs the normal global index so ``num_unique_motions`` /
    ``GlobalCurriculum`` work exactly as before. ``_read_many`` then slices the mmap
    by global row range and applies ``body_indices`` at read time — same semantics as
    ``read_motion_npz`` — so all of ``prepare_working_set`` / ``commit_prepared`` /
    atomic-swap logic is inherited unchanged.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pack: _PackHandles | None = None
        self._path_to_global: dict[str, int] = {}
        self._metadata_by_global: dict[int, dict] = {}

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def set_packed_index(
        self, pack_dir: str, body_indices: Tensor | Sequence[int]
    ) -> None:
        """Open the pack, build the global index from its sidecar, set body selection."""
        index = load_pack_index(pack_dir)
        np_dtype = np.dtype(index["dtype"])
        tail_shapes = {k: tuple(v) for k, v in index["tail_shapes"].items()}
        total_frames = int(index["total_frames"])

        mmaps = {}
        for k, tail in tail_shapes.items():
            shape = (total_frames, *tail)
            path = os.path.join(pack_dir, f"{k}.bin")
            expected = int(np.prod(shape)) * np_dtype.itemsize
            actual = os.path.getsize(path)
            if actual != expected:
                raise ValueError(
                    f"pack file {k}.bin size {actual} != expected {expected} "
                    f"(shape {shape}, dtype {np_dtype}); pack is corrupt or stale"
                )
            mmaps[k] = np.memmap(path, dtype=np_dtype, mode="r", shape=shape)

        entries = index["entries"]
        offsets = np.array([e["offset"] for e in entries], dtype=np.int64)
        lengths = np.array([e["num_frames"] for e in entries], dtype=np.int64)
        self._pack = _PackHandles(
            mmaps=mmaps,
            offsets=offsets,
            lengths=lengths,
            dtype=np_dtype,
            num_bodies_raw=int(index["num_bodies_raw"]),
        )
        self._path_to_global = {e["path"]: g for g, e in enumerate(entries)}
        self._metadata_by_global = {
            g: dict(e.get("metadata", {})) for g, e in enumerate(entries)
        }

        motion_index = [
            MotionEntry(path=e["path"], num_frames=int(e["num_frames"]), fps=float(e["fps"]))
            for e in entries
        ]
        self.set_index(motion_index, body_indices)

    # ------------------------------------------------------------------ #
    # Read path override (mmap row-range slice instead of zlib decode)
    # ------------------------------------------------------------------ #

    def _read_one_packed(self, global_id: int, body_idx_np: np.ndarray) -> LoadedMotion:
        assert self._pack is not None
        off = int(self._pack.offsets[global_id])
        t = int(self._pack.lengths[global_id])
        sl = slice(off, off + t)
        mm = self._pack.mmaps

        def take(field: str, select_bodies: bool) -> Tensor:
            view = mm[field][sl]
            if select_bodies:
                view = view[:, body_idx_np]
            # Copy out of the read-only mmap into an owned, writable, contiguous
            # float32 array (avoids the non-writable-tensor warning and pins the
            # pages we need into a normal allocation for the downstream GPU copy).
            arr = np.array(view, dtype=np.float32, order="C", copy=True)
            return torch.from_numpy(arr).to(self.dtype)

        entry = self.index[global_id]
        return LoadedMotion(
            fps=float(entry.fps),
            num_frames=t,
            joint_pos=take("joint_pos", False),
            joint_vel=take("joint_vel", False),
            body_pos_w=take("body_pos_w", True),
            body_quat_w=take("body_quat_w", True),
            body_lin_vel_w=take("body_lin_vel_w", True),
            body_ang_vel_w=take("body_ang_vel_w", True),
            metadata=dict(self._metadata_by_global.get(global_id, {})),
            source_file=entry.path,
        )

    def _read_many(
        self, file_paths: Sequence[str], body_indices: Tensor
    ) -> list[LoadedMotion]:
        """Slice the requested clips out of the mmap (no decompression)."""
        if self._pack is None:
            # Not in packed mode (e.g. load_files used on raw npz) — defer to parent.
            return super()._read_many(file_paths, body_indices)
        body_idx_np = np.asarray(body_indices.detach().cpu().numpy(), dtype=np.int64)
        out: list[LoadedMotion] = []
        for p in file_paths:
            g = self._path_to_global.get(p)
            if g is None:
                print(f"[PackedMotionStore] path not in pack, skipping: {p}")
                continue
            out.append(self._read_one_packed(g, body_idx_np))
        return out


# ====================================================================== #
# Phase 3 — seeded reshuffling epoch sweep
# ====================================================================== #


class _EpochIterator:
    """Deterministic, fair round-robin over ``[0, n)`` that reshuffles each epoch.

    A fixed-seed ``torch.Generator`` drives the permutation so the same seed always
    yields the same sequence (reproducible coverage). ``draw(k)`` returns the next
    ``k`` unique-within-epoch ids, crossing epoch boundaries as needed (so across two
    epochs an id may repeat — that is correct round-robin, not a bug).
    """

    def __init__(self, n: int, device: torch.device | str = "cpu", *, seed: int = 0):
        self.n = int(n)
        self.device = torch.device(device)
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(int(seed))
        self._perm = self._shuffle()
        self._cursor = 0
        self.epochs_completed = 0

    def _shuffle(self) -> Tensor:
        if self.n == 0:
            return torch.empty(0, dtype=torch.long)
        return torch.randperm(self.n, generator=self._gen)

    def draw(self, k: int) -> Tensor:
        """Return the next ``k`` ids (on ``self.device``), reshuffling on drain."""
        k = int(k)
        if self.n == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        picks: list[Tensor] = []
        remaining = k
        while remaining > 0:
            avail = self._perm[self._cursor : self._cursor + remaining]
            picks.append(avail)
            taken = avail.numel()
            self._cursor += taken
            remaining -= taken
            if self._cursor >= self.n:
                self._perm = self._shuffle()
                self._cursor = 0
                self.epochs_completed += 1
        return torch.cat(picks).to(self.device)


# ====================================================================== #
# Phase 2 (global) + Phase 3 — failure cap + coverage curriculum
# ====================================================================== #


class CoverageCurriculum(GlobalCurriculum):
    """``GlobalCurriculum`` with a per-motion probability cap and an epoch sweep.

    * ``max_prob_per_motion`` (Phase 2): water-fill cap so no single clip exceeds the
      cap in the sampling distribution — prevents one broken clip from dominating.
      ``None`` disables the cap (identical to the parent distribution).
    * ``coverage_ratio`` (Phase 3): fraction of each working set drawn from a seeded
      reshuffling epoch sweep (guaranteed coverage); the remainder is drawn from the
      capped failure-weighted multinomial. ``0.0`` reproduces the parent exactly.

    The two groups are kept disjoint so the working set stays unique (the resident set
    must not contain duplicate clips), matching the parent contract.
    """

    def __init__(
        self,
        num_unique_motions: int,
        device: torch.device | str = "cpu",
        *,
        beta: float = 1.0,
        alpha: float = 0.01,
        uniform_ratio: float = 0.1,
        max_prob_per_motion: float | None = None,
        coverage_ratio: float = 0.0,
        coverage_seed: int = 0,
    ) -> None:
        super().__init__(
            num_unique_motions,
            device,
            beta=beta,
            alpha=alpha,
            uniform_ratio=uniform_ratio,
        )
        if max_prob_per_motion is not None and not (0.0 < max_prob_per_motion <= 1.0):
            raise ValueError("max_prob_per_motion must be in (0, 1] or None")
        if not (0.0 <= coverage_ratio <= 1.0):
            raise ValueError("coverage_ratio must be in [0, 1]")
        self.max_prob_per_motion = max_prob_per_motion
        self.coverage_ratio = float(coverage_ratio)
        self._epoch = _EpochIterator(
            self.num_unique_motions, device=self.device, seed=coverage_seed
        )

    # ---------------------------- Phase 2 ---------------------------- #

    @staticmethod
    def _apply_cap(probs: Tensor, cap: float) -> Tensor:
        """Water-filling cap: clamp the largest entries to ``cap`` and redistribute
        the freed mass over the uncapped entries until none exceed ``cap`` (or the cap
        is infeasible, i.e. ``cap * n <= 1``, in which case the closest feasible
        distribution — uniform at ``1/n`` — is returned).

        Redistribution is proportional to the uncapped entries' current mass; when that
        mass is ~0 (e.g. one clip carried essentially all probability), the freed mass
        is spread uniformly over the uncapped set so the result still sums to 1 with no
        entry above ``cap``.
        """
        n = probs.numel()
        if n == 0:
            return probs
        if cap * n <= 1.0:
            # Infeasible to keep sum=1 with every entry <= cap; uniform is the cap.
            return torch.full_like(probs, 1.0 / n)
        p = (probs / probs.sum()).clone()
        capped = torch.zeros(n, dtype=torch.bool, device=p.device)
        for _ in range(n + 1):
            over = (p > cap + 1e-15) & (~capped)
            if not bool(over.any()):
                break
            capped = capped | over
            p[capped] = cap
            free_mass = 1.0 - cap * float(int(capped.sum()))
            uncapped = ~capped
            if not bool(uncapped.any()):
                break
            denom = float(p[uncapped].sum())
            if denom > 0:
                p[uncapped] = p[uncapped] * (free_mass / denom)
            else:
                p[uncapped] = free_mass / float(int(uncapped.sum()))
        return p / p.sum()

    def probabilities(self) -> Tensor:
        probs = super().probabilities()
        if self.max_prob_per_motion is None:
            return probs
        return self._apply_cap(probs.double(), float(self.max_prob_per_motion)).to(
            probs.dtype
        )

    # ---------------------------- Phase 3 ---------------------------- #

    def _sample_failure_weighted(self, k: int, exclude: Tensor | None) -> Tensor:
        """Draw ``k`` unique ids from the capped distribution, excluding ``exclude``."""
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        probs = self.probabilities().clone()
        if exclude is not None and exclude.numel() > 0:
            probs[exclude.to(self.device)] = 0.0
        total = float(probs.sum())
        if total <= 0 or not torch.isfinite(probs).all():
            # Degenerate (everything excluded / underflow): fall back to uniform over
            # the still-available ids.
            mask = torch.ones(self.num_unique_motions, device=self.device)
            if exclude is not None and exclude.numel() > 0:
                mask[exclude.to(self.device)] = 0.0
            if float(mask.sum()) <= 0:
                return torch.empty(0, dtype=torch.long, device=self.device)
            probs = mask / mask.sum()
        else:
            probs = probs / total
        k = min(k, int((probs > 0).sum().item()))
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.multinomial(probs, k, replacement=False)

    def sample_working_set(self, num_motions: int) -> Tensor:
        n = self.num_unique_motions
        if num_motions >= n:
            return torch.arange(n, device=self.device)
        if self.coverage_ratio <= 0.0:
            # No coverage quota: capped failure-weighted draw only.
            return self._sample_failure_weighted(num_motions, exclude=None)

        n_cov = min(num_motions, int(round(self.coverage_ratio * num_motions)))
        cov_ids = self._epoch.draw(n_cov)
        # Dedup the coverage ids against themselves (rare cross-epoch repeat within one
        # draw) so the resident set stays unique.
        cov_ids = torch.unique(cov_ids)
        n_rest = num_motions - cov_ids.numel()
        rest_ids = self._sample_failure_weighted(n_rest, exclude=cov_ids)
        ids = torch.cat([cov_ids, rest_ids])
        # If exclusion shrank the pool below the target, top up from the epoch sweep
        # so the working set is always full when the dataset can fill it.
        if ids.numel() < num_motions:
            seen = set(ids.tolist())
            while ids.numel() < num_motions:
                extra = self._epoch.draw(num_motions - ids.numel())
                extra = torch.tensor(
                    [int(x) for x in extra.tolist() if int(x) not in seen],
                    dtype=torch.long,
                    device=self.device,
                )
                if extra.numel() == 0:
                    break
                seen.update(extra.tolist())
                ids = torch.cat([ids, extra])
        return ids
