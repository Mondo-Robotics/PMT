"""Streaming motion library — pure-torch data layer (no Isaac Lab dependency).

This module is the foundation for a memory-bounded multi-motion loader. It is
deliberately free of any ``isaaclab``/``omni`` imports so it can be unit-tested
with a plain Python + torch interpreter (no simulator runtime required).

Step 1 scope (this file): ``FlatMotionStore`` — flat *ragged* storage that
replaces the dense ``[N, max_T, ...]`` padded layout of ``MotionDataStore``.
Instead of padding every clip to ``max_motion_length`` (wasting
``max_T / mean_T`` memory), all clips are concatenated into ``[sum(T_i), ...]``
tensors and addressed via a ``length_starts`` offset table, exactly like the
SONIC ``MotionLibBase`` (motion_lib_base.py:1479-1528).

Robot-only: human motion paths are intentionally not handled here.

Later steps add: lazy file indexing (Step 2), a swappable working set
(Step 3), and a two-level sampler (Step 4). Those build on this class.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import zipfile
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.lib.format as npy_format
import torch
from torch import Tensor

# Per-clip array fields required to build a motion (all read into the flat store).
REQUIRED_ARRAY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

# Default fps when a clip omits the ``fps`` field. MUST match between the index
# (read_npz_header) and the loader (read_motion_npz) so a file indexed with the
# default never crashes when actually loaded.
DEFAULT_FPS = 30.0

# Core npz keys that constitute a robot motion clip. Any other key in the file
# is treated as opaque per-clip metadata (e.g. terrain ``transform_*`` scalars).
CORE_KEYS = (
    "fps",
    *REQUIRED_ARRAY_KEYS,
    # Excluded from metadata so a stray large human "positions" array (present in
    # some npz) is not retained and silently undo the memory savings.
    "positions",
)


@dataclass
class LoadedMotion:
    """A single motion's tensors after reading + body-index selection.

    These are short-lived: they exist only between file read and concatenation
    into the flat store, after which they are dropped so memory is not doubled.
    """

    fps: float
    num_frames: int
    joint_pos: Tensor
    joint_vel: Tensor
    body_pos_w: Tensor
    body_quat_w: Tensor
    body_lin_vel_w: Tensor
    body_ang_vel_w: Tensor
    metadata: dict = field(default_factory=dict)
    source_file: str = ""


@dataclass
class PreparedWorkingSet:
    """A fully-built working set NOT yet installed into the live store.

    Produced by ``prepare_working_set`` (which never mutates the live store) and
    consumed by ``commit_prepared``. Holding the built flat tensors here lets a
    swap be atomic: if prepare fails, the live store is untouched; only commit
    frees the old buffers and installs the new ones.
    """

    flat_joint_pos: Tensor
    flat_joint_vel: Tensor
    flat_body_pos_w: Tensor
    flat_body_quat_w: Tensor
    flat_body_lin_vel_w: Tensor
    flat_body_ang_vel_w: Tensor
    motion_lengths: Tensor
    length_starts: Tensor
    num_motions: int
    num_joints: int
    num_bodies: int
    max_motion_length: int
    fps: list
    metadata: list
    source_files: list
    working_to_global: Tensor
    global_to_working: dict


def read_motion_npz(
    file_path: str,
    body_indices: Tensor,
    dtype: torch.dtype,
    storage_device: torch.device,
) -> LoadedMotion | None:
    """Read one robot motion ``.npz`` and select the requested bodies.

    Returns ``None`` if the file is missing or unreadable. Pure function — safe
    to call from worker threads (used by the parallel loader in Step 3).
    """
    if not os.path.isfile(file_path):
        print(f"[FlatMotionStore] File not found: {file_path}")
        return None
    # Wrap the WHOLE parse (not just np.load): a missing key, shape mismatch, or
    # out-of-range body index would otherwise throw in a worker and abort the swap.
    try:
        # Context manager closes the underlying zip handle — matters for the
        # large threaded/persistent loaders (avoids fd / RSS growth).
        with np.load(file_path) as data:
            files = set(data.files)
            for key in REQUIRED_ARRAY_KEYS:
                if key not in files:
                    print(f"[FlatMotionStore] {file_path} missing '{key}'; skipping")
                    return None

            fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in files else DEFAULT_FPS

            jp_np = np.asarray(data["joint_pos"])
            jv_np = np.asarray(data["joint_vel"])
            bp_np = np.asarray(data["body_pos_w"])
            bq_np = np.asarray(data["body_quat_w"])
            blv_np = np.asarray(data["body_lin_vel_w"])
            bav_np = np.asarray(data["body_ang_vel_w"])

            num_frames = int(jp_np.shape[0])
            if num_frames == 0:
                # length-1 == -1 in the frame clamp would wrap to another clip.
                print(f"[FlatMotionStore] Skipping zero-frame motion: {file_path}")
                return None

            # All per-frame arrays must agree on the time dimension.
            for name, arr in (
                ("joint_vel", jv_np), ("body_pos_w", bp_np), ("body_quat_w", bq_np),
                ("body_lin_vel_w", blv_np), ("body_ang_vel_w", bav_np),
            ):
                if arr.shape[0] != num_frames:
                    print(f"[FlatMotionStore] {file_path} '{name}' frames {arr.shape[0]} "
                          f"!= joint_pos {num_frames}; skipping")
                    return None

            # Body dim must cover the requested indices; quat last dim must be 4.
            max_body = int(body_indices.max().item()) if body_indices.numel() else -1
            n_body = bp_np.shape[1]
            if max_body >= n_body:
                print(f"[FlatMotionStore] {file_path} body dim {n_body} < requested "
                      f"index {max_body}; skipping")
                return None
            if bq_np.shape[-1] != 4:
                print(f"[FlatMotionStore] {file_path} body_quat last dim "
                      f"{bq_np.shape[-1]} != 4; skipping")
                return None

            metadata = {k: data[k] for k in files if k not in CORE_KEYS}

        body_idx = body_indices.to(storage_device)
        joint_pos = torch.as_tensor(jp_np, dtype=dtype, device=storage_device)
        joint_vel = torch.as_tensor(jv_np, dtype=dtype, device=storage_device)
        body_pos_w = torch.as_tensor(bp_np, dtype=dtype, device=storage_device)[:, body_idx]
        body_quat_w = torch.as_tensor(bq_np, dtype=dtype, device=storage_device)[:, body_idx]
        body_lin_vel_w = torch.as_tensor(blv_np, dtype=dtype, device=storage_device)[:, body_idx]
        body_ang_vel_w = torch.as_tensor(bav_np, dtype=dtype, device=storage_device)[:, body_idx]
    except Exception as e:  # noqa: BLE001
        print(f"[FlatMotionStore] Failed to load/parse {file_path}: {e}")
        return None

    return LoadedMotion(
        fps=fps,
        num_frames=num_frames,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        metadata=metadata,
        source_file=file_path,
    )


# --- Process-pool decode (P2): break the GIL on zlib decompression --------- #
# The worker is a TOP-LEVEL function (picklable for spawn) and is NUMPY-ONLY —
# it never imports torch or touches CUDA, so a spawn pool created after Isaac Sim
# init is safe. It returns fp16 numpy arrays (bodies already selected) which the
# main process turns into tensors. fp16 halves the IPC bytes shipped back.

# numpy dtype for decode output, set per-process by the pool initializer.
_WORKER_FP16 = True


def _proc_pool_init(use_fp16: bool) -> None:
    global _WORKER_FP16
    _WORKER_FP16 = bool(use_fp16)


def read_motion_arrays(args):
    """Process-pool worker: decode one npz to fp16 numpy with bodies selected.

    Returns ``(path, fps, jp, jv, bp, bq, blv, bav)`` or ``None`` on any failure.
    Mirrors ``read_motion_npz`` validation but stays in numpy (no torch/CUDA).
    """
    file_path, body_indices_list = args
    out_dtype = np.float16 if _WORKER_FP16 else np.float32
    if not os.path.isfile(file_path):
        return None
    try:
        body = np.asarray(body_indices_list, dtype=np.int64)
        with np.load(file_path) as data:
            files = set(data.files)
            for key in REQUIRED_ARRAY_KEYS:
                if key not in files:
                    return None
            fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in files else DEFAULT_FPS
            jp = np.asarray(data["joint_pos"])
            jv = np.asarray(data["joint_vel"])
            bp = np.asarray(data["body_pos_w"])
            bq = np.asarray(data["body_quat_w"])
            blv = np.asarray(data["body_lin_vel_w"])
            bav = np.asarray(data["body_ang_vel_w"])
            n = int(jp.shape[0])
            if n == 0:
                return None
            for arr in (jv, bp, bq, blv, bav):
                if arr.shape[0] != n:
                    return None
            if (body.max() if body.size else -1) >= bp.shape[1] or bq.shape[-1] != 4:
                return None
            return (
                file_path, fps,
                jp.astype(out_dtype), jv.astype(out_dtype),
                bp[:, body].astype(out_dtype), bq[:, body].astype(out_dtype),
                blv[:, body].astype(out_dtype), bav[:, body].astype(out_dtype),
            )
    except Exception:  # noqa: BLE001
        return None


def _loaded_from_arrays(rec, dtype: torch.dtype, storage_device: torch.device) -> LoadedMotion:
    """Build a LoadedMotion from a process-worker numpy record (main process)."""
    path, fps, jp, jv, bp, bq, blv, bav = rec
    t = lambda a: torch.as_tensor(a, dtype=dtype, device=storage_device)  # noqa: E731
    return LoadedMotion(
        fps=fps, num_frames=int(jp.shape[0]),
        joint_pos=t(jp), joint_vel=t(jv),
        body_pos_w=t(bp), body_quat_w=t(bq),
        body_lin_vel_w=t(blv), body_ang_vel_w=t(bav),
        metadata={}, source_file=path,
    )


@dataclass
class MotionEntry:
    """Lightweight global index entry — built without loading array bodies.

    One per motion clip. ``num_frames``/``fps`` come from the npz header so the
    full dataset can be indexed cheaply at startup; arrays are read only when the
    clip enters the working set (Step 3).
    """

    path: str
    num_frames: int
    fps: float


def _read_npy_header_shape(f) -> tuple:
    """Parse an npy member's shape using public version-dispatched readers.

    Avoids the private ``_read_array_header`` for robustness across numpy
    versions; falls back to it only for unforeseen header versions.
    """
    version = npy_format.read_magic(f)
    if version == (1, 0):
        shape, _fortran, _dtype = npy_format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, _fortran, _dtype = npy_format.read_array_header_2_0(f)
    else:  # (3, 0) or future — private fallback
        shape, _fortran, _dtype = npy_format._read_array_header(f, version)
    return shape


def read_npz_header(file_path: str) -> tuple[int, float] | None:
    """Read ``(num_frames, fps)`` from an npz WITHOUT loading the big arrays.

    Validates that ALL required array members are present (so a partial/bad npz
    is rejected at index time, not later in the working-set loader), reads
    ``num_frames`` from ``joint_pos.npy``'s header (shape only), and reads the
    tiny ``fps`` array in full (defaulting to ``DEFAULT_FPS`` if absent — matching
    ``read_motion_npz``). Returns ``None`` on any error. Pure / thread-safe.
    """
    try:
        with zipfile.ZipFile(file_path) as z:
            names = set(z.namelist())
            # Reject incomplete clips up front — every required array must exist.
            for key in REQUIRED_ARRAY_KEYS:
                if f"{key}.npy" not in names:
                    print(f"[MotionIndex] Missing '{key}' in {file_path}; skipping")
                    return None
            with z.open("joint_pos.npy") as f:
                shape = _read_npy_header_shape(f)
            num_frames = int(shape[0])
            fps = DEFAULT_FPS
            if "fps.npy" in names:
                with z.open("fps.npy") as f:
                    fps = float(np.asarray(npy_format.read_array(f)).reshape(-1)[0])
        return num_frames, fps
    except Exception as e:  # noqa: BLE001
        print(f"[MotionIndex] Failed to read header {file_path}: {e}")
        return None


def build_motion_index(file_paths: Sequence[str], num_workers: int = 16) -> list[MotionEntry]:
    """Build the global index over ``file_paths`` using a bounded thread pool.

    Header reads are I/O-bound and release the GIL, so threads (not processes —
    no fork-after-CUDA hazard) parallelize the scan safely. Zero-frame clips are
    dropped here so they never reach the flat store. Order follows ``file_paths``.
    """
    paths = list(file_paths)
    if not paths:
        return []
    num_workers = max(1, min(num_workers, len(paths)))
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        headers = list(ex.map(read_npz_header, paths))

    entries: list[MotionEntry] = []
    n_skipped = 0
    for path, hdr in zip(paths, headers):
        if hdr is None:
            n_skipped += 1
            continue
        num_frames, fps = hdr
        if num_frames <= 0:
            n_skipped += 1
            continue
        entries.append(MotionEntry(path=path, num_frames=num_frames, fps=fps))
    if n_skipped:
        print(f"[MotionIndex] Skipped {n_skipped} unreadable/empty files")
    print(f"[MotionIndex] Indexed {len(entries)} motions from {len(paths)} files")
    return entries


class FlatMotionStore:
    """Flat *ragged* motion storage with per-motion frame clamping.

    All clips are concatenated along the time axis into contiguous tensors:

        ``_flat_joint_pos``     : [sum(T_i), num_joints]
        ``_flat_body_pos_w``    : [sum(T_i), num_bodies, 3]
        ... etc.

    A clip ``m`` occupies rows ``[length_starts[m], length_starts[m] + T_m)``.
    A ``(motion_id, frame_id)`` pair maps to the global row
    ``length_starts[motion_id] + clamp(frame_id, 0, T_m - 1)``.

    This holds exactly ``sum(T_i)`` frames — no padding — vs. the dense layout's
    ``num_motions * max_motion_length``.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        storage_device: str | torch.device = "cuda",
        use_fp16: bool = False,
        num_workers: int = 16,
        use_process_pool: bool = False,
    ):
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_workers = num_workers
        # P2: decode in a spawn process pool to break the GIL on zlib decompression
        # of compressed .npz (threads plateau; processes parallelize). Lazily
        # created (after CUDA init). Falls back to threads on BrokenProcessPool.
        self.use_process_pool = bool(use_process_pool)
        self._proc_pool = None

        # Global index over the full dataset (Step 2). Populated by set_index();
        # working sets are loaded from it by load_working_set().
        self.index: list[MotionEntry] = []
        self.body_indices: Tensor | None = None

        # Working-set <-> global id maps (Step 3). working_to_global[local] = global.
        self.working_to_global: Tensor | None = None
        self._global_to_working: dict[int, int] = {}

        self._reset_state()

    def _get_proc_pool(self):
        """Lazily create the persistent spawn process pool (after CUDA init)."""
        if self._proc_pool is None:
            ctx = mp.get_context("spawn")
            self._proc_pool = ProcessPoolExecutor(
                max_workers=max(1, self.num_workers),
                mp_context=ctx,
                initializer=_proc_pool_init,
                initargs=(self.dtype == torch.float16,),
            )
        return self._proc_pool

    def shutdown(self) -> None:
        """Tear down the process pool (call at training end / on pool error)."""
        if self._proc_pool is not None:
            try:
                self._proc_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass
            self._proc_pool = None

    def __del__(self):
        # Best-effort cleanup; explicit shutdown() is preferred.
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass

    @property
    def num_unique_motions(self) -> int:
        """Total clips in the global index (the full dataset)."""
        return len(self.index)

    def set_index(self, index: list[MotionEntry], body_indices: Tensor | Sequence[int]) -> None:
        """Install the global index and the body selection used at load time."""
        self.index = list(index)
        self.body_indices = torch.as_tensor(body_indices, dtype=torch.long)

    def _reset_state(self) -> None:
        """Clear all store state. Called on construct and before every (re)load so
        an empty working set never leaves stale data addressable by the getters."""
        self.num_motions: int = 0
        self.num_joints: int = 0
        self.num_bodies: int = 0
        self.max_motion_length: int = 0

        # [N] length / offset tables live on the compute device for cheap indexing.
        self.motion_lengths: Tensor | None = None
        self.length_starts: Tensor | None = None

        # Per-clip lightweight metadata + fps (kept; tensors are dropped post-stack).
        self.fps: list[float] = []
        self.metadata: list[dict] = []
        self.source_files: list[str] = []

        # Flat storage tensors.
        self._flat_joint_pos: Tensor | None = None
        self._flat_joint_vel: Tensor | None = None
        self._flat_body_pos_w: Tensor | None = None
        self._flat_body_quat_w: Tensor | None = None
        self._flat_body_lin_vel_w: Tensor | None = None
        self._flat_body_ang_vel_w: Tensor | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _read_many(self, file_paths: Sequence[str], body_indices: Tensor) -> list[LoadedMotion]:
        """Read a batch of motion files via a bounded thread pool.

        Decode ALWAYS targets host (CPU) tensors so threads are always safe and
        parallel (numpy decode + CPU tensor creation release the GIL; no CUDA
        context is touched in a worker — important for fork/thread safety). The
        CPU→``storage_device`` copy happens later in ``_build_prepared`` via the
        flat-tensor slice assignment. Threads, not processes (fork-after-CUDA is
        unsafe). Order follows ``file_paths``.
        """
        paths = list(file_paths)
        if not paths:
            return []
        nw = max(1, min(self.num_workers, len(paths)))
        cpu = torch.device("cpu")

        # P2: process-pool decode breaks the GIL on zlib (compressed .npz). Workers
        # return numpy; we build CPU tensors here. Fall back to threads on a broken
        # pool so a worker crash never kills training.
        if self.use_process_pool and len(paths) > 1:
            try:
                body_list = body_indices.detach().cpu().tolist()
                pool = self._get_proc_pool()
                recs = list(pool.map(read_motion_arrays,
                                     [(p, body_list) for p in paths], chunksize=16))
                # Build CPU tensors (decode contract); GPU copy happens in build.
                return [_loaded_from_arrays(r, self.dtype, cpu)
                        for r in recs if r is not None]
            except (BrokenExecutor, OSError, RuntimeError) as e:
                # RuntimeError covers "cannot schedule new futures after shutdown"
                # (a stale/dead pool handle). Drop the pool and retry on threads.
                print(f"[FlatMotionStore] process pool failed ({e}); "
                      f"falling back to threaded decode")
                self.shutdown()
                # fall through to threaded path

        def _read(p: str) -> LoadedMotion | None:
            # Always decode to CPU regardless of storage_device.
            return read_motion_npz(p, body_indices, self.dtype, cpu)

        if nw == 1:
            results = [_read(p) for p in paths]
        else:
            with ThreadPoolExecutor(max_workers=nw) as ex:
                results = list(ex.map(_read, paths))
        return [m for m in results if m is not None]

    def load_files(self, file_paths: Sequence[str], body_indices: Tensor | Sequence[int]) -> None:
        """Read every file in ``file_paths`` and build the flat store (non-streaming).

        Kept for tests / the simple "load everything" path. Streaming callers use
        ``set_index`` + ``load_working_set`` instead.
        """
        body_indices = torch.as_tensor(body_indices, dtype=torch.long)
        if self.body_indices is None:
            self.body_indices = body_indices
        motions = self._read_many(file_paths, body_indices)
        # _build_from_motions installs an ascending [0..N) working->global map via
        # commit_prepared (so getters/metadata stay consistent for this path).
        self._build_from_motions(motions)

    def prepare_working_set(self, global_ids: Sequence[int] | Tensor) -> PreparedWorkingSet:
        """Build a new working set WITHOUT touching the live store (atomic swap).

        Reads + body-selects + concatenates the selected clips into fresh flat
        tensors and returns them in a ``PreparedWorkingSet``. The current resident
        store is left fully intact, so if this raises (bad ids, all reads fail) the
        caller can keep training on the old set. Install it via ``commit_prepared``.

        ``global_ids`` should be unique; dedup defensively, order-preserving.
        """
        assert self.body_indices is not None, "set_index() must be called before prepare_working_set()"
        if isinstance(global_ids, Tensor):
            global_ids = global_ids.detach().cpu().tolist()
        # Validate BEFORE doing work: an out-of-range id signals a sampler/index bug.
        for g in global_ids:
            if not (0 <= int(g) < self.num_unique_motions):
                raise IndexError(f"global_id {g} out of range [0, {self.num_unique_motions})")
        seen: set[int] = set()
        unique_ids: list[int] = []
        for g in global_ids:
            g = int(g)
            if g not in seen:
                seen.add(g)
                unique_ids.append(g)

        paths = [self.index[g].path for g in unique_ids]
        motions = self._read_many(paths, self.body_indices)

        # Map surviving motions back to their global ids (some reads may fail).
        survived_by_path = {m.source_file: m for m in motions}
        kept_global: list[int] = []
        kept_motions: list[LoadedMotion] = []
        for g, p in zip(unique_ids, paths):
            m = survived_by_path.get(p)
            if m is not None:
                kept_global.append(g)
                kept_motions.append(m)

        # Distinguish "asked for nothing" (valid empty swap) from "asked for
        # clips but all failed to load" (a real error — keep the old set).
        if not kept_motions:
            if len(unique_ids) == 0:
                return self._empty_prepared()
            raise RuntimeError(
                f"prepare_working_set: all {len(unique_ids)} requested clips failed to load"
            )

        prepared = self._build_prepared(kept_motions, kept_global)
        return prepared

    def _empty_prepared(self) -> PreparedWorkingSet:
        """A PreparedWorkingSet representing an empty working set (0 motions)."""
        dev, dt = self.storage_device, self.dtype
        empty2 = torch.empty(0, 1, dtype=dt, device=dev)
        return PreparedWorkingSet(
            flat_joint_pos=empty2, flat_joint_vel=empty2,
            flat_body_pos_w=torch.empty(0, 1, 3, dtype=dt, device=dev),
            flat_body_quat_w=torch.empty(0, 1, 4, dtype=dt, device=dev),
            flat_body_lin_vel_w=torch.empty(0, 1, 3, dtype=dt, device=dev),
            flat_body_ang_vel_w=torch.empty(0, 1, 3, dtype=dt, device=dev),
            motion_lengths=torch.empty(0, dtype=torch.long, device=self.device),
            length_starts=torch.empty(0, dtype=torch.long, device=self.device),
            num_motions=0, num_joints=0, num_bodies=0, max_motion_length=0,
            fps=[], metadata=[], source_files=[],
            working_to_global=torch.empty(0, dtype=torch.long, device=self.device),
            global_to_working={},
        )

    def commit_prepared(self, prepared: PreparedWorkingSet) -> None:
        """Install a PreparedWorkingSet, freeing the old resident buffers first.

        This is the only step that mutates the live store. Old flat tensors are
        dropped + CUDA cache emptied immediately before the new ones become live,
        so peak (old per-clip already freed by prepare) stays bounded.
        """
        self._free_flat_tensors()
        self._reset_state()
        if prepared.num_motions == 0:
            # Empty working set: leave the store fully reset (flat tensors None) so
            # getters see "no data loaded", matching the empty-reload contract.
            print("[FlatMotionStore] Committed empty working set (0 motions)")
            return
        self.num_motions = prepared.num_motions
        self.num_joints = prepared.num_joints
        self.num_bodies = prepared.num_bodies
        self.max_motion_length = prepared.max_motion_length
        # prepared tensors are CPU-built; move flat data to storage_device and the
        # small index tables to the compute device here on the main thread (this is
        # the CPU->GPU copy for storage_device='cuda').
        sd = self.storage_device

        def _to_sd(t: Tensor) -> Tensor:
            return t if t.device == sd else t.to(sd)

        self._flat_joint_pos = _to_sd(prepared.flat_joint_pos)
        self._flat_joint_vel = _to_sd(prepared.flat_joint_vel)
        self._flat_body_pos_w = _to_sd(prepared.flat_body_pos_w)
        self._flat_body_quat_w = _to_sd(prepared.flat_body_quat_w)
        self._flat_body_lin_vel_w = _to_sd(prepared.flat_body_lin_vel_w)
        self._flat_body_ang_vel_w = _to_sd(prepared.flat_body_ang_vel_w)
        self.motion_lengths = prepared.motion_lengths.to(self.device)
        self.length_starts = prepared.length_starts.to(self.device)
        self.fps = prepared.fps
        self.metadata = prepared.metadata
        self.source_files = prepared.source_files
        self.working_to_global = prepared.working_to_global.to(self.device)
        self._global_to_working = prepared.global_to_working

    def load_working_set(self, global_ids: Sequence[int] | Tensor) -> None:
        """Convenience: prepare + commit in one call (non-overlapped path).

        Atomic at the parse level — if prepare raises, the live store is untouched.
        The overlapped/background path calls prepare_working_set and commit_prepared
        separately so the read can run during the PPO update.
        """
        prepared = self.prepare_working_set(global_ids)
        self.commit_prepared(prepared)

    def _free_flat_tensors(self) -> None:
        """Drop flat tensors and reclaim CUDA memory before a reload (swap)."""
        self._flat_joint_pos = None
        self._flat_joint_vel = None
        self._flat_body_pos_w = None
        self._flat_body_quat_w = None
        self._flat_body_lin_vel_w = None
        self._flat_body_ang_vel_w = None
        import gc

        gc.collect()
        if self.storage_device.type == "cuda" or self.device.type == "cuda":
            torch.cuda.empty_cache()

    def global_to_working(self, global_ids: Tensor) -> Tensor:
        """Map global ids -> current working-set local ids (-1 if not resident)."""
        out = torch.full_like(
            torch.as_tensor(global_ids, dtype=torch.long), -1
        )
        flat = out.view(-1)
        gids = torch.as_tensor(global_ids, dtype=torch.long).view(-1)
        for i, g in enumerate(gids.tolist()):
            flat[i] = self._global_to_working.get(int(g), -1)
        return out

    def _build_prepared(
        self, motions: list[LoadedMotion], kept_global: list[int]
    ) -> PreparedWorkingSet:
        """Pure builder: concatenate clips into flat tensors WITHOUT touching the
        live store. Frees each clip's per-frame tensors as it copies them in.
        Returns a PreparedWorkingSet ready to commit."""
        num_motions = len(motions)
        num_joints = int(motions[0].joint_pos.shape[1])
        num_bodies = int(motions[0].body_pos_w.shape[1])
        lengths = [m.num_frames for m in motions]
        total = int(sum(lengths))
        max_motion_length = int(max(lengths))

        # Build flat tensors on CPU (NOT storage_device): this keeps the whole
        # prepare step CUDA-free so it can run in a background thread during the
        # PPO update (P3). commit_prepared moves them to storage_device on the main
        # thread. Pre-allocate + copy-in + free each clip to avoid the torch.cat 2x
        # peak. Length tables stay CPU here; commit moves them to self.device.
        dt = self.dtype
        cpu = torch.device("cpu")
        f_jp = torch.empty(total, num_joints, dtype=dt, device=cpu)
        f_jv = torch.empty(total, num_joints, dtype=dt, device=cpu)
        f_bp = torch.empty(total, num_bodies, 3, dtype=dt, device=cpu)
        f_bq = torch.empty(total, num_bodies, 4, dtype=dt, device=cpu)
        f_blv = torch.empty(total, num_bodies, 3, dtype=dt, device=cpu)
        f_bav = torch.empty(total, num_bodies, 3, dtype=dt, device=cpu)

        offset = 0
        for m in motions:
            t = m.num_frames
            sl = slice(offset, offset + t)
            f_jp[sl] = m.joint_pos
            f_jv[sl] = m.joint_vel
            f_bp[sl] = m.body_pos_w
            f_bq[sl] = m.body_quat_w
            f_blv[sl] = m.body_lin_vel_w
            f_bav[sl] = m.body_ang_vel_w
            offset += t
            m.joint_pos = m.joint_vel = None  # type: ignore[assignment]
            m.body_pos_w = m.body_quat_w = None  # type: ignore[assignment]
            m.body_lin_vel_w = m.body_ang_vel_w = None  # type: ignore[assignment]

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=cpu)
        shifted = lengths_t.roll(1)
        shifted[0] = 0
        return PreparedWorkingSet(
            flat_joint_pos=f_jp, flat_joint_vel=f_jv,
            flat_body_pos_w=f_bp, flat_body_quat_w=f_bq,
            flat_body_lin_vel_w=f_blv, flat_body_ang_vel_w=f_bav,
            motion_lengths=lengths_t, length_starts=shifted.cumsum(0),
            num_motions=num_motions, num_joints=num_joints, num_bodies=num_bodies,
            max_motion_length=max_motion_length,
            fps=[m.fps for m in motions],
            metadata=[m.metadata for m in motions],
            source_files=[m.source_file for m in motions],
            # built on CPU; commit moves to self.device for device-consistent
            # translation of device-side start_motion_ids.
            working_to_global=torch.tensor(kept_global, dtype=torch.long, device=cpu),
            global_to_working={g: i for i, g in enumerate(kept_global)},
        )

    def _build_from_motions(self, motions: list[LoadedMotion]) -> None:
        """Build + install directly (used by the non-streaming load_files path)."""
        self._reset_state()
        self.num_motions = len(motions)
        if self.num_motions == 0:
            print("[FlatMotionStore] Warning: no motions loaded")
            return
        prepared = self._build_prepared(motions, list(range(len(motions))))
        self.commit_prepared(prepared)

    # ------------------------------------------------------------------ #
    # Indexing helpers
    # ------------------------------------------------------------------ #

    def _clamp_frames(self, motion_ids: Tensor, frame_ids: Tensor) -> Tensor:
        """Clamp ``frame_ids`` to each motion's own valid range ``[0, T_m - 1]``.

        Critical for ragged storage: a global ``max_motion_length`` clamp would
        let a short clip's index spill into the next clip's rows.
        """
        assert self.motion_lengths is not None, "Motion data not loaded"
        lengths = self.motion_lengths.to(frame_ids.device)[motion_ids]
        frame_ids = torch.clamp(frame_ids, min=0)
        return torch.minimum(frame_ids, lengths - 1)

    def _global_index(self, motion_ids: Tensor, frame_ids: Tensor) -> Tensor:
        """Map ``(motion_id, frame_id)`` to flat row indices with per-motion clamp."""
        assert self.length_starts is not None, "Motion data not loaded"
        frame_ids = self._clamp_frames(motion_ids, frame_ids)
        starts = self.length_starts.to(frame_ids.device)[motion_ids]
        return starts + frame_ids

    def _to_storage(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        # CUDA->CPU copies of index tensors (motion_ids/frame_ids) MUST be blocking: a
        # non_blocking device-to-host copy can return a CPU tensor still holding stale
        # allocator memory, which then indexes the flat store out of bounds (intermittent,
        # masked by CUDA_LAUNCH_BLOCKING=1). Only the CUDA->CPU direction is unsafe here.
        out = []
        for t in tensors:
            if t.device == self.storage_device:
                out.append(t)
                continue
            non_blocking = not (t.device.type == "cuda" and self.storage_device.type == "cpu")
            out.append(t.to(self.storage_device, non_blocking=non_blocking))
        return tuple(out)

    def _to_device(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        if self.storage_device == self.device:
            return tensors
        return tuple(t.to(self.device, non_blocking=True) for t in tensors)

    # ------------------------------------------------------------------ #
    # Public getters (API-compatible with the dense MotionDataStore)
    # ------------------------------------------------------------------ #

    def get_motion_data(
        self, motion_ids: Tensor, frame_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Gather a single frame per env. Returns the 6 core fields."""
        assert self._flat_joint_pos is not None, "Motion data not loaded"
        motion_ids, frame_ids = self._to_storage(motion_ids, frame_ids)
        gidx = self._global_index(motion_ids, frame_ids)
        out = (
            self._flat_joint_pos[gidx],
            self._flat_joint_vel[gidx],
            self._flat_body_pos_w[gidx],
            self._flat_body_quat_w[gidx],
            self._flat_body_lin_vel_w[gidx],
            self._flat_body_ang_vel_w[gidx],
        )
        return self._to_device(*out)

    def _window_indices(
        self, motion_ids: Tensor, start_frames: Tensor, window_size: int, stride: int
    ) -> tuple[Tensor, Tensor]:
        """Compute [B, W] flat row indices for a forward window from ``start_frames``.

        A centered window of half-width ``L`` (centered window) is just a forward window of
        size ``2L+1`` starting at ``center - L*stride`` — same clamping applies.
        """
        offsets = torch.arange(window_size, device=start_frames.device) * stride
        frame_indices = start_frames.unsqueeze(1) + offsets.unsqueeze(0)  # [B, W]
        b_motion = motion_ids.unsqueeze(1).expand(-1, window_size)  # [B, W]
        gidx = self._global_index(b_motion.reshape(-1), frame_indices.reshape(-1))
        return gidx.view(motion_ids.shape[0], window_size), frame_indices

    def get_motion_window(
        self, motion_ids: Tensor, start_frames: Tensor, window_size: int, stride: int = 1
    ) -> tuple[Tensor, Tensor]:
        """Windowed (joint_pos, joint_vel), each [B, W, num_joints]."""
        assert self._flat_joint_pos is not None, "Motion data not loaded"
        motion_ids, start_frames = self._to_storage(motion_ids, start_frames)
        gidx, _ = self._window_indices(motion_ids, start_frames, window_size, stride)
        jp = self._flat_joint_pos[gidx]
        jv = self._flat_joint_vel[gidx]
        return self._to_device(jp, jv)

    def get_motion_window_full(
        self, motion_ids: Tensor, start_frames: Tensor, window_size: int, stride: int = 1
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Windowed version of all 6 core fields, each [B, W, ...]."""
        assert self._flat_joint_pos is not None, "Motion data not loaded"
        motion_ids, start_frames = self._to_storage(motion_ids, start_frames)
        gidx, _ = self._window_indices(motion_ids, start_frames, window_size, stride)
        out = (
            self._flat_joint_pos[gidx],
            self._flat_joint_vel[gidx],
            self._flat_body_pos_w[gidx],
            self._flat_body_quat_w[gidx],
            self._flat_body_lin_vel_w[gidx],
            self._flat_body_ang_vel_w[gidx],
        )
        return self._to_device(*out)

    def gather_centered_window(
        self, motion_ids: Tensor, center_frames: Tensor, half_window: int, stride: int = 1
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Centered window of all 6 fields, each [B, 2*half_window+1, ...].

        Used by the command's windowing methods so they never touch ``_flat_*``
        directly. Equivalent to a forward window starting ``half_window`` back.
        """
        window_size = 2 * half_window + 1
        start = center_frames - half_window * stride
        return self.get_motion_window_full(motion_ids, start, window_size, stride)


# ====================================================================== #
# Global curriculum (Step 4) — picks which clips form the working set
# ====================================================================== #


class GlobalCurriculum:
    """Per-motion curriculum over the FULL dataset, persisting across swaps.

    This is the "global level" of the two-level sampler. It tracks per-motion
    failure/success statistics over ``num_unique_motions`` (the whole dataset,
    NOT the resident working set) and produces a probability distribution used to
    choose which unique clips load into the next working set.

    Working sets load whole clips, so global selection is per-motion (no bin/frame
    structure needed here — that lives in the resident local sampler). Selection
    uses ``multinomial(replacement=False)`` so the resident set is unique.

    State is keyed by GLOBAL motion id and never reset on swap, so curriculum
    progress accumulates across the whole run. Tiny: O(num_unique_motions).
    """

    def __init__(
        self,
        num_unique_motions: int,
        device: torch.device | str = "cpu",
        *,
        beta: float = 1.0,
        alpha: float = 0.01,
        uniform_ratio: float = 0.1,
        # ---- Phase 3 anti-forgetting (default 0 => base streaming unchanged) ----
        age_ratio: float = 0.0,        # budget for an age-weighted reload term
        age_tau: float = 10.0,         # folds-since-seen scale for the age weight
    ) -> None:
        self.num_unique_motions = int(num_unique_motions)
        self.device = torch.device(device)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.uniform_ratio = float(uniform_ratio)
        self.age_ratio = float(age_ratio)
        self.age_tau = float(max(1e-6, age_tau))
        # Validate EACH component (not just the sum): a negative component could pass a
        # sum check while still producing an invalid (negative-mass) mixture.
        if not (0.0 <= self.uniform_ratio <= 1.0):
            raise ValueError(f"uniform_ratio must be in [0,1], got {self.uniform_ratio}")
        if not (0.0 <= self.age_ratio <= 1.0):
            raise ValueError(f"age_ratio must be in [0,1], got {self.age_ratio}")
        if self.uniform_ratio + self.age_ratio > 1.0:
            raise ValueError(
                f"uniform_ratio + age_ratio must be <= 1, got "
                f"{self.uniform_ratio} + {self.age_ratio}"
            )

        n = self.num_unique_motions
        # Laplace-smoothed counts (start at 1) so every clip has nonzero mass.
        self.failed = torch.ones(n, device=self.device, dtype=torch.float32)
        self.success = torch.ones(n, device=self.device, dtype=torch.float32)
        # Accumulators between EMA folds.
        self._cur_failed = torch.zeros(n, device=self.device, dtype=torch.float32)
        self._cur_success = torch.zeros(n, device=self.device, dtype=torch.float32)
        # Phase 3: folds since a clip last received ANY outcome (proxy for "rounds
        # since last resident/sampled"). Incremented for untouched clips each fold;
        # reset to 0 for touched clips. Drives the age-weighted reload term so easy
        # clips that stopped being sampled get periodically pulled back into a working
        # set (anti-starvation), not forgotten.
        self._folds_since_seen = torch.zeros(n, device=self.device, dtype=torch.float32)

    def probabilities(self) -> Tensor:
        """Per-motion sampling probability over the full dataset (sums to 1).

        Computed in float64 and guarded against underflow: with ``uniform_ratio=0``
        and a high ``beta``, ``p_fail**beta`` can underflow to all-zero, which would
        make ``multinomial`` raise on a NaN/zero distribution. We fall back to a
        uniform distribution if the weighted mass collapses.

        Phase 3: when ``age_ratio>0``, reserve that budget for an age-weighted term so
        long-unsampled clips get reloaded (anti-forgetting). ``age_ratio=0`` (default)
        reproduces the original failure+uniform mix exactly.
        """
        total = self.failed + self.success
        p_fail = torch.clamp((self.failed / (total + 1e-8)).double(), min=1e-12)
        weighted = torch.pow(p_fail, self.beta)
        wsum = weighted.sum()
        n = max(1, self.num_unique_motions)
        if not torch.isfinite(wsum) or wsum <= 0:
            weighted = torch.ones_like(weighted)  # underflow -> uniform fallback
            wsum = weighted.sum()
        weighted = weighted / wsum
        uniform = 1.0 / n
        if self.age_ratio <= 0.0:
            probs = weighted * (1.0 - self.uniform_ratio) + uniform * self.uniform_ratio
        else:
            # Age weight increases with folds-since-seen (saturating): older clips get
            # more reload mass. Normalized to a distribution before mixing.
            age_w = (1.0 - torch.exp(-self._folds_since_seen.double() / self.age_tau))
            asum = age_w.sum()
            age = (age_w / asum) if asum > 0 else torch.full_like(age_w, 1.0 / n)
            hard_w = 1.0 - self.uniform_ratio - self.age_ratio
            probs = weighted * hard_w + uniform * self.uniform_ratio + age * self.age_ratio
        probs = probs / probs.sum()
        return probs.to(self.failed.dtype)

    def sample_working_set(self, num_motions: int) -> Tensor:
        """Pick ``num_motions`` UNIQUE global ids for the next working set.

        ``replacement=False`` guarantees uniqueness (the resident set must not
        contain duplicate clips). If ``num_motions >= num_unique_motions`` the
        whole dataset is returned.
        """
        n = self.num_unique_motions
        if num_motions >= n:
            return torch.arange(n, device=self.device)
        probs = self.probabilities()
        return torch.multinomial(probs, num_motions, replacement=False)

    def update(self, global_motion_ids: Tensor, terminated: Tensor) -> None:
        """Record per-episode outcomes, keyed by GLOBAL motion id.

        ``terminated`` True = failure. Callers translate resident-local start ids
        to global ids (via ``FlatMotionStore.working_to_global``) before calling.
        """
        if global_motion_ids.numel() == 0:
            return
        gids = global_motion_ids.to(self.device, dtype=torch.long)
        term = terminated.to(self.device, dtype=torch.bool)
        if term.any():
            self._cur_failed += torch.bincount(gids[term], minlength=self.num_unique_motions).float()
        if (~term).any():
            self._cur_success += torch.bincount(gids[~term], minlength=self.num_unique_motions).float()

    def sync_accumulators(self, all_reduce_sum) -> None:
        """Pool the RAW per-window outcome accumulators across distributed ranks.

        Must be called at a swap boundary BEFORE ``fold()`` so every rank folds the
        SAME globally-pooled counts (matching single-GPU semantics: "one big
        rollout"). ``all_reduce_sum(tensor)`` is injected by the runner (an
        in-place SUM all-reduce, e.g. ``torch.distributed.all_reduce(t, SUM)``) so
        this module stays free of a hard torch.distributed dependency.

        NOTE: SUM with no divide is intentional. Each rank contributes only its own
        envs' outcomes; summing reconstructs the full multi-rank outcome set. The
        persistent ``failed``/``success`` (which carry the Laplace prior) are NOT
        reduced — only the fresh ``_cur_*`` deltas are — so the prior is not scaled.
        """
        all_reduce_sum(self._cur_failed)
        all_reduce_sum(self._cur_success)

    def fold(self) -> None:
        """EMA-fold accumulated outcomes into the persistent stats and reset.

        Call once per swap (before sampling the next working set) so the
        distribution reflects recent performance without unbounded growth.
        In distributed runs, call ``sync_accumulators`` first so all ranks fold
        identical pooled counts.

        Only motions that received an outcome this round are updated; a motion
        absent from the working set keeps its prior stats UNCHANGED (its history
        must not silently decay just because it wasn't sampled).
        """
        a = self.alpha
        touched = (self._cur_failed + self._cur_success) > 0
        self.failed = torch.where(touched, a * self._cur_failed + (1.0 - a) * self.failed, self.failed)
        self.success = torch.where(touched, a * self._cur_success + (1.0 - a) * self.success, self.success)
        # Phase 3: bump age for clips with NO outcome this round; reset touched to 0.
        # Maintained unconditionally (cheap, O(n)); only consumed when age_ratio>0.
        self._folds_since_seen = torch.where(
            touched,
            torch.zeros_like(self._folds_since_seen),
            self._folds_since_seen + 1.0,
        )
        self._cur_failed.zero_()
        self._cur_success.zero_()

    def state_dict(self) -> dict:
        return {
            "failed": self.failed.detach().cpu(),
            "success": self.success.detach().cpu(),
            "cur_failed": self._cur_failed.detach().cpu(),
            "cur_success": self._cur_success.detach().cpu(),
            "folds_since_seen": self._folds_since_seen.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        # Reject curriculum state whose size doesn't match this dataset.
        for key in ("failed", "success"):
            if key in state and state[key].numel() != self.num_unique_motions:
                raise ValueError(
                    f"GlobalCurriculum '{key}' size {state[key].numel()} != "
                    f"num_unique_motions {self.num_unique_motions}"
                )
        if "failed" in state:
            self.failed = state["failed"].to(self.device).float()
        if "success" in state:
            self.success = state["success"].to(self.device).float()
        if "cur_failed" in state:
            self._cur_failed = state["cur_failed"].to(self.device).float()
        if "cur_success" in state:
            self._cur_success = state["cur_success"].to(self.device).float()
        # Backward-compatible: pre-Phase-3 checkpoints have no folds_since_seen.
        if "folds_since_seen" in state and \
                state["folds_since_seen"].numel() == self.num_unique_motions:
            self._folds_since_seen = state["folds_since_seen"].to(self.device).float()
