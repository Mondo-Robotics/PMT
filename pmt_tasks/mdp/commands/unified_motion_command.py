"""Unified single-store command for plane + terrain motions (one policy, one store).

Motivation (validated on real data): every clip — plane *or* terrain — stores
``body_pos_w`` in the **world frame**, already placed on the mesh. Terrain clips also
carry ``transform_d{x,y,yaw}``, but that transform is *already baked into* ``body_pos_w``
(verified: ``body_pos_w[0, base, :2] == [transform_dx, transform_dy]``), so it does NOT
need to be applied at load time. The robot root state can therefore be set directly from
``body_pos_w[base]`` for **any** clip.

With ``env_spacing = 0`` for all envs (terrain *must* be 0; plane is set to 0 for
simplicity), the only remaining plane/terrain difference is **reset noise**: terrain
clips must reset with zero positional/joint noise to stay aligned to the geometry, while
plane clips may use noise for robustness. That is a *per-clip* property, not a reason to
maintain two separate stores + two samplers + an env partition (as
``GroupedStreamingMultiMotionCommandV2`` does).

This command therefore:

* loads plane + terrain clips into **one** store (``PackedMotionStore`` when a pre-packed
  mmap dataset is given, else the base store),
* tags each clip ``is_terrain`` from the caller's file lists (per-clip flag),
* applies reset noise **per env, keyed on the clip that env is currently playing**
  (terrain → zero, plane → cfg ranges), by overriding the base command's two noise hooks,
* keeps a single sampler / curriculum over the whole mixed dataset.

The file is import-light: the pure-python noise/flag logic lives in free functions
(``terrain_flag_for_files``, ``per_env_pose_velocity_noise``,
``per_env_joint_position_noise``) that are unit-testable without isaaclab. The
``MultiMotionCommandV2`` subclass is only imported lazily so tests for the logic don't
need the simulator.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


# ====================================================================== #
# Pure logic (no isaaclab) — unit-testable
# ====================================================================== #


def strip_chunk_suffix(path: str) -> str:
    """Drop the ``[chunk_N]`` suffix that ``_chunk_motions`` appends to ``source_file``.

    The eager store chunks long clips into multiple MotionData entries tagged
    ``f"{source_file}[chunk_i]"``; stripping the suffix recovers the original clip path
    so it can be matched against ``terrain_motion_files``.
    """
    idx = path.rfind("[chunk_")
    return path[:idx] if idx != -1 else path


def terrain_flag_for_paths(
    clip_paths: Sequence[str], terrain_files: Sequence[str]
) -> np.ndarray:
    """Bool array ``[len(clip_paths)]``: True where the (chunk-suffixed) clip is terrain.

    Like ``terrain_flag_for_files`` but tolerates the ``[chunk_N]`` suffix on each path,
    so it can be applied to a store's *resident* clip list (post-chunk/post-skip) whose
    length is the index space of ``motion_ids``.
    """
    terrain_set = {os.path.abspath(p) for p in terrain_files}
    return np.array(
        [os.path.abspath(strip_chunk_suffix(p)) in terrain_set for p in clip_paths],
        dtype=bool,
    )


def terrain_flag_for_files(
    motion_files: Sequence[str], terrain_files: Sequence[str]
) -> np.ndarray:
    """Return a bool array ``[num_files]``: True where the clip is terrain-anchored.

    Membership is by exact path; ``terrain_files`` is the caller's explicit terrain
    list (the "per-clip flag from the file list" contract). Plane clips are simply
    those not in that set.
    """
    terrain_set = {os.path.abspath(p) for p in terrain_files}
    return np.array(
        [os.path.abspath(p) in terrain_set for p in motion_files], dtype=bool
    )


def per_env_pose_velocity_noise(
    is_terrain_env: Tensor,
    env_ids: Tensor,
    pose_range: dict,
    velocity_range: dict,
    device,
) -> tuple[Tensor, Tensor]:
    """Pose/velocity reset noise per env: zero for terrain rows, cfg ranges for plane rows.

    ``is_terrain_env`` is a ``[num_envs]`` bool tensor telling, for each env, whether the
    clip it is currently playing is terrain-anchored. Returns two ``[len(env_ids), 6]``
    tensors (pose, velocity) in ``env_ids`` order. Pure torch (no isaaclab) so it is
    unit-testable; the command passes ``sample_uniform`` results through this shape.
    """
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    n = env_ids.numel()
    pose_rand = torch.zeros(n, 6, device=device)
    vel_rand = torch.zeros(n, 6, device=device)

    plane_mask = ~is_terrain_env[env_ids].bool()
    if bool(plane_mask.any()):
        k = int(plane_mask.sum().item())
        pose_ranges = torch.tensor(
            [pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=device
        )
        vel_ranges = torch.tensor(
            [velocity_range.get(key, (0.0, 0.0)) for key in pose_keys], device=device
        )
        lo, hi = pose_ranges[:, 0], pose_ranges[:, 1]
        pose_rand[plane_mask] = lo + (hi - lo) * torch.rand(k, 6, device=device)
        lo, hi = vel_ranges[:, 0], vel_ranges[:, 1]
        vel_rand[plane_mask] = lo + (hi - lo) * torch.rand(k, 6, device=device)
    return pose_rand, vel_rand


def per_env_joint_position_noise(
    is_terrain_env: Tensor,
    joint_pos_shape: torch.Size,
    joint_position_range: tuple,
    device,
) -> Tensor:
    """Joint-position reset noise for ALL envs: zero for terrain rows, cfg range for plane."""
    noise = torch.zeros(joint_pos_shape, device=device)
    lo, hi = joint_position_range
    if lo != 0.0 or hi != 0.0:
        plane_mask = ~is_terrain_env.bool()
        if bool(plane_mask.any()):
            full = lo + (hi - lo) * torch.rand(joint_pos_shape, device=device)
            noise[plane_mask] = full[plane_mask]
    return noise


def per_env_origin(
    is_terrain_env: Tensor,
    env_ids: Tensor,
    flat_origin,
    device,
) -> Tensor:
    """Per-clip world origin for each env, keyed on its current clip's terrain flag.

    Terrain clips are world-placed on the mesh (their ``body_pos_w`` already carries the
    baked transform), so a terrain env's origin offset must be **zero**. Flat clips start
    at the world origin in the data (verified: ``body_pos_w[0, pelvis, :2] ~= (0, 0)``),
    so a flat env's origin must be ``flat_origin`` (e.g. ``(90, 0, 0)``) to shift the clip
    onto the dedicated flat patch baked into the combined mesh — keeping flat envs
    spatially separated from the terrain region (no collision).

    ``is_terrain_env`` is a ``[num_envs]`` bool tensor (clip-the-env-plays terrain flag).
    Returns a ``[len(env_ids), 3]`` tensor in ``env_ids`` order: zeros for terrain rows,
    ``flat_origin`` for plane rows. Pure torch (no isaaclab) so it is unit-testable; the
    command index-assigns it into ``scene.terrain.env_origins`` at reset.
    """
    flat_o = torch.as_tensor(flat_origin, device=device, dtype=torch.float32)
    n = env_ids.numel()
    origins = torch.zeros(n, 3, device=device, dtype=torch.float32)
    plane_mask = ~is_terrain_env[env_ids].bool()
    if bool(plane_mask.any()):
        origins[plane_mask] = flat_o
    return origins


# ====================================================================== #
# Command (imports isaaclab lazily via the base class)
# ====================================================================== #


try:  # isaaclab present (cluster/training) -> define real module-level classes.
    from isaaclab.utils import configclass as _configclass

    _HAS_ISAACLAB_UNIFIED = True
except Exception:  # pragma: no cover - test/standalone env without isaaclab
    _HAS_ISAACLAB_UNIFIED = False


# Define the command + cfg at TRUE MODULE LEVEL (inside a module-level ``if`` block, NOT
# inside a function) when isaaclab is available AND we are imported as a package. This is
# REQUIRED so the classes have a clean importable ``__qualname__`` — the runner pickles the
# env-cfg (``params/env.pkl``), and a class built inside a function gets a
# ``..._build_unified_command_class.<locals>.UnifiedMultiMotionCommandCfg`` qualname that
# ``pickle.dump`` cannot serialize (Codex-flagged; confirmed as a runtime FAILED). In the
# standalone unit-test context (top-level module, ``__package__`` empty) the block is
# skipped and only the pure helper functions above are used.
if _HAS_ISAACLAB_UNIFIED and __package__:
    configclass = _configclass

    from .multi_motion_command import MultiMotionCommandV2, MultiMotionCommandV2Cfg
    from .packed_motion_lib import PackedMotionStore
    from .streaming_motion_lib import GlobalCurriculum, build_motion_index

    class UnifiedMultiMotionCommand(MultiMotionCommandV2):
        """One store for plane + terrain; per-clip reset-noise; root from body_pos_w."""

        cfg: "UnifiedMultiMotionCommandCfg"

        def __init__(self, cfg, env):
            # Resolve the per-clip terrain flag BEFORE base __init__ loads data, since the
            # base loader fixes self.cfg.motion_files order = clip order.
            self._is_terrain_clip_np = terrain_flag_for_files(
                cfg.motion_files, cfg.terrain_motion_files
            )
            super().__init__(cfg, env)
            self._is_terrain_clip = torch.as_tensor(
                self._is_terrain_clip_np, device=self.device, dtype=torch.bool
            )
            # Per-env flag: which group is the clip this env currently plays. Maintained
            # alongside motion_ids; refreshed whenever a command is (re)sampled.
            self._is_terrain_env = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._refresh_is_terrain_env(torch.arange(self.num_envs, device=self.device))

        # -- data store: optionally packed/mmap, else the base store -------- #

        def _init_data_store(self) -> None:
            pack_dir = getattr(self.cfg, "pack_dir", None)
            if pack_dir:
                store = PackedMotionStore(
                    device=self.device,
                    storage_device=self.cfg.storage_device,
                    use_fp16=self.cfg.use_fp16,
                    num_workers=getattr(self.cfg, "num_load_workers", 16),
                )
                store.set_packed_index(pack_dir, self.body_indices)
                # Map cfg.motion_files (terrain-flag order) onto pack global ids so the
                # is_terrain flag stays aligned to whatever subset the pack resolved.
                self._align_terrain_flag_to_index(store)
                self.global_curriculum = GlobalCurriculum(
                    num_unique_motions=store.num_unique_motions,
                    device=self.device,
                    beta=getattr(self.cfg, "global_beta", 1.0),
                    alpha=getattr(self.cfg, "global_alpha", 0.01),
                    uniform_ratio=getattr(self.cfg, "global_uniform_ratio", 0.1),
                )
                ws = getattr(self.cfg, "max_working_set", 0) or self.num_envs
                ws = min(ws, store.num_unique_motions)
                store.load_working_set(
                    self.global_curriculum.sample_working_set(ws)
                )
                self.data_store = store
                print(
                    f"[Unified] packed store: {store.num_unique_motions} clips, "
                    f"resident {store.num_motions}, "
                    f"terrain={int(self._is_terrain_clip_np.sum())} "
                    f"plane={int((~self._is_terrain_clip_np).sum())}"
                )
            else:
                super()._init_data_store()
                # The eager store may DROP unreadable clips and CHUNK long clips
                # (chunk_length>0 turns one source clip into N MotionData entries), so
                # store.num_motions != len(cfg.motion_files) and motion_ids index the
                # chunked store, not cfg.motion_files. Re-derive the per-clip flag from
                # each resident MotionData's source_file so its length == store clip
                # count (mirrors _align_terrain_flag_to_index for the packed store).
                self._align_terrain_flag_to_eager_store(self.data_store)

        def _align_terrain_flag_to_index(self, store) -> None:
            """Re-order the per-clip terrain flag to match the store's global id order.

            The packer may skip unreadable clips, so pack order can differ from
            cfg.motion_files order. Re-derive the flag from each resident entry's path.
            """
            terrain_set = {os.path.abspath(p) for p in self.cfg.terrain_motion_files}
            flag = np.array(
                [os.path.abspath(e.path) in terrain_set for e in store.index], dtype=bool
            )
            self._is_terrain_clip_np = flag

        def _align_terrain_flag_to_eager_store(self, store) -> None:
            """Rebuild the per-clip terrain flag for the (non-packed) eager store.

            ``store.motions`` is the post-chunk, post-skip clip list whose order matches
            ``motion_ids``. Each entry's ``source_file`` carries the original path (plus a
            ``[chunk_i]`` suffix for chunked clips), so re-derive ``is_terrain`` per
            resident clip. Length is then guaranteed == store.num_motions == the index
            space of motion_ids.
            """
            self._is_terrain_clip_np = terrain_flag_for_paths(
                [m.source_file for m in store.motions], self.cfg.terrain_motion_files
            )

        # -- per-env terrain flag tracking ---------------------------------- #

        def _refresh_is_terrain_env(self, env_ids: Tensor) -> None:
            """Set is_terrain_env[env_ids] from the clip each env currently plays.

            Uses ``working_to_global`` when the store streams (motion_ids are resident-
            local) so the flag is read in global-clip space; falls back to direct
            indexing for the non-streaming store.
            """
            local_ids = self.motion_ids[env_ids]
            global_ids = self._local_to_global_safe(local_ids)
            # Index the flag on its own device with a long index, and assign on the
            # flag's device then move to the per-env buffer's device. Guards against a
            # device/dtype mismatch triggering an async CUDA index assert.
            global_ids = global_ids.to(self._is_terrain_clip.device, dtype=torch.long)
            self._is_terrain_env[env_ids] = self._is_terrain_clip[global_ids].to(
                self._is_terrain_env.device
            )

        def _local_to_global_safe(self, local_ids: Tensor) -> Tensor:
            local_ids = local_ids.to(dtype=torch.long)
            w2g = getattr(self.data_store, "working_to_global", None)
            if w2g is None:
                return local_ids
            return w2g.to(local_ids.device)[local_ids]

        def _reset_robot_state(self, env_ids) -> None:
            # CRITICAL ORDERING: the base _resample_command sets self.motion_ids[env_ids]
            # to the newly-sampled clips and THEN calls this method, which applies reset
            # noise via _pose_velocity_noise / _joint_position_noise (both read
            # _is_terrain_env) AND reads self.body_pos_w (= _body_pos_w_buf +
            # scene.env_origins). So we must, in this exact order, BEFORE the base runs:
            #   1. refresh the per-env terrain flag from the just-assigned motion_ids,
            #      otherwise a terrain clip could be reset with the *previous* clip's
            #      (possibly plane) noise / origin;
            #   2. inject the per-clip env origin (terrain -> 0, flat -> flat_origin) into
            #      scene.terrain.env_origins, so the root pos the base computes from
            #      body_pos_w lands at the flat patch for flat clips (no collision on the
            #      terrain mesh) and at the baked mesh location for terrain clips.
            # Refreshing/injecting after super() returns is too late (root already written).
            ids = env_ids
            if not isinstance(ids, torch.Tensor):
                ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
            if ids.numel() > 0:
                self._refresh_is_terrain_env(ids)
                if self.cfg.inject_env_origins:
                    self._inject_env_origins(ids)
            super()._reset_robot_state(env_ids)

        # -- per-clip env-origin injection (terrain -> 0, flat -> flat_origin) -- #

        def _inject_env_origins(self, env_ids: Tensor) -> None:
            """Index-assign per-clip world origins into scene.terrain.env_origins.

            ``scene.env_origins`` is a live read-only property delegating to
            ``terrain.env_origins``; there is no setter, so we index-assign into the
            underlying tensor (same mechanism as grouped_motion_common.apply_group_env_
            origins, but keyed per-clip rather than per fixed env partition).
            """
            origins = self._env.scene.terrain.env_origins  # live [num_envs, 3] tensor
            new = per_env_origin(
                self._is_terrain_env, env_ids, self.cfg.flat_origin, origins.device
            )
            origins[env_ids] = new.to(origins.dtype)

        # -- noise hooks: per env, keyed on the clip's terrain flag --------- #

        def _pose_velocity_noise(self, env_ids: Tensor) -> tuple[Tensor, Tensor]:
            return per_env_pose_velocity_noise(
                self._is_terrain_env,
                env_ids,
                self.cfg.pose_range,
                self.cfg.velocity_range,
                self.device,
            )

        def _joint_position_noise(self, joint_pos_shape: torch.Size) -> Tensor:
            return per_env_joint_position_noise(
                self._is_terrain_env,
                joint_pos_shape,
                self.cfg.joint_position_range,
                self.device,
            )

    @configclass
    class UnifiedMultiMotionCommandCfg(MultiMotionCommandV2Cfg):
        """Config: one store, plane+terrain mixed, per-clip noise.

        ``motion_files`` is the full mixed list (plane + terrain). ``terrain_motion_files``
        is the subset to treat as terrain-anchored (zero reset noise). ``pack_dir``, when
        set, points at a pre-packed mmap dataset built by ``prepack_motion_dataset``.
        ``pose_range`` / ``velocity_range`` / ``joint_position_range`` are the PLANE noise
        ranges (terrain rows always get zero).
        """

        class_type: type = UnifiedMultiMotionCommand

        terrain_motion_files: list = []
        pack_dir: str | None = None
        max_working_set: int = 0
        num_load_workers: int = 16
        global_beta: float = 1.0
        global_alpha: float = 0.01
        global_uniform_ratio: float = 0.1

        # Per-clip env-origin injection. When True, each env's world origin is set at
        # reset from the clip it plays: terrain clip -> 0 (world-placed on mesh), flat
        # clip -> ``flat_origin`` (shifted onto the flat patch baked in the combined
        # mesh). This subsumes GroupedMultiMotionCommandV2's hard env partition WITHOUT
        # pinning an env to a group spatially. Requires env_spacing == 0 so the importer
        # adds no origin of its own. Default False -> behaves like the noise-only Unified
        # command (single-mesh / plane-only use).
        inject_env_origins: bool = False
        flat_origin: list = [90.0, 0.0, 0.0]

    # Classes are now real module-level objects (UnifiedMultiMotionCommand /
    # UnifiedMultiMotionCommandCfg) with importable qualnames — picklable by the runner.
