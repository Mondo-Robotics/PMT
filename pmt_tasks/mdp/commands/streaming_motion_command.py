"""Streaming multi-motion command — memory-bounded variant of MultiMotionCommandV2.

Composes the pure-torch streaming data layer (``streaming_motion_lib``) into an
Isaac Lab ``CommandTerm`` so training can use a dataset far larger than GPU
memory. Only a working set of ~``num_envs`` clips is resident at any time; the
set is swapped periodically by the runner via :meth:`resample_working_set`.

Two-level sampling (per the plan):
  * GLOBAL: ``GlobalCurriculum`` over the full dataset chooses which unique clips
    form the next working set. Persists across swaps.
  * LOCAL: the inherited per-resident-clip sampler (uniform / adaptive /
    bin_adaptive) picks which resident clip + frame each env resets to.

The class reuses ALL of MultiMotionCommandV2's reset / pose / metric / windowing
machinery; it only swaps the data store and adds the streaming hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

from isaaclab.utils import configclass

from .multi_motion_command import (
    MultiMotionCommandV2,
    MultiMotionCommandV2Cfg,
)
from .streaming_motion_lib import FlatMotionStore, GlobalCurriculum, build_motion_index

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _FlatStoreAdapter(FlatMotionStore):
    """FlatMotionStore exposing the few attributes the base command reads.

    The base ``MultiMotionCommandV2`` reads ``data_store.num_joints`` and
    ``data_store.motion_lengths`` (compute-device tensor) directly; FlatMotionStore
    already provides both with matching semantics, so no shim is needed beyond
    naming. Kept as a thin subclass for clarity / future hooks.
    """

    pass


class StreamingMultiMotionCommand(MultiMotionCommandV2):
    """MultiMotionCommandV2 backed by a swappable, memory-bounded working set."""

    cfg: "StreamingMultiMotionCommandV2Cfg"

    def __init__(self, cfg: "StreamingMultiMotionCommandV2Cfg", env: "ManagerBasedRLEnv"):
        # Resolve the resident working-set size before base __init__ loads data.
        self._cap = int(cfg.max_working_set) if cfg.max_working_set else env.num_envs
        self._working_set_size = min(env.num_envs, self._cap)
        super().__init__(cfg, env)

        # start_global_motion_ids mirrors start_motion_ids but in GLOBAL id space,
        # captured at reset time so outcomes route correctly even if a swap occurs
        # before the next update (Codex Step-4 contract).
        self._start_global_motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Data store: index the full dataset, load only a working set
    # ------------------------------------------------------------------ #

    def _init_data_store(self) -> None:
        """Build the streaming store: index everything, load the first working set."""
        self.data_store = _FlatStoreAdapter(
            device=self.device,
            storage_device=self.cfg.storage_device,
            use_fp16=self.cfg.use_fp16,
            num_workers=self.cfg.num_load_workers,
            use_process_pool=getattr(self.cfg, "use_process_pool", False),
        )
        index = build_motion_index(self.cfg.motion_files, num_workers=self.cfg.num_load_workers)
        if len(index) == 0:
            raise ValueError("[Streaming] No valid motions indexed from cfg.motion_files")
        self.data_store.set_index(index, self.body_indices)

        # Global curriculum over the FULL dataset.
        self.global_curriculum = GlobalCurriculum(
            num_unique_motions=self.data_store.num_unique_motions,
            device=self.device,
            beta=self.cfg.global_beta,
            alpha=self.cfg.global_alpha,
            uniform_ratio=self.cfg.global_uniform_ratio,
        )

        # Load the initial working set.
        initial_ids = self.global_curriculum.sample_working_set(self._working_set_size)
        self.data_store.load_working_set(initial_ids)
        print(
            f"[Streaming] Indexed {self.data_store.num_unique_motions} motions; "
            f"resident working set = {self.data_store.num_motions} "
            f"(cap {self._working_set_size})"
        )

    # ------------------------------------------------------------------ #
    # Episode outcomes -> global curriculum (translated local -> global)
    # ------------------------------------------------------------------ #

    def _on_episode_outcomes(self, env_ids: Tensor, terminated: Tensor) -> None:
        """Route resetting envs' outcomes to the global curriculum by GLOBAL id.

        Uses the GLOBAL start id captured when the env last started its episode
        (``_start_global_motion_ids``), NOT a fresh translation of the local id.
        This is swap-safe: if the working set changed since the env started, the
        current ``working_to_global`` no longer maps that env's local id to the
        clip it was actually tracking.
        """
        global_ids = self._start_global_motion_ids[env_ids]
        self.global_curriculum.update(global_ids, terminated)

    def _local_to_global(self, local_ids: Tensor) -> Tensor:
        """Translate resident-local motion ids to global ids via working_to_global."""
        w2g = self.data_store.working_to_global
        assert w2g is not None, "working_to_global not set"
        return w2g.to(local_ids.device)[local_ids]

    def get_sonic_human_window(self, env_ids: Tensor) -> Tensor:
        """Robot-only streaming: human motion is not loaded by FlatMotionStore.

        The base implementation indexes dense ``_stacked_human_*`` fields that do
        not exist on the streaming store. Return zeros so a SONIC-human obs term
        does not crash. If a real human stream is ever needed for streaming, this
        must be replaced with a ragged human store rather than silently zeroed.
        """
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # 10-frame window × 22 human joints × 3 — matches the base zero-shape.
        return torch.zeros(len(env_ids), 10 * 22 * 3, device=self.device)

    # Override resample to also capture the GLOBAL start id for each env. The base
    # _resample_command sets self.start_motion_ids[env_ids] (local); we mirror it
    # to global space right after so outcomes survive a working-set swap.
    def _resample_command(self, env_ids: Sequence[int] | Tensor) -> None:
        super()._resample_command(env_ids)
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self._start_global_motion_ids[env_ids] = self._local_to_global(
            self.start_motion_ids[env_ids]
        )

    # ------------------------------------------------------------------ #
    # Streaming control (called by the runner at rollout boundaries)
    # ------------------------------------------------------------------ #

    def sync_curriculum_accumulators(self, all_reduce_sum) -> None:
        """Distributed: pool raw per-window outcome counts across ranks BEFORE fold
        so every rank folds identical globally-pooled stats. ``all_reduce_sum`` is
        an in-place SUM all-reduce injected by the runner."""
        self.global_curriculum.sync_accumulators(all_reduce_sum)

    def fold_curriculum(self) -> None:
        """EMA-fold accumulated outcomes into the persistent global curriculum.

        Separated from sampling so a distributed runner can sync raw accumulators
        across ranks first, then fold identical pooled stats. Single-GPU callers
        can just use :meth:`resample_working_set`.
        """
        self.global_curriculum.fold()

    def prepare_working_set_ids(self, new_ids):
        """Build (but do not install) the next working set. Returns a prepared
        object, or None on failure. Safe to run in a background thread while the
        live store keeps serving the current set (P3 overlap)."""
        try:
            return self.data_store.prepare_working_set(new_ids)
        except Exception as e:  # noqa: BLE001
            print(f"[Streaming] prepare_working_set failed, keeping current set: {e}")
            return None

    def commit_working_set(self, prepared) -> bool:
        """Install a prepared working set and rebuild the local sampler.

        Returns True on commit, False if ``prepared`` is None (prepare failed —
        the old resident set stays live so training continues uninterrupted)."""
        if prepared is None:
            return False
        self.data_store.commit_prepared(prepared)
        # Local sampler is sized to the (now-changed) resident set; rebuild it.
        # Curriculum continuity lives in the GLOBAL curriculum, so a fresh local
        # sampler each swap is correct — its per-resident stats are transient.
        self.sampler = self._create_sampler()
        return True

    def load_working_set_ids(self, new_ids) -> bool:
        """Load a specific set of global ids as the resident working set.

        Used by the distributed runner after broadcasting rank-0's sampled ids so
        every rank holds the identical working set. Atomic: on load failure the
        old resident set stays live and we return False.
        """
        return self.commit_working_set(self.prepare_working_set_ids(new_ids))

    def sample_working_set_ids(self):
        """Sample a fresh unique working set from the global curriculum (no load)."""
        return self.global_curriculum.sample_working_set(self._working_set_size)

    def resample_working_set(self) -> None:
        """Fold + sample + load a fresh resident working set (single-process path).

        Call at a PPO rollout boundary only. The runner must then call
        :meth:`forced_reset_all` and reacquire obs. For distributed runs the
        runner instead does fold -> all-reduce -> rank0 sample -> broadcast ->
        :meth:`load_working_set_ids` to keep ranks consistent.
        """
        self.fold_curriculum()
        return self.load_working_set_ids(self.sample_working_set_ids())

    def forced_reset_all(self) -> None:
        """Fully reset every env onto the new working set after a swap.

        Routed through the env's own ``reset()`` so the WHOLE reset stack runs
        (scene, action/observation/reward/termination/event managers, observation
        history, ``episode_length_buf=0``) — not just command + robot state. The
        CommandManager.reset cascade calls this term's ``_resample_command(all)``,
        so we hold the ``_is_first_reset`` guard across the cascade: with the full
        env set that makes the base path SKIP the curriculum/sampler update (a
        scheduled swap is a truncation, not an episode outcome). The guard is
        cleared afterward so subsequent per-episode resets count normally.

        Note: recurrent-policy hidden state is the runner/policy's concern; the
        runner reacquires observations after this call.
        """
        self._is_first_reset = True
        try:
            # Full reset of all envs via the env's reset stack.
            self._env.reset()
        finally:
            self._is_first_reset = False


# ====================================================================== #
# Config
# ====================================================================== #


@configclass
class StreamingMultiMotionCommandV2Cfg(MultiMotionCommandV2Cfg):
    """Config for the streaming, memory-bounded multi-motion command."""

    class_type: type = StreamingMultiMotionCommand

    # Resident working-set cap. 0 -> use num_envs. The resident set holds
    # min(num_envs, max_working_set) UNIQUE clips.
    max_working_set: int = 0

    # Parallel file loading at swap time. Decode is always CPU-side.
    num_load_workers: int = 16

    # P2: use a spawn process pool to decode clips (breaks the GIL on zlib for
    # compressed .npz; ~8x faster in isolation). DEFAULT OFF: inside the Isaac Sim
    # training process the spawned workers re-import the entry script (which boots
    # Isaac Sim via AppLauncher), causing BrokenProcessPool + GPU-leaking orphans.
    # The threaded decode + P3 background-overlap give the practical win safely.
    # Only enable for offline/standalone tools whose __main__ is import-guarded and
    # does NOT launch Isaac Sim.
    use_process_pool: bool = False

    # Global curriculum (per-motion, over the full dataset, persists across swaps).
    global_beta: float = 1.0
    global_alpha: float = 0.01
    global_uniform_ratio: float = 0.1
