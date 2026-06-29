"""Adaptive-sampling motion command (Isaac Lab glue).

Subclasses ``StreamingMultiMotionCommand`` so it inherits the entire indexed,
memory-bounded, two-level streaming stack (global curriculum picks the resident
working set; a local sampler picks clip+frame). The ONLY thing this class changes
is the LOCAL sampler: when ``sampler_type == "hybrid"`` it installs the pure-torch
``HybridBinSampler`` from ``adaptive_sampling_lib`` instead of the stock
``BinBasedAdaptiveSampler``.

Phase 0 (this commit): ``HybridBinSampler`` is configured for parity with
``bin_adaptive`` (all hybrid hooks off), so a hybrid run reproduces the existing
streaming + bin-adaptive behavior. Later phases flip on:
  * P1 composite tracking-error + backward-from-termination attribution,
  * P2 offline frequency/jerk/biomech prior,
  * P3 retention/age anti-forgetting budgets,
  * P4 policy-uncertainty + adversarial hard-buffer.

The heavy sampling math lives in the isaaclab-free sibling module so it can be
unit-tested without booting Isaac Sim (see tests/test_adaptive_sampling_lib.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

from isaaclab.utils import configclass

import os

from .adaptive_sampling_lib import HybridBinSampler
from .adaptive_sampling_prior import PRIOR_CACHE_VERSION, slice_prior_to_working_set
from .multi_motion_command import MotionSampler, SamplingResult, UniformSampler
from .streaming_motion_command import (
    StreamingMultiMotionCommand,
    StreamingMultiMotionCommandV2Cfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _HybridSamplerAdapter(MotionSampler):
    """Adapt the isaaclab-free ``HybridBinSampler`` to the ``MotionSampler`` API.

    The base command treats the local sampler as a ``MotionSampler`` (calls
    ``sample`` / ``update`` / ``step`` / ``get_metrics``). ``HybridBinSampler`` is
    intentionally NOT a ``MotionSampler`` subclass (that base pulls in isaaclab and
    would make the math untestable offline). This thin adapter bridges the two and
    re-wraps the lib's ``SamplingResult`` into the command's own so downstream code
    that ``isinstance``-checks the command result keeps working.
    """

    def __init__(self, core: HybridBinSampler):
        # Deliberately do NOT call super().__init__: we delegate all state to core.
        self.core = core
        self.num_motions = core.num_motions
        self.motion_lengths = core.motion_lengths
        self.device = core.device

    def sample(self, num_samples: int) -> SamplingResult:
        r = self.core.sample(num_samples)
        return SamplingResult(motion_ids=r.motion_ids, frame_ids=r.frame_ids)

    def update(self, motion_ids, frame_ids, terminated, **kwargs) -> None:
        self.core.update(motion_ids, frame_ids, terminated, **kwargs)

    def step(self) -> None:
        self.core.step()

    def get_metrics(self) -> dict[str, float]:
        return self.core.get_metrics()


class AdaptiveSamplingMotionCommand(StreamingMultiMotionCommand):
    """Streaming multi-motion command whose LOCAL sampler is the hybrid sampler."""

    cfg: "AdaptiveSamplingMotionCommandCfg"

    def __init__(self, cfg: "AdaptiveSamplingMotionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # Phase 1: per-env composite tracking-error accumulators (running sum + max
        # over the live episode). Folded into a per-episode scalar at reset time and
        # fed to the hybrid sampler so difficulty = failure AND poor tracking, not
        # failure alone. Only meaningful when error_weight > 0 (hybrid sampler).
        self._episode_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_error_max = torch.zeros(self.num_envs, device=self.device)
        self._episode_error_count = torch.zeros(self.num_envs, device=self.device)
        self._uses_error = float(self.cfg.hybrid_error_weight) > 0.0

        # Phase 4: per-env policy-uncertainty accumulators. The runner pushes per-env
        # action-std (normalized) each rollout step via receive_policy_uncertainty();
        # we accumulate the running mean per episode and feed it to the sampler at reset.
        # Only used when hybrid_uncertainty_weight > 0.
        self._uses_uncertainty = float(getattr(self.cfg, "hybrid_uncertainty_weight", 0.0)) > 0.0
        self._episode_unc_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_unc_count = torch.zeros(self.num_envs, device=self.device)
        # Latest per-env uncertainty pushed by the runner (mean action-std, normalized).
        self._latest_uncertainty = torch.zeros(self.num_envs, device=self.device)
        self._unc_norm = float(getattr(self.cfg, "hybrid_uncertainty_norm", 1.0)) or 1.0

    # ------------------------------------------------------------------ #
    # Phase 2: offline difficulty prior (loaded once, sliced per working set)
    # ------------------------------------------------------------------ #
    def _init_data_store(self) -> None:
        """Build the streaming store/index, THEN load the global offline prior.

        Runs before the base __init__ calls _create_sampler(), so the global prior is
        available when the first (and every subsequent) local sampler is built. The
        prior is aligned to the data-store's GLOBAL motion index by absolute path; any
        clip missing from the cache gets a zero (neutral) prior row.
        """
        super()._init_data_store()  # indexes dataset + builds global curriculum + first working set

        # Phase 3: enable the GLOBAL anti-forgetting age term on the (already-built)
        # global curriculum. _folds_since_seen is always allocated by its __init__, so
        # we only set the budgets here. age_ratio=0 (default) leaves base behavior
        # untouched. Validated against uniform_ratio to keep the mix in [0,1]. The
        # first working set was sampled before this with all ages 0 (age term == uniform),
        # which is harmless.
        age_ratio = float(getattr(self.cfg, "global_age_ratio", 0.0))
        age_tau = float(getattr(self.cfg, "global_age_tau", 10.0))
        if age_ratio != 0.0:
            gc = self.global_curriculum
            # Validate EACH component (not just the sum) so a negative ratio can't slip
            # through. age_ratio must be in [0,1] and uniform+age <= 1.
            if not (0.0 <= age_ratio <= 1.0):
                raise ValueError(f"global_age_ratio must be in [0,1], got {age_ratio}")
            if gc.uniform_ratio + age_ratio > 1.0:
                raise ValueError(
                    f"global_uniform_ratio + global_age_ratio must be <= 1, got "
                    f"{gc.uniform_ratio} + {age_ratio}"
                )
            gc.age_ratio = age_ratio
            gc.age_tau = max(1e-6, age_tau)

        self._global_prior = None   # [N_global, prior_max_bins] in [0,1], or None
        self._prior_max_bins = 0
        path = getattr(self.cfg, "offline_prior_path", "") or ""
        strength = float(getattr(self.cfg, "offline_prior_strength", 0.0))
        if not path or strength <= 0.0:
            return
        self._global_prior, self._prior_max_bins = self._load_global_prior(path)

    def _load_global_prior(self, path: str):
        """Load the precomputed prior cache and align rows to the global motion index.

        Returns (global_prior[N_global, max_bins] on self.device, max_bins) or (None, 0)
        on any problem (missing file, version mismatch, no path overlap) — failing SOFT
        so a missing/incompatible cache degrades to Phase-0/1 behavior instead of crashing.
        """
        # Normalize + existence-check inside try too: a non-string offline_prior_path
        # (config typed str, but be defensive) must soft-disable, not raise (Codex finding).
        try:
            path = os.path.realpath(os.path.expanduser(str(path)))
            if not os.path.isfile(path):
                print(f"[AdaptiveSampling] offline_prior_path not found: {path}; ignoring prior")
                return None, 0
            cache = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[AdaptiveSampling] failed to resolve/load prior cache {path!r}: {e!r}; ignoring prior")
            return None, 0
        # Validate the cache structurally so a malformed file degrades to Phase-0/1
        # instead of raising (Codex Phase-2 finding): require version, the expected
        # keys, a 2-D float prior, and a paths list whose length matches the rows.
        # The ENTIRE alignment is wrapped so ANY malformed field (missing key, wrong
        # type, non-pathlike entry in ``paths``, bad ``num_bins``) degrades to
        # Phase-0/1 instead of raising (Codex Phase-2 findings).
        try:
            if int(cache.get("version", -1)) != PRIOR_CACHE_VERSION:
                print(f"[AdaptiveSampling] prior cache version {cache.get('version')} != "
                      f"{PRIOR_CACHE_VERSION}; ignoring prior")
                return None, 0
            cache_prior = cache["prior"].float()       # [Nc, Bc]
            cache_paths = list(cache["paths"])
            cache_bin_size = int(cache["bin_size"])
            # num_bins is required for the stale-clip guard below.
            cache_num_bins = cache["num_bins"]
            cache_num_bins = [int(x) for x in cache_num_bins]
            if cache_prior.dim() != 2 or cache_prior.shape[0] != len(cache_paths) \
                    or len(cache_num_bins) != len(cache_paths):
                print(f"[AdaptiveSampling] malformed prior cache (prior {tuple(cache_prior.shape)}, "
                      f"{len(cache_paths)} paths, {len(cache_num_bins)} num_bins); ignoring prior")
                return None, 0

            # BIN-GRID GUARD (Codex Phase-2 HIGH): the cache's per-bin columns are only
            # meaningful if its bin_size matches the live sampler's grid (sampler bins
            # at round(cfg.motion_fps * cfg.bin_duration)). A cache built with a
            # different fps/bin_duration would inject column k into the WRONG bin.
            live_bin_size = max(1, int(round(float(self.cfg.motion_fps) * float(self.cfg.bin_duration))))
            if cache_bin_size != live_bin_size:
                print(f"[AdaptiveSampling] prior cache bin_size {cache_bin_size} != live "
                      f"bin_size {live_bin_size} (motion_fps={self.cfg.motion_fps}, "
                      f"bin_duration={self.cfg.bin_duration}); ignoring prior to avoid misalignment")
                return None, 0

            # Align to the store's global index by REALPATH (symlink-robust).
            path_to_row = {os.path.realpath(str(p)): i for i, p in enumerate(cache_paths)}
            max_bins = int(cache_prior.shape[1])
            index = self.data_store.index
            n_global = len(index)
            lengths = self.data_store.motion_lengths  # [N_global] on compute device
            global_prior = torch.zeros(n_global, max_bins, dtype=torch.float32)
            hits = 0
            stale = 0
            for g, entry in enumerate(index):
                row = path_to_row.get(os.path.realpath(str(entry.path)))
                if row is None:
                    continue
                # STALE-CLIP GUARD (Codex Phase-2 MEDIUM): a same-path, same-bin_size
                # cache from an EDITED clip (different length) would map stale priors.
                # Require the cached per-clip bin count to match the live clip's
                # expected bin count; skip (leave zero/neutral) any row that disagrees.
                expected_bins = max(1, (int(entry.num_frames) + cache_bin_size - 1) // cache_bin_size)
                if cache_num_bins[row] != expected_bins:
                    stale += 1
                    continue
                global_prior[g] = cache_prior[row]
                hits += 1
        except Exception as e:  # noqa: BLE001
            print(f"[AdaptiveSampling] malformed prior cache {path}: {e!r}; ignoring prior")
            return None, 0

        if hits == 0:
            print(f"[AdaptiveSampling] prior cache has NO usable path overlap with the "
                  f"motion index ({n_global} clips, {stale} stale); ignoring prior")
            return None, 0
        if stale:
            print(f"[AdaptiveSampling] prior: skipped {stale} stale clip(s) (num_bins mismatch)")
        # Guard the final device transfer too (e.g. CUDA OOM / unavailable device) so the
        # ENTIRE loader honors the soft-fail contract — degrade to Phase-0/1, never raise.
        try:
            global_prior = global_prior.to(self.device)
        except Exception as e:  # noqa: BLE001
            print(f"[AdaptiveSampling] failed to move prior to {self.device}: {e!r}; ignoring prior")
            return None, 0
        print(f"[AdaptiveSampling] loaded offline prior: {hits}/{n_global} clips matched, "
              f"max_bins={max_bins}, bin_size={cache_bin_size}, "
              f"strength={self.cfg.offline_prior_strength}")
        return global_prior, max_bins

    def _resident_prior(self):
        """Slice the global prior to the CURRENT resident working set, or None.

        Returns a ``[num_resident, local_max_bins]`` tensor matching the local sampler's
        bin grid, where local_max_bins is derived from the resident motion lengths the
        same way HybridBinSampler computes max_bins.
        """
        if getattr(self, "_global_prior", None) is None:
            return None
        w2g = getattr(self.data_store, "working_to_global", None)
        if w2g is None:
            return None
        lengths = self.data_store.motion_lengths
        if lengths is None or lengths.numel() == 0:
            return None
        bin_size = max(1, int(round(float(self.cfg.motion_fps) * float(self.cfg.bin_duration))))
        local_max_bins = int(((lengths + (bin_size - 1)) // bin_size).clamp(min=1).max().item())
        return slice_prior_to_working_set(self._global_prior, w2g, local_max_bins)

    # ------------------------------------------------------------------ #
    # Phase 1: composite tracking error + backward-from-termination blame
    # ------------------------------------------------------------------ #
    def _composite_error(self) -> Tensor:
        """Per-env composite tracking error (weighted sum of the command metrics).

        Uses the SAME error tensors the base command already computes in
        ``_update_metrics`` (error_anchor_pos/rot, error_body_pos/rot, error_joint_pos),
        so this adds no new forward kinematics. Weights follow
        adaptive_sampling_discussion.md §3 (Phase 1). Returns a [num_envs] tensor.
        """
        m = self.metrics
        return (
            0.15 * m["error_anchor_pos"]
            + 0.10 * m["error_anchor_rot"]
            + 0.35 * m["error_body_pos"]
            + 0.20 * m["error_body_rot"]
            + 0.20 * m["error_joint_pos"]
        )

    def _update_command(self) -> None:
        # CRITICAL ORDERING: accumulate the just-simulated step's error BEFORE
        # super()._update_command(). IsaacLab's CommandTerm.compute() calls
        # _update_metrics() (refreshing self.metrics for the CURRENT, pre-increment
        # frame) immediately before _update_command(). The base _update_command then
        # advances frame_ids and may resample ended (motion-end) clips, which calls our
        # _record_outcomes and then clears that env's accumulator. Accumulating FIRST
        # guarantees the just-tracked frame is in the accumulator when the motion-end
        # path consumes it, and that the post-clear accumulator stays clean.
        #
        # NOTE on the terminal frame (Codex reviewer-B finding): we deliberately do NOT
        # try to fold an extra "terminal" error inside _record_outcomes. On the external
        # env-termination path, IsaacLab runs termination -> _reset_idx ->
        # command_manager.reset(), and reset() ZEROES self.metrics[env_ids] (command_
        # manager.py:142) BEFORE calling _resample -> _record_outcomes. So self.metrics
        # is already zero there and is NOT a valid terminal sample. The accumulator is
        # the single source of truth: it holds every per-step error the sim actually
        # computed for the episode. The divergence frame that triggered termination is
        # never simulated/metric-evaluated before reset, so it is simply not part of the
        # episode's tracked error — which is the correct, unbiased behavior.
        if self._uses_error:
            err = self._composite_error().detach()
            self._episode_error_sum += err
            self._episode_error_max = torch.maximum(self._episode_error_max, err)
            self._episode_error_count += 1.0
        if self._uses_uncertainty:
            # Accumulate the most recent per-env uncertainty pushed by the runner. Same
            # ordering rationale as error: fold BEFORE super() so the motion-end path
            # sees a complete accumulator. The runner pushes BEFORE env.step (so this
            # step's value reflects the action just taken); if it never pushes (no hook),
            # _latest_uncertainty stays 0 and contributes nothing.
            self._episode_unc_sum += self._latest_uncertainty
            self._episode_unc_count += 1.0
        super()._update_command()

    def receive_policy_uncertainty(self, uncertainty: Tensor) -> None:
        """Runner hook (Phase 4): push per-env policy uncertainty for the live step.

        ``uncertainty`` is a [num_envs] (or [num_envs, act_dim]) tensor — typically the
        actor action-std. We reduce to per-env mean, normalize by ``hybrid_uncertainty_norm``
        into ~[0,1], and stash it; _update_command folds it into the episode accumulator.
        No-op (cheap) when uncertainty is disabled. Fully shape/device/NaN tolerant: any
        malformed push is ignored rather than corrupting the accumulator or raising.
        """
        if not self._uses_uncertainty:
            return
        if not isinstance(uncertainty, torch.Tensor):
            return
        u = uncertainty.detach()
        # Reduce everything except the leading (env) dim to a per-env scalar.
        if u.dim() == 0:
            return  # scalar: no per-env structure -> ignore
        if u.dim() > 1:
            u = u.reshape(u.shape[0], -1).mean(dim=-1)
        if u.shape[0] != self.num_envs:
            return  # wrong env count -> ignore malformed push
        u = u.to(self.device, dtype=torch.float32)
        # Sanitize BEFORE storing so a NaN/Inf never poisons the episode accumulator
        # (the sampler also sanitizes, but only at fold time — by then later valid
        # samples in the episode would already be lost).
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        self._latest_uncertainty = (u / self._unc_norm).clamp(0.0, 1.0)

    def _episode_error_scalar(self, env_ids: Tensor) -> Tensor:
        """Blend mean+max episode error for env_ids (0.5/0.5 per the discussion)."""
        cnt = self._episode_error_count[env_ids].clamp(min=1.0)
        mean_err = self._episode_error_sum[env_ids] / cnt
        max_err = self._episode_error_max[env_ids]
        return 0.5 * mean_err + 0.5 * max_err

    def _episode_uncertainty_scalar(self, env_ids: Tensor) -> Tensor:
        """Per-episode mean uncertainty for env_ids (0 if no steps accumulated)."""
        cnt = self._episode_unc_count[env_ids].clamp(min=1.0)
        return self._episode_unc_sum[env_ids] / cnt

    def _reset_episode_error(self, env_ids: Tensor) -> None:
        self._episode_error_sum[env_ids] = 0.0
        self._episode_error_max[env_ids] = 0.0
        self._episode_error_count[env_ids] = 0.0

    def _reset_episode_uncertainty(self, env_ids: Tensor) -> None:
        self._episode_unc_sum[env_ids] = 0.0
        self._episode_unc_count[env_ids] = 0.0

    def _record_outcomes(self, env_ids: Tensor, terminated: Tensor) -> None:
        """Feed outcomes to the LOCAL hybrid sampler with Phase-1 signals.

        Adds two kwargs over the base (which only passes start ids + terminated):
          * end_frame_ids: the frame the env was on when it ended -> the hybrid
            sampler attributes a FAILURE to the TERMINATION bin (backward blame),
            not the start bin. (self.frame_ids holds the current/last frame.)
          * tracking_error: the per-episode composite error scalar.
        Falls back to the base 3-arg call when error is disabled or the sampler does
        not accept the kwargs (e.g. a non-hybrid fallback sampler).
        """
        # Only the hybrid adapter accepts the Phase-1 kwargs. When error is disabled,
        # OR the resident store was empty so _create_sampler fell back to a plain
        # UniformSampler (whose update() has no kwargs), use the base 3-arg path. This
        # explicit capability check replaces a broad ``except TypeError`` that could
        # otherwise mask a genuine internal TypeError raised inside sampler.update.
        # The hybrid adapter accepts the Phase-1/4 kwargs. Use it whenever error OR
        # uncertainty is enabled; otherwise (or for a UniformSampler fallback) take the
        # base 3-arg path.
        uses_signals = self._uses_error or self._uses_uncertainty
        if not uses_signals or not isinstance(self.sampler, _HybridSamplerAdapter):
            return super()._record_outcomes(env_ids, terminated)

        # The per-episode error/uncertainty come entirely from the step accumulators
        # (filled by _update_command). We do NOT read self.metrics here: on the
        # termination path CommandManager.reset() has already zeroed self.metrics[env_ids]
        # before this call, so it is not a valid terminal sample (see _update_command note).
        end_frame_ids = self.frame_ids[env_ids]
        episode_error = self._episode_error_scalar(env_ids) if self._uses_error else None
        episode_unc = self._episode_uncertainty_scalar(env_ids) if self._uses_uncertainty else None
        self.sampler.update(
            self.start_motion_ids[env_ids],
            self.start_frame_ids[env_ids],
            terminated,
            end_frame_ids=end_frame_ids,
            tracking_error=episode_error,
            uncertainty=episode_unc,
        )

    def _resample_command(self, env_ids: Sequence[int] | Tensor) -> None:
        # Base records outcomes (via _record_outcomes above) and re-seeds the env.
        # Reset the per-env accumulators AFTER so the just-ended episode's signals have
        # already been consumed by _record_outcomes and the new episode starts clean.
        super()._resample_command(env_ids)
        if not (self._uses_error or self._uses_uncertainty):
            return
        if not isinstance(env_ids, Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if self._uses_error:
            self._reset_episode_error(env_ids)
        if self._uses_uncertainty:
            self._reset_episode_uncertainty(env_ids)

    def _create_sampler(self) -> MotionSampler:
        """Install the hybrid local sampler; fall back to base for other types.

        Mirrors the base ``_create_sampler`` empty-store fallback so a working set
        that fails to load does not crash sampler construction.
        """
        if self.cfg.sampler_type != "hybrid":
            # Defer to the inherited factory (uniform / adaptive / bin_adaptive).
            return super()._create_sampler()

        if self.data_store.num_motions == 0 or self.data_store.motion_lengths is None:
            dummy_lengths = torch.tensor([1], device=self.device)
            return UniformSampler(1, dummy_lengths, self.device)

        core = HybridBinSampler(
            num_motions=self.data_store.num_motions,
            motion_lengths=self.data_store.motion_lengths,
            device=self.device,
            motion_fps=self.cfg.motion_fps,
            bin_duration=self.cfg.bin_duration,
            beta=self.cfg.adaptive_beta,
            alpha=self.cfg.adaptive_alpha,
            uniform_ratio=self.cfg.adaptive_uniform_ratio,
            update_interval=self.cfg.adaptive_update_interval,
            kernel_size=self.cfg.adaptive_kernel_size,
            kernel_lambda=self.cfg.adaptive_kernel_lambda,
            # Phase 1+ hooks (defaults keep Phase 0 parity with bin_adaptive).
            error_weight=float(self.cfg.hybrid_error_weight),
            failure_weight=float(self.cfg.hybrid_failure_weight),
            error_good=float(self.cfg.hybrid_error_good),
            error_bad=float(self.cfg.hybrid_error_bad),
            # Phase 2: offline difficulty prior sliced to the CURRENT resident set.
            # Rebuilt here so every working-set swap re-injects the matching rows.
            offline_bin_prior=self._resident_prior(),
            offline_prior_strength=float(getattr(self.cfg, "offline_prior_strength", 0.0)),
            # Phase 3: local anti-forgetting (retention budget + softened motion score).
            retention_ratio=float(getattr(self.cfg, "hybrid_retention_ratio", 0.0)),
            topk_motion=int(getattr(self.cfg, "hybrid_topk_motion", 1)),
            topk_motion_weight=float(getattr(self.cfg, "hybrid_topk_motion_weight", 0.3)),
            retention_success_thresh=float(getattr(self.cfg, "hybrid_retention_success_thresh", 0.85)),
            # Phase 4: success-gated policy uncertainty + adversarial hard-buffer.
            uncertainty_weight=float(getattr(self.cfg, "hybrid_uncertainty_weight", 0.0)),
            uncertainty_gate_lo=float(getattr(self.cfg, "hybrid_uncertainty_gate_lo", 0.2)),
            uncertainty_gate_hi=float(getattr(self.cfg, "hybrid_uncertainty_gate_hi", 0.8)),
            hard_buffer_ratio=float(getattr(self.cfg, "hybrid_hard_buffer_ratio", 0.0)),
            hard_buffer_k=int(getattr(self.cfg, "hybrid_hard_buffer_k", 64)),
        )
        return _HybridSamplerAdapter(core)


@configclass
class AdaptiveSamplingMotionCommandCfg(StreamingMultiMotionCommandV2Cfg):
    """Config for the adaptive-sampling (hybrid) streaming command.

    Inherits every streaming knob (max_working_set, num_load_workers,
    global_beta/alpha/uniform_ratio, motion_fps, bin_duration, ...). Adds the
    hybrid-sampler weights. ``sampler_type='hybrid'`` selects this code path; any
    other value falls back to the inherited samplers so this cfg is a safe drop-in.
    """

    class_type: type = AdaptiveSamplingMotionCommand

    sampler_type: str = "hybrid"

    # ---- Phase 1 composite-error weights (0 => Phase 0 parity / failure-only) ----
    hybrid_error_weight: float = 0.0
    hybrid_failure_weight: float = 1.0
    hybrid_error_good: float = 0.0
    hybrid_error_bad: float = 1.0

    # ---- Phase 2 offline difficulty prior ----
    # Absolute path to a cache produced by scripts/precompute_motion_prior.py. Empty or
    # strength<=0 disables the prior (=> Phase 0/1 behavior). The prior seeds the hybrid
    # sampler's failed_bin_count as `failed = 1 + strength * prior`, so high-frequency /
    # high-jerk / high-acceleration bins are sampled MORE before any online failures.
    offline_prior_path: str = ""
    offline_prior_strength: float = 0.0

    # ---- Phase 3 anti-forgetting (all default to no-op => Phase 0-2 parity) ----
    # LOCAL (resident sampler): reserve `retention_ratio` of the per-clip budget for
    # replaying already-learned clips (success-EMA >= thresh), and soften the motion
    # score to (1-w)*max_bin + w*mean(top-k) so one noisy hard bin can't monopolize.
    hybrid_retention_ratio: float = 0.0
    hybrid_topk_motion: int = 1
    hybrid_topk_motion_weight: float = 0.3
    hybrid_retention_success_thresh: float = 0.85
    # GLOBAL (working-set curriculum): reserve `global_age_ratio` for an age-weighted
    # term so clips not sampled for many folds get reloaded (anti-starvation).
    global_age_ratio: float = 0.0
    global_age_tau: float = 10.0

    # ---- Phase 4 uncertainty + adversarial hard-buffer (default no-op => parity) ----
    # Success-gated policy uncertainty: blends per-bin policy action-std into difficulty,
    # but ONLY where the bin's success rate is in [gate_lo, gate_hi] (neither hopeless nor
    # mastered). The runner pushes per-env action-std via receive_policy_uncertainty();
    # hybrid_uncertainty_norm scales raw std into ~[0,1]. weight=0 disables.
    hybrid_uncertainty_weight: float = 0.0
    hybrid_uncertainty_gate_lo: float = 0.2
    hybrid_uncertainty_gate_hi: float = 0.8
    hybrid_uncertainty_norm: float = 1.0
    # Adversarial hard-buffer: reserve `hard_buffer_ratio` of the per-clip budget for a
    # uniform draw over the current top-K hardest clips (guaranteed hard-example share).
    hybrid_hard_buffer_ratio: float = 0.0
    hybrid_hard_buffer_k: int = 64
