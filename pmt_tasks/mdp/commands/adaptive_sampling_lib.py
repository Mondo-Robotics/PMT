"""Pure-torch sampler core for the adaptive-sampling motion command.

This module is deliberately free of any ``isaaclab`` / ``omni`` imports so the
sampling math can be unit-tested with a plain Python + torch interpreter (no
Isaac Sim runtime). The Isaac Lab command glue lives in the sibling module
``adaptive_sampling_motion_command.py`` (which subclasses the streaming command
and injects the sampler defined here).

Design (per ``adaptive_sampling_discussion.md``):
  * Hierarchical bin sampling: motion -> bin -> frame (uniform within bin).
  * Phase 0 (this commit): behaves IDENTICALLY to ``BinBasedAdaptiveSampler`` —
    failure-rate EMA, ``p_fail ** beta``, uniform mix, one-sided (forward) kernel
    smoothing. All hybrid hooks below are present but DISABLED by default so the
    output distribution matches the plain bin sampler bit-for-bit in expectation.
  * Phases 1-4 add: composite tracking-error EMA + backward-from-termination
    attribution (P1), offline frequency/jerk/biomech prior (P2), retention/age
    anti-forgetting budgets (P3), policy-uncertainty + hard-buffer (P4). The
    constructor already accepts the hooks so later phases only flip them on.

Keeping the math here (not inheriting ``MotionSampler`` from
``multi_motion_command``) is intentional: that base class pulls in isaaclab and
would make the sampler untestable without booting Isaac Sim. The command-side
wrapper adapts this sampler to the ``MotionSampler`` duck-typed interface
(``sample`` / ``update`` / ``step`` / ``get_metrics``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SamplingResult:
    """Result of a sampling operation (field-compatible with the command's own).

    The Isaac Lab command reads ``result.motion_ids`` and ``result.frame_ids``
    only (duck-typed), so this lightweight stand-in works without importing the
    isaaclab-bound ``SamplingResult`` from ``multi_motion_command``.
    """

    motion_ids: Tensor
    frame_ids: Tensor


class HybridBinSampler:
    """Hybrid bin-level adaptive sampler (pure torch).

    Phase 0 contract: with all hybrid weights at their defaults
    (``error_weight=0``, ``offline_bin_prior=None``, retention/age/uncertainty
    off), the sampling distribution equals the existing ``BinBasedAdaptiveSampler``:
    motion score = max over bins of smoothed ``p_fail``; bin score = smoothed
    ``p_fail`` within the chosen motion; frame uniform within the chosen bin.
    """

    def __init__(
        self,
        num_motions: int,
        motion_lengths: Tensor,
        device: torch.device | str,
        *,
        motion_fps: float = 50.0,
        bin_duration: float = 1.0,
        beta: float = 0.5,
        alpha: float = 0.001,
        uniform_ratio: float = 0.1,
        update_interval: int = 240,
        kernel_size: int = 5,
        kernel_lambda: float = 0.8,
        # ---- Phase 1+ hooks (disabled by default => Phase 0 parity) ----
        error_weight: float = 0.0,          # P1: weight of tracking-error term
        failure_weight: float = 1.0,        # P1: weight of failure term
        error_good: float = 0.0,            # P1: error mapped to 0 difficulty
        error_bad: float = 1.0,             # P1: error mapped to 1 difficulty
        offline_bin_prior: Tensor | None = None,  # P2: [num_motions, max_bins] in [0,1]
        offline_prior_strength: float = 0.0,      # P2: pseudo-fail mass
        # ---- Phase 3 anti-forgetting hooks (disabled by default => P0-2 parity) ----
        retention_ratio: float = 0.0,        # P3: kappa, budget for learned-clip replay
        topk_motion: int = 1,                # P3: motion score = blend of top-k hard bins
        topk_motion_weight: float = 0.3,     # P3: weight of mean(top-k) vs max bin
        retention_success_thresh: float = 0.85,  # P3: success-EMA to count as "learned"
        # ---- Phase 4 hooks (disabled by default => P0-3 parity) ----
        uncertainty_weight: float = 0.0,     # P4: weight of success-gated policy uncertainty
        uncertainty_gate_lo: float = 0.2,    # P4: success-rate band [lo,hi] where U counts
        uncertainty_gate_hi: float = 0.8,
        hard_buffer_ratio: float = 0.0,      # P4: budget reserved for top-K hardest clips
        hard_buffer_k: int = 64,             # P4: size of the hard buffer (top-K clips)
    ) -> None:
        if motion_fps <= 0:
            raise ValueError(f"motion_fps must be > 0, got {motion_fps}")
        if bin_duration <= 0:
            raise ValueError(f"bin_duration must be > 0, got {bin_duration}")
        if update_interval < 1:
            raise ValueError(f"update_interval must be >= 1, got {update_interval}")
        if not (0.0 <= uniform_ratio <= 1.0):
            raise ValueError(f"uniform_ratio must be in [0, 1], got {uniform_ratio}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if not (0.0 < kernel_lambda <= 1.0):
            raise ValueError(f"kernel_lambda must be in (0, 1], got {kernel_lambda}")
        # Blend weights must be finite and >= 0 so _pf_bin stays in [0,1] (Codex finding:
        # a negative failure_weight produced out-of-range difficulty).
        if not (math.isfinite(float(failure_weight)) and float(failure_weight) >= 0.0):
            raise ValueError(f"failure_weight must be finite and >= 0, got {failure_weight}")
        if not (math.isfinite(float(error_weight)) and float(error_weight) >= 0.0):
            raise ValueError(f"error_weight must be finite and >= 0, got {error_weight}")
        if not (0.0 <= retention_ratio <= 1.0):
            raise ValueError(f"retention_ratio must be in [0, 1], got {retention_ratio}")
        if not (0.0 <= uniform_ratio + retention_ratio <= 1.0):
            raise ValueError(
                f"uniform_ratio + retention_ratio must be in [0, 1], got "
                f"{uniform_ratio} + {retention_ratio}"
            )
        if topk_motion < 1:
            raise ValueError(f"topk_motion must be >= 1, got {topk_motion}")
        if not (0.0 <= float(topk_motion_weight) <= 1.0):
            raise ValueError(f"topk_motion_weight must be in [0, 1], got {topk_motion_weight}")
        if not math.isfinite(float(topk_motion_weight)):
            raise ValueError(f"topk_motion_weight must be finite, got {topk_motion_weight}")
        if not (math.isfinite(float(uncertainty_weight)) and float(uncertainty_weight) >= 0.0):
            raise ValueError(f"uncertainty_weight must be finite and >= 0, got {uncertainty_weight}")
        if not (0.0 <= float(uncertainty_gate_lo) <= float(uncertainty_gate_hi) <= 1.0):
            raise ValueError(
                f"uncertainty gate must satisfy 0<=lo<=hi<=1, got "
                f"[{uncertainty_gate_lo}, {uncertainty_gate_hi}]"
            )
        if not (0.0 <= float(hard_buffer_ratio) <= 1.0):
            raise ValueError(f"hard_buffer_ratio must be in [0, 1], got {hard_buffer_ratio}")
        if float(uniform_ratio) + float(retention_ratio) + float(hard_buffer_ratio) > 1.0:
            raise ValueError(
                f"uniform_ratio + retention_ratio + hard_buffer_ratio must be <= 1, got "
                f"{uniform_ratio} + {retention_ratio} + {hard_buffer_ratio}"
            )
        if hard_buffer_k < 1:
            raise ValueError(f"hard_buffer_k must be >= 1, got {hard_buffer_k}")

        self.device = torch.device(device)
        self.num_motions = int(num_motions)
        self.motion_lengths = motion_lengths.to(self.device).long()

        self.beta = float(beta)
        self.alpha = float(alpha)
        self.uniform_ratio = float(uniform_ratio)
        self.update_interval = int(update_interval)

        # Phase 1 hooks.
        self.error_weight = float(error_weight)
        self.failure_weight = float(failure_weight)
        self.error_good = float(error_good)
        self.error_bad = float(error_bad)

        # Phase 3 anti-forgetting hooks.
        self.retention_ratio = float(retention_ratio)
        self.topk_motion = int(topk_motion)
        self.topk_motion_weight = float(topk_motion_weight)
        self.retention_success_thresh = float(retention_success_thresh)

        # Phase 4 hooks.
        self.uncertainty_weight = float(uncertainty_weight)
        self.uncertainty_gate_lo = float(uncertainty_gate_lo)
        self.uncertainty_gate_hi = float(uncertainty_gate_hi)
        self.hard_buffer_ratio = float(hard_buffer_ratio)
        self.hard_buffer_k = int(hard_buffer_k)

        bin_size = int(round(float(motion_fps) * float(bin_duration)))
        self.bin_size = max(1, bin_size)

        self.motion_num_bins = (
            (self.motion_lengths + (self.bin_size - 1)) // self.bin_size
        ).clamp(min=1).long()
        self.max_bins = int(self.motion_num_bins.max().item())

        dev = self.device
        f32 = torch.float32  # explicit (matches BinBasedAdaptiveSampler; immune to a
        #                      nonstandard torch default dtype that could break parity).
        # Motion-level counts (Laplace-smoothed).
        self.failed_motion_count = torch.ones(self.num_motions, device=dev, dtype=f32)
        self.success_motion_count = torch.ones(self.num_motions, device=dev, dtype=f32)
        self.current_failed_motion_count = torch.zeros(self.num_motions, device=dev, dtype=f32)
        self.current_success_motion_count = torch.zeros(self.num_motions, device=dev, dtype=f32)

        # Bin-level counts (Laplace-smoothed).
        self.failed_bin_count = torch.ones(self.num_motions, self.max_bins, device=dev, dtype=f32)
        self.success_bin_count = torch.ones(self.num_motions, self.max_bins, device=dev, dtype=f32)
        self.current_failed_bin_count = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)
        self.current_success_bin_count = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)

        # Phase 1: per-bin tracking-error EMA (in [0,1] after normalization).
        self.error_bin_ema = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)
        self.current_error_bin_sum = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)
        self.current_error_bin_count = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)

        # Phase 4: per-bin policy-uncertainty EMA (mean action-std proxy in [0,1] after
        # normalization by the caller) + accumulators, mirroring the error channel.
        self.uncertainty_bin_ema = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)
        self.current_unc_bin_sum = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)
        self.current_unc_bin_count = torch.zeros(self.num_motions, self.max_bins, device=dev, dtype=f32)

        # Valid bins mask.
        bin_indices = torch.arange(self.max_bins, device=dev).unsqueeze(0)
        self.valid_bins_mask = bin_indices < self.motion_num_bins.unsqueeze(1)
        self.total_valid_bins = int(self.valid_bins_mask.sum().item())

        # One-sided exponential smoothing kernel over bins (forward spread).
        if kernel_size > 1:
            kernel = torch.tensor(
                [kernel_lambda ** (kernel_size - 1 - i) for i in range(kernel_size)],
                device=dev,
                dtype=torch.float32,
            )
            self._kernel = (kernel / kernel.sum()).view(1, 1, -1)
            self._kernel_size = int(kernel_size)
        else:
            self._kernel = None
            self._kernel_size = 0

        # Phase 2: offline prior injected as pseudo-failures (online EMA overrides it).
        self._apply_offline_bin_prior(offline_bin_prior, float(offline_prior_strength))

        self._steps_since_update = 0

        self._last_entropy = 0.0
        self._last_pfail_mean = 0.0
        self._last_top1_prob = 0.0
        self._cached_motion_probs: Tensor | None = None
        self._update_metrics()

    # ------------------------------------------------------------------ #
    # Phase 2 prior injection
    # ------------------------------------------------------------------ #
    def _apply_offline_bin_prior(
        self, offline_bin_prior: Tensor | None, offline_prior_strength: float
    ) -> None:
        if offline_bin_prior is None or offline_prior_strength <= 0.0:
            return
        prior = offline_bin_prior.to(device=self.device, dtype=torch.float32)
        if prior.shape != self.failed_bin_count.shape:
            raise ValueError(
                f"offline_bin_prior must have shape {tuple(self.failed_bin_count.shape)}, "
                f"got {tuple(prior.shape)}"
            )
        prior = torch.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        prior = prior * self.valid_bins_mask.to(prior.dtype)
        self.failed_bin_count = self.failed_bin_count + offline_prior_strength * prior

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, num_samples: int) -> SamplingResult:
        if num_samples <= 0:
            empty = torch.empty((0,), device=self.device, dtype=torch.long)
            return SamplingResult(motion_ids=empty, frame_ids=empty)

        motion_probs = self._get_motion_probs()
        motion_ids = torch.multinomial(motion_probs, num_samples, replacement=True)

        valid_selected = self.valid_bins_mask[motion_ids]
        pf_selected = self._compute_score_selected_smoothed(motion_ids, valid_selected)

        pf_selected = pf_selected.clone()
        pf_selected[~valid_selected] = 0.0
        pf_selected = torch.clamp(pf_selected, min=1e-6)
        pf_selected[~valid_selected] = 0.0

        pf_weighted = torch.pow(pf_selected, self.beta)
        pf_weighted[~valid_selected] = 0.0
        pf_weighted = pf_weighted / (pf_weighted.sum(dim=1, keepdim=True) + 1e-8)

        valid_counts = self.motion_num_bins[motion_ids].clamp(min=1).unsqueeze(1)
        uniform_row = valid_selected.to(dtype=torch.float32) / valid_counts

        bin_probs = pf_weighted * (1.0 - self.uniform_ratio) + uniform_row * self.uniform_ratio
        bin_probs[~valid_selected] = 0.0
        bin_probs = bin_probs / (bin_probs.sum(dim=1, keepdim=True) + 1e-8)

        bin_ids = torch.multinomial(bin_probs, 1, replacement=True).squeeze(1)

        bin_start = bin_ids * self.bin_size
        lengths = self.motion_lengths[motion_ids]
        bin_len = torch.clamp(lengths - bin_start, min=1, max=self.bin_size)
        offset = (torch.rand(num_samples, device=self.device) * bin_len.to(dtype=torch.float32)).long()
        frame_ids = bin_start + offset
        frame_ids = torch.minimum(frame_ids, lengths - 1)

        return SamplingResult(motion_ids=motion_ids, frame_ids=frame_ids)

    # ------------------------------------------------------------------ #
    # Updates
    # ------------------------------------------------------------------ #
    def update(
        self,
        motion_ids: Tensor,
        frame_ids: Tensor,
        terminated: Tensor,
        *,
        end_frame_ids: Tensor | None = None,   # P1: termination frame for backward blame
        tracking_error: Tensor | None = None,  # P1: per-episode composite error in [0,1]
        uncertainty: Tensor | None = None,      # P4: per-episode policy uncertainty in [0,1]
    ) -> None:
        if motion_ids.numel() == 0:
            return

        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        frame_ids = frame_ids.to(device=self.device, dtype=torch.long)
        terminated = terminated.to(device=self.device, dtype=torch.bool)

        failed_mask = terminated
        success_mask = ~terminated

        # Motion-level counts (vectorized).
        if failed_mask.any():
            counts = torch.bincount(motion_ids[failed_mask], minlength=self.num_motions).float()
            self.current_failed_motion_count += counts
        if success_mask.any():
            counts = torch.bincount(motion_ids[success_mask], minlength=self.num_motions).float()
            self.current_success_motion_count += counts

        # Phase 0/1: choose which bin an outcome is attributed to. Default (P0) =
        # the START bin (frame_ids). When end_frame_ids is provided AND the episode
        # terminated, P1 attributes failure to the TERMINATION bin (closer to the
        # actual hard segment); successes still attribute to the start bin.
        att_frames = frame_ids.clone()
        if end_frame_ids is not None:
            end_frame_ids = end_frame_ids.to(device=self.device, dtype=torch.long)
            att_frames = torch.where(failed_mask, end_frame_ids, frame_ids)

        bin_ids = (att_frames // self.bin_size).long()
        max_valid_bin = (self.motion_num_bins[motion_ids] - 1).clamp(min=0)
        bin_ids = torch.minimum(bin_ids, max_valid_bin).clamp(min=0, max=self.max_bins - 1)

        flat_index = motion_ids * self.max_bins + bin_ids
        flat_failed = self.current_failed_bin_count.view(-1)
        flat_success = self.current_success_bin_count.view(-1)

        if failed_mask.any():
            idx = flat_index[failed_mask]
            flat_failed.scatter_add_(0, idx, torch.ones_like(idx, dtype=flat_failed.dtype))
        if success_mask.any():
            idx = flat_index[success_mask]
            flat_success.scatter_add_(0, idx, torch.ones_like(idx, dtype=flat_success.dtype))

        # Phase 1: accumulate per-bin tracking error (folded into EMA on step()).
        if tracking_error is not None and self.error_weight > 0.0:
            # Store the RAW per-episode composite error (e.g. meters + radians). It is
            # mapped to [0,1] difficulty later in ``_pf_bin`` via (error - error_good) /
            # (error_bad - error_good); clamping to [0,1] here would destroy that scale.
            err = tracking_error.to(device=self.device, dtype=torch.float32)
            err = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
            self.current_error_bin_sum.view(-1).scatter_add_(0, flat_index, err)
            self.current_error_bin_count.view(-1).scatter_add_(
                0, flat_index, torch.ones_like(err)
            )

        # Phase 4: accumulate per-bin policy uncertainty (attributed to the SAME bin as
        # the outcome). Expected pre-normalized to ~[0,1] by the caller; sanitized here.
        if uncertainty is not None and self.uncertainty_weight > 0.0:
            unc = uncertainty.to(device=self.device, dtype=torch.float32)
            unc = torch.nan_to_num(unc, nan=0.0, posinf=0.0, neginf=0.0)
            self.current_unc_bin_sum.view(-1).scatter_add_(0, flat_index, unc)
            self.current_unc_bin_count.view(-1).scatter_add_(
                0, flat_index, torch.ones_like(unc)
            )

    def step(self) -> None:
        self._steps_since_update += 1
        if self._steps_since_update >= self.update_interval:
            self._ema_update()
            self._reset_counts()
            self._steps_since_update = 0

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def get_metrics(self) -> dict[str, float]:
        return {
            "adaptive_entropy": float(self._last_entropy),
            "adaptive_pfail_mean": float(self._last_pfail_mean),
            "adaptive_top1_prob": float(self._last_top1_prob),
            "total_failed": float(self.failed_motion_count.sum().item()),
            "total_success": float(self.success_motion_count.sum().item()),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ema_update(self) -> None:
        a = self.alpha
        self.failed_motion_count = a * self.current_failed_motion_count + (1.0 - a) * self.failed_motion_count
        self.success_motion_count = a * self.current_success_motion_count + (1.0 - a) * self.success_motion_count
        self.failed_bin_count = a * self.current_failed_bin_count + (1.0 - a) * self.failed_bin_count
        self.success_bin_count = a * self.current_success_bin_count + (1.0 - a) * self.success_bin_count

        if self.error_weight > 0.0:
            touched = self.current_error_bin_count > 0
            mean_err = torch.where(
                touched,
                self.current_error_bin_sum / self.current_error_bin_count.clamp(min=1.0),
                self.error_bin_ema,
            )
            self.error_bin_ema = torch.where(
                touched, a * mean_err + (1.0 - a) * self.error_bin_ema, self.error_bin_ema
            )

        if self.uncertainty_weight > 0.0:
            touched_u = self.current_unc_bin_count > 0
            mean_unc = torch.where(
                touched_u,
                self.current_unc_bin_sum / self.current_unc_bin_count.clamp(min=1.0),
                self.uncertainty_bin_ema,
            )
            self.uncertainty_bin_ema = torch.where(
                touched_u, a * mean_unc + (1.0 - a) * self.uncertainty_bin_ema, self.uncertainty_bin_ema
            )

        self._update_metrics()

    def _reset_counts(self) -> None:
        self.current_failed_motion_count.zero_()
        self.current_success_motion_count.zero_()
        self.current_failed_bin_count.zero_()
        self.current_success_bin_count.zero_()
        self.current_error_bin_sum.zero_()
        self.current_error_bin_count.zero_()
        self.current_unc_bin_sum.zero_()
        self.current_unc_bin_count.zero_()

    def _pf_bin(self) -> Tensor:
        """Per-bin difficulty BEFORE smoothing: failure rate, optionally blended with
        normalized tracking error (Phase 1) and success-gated policy uncertainty
        (Phase 4). With error_weight=0 AND uncertainty_weight=0 this is exactly
        ``failed/(failed+success)`` (Phase 0 parity)."""
        total = self.failed_bin_count + self.success_bin_count
        pf = self.failed_bin_count / (total + 1e-8)
        if self.error_weight <= 0.0 and self.uncertainty_weight <= 0.0:
            return pf
        # Weighted blend of failure (+error +uncertainty). The denominator is the sum of
        # ACTIVE weights so the result stays in [0,1] and reduces to each single term
        # when others are off. CRITICAL (Codex Phase-4 HIGH): the uncertainty term is
        # PER-BIN gated, so its weight must be added to the denominator ONLY on gated-ON
        # bins. Otherwise a gated-OFF bin would divide pure failure by (fw+uw) and shrink
        # it (e.g. 0.99 -> 0.495), wrongly weakening failure pressure on hopeless bins.
        num = self.failure_weight * pf
        denom = torch.full_like(pf, float(self.failure_weight))
        if self.error_weight > 0.0:
            e = ((self.error_bin_ema - self.error_good) / (self.error_bad - self.error_good + 1e-8)).clamp(0.0, 1.0)
            num = num + self.error_weight * e
            denom = denom + self.error_weight  # error has no gate -> applies to all bins
        if self.uncertainty_weight > 0.0:
            # SUCCESS GATE (discussion §3 Phase 4): uncertainty only counts where the
            # bin's success rate is in [lo, hi] — i.e. the policy is neither hopeless
            # (still failing -> failure term already drives it) nor mastered (don't chase
            # residual noise).
            sr = self.success_bin_count / (total + 1e-8)
            gate = ((sr >= self.uncertainty_gate_lo) & (sr <= self.uncertainty_gate_hi)).to(pf.dtype)
            u = self.uncertainty_bin_ema.clamp(0.0, 1.0) * gate
            num = num + self.uncertainty_weight * u
            denom = denom + self.uncertainty_weight * gate  # gated weight (per-bin)
        return num / denom.clamp(min=1e-8)

    def _smooth_bins(self, pf: Tensor, valid_mask: Tensor) -> Tensor:
        pf = pf.clone()
        pf[~valid_mask] = 0.0
        if self._kernel is not None:
            pf_3d = pf.unsqueeze(1)
            padding = self._kernel_size - 1
            pf_padded = torch.nn.functional.pad(pf_3d, (padding, 0), mode="constant", value=0.0)
            pf = torch.nn.functional.conv1d(pf_padded, self._kernel).squeeze(1)
            pf[~valid_mask] = 0.0
        return pf

    def _compute_pf_bin_smoothed(self) -> Tensor:
        return self._smooth_bins(self._pf_bin(), self.valid_bins_mask)

    def _compute_score_selected_smoothed(self, motion_ids: Tensor, valid_selected: Tensor) -> Tensor:
        pf_full = self._pf_bin()[motion_ids]
        return self._smooth_bins(pf_full, valid_selected)

    @staticmethod
    def _pow_and_normalize(values: Tensor, beta: float) -> Tensor:
        values = torch.clamp(values, min=1e-6)
        weighted = torch.pow(values, beta)
        return weighted / (weighted.sum() + 1e-8)

    def _mix_with_uniform(self, probs: Tensor) -> Tensor:
        uniform = 1.0 / float(self.num_motions)
        mixed = probs * (1.0 - self.uniform_ratio) + uniform * self.uniform_ratio
        return mixed / mixed.sum()

    def _motion_score(self, pf_bin: Tensor) -> Tensor:
        """Per-motion difficulty score from smoothed per-bin difficulty.

        Phase 0/1: hardest-bin-dominates (max over bins). Phase 3 (topk_motion>1)
        softens this to ``(1-w)*max + w*mean(top-k)`` so a single noisy hard bin does
        not let one clip monopolize the budget while its other bins are easy. With
        topk_motion==1 OR topk_motion_weight==0 this reduces EXACTLY to the max
        (Phase 0/1 parity).
        """
        max_score = pf_bin.max(dim=1).values
        if self.topk_motion <= 1 or self.topk_motion_weight <= 0.0:
            return max_score
        # PER-MOTION top-k over VALID bins only (Codex Phase-3 finding). pf_bin padding
        # bins are 0; if we naively topk over max_bins, a short clip's zero-padding
        # enters the mean and biases its score DOWN purely by length. Instead take, for
        # each motion, the mean of its top-k among its OWN valid bins (k capped at that
        # motion's bin count), dividing by the actual number of terms.
        kcap = int(min(self.topk_motion, self.max_bins))
        topk_vals = pf_bin.topk(kcap, dim=1).values            # [N, kcap], padding->0 sinks low
        # Number of valid terms per motion = min(kcap, motion_num_bins).
        k_per = torch.minimum(
            torch.full_like(self.motion_num_bins, kcap), self.motion_num_bins
        ).clamp(min=1).to(topk_vals.dtype)                      # [N]
        # Sum only the first k_per entries of each row (top values are first, padding
        # zeros are last after topk's descending sort), then divide by k_per.
        ar = torch.arange(kcap, device=pf_bin.device).unsqueeze(0)   # [1, kcap]
        keep = ar < k_per.unsqueeze(1)                          # [N, kcap]
        topk_mean = (topk_vals * keep).sum(dim=1) / k_per
        w = self.topk_motion_weight
        return (1.0 - w) * max_score + w * topk_mean

    def _learned_retention_probs(self) -> Tensor:
        """Uniform distribution over clips judged 'learned' (high success-EMA).

        Phase 3 anti-forgetting: a fixed budget kappa is reserved for replaying clips
        the policy already tracks well, so adding new/hard clips does not silently
        starve and forget them. 'Learned' = motion-level success rate >= threshold.
        Falls back to all-clips-uniform if none qualify yet (early training), so the
        retention mass is never lost. Returns a [num_motions] distribution summing to 1.
        """
        total = self.failed_motion_count + self.success_motion_count
        success_rate = self.success_motion_count / (total + 1e-8)
        learned = success_rate >= self.retention_success_thresh
        mass = learned.to(torch.float32)
        s = mass.sum()
        if s <= 0:
            # No learned clips yet -> spread retention budget uniformly (harmless).
            return torch.full((self.num_motions,), 1.0 / self.num_motions, device=self.device)
        return mass / s

    def _hard_buffer_probs(self, motion_score: Tensor) -> Tensor:
        """Uniform distribution over the current top-K hardest clips (Phase 4).

        Adversarial hard-buffer: reserve a fixed budget for the hardest clips so they
        always keep a guaranteed share of resets even if the softmax over scores would
        spread mass thinly across a huge dataset. Returns a [num_motions] distribution.

        Cold-start guard (Codex Phase-4 LOW): when scores are (near-)all-equal — e.g.
        before any outcomes — an arbitrary top-K subset would get a large reserved
        budget purely by index order. In that case fall back to uniform over ALL clips
        so the hard-buffer budget injects no spurious bias until scores differentiate.
        """
        k = int(min(self.hard_buffer_k, self.num_motions))
        if float(motion_score.max() - motion_score.min()) <= 1e-8:
            return torch.full((self.num_motions,), 1.0 / self.num_motions, device=self.device)
        topk_idx = motion_score.topk(k).indices
        mass = torch.zeros(self.num_motions, device=self.device, dtype=torch.float32)
        mass[topk_idx] = 1.0
        return mass / mass.sum()

    def _compute_motion_probs(self) -> Tensor:
        """Full motion sampling distribution with the budget mixture:

            p = rho*uniform + kappa*retention + eta*hard_buffer + (1-rho-kappa-eta)*hard

        With retention_ratio=0 AND hard_buffer_ratio=0 this is exactly the Phase-0/1
        ``hard + uniform`` mix (``_mix_with_uniform`` of the pow-normalized score),
        preserving parity.
        """
        pf_bin = self._compute_pf_bin_smoothed()
        motion_score = self._motion_score(pf_bin)
        hard = self._pow_and_normalize(motion_score, self.beta)
        if self.retention_ratio <= 0.0 and self.hard_buffer_ratio <= 0.0:
            return self._mix_with_uniform(hard)
        # Four-way budget mix. Validated at construction so the weights are in [0,1] and
        # rho+kappa+eta <= 1, leaving a nonneg residual for the hard term.
        rho = self.uniform_ratio
        kappa = self.retention_ratio
        eta = self.hard_buffer_ratio
        uniform = torch.full((self.num_motions,), 1.0 / self.num_motions, device=self.device)
        probs = (1.0 - rho - kappa - eta) * hard + rho * uniform
        if kappa > 0.0:
            probs = probs + kappa * self._learned_retention_probs()
        if eta > 0.0:
            probs = probs + eta * self._hard_buffer_probs(motion_score)
        return probs / probs.sum()

    def _update_metrics(self) -> None:
        motion_probs = self._compute_motion_probs()
        self._cached_motion_probs = motion_probs
        H = -(motion_probs * torch.log(motion_probs + 1e-12)).sum()
        self._last_entropy = (H / math.log(max(2, self.num_motions))).item()
        self._last_pfail_mean = self._motion_score(self._compute_pf_bin_smoothed()).mean().item()
        self._last_top1_prob = motion_probs.max().item()

    def _get_motion_probs(self) -> Tensor:
        if self._cached_motion_probs is None:
            self._update_metrics()
        assert self._cached_motion_probs is not None
        return self._cached_motion_probs
