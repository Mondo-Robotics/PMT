"""Bin-based adaptive sampler for large-scale motion datasets.

This module provides a sampler that tracks success/failure statistics at a
temporal-bin level instead of per-frame. It is designed for datasets with many
motions and long trajectories where frame-level adaptive sampling becomes
prohibitively expensive (memory + multinomial over huge categorical spaces).

Key behaviors:
1) Sample hierarchy: motion -> bin -> frame (uniform within sampled bin).
2) Motion selection uses "hardest bin dominates": score(motion) = max_b p_fail(bin).
3) Bin selection uses per-bin failure probability with optional kernel smoothing.
4) Counts are updated for the *starting* frame/bin of each episode (same as the
   existing AdaptiveSampler behavior that attributes an outcome to start_frame).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .multi_motion_command import MotionSampler, SamplingResult


class BinBasedAdaptiveSampler(MotionSampler):
    """Adaptive sampling using temporal bins instead of per-frame statistics."""

    def __init__(
        self,
        num_motions: int,
        motion_lengths: Tensor,
        device: torch.device,
        *,
        motion_fps: float = 50.0,
        bin_duration: float = 1.0,
        beta: float = 0.5,
        alpha: float = 0.001,
        uniform_ratio: float = 0.1,
        update_interval: int = 240,
        kernel_size: int = 5,
        kernel_lambda: float = 0.8,
    ) -> None:
        super().__init__(num_motions, motion_lengths, device)

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

        self.beta = float(beta)
        self.alpha = float(alpha)
        self.uniform_ratio = float(uniform_ratio)
        self.update_interval = int(update_interval)

        bin_size = int(round(float(motion_fps) * float(bin_duration)))
        self.bin_size = max(1, bin_size)

        # Number of bins per motion (>= 1)
        self.motion_num_bins = ((self.motion_lengths + (self.bin_size - 1)) // self.bin_size).clamp(min=1).long()
        self.max_bins = int(self.motion_num_bins.max().item())

        # Motion-level counts (always tracked).
        self.failed_motion_count = torch.ones(num_motions, device=device, dtype=torch.float32)
        self.success_motion_count = torch.ones(num_motions, device=device, dtype=torch.float32)
        self.current_failed_motion_count = torch.zeros(num_motions, device=device, dtype=torch.float32)
        self.current_success_motion_count = torch.zeros(num_motions, device=device, dtype=torch.float32)

        # Bin-level counts.
        self.failed_bin_count = torch.ones(num_motions, self.max_bins, device=device, dtype=torch.float32)
        self.success_bin_count = torch.ones(num_motions, self.max_bins, device=device, dtype=torch.float32)
        self.current_failed_bin_count = torch.zeros(num_motions, self.max_bins, device=device, dtype=torch.float32)
        self.current_success_bin_count = torch.zeros(num_motions, self.max_bins, device=device, dtype=torch.float32)

        # Valid bins mask: [num_motions, max_bins]
        bin_indices = torch.arange(self.max_bins, device=device).unsqueeze(0)
        self.valid_bins_mask = bin_indices < self.motion_num_bins.unsqueeze(1)
        self.total_valid_bins = int(self.valid_bins_mask.sum().item())

        # Optional smoothing kernel over bins.
        # We use LEFT padding so that a failure at bin i increases probability for i, i+1, ..., i+k-1.
        if kernel_size > 1:
            kernel = torch.tensor(
                [kernel_lambda ** (kernel_size - 1 - i) for i in range(kernel_size)],
                device=device,
                dtype=torch.float32,
            )
            self._kernel = (kernel / kernel.sum()).view(1, 1, -1)
            self._kernel_size = int(kernel_size)
        else:
            self._kernel = None
            self._kernel_size = 0

        self._steps_since_update = 0

        # Cached metrics.
        self._last_entropy = 0.0
        self._last_pfail_mean = 0.0
        self._last_top1_prob = 0.0
        self._cached_motion_probs: Tensor | None = None

        # Initialize cached metrics/probabilities from the (uniform) priors.
        self._update_metrics()

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    def sample(self, num_samples: int) -> SamplingResult:
        if num_samples <= 0:
            empty = torch.empty((0,), device=self.device, dtype=torch.long)
            return SamplingResult(motion_ids=empty, frame_ids=empty)

        motion_probs = self._get_motion_probs()

        motion_ids = torch.multinomial(motion_probs, num_samples, replacement=True)

        # Bin selection within each selected motion.
        valid_selected = self.valid_bins_mask[motion_ids]  # [N, B]
        pf_selected = self._compute_pf_selected_smoothed(motion_ids, valid_selected)  # [N, B]

        # Build row-wise weighted probabilities (masked).
        pf_selected = pf_selected.clone()
        pf_selected[~valid_selected] = 0.0

        pf_selected = torch.clamp(pf_selected, min=1e-6)
        pf_selected[~valid_selected] = 0.0

        pf_weighted = torch.pow(pf_selected, self.beta)
        pf_weighted[~valid_selected] = 0.0
        pf_weighted = pf_weighted / (pf_weighted.sum(dim=1, keepdim=True) + 1e-8)

        # Uniform distribution per row over valid bins.
        valid_counts = self.motion_num_bins[motion_ids].clamp(min=1).unsqueeze(1)
        uniform_row = valid_selected.to(dtype=torch.float32) / valid_counts

        bin_probs = pf_weighted * (1.0 - self.uniform_ratio) + uniform_row * self.uniform_ratio
        bin_probs[~valid_selected] = 0.0
        bin_probs = bin_probs / (bin_probs.sum(dim=1, keepdim=True) + 1e-8)

        bin_ids = torch.multinomial(bin_probs, 1, replacement=True).squeeze(1)

        # Convert bin -> frame with within-bin diversity.
        bin_start = bin_ids * self.bin_size
        lengths = self.motion_lengths[motion_ids]

        # Bin length (last bin may be shorter).
        bin_len = torch.clamp(lengths - bin_start, min=1, max=self.bin_size)
        offset = (torch.rand(num_samples, device=self.device) * bin_len.to(dtype=torch.float32)).long()

        frame_ids = bin_start + offset
        frame_ids = torch.minimum(frame_ids, lengths - 1)

        return SamplingResult(motion_ids=motion_ids, frame_ids=frame_ids)

    # ---------------------------------------------------------------------
    # Updates
    # ---------------------------------------------------------------------

    def update(self, motion_ids: Tensor, frame_ids: Tensor, terminated: Tensor) -> None:
        if motion_ids.numel() == 0:
            return

        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        frame_ids = frame_ids.to(device=self.device, dtype=torch.long)
        terminated = terminated.to(device=self.device, dtype=torch.bool)

        failed_mask = terminated
        success_mask = ~terminated

        # Motion-level counts.
        if failed_mask.any():
            failed_ids = motion_ids[failed_mask]
            counts = torch.bincount(failed_ids, minlength=self.num_motions).to(dtype=torch.float32, device=self.device)
            self.current_failed_motion_count += counts

        if success_mask.any():
            success_ids = motion_ids[success_mask]
            counts = torch.bincount(success_ids, minlength=self.num_motions).to(dtype=torch.float32, device=self.device)
            self.current_success_motion_count += counts

        # Bin-level counts: attribute outcome to the start bin.
        bin_ids = (frame_ids // self.bin_size).to(dtype=torch.long)
        # Clamp to valid range for each motion.
        max_valid_bin = (self.motion_num_bins[motion_ids] - 1).clamp(min=0)
        bin_ids = torch.minimum(bin_ids, max_valid_bin)
        bin_ids = torch.clamp(bin_ids, min=0, max=self.max_bins - 1)

        flat_index = motion_ids * self.max_bins + bin_ids
        flat_failed = self.current_failed_bin_count.view(-1)
        flat_success = self.current_success_bin_count.view(-1)

        if failed_mask.any():
            idx = flat_index[failed_mask]
            inc = torch.ones_like(idx, dtype=flat_failed.dtype, device=self.device)
            flat_failed.scatter_add_(0, idx, inc)

        if success_mask.any():
            idx = flat_index[success_mask]
            inc = torch.ones_like(idx, dtype=flat_success.dtype, device=self.device)
            flat_success.scatter_add_(0, idx, inc)

    def step(self) -> None:
        """Called each simulation step to update internal EMA state."""
        self._steps_since_update += 1

        if self._steps_since_update >= self.update_interval:
            self._ema_update()
            self._reset_counts()
            self._steps_since_update = 0

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------

    def get_metrics(self) -> dict[str, float]:
        return {
            "adaptive_entropy": float(self._last_entropy),
            "adaptive_pfail_mean": float(self._last_pfail_mean),
            "adaptive_top1_prob": float(self._last_top1_prob),
            "total_failed": float(self.failed_motion_count.sum().item()),
            "total_success": float(self.success_motion_count.sum().item()),
        }

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _ema_update(self) -> None:
        alpha = float(self.alpha)

        self.failed_motion_count = alpha * self.current_failed_motion_count + (1.0 - alpha) * self.failed_motion_count
        self.success_motion_count = (
            alpha * self.current_success_motion_count + (1.0 - alpha) * self.success_motion_count
        )

        self.failed_bin_count = alpha * self.current_failed_bin_count + (1.0 - alpha) * self.failed_bin_count
        self.success_bin_count = alpha * self.current_success_bin_count + (1.0 - alpha) * self.success_bin_count

        self._update_metrics()

    def _reset_counts(self) -> None:
        self.current_failed_motion_count.zero_()
        self.current_success_motion_count.zero_()
        self.current_failed_bin_count.zero_()
        self.current_success_bin_count.zero_()

    def _update_metrics(self) -> None:
        # Recompute motion probabilities for logging/caching.
        pf_bin = self._compute_pf_bin_smoothed()
        motion_score = pf_bin.max(dim=1).values
        motion_probs = self._mix_with_uniform(self._pow_and_normalize(motion_score, self.beta), self.uniform_ratio)
        self._cached_motion_probs = motion_probs

        # Normalized entropy.
        H = -(motion_probs * torch.log(motion_probs + 1e-12)).sum()
        self._last_entropy = (H / math.log(max(2, self.num_motions))).item()

        self._last_pfail_mean = motion_score.mean().item()
        self._last_top1_prob = motion_probs.max().item()

    def _get_motion_probs(self) -> Tensor:
        if self._cached_motion_probs is None:
            self._update_metrics()
        assert self._cached_motion_probs is not None
        return self._cached_motion_probs

    def _compute_pf_bin_smoothed(self) -> Tensor:
        total = self.failed_bin_count + self.success_bin_count
        pf_bin = self.failed_bin_count / (total + 1e-8)

        pf_bin = pf_bin.clone()
        pf_bin[~self.valid_bins_mask] = 0.0

        if self._kernel is not None:
            pf_3d = pf_bin.unsqueeze(1)  # [M, 1, B]
            padding = self._kernel_size - 1
            pf_padded = torch.nn.functional.pad(pf_3d, (padding, 0), mode="constant", value=0.0)
            pf_smoothed = torch.nn.functional.conv1d(pf_padded, self._kernel)
            pf_bin = pf_smoothed.squeeze(1)
            pf_bin[~self.valid_bins_mask] = 0.0

        return pf_bin

    def _compute_pf_selected_smoothed(self, motion_ids: Tensor, valid_selected: Tensor) -> Tensor:
        failed = self.failed_bin_count[motion_ids]
        success = self.success_bin_count[motion_ids]
        pf = failed / (failed + success + 1e-8)

        pf = pf.clone()
        pf[~valid_selected] = 0.0

        if self._kernel is not None:
            pf_3d = pf.unsqueeze(1)  # [N, 1, B]
            padding = self._kernel_size - 1
            pf_padded = torch.nn.functional.pad(pf_3d, (padding, 0), mode="constant", value=0.0)
            pf_smoothed = torch.nn.functional.conv1d(pf_padded, self._kernel)
            pf = pf_smoothed.squeeze(1)
            pf[~valid_selected] = 0.0

        return pf

    @staticmethod
    def _pow_and_normalize(values: Tensor, beta: float) -> Tensor:
        values = torch.clamp(values, min=1e-6)
        weighted = torch.pow(values, beta)
        return weighted / (weighted.sum() + 1e-8)

    def _mix_with_uniform(self, probs: Tensor, uniform_ratio: float) -> Tensor:
        uniform = 1.0 / float(self.num_motions)
        mixed = probs * (1.0 - uniform_ratio) + uniform * uniform_ratio
        return mixed / mixed.sum()
