"""PerceptiveResidualBehaviorTokenTracker — the P-CaRBT network.

P-CaRBT = Perception-Conditioned Contact-Aware Residual Behavior Tokenizer
(design: docs/latent_space_represent_pmt.md; plan: docs/pcrbt_implementation_plan.md).

This subclasses ``PerceptiveMotionTokenTracker`` and changes exactly three things:
  1. swaps the continuous ``_MotionTokenizer`` for the discrete
     ``_ResidualFSQBehaviorTokenizer`` (FSQ behavior tokens);
  2. enables the optional sin/cos gait-phase head on the aux decoder;
  3. adds ``compute_sonic_aux_losses`` so PPO (motion_tracking_rl/algorithms/ppo.py:562)
     can train the tokenizer with FSQ-usage-entropy + motion-reconstruction (and,
     when contact/phase labels exist, contact/phase) auxiliary losses.

Everything else — the closed-loop ``_TokenConditionedPMTDecoder``, the
``_TerrainMotionAdapter`` (continuous perception residual), the critic, the
checkpoint/freeze machinery — is inherited unchanged. The class registers under the
EXISTING compat axis ``perceptive_motion_token_tracker`` so it stays compatible with
the ``ppo`` algorithm spec without a compat.py edit (plan-review C1).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from motion_tracking_rl.registry import register_network

from .token_tracker import (
    PerceptiveMotionTokenTracker,
    _MotionAuxDecoder,
    _ResidualFSQBehaviorTokenizer,
)


@register_network(
    "PerceptiveResidualBehaviorTokenTracker",
    compat_name="perceptive_motion_token_tracker",
)
class PerceptiveResidualBehaviorTokenTracker(PerceptiveMotionTokenTracker):
    """PMT token tracker with a discrete residual-FSQ behavior tokenizer (P-CaRBT)."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        fsq_levels: Sequence[int] = (8, 8, 8, 5, 5),
        num_residual_levels: int = 1,
        token_groups: dict[str, list[str]] | None = None,
        use_phase_head: bool = False,
        aux_loss_coef: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        # Stash P-CaRBT-specific knobs that the parent __init__ would reject, then
        # restore the tokenizer / aux-decoder the parent built with the FSQ versions.
        self.fsq_levels = [int(level) for level in fsq_levels]
        self.num_residual_levels = int(num_residual_levels)
        self.use_phase_head = bool(use_phase_head)
        self.aux_loss_coef = {
            str(name): float(value) for name, value in (aux_loss_coef or {}).items()
        }
        # Grouped tokenization is deferred (q_ref joint order is interleaved BFS; a
        # hard-coded leg slice is wrong — see the PCRBT joint-order note).
        # The knob is accepted so configs can carry it, but must be unset/off here.
        if token_groups:
            raise NotImplementedError(
                "token_groups (grouped behavior tokenization) is not supported yet. "
                "Leave it unset for the feasibility run; grouping requires name-resolved "
                "joint indices ."
            )

        # P-CaRBT trains the FSQ tokenizer FROM SCRATCH. A continuous-PMT checkpoint
        # cannot seed an FSQ tokenizer (the fsq_in/fsq_quantizers/fsq_out keys do not
        # exist in it), and the parent would load it into the to-be-discarded continuous
        # tokenizer. Forbid it loudly rather than silently discarding the load.
        ckpt = next(
            (
                kwargs.get(key)
                for key in (
                    "pmt_ckpt_path",
                    "perceptive_motion_tracker_ckpt_path",
                    "percaptive_motion_tracker_ckpt_path",
                    "teacher_ckpt_path",
                )
                if kwargs.get(key)
            ),
            None,
        )
        if ckpt:
            raise ValueError(
                "PerceptiveResidualBehaviorTokenTracker trains the FSQ tokenizer from "
                "scratch and does not accept a PMT checkpoint path (a continuous-PMT "
                f"checkpoint cannot seed an FSQ tokenizer). Got: {ckpt}."
            )

        super().__init__(obs, obs_groups, num_actions, **kwargs)

        # Resolve the future-motion dim exactly as the parent did.
        future_motion_dim = self._obs_group_dim(obs, self.future_motion_set_name)

        # Replace the continuous tokenizer with the discrete residual-FSQ tokenizer.
        self.motion_tokenizer = _ResidualFSQBehaviorTokenizer(
            future_motion_dim=future_motion_dim,
            future_motion_len=self.future_motion_len,
            model_dim=self.model_dim,
            token_dim=self.motion_token_dim,
            num_tokens=self.num_motion_tokens,
            activation=self.activation,
            num_heads=self.token_num_heads,
            fsq_levels=self.fsq_levels,
            num_residual_levels=self.num_residual_levels,
        )

        # Rebuild the aux decoder with the phase head enabled if requested.
        if self.motion_aux_decoder is not None and self.use_phase_head:
            self.motion_aux_decoder = _MotionAuxDecoder(
                self.motion_token_dim,
                future_motion_dim,
                self.model_dim,
                self.activation,
                use_phase_head=True,
            )

        # The parent may have re-frozen modules / loaded a checkpoint before we swapped
        # the tokenizer. Re-apply the freeze policy so the new modules match.
        if self.freeze_pmt:
            self._freeze_pmt()
        if self.pmt_only_mode:
            self._freeze_pma(include_critic=False)

    # ------------------------------------------------------------------
    # PPO auxiliary-loss hook (consumed at ppo.py:562)
    # ------------------------------------------------------------------
    def compute_sonic_aux_losses(self, obs: TensorDict) -> dict[str, Any]:
        """Return behavior-tokenizer auxiliary losses for the PPO aux path.

        Contract (ppo.py:562-595): return ``{"aux_losses": {name: scalar_tensor},
        "aux_loss_coef": {name: float}}``. PPO multiplies each term by its coef
        (algorithm-cfg ``aux_loss_coef`` OVERRIDES the dict returned here) and by the
        global ``aux_loss_scale``; a term with no configured coef is weighted 0.0.

        Active on flat-ground feasibility: ``fsq_usage_entropy`` (anti-dead-code) and
        ``motion_recon`` (pose/velocity reconstruction of the future window). Contact /
        phase terms are emitted only when their supervision targets are present in obs.
        """
        outputs = self._compute_student_outputs(obs, include_teacher=False)
        aux_losses: dict[str, torch.Tensor] = {}

        # --- FSQ usage entropy: maximize per-dimension code-usage entropy so the
        # quantizer does not collapse onto a few codes. We MINIMIZE the negated
        # entropy. This uses the DIFFERENTIABLE soft-usage surrogate on the bounded
        # pre-quant grid (carries gradient into fsq_in), not a detached histogram.
        get_soft_entropy = getattr(self.motion_tokenizer, "soft_usage_entropy", None)
        usage_entropy = get_soft_entropy() if get_soft_entropy is not None else None
        if usage_entropy is not None:
            aux_losses["fsq_usage_entropy"] = -usage_entropy

        # --- Motion reconstruction (L_pose / L_vel): the aux decoder reconstructs the
        # future-motion window from the (quantized) token; FSQ STE lets gradient flow
        # back into the tokenizer.
        if "future_motion_hat" in outputs:
            target = self._get_concat_flat(obs, self.future_motion_set_name)
            pred = outputs["future_motion_hat"]
            if pred.shape == target.shape:
                aux_losses["motion_recon"] = F.mse_loss(pred, target)

        # --- Optional contact / phase supervision (disabled on flat: no labels).
        contact_target = self._optional_obs_tensor(obs, "contact_target")
        if contact_target is not None and "contact_logits_aux" in outputs:
            aux_losses["contact_bce"] = F.binary_cross_entropy_with_logits(
                outputs["contact_logits_aux"], contact_target
            )
        phase_target = self._optional_obs_tensor(obs, "phase_target")
        if phase_target is not None and "phase_hat" in outputs:
            aux_losses["phase_mse"] = F.mse_loss(outputs["phase_hat"], phase_target)

        return {"aux_losses": aux_losses, "aux_loss_coef": dict(self.aux_loss_coef)}

    @torch.no_grad()
    def fsq_usage_entropy_monitor(self) -> float | None:
        """Monitor-only HARD code-usage entropy from the last forward's FSQ indices.

        Histogram of the (detached, argmax) integer codes per FSQ dimension, normalized
        by log(num_levels) and averaged. This is the true discrete usage (0=collapsed,
        1=uniform) for LOGGING/dead-code detection — it has no gradient. The trainable
        anti-collapse signal is the differentiable ``soft_usage_entropy`` surrogate used
        in ``compute_sonic_aux_losses``.
        """
        tokenizer = self.motion_tokenizer
        get_indices = getattr(tokenizer, "last_code_indices", None)
        levels = get_indices() if get_indices is not None else None
        if not levels:
            return None
        fsq_levels = tokenizer.fsq_levels
        device = levels[0].device
        entropies: list[torch.Tensor] = []
        for idx in levels:  # idx: [B, num_tokens, fsq_dim] long
            flat = idx.reshape(-1, idx.shape[-1])  # [N, fsq_dim]
            for dim, num_levels in enumerate(fsq_levels):
                counts = torch.bincount(
                    flat[:, dim].clamp(0, num_levels - 1),
                    minlength=num_levels,
                ).float()
                probs = counts / counts.sum().clamp_min(1.0)
                ent = -(probs * probs.clamp_min(1.0e-9).log()).sum()
                entropies.append(ent / torch.log(torch.tensor(float(num_levels), device=device)))
        return float(torch.stack(entropies).mean().item())

    def _optional_obs_tensor(self, obs: TensorDict, set_name: str) -> torch.Tensor | None:
        if set_name in self.obs_groups and self._obs_has_group_tensors(obs, set_name):
            return self._get_concat_flat(obs, set_name)
        if set_name in obs:
            return obs[set_name].reshape(obs[set_name].shape[0], -1)
        return None
