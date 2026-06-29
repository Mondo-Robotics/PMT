"""Backward-compatible aliases for the perceptive-motion family.

``PerceptiveMotionTracker`` / ``PercaptiveMotionTracker`` (note the misspelling
preserved for old checkpoints/configs) are NOT separate classes — they resolve to
``TransformerActorCritic``. Kept as standalone module to avoid a circular import
between adapter_tracker / token_tracker.
"""
from __future__ import annotations

from motion_tracking_rl.networks.transformer_actor_critic import TransformerActorCritic

PerceptiveMotionTracker = TransformerActorCritic
PercaptiveMotionTracker = TransformerActorCritic  # Backward-compatible misspelled alias.
