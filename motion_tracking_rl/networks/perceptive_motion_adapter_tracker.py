"""Backward-compat shim for the OLD monolith module path.

The 2017-line monolith was split into the ``motion_tracking_rl.networks.perceptive_motion``
subpackage. This shim keeps the OLD module path importable so that:
  * pickled cfg __qualname__/__module__ that referenced this module still resolve,
  * string class_name dispatch (``PerceptiveMotionTokenTracker`` etc.) resolves to
    the SAME class object as the new subpackage,
  * existing imports `from ...perceptive_motion_adapter_tracker import X` keep working.

Do NOT redefine the classes here — re-export them so identity is preserved
(`old.PerceptiveMotionTokenTracker is new.PerceptiveMotionTokenTracker`).
"""
from __future__ import annotations

from motion_tracking_rl.networks.perceptive_motion import (
    PercaptiveMotionTracker,
    PerceptiveMotionAdapter,
    PerceptiveMotionAdapterTracker,
    PerceptiveMotionTokenTracker,
    PerceptiveMotionTracker,
)

# Internal helpers / nested classes that older code or checkpoints may import by
# the old module path. Re-exported so the old qualified names still resolve.
from motion_tracking_rl.networks.perceptive_motion.adapter import (
    _small_init_last_linear,
    _zero_init_last_linear,
)
from motion_tracking_rl.networks.perceptive_motion.token_tracker import (
    _FootEventLatentEncoder,
    _HeightScanEncoder,
    _MotionAuxDecoder,
    _MotionTokenizer,
    _ProprioHistoryEncoder,
    _TerrainMotionAdapter,
    _TokenConditionedPMTDecoder,
)

__all__ = [
    "PercaptiveMotionTracker",
    "PerceptiveMotionTracker",
    "PerceptiveMotionAdapter",
    "PerceptiveMotionAdapterTracker",
    "PerceptiveMotionTokenTracker",
]
