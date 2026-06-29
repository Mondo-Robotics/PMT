"""PerceptiveMotion network family (split from the 2017-line monolith).

Public classes:
  - PerceptiveMotionAdapter           (adapter.py)
  - PerceptiveMotionAdapterTracker    (adapter_tracker.py)
  - PerceptiveMotionTokenTracker      (token_tracker.py)
  - PerceptiveMotionTracker / PercaptiveMotionTracker  (= TransformerActorCritic, _aliases.py)

CHECKPOINT/IMPORT SAFETY: state_dict keys are by attribute-path within each
module (unchanged by this split). The OLD module path
``motion_tracking_rl.networks.perceptive_motion_adapter_tracker`` is kept
importable as a compat shim that re-exports these SAME class objects, so pickled
cfg __qualname__/__module__ and string class_name dispatch still resolve.
"""
from __future__ import annotations

from ._aliases import PercaptiveMotionTracker, PerceptiveMotionTracker
from .adapter import PerceptiveMotionAdapter
from .adapter_tracker import PerceptiveMotionAdapterTracker
from .token_tracker import PerceptiveMotionTokenTracker
from .behavior_token_tracker import PerceptiveResidualBehaviorTokenTracker

__all__ = [
    "PercaptiveMotionTracker",
    "PerceptiveMotionTracker",
    "PerceptiveMotionAdapter",
    "PerceptiveMotionAdapterTracker",
    "PerceptiveMotionTokenTracker",
    "PerceptiveResidualBehaviorTokenTracker",
]
