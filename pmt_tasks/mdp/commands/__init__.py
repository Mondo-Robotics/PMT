"""Motion-tracking command stack (PMT reorg 2026-06-25).

All command-term classes, samplers, motion stores, and the terrain-IK adapter live
here. The parent ``mdp/__init__.py`` does ``from .commands import *`` so the public
surface (``pmt_tasks.mdp.MotionCommand`` etc.) is unchanged after the reorg.

Re-exports mirror the OLD flat ``mdp/__init__.py`` exactly, so qualified imports like
``from pmt_tasks.mdp.commands import MotionCommand`` resolve to this package's __init__
(NOT the inner ``commands.py`` module — the re-export below makes them identical).
"""
from __future__ import annotations

# Base command terms + loaders (was: `from .commands import *`).
from .commands import (  # noqa: F401
    MotionLoader,
    MotionCommand,
    MotionCommandCfg,
    MultiMotionCommand,
    MultiMotionCommandCfg,
    SingleMotionLoader,
    SingleMotionCommand,
    SingleMotionCommandCfg,
    SyncedStudentMotionCommand,
    SyncedStudentMotionCommandCfg,
)

# V2 multi-motion architecture (store / samplers / command).
from .multi_motion_command import (  # noqa: F401
    MotionData,
    MotionDataStore,
    MotionSampler,
    UniformSampler,
    AdaptiveSampler,
    SamplingResult,
    MultiMotionCommandV2,
    MultiMotionCommandV2Cfg,
    SyncedStudentMultiMotionCommandV2,
    SyncedStudentMultiMotionCommandV2Cfg,
)

# Specialized command variants.
from .bin_based_sampler import BinBasedAdaptiveSampler  # noqa: F401
from .streaming_motion_command import (  # noqa: F401
    StreamingMultiMotionCommand,
    StreamingMultiMotionCommandV2Cfg,
)
from .unified_motion_command import (  # noqa: F401
    UnifiedMultiMotionCommand,
    UnifiedMultiMotionCommandCfg,
)
from .adaptive_sampling_lib import HybridBinSampler  # noqa: F401
from .adaptive_sampling_motion_command import (  # noqa: F401
    AdaptiveSamplingMotionCommand,
    AdaptiveSamplingMotionCommandCfg,
)
