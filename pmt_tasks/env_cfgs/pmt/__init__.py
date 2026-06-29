"""PMT pipeline env-cfg exports.

Submodules are imported lazily so a targeted PMT env import does not also import every
other PMT task and its Isaac dependencies.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SteppingStoneSceneCfg": ".stepping_stone",
    "SteppingStoneCommandsCfg": ".stepping_stone",
    "SteppingStoneObservationsCfg": ".stepping_stone",
    "TransformerSteppingStoneObservationsCfg": ".stepping_stone",
    "SteppingStoneRewardsCfg": ".stepping_stone",
    "SteppingStoneTerminationsCfg": ".stepping_stone",
    "PMTSteppingStoneEnvCfg": ".stepping_stone",
    "DistillSteppingStoneCommandsCfg": ".distill_stepping_stone",
    "DistillSteppingStoneObservationsCfg": ".distill_stepping_stone",
    "PMTSteppingStoneDistillEnvCfg": ".distill_stepping_stone",
    "TransformerDistillSteppingStoneObservationsCfg": ".distill_stepping_stone",
    "PMTSteppingStoneVisionLatentAnchorDistillEnvCfg": ".distill_stepping_stone",
    "BackFlipRewardsCfg": ".backflip",
    "PMTBackFlipEnvCfg": ".backflip",
    "TerrainFlatUnifiedCommandsCfg": ".terrain_flat_mix",
    "PMTTerrainFlatMixEnvCfg": ".terrain_flat_mix",
    "PMTTerrainFlatMixEnvCfg_PLAY": ".terrain_flat_mix",
    "PMTAdaptiveSamplingObservationsCfg": ".adaptive_sampling",
    "PMTAdaptiveSamplingEnvCfg": ".adaptive_sampling",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
