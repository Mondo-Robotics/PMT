"""BFM-Zero env-cfgs (off-policy FB-CPR tracking). The RL machinery lives in
motion_tracking_rl.bfm_zero; this package holds only the IsaacLab env cfgs."""
from .bfm_zero import (
    BFMZeroZeroRewardsCfg,
    BFMZeroTimeoutTerminationsCfg,
    BFMZeroG1TerrainFlatStreamingEnvCfg,
    BFMZeroG1FlatMultiMotionV2EnvCfg,
    BFMZeroG1FlatMultiMotionV2EnvCfg_PLAY,
)
