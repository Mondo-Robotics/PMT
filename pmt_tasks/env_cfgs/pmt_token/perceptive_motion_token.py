"""PMT (PerceptiveMotionTokenTracker) PPO-pretrain env cfg (plan §6 Phase 2.5).

Reuses the stepping-stone base (PMTSteppingStoneEnvCfg) and adds a flat
height-scan ``vision`` obs group so the env exposes BOTH the future-motion-window
groups (command_window + motion_anchor_delta_window) AND a height_scan
group. The PMT token-tracker network maps:

    obs_groups = {
      "policy":               ["policy", "proprio"],
      "policy_history":       ["proprio_history"],
      "future_motion_window": ["command_window", "motion_anchor_delta_window"],
      "height_scan":          ["vision"],
      "critic":               ["critic"],
    }

The PMT-pretrain runner runs the token tracker in ``pmt_only_mode=True`` /
``require_height_scan=False`` so it trains the PMT decoder + tokenizer FROM SCRATCH
(no pretrained PMT checkpoint, require_pmt_checkpoint=False) — this is the data/ckpt
un-gated gate target. The height_scan group is still present (PMA path can be
enabled by a config flip later).

Source faithfulness: mirrors
``PerceptiveMotionTokenTrackerTerrainFlatAllMotionsEnvCfg`` obs_groups
(distill_stepping_stone_env_cfg.py:285), reusing the local stepping-stone terrain
+ clips rather than the cluster terrain-flat dataset.
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.env_cfgs.pmt.stepping_stone import (
    PMTSteppingStoneEnvCfg,
    TransformerSteppingStoneObservationsCfg,
)


@configclass
class PMTTokenObservationsCfg(TransformerSteppingStoneObservationsCfg):
    """stepping-stone obs + a flat height-scan ``vision`` group for PMT."""

    @configclass
    class VisionCfg(ObsGroup):
        vision = ObsTerm(
            func=mdp.height_scan_for_vision,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "nan_dropout_prob": 0.0,
                "append_validity_mask": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    vision: VisionCfg = VisionCfg()


@configclass
class PMTPerceptiveMotionTokenTrackerEnvCfg(PMTSteppingStoneEnvCfg):
    """One-stage PMT token-tracker PPO-pretrain env (stepping-stone terrain)."""

    observations: PMTTokenObservationsCfg = PMTTokenObservationsCfg()
