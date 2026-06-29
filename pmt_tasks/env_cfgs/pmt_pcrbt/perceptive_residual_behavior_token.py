"""P-CaRBT flat-ground env cfg (FSQ behavior tokenizer feasibility run).

This is the FLAT-plane env for the ``pmt_pcrbt`` task. Per the plan-review C2/C3
corrections (docs/pcrbt_implementation_plan.md Phase 2):

- It is built on the FLAT multimotion env (``PMTMultiMotionFlatEnvCfg``: flat plane,
  V2 multi-clip command named ``motion``, no terrain mesh) — NOT the stepping-stone /
  big_map token env. This is what lets the feasibility run train on flat lafan1.
- It exposes ONLY the obs groups the ``PerceptiveResidualBehaviorTokenTracker`` needs
  in ``pmt_only_mode`` (proprio current + history, the future-motion window, critic).
  The ``vision``/height-scan group is intentionally DROPPED (reduced OBS is permitted;
  ``pmt_only_mode`` skips the terrain adapter, so a height scan would be dead weight and
  the flat plane has no scanner geometry anyway).

The token/window obs groups are the same ones the transformer/token stack uses
(``proprio``, ``proprio_history`` len-10 unflattened, ``command_window``,
``motion_anchor_delta_window``), all reading the flat env's ``motion`` command.
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.env_cfgs.multi_motion_flat import (
    ObservationsCfg,
    PMTMultiMotionFlatEnvCfg,
)


@configclass
class PCaRBTFlatObservationsCfg(ObservationsCfg):
    """Flat multimotion policy/critic obs + the token/window groups (no vision).

    Inherits the flat env's ``policy`` and ``critic`` groups (the multimotion base
    ``ObservationsCfg``) and adds the four token-tracker groups. All groups read the
    flat env's ``motion`` command.
    """

    @configclass
    class ProprioCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CommandWindowCfg(ObsGroup):
        command_window = ObsTerm(
            func=mdp.command_window,
            params={"command_name": "motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MotionAnchorDeltaWindowCfg(ObsGroup):
        motion_anchor_delta_window = ObsTerm(
            func=mdp.motion_anchor_delta_window,
            params={"command_name": "motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    command_window: CommandWindowCfg = CommandWindowCfg()
    motion_anchor_delta_window: MotionAnchorDeltaWindowCfg = MotionAnchorDeltaWindowCfg()


@configclass
class PMTPCaRBTFlatEnvCfg(PMTMultiMotionFlatEnvCfg):
    """Flat-plane P-CaRBT env: multimotion-flat base + token/window obs groups."""

    observations: PCaRBTFlatObservationsCfg = PCaRBTFlatObservationsCfg()
