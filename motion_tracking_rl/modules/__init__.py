# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy compatibility shim for the former ``motion_tracking_rl.modules`` package.

The high-level network modules moved to ``motion_tracking_rl.networks`` (PMT
plan §2). This shim keeps the old import paths working so that:

* ``from motion_tracking_rl.modules import ActorCritic`` resolves to the new
  class (symbol re-export below), and
* ``from motion_tracking_rl.modules.transformer_actor_critic import ...`` and
  unpickling old configs that reference ``motion_tracking_rl.modules.<sub>``
  resolve to the SAME module object as the new location (sys.modules aliasing
  below) — checkpoint / qualname safety, PMT plan §4.

Both old and new paths therefore return identical class objects (``is``).
"""

from __future__ import annotations

import importlib
import sys as _sys

# Submodules that physically moved from modules/ -> networks/. Each old dotted
# path is aliased in sys.modules to the corresponding new module object, so any
# old fully-qualified reference (imports, pickled __module__) keeps resolving.
_MOVED_SUBMODULES = (
    "actor_critic",
    "actor_critic_recurrent",
    "deploy_residual_vision_sonic",
    "diff_normalizer",
    "diffusion_actor_critic",
    "ode_solver",
    "official_sonic_actor_critic",
    "perceptive_motion_adapter_tracker",
    "residual_sonic_teacher",
    "residual_vision_action",
    "residual_vision_sonic",
    "vision_ablation_actor_critic",
    "rnd",
    "sonic_diffusion_student_teacher",
    "student_teacher",
    "student_teacher_recurrent",
    "symmetry",
    "transformer_actor_critic",
    "vision_student_teacher",
    "vision_sonic_latent_distill",
    "vision_sonic",
    "vision_transformer_actor_critic",
)

for _name in _MOVED_SUBMODULES:
    _new = importlib.import_module(f"motion_tracking_rl.networks.{_name}")
    _sys.modules[f"{__name__}.{_name}"] = _new
    globals()[_name] = _new

# Also alias the relocated torchdiffeq vendored package (modules.torchdiffeq -> networks.torchdiffeq).
try:
    _td = importlib.import_module("motion_tracking_rl.networks.torchdiffeq")
    _sys.modules[f"{__name__}.torchdiffeq"] = _td
    globals()["torchdiffeq"] = _td
except Exception:  # pragma: no cover - optional vendored dep
    pass

# Re-export the public symbol surface so ``from motion_tracking_rl.modules import X`` works.
from motion_tracking_rl.networks import *  # noqa: F401,F403,E402
from motion_tracking_rl.networks import __all__ as _networks_all  # noqa: E402

__all__ = list(_networks_all)

del _name, _new, importlib, _sys
