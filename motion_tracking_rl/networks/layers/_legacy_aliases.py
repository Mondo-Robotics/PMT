# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy import-path aliases for checkpoint / pickle compatibility (PMT plan §4).

State_dict keys are qualified by the *attribute path within the nn.Module*, not by
the module file path, so moving files does not change state_dict keys. What CAN
break are pickled configs / fully-qualified references that name the OLD module
path (``motion_tracking_rl.modules.<sub>``). This module exists as the documented,
single place that establishes those aliases.

What's covered:

* ``motion_tracking_rl.modules`` -> ``motion_tracking_rl.networks`` package and
  every moved submodule. This is actually registered by importing the
  ``motion_tracking_rl.modules`` shim package (its ``__init__`` aliases each moved
  submodule into ``sys.modules``). Importing this module ensures that has run.
* The canonical ``MapTransformer``/``MapCNN`` (extracted to
  ``networks/layers/map_transformer.py``) are re-exported here for convenience.
  Note: the three *divergent* MapTransformer variants (vision_sonic,
  residual_vision_sonic, deploy_residual_vision_sonic) are NOT aliased to the
  canonical class on purpose — their state_dict layouts differ and stay bound to
  their own modules.
"""

from __future__ import annotations

# Importing the shim package runs its sys.modules aliasing (modules.* -> networks.*).
import motion_tracking_rl.modules as _legacy_modules_shim  # noqa: F401

from motion_tracking_rl.networks.layers.map_transformer import MapCNN, MapTransformer

__all__ = ["MapCNN", "MapTransformer"]
