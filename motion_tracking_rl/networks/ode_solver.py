# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for use in motion_tracking_rl.

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

try:
    from motion_tracking_rl.networks.torchdiffeq.torchdiffeq import odeint
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "torchdiffeq is required for ODESolver. Install it to enable PHC-style solvers."
    ) from exc
  # type: ignore

class ODESolver(torch.nn.Module):
    """Solve ODEs over a time grid using a velocity model (PHC-style wrapper)."""

    def __init__(self) -> None:
        super().__init__()

    def sample(
        self,
        velocity_model: Callable,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        """Solve the ODE with a velocity field model."""

        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        return sol[-1]
