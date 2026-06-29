"""Tracking-task MDP modules.

This package exposes the full Isaac Lab command stack when Isaac Lab is
available, while still allowing pure-torch helpers and tests to import in
lighter environments.
"""

try:
    from isaaclab.envs.mdp import *  # noqa: F401, F403
    _HAS_ISAACLAB = True
except ModuleNotFoundError as exc:
    # Degrade gracefully when isaaclab OR its Isaac-Sim deps (omni/pxr) are absent, so
    # `import pmt_tasks.mdp` stays USD-free on the mjlab backend (MJLAB_BACKEND_PLAN.md
    # Phase B). Note bare "omni"/"pxr" have no dot, so match the root too.
    _name = exc.name or ""
    _root = _name.split(".")[0]
    if _root not in ("isaaclab", "omni", "pxr"):
        raise
    _HAS_ISAACLAB = False

# NOTE (PMT port): the minco_* / mppi_* planner modules and their command
# classes (MincoMotionCommand, MppiMotionCommand) are intentionally NOT ported
# into PMT to reduce surface. The duck-typed minco/mppi observation helpers in
# observations.py remain (they only call command methods, no module imports).

if _HAS_ISAACLAB:
    # The command stack lives in the mdp/commands/ subpackage (PMT reorg 2026-06-25);
    # its __init__ re-exports the full command surface, so `from .commands import *`
    # exposes the same names the old flat layout did.
    from .commands import *  # noqa: F401, F403
    from .events import *  # noqa: F401, F403
    from .observations import *  # noqa: F401, F403
    from .rewards import *  # noqa: F401, F403
    from .terminations import *  # noqa: F401, F403
