"""BFM-Zero (FB-CPR-Aux) port onto PMT IsaacLab G1 tasks.

This package adapts the off-policy, latent-conditioned BFM-Zero algorithm (Forward-Backward
representation + Critic-Preferred Rollouts + auxiliary safety critic) so it can train inside
the IsaacLab ``ManagerBasedRLEnv`` used by PMT while the original transformer PPO
tasks remain untouched.

The BFM-Zero networks/agent are **vendored** inside this package under ``._vendor`` (the minimal
FB-CPR-Aux import-closure of the upstream ``humanoidverse`` package). PMT therefore needs nothing
from any external ``BFM-Zero`` repo — see ``._vendor.__init__`` for provenance and license.

Sub-modules
-----------
- ``obs_math``      : pure-torch builders for the BFM observation dict (no Isaac dependency)
- ``expert_streaming`` : expert buffer built from the streaming motion store (no Isaac dependency)
- ``aux_rewards``   : pure-torch aux-reward formulas (no Isaac dependency)
- ``config``        : builds the ``FBcprAuxAgentConfig`` for this task
- ``vec_env``       : IsaacLab ManagerBasedRLEnv -> BFM dict-obs Gym vector-env adapter
- ``runner``        : off-policy training loop mirroring ``humanoidverse.train.Workspace``
"""

# ``ensure_bfm_zero_on_path`` is retained as a backward-compatible no-op (the FB-CPR-Aux code is
# vendored under ``._vendor``); re-exported so existing imports keep resolving.
from .config import ensure_bfm_zero_on_path  # noqa: F401
