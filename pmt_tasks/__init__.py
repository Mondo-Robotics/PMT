"""PMT task composition layer (plan §2, §3).

SELECT -> DERIVE -> VALIDATE -> emit (plan §3a/§3b/§10). ``build_task_config`` is the
pure (no isaaclab/omni) compose+derive+validate path used by the test suite; the
@configclass-emitting helpers (``build_env_cfg`` / ``build_agent_cfg`` in builder.py)
import Isaac Lab lazily and are consumed by ``registry_gym.register_pmt_tasks`` at
gym-registration time.
"""
from pmt_tasks.builder import build_task_config, load_paths  # noqa: F401
from pmt_tasks.derive import derive_obs_groups, derive_reward_weights  # noqa: F401
