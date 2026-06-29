"""Off-policy BFM-Zero (FB-CPR-Aux) training runner over the IsaacLab vec-env adapter.

Mirrors ``humanoidverse.train.Workspace.train_online`` but:
  - drives the ``BFMZeroVecEnv`` adapter (Option B) instead of the humanoidverse Isaac wrapper,
  - sources expert mocap from a ``StaticExpertBuffer`` built off the streaming motion store,
  - handles IsaacLab same-step autoreset with a ``~(prev_done | curr_done)`` transition mask so
    no bogus post-reset ``next_obs`` (or done-row aux) is ever stored.

The agent is the unmodified ``FBcprAuxAgent`` (built via ``bfm_zero.config.build_agent_config``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_map

from . import config as bfm_config


def _to_torch_obs(obs_np: dict, device, dtype_lower=True):
    def conv(x):
        t = torch.as_tensor(x, device=device)
        if dtype_lower and t.dtype == torch.float64:
            t = t.float()
        return t
    return {k: conv(v) for k, v in obs_np.items()}


class BFMZeroRunner:
    """Owns the env-step / replay / agent-update loop for BFM-Zero on IsaacLab."""

    def __init__(
        self,
        vec_env,
        expert_buffer,
        *,
        device: str = "cuda",
        seed: int = 0,
        num_seed_steps: int = 10240,
        update_agent_every: int = 1024,
        num_agent_updates: int = 16,
        buffer_size: int = 5_000_000,
        batch_size: int = 1024,
        log_every: int = 2048,
        checkpoint_every: int = 0,
        eval_every: int = 0,
        eval_horizon: int = 250,
        work_dir: str = "logs/bfm_zero/run",
        agent=None,
        agent_cfg=None,
        motion_resample_frequency: int = 0,
        expert_builder=None,
        buffer_device: str | None = None,
    ):
        from ._vendor.agents.buffers.transition import DictBuffer

        self.env = vec_env
        self.expert_buffer = expert_buffer
        self.device = device
        self.num_envs = vec_env.num_envs
        self.num_seed_steps = num_seed_steps
        self.update_agent_every = update_agent_every
        self.num_agent_updates = num_agent_updates
        self.batch_size = batch_size
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.eval_horizon = eval_horizon
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.motion_resample_frequency = motion_resample_frequency

        # TensorBoard logging. Events are written under ``<work_dir>/tb`` so they can be viewed with
        # ``tensorboard --logdir <work_dir>/tb``. Falls back to stdout-only if TB is unavailable.
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.work_dir / "tb"))
        except Exception as e:  # noqa: BLE001
            print(f"[BFMZeroRunner] TensorBoard unavailable ({e}); logging to stdout only")
        # Optional callable returning a freshly-built expert buffer (used to refresh after swaps).
        self._expert_builder = expert_builder

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build (or accept) the agent config. The pydantic config is frozen, so batch_size and the
        # rollout-trajectory-length clamp must be applied at BUILD time, not mutated afterwards.
        if agent_cfg is None:
            feasible_len = self._feasible_rollout_length(expert_buffer)
            agent_cfg = bfm_config.build_agent_config(
                device=device, batch_size=batch_size, rollout_expert_trajectories_length=feasible_len
            )
        else:
            self._warn_if_batch_or_rollout_mismatch(agent_cfg, batch_size, expert_buffer)
        self.agent_cfg = agent_cfg
        self.agent = agent or self.agent_cfg.build(
            obs_space=self._agent_obs_space(), action_dim=vec_env.single_action_space.shape[0]
        )
        self.agent._model.train()

        self.replay_buffer = {
            "train": DictBuffer(capacity=buffer_size, device=buffer_device or device),
            "expert_slicer": expert_buffer,
        }

    def _feasible_rollout_length(self, expert_buffer, desired: int = 250) -> int:
        """Largest expert rollout-trajectory length the resident clips can serve.

        ``maybe_update_rollout_context`` samples expert sub-trajectories of this length; a clip
        needs >= length+1 frames. Clamp to ``max_clip_len - 1`` if the longest resident clip is
        shorter (otherwise the first expert rollout would raise).
        """
        max_len = getattr(expert_buffer, "_max_len", None)
        if max_len is None:
            return desired
        feasible = max(1, min(desired, int(max_len) - 1))
        if feasible < desired:
            print(
                f"[BFMZeroRunner] clamping rollout_expert_trajectories_length {desired} -> {feasible} "
                f"(longest resident clip = {max_len} frames)"
            )
        return feasible

    def _warn_if_batch_or_rollout_mismatch(self, agent_cfg, batch_size, expert_buffer):
        train = agent_cfg.train
        if train.batch_size != batch_size:
            print(
                f"[BFMZeroRunner] WARNING: runner batch_size={batch_size} != agent_cfg.train.batch_size="
                f"{train.batch_size}; the agent uses its own cfg value."
            )
        max_len = getattr(expert_buffer, "_max_len", None)
        if (
            getattr(train, "rollout_expert_trajectories", False)
            and max_len is not None
            and train.rollout_expert_trajectories_length > int(max_len) - 1
        ):
            print(
                f"[BFMZeroRunner] WARNING: rollout_expert_trajectories_length="
                f"{train.rollout_expert_trajectories_length} exceeds feasible {int(max_len) - 1} "
                f"(longest resident clip = {max_len}); expert rollouts may raise."
            )

    def _agent_obs_space(self):
        """Agent obs space = adapter obs minus 'time'."""
        import gymnasium

        spaces = dict(self.env.single_observation_space.spaces)
        spaces.pop("time", None)
        return gymnasium.spaces.Dict(spaces)

    def train(self, total_env_steps: int):
        env = self.env
        device = self.device
        n = self.num_envs

        obs_np, _ = env.reset()
        prev_done = np.zeros(n, dtype=bool)
        context = None
        total_metrics = None
        num_metrics_updates = 0
        start_time = time.time()

        # Counters are in ENV-TRANSITION units (like humanoidverse Workspace ``t``): each loop
        # iteration advances by ``n`` transitions. Thresholds (num_seed_steps, update_agent_every,
        # log_every, swap cadence) are therefore in transition units too.
        last_update_t = 0
        last_log_t = 0
        last_swap_t = 0
        last_ckpt_t = 0
        last_eval_t = 0
        t = 0
        while t < total_env_steps:
            with torch.no_grad():
                obs = _to_torch_obs(obs_np, device)
                # Keep ``time`` as [N, 1] (matching humanoidverse) so the z-reset mask broadcasts
                # against z of shape [N, z_dim].
                step_count = obs.pop("time").long()

                context = self.agent.maybe_update_rollout_context(
                    z=context, step_count=step_count, replay_buffer=self.replay_buffer
                )
                if t < self.num_seed_steps:
                    action = np.stack([env.single_action_space.sample() for _ in range(n)]).astype(np.float32)
                    action_t = torch.as_tensor(action, device=device)
                else:
                    action_t = self.agent.act(obs=obs, z=context, mean=False)
                    if not torch.is_tensor(action_t):
                        action_t = torch.as_tensor(action_t, device=device)

            next_obs_np, reward_np, terminated_np, truncated_np, info = env.step(action_t)
            curr_done = np.logical_or(terminated_np, truncated_np)

            # Store only transitions whose next_obs is a TRUE successor: skip envs that were done
            # in the previous step (their obs is a reset state) AND envs done this step (their
            # next_obs is the post-reset state under IsaacLab same-step autoreset).
            valid = ~np.logical_or(prev_done, curr_done)
            if valid.any():
                idx = valid
                real_next = _to_torch_obs(next_obs_np, device)
                real_next.pop("time", None)
                data = {
                    "observation": {k: obs[k][idx] for k in obs},
                    "action": action_t[idx],
                    "reward": torch.as_tensor(reward_np[idx], device=device).reshape(-1, 1),
                    "z": context[idx],
                    "next": {
                        "observation": {k: real_next[k][idx] for k in real_next},
                        "terminated": torch.as_tensor(terminated_np[idx], device=device).reshape(-1, 1),
                        "truncated": torch.as_tensor(truncated_np[idx], device=device).reshape(-1, 1),
                    },
                    "aux_rewards": {
                        k: torch.as_tensor(v[idx], device=device).reshape(-1, 1)
                        for k, v in info["aux_rewards"].items()
                    },
                }
                self.replay_buffer["train"].extend(data)

            # Agent updates (every ``update_agent_every`` transitions, after the seed phase).
            if len(self.replay_buffer["train"]) > 0 and t >= self.num_seed_steps and (t - last_update_t) >= self.update_agent_every:
                last_update_t = t
                for _ in range(self.num_agent_updates):
                    metrics = self.agent.update(self.replay_buffer, t)
                    if total_metrics is None:
                        total_metrics = {k: metrics[k].float().clone() for k in metrics}
                        num_metrics_updates = 1
                    else:
                        for k in metrics:
                            total_metrics[k] = total_metrics[k] + metrics[k].float()
                        num_metrics_updates += 1

            # Streaming working-set swap (transition-count cadence).
            if self.motion_resample_frequency > 0 and t > 0 and (t - last_swap_t) >= (self.motion_resample_frequency * self.update_agent_every):
                last_swap_t = t
                swapped = self._maybe_swap_working_set()
                if swapped:
                    # After a swap the resident clips changed: rebuild the expert buffer and reset
                    # the rollout state (the command force-reset all envs).
                    self._rebuild_expert_buffer()
                    obs_np, _ = env.reset()
                    prev_done = np.zeros(n, dtype=bool)
                    context = None
                    t += n
                    continue

            # Logging.
            if total_metrics is not None and (t - last_log_t) >= self.log_every:
                last_log_t = t
                m = {k: round((total_metrics[k] / max(num_metrics_updates, 1)).mean().item(), 6) for k in sorted(total_metrics)}
                fps = round((self.log_every) / (time.time() - start_time + 1e-9), 1)
                print(f"[BFMZeroRunner t={t}] env_steps={t} fps={fps} " + " ".join(f"{k}={v}" for k, v in m.items()))
                if self._tb is not None:
                    # Group metrics so TensorBoard nests them sensibly (aux_rew/* already grouped).
                    for k, v in m.items():
                        tag = k if "/" in k else f"train/{k}"
                        self._tb.add_scalar(tag, v, t)
                    self._tb.add_scalar("perf/fps", fps, t)
                    self._tb.flush()
                total_metrics = None
                num_metrics_updates = 0
                start_time = time.time()

            # Periodic checkpoint (transition-count cadence). 0 disables (only final save).
            if self.checkpoint_every > 0 and t > 0 and (t - last_ckpt_t) >= self.checkpoint_every:
                last_ckpt_t = t
                self.save("checkpoint")

            # Periodic zero-shot tracking eval. Runs in eval mode, resets the env, and does NOT
            # touch the replay buffer. We restore the rollout state (fresh reset) afterwards.
            if self.eval_every > 0 and t > 0 and (t - last_eval_t) >= self.eval_every:
                last_eval_t = t
                self._run_eval(t)
                obs_np, _ = env.reset()
                prev_done = np.zeros(n, dtype=bool)
                context = None
                t += n
                continue

            obs_np = next_obs_np
            prev_done = curr_done
            t += n

        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        return self.agent

    def _run_eval(self, t: int):
        """Run the zero-shot tracking eval and log/print its metrics."""
        try:
            from . import tracking_eval
        except ImportError:
            from . import tracking_eval  # type: ignore
        try:
            metrics = tracking_eval.run_tracking_eval(
                self.env, self.agent, horizon=self.eval_horizon, device=self.device
            )
        except Exception as e:  # noqa: BLE001
            print(f"[BFMZeroRunner] tracking eval skipped: {e}")
            return
        msg = " ".join(f"{k}={round(v, 5)}" for k, v in metrics.items())
        print(f"[BFMZeroRunner EVAL t={t}] {msg}")
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, t)
            self._tb.flush()

    def _maybe_swap_working_set(self) -> bool:
        cmd = getattr(self.env, "motion_command", None)
        if cmd is None:
            return False
        try:
            ids = cmd.sample_working_set_ids()
            prepared = cmd.prepare_working_set_ids(ids)
            cmd.commit_working_set(prepared)
            cmd.forced_reset_all()
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[BFMZeroRunner] working-set swap skipped: {e}")
            return False

    def _rebuild_expert_buffer(self):
        """Rebuild the expert buffer from the (post-swap) resident motion store, if possible."""
        builder = getattr(self, "_expert_builder", None)
        if builder is None:
            return  # static buffer; keep as-is (acceptable when swaps are disabled)
        try:
            self.expert_buffer = builder()
            self.replay_buffer["expert_slicer"] = self.expert_buffer
            print("[BFMZeroRunner] rebuilt expert buffer after working-set swap")
        except Exception as e:  # noqa: BLE001
            print(f"[BFMZeroRunner] expert buffer rebuild skipped: {e}")

    def save(self, tag: str = "checkpoint"):
        out = self.work_dir / tag
        self.agent.save(str(out))
        print(f"[BFMZeroRunner] saved agent to {out}")
