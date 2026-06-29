# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict

import motion_tracking_rl
from motion_tracking_rl.algorithms import PPO, BPO, FPOPlus, ADDPPO
from motion_tracking_rl.env import VecEnv
from motion_tracking_rl.networks import (
    ActorCritic,
    ActorCriticRecurrent,
    DiffusionActorCritic,
    OfficialSonicActorCritic,
    SonicActorCritic,
    TransformerActorCritic,
    VisionTransformerActorCritic,
    VisionSonicActorCritic,
    ResidualVisionSonicActorCritic,
    ModularVisionSonicActorCritic,
    DeployResidualVisionSonicActorCritic,
    VisionAblationActorCritic,
    VisionAblationRecurrentActorCritic,
    PerceptiveMotionTokenTracker,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from motion_tracking_rl.registry import NETWORKS, ALGORITHMS, register_runner


def _resolve_class(name: str, registry: dict, _globals: dict):
    """Resolve a config class_name to a class: prefer the decorator registry,
    fall back to module-scope eval() for classes not yet @register_*'d.

    This completes the eval()->registry swap (plan §4): registered networks/
    algorithms route through the registry; anything not yet decorated still
    resolves via the imported module scope (zero behavior change)."""
    if name in registry:
        return registry[name]
    return eval(name, _globals)
from motion_tracking_rl.utils import resolve_obs_groups, store_code_state


@register_runner("on_policy")
@register_runner("OnPolicyRunner")
class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        # Streaming motion working-set swap cadence (0 = disabled).
        self.motion_resample_frequency = int(self.cfg.get("motion_resample_frequency", 0))
        self._streaming_command = (
            self._find_streaming_command() if self.motion_resample_frequency > 0 else None
        )
        # Phase 4: command that wants per-step policy uncertainty (action-std), or None.
        # Resolved once; the rollout loop pushes action-std only if a command exposes
        # receive_policy_uncertainty (the adaptive-sampling command with uncertainty on).
        self._uncertainty_command = self._find_uncertainty_command()

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)
        self.policy_metadata = self._get_policy_metadata()

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [motion_tracking_rl.__file__]

    def _find_streaming_command(self):
        """Locate a command term exposing resample_working_set() (streaming), or None.

        Duck-typed so the runner stays decoupled from the streaming command's
        concrete class.
        """
        env = getattr(self.env, "unwrapped", self.env)
        cmd_mgr = getattr(env, "command_manager", None)
        if cmd_mgr is None:
            return None
        try:
            term_names = list(cmd_mgr.active_terms)
        except Exception:
            term_names = ["motion"]
        for name in term_names:
            try:
                term = cmd_mgr.get_term(name)
            except Exception:
                continue
            if hasattr(term, "resample_working_set") and hasattr(term, "forced_reset_all"):
                print(f"[Streaming] Runner will swap working set every "
                      f"{self.motion_resample_frequency} iters via command '{name}'")
                return term
        print("[Streaming] motion_resample_frequency set but no streaming command found; "
              "swapping disabled")
        return None

    def _find_uncertainty_command(self):
        """Locate a command exposing receive_policy_uncertainty() (Phase 4), or None.

        Duck-typed and fully optional: if no command wants uncertainty (the common
        case), the rollout loop skips the push entirely (zero overhead).
        """
        env = getattr(self.env, "unwrapped", self.env)
        cmd_mgr = getattr(env, "command_manager", None)
        if cmd_mgr is None:
            return None
        try:
            term_names = list(cmd_mgr.active_terms)
        except Exception:
            term_names = ["motion"]
        for name in term_names:
            try:
                term = cmd_mgr.get_term(name)
            except Exception:
                continue
            # Only treat it as active if the hook exists AND the command actually uses
            # uncertainty (the command sets _uses_uncertainty from its cfg).
            if hasattr(term, "receive_policy_uncertainty") and getattr(term, "_uses_uncertainty", False):
                print(f"[AdaptiveSampling] Runner will push policy uncertainty to command '{name}'")
                return term
        return None

    def _push_policy_uncertainty(self) -> None:
        """Push the actor's current per-env action-std to the uncertainty command.

        Called each rollout step AFTER alg.act() (so the distribution is populated) and
        BEFORE env.step(). Guarded: any failure to read action_std is swallowed so a
        policy without that attribute never breaks training.
        """
        cmd = self._uncertainty_command
        if cmd is None:
            return
        try:
            std = self.alg.policy.action_std  # [num_envs, act_dim]
        except Exception:
            return
        if std is None:
            return
        cmd.receive_policy_uncertainty(std)

    def _get_streaming_curriculum(self):
        """Return the streaming command's GlobalCurriculum, or None.

        Independent of ``motion_resample_frequency`` so a streaming run that
        never swaps still checkpoints its curriculum. Looks up the command
        directly (the cached ``_streaming_command`` is only set when swapping is
        enabled).
        """
        # getattr-guarded: runner subclasses (e.g. DistillationRunner) inherit save()
        # but do not set _streaming_command in their __init__.
        cmd = getattr(self, "_streaming_command", None)
        if cmd is None:
            env = getattr(self.env, "unwrapped", self.env)
            cmd_mgr = getattr(env, "command_manager", None)
            if cmd_mgr is None:
                return None
            try:
                term_names = list(cmd_mgr.active_terms)
            except Exception:
                term_names = ["motion"]
            for name in term_names:
                try:
                    term = cmd_mgr.get_term(name)
                except Exception:
                    continue
                if hasattr(term, "global_curriculum"):
                    cmd = term
                    break
        return getattr(cmd, "global_curriculum", None) if cmd is not None else None

    def _swap_streaming_working_set(self) -> None:
        """Swap the streaming command's resident working set at a rollout boundary.

        Single-process: fold + sample + load in one call.
        Distributed: fold locally, all-reduce the folded curriculum so every rank
        agrees, then rank 0 samples and broadcasts the chosen global ids so all
        ranks load the IDENTICAL working set (independent multinomial draws would
        diverge even with equal probabilities). Finally every env is fully reset.
        """
        cmd = self._streaming_command
        if not self.is_distributed:
            cmd.resample_working_set()
            cmd.forced_reset_all()
            return

        # Distributed swap — keep every rank's curriculum and resident set identical.
        # 1) Pool RAW per-window outcome counts across ranks (SUM, no divide) BEFORE
        #    folding, so all ranks fold the same globally-pooled deltas — matching
        #    single-GPU "one big rollout" semantics (each rank only saw its own envs).
        def _all_reduce_sum(t):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

        cmd.sync_curriculum_accumulators(_all_reduce_sum)
        # 2) Fold identical pooled counts on every rank (deterministic, no broadcast
        #    of failed/success needed — they were already identical and the pooled
        #    delta is identical).
        cmd.fold_curriculum()
        # 3) Rank 0 samples; broadcast the chosen ids so all ranks load the SAME set
        #    (independent multinomial draws would diverge even with equal probs).
        ids = cmd.sample_working_set_ids().to(self.device)
        torch.distributed.broadcast(ids, src=0)
        # 4) PREPARE on every rank (atomic — does NOT touch the live set yet), then
        #    collectively agree before committing. This avoids the divergence where
        #    some ranks commit the new set while a failed rank keeps the old one.
        prepared = cmd.prepare_working_set_ids(ids)
        ok_t = torch.tensor([1.0 if prepared is not None else 0.0], device=self.device)
        torch.distributed.all_reduce(ok_t, op=torch.distributed.ReduceOp.MIN)
        if ok_t.item() < 1.0:
            # At least one rank failed to prepare → NO rank commits. Every rank keeps
            # its current resident set, stays consistent, and training continues.
            if self.gpu_global_rank == 0:
                print("[Streaming] a rank failed to prepare new working set; "
                      "all ranks keep current set this swap")
            return
        # 5) All ranks prepared successfully → all commit + reset together.
        cmd.commit_working_set(prepared)
        cmd.forced_reset_all()

    def _launch_background_prepare(self) -> None:
        """P3: fold + sample + decode/build the next working set in a background
        thread (CUDA-free CPU work) so it overlaps the PPO update. The built
        (CPU) PreparedWorkingSet is stashed in self._pending_prepare; the main
        thread commits it (CPU->GPU copy) after the update."""
        import threading

        cmd = self._streaming_command
        cmd.fold_curriculum()
        ids = cmd.sample_working_set_ids()

        self._pending_prepare = None
        self._prepare_error = None

        def _work():
            try:
                # prepare_working_set_ids builds CPU tensors only (CUDA-free).
                self._pending_prepare = cmd.prepare_working_set_ids(ids)
            except Exception as e:  # noqa: BLE001
                self._prepare_error = e

        self._prepare_thread = threading.Thread(target=_work, daemon=True)
        self._prepare_thread.start()

    def _commit_streaming_swap(self) -> None:
        """Install the background-prepared working set (or do a synchronous swap if
        no prepare is pending / it failed), then force-reset all envs."""
        cmd = self._streaming_command
        if self.is_distributed:
            # Distributed path stays synchronous (needs all-reduce + broadcast).
            self._swap_streaming_working_set()
            return

        thread = getattr(self, "_prepare_thread", None)
        if thread is not None:
            thread.join()
            self._prepare_thread = None
        prepared = getattr(self, "_pending_prepare", None)
        err = getattr(self, "_prepare_error", None)
        committed = False
        if prepared is not None:
            committed = cmd.commit_working_set(prepared)
        if not committed:
            # Background prepare missing/failed — fall back to a synchronous swap so
            # training still rotates the working set (old set stays live if that
            # also fails).
            if err is not None:
                print(f"[Streaming] background prepare failed ({err}); synchronous swap")
            cmd.resample_working_set()
        self._pending_prepare = None
        cmd.forced_reset_all()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            self.alg.set_learning_iteration(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Phase 4 (optional): push actor action-std to the adaptive-sampling
                    # command. No-op when no command requested uncertainty.
                    if self._uncertainty_command is not None:
                        self._push_policy_uncertainty()
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # P3: if a swap is due this iteration, fold the curriculum, sample the
            # next ids, and launch the CPU decode+build in a BACKGROUND thread so it
            # overlaps the PPO update below. Single-process only (distributed needs a
            # synchronous all-reduce/broadcast, handled in _swap_streaming_working_set).
            swap_due = (
                self._streaming_command is not None
                and (it + 1) % self.motion_resample_frequency == 0
            )
            self._pending_prepare = None
            if swap_due and not self.is_distributed:
                self._launch_background_prepare()

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Streaming motion swap — AFTER the PPO update, at the rollout boundary,
            # so trajectory continuity / bootstrapping is never corrupted. Swaps the
            # resident working set, force-resets all envs (not counted as episode
            # outcomes), reacquires observations, and clears episode accumulators
            # that must not span the artificial truncation.
            if swap_due:
                # env.reset() does inplace writes on sim tensors created during the
                # rollout's inference_mode; those updates are only legal inside an
                # inference_mode context (matching how env.step is invoked).
                _swap_t0 = time.time()
                with torch.inference_mode():
                    self._commit_streaming_swap()
                    obs = self.env.get_observations().to(self.device)
                print(f"[Streaming] iter {it+1}: working-set swap took {time.time() - _swap_t0:.2f}s")
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                if self.alg.rnd:
                    cur_ereward_sum.zero_()
                    cur_ireward_sum.zero_()
                # Reset recurrent policy hidden state for ALL envs: the forced swap
                # reset is an artificial truncation outside env.step, so PPO never
                # sees the dones that would clear recurrent memory. No-op for
                # feed-forward policies.
                if getattr(self.alg.policy, "is_recurrent", False):
                    all_dones = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
                    self.alg.policy.reset(all_dones)

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if hasattr(self.alg, "get_optimizer_lrs"):
            for group_name, lr_value in self.alg.get_optimizer_lrs().items():
                self.writer.add_scalar(f"Loss/learning_rate_{group_name}", lr_value, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # Print rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # Print episode information
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None) -> None:
        policy_metadata = self._get_policy_metadata()
        # If the algorithm has an EMA (e.g. FPOPlus), temporarily swap EMA weights
        # into the live policy so the exported model_state_dict reflects them.
        ema_backup = None
        if hasattr(self.alg, "swap_ema_into_policy"):
            ema_backup = self.alg.swap_ema_into_policy()
        # Save model. Clone tensors so the dict remains correct after we restore
        # live weights below (state_dict() returns references, not copies).
        policy_state_dict = {k: v.detach().clone() for k, v in self.alg.policy.state_dict().items()}
        optimizer = getattr(self.alg, "optimizer", None)
        runner_metadata = {
            "experiment_name": self.cfg.get("experiment_name"),
            "obs_groups": dict(self.cfg.get("obs_groups", {})),
        }
        saved_dict = {
            "model_state_dict": policy_state_dict,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "iter": self.current_learning_iteration,
            "infos": infos,
            "policy_metadata": policy_metadata,
            "runner_metadata": runner_metadata,
        }
        if hasattr(self.alg, "ema_state_dict"):
            ema_state = self.alg.ema_state_dict()
            if ema_state is not None:
                saved_dict["ema_state_dict"] = ema_state
        # Restore live weights now that the state dict has been built.
        if ema_backup is not None and hasattr(self.alg, "restore_from_ema_swap"):
            self.alg.restore_from_ema_swap(ema_backup)
        # Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # Save ADD discriminator if used
        if hasattr(self.alg, "discriminator") and self.alg.discriminator is not None:
            saved_dict["disc_state_dict"] = self.alg.discriminator.state_dict()
            saved_dict["diff_normalizer_state_dict"] = self.alg.diff_normalizer.state_dict()
        # Save streaming global curriculum so resume keeps difficulty progress
        # (otherwise the global sampling distribution silently resets on resume).
        streaming_curriculum = self._get_streaming_curriculum()
        if streaming_curriculum is not None:
            saved_dict["motion_curriculum_state_dict"] = streaming_curriculum.state_dict()
        torch.save(saved_dict, path)

        adapter_state_dict = (
            self.alg.policy.get_adapter_state_dict() if hasattr(self.alg.policy, "get_adapter_state_dict") else None
        )
        if adapter_state_dict:
            root, ext = os.path.splitext(path)
            adapter_path = f"{root}_pma{ext or '.pt'}"
            torch.save(
                {
                    "adapter_state_dict": adapter_state_dict,
                    "iter": self.current_learning_iteration,
                    "infos": infos,
                    "policy_metadata": policy_metadata,
                    "runner_metadata": runner_metadata,
                },
                adapter_path,
            )

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        loaded_policy_metadata = loaded_dict.get("policy_metadata")
        current_policy_metadata = self._get_policy_metadata()
        metadata_requires_transfer = False
        if isinstance(loaded_policy_metadata, dict) and isinstance(current_policy_metadata, dict):
            loaded_policy_metadata = self._normalize_policy_metadata_for_resume(loaded_policy_metadata)
            current_policy_metadata = self._normalize_policy_metadata_for_resume(current_policy_metadata)
            metadata_issues: list[str] = []
            if loaded_policy_metadata.get("policy_class") != current_policy_metadata.get("policy_class"):
                metadata_issues.append(
                    f"policy_class checkpoint={loaded_policy_metadata.get('policy_class')} "
                    f"current={current_policy_metadata.get('policy_class')}"
                )
            loaded_schema_hash = loaded_policy_metadata.get("obs_schema", {}).get("hash")
            current_schema_hash = current_policy_metadata.get("obs_schema", {}).get("hash")
            if loaded_policy_metadata.get("signature") != current_policy_metadata.get("signature"):
                metadata_issues.append("policy_signature differs from the current runner configuration")
            if metadata_issues:
                metadata_requires_transfer = True
                warnings.warn(
                    f"Checkpoint '{path}' metadata does not match the current runner configuration. "
                    "Attempting policy-defined transfer loading instead of resume loading: "
                    + "; ".join(metadata_issues),
                    stacklevel=2,
                )
            if loaded_schema_hash is not None and loaded_schema_hash != current_schema_hash:
                warnings.warn(
                    f"Checkpoint '{path}' was created with a different observation schema hash "
                    f"(checkpoint={loaded_schema_hash}, current={current_schema_hash}). "
                    "The policy signature still matches, so loading will continue.",
                    stacklevel=2,
                )
        elif current_policy_metadata is not None:
            warnings.warn(
                f"Checkpoint '{path}' has no policy_metadata. Resume compatibility can not be validated early.",
                stacklevel=2,
            )
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if not resumed_training:
            metadata_requires_transfer = True
            warnings.warn(
                f"Checkpoint '{path}' was loaded as a transfer-only state_dict. "
                "Optimizer state and iteration will not be restored.",
                stacklevel=2,
            )
        # Load EMA shadow if present (FPOPlus).
        if hasattr(self.alg, "load_ema_state_dict") and "ema_state_dict" in loaded_dict:
            self.alg.load_ema_state_dict(loaded_dict["ema_state_dict"])
        # Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load ADD discriminator if used
        if (
            hasattr(self.alg, "discriminator")
            and self.alg.discriminator is not None
            and "disc_state_dict" in loaded_dict
        ):
            self.alg.discriminator.load_state_dict(loaded_dict["disc_state_dict"])
            self.alg.diff_normalizer.load_state_dict(loaded_dict["diff_normalizer_state_dict"])
        # Restore streaming global curriculum so resume keeps difficulty progress.
        if "motion_curriculum_state_dict" in loaded_dict:
            curriculum = self._get_streaming_curriculum()
            if curriculum is not None:
                try:
                    curriculum.load_state_dict(loaded_dict["motion_curriculum_state_dict"])
                except ValueError as e:
                    warnings.warn(f"[Streaming] curriculum state not restored: {e}", stacklevel=2)
        # Load optimizer if used
        if metadata_requires_transfer and resumed_training:
            warnings.warn(
                f"Checkpoint '{path}' weights loaded successfully, but metadata differences require transfer-only "
                "semantics. Optimizer state and iteration will not be restored.",
                stacklevel=2,
            )
        if load_optimizer and resumed_training and not metadata_requires_transfer:
            # Algorithm optimizer
            optimizer = getattr(self.alg, "optimizer", None)
            optimizer_state = loaded_dict.get("optimizer_state_dict")
            if optimizer is not None and optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load optimizer state from checkpoint '{path}': {exc}. "
                        "Continuing with freshly initialized optimizer.",
                        stacklevel=2,
                    )
            elif optimizer is not None:
                warnings.warn(
                    f"Checkpoint '{path}' has no optimizer_state_dict. "
                    "Continuing with freshly initialized optimizer.",
                    stacklevel=2,
                )
            # RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                if "rnd_optimizer_state_dict" in loaded_dict:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
                else:
                    warnings.warn(
                        f"Checkpoint '{path}' has no rnd_optimizer_state_dict. "
                        "Continuing with freshly initialized RND optimizer.",
                        stacklevel=2,
                    )
        # Load current learning iteration
        if resumed_training and not metadata_requires_transfer:
            self.current_learning_iteration = int(loaded_dict.get("iter", 0))
        reset_action_std = self.cfg.get("reset_action_std_on_load")
        if reset_action_std is not None and float(reset_action_std) > 0.0:
            self._reset_policy_action_std(float(reset_action_std))
        return loaded_dict.get("infos", {})

    def _reset_policy_action_std(self, target_std: float) -> None:
        """Reset common action-std parameterizations after checkpoint loading."""
        if target_std <= 0.0:
            raise ValueError(f"reset_action_std_on_load must be positive, got {target_std}.")

        target = torch.tensor(float(target_std), device=self.device)
        reset_terms: list[str] = []

        def inv_softplus(value: torch.Tensor) -> torch.Tensor:
            return value + torch.log(-torch.expm1(-value))

        def reset_module(module: torch.nn.Module, prefix: str = "") -> None:
            raw_std_param = getattr(module, "raw_std_param", None)
            if raw_std_param is not None:
                min_std = float(getattr(module, "min_std", 0.0))
                raw_target = torch.clamp(target - min_std, min=1.0e-6)
                raw_std_param.data.copy_(torch.full_like(raw_std_param.data, inv_softplus(raw_target).item()))
                reset_terms.append(prefix + "raw_std_param")

            log_std_param = getattr(module, "log_std_param", None)
            if log_std_param is not None:
                log_std_param.data.copy_(torch.full_like(log_std_param.data, torch.log(target).item()))
                reset_terms.append(prefix + "log_std_param")

            std = getattr(module, "std", None)
            if isinstance(std, torch.nn.Parameter):
                std.data.copy_(torch.full_like(std.data, target.item()))
                reset_terms.append(prefix + "std")

            log_std = getattr(module, "log_std", None)
            if isinstance(log_std, torch.nn.Parameter):
                log_std.data.copy_(torch.full_like(log_std.data, torch.log(target).item()))
                reset_terms.append(prefix + "log_std")

        reset_module(self.alg.policy)
        student = getattr(self.alg.policy, "student", None)
        if isinstance(student, torch.nn.Module):
            reset_module(student, "student.")

        if self.gpu_global_rank == 0:
            if reset_terms:
                print(
                    f"[OnPolicyRunner] Reset loaded action std to {target_std:g} for: "
                    + ", ".join(sorted(set(reset_terms)))
                )
            else:
                warnings.warn(
                    "reset_action_std_on_load was set, but no known action std parameters were found on the policy.",
                    stacklevel=2,
                )

    def _get_policy_metadata(self) -> dict | None:
        if hasattr(self.alg.policy, "get_checkpoint_metadata"):
            metadata = self.alg.policy.get_checkpoint_metadata()
            if isinstance(metadata, dict):
                return metadata
        return None

    @staticmethod
    def _normalize_policy_metadata_for_resume(metadata: dict) -> dict:
        """Fill legacy default signature fields before strict resume comparison."""
        normalized = dict(metadata)
        signature = dict(normalized.get("signature", {}))
        policy_family = normalized.get("policy_family")
        policy_class = normalized.get("policy_class")
        if policy_family == "vision_transformer_actor_critic" or policy_class == "VisionTransformerActorCritic":
            signature.setdefault("use_action_residual", True)
            signature.setdefault("use_map_proprio_cross_attention", False)
        normalized["signature"] = signature
        return normalized

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # ADD discriminator
        if hasattr(self.alg, "discriminator") and self.alg.discriminator is not None:
            self.alg.discriminator.train()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # ADD discriminator
        if hasattr(self.alg, "discriminator") and self.alg.discriminator is not None:
            self.alg.discriminator.eval()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Device assignment: this cluster's Ray launcher (md_ai_kit) exposes ALL GPUs to
        # every worker (CUDA_VISIBLE_DEVICES=0,1,... ; torch.cuda.device_count()==world_size)
        # and sets LOCAL_RANK to the TRUE per-node GPU index. So each worker must run on
        # cuda:{LOCAL_RANK} — and the caller (train_pmt.py) already builds the env + agent on
        # cuda:{local_rank}. We therefore enforce the standard torchrun-style invariant
        # device==cuda:{local_rank}; this also catches the previous device-mismatch bug where
        # the env lived on cuda:1 while the runner pinned the default device to cuda:0.
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed — UNLESS the launcher already did. Under Ray
        # (md_ai_kit DataParallelTrainer) the default process group is pre-initialized
        # for the worker group; calling init_process_group again raises
        # "trying to initialize the default process group twice!". Guard for it so PMT
        # works both under torchrun (we init) and under Ray (launcher already inited).
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
            )
        else:
            # Trust the launcher's group; align our rank bookkeeping with it.
            self.gpu_global_rank = torch.distributed.get_rank()
            self.gpu_world_size = torch.distributed.get_world_size()
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = _resolve_class(self.policy_cfg.pop("class_name"), NETWORKS, globals())
        actor_critic: (
            ActorCritic
            | ActorCriticRecurrent
            | OfficialSonicActorCritic
            | SonicActorCritic
            | VisionTransformerActorCritic
            | VisionSonicActorCritic
            | ResidualVisionSonicActorCritic
            | ModularVisionSonicActorCritic
            | DeployResidualVisionSonicActorCritic
            | VisionAblationActorCritic
            | VisionAblationRecurrentActorCritic
            | PerceptiveMotionTokenTracker
        ) = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = _resolve_class(self.alg_cfg.pop("class_name"), ALGORITHMS, globals())
        alg: PPO | BPO | FPOPlus | ADDPPO = alg_class(
            actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self) -> None:
        """Prepare the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune or Tensorboard summary writer, default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from motion_tracking_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
