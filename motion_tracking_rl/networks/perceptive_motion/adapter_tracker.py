"""PerceptiveMotionAdapterTracker (split from the 2017-line monolith).

A trainable PMA wrapped around a frozen Perceptive Motion Tracker
(``TransformerActorCritic``). State-dict keys are by attribute-path within this
module, so moving the class here is checkpoint-safe (internal structure unchanged).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.networks.transformer_actor_critic import TransformerActorCritic
from motion_tracking_rl.registry import register_network

from .adapter import PerceptiveMotionAdapter


@register_network("PerceptiveMotionAdapterTracker", compat_name="perceptive_motion_adapter_tracker")
class PerceptiveMotionAdapterTracker(nn.Module):
    """Trainable PMA wrapped around a frozen Perceptive Motion Tracker."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        perceptive_motion_tracker_cfg: dict[str, Any] | None = None,
        perceptive_motion_tracker_ckpt_path: str | None = None,
        percaptive_motion_tracker_cfg: dict[str, Any] | None = None,
        percaptive_motion_tracker_ckpt_path: str | None = None,
        teacher_cfg: dict[str, Any] | None = None,
        teacher_ckpt_path: str | None = None,
        teacher_load_strict: bool = True,
        perceptive_motion_adapter_ckpt_path: str | None = None,
        adapter_load_strict: bool = True,
        adapter_mode: str = "gated_residual",
        adapter_context_set_name: str = "pma_context",
        adapter_hidden_dims: Sequence[int] = (512, 256, 128),
        adapter_activation: str = "elu",
        adapter_use_z_flat: bool = True,
        adapter_delta_scale: float = 1.0,
        adapter_gate_bias: float = -2.0,
        critic_hidden_dims: Sequence[int] = (512, 256),
        critic_set_name: str = "critic",
        tracker_policy_set_name: str = "policy",
        update_tracker_normalization: bool = False,
        require_tracker_checkpoint: bool = True,
        require_teacher_latent_target: bool = True,
        flat_identity_obs_key: str | None = None,
        flat_identity_threshold: float | None = None,
        partial_tracker_load_min_match_fraction: float = 0.9,
        partial_adapter_load_min_match_fraction: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[PerceptiveMotionAdapterTracker] Ignoring unused config keys: {sorted(kwargs.keys())}")

        required_groups = (
            "policy",
            "policy_history",
            "command_window",
            adapter_context_set_name,
            critic_set_name,
            tracker_policy_set_name,
        )
        missing_groups = [name for name in required_groups if name not in obs_groups]
        if missing_groups:
            raise KeyError(f"Missing obs_groups required by PerceptiveMotionAdapterTracker: {missing_groups}")

        self.obs_groups = obs_groups
        self.adapter_mode = str(adapter_mode)
        self.adapter_context_set_name = adapter_context_set_name
        self.adapter_hidden_dims = tuple(adapter_hidden_dims)
        self.adapter_activation = str(adapter_activation)
        self.adapter_use_z_flat = bool(adapter_use_z_flat)
        self.adapter_delta_scale = float(adapter_delta_scale)
        self.adapter_gate_bias = float(adapter_gate_bias)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.critic_set_name = str(critic_set_name)
        self.tracker_policy_set_name = str(tracker_policy_set_name)
        self.update_tracker_normalization = bool(update_tracker_normalization)
        self.require_tracker_checkpoint = bool(require_tracker_checkpoint)
        self.require_teacher_latent_target = bool(require_teacher_latent_target)
        self.flat_identity_obs_key = flat_identity_obs_key
        self.flat_identity_threshold = flat_identity_threshold
        self.perceptive_motion_adapter_ckpt_path = perceptive_motion_adapter_ckpt_path
        self.partial_tracker_load_min_match_fraction = float(partial_tracker_load_min_match_fraction)
        self.partial_adapter_load_min_match_fraction = float(partial_adapter_load_min_match_fraction)
        self._last_bridge_debug: dict[str, float] = {}
        self._prev_delta_z: torch.Tensor | None = None
        self.foot_traj_target_obs_key = None

        tracker_cfg = dict(perceptive_motion_tracker_cfg or percaptive_motion_tracker_cfg or teacher_cfg or {})
        ckpt_path = perceptive_motion_tracker_ckpt_path or percaptive_motion_tracker_ckpt_path or teacher_ckpt_path
        if self.require_tracker_checkpoint and not ckpt_path:
            raise ValueError(
                "PerceptiveMotionAdapterTracker requires a frozen Perceptive Motion Tracker checkpoint. "
                "Set perceptive_motion_tracker_ckpt_path or PERCEPTIVE_MOTION_TRACKER_CKPT; pass "
                "require_tracker_checkpoint=False only for synthetic tests."
            )
        self.teacher_ckpt_path = ckpt_path
        self.perceptive_motion_tracker_ckpt_path = ckpt_path
        self.percaptive_motion_tracker_ckpt_path = ckpt_path
        self.loaded_teacher = False
        if self.require_teacher_latent_target:
            missing_teacher_groups = [
                name for name in ("teacher_policy_history", "teacher_command_window") if name not in obs_groups
            ]
            if missing_teacher_groups:
                raise KeyError(
                    "PMA training requires teacher latent target groups, missing: "
                    f"{missing_teacher_groups}. Set require_teacher_latent_target=False only for identity/no-teacher ablations."
                )

        tracker_obs_groups = {
            "policy": list(obs_groups[self.tracker_policy_set_name]),
            "policy_history": list(obs_groups["policy_history"]),
            "command_window": list(obs_groups["command_window"]),
            "critic": list(obs_groups.get("critic", obs_groups[self.tracker_policy_set_name])),
        }
        for group_name in (
            "vel_gt",
            "anchor_gt",
            "anchor_estimator",
            "teacher",
            "teacher_policy_history",
            "teacher_command_window",
            "teacher_anchor_estimator",
        ):
            if group_name in obs_groups:
                tracker_obs_groups[group_name] = list(obs_groups[group_name])

        self.percaptive_motion_tracker = TransformerActorCritic(
            obs=obs,
            obs_groups=tracker_obs_groups,
            num_actions=num_actions,
            **tracker_cfg,
        )

        if ckpt_path:
            self._load_tracker_checkpoint(ckpt_path, strict=teacher_load_strict)
        self._freeze_tracker()

        latent_dim = int(getattr(self.percaptive_motion_tracker, "n_embd", tracker_cfg.get("n_embd", 128)))
        context_dim = self._obs_group_dim(obs, adapter_context_set_name)
        adapter_input_dim = context_dim + (latent_dim if self.adapter_use_z_flat else 0)

        self.perceptive_motion_adapter = PerceptiveMotionAdapter(
            input_dim=adapter_input_dim,
            latent_dim=latent_dim,
            hidden_dims=self.adapter_hidden_dims,
            activation=self.adapter_activation,
            mode=self.adapter_mode,
            delta_scale=self.adapter_delta_scale,
            gate_bias=self.adapter_gate_bias,
        )
        critic_dim = self._obs_group_dim(obs, self.critic_set_name)
        self.pma_critic = MLP(
            input_dim=critic_dim,
            output_dim=1,
            hidden_dims=list(self.critic_hidden_dims),
            activation=self.adapter_activation,
        )
        if perceptive_motion_adapter_ckpt_path:
            self._load_adapter_checkpoint(perceptive_motion_adapter_ckpt_path, strict=adapter_load_strict)

    def forward(self) -> NoReturn:
        raise NotImplementedError("PerceptiveMotionAdapterTracker is driven through act/evaluate APIs.")

    @property
    def perceptive_motion_tracker(self) -> TransformerActorCritic:
        return self.percaptive_motion_tracker

    @property
    def teacher(self) -> TransformerActorCritic:
        return self.percaptive_motion_tracker

    @property
    def student(self) -> PerceptiveMotionAdapter:
        return self.perceptive_motion_adapter

    @property
    def action_mean(self) -> torch.Tensor:
        return self.percaptive_motion_tracker.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.percaptive_motion_tracker.action_std

    @property
    def entropy(self) -> torch.Tensor:
        return self.percaptive_motion_tracker.entropy

    def _freeze_tracker(self) -> None:
        self.percaptive_motion_tracker.eval()
        for param in self.percaptive_motion_tracker.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _resolve_checkpoint_path(path: str | Path) -> Path:
        ckpt_path = Path(path).expanduser()
        if ckpt_path.is_file():
            return ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Perceptive Motion Tracker checkpoint not found: {ckpt_path}")
        candidates = list(ckpt_path.glob("model_*.pt"))
        if not candidates:
            candidates = list(ckpt_path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt checkpoint found under: {ckpt_path}")

        def sort_key(candidate: Path) -> tuple[int, float]:
            match = re.search(r"model_(\d+)\.pt$", candidate.name)
            return (int(match.group(1)) if match else -1, candidate.stat().st_mtime)

        return max(candidates, key=sort_key)

    @staticmethod
    def _strip_prefix_state(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        if not prefix:
            return dict(state_dict)
        return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}

    @staticmethod
    def _select_checkpoint_state_dict(checkpoint: Any, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            return checkpoint
        for key in keys:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        return checkpoint.get("model_state_dict", checkpoint)

    def _load_tracker_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool) -> None:
        candidate_prefixes = (
            "",
            "percaptive_motion_tracker.",
            "perceptive_motion_tracker.",
            "motion_tracker.",
            "teacher.",
            "student.",
            "actor_critic.",
            "policy.",
            "module.",
            "actor.",
        )
        tracker_state = self.percaptive_motion_tracker.state_dict()
        last_error: Exception | None = None
        best_match: tuple[str, dict[str, torch.Tensor], int] | None = None
        for prefix in candidate_prefixes:
            candidate = self._strip_prefix_state(state_dict, prefix) if prefix else state_dict
            if prefix and not candidate:
                continue
            if strict:
                try:
                    self.percaptive_motion_tracker.load_state_dict(candidate, strict=True)
                    if prefix:
                        print(
                            "[PerceptiveMotionAdapterTracker] Loaded Perceptive Motion Tracker "
                            f"state with prefix '{prefix}'"
                        )
                    return
                except RuntimeError as exc:
                    if last_error is None:
                        last_error = exc
                    continue

            filtered = {
                key: value
                for key, value in candidate.items()
                if key in tracker_state and tracker_state[key].shape == value.shape
            }
            if filtered:
                score = len(filtered)
                if best_match is None or score > best_match[2]:
                    best_match = (prefix, filtered, score)

        if best_match is not None:
            prefix, filtered, score = best_match
            match_frac = score / max(len(tracker_state), 1)
            if match_frac < self.partial_tracker_load_min_match_fraction:
                raise RuntimeError(
                    "Refusing partial Perceptive Motion Tracker load: "
                    f"matched={score}/{len(tracker_state)} ({match_frac:.1%}) using prefix "
                    f"'{prefix}', below required {self.partial_tracker_load_min_match_fraction:.1%}."
                )
            self.percaptive_motion_tracker.load_state_dict(filtered, strict=False)
            missing = len(tracker_state) - score
            print(
                "[PerceptiveMotionAdapterTracker] Loaded partial Perceptive Motion Tracker "
                f"state: matched={score} missing_or_shape_mismatch={missing}"
            )
            return

        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not match checkpoint tensors to Perceptive Motion Tracker state_dict.")

    def _load_tracker_checkpoint(self, ckpt_path: str | Path, strict: bool = True) -> None:
        resolved_path = self._resolve_checkpoint_path(ckpt_path)
        try:
            checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(resolved_path, map_location="cpu")
        state_dict = self._select_checkpoint_state_dict(checkpoint, ("model_state_dict",))
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint format at {resolved_path}")
        self._load_tracker_state_dict(state_dict, strict=strict)
        self.loaded_teacher = True
        self.teacher_ckpt_path = str(resolved_path)
        self.perceptive_motion_tracker_ckpt_path = str(resolved_path)
        self.percaptive_motion_tracker_ckpt_path = str(resolved_path)
        print(f"[PerceptiveMotionAdapterTracker] Loaded frozen Perceptive Motion Tracker from {resolved_path}")

    def _load_adapter_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool) -> None:
        candidate_prefixes = (
            "",
            "perceptive_motion_adapter.",
            "student.",
            "module.perceptive_motion_adapter.",
            "module.student.",
        )
        adapter_state = self.perceptive_motion_adapter.state_dict()
        last_error: Exception | None = None
        best_match: tuple[str, dict[str, torch.Tensor], int] | None = None
        for prefix in candidate_prefixes:
            candidate = self._strip_prefix_state(state_dict, prefix) if prefix else state_dict
            if prefix and not candidate:
                continue
            if strict:
                try:
                    self.perceptive_motion_adapter.load_state_dict(candidate, strict=True)
                    if prefix:
                        print(
                            "[PerceptiveMotionAdapterTracker] Loaded Perceptive Motion Adapter "
                            f"state with prefix '{prefix}'"
                        )
                    return
                except RuntimeError as exc:
                    if last_error is None:
                        last_error = exc
                    continue

            filtered = {
                key: value
                for key, value in candidate.items()
                if key in adapter_state and adapter_state[key].shape == value.shape
            }
            if filtered:
                score = len(filtered)
                if best_match is None or score > best_match[2]:
                    best_match = (prefix, filtered, score)

        if best_match is not None:
            prefix, filtered, score = best_match
            match_frac = score / max(len(adapter_state), 1)
            if match_frac < self.partial_adapter_load_min_match_fraction:
                raise RuntimeError(
                    "Refusing partial Perceptive Motion Adapter load: "
                    f"matched={score}/{len(adapter_state)} ({match_frac:.1%}) using prefix "
                    f"'{prefix}', below required {self.partial_adapter_load_min_match_fraction:.1%}."
                )
            self.perceptive_motion_adapter.load_state_dict(filtered, strict=False)
            missing = len(adapter_state) - score
            print(
                "[PerceptiveMotionAdapterTracker] Loaded partial Perceptive Motion Adapter state: "
                f"matched={score} missing_or_shape_mismatch={missing}"
            )
            return

        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not match checkpoint tensors to Perceptive Motion Adapter state_dict.")

    def _load_adapter_checkpoint(self, ckpt_path: str | Path, strict: bool = True) -> None:
        resolved_path = self._resolve_checkpoint_path(ckpt_path)
        try:
            checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(resolved_path, map_location="cpu")
        state_dict = self._select_checkpoint_state_dict(
            checkpoint,
            (
                "adapter_state_dict",
                "perceptive_motion_adapter_state_dict",
                "pma_state_dict",
                "model_state_dict",
            ),
        )
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported adapter checkpoint format at {resolved_path}")
        self._load_adapter_state_dict(state_dict, strict=strict)
        self.perceptive_motion_adapter_ckpt_path = str(resolved_path)
        print(f"[PerceptiveMotionAdapterTracker] Loaded Perceptive Motion Adapter from {resolved_path}")

    def get_adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.perceptive_motion_adapter.state_dict().items()
        }

    def _obs_group_dim(self, obs: TensorDict, group_name: str) -> int:
        dim = 0
        for key in self.obs_groups.get(group_name, []):
            if key not in obs.keys():
                raise KeyError(f"Observation key '{key}' from obs_groups['{group_name}'] is missing.")
            dim += int(obs[key].reshape(obs[key].shape[0], -1).shape[-1])
        return dim

    def _get_concat_flat(self, obs: TensorDict, set_name: str) -> torch.Tensor:
        keys = self.obs_groups.get(set_name, [])
        if not keys:
            raise KeyError(f"obs_groups['{set_name}'] is empty or missing.")
        values = []
        for key in keys:
            if key not in obs.keys():
                raise KeyError(f"Observation key '{key}' from obs_groups['{set_name}'] is missing.")
            value = obs[key]
            values.append(value.reshape(value.shape[0], -1))
        return torch.cat(values, dim=-1)

    def _has_obs_group(self, group_name: str, obs: TensorDict | None = None) -> bool:
        keys = self.obs_groups.get(group_name, [])
        if not keys:
            return False
        if obs is None:
            return True
        return all(key in obs.keys() for key in keys)

    def _teacher_has_latent_target(self, obs: TensorDict) -> bool:
        return self._has_obs_group("teacher_policy_history", obs) and self._has_obs_group("teacher_command_window", obs)

    def _flat_identity_mask(self, obs: TensorDict) -> torch.Tensor | None:
        key = self.flat_identity_obs_key
        if key is None or key not in obs.keys():
            return None
        value = obs[key].detach().float().reshape(obs[key].shape[0], -1)
        if self.flat_identity_threshold is not None:
            return (value.abs().max(dim=-1).values <= self.flat_identity_threshold).float()
        return value.mean(dim=-1).clamp(0.0, 1.0)

    def _anchor_group_matches_expected(self, group_name: str, obs: TensorDict) -> bool:
        expected_dim = int(getattr(self.percaptive_motion_tracker, "anchor_estimator_obs_dim", 0))
        if expected_dim == 0:
            return True
        return self._has_obs_group(group_name, obs) and self._obs_group_dim(obs, group_name) == expected_dim

    def _select_anchor_set_name(self, obs: TensorDict, candidates: tuple[str, ...]) -> str:
        expected_dim = int(getattr(self.percaptive_motion_tracker, "anchor_estimator_obs_dim", 0))
        if expected_dim == 0:
            return candidates[0]

        for name in candidates:
            if self._anchor_group_matches_expected(name, obs):
                return name

        if self._anchor_group_matches_expected("policy", obs):
            return "policy"

        raise ValueError(
            f"Could not find anchor estimator obs group with expected_dim={expected_dim}. "
            f"Tried {candidates} and policy."
        )

    def _student_anchor_set_name(self, obs: TensorDict) -> str:
        return self._select_anchor_set_name(obs, ("anchor_estimator",))

    def _teacher_anchor_set_name(self, obs: TensorDict) -> str:
        return self._select_anchor_set_name(obs, ("teacher_anchor_estimator", "anchor_estimator"))

    def _compute_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        tracker = self.percaptive_motion_tracker
        with torch.no_grad():
            z_flat, _, h_last = tracker.encode_motion_token(
                obs,
                history_set_name="policy_history",
                motion_set_name="command_window",
                return_history=True,
            )
            if self._teacher_has_latent_target(obs):
                z_opt = tracker.encode_motion_token(
                    obs,
                    history_set_name="teacher_policy_history",
                    motion_set_name="teacher_command_window",
                )
            elif self.require_teacher_latent_target:
                raise KeyError(
                    "PMA requires teacher_policy_history and teacher_command_window to compute z_opt. "
                    "Set require_teacher_latent_target=False only for identity/no-teacher ablations."
                )
            else:
                z_opt = z_flat

        context = self._get_concat_flat(obs, self.adapter_context_set_name)
        adapter_input = torch.cat((z_flat, context), dim=-1) if self.adapter_use_z_flat else context
        adapter_outputs = self.perceptive_motion_adapter(adapter_input, z_flat)

        tracker.update_distribution_from_token(
            obs,
            adapter_outputs["z_task"],
            policy_set_name=self.tracker_policy_set_name,
            history_set_name="policy_history",
            anchor_estimator_set_name=self._student_anchor_set_name(obs),
            h_last=h_last,
        )

        outputs = {
            "action": tracker.action_mean,
            "actions": tracker.action_mean,
            "z_flat": z_flat,
            "z_opt": z_opt,
            "z_flat_token": z_flat,
            "z_opt_token": z_opt,
            "motion_token": adapter_outputs["z_task"],
            **adapter_outputs,
        }
        if self._prev_delta_z is not None and self._prev_delta_z.shape == adapter_outputs["delta_z"].shape:
            outputs["delta_z_prev"] = self._prev_delta_z
            # This cache is not guaranteed to be the previous timestep in shuffled learner minibatches.
            outputs["delta_z_prev_is_temporal"] = torch.zeros(
                (), dtype=torch.bool, device=adapter_outputs["delta_z"].device
            )
        self._prev_delta_z = adapter_outputs["delta_z"].detach()

        flat_identity_mask = self._flat_identity_mask(obs)
        if flat_identity_mask is not None:
            outputs["flat_identity_mask"] = flat_identity_mask

        with torch.no_grad():
            outputs_detached = {key: value.detach().float() for key, value in outputs.items() if torch.is_tensor(value)}
            gate = outputs_detached.get("gate")
            self._last_bridge_debug = {
                "bridge_z_flat_abs_mean": float(outputs_detached["z_flat"].abs().mean().item()),
                "bridge_z_task_abs_mean": float(outputs_detached["z_task"].abs().mean().item()),
                "bridge_z_opt_abs_mean": float(outputs_detached["z_opt"].abs().mean().item()),
                "bridge_delta_z_abs_mean": float(outputs_detached["delta_z"].abs().mean().item()),
                "bridge_identity_residual_abs_mean": float(outputs_detached["identity_residual"].abs().mean().item()),
                "bridge_latent_cosine": float(
                    F.cosine_similarity(outputs_detached["z_task"], outputs_detached["z_opt"], dim=-1).mean().item()
                ),
                "bridge_latent_norm_ratio": float(
                    (
                        outputs_detached["z_task"].norm(dim=-1)
                        / outputs_detached["z_opt"].norm(dim=-1).clamp_min(1.0e-6)
                    )
                    .mean()
                    .item()
                ),
            }
            if gate is not None:
                self._last_bridge_debug["bridge_gate_mean"] = float(gate.mean().item())
                self._last_bridge_debug["bridge_gate_abs_mean"] = float(gate.abs().mean().item())

        return outputs

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        return self._compute_student_outputs(obs)

    def act(self, obs: TensorDict) -> torch.Tensor:
        self._compute_student_outputs(obs)
        return self.percaptive_motion_tracker.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self._compute_student_outputs(obs)["action"]

    def infer_teacher_action(self, obs: TensorDict) -> torch.Tensor:
        tracker = self.percaptive_motion_tracker
        with torch.no_grad():
            if self._teacher_has_latent_target(obs):
                z_opt, _, h_last = tracker.encode_motion_token(
                    obs,
                    history_set_name="teacher_policy_history",
                    motion_set_name="teacher_command_window",
                    return_history=True,
                )
                policy_set_name = "teacher" if self._has_obs_group("teacher", obs) else "policy"
                tracker.update_distribution_from_token(
                    obs,
                    z_opt,
                    policy_set_name=policy_set_name,
                    history_set_name="teacher_policy_history",
                    anchor_estimator_set_name=self._teacher_anchor_set_name(obs),
                    h_last=h_last,
                )
                return tracker.action_mean
            if self.require_teacher_latent_target:
                raise KeyError(
                    "PMA requires teacher latent target groups to infer teacher actions. "
                    "Set require_teacher_latent_target=False only for identity/no-teacher ablations."
                )
            tracker._update_distribution(obs)
            return tracker.action_mean

    def evaluate(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        del kwargs
        critic_obs = self._get_concat_flat(obs, self.critic_set_name)
        return self.pma_critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.percaptive_motion_tracker.get_actions_log_prob(actions)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.update_tracker_normalization:
            self.percaptive_motion_tracker.update_normalization(obs)

    def get_last_aux_outputs(self, clear: bool = True) -> dict[str, torch.Tensor]:
        return self.percaptive_motion_tracker.get_last_aux_outputs(clear=clear)

    def normalize_velocity(self, value: torch.Tensor) -> torch.Tensor:
        return self.percaptive_motion_tracker.normalize_velocity(value)

    def normalize_anchor(self, value: torch.Tensor) -> torch.Tensor:
        return self.percaptive_motion_tracker.normalize_anchor(value)

    def reset(self, dones: torch.Tensor | None = None, hidden_states: tuple | None = None) -> None:
        if dones is None or bool(torch.as_tensor(dones).any().item()):
            self._prev_delta_z = None
        reset_fn = getattr(self.percaptive_motion_tracker, "reset", None)
        if reset_fn is None:
            return
        try:
            if hidden_states is not None:
                reset_fn(dones=dones, hidden_states=hidden_states)
            else:
                reset_fn(dones=dones)
        except TypeError:
            reset_fn(dones=dones)

    def get_hidden_states(self) -> tuple[None, None]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        return None

    def get_last_bridge_debug(self, clear: bool = True) -> dict[str, float]:
        debug = dict(self._last_bridge_debug)
        if clear:
            self._last_bridge_debug.clear()
        return debug

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        tracker_metadata = {}
        if hasattr(self.percaptive_motion_tracker, "get_checkpoint_metadata"):
            tracker_metadata = self.percaptive_motion_tracker.get_checkpoint_metadata()
        tracker_signature = dict(tracker_metadata.get("signature", {}))
        return {
            "policy_class": self.__class__.__name__,
            "policy_family": "perceptive_motion_adapter_tracker",
            "architecture": self.__class__.__name__,
            "tracker_metadata": tracker_metadata,
            "obs_schema": tracker_metadata.get("obs_schema"),
            "signature": {
                **tracker_signature,
                "adapter_mode": self.adapter_mode,
                "adapter_hidden_dims": self.adapter_hidden_dims,
                "adapter_activation": self.adapter_activation,
                "adapter_context_set_name": self.adapter_context_set_name,
                "adapter_use_z_flat": self.adapter_use_z_flat,
                "adapter_delta_scale": self.adapter_delta_scale,
                "adapter_gate_bias": self.adapter_gate_bias,
                "critic_hidden_dims": self.critic_hidden_dims,
                "critic_set_name": self.critic_set_name,
                "tracker_policy_set_name": self.tracker_policy_set_name,
                "update_tracker_normalization": self.update_tracker_normalization,
                "require_tracker_checkpoint": self.require_tracker_checkpoint,
                "require_teacher_latent_target": self.require_teacher_latent_target,
                "flat_identity_obs_key": self.flat_identity_obs_key,
                "flat_identity_threshold": self.flat_identity_threshold,
                "partial_tracker_load_min_match_fraction": self.partial_tracker_load_min_match_fraction,
                "partial_adapter_load_min_match_fraction": self.partial_adapter_load_min_match_fraction,
            },
            "perceptive_motion_tracker_ckpt_path": self.perceptive_motion_tracker_ckpt_path,
            "percaptive_motion_tracker_ckpt_path": self.percaptive_motion_tracker_ckpt_path,
            "perceptive_motion_adapter_ckpt_path": self.perceptive_motion_adapter_ckpt_path,
            "adapter_mode": self.adapter_mode,
            "adapter_hidden_dims": self.adapter_hidden_dims,
            "adapter_activation": self.adapter_activation,
            "adapter_context_set_name": self.adapter_context_set_name,
            "adapter_use_z_flat": self.adapter_use_z_flat,
            "adapter_delta_scale": self.adapter_delta_scale,
            "adapter_gate_bias": self.adapter_gate_bias,
            "critic_hidden_dims": self.critic_hidden_dims,
            "critic_set_name": self.critic_set_name,
            "tracker_policy_set_name": self.tracker_policy_set_name,
            "update_tracker_normalization": self.update_tracker_normalization,
            "require_tracker_checkpoint": self.require_tracker_checkpoint,
            "require_teacher_latent_target": self.require_teacher_latent_target,
            "flat_identity_obs_key": self.flat_identity_obs_key,
            "flat_identity_threshold": self.flat_identity_threshold,
            "partial_tracker_load_min_match_fraction": self.partial_tracker_load_min_match_fraction,
            "partial_adapter_load_min_match_fraction": self.partial_adapter_load_min_match_fraction,
        }

    def train(self, mode: bool = True) -> "PerceptiveMotionAdapterTracker":
        super().train(mode)
        self.perceptive_motion_adapter.train(mode)
        self.percaptive_motion_tracker.train(False)
        self._freeze_tracker()
        return self

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        result = nn.Module.load_state_dict(self, state_dict, strict=strict)
        self._freeze_tracker()
        self.loaded_teacher = True
        return result
