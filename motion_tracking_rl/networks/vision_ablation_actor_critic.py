from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from motion_tracking_rl.networks._teacher_alignment import align_teacher_actions
from motion_tracking_rl.networks.layers import EmpiricalNormalization, HiddenState, MLP, Memory
from motion_tracking_rl.networks.transformer_actor_critic import TransformerActorCritic
from motion_tracking_rl.networks.vision_transformer_actor_critic import VisionTransformerActorCritic
from motion_tracking_rl.registry import register_network
from motion_tracking_rl.utils import build_obs_schema, resolve_nn_activation, unpad_trajectories


class HeightScanCnnEncoder(nn.Module):
    """Small CNN encoder for flattened height scans."""

    def __init__(
        self,
        vision_dim: int,
        output_dim: int,
        *,
        map_height: int = 17,
        map_width: int = 11,
        z_clip: float = 3.0,
        normalize_height: bool = True,
        channels: tuple[int, ...] | list[int] = (16, 32),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.z_clip = float(z_clip)
        self.normalize_height = bool(normalize_height)
        cells = self.map_height * self.map_width
        if vision_dim == cells:
            self.input_channels = 1
        elif vision_dim == 2 * cells:
            self.input_channels = 2
        else:
            raise ValueError(
                f"Height scan dim must be H*W or 2*H*W with H={map_height}, W={map_width}; got {vision_dim}."
            )

        layers: list[nn.Module] = []
        in_channels = self.input_channels
        act = resolve_nn_activation(activation)
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, int(out_channels), kernel_size=3, stride=1, padding=1))
            layers.append(act)
            in_channels = int(out_channels)
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = MLP(in_channels, output_dim, [max(in_channels, output_dim)], activation)

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        batch_size = vision.shape[0]
        cells = self.map_height * self.map_width
        height = vision[..., :cells]
        if self.z_clip > 0.0:
            height = height.clamp(-self.z_clip, self.z_clip)
            if self.normalize_height:
                height = height / self.z_clip
        if self.input_channels == 2:
            mask = vision[..., cells:]
            image = torch.cat([height, mask], dim=-1).reshape(batch_size, 2, self.map_height, self.map_width)
        else:
            image = height.reshape(batch_size, 1, self.map_height, self.map_width)
        features = self.pool(self.cnn(image)).flatten(start_dim=1)
        return self.proj(features)


@register_network("VisionAblationActorCritic", compat_name="vision_ablation")
class VisionAblationActorCritic(nn.Module):
    """Actor-critic for architecture ablations using the same proprio and vision observations.

    Supported architectures:
    - ``flat_mlp``: concatenate proprio/token features and height scan, then feed one MLP.
    - ``split_mlp``: encode proprio/token features and height scan with separate MLPs, then fuse.
    - ``split_cnn``: encode proprio/token features with an MLP and height scan with a CNN, then fuse.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        architecture: str = "flat_mlp",
        actor_obs_normalization: bool = True,
        vision_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        policy_encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        vision_encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        fusion_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        cnn_channels: tuple[int, ...] | list[int] = (16, 32),
        feature_dim: int = 128,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        map_height: int = 17,
        map_width: int = 11,
        z_clip: float = 3.0,
        normalize_height: bool = True,
        use_vel_estimator: bool = True,
        vel_estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        vel_estimator_output_dim: int = 3,
        vel_gt_normalization: bool = True,
        use_anchor_estimator: bool = True,
        anchor_estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        anchor_estimator_output_dim: int = 3,
        anchor_gt_normalization: bool = True,
        use_foot_traj_head: bool = True,
        foot_traj_output_dim: int | None = None,
        foot_traj_target_obs_key: str | None = "foot_traj_target",
        foot_traj_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "VisionAblationActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()
        self.obs_groups = obs_groups
        self.architecture = architecture.lower()
        if self.architecture not in {"flat_mlp", "split_mlp", "split_cnn"}:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. Expected one of: flat_mlp, split_mlp, split_cnn."
            )

        self.num_actions = int(num_actions)
        self.noise_std_type = noise_std_type
        self.use_vel_estimator = bool(use_vel_estimator)
        self.use_anchor_estimator = bool(use_anchor_estimator)
        self.use_foot_traj_head = bool(use_foot_traj_head)
        self.vel_output_dim = int(vel_estimator_output_dim)
        self.anchor_output_dim = int(anchor_estimator_output_dim)
        self.vel_gt_normalization = bool(vel_gt_normalization)
        self.anchor_gt_normalization = bool(anchor_gt_normalization)
        self.foot_traj_target_obs_key = foot_traj_target_obs_key
        self.foot_traj_output_dim = self._resolve_foot_traj_dim(obs, foot_traj_output_dim, foot_traj_target_obs_key)
        self._obs_schema = build_obs_schema(obs, obs_groups)

        self.policy_obs_dim = self._infer_flat_dim(obs, obs_groups.get("policy", []))
        self.vision_obs_dim = self._infer_flat_dim(obs, obs_groups.get("vision", []))
        self.critic_obs_dim = self._infer_flat_dim(obs, obs_groups.get("critic", []))
        if self.vision_obs_dim <= 0:
            raise ValueError("vision ablations require obs_groups['vision'].")

        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.vision_obs_normalization = bool(vision_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.actor_obs_normalizer = EmpiricalNormalization(self.policy_obs_dim) if actor_obs_normalization else nn.Identity()
        self.vision_obs_normalizer = EmpiricalNormalization(self.vision_obs_dim) if vision_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(self.critic_obs_dim) if critic_obs_normalization else nn.Identity()

        self.feature_dim = int(feature_dim)
        if self.architecture == "flat_mlp":
            actor_input_dim = self.policy_obs_dim + self.vision_obs_dim
            hidden = list(actor_hidden_dims[:-1]) if len(actor_hidden_dims) > 1 else [self.feature_dim]
            self.actor_encoder = MLP(actor_input_dim, self.feature_dim, hidden, activation)
            self.policy_encoder = None
            self.vision_encoder = None
        else:
            self.policy_encoder = MLP(
                self.policy_obs_dim,
                self.feature_dim,
                list(policy_encoder_hidden_dims),
                activation,
            )
            if self.architecture == "split_mlp":
                self.vision_encoder = MLP(
                    self.vision_obs_dim,
                    self.feature_dim,
                    list(vision_encoder_hidden_dims),
                    activation,
                )
            else:
                self.vision_encoder = HeightScanCnnEncoder(
                    self.vision_obs_dim,
                    self.feature_dim,
                    map_height=map_height,
                    map_width=map_width,
                    z_clip=z_clip,
                    normalize_height=normalize_height,
                    channels=cnn_channels,
                    activation=activation,
                )
            self.actor_encoder = MLP(2 * self.feature_dim, self.feature_dim, list(fusion_hidden_dims), activation)

        self.mean_head = nn.Linear(self.feature_dim, self.num_actions)
        self.critic = MLP(self.critic_obs_dim + self.feature_dim, 1, list(critic_hidden_dims), activation)

        self.vel_head = (
            MLP(self.feature_dim, self.vel_output_dim, list(vel_estimator_hidden_dims), activation)
            if self.use_vel_estimator
            else None
        )
        self.vel_gt_normalizer = EmpiricalNormalization(self.vel_output_dim) if vel_gt_normalization else nn.Identity()
        self.anchor_head = (
            MLP(self.feature_dim, self.anchor_output_dim, list(anchor_estimator_hidden_dims), activation)
            if self.use_anchor_estimator
            else None
        )
        self.anchor_gt_normalizer = (
            EmpiricalNormalization(self.anchor_output_dim) if anchor_gt_normalization else nn.Identity()
        )
        self.foot_traj_head = (
            MLP(self.feature_dim, self.foot_traj_output_dim, list(foot_traj_hidden_dims), activation)
            if self.use_foot_traj_head and self.foot_traj_output_dim > 0
            else None
        )

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
            self.log_std = None
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
            self.std = None
        else:
            raise ValueError(f"Unknown noise_std_type '{noise_std_type}'. Expected 'scalar' or 'log'.")

        self.distribution: Normal | None = None
        self._last_aux_outputs: dict[str, torch.Tensor] | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _infer_flat_dim(obs: TensorDict, groups: list[str]) -> int:
        total = 0
        for group in groups:
            if group not in obs.keys():
                raise KeyError(f"Observation group '{group}' not found. Available groups: {list(obs.keys())}")
            total += int(torch.tensor(obs[group].shape[1:]).prod().item())
        return total

    @staticmethod
    def _resolve_foot_traj_dim(
        obs: TensorDict,
        foot_traj_output_dim: int | None,
        foot_traj_target_obs_key: str | None,
    ) -> int:
        if foot_traj_output_dim is not None:
            return int(foot_traj_output_dim)
        if foot_traj_target_obs_key is None or foot_traj_target_obs_key not in obs.keys():
            return 0
        return int(torch.tensor(obs[foot_traj_target_obs_key].shape[1:]).prod().item())

    def _concat_flat(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        values = [obs[group].reshape(*obs.batch_size, -1) for group in groups]
        return torch.cat(values, dim=-1)

    def _policy_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._concat_flat(obs, self.obs_groups.get("policy", []))

    def _vision_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._concat_flat(obs, self.obs_groups.get("vision", []))

    def _critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._concat_flat(obs, self.obs_groups.get("critic", []))

    def _encode_actor(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs = self.actor_obs_normalizer(self._policy_obs(obs))
        vision_obs = self.vision_obs_normalizer(self._vision_obs(obs))
        if self.architecture == "flat_mlp":
            feature = self.actor_encoder(torch.cat([policy_obs, vision_obs], dim=-1))
            vision_feature = feature
        else:
            policy_feature = self.policy_encoder(policy_obs)
            vision_feature = self.vision_encoder(vision_obs)
            feature = self.actor_encoder(torch.cat([policy_feature, vision_feature], dim=-1))
        return feature, vision_feature

    def _make_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return torch.clamp(F.softplus(self.std), min=1.0e-6, max=math.exp(2.0)).expand_as(mean)
        return torch.exp(torch.clamp(self.log_std, -5.0, 2.0)).expand_as(mean)

    def _update_distribution(self, obs: TensorDict) -> None:
        feature, _ = self._encode_actor(obs)
        mean = self.mean_head(feature)
        self.distribution = Normal(mean, self._make_std(mean))
        aux: dict[str, torch.Tensor] = {}
        if self.vel_head is not None:
            aux["v_hat"] = self.vel_head(feature)
        if self.anchor_head is not None:
            aux["anchor_hat"] = self.anchor_head(feature)
        if self.foot_traj_head is not None:
            aux["foot_traj"] = self.foot_traj_head(feature)
        self._last_aux_outputs = aux if aux else None

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.mean

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        self._update_distribution(obs)
        outputs = {"action": self.distribution.mean}
        aux = self.get_last_aux_outputs(clear=False)
        if "foot_traj" in aux:
            outputs["foot_traj"] = aux["foot_traj"]
        return outputs

    def evaluate(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self._critic_obs(obs))
        _, vision_feature = self._encode_actor(obs)
        return self.critic(torch.cat([critic_obs, vision_feature], dim=-1))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_last_aux_outputs(self, *, clear: bool = True) -> dict[str, torch.Tensor]:
        aux = self._last_aux_outputs if self._last_aux_outputs is not None else {}
        if clear:
            self._last_aux_outputs = None
        return aux

    def normalize_velocity(self, v: torch.Tensor) -> torch.Tensor:
        return self.vel_gt_normalizer(v) if self.vel_gt_normalization else v

    def normalize_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        return self.anchor_gt_normalizer(anchor) if self.anchor_gt_normalization else anchor

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self._policy_obs(obs))
        if self.vision_obs_normalization:
            self.vision_obs_normalizer.update(self._vision_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self._critic_obs(obs))
        if self.use_vel_estimator and self.vel_gt_normalization and "vel_gt" in self.obs_groups:
            self.vel_gt_normalizer.update(self._concat_flat(obs, self.obs_groups["vel_gt"]))
        if self.use_anchor_estimator and self.anchor_gt_normalization and "anchor_gt" in self.obs_groups:
            self.anchor_gt_normalizer.update(self._concat_flat(obs, self.obs_groups["anchor_gt"]))

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "policy_class": self.__class__.__name__,
            "policy_family": "vision_ablation_actor_critic",
            "architecture": self.architecture,
            "obs_schema": self._obs_schema,
            "signature": {
                "num_actions": self.num_actions,
                "policy_obs_dim": self.policy_obs_dim,
                "vision_obs_dim": self.vision_obs_dim,
                "critic_obs_dim": self.critic_obs_dim,
                "feature_dim": self.feature_dim,
                "use_vel_estimator": self.use_vel_estimator,
                "use_anchor_estimator": self.use_anchor_estimator,
                "use_foot_traj_head": self.use_foot_traj_head,
            },
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any(key.startswith("student.") for key in state_dict.keys()):
            student_state = {
                key.replace("student.", "", 1): value for key, value in state_dict.items() if key.startswith("student.")
            }
            if not student_state:
                raise ValueError("No 'student.*' parameters found in provided state_dict.")
            nn.Module.load_state_dict(self, student_state, strict=strict)
            return False
        nn.Module.load_state_dict(self, state_dict, strict=strict)
        return True


@register_network("VisionAblationRecurrentActorCritic", compat_name="vision_ablation")
class VisionAblationRecurrentActorCritic(VisionAblationActorCritic):
    """GRU/LSTM ablation over the same flattened proprio and vision inputs.

    Same compat axis as the non-recurrent class (``vision_ablation``): the GRU
    ablation is a CLASS SELECTION (different class_name), not a pure ``architecture``
    field flip. PPO supports recurrent, so this pairs with the on_policy runner.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("architecture", "flat_mlp")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.rnn_type = str(rnn_type)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_num_layers = int(rnn_num_layers)
        recurrent_input_dim = self.policy_obs_dim + self.vision_obs_dim
        critic_input_dim = self.critic_obs_dim + self.vision_obs_dim
        self.memory_a = Memory(recurrent_input_dim, self.rnn_hidden_dim, self.rnn_num_layers, self.rnn_type)
        self.memory_c = Memory(critic_input_dim, self.rnn_hidden_dim, self.rnn_num_layers, self.rnn_type)

        activation = kwargs.get("activation", "elu")
        actor_hidden_dims = list(kwargs.get("actor_hidden_dims", (512, 256, 128)))
        critic_hidden_dims = list(kwargs.get("critic_hidden_dims", (512, 256, 128)))
        hidden = actor_hidden_dims[:-1] if len(actor_hidden_dims) > 1 else [self.feature_dim]
        self.actor_encoder = MLP(self.rnn_hidden_dim, self.feature_dim, hidden, activation)
        self.policy_encoder = None
        self.vision_encoder = None
        self.critic = MLP(self.rnn_hidden_dim, 1, critic_hidden_dims, activation)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: tuple[HiddenState, HiddenState] = (None, None),
    ) -> None:
        self.memory_a.reset(dones, hidden_states[0])
        self.memory_c.reset(dones, hidden_states[1])

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def _actor_recurrent_input(self, obs: TensorDict) -> torch.Tensor:
        policy_obs = self.actor_obs_normalizer(self._policy_obs(obs))
        vision_obs = self.vision_obs_normalizer(self._vision_obs(obs))
        return torch.cat([policy_obs, vision_obs], dim=-1)

    def _set_aux_outputs_from_feature(self, feature: torch.Tensor) -> None:
        aux: dict[str, torch.Tensor] = {}
        if self.vel_head is not None:
            aux["v_hat"] = self.vel_head(feature)
        if self.anchor_head is not None:
            aux["anchor_hat"] = self.anchor_head(feature)
        if self.foot_traj_head is not None:
            aux["foot_traj"] = self.foot_traj_head(feature)
        self._last_aux_outputs = aux if aux else None

    def _update_distribution_from_feature(self, feature: torch.Tensor) -> None:
        mean = self.mean_head(feature)
        self.distribution = Normal(mean, self._make_std(mean))
        self._set_aux_outputs_from_feature(feature)

    def act(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        rnn_input = self._actor_recurrent_input(obs)
        if masks is not None:
            if hidden_state is None:
                raise ValueError("Hidden states not passed to recurrent ablation policy during PPO update")
            rnn_out_padded, _ = self.memory_a.rnn(rnn_input, hidden_state)
            aux_feature = self.actor_encoder(rnn_out_padded)
            rnn_out = unpad_trajectories(rnn_out_padded, masks).squeeze(0)
        else:
            aux_feature = None
            rnn_out = self.memory_a(rnn_input, masks, hidden_state).squeeze(0)
        feature = self.actor_encoder(rnn_out)
        self._update_distribution_from_feature(feature)
        if aux_feature is not None:
            self._set_aux_outputs_from_feature(aux_feature)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        rnn_input = self._actor_recurrent_input(obs)
        rnn_out = self.memory_a(rnn_input).squeeze(0)
        feature = self.actor_encoder(rnn_out)
        return self.mean_head(feature)

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        rnn_input = self._actor_recurrent_input(obs)
        rnn_out = self.memory_a(rnn_input).squeeze(0)
        feature = self.actor_encoder(rnn_out)
        self._update_distribution_from_feature(feature)
        outputs = {"action": self.distribution.mean}
        aux = self.get_last_aux_outputs(clear=False)
        if "foot_traj" in aux:
            outputs["foot_traj"] = aux["foot_traj"]
        return outputs

    def evaluate(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self._critic_obs(obs))
        vision_obs = self.vision_obs_normalizer(self._vision_obs(obs))
        rnn_input = torch.cat([critic_obs, vision_obs], dim=-1)
        rnn_out = self.memory_c(rnn_input, masks, hidden_state).squeeze(0)
        return self.critic(rnn_out)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().get_checkpoint_metadata()
        metadata["policy_family"] = "vision_ablation_recurrent_actor_critic"
        metadata["signature"].update(
            {
                "rnn_type": self.rnn_type,
                "rnn_hidden_dim": self.rnn_hidden_dim,
                "rnn_num_layers": self.rnn_num_layers,
            }
        )
        return metadata

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.detach_hidden_state(dones)
        self.memory_c.detach_hidden_state(dones)


class VisionAblationStudentTeacher(nn.Module):
    """Distill an vision ablation student from the fixed vision-transformer teacher."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        student_class_name: str = "VisionAblationActorCritic",
        student_cfg: dict[str, Any] | None = None,
        teacher_cfg: dict[str, Any] | None = None,
        teacher_ckpt_path: str | None = None,
        teacher_load_strict: bool = True,
        align_teacher_to_student_reference: bool = True,
        foot_traj_target_obs_key: str | None = "foot_traj_target",
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "VisionAblationStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        required_sets = ["policy", "teacher", "teacher_policy_history", "teacher_command_window"]
        missing_sets = [name for name in required_sets if name not in obs_groups]
        if "vision" not in obs_groups and "teacher_vision" not in obs_groups:
            missing_sets.append("vision or teacher_vision")
        if missing_sets:
            raise ValueError(
                f"VisionAblationStudentTeacher requires observation sets {required_sets}. Missing: {missing_sets}"
            )
        if align_teacher_to_student_reference and "command_window" not in obs_groups:
            raise ValueError("align_teacher_to_student_reference=True requires obs_groups['command_window'].")

        self.obs_groups = obs_groups
        self.loaded_teacher = False
        self.teacher_ckpt_path = teacher_ckpt_path
        self.student_class_name = str(student_class_name)
        self.align_teacher_to_student_reference = bool(align_teacher_to_student_reference)
        self.foot_traj_target_obs_key = foot_traj_target_obs_key
        self._obs_schema = build_obs_schema(obs, obs_groups)
        self._last_bridge_debug: dict[str, float] = {}

        student_cfg = {} if student_cfg is None else dict(student_cfg)
        teacher_cfg = {} if teacher_cfg is None else dict(teacher_cfg)

        student_obs_groups = {
            "policy": list(obs_groups["policy"]),
            "critic": list(obs_groups.get("critic", obs_groups["policy"])),
        }
        if self.student_class_name in {"VisionTransformerActorCritic", "TransformerActorCritic"}:
            student_obs_groups["policy_history"] = list(obs_groups["policy_history"])
            student_obs_groups["command_window"] = list(obs_groups["command_window"])
        if "vision" in obs_groups:
            student_obs_groups["vision"] = list(obs_groups["vision"])
        if "vel_gt" in obs_groups:
            student_obs_groups["vel_gt"] = list(obs_groups["vel_gt"])
        if "anchor_gt" in obs_groups:
            student_obs_groups["anchor_gt"] = list(obs_groups["anchor_gt"])
        if "foot_traj_target" in obs_groups:
            student_obs_groups["foot_traj_target"] = list(obs_groups["foot_traj_target"])

        teacher_vision_groups = obs_groups["teacher_vision"] if "teacher_vision" in obs_groups else obs_groups["vision"]
        teacher_obs_groups = {
            "policy": list(obs_groups["teacher"]),
            "policy_history": list(obs_groups["teacher_policy_history"]),
            "command_window": list(obs_groups["teacher_command_window"]),
            "critic": list(obs_groups.get("teacher_critic", obs_groups.get("critic", obs_groups["teacher"]))),
            "vision": list(teacher_vision_groups),
        }
        if "teacher_vel_gt" in obs_groups:
            teacher_obs_groups["vel_gt"] = list(obs_groups["teacher_vel_gt"])
        elif "vel_gt" in obs_groups:
            teacher_obs_groups["vel_gt"] = list(obs_groups["vel_gt"])
        if "teacher_anchor_gt" in obs_groups:
            teacher_obs_groups["anchor_gt"] = list(obs_groups["teacher_anchor_gt"])
        elif "anchor_gt" in obs_groups:
            teacher_obs_groups["anchor_gt"] = list(obs_groups["anchor_gt"])
        if "teacher_anchor_estimator" in obs_groups and obs_groups["teacher_anchor_estimator"]:
            teacher_obs_groups["anchor_estimator"] = list(obs_groups["teacher_anchor_estimator"])

        student_classes = {
            "TransformerActorCritic": TransformerActorCritic,
            "VisionTransformerActorCritic": VisionTransformerActorCritic,
            "VisionAblationActorCritic": VisionAblationActorCritic,
            "VisionAblationRecurrentActorCritic": VisionAblationRecurrentActorCritic,
        }
        if self.student_class_name not in student_classes:
            raise ValueError(
                f"Unsupported ablation student_class_name '{self.student_class_name}'. "
                f"Expected one of {sorted(student_classes)}."
            )
        student_class = student_classes[self.student_class_name]
        self.student = student_class(
            obs=obs,
            obs_groups=student_obs_groups,
            num_actions=num_actions,
            **student_cfg,
        )
        self.is_recurrent = bool(getattr(self.student, "is_recurrent", False))

        self.teacher = VisionTransformerActorCritic(
            obs=obs,
            obs_groups=teacher_obs_groups,
            num_actions=num_actions,
            **teacher_cfg,
        )
        self._freeze_teacher()

        if teacher_ckpt_path:
            self._load_teacher_checkpoint(Path(teacher_ckpt_path), strict=teacher_load_strict)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.student.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.student.action_std

    @property
    def entropy(self) -> torch.Tensor:
        return self.student.entropy

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def _load_teacher_checkpoint(self, ckpt_path: Path, strict: bool) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
        loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported teacher checkpoint format in: {ckpt_path}")
        if any(key.startswith("teacher.") for key in state_dict.keys()):
            teacher_state = {
                key.replace("teacher.", "", 1): value for key, value in state_dict.items() if key.startswith("teacher.")
            }
        else:
            teacher_state = state_dict
        self.teacher.load_state_dict(teacher_state, strict=strict)
        self._freeze_teacher()
        self.loaded_teacher = True
        print(f"[VisionAblationStudentTeacher] Loaded VisionTransformerActorCritic teacher: {ckpt_path}")

    def act(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act(obs)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act_inference(obs)

    def infer_student_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        if hasattr(self.student, "infer_student_outputs"):
            return self.student.infer_student_outputs(obs)
        action = self.student.act_inference(obs)
        return {"action": action}

    def get_last_aux_outputs(self, *, clear: bool = True) -> dict[str, torch.Tensor]:
        return self.student.get_last_aux_outputs(clear=clear) if hasattr(self.student, "get_last_aux_outputs") else {}

    def get_last_bridge_debug(self, *, clear: bool = True) -> dict[str, float]:
        debug = dict(self._last_bridge_debug)
        if clear:
            self._last_bridge_debug = {}
        return debug

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        student_metadata = self.student.get_checkpoint_metadata()
        teacher_metadata = self.teacher.get_checkpoint_metadata()
        return {
            "policy_class": self.__class__.__name__,
            "policy_family": "vision_ablation_student_teacher",
            "obs_schema": self._obs_schema,
            "signature": {
                "student_class_name": self.student_class_name,
                "align_teacher_to_student_reference": bool(self.align_teacher_to_student_reference),
                "foot_traj_target_obs_key": self.foot_traj_target_obs_key,
                "student": student_metadata.get("signature", {}),
                "teacher": teacher_metadata.get("signature", {}),
            },
            "subpolicies": {
                "student": student_metadata,
                "teacher": teacher_metadata,
            },
        }

    def normalize_velocity(self, v: torch.Tensor) -> torch.Tensor:
        return self.student.normalize_velocity(v) if hasattr(self.student, "normalize_velocity") else v

    def normalize_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        return self.student.normalize_anchor(anchor) if hasattr(self.student, "normalize_anchor") else anchor

    @staticmethod
    def _get_concat_seq(obs: TensorDict, groups: list[str], expected_seq_len: int | None = None) -> torch.Tensor:
        xs = []
        seq_len = expected_seq_len
        for key in groups:
            value = obs[key]
            if value.ndim == 2:
                if seq_len is None:
                    raise ValueError(f"Cannot infer sequence length for observation '{key}'.")
                if value.shape[-1] % seq_len != 0:
                    raise ValueError(
                        f"Observation '{key}' last dim {value.shape[-1]} is not divisible by seq_len={seq_len}."
                    )
                value = value.reshape(value.shape[0], seq_len, -1)
            elif value.ndim != 3:
                raise ValueError(f"Observation '{key}' must be 2D or 3D. Got shape: {tuple(value.shape)}")
            if seq_len is None:
                seq_len = value.shape[1]
            elif value.shape[1] != seq_len:
                raise ValueError(
                    f"Command-window obs groups must share the same sequence length. Got {seq_len} and {value.shape[1]}."
                )
            xs.append(value)
        if not xs:
            raise KeyError("Command-window observation set is empty.")
        return torch.cat(xs, dim=-1)

    def _align_teacher_actions(self, obs: TensorDict, teacher_actions: torch.Tensor) -> torch.Tensor:
        return align_teacher_actions(self, obs, teacher_actions)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            teacher_actions = self.teacher.act_inference(obs)
            return self._align_teacher_actions(obs, teacher_actions)

    def update_normalization(self, obs: TensorDict) -> None:
        self.student.update_normalization(obs)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: tuple[HiddenState, HiddenState] | None = None,
    ) -> None:
        if hidden_states is None:
            hidden_states = (None, None)
        if self.is_recurrent:
            self.student.reset(dones, hidden_states)
        else:
            self.student.reset(dones)
        self.teacher.reset(dones)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        if self.is_recurrent and hasattr(self.student, "get_hidden_states"):
            return self.student.get_hidden_states()
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        if hasattr(self.student, "detach_hidden_states"):
            self.student.detach_hidden_states(dones)

    def train(self, mode: bool = True):
        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()
        return self

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any(key.startswith("student.") for key in state_dict.keys()):
            nn.Module.load_state_dict(self, state_dict, strict=strict)
            self._freeze_teacher()
            self.loaded_teacher = True
            return True
        self.teacher.load_state_dict(state_dict, strict=strict)
        self._freeze_teacher()
        self.loaded_teacher = True
        return False
