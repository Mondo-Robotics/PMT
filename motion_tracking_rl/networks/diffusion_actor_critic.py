# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion-based actor-critic policy for Flow Policy Optimization++ (FPO++)."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from tensordict import TensorDict

from motion_tracking_rl.networks.layers import EmpiricalNormalization, MLP
from motion_tracking_rl.networks.ode_solver import ODESolver
from motion_tracking_rl.utils import resolve_nn_activation
from motion_tracking_rl.registry import register_network


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.view(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaLN(nn.Module):
    """Adaptive LayerNorm modulated by timestep embedding."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


@register_network("DiffusionActorCritic", compat_name="diffusion")
class DiffusionActorCritic(nn.Module):
    """Actor-critic with a diffusion-style policy used by FPO++."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256),
        activation: str = "elu",
        num_steps: int = 64,
        solver_method: str = "euler",
        parameterization: str = "velocity",
        timestep_embed_dim: int = 8,
        sample_t_strategy: str = "uniform",
        p_mean: float = -1.2,
        p_std: float = 1.2,
        perturb_action_std: float = 0.02,
        cfm_target_std: float | None = None,
        cfm_loss_reduction: str = "sqrt",
        cfm_loss_t_inverse_cdf_beta: float = 1.0,
        actor_scale: float | list[float] | tuple[float, ...] = 1.0,
        action_bound: float = 0.9,
        action_clip: float | None = None,
        zero_sampling_inference: bool = False,
        rollout_zero_noise: bool = False,
        loss_dim_mask: list[float] | None = None,
        **kwargs: dict[str, object],
    ) -> None:
        sampling_steps = kwargs.pop("sampling_steps", None)
        if sampling_steps is not None:
            num_steps = int(sampling_steps)
        action_perturb_std = kwargs.pop("action_perturb_std", None)
        if action_perturb_std is not None:
            perturb_action_std = float(action_perturb_std)
        if kwargs:
            print(
                "DiffusionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if "policy" not in obs_groups or "critic" not in obs_groups:
            raise ValueError("DiffusionActorCritic requires 'policy' and 'critic' observation groups.")

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.solver_method = solver_method
        if parameterization not in {"velocity", "data"}:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        self.parameterization = parameterization
        self.sample_t_strategy = sample_t_strategy
        self.p_mean = p_mean
        self.p_std = p_std
        self.perturb_action_std = perturb_action_std
        if cfm_target_std is not None and (not math.isfinite(cfm_target_std) or cfm_target_std <= 0.0):
            raise ValueError(f"cfm_target_std must be > 0 and finite when provided, got {cfm_target_std}.")
        self.cfm_target_std = cfm_target_std
        if cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError(f"Unknown cfm_loss_reduction: {cfm_loss_reduction}")
        self.cfm_loss_reduction = cfm_loss_reduction
        if not math.isfinite(cfm_loss_t_inverse_cdf_beta) or cfm_loss_t_inverse_cdf_beta <= 0.0:
            raise ValueError(
                "cfm_loss_t_inverse_cdf_beta must be > 0 and finite, "
                f"got {cfm_loss_t_inverse_cdf_beta}."
            )
        self.cfm_loss_t_inverse_cdf_beta = float(cfm_loss_t_inverse_cdf_beta)
        self.action_bound = action_bound
        self.action_clip = action_clip
        self.zero_sampling_inference = zero_sampling_inference
        self.rollout_zero_noise = rollout_zero_noise
        self._activation = activation
        actor_scale_tensor = torch.as_tensor(actor_scale, dtype=torch.float32)
        if actor_scale_tensor.numel() == 1:
            actor_scale_tensor = actor_scale_tensor.repeat(num_actions)
        if actor_scale_tensor.numel() != num_actions:
            raise ValueError(f"actor_scale has {actor_scale_tensor.numel()} values, expected {num_actions}.")
        if torch.any(actor_scale_tensor <= 0):
            raise ValueError("actor_scale values must be positive.")
        self.register_buffer("actor_scale", actor_scale_tensor.reshape(1, num_actions))
        if loss_dim_mask is not None:
            mask = torch.tensor(loss_dim_mask, dtype=torch.float32)
            if mask.numel() != num_actions:
                raise ValueError(
                    f"loss_dim_mask length {mask.numel()} does not match num_actions {num_actions}."
                )
            self.register_buffer("_loss_dim_mask", mask)
        else:
            self._loss_dim_mask = None

        # Warn if SONIC-specific observation keys are present.
        if "robot_encoder" in obs.keys() or "human_encoder" in obs.keys():
            warnings.warn(
                "DiffusionActorCritic ignores SONIC encoder observations. "
                "Use non-SONIC tasks for FPO++.",
                stacklevel=2,
            )

        # Compute observation sizes.
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) != 2:
                raise ValueError("DiffusionActorCritic only supports 1D observations.")
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) != 2:
                raise ValueError("DiffusionActorCritic only supports 1D observations.")
            num_critic_obs += obs[obs_group].shape[-1]

        # Actor: diffusion policy backbone with timestep conditioning.
        hidden_size = actor_hidden_dims[-1] if len(actor_hidden_dims) > 0 else 256
        actor_hidden = list(actor_hidden_dims[:-1]) if len(actor_hidden_dims) > 1 else []
        self.actor_mlp = MLP(num_actor_obs + num_actions, hidden_size, actor_hidden, activation)
        self.actor_norm = AdaLN(hidden_size)
        self.post_adaln_act = resolve_nn_activation(activation)
        self.actor_head = nn.Linear(hidden_size, num_actions)

        # Noise embedder.
        self.noise_emb = TimestepEmbedder(hidden_size, timestep_embed_dim)

        # Critic: standard value function.
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)

        # Observation normalization.
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self._last_action_mean: torch.Tensor | None = None
        self._last_action_std: torch.Tensor | None = None
        self.mean_bound_loss: torch.Tensor | None = None
        self._solver = ODESolver()

    @property
    def action_mean(self) -> torch.Tensor:
        if self._last_action_mean is None:
            device = next(self.parameters()).device
            return torch.zeros(1, self.num_actions, device=device)
        return self._last_action_mean

    @property
    def action_std(self) -> torch.Tensor:
        if self._last_action_std is None:
            device = next(self.parameters()).device
            return torch.ones(1, self.num_actions, device=device)
        return self._last_action_std

    @property
    def entropy(self) -> torch.Tensor:
        if self._last_action_mean is None:
            device = next(self.parameters()).device
            return torch.zeros(1, device=device)
        return torch.zeros(self._last_action_mean.shape[0], device=self._last_action_mean.device)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def _sample_t(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sample_t_strategy == "uniform":
            return torch.rand(shape, device=device, dtype=dtype)
        if self.sample_t_strategy == "lognormal":
            rnd_normal = torch.randn(shape, device=device, dtype=dtype)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()
            time = 1.0 / (1.0 + sigma)
            return torch.clamp(time, min=1e-4, max=1.0)
        raise ValueError(f"Unknown sample_t_strategy: {self.sample_t_strategy}")

    def _sample_cfm_loss_t(
        self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        u = torch.rand(shape, device=device, dtype=dtype)
        beta = self.cfm_loss_t_inverse_cdf_beta
        return 0.005 + 0.99 * (1.0 - (1.0 - u).pow(1.0 / beta))

    def _compute_squared_error(self, diff: torch.Tensor) -> torch.Tensor:
        squared_error = diff.square()
        if self._loss_dim_mask is not None:
            mask = self._loss_dim_mask.to(device=diff.device, dtype=diff.dtype)
            squared_error = squared_error * mask
            action_dim = mask.sum().clamp(min=1.0)
        else:
            action_dim = torch.tensor(diff.shape[-1], device=diff.device, dtype=diff.dtype)

        if self.cfm_loss_reduction == "mean":
            return squared_error.sum(dim=-1) / action_dim
        if self.cfm_loss_reduction == "sum":
            return squared_error.sum(dim=-1)
        if self.cfm_loss_reduction == "sqrt":
            return squared_error.sum(dim=-1) / torch.sqrt(action_dim)
        raise ValueError(f"Unknown cfm_loss_reduction: {self.cfm_loss_reduction}")

    def _predict(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.noise_emb(t)
        x_inp = torch.cat([x_t, obs], dim=-1)
        hidden = self.actor_mlp(x_inp)
        hidden = self.actor_norm(hidden, t_emb)
        hidden = self.post_adaln_act(hidden)
        return self.actor_head(hidden)

    def _predict_velocity(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.parameterization == "velocity":
            return self._predict(obs, x_t, t)
        if self.parameterization == "data":
            x1 = self._predict(obs, x_t, t)
            denom = (1.0 - t).clamp(min=1e-3).unsqueeze(-1) if t.dim() == 1 else (1.0 - t).clamp(min=1e-3)
            return (x1 - x_t) / denom
        raise ValueError(f"Unknown parameterization: {self.parameterization}")

    def sample_diffusion_action(
        self, obs: torch.Tensor, zero_noise: bool = False, enable_grad: bool = False
    ) -> torch.Tensor:
        """Sample actions using Euler integration of a learned velocity field."""
        batch_size = obs.shape[0]
        device = obs.device
        dtype = obs.dtype

        # Flow direction matches the original FPO: noise lives at t=1, the action at
        # t=0 (consistent with the CFM forward process x_t = t*eps + (1-t)*scaled_actions
        # and target velocity eps - scaled_actions). We therefore start from x(t=1)=noise
        # and integrate the velocity field BACKWARD in time 1.0 -> 0.0.
        if zero_noise:
            x_1 = torch.zeros(batch_size, self.num_actions, device=device, dtype=dtype)
        else:
            x_1 = torch.randn(batch_size, self.num_actions, device=device, dtype=dtype)
        time_grid = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        step_size = 1.0 / self.num_steps

        def velocity_fn(x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
            t_batch = torch.ones((batch_size,), device=device, dtype=dtype) * t
            return self._predict_velocity(obs, x, t_batch)

        x_t = self._solver.sample(
            velocity_fn,
            x_init=x_1,
            step_size=step_size,
            method=self.solver_method,
            time_grid=time_grid,
            enable_grad=enable_grad,
            obs=obs,
        )

        # Original perturbs AFTER scaling by actor_scale (entropy-style regularizer on
        # the final action), so apply the scale first then add noise.
        actor_scale = self.actor_scale.to(device=x_t.device, dtype=x_t.dtype)
        actions = x_t * actor_scale
        if self.training and self.perturb_action_std > 0 and not zero_noise:
            actions = actions + torch.randn_like(actions) * self.perturb_action_std
        return actions

    def act(self, obs: TensorDict, **kwargs: dict[str, object]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actions = self.sample_diffusion_action(actor_obs, zero_noise=self.rollout_zero_noise)
        if self.action_clip is not None:
            actions = torch.clamp(actions, -self.action_clip, self.action_clip)
        self._last_action_mean = actions.detach()
        self._last_action_std = torch.ones_like(actions)
        return actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actions = self.sample_diffusion_action(actor_obs, zero_noise=self.zero_sampling_inference)
        if self.action_clip is not None:
            actions = torch.clamp(actions, -self.action_clip, self.action_clip)
        self._last_action_mean = actions.detach()
        self._last_action_std = torch.ones_like(actions)
        return actions

    def act_with_cfm_info(self, obs_dict: TensorDict, num_samples: int):
        """Sample actions along with CFM information used for FPO++ updates."""
        actor_obs = self.get_actor_obs(obs_dict)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        with torch.no_grad():
            actions = self.sample_diffusion_action(actor_obs, zero_noise=self.rollout_zero_noise)
            if self.action_clip is not None:
                actions = torch.clamp(actions, -self.action_clip, self.action_clip)

        self._last_action_mean = actions.detach()
        # Keep vector-shaped std for logging/storage compatibility.
        std = actions.std(dim=0, keepdim=True)
        self._last_action_std = std.expand_as(actions)

        batch_size = actor_obs.shape[0]
        device = actor_obs.device
        dtype = actor_obs.dtype
        eps_sample = torch.randn(batch_size, num_samples, self.num_actions, device=device, dtype=dtype)
        t_sample = self._sample_cfm_loss_t((batch_size, num_samples, 1), device=device, dtype=dtype)

        with torch.no_grad():
            B, N, D = eps_sample.shape
            obs_tile = actor_obs.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            actions_tile = actions.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            flat_eps = eps_sample.reshape(B * N, D)
            flat_t = t_sample.reshape(B * N, 1)
            initial_cfm_loss = self.compute_cfm_loss(obs_tile, actions_tile, flat_eps, flat_t)
            initial_cfm_loss = initial_cfm_loss.view(B, N)

        cfm_info = {
            "initial_cfm_loss": initial_cfm_loss,
            "loss_eps": eps_sample,
            "loss_t": t_sample,
        }
        return actions, cfm_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, object]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def bound_loss(self, actions: torch.Tensor) -> torch.Tensor:
        bound = self.action_bound
        loss = torch.zeros_like(actions)
        loss = torch.where(actions > bound, (actions - bound) ** 2, loss)
        loss = torch.where(actions < -bound, (actions + bound) ** 2, loss)
        return loss.mean()

    def compute_cfm_loss(
        self, obs: torch.Tensor, actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute Conditional Flow Matching loss for a batch."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        actor_scale = self.actor_scale.to(device=actions.device, dtype=actions.dtype)
        scaled_actions = actions / actor_scale
        x_t = t * eps + (1.0 - t) * scaled_actions
        t_flat = t.view(-1)
        pred = self._predict(obs, x_t, t_flat)
        if self.parameterization == "velocity":
            velocity_target = eps - scaled_actions
            diff = pred - velocity_target
        elif self.parameterization == "data":
            diff = pred - scaled_actions
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        self.mean_bound_loss = None
        return self._compute_squared_error(diff)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load checkpoint weights.

        Returns:
            True when full training state resume is expected.
            False when this is a transfer load (e.g. distillation checkpoint).
        """
        # Distillation checkpoint path: model_state_dict contains "student.*" and "teacher.*" keys.
        if any(key.startswith("student.") for key in state_dict.keys()):
            student_state = {
                key.replace("student.", "", 1): value for key, value in state_dict.items() if key.startswith("student.")
            }
            if not student_state:
                raise ValueError("No 'student.*' parameters found in provided state_dict.")
            nn.Module.load_state_dict(self, student_state, strict=strict)
            # Transfer-load only; optimizer/iteration should not be resumed.
            return False

        # Standard FPO++ checkpoint.
        nn.Module.load_state_dict(self, state_dict, strict=strict)
        return True
