from __future__ import annotations

from typing import Any

import torch.nn as nn

from motion_tracking_rl.networks.actor_critic import SonicActorCritic


class OfficialSonicActorCritic(SonicActorCritic):
    """Clean SONIC policy wrapper for official-style G1 training modes.

    The legacy ``SonicActorCritic`` mixes enabled branches with trainability.
    This wrapper keeps the deployed action/encoder contract but adds explicit
    freeze and auxiliary-loss switches so configs can choose scratch training,
    finetuning, or encoder-only updates without changing the rollout code.
    """

    def __init__(
        self,
        *args: Any,
        training_mode: str = "scratch_robot",
        freeze_robot_encoder: bool = False,
        freeze_human_encoder: bool = False,
        freeze_hybrid_encoder: bool = False,
        freeze_control_decoder: bool = False,
        freeze_motion_decoder: bool = False,
        freeze_critic: bool = False,
        freeze_action_std: bool = False,
        aux_train_robot_encoder: bool | None = None,
        aux_train_human_encoder: bool | None = None,
        aux_train_hybrid_encoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.training_mode = training_mode
        self.freeze_robot_encoder = bool(freeze_robot_encoder)
        self.freeze_human_encoder = bool(freeze_human_encoder)
        self.freeze_hybrid_encoder = bool(freeze_hybrid_encoder)
        self.freeze_control_decoder = bool(freeze_control_decoder)
        self.freeze_motion_decoder = bool(freeze_motion_decoder)
        self.freeze_critic = bool(freeze_critic)
        self.freeze_action_std = bool(freeze_action_std)

        super().__init__(*args, **kwargs)

        self.aux_train_robot_encoder = (
            bool(aux_train_robot_encoder)
            if aux_train_robot_encoder is not None
            else bool(self.train_robot_encoder and not self.freeze_robot_encoder)
        )
        self.aux_train_human_encoder = (
            bool(aux_train_human_encoder)
            if aux_train_human_encoder is not None
            else bool(self.train_human_encoder and not self.freeze_human_encoder)
        )
        self.aux_train_hybrid_encoder = (
            bool(aux_train_hybrid_encoder)
            if aux_train_hybrid_encoder is not None
            else bool(self.train_hybrid_encoder and not self.freeze_hybrid_encoder)
        )

        self._apply_component_freeze()

    def _set_module_trainable(self, module: nn.Module | None, trainable: bool) -> None:
        self._set_trainable(module, trainable)

    def _apply_component_freeze(self) -> None:
        self._set_module_trainable(self.robot_encoder, self.train_robot_encoder and not self.freeze_robot_encoder)
        self._set_module_trainable(self.robot_encoder_proj, self.train_robot_encoder and not self.freeze_robot_encoder)
        self._set_module_trainable(self.human_encoder, self.train_human_encoder and not self.freeze_human_encoder)
        self._set_module_trainable(self.human_encoder_proj, self.train_human_encoder and not self.freeze_human_encoder)
        self._set_module_trainable(self.hybrid_encoder, self.train_hybrid_encoder and not self.freeze_hybrid_encoder)
        self._set_module_trainable(self.hybrid_encoder_proj, self.train_hybrid_encoder and not self.freeze_hybrid_encoder)
        self._set_module_trainable(self.control_decoder, not self.freeze_control_decoder)
        self._set_module_trainable(self.motion_decoder, not self.freeze_motion_decoder)
        self._set_module_trainable(self.critic, not self.freeze_critic)
        self.std.requires_grad_(not self.freeze_action_std)

    def _apply_hard_freezes_only(self) -> None:
        if self.freeze_robot_encoder:
            self._set_trainable(self.robot_encoder, False)
            self._set_trainable(self.robot_encoder_proj, False)
        if self.freeze_human_encoder:
            self._set_trainable(self.human_encoder, False)
            self._set_trainable(self.human_encoder_proj, False)
        if self.freeze_hybrid_encoder:
            self._set_trainable(self.hybrid_encoder, False)
            self._set_trainable(self.hybrid_encoder_proj, False)
        if self.freeze_control_decoder:
            self._set_trainable(self.control_decoder, False)
        if self.freeze_motion_decoder:
            self._set_trainable(self.motion_decoder, False)
        if self.freeze_critic:
            self._set_trainable(self.critic, False)
        if self.freeze_action_std:
            self.std.requires_grad_(False)

    def get_sonic_encoder_train_cfg(self) -> dict[str, bool]:
        return {
            "robot": self.aux_train_robot_encoder,
            "human": self.aux_train_human_encoder,
            "hybrid": self.aux_train_hybrid_encoder,
        }

    def set_warmup_freeze(
        self,
        *,
        freeze_encoders: bool,
        freeze_control_decoder: bool,
        freeze_action_std: bool,
    ) -> dict[str, bool]:
        state = super().set_warmup_freeze(
            freeze_encoders=freeze_encoders,
            freeze_control_decoder=freeze_control_decoder,
            freeze_action_std=freeze_action_std,
        )
        self._apply_hard_freezes_only()
        state.update(
            {
                "hard_freeze_robot_encoder": self.freeze_robot_encoder,
                "hard_freeze_human_encoder": self.freeze_human_encoder,
                "hard_freeze_hybrid_encoder": self.freeze_hybrid_encoder,
                "hard_freeze_control_decoder": self.freeze_control_decoder,
                "hard_freeze_motion_decoder": self.freeze_motion_decoder,
                "hard_freeze_critic": self.freeze_critic,
                "hard_freeze_action_std": self.freeze_action_std,
            }
        )
        self._warmup_freeze_state = dict(state)
        return state

    def get_component_trainable_summary(self) -> dict[str, bool]:
        def any_trainable(module: nn.Module | None) -> bool:
            return module is not None and any(param.requires_grad for param in module.parameters())

        return {
            "robot_encoder": any_trainable(self.robot_encoder) or any_trainable(self.robot_encoder_proj),
            "human_encoder": any_trainable(self.human_encoder) or any_trainable(self.human_encoder_proj),
            "hybrid_encoder": any_trainable(self.hybrid_encoder) or any_trainable(self.hybrid_encoder_proj),
            "control_decoder": any_trainable(self.control_decoder),
            "motion_decoder": any_trainable(self.motion_decoder),
            "critic": any_trainable(self.critic),
            "action_std": bool(self.std.requires_grad),
        }
