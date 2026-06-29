# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural-network components for RL agents.

High-level actor-critic / student-teacher networks live directly under this
package (formerly ``motion_tracking_rl.modules``). Low-level shared building
blocks (MLP, normalization, memory, SRU cells) live under
``motion_tracking_rl.networks.layers`` and are re-exported here for
backward compatibility.
"""

# --- Low-level building blocks (re-exported from layers/) ---
from .layers import (
    GRU_SRU,
    GRUSRUCell,
    LSTM_SRU,
    LSTMSRUCell,
    HiddenState,
    Memory,
    MLP,
    EmpiricalDiscountedVariationNormalization,
    EmpiricalNormalization,
)

# --- High-level networks (moved from modules/) ---
from .actor_critic import ActorCritic, SonicActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .diff_normalizer import DiffNormalizer
from .diffusion_actor_critic import DiffusionActorCritic
from .transformer_actor_critic import TransformerActorCritic
from .vision_transformer_actor_critic import VisionTransformerActorCritic
from .rnd import RandomNetworkDistillation, resolve_rnd_config
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .official_sonic_actor_critic import OfficialSonicActorCritic
from .symmetry import resolve_symmetry_config
from .vision_sonic import VisionSonicActorCritic, MapTransformer
from .vision_student_teacher import VisionStudentTeacher
from .perceptive_motion import (
    PercaptiveMotionTracker,
    PerceptiveMotionTracker,
    PerceptiveMotionAdapter,
    PerceptiveMotionAdapterTracker,
    PerceptiveMotionTokenTracker,
    PerceptiveResidualBehaviorTokenTracker,
)
from .residual_vision_sonic import ResidualVisionSonicActorCritic
from .residual_vision_action import ModularVisionSonicActorCritic
from .deploy_residual_vision_sonic import DeployResidualVisionSonicActorCritic
from .vision_ablation_actor_critic import (
    VisionAblationActorCritic,
    VisionAblationRecurrentActorCritic,
    VisionAblationStudentTeacher,
)
from .sonic_diffusion_student_teacher import SonicDiffusionStudentTeacher

__all__ = [
    # high-level
    "ActorCritic",
    "SonicActorCritic",
    "OfficialSonicActorCritic",
    "DiffusionActorCritic",
    "TransformerActorCritic",
    "VisionTransformerActorCritic",
    "VisionSonicActorCritic",
    "VisionStudentTeacher",
    "PercaptiveMotionTracker",
    "PerceptiveMotionTracker",
    "PerceptiveMotionAdapter",
    "PerceptiveMotionAdapterTracker",
    "PerceptiveMotionTokenTracker",
    "PerceptiveResidualBehaviorTokenTracker",
    "ResidualVisionSonicActorCritic",
    "ModularVisionSonicActorCritic",
    "DeployResidualVisionSonicActorCritic",
    "VisionAblationActorCritic",
    "VisionAblationRecurrentActorCritic",
    "VisionAblationStudentTeacher",
    "MapTransformer",
    "ActorCriticRecurrent",
    "DiffNormalizer",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "SonicDiffusionStudentTeacher",
    "resolve_rnd_config",
    "resolve_symmetry_config",
    # low-level building blocks (re-exported from .layers)
    "MLP",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "GRU_SRU",
    "GRUSRUCell",
    "HiddenState",
    "LSTM_SRU",
    "LSTMSRUCell",
    "Memory",
]
