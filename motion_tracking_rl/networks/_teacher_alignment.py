"""Shared teacher→student action-alignment helper for distillation wrappers.

Both ``VisionStudentTeacher`` (vision_student_teacher.py) and
``VisionAblationStudentTeacher`` (vision_ablation_actor_critic.py) need to align a
teacher's actions to the student's command-window reference (a ``q_ref`` delta
compensation) and record bridge-debug stats. The two classes are independent
``nn.Module`` siblings (no shared base), and the alignment body was duplicated
verbatim in both. This module factors that body into one function that operates on
the policy instance, so the logic lives in a single place.

The ``policy`` argument must expose:
  * ``teacher`` (with an optional ``cmd_len`` attr),
  * ``obs_groups`` with ``teacher_command_window`` / ``command_window`` keys,
  * ``align_teacher_to_student_reference`` (bool),
  * ``_get_concat_seq(obs, groups, expected_seq_len=...)``,
and it will have ``_last_bridge_debug`` set as a side effect.
"""
from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict


def align_teacher_actions(policy: Any, obs: TensorDict, teacher_actions: torch.Tensor) -> torch.Tensor:
    """Align teacher actions to the student command-window reference (q_ref delta).

    Returns the (optionally) aligned teacher actions and records bridge-debug stats on
    ``policy._last_bridge_debug``. Behaviour is identical to the previously-duplicated
    ``_align_teacher_actions`` methods.
    """
    expected_len = getattr(policy.teacher, "cmd_len", None)
    teacher_primary_window = policy._get_concat_seq(
        obs,
        [policy.obs_groups["teacher_command_window"][0]],
        expected_seq_len=expected_len,
    )
    student_primary_window = policy._get_concat_seq(
        obs,
        [policy.obs_groups["command_window"][0]],
        expected_seq_len=expected_len,
    )
    center_index = teacher_primary_window.shape[1] // 2

    # The first command-window group keeps the original command-window layout:
    # [v_ref_b(3), w_ref_b(3), g_ref_b(3), q_ref(29)].
    q_ref_teacher = teacher_primary_window[:, center_index, 9:]
    q_ref_student = student_primary_window[:, center_index, 9:]
    if q_ref_teacher.shape[-1] != teacher_actions.shape[-1]:
        raise ValueError(
            f"Teacher reference dim {q_ref_teacher.shape[-1]} does not match action dim {teacher_actions.shape[-1]}."
        )
    q_ref_delta = q_ref_teacher - q_ref_student
    aligned_teacher_actions = (
        teacher_actions + q_ref_delta if policy.align_teacher_to_student_reference else teacher_actions
    )
    primary_delta = teacher_primary_window - student_primary_window
    action_delta = aligned_teacher_actions - teacher_actions
    policy._last_bridge_debug = {
        "bridge_align_enabled": float(policy.align_teacher_to_student_reference),
        "bridge_primary_window_mae": float(primary_delta.abs().mean().item()),
        "bridge_primary_window_max_abs": float(primary_delta.abs().max().item()),
        "bridge_qref_mae": float(q_ref_delta.abs().mean().item()),
        "bridge_qref_max_abs": float(q_ref_delta.abs().max().item()),
        "bridge_teacher_action_abs_mean": float(teacher_actions.abs().mean().item()),
        "bridge_aligned_teacher_action_abs_mean": float(aligned_teacher_actions.abs().mean().item()),
        "bridge_alignment_delta_mae": float(action_delta.abs().mean().item()),
        "bridge_alignment_delta_max_abs": float(action_delta.abs().max().item()),
    }
    return aligned_teacher_actions
