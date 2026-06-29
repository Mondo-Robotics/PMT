"""Canonical body-name lists for the BFM-Zero port (no torch / no isaac imports).

Kept import-light so the IsaacLab env cfg can reference ``PRIVILEGED_BODY_NAMES`` at task
registration time without pulling in torch_utils / humanoidverse.
"""

from __future__ import annotations

# Pelvis-first 30-body order for privileged_state. MUST start with "pelvis" (root); the remaining
# 29 may be in any FIXED order as long as online (vec_env) and expert (motion store) use this same
# list. These are the exact 30 IsaacLab G1 body names (verified against robot.body_names).
PRIVILEGED_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]

NUM_PRIVILEGED_BODIES = len(PRIVILEGED_BODY_NAMES)
assert NUM_PRIVILEGED_BODIES == 30
assert PRIVILEGED_BODY_NAMES[0] == "pelvis"
