"""Convert a PMT/SONIC motion .npz (Isaac-Lab BFS body/joint axis order) into mjlab's
MJCF body/joint axis order.

Phase A discovery (MJLAB_BACKEND_PLAN.md Risk #1, CONFIRMED): PMT clips store
body_pos_w/body_quat_w/... and joint_pos/joint_vel in Isaac Lab's BFS articulation order
(see whole_body_tracking/g1_motion_gen/g1mg/constants.py). mjlab indexes motion arrays
positionally against its MJCF body/joint order. The two orders differ, so clips load and
step but track garbage unless remapped. The permutations below are verified by forward
kinematics (max body-position error 0.000 m).

Usage:
  <mjlab-repo>/.venv/bin/python scripts/pmt_npz_to_mjlab.py in.npz out.npz
"""

from __future__ import annotations

import sys

import numpy as np

# Isaac-Lab BFS order the PMT/SONIC npz are saved in.
BFS_BODIES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
    "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link", "left_shoulder_roll_link",
    "right_shoulder_roll_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_elbow_link",
    "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link", "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
BFS_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
JOINT_KEYS = ("joint_pos", "joint_vel")


def mjlab_orders() -> tuple[list[str], list[str]]:
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    m = get_spec().compile()
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
    return bodies, joints


def convert(in_path: str, out_path: str) -> None:
    mj_bodies, mj_joints = mjlab_orders()
    body_perm = [BFS_BODIES.index(n) for n in mj_bodies]
    joint_perm = [BFS_JOINTS.index(n) for n in mj_joints]

    d = np.load(in_path)
    out = {}
    for k in d.files:
        a = d[k]
        if k in BODY_KEYS:
            out[k] = a[:, body_perm]
        elif k in JOINT_KEYS:
            out[k] = a[:, joint_perm]
        else:
            out[k] = a  # fps, etc.
    np.savez(out_path, **out)
    print(f"[convert] {in_path} -> {out_path}  (bodies+joints remapped BFS->MJCF)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2])
