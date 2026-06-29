"""Phase D (MJLAB_BACKEND_PLAN.md): load a trained PMT checkpoint and measure whether it
tracks motion inside mjlab.

The trained PMT G1 flat policy is a plain ActorCritic MLP (512/256/128, ELU) with actor obs
= 286 dims in this order (== mjlab's CRITIC layout):
  command(58) anchor_pos_b(3) anchor_ori_b(6) body_pos(42) body_ori(84)
  base_lin_vel(3) base_ang_vel(3) joint_pos(29) joint_vel(29) actions(29)
plus a built-in obs normalizer. We build an mjlab flat-tracking env whose ACTOR group uses
mjlab's critic terms (same layout), load the actor weights + normalizer, and roll out on a
(remapped) training clip, reporting joint/anchor/body tracking error.

Run:
  <mjlab-repo>/.venv/bin/python scripts/mjlab_eval_pmt_ckpt.py \
      --ckpt /tmp/pmt_ckpt/model_39999.pt --motion /tmp/mvp_motion/Aeroplane_BR_mjlab.npz
"""

from __future__ import annotations

import argparse

import torch


def build_mlp(obs_dim: int, act_dim: int, hidden=(512, 256, 128)):
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = obs_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers += [nn.Linear(prev, act_dim)]
    return nn.Sequential(*layers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--motion", required=True, help="mjlab-ORDER npz (already remapped)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument(
        "--root-frame-basevel",
        action="store_true",
        help="override IMU base_lin/ang_vel obs slices with PMT-style root-frame velocity",
    )
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import (
        unitree_g1_flat_tracking_env_cfg,
    )

    cfg = unitree_g1_flat_tracking_env_cfg(play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.commands["motion"].motion_file = args.motion
    # actor must see the SAME 286-dim layout the PMT policy was trained on => use the
    # critic term set for the actor group too.
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=dict(cfg.observations["critic"].terms),
        concatenate_terms=True,
        enable_corruption=False,
    )

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    obs, _ = env.reset()
    obs_dim = obs["actor"].shape[-1]
    act_dim = env.action_manager.total_action_dim
    print(f"[env] actor obs dim={obs_dim}  act dim={act_dim}")

    # --- load PMT policy ---
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    exp_dim = sd["actor.0.weight"].shape[1]
    assert obs_dim == exp_dim, (
        f"obs dim mismatch: mjlab actor={obs_dim} vs ckpt={exp_dim}. "
        "Term layout differs — align before loading."
    )
    actor = build_mlp(obs_dim, act_dim).to(device)
    with torch.no_grad():
        # rsl-rl stores actor as Sequential indices 0,2,4,6 (Linear) — same as our build.
        for i in (0, 2, 4, 6):
            actor[i].weight.copy_(sd[f"actor.{i}.weight"])
            actor[i].bias.copy_(sd[f"actor.{i}.bias"])
        mean = sd["actor_obs_normalizer._mean"].to(device)
        std = sd["actor_obs_normalizer._std"].to(device)
    actor.eval()

    # --- joint-order bridge (Phase D finding) ---------------------------------
    # The PMT policy was trained in Isaac BFS joint order; mjlab obs/actions are in MJCF
    # order. body/anchor terms share order (tracked-body name list identical), but the
    # joint-indexed slices of the 286-dim obs and the 29-dim action must be permuted.
    # Actor obs layout (dims): command[0:58]=motion(jp29,jv29) anchor_pos[58:61]
    #   anchor_ori[61:67] body_pos[67:109] body_ori[109:193] base_lin[193:196]
    #   base_ang[196:199] joint_pos[199:228] joint_vel[228:257] actions[257:286]
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    m = get_spec().compile()
    mjj = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
    BFS = [
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
    mjcf_to_bfs = torch.tensor([mjj.index(n) for n in BFS], device=device)
    bfs_to_mjcf = torch.tensor([BFS.index(n) for n in mjj], device=device)

    # joint-indexed slice starts within the 286-dim actor obs (each 29 wide).
    JSLICES = [(0, 29), (29, 58), (199, 228), (228, 257), (257, 286)]

    def to_bfs(o):
        o = o.clone()
        for a, b in JSLICES:
            o[:, a:b] = o[:, a:b][:, mjcf_to_bfs]
        return o

    def policy(o):
        a_bfs = actor((to_bfs(o) - mean) / (std + 1e-8))
        return a_bfs[:, bfs_to_mjcf]  # BFS action -> MJCF for the env

    cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    # base_lin_vel[193:196], base_ang_vel[196:199] in the 286-dim actor obs.
    BLV, BAV = slice(193, 196), slice(196, 199)

    def patch_basevel(o):
        if not args.root_frame_basevel:
            return o
        o = o.clone()
        o[:, BLV] = robot.data.root_link_lin_vel_b
        o[:, BAV] = robot.data.root_link_ang_vel_b
        return o

    je, ae, be = [], [], []
    with torch.no_grad():
        for _ in range(args.steps):
            act = policy(patch_basevel(obs["actor"]))
            obs, rew, term, trunc, extra = env.step(act)
            je.append(torch.norm(cmd.joint_pos - cmd.robot_joint_pos, dim=-1).mean().item())
            ae.append(torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1).mean().item())
            be.append(torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1).mean().item())

    import numpy as np

    je, ae, be = np.array(je), np.array(ae), np.array(be)
    print(f"[track] joint_pos err  mean={je.mean():.4f} rad  final={je[-1]:.4f}")
    print(f"[track] anchor_pos err mean={ae.mean():.4f} m   final={ae[-1]:.4f}")
    print(f"[track] body_pos  err  mean={be.mean():.4f} m   final={be[-1]:.4f}")
    # A tracking policy should hold joint err well under the ~0.6 rad zero-action drift.
    verdict = "TRACKS ✅" if je.mean() < 0.3 and be.mean() < 0.3 else "DOES NOT TRACK ❌"
    print(f"\nPHASE D: {verdict}  (joint {je.mean():.3f} rad, body {be.mean():.3f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
