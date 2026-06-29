"""View a trained PMT checkpoint tracking a motion in mjlab's interactive viewer.

Loads the PMT ActorCritic MLP (obs 286 = mjlab critic layout), wires the joint-order bridge
(MJCF<->BFS) at the policy boundary, and launches mjlab's viewer (native window if a display
is available, else a viser web URL).

Examples:
  # auto: native window if $DISPLAY, else viser web URL
  <mjlab-repo>/.venv/bin/python scripts/mjlab_view_pmt_ckpt.py \
      --ckpt /tmp/pmt_ckpt/model_39999.pt --motion /tmp/mvp_motion/Aeroplane_BR.npz

  # force the viser web viewer (good over SSH) — prints a URL to open in your browser
  ... scripts/mjlab_view_pmt_ckpt.py --ckpt ... --motion ... --viewer viser

Notes:
  * --motion takes a RAW PMT/SONIC npz (Isaac BFS order); it is auto-remapped to mjlab MJCF
    order via scripts/pmt_npz_to_mjlab.py. Pass --already-mjlab to skip remap.
  * 1 env by default so the viewer shows a single robot.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

# joint-order bridge (same perms as scripts/pmt_npz_to_mjlab.py)
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

# joint-indexed slices within the 286-dim actor obs (each 29 wide).
JSLICES = [(0, 29), (29, 58), (199, 228), (228, 257), (257, 286)]


def build_mlp(obs_dim: int, act_dim: int, hidden=(512, 256, 128)):
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
    ap.add_argument("--motion", required=True, help="PMT/SONIC npz (BFS order by default)")
    ap.add_argument("--already-mjlab", action="store_true", help="motion is already MJCF order")
    ap.add_argument("--viewer", choices=["auto", "native", "viser"], default="auto")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    # --- remap the clip to mjlab order unless told it already is ---
    motion = args.motion
    if not args.already_mjlab:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pmt_npz_to_mjlab import convert

        cache = Path("/tmp/pmt_mjlab_clips")
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / (Path(motion).stem + "_mjlab.npz")
        if not out.exists():
            convert(motion, str(out))
        motion = str(out)
        print(f"[motion] using mjlab-order clip: {motion}")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    # play=True => infinite episode, deterministic "start" sampling, no push/corruption.
    cfg = unitree_g1_flat_tracking_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = motion
    # actor must see the 286-dim layout the PMT policy trained on == mjlab critic terms.
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=dict(cfg.observations["critic"].terms),
        concatenate_terms=True,
        enable_corruption=False,
    )

    has_display = bool(os.environ.get("DISPLAY"))
    render_mode = None  # viewers drive their own rendering
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)

    # --- joint-order bridge ---
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    m = get_spec().compile()
    mjj = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
    mjcf_to_bfs = torch.tensor([mjj.index(n) for n in BFS_JOINTS], device=device)
    bfs_to_mjcf = torch.tensor([BFS_JOINTS.index(n) for n in mjj], device=device)

    # --- load policy ---
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    obs0 = env.get_observations()
    obs_dim = obs0["actor"].shape[-1]
    act_dim = env.action_manager.total_action_dim
    exp = sd["actor.0.weight"].shape[1]
    assert obs_dim == exp, f"obs dim {obs_dim} != ckpt {exp}"
    actor = build_mlp(obs_dim, act_dim).to(device)
    with torch.no_grad():
        for i in (0, 2, 4, 6):
            actor[i].weight.copy_(sd[f"actor.{i}.weight"])
            actor[i].bias.copy_(sd[f"actor.{i}.bias"])
        mean = sd["actor_obs_normalizer._mean"].to(device)
        std = sd["actor_obs_normalizer._std"].to(device)
    actor.eval()
    print(f"[policy] loaded {args.ckpt}  obs={obs_dim} act={act_dim}")

    def policy(obs):
        o = obs["actor"].clone()
        for a, b in JSLICES:
            o[:, a:b] = o[:, a:b][:, mjcf_to_bfs]
        a_bfs = actor((o - mean) / (std + 1e-8))
        return a_bfs[:, bfs_to_mjcf]

    # --- launch viewer ---
    resolved = args.viewer
    if resolved == "auto":
        resolved = "native" if has_display else "viser"
    print(f"[viewer] launching '{resolved}'  (DISPLAY={os.environ.get('DISPLAY') or 'none'})")

    if resolved == "native":
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(env, policy).run()
    else:
        from mjlab.viewer import ViserPlayViewer

        print("[viewer] open the printed viser URL in your browser (works over SSH).")
        ViserPlayViewer(env, policy).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
