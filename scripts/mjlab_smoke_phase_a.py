"""Phase A smoke test (MJLAB_BACKEND_PLAN.md): prove a real PMT motion clip loads, resets,
and steps inside mjlab's stock G1 flat-tracking env, with FINITE command/body tracking errors.

This is the runtime confirmation of the (statically verified) semantic contract for the
G1-flat family. Also doubles as the NPZ validator (keys/dtype/shape/fps/body-order).

Run with mjlab's venv:
  <mjlab-repo>/.venv/bin/python scripts/mjlab_smoke_phase_a.py \
      --motion /tmp/mvp_motion/Aeroplane_BR.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

EXPECTED_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}


def validate_npz(path: str, env_rate_hz: float) -> dict:
    """NPZ validator (plan Phase A): keys, dtype, shape, fps==env-rate."""
    d = np.load(path)
    keys = set(d.files)
    missing = EXPECTED_KEYS - keys
    assert not missing, f"NPZ missing keys: {missing}"
    T, nj = d["joint_pos"].shape
    Tb, nb, three = d["body_pos_w"].shape
    assert d["body_quat_w"].shape == (Tb, nb, 4), d["body_quat_w"].shape
    assert T == Tb, f"frame count mismatch joint={T} body={Tb}"
    fps = int(d["fps"][0]) if "fps" in keys else None
    info = {"frames": T, "n_joints": nj, "n_bodies": nb, "fps": fps}
    if fps is not None:
        # mjlab advances one motion frame per env step → fps must equal env control rate.
        assert abs(fps - env_rate_hz) < 1e-6, (
            f"fps {fps} != env rate {env_rate_hz}Hz → clip would play at wrong speed"
        )
    print(f"[validate] OK  {info}")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="path to a PMT .npz motion clip")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.tracking.config.g1.env_cfgs import (
        unitree_g1_flat_tracking_env_cfg,
    )
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # play=True: infinite episode, RSI off, deterministic "start" sampling.
    env_cfg = unitree_g1_flat_tracking_env_cfg(play=True)
    env_cfg.scene.num_envs = args.num_envs
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = args.motion

    env_rate_hz = 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)
    print(f"[env] dt={env_cfg.sim.mujoco.timestep} decim={env_cfg.decimation} "
          f"-> control rate {env_rate_hz:.1f} Hz")
    validate_npz(args.motion, env_rate_hz)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    print(f"[env] built  num_envs={env.num_envs}  device={device}")
    print(f"[env] robot bodies={len(env.scene['robot'].body_names)} "
          f"joints={len(env.scene['robot'].joint_names)}")

    obs, _ = env.reset()
    act_dim = env.action_manager.total_action_dim
    print(f"[reset] obs groups={list(obs.keys())}  action_dim={act_dim}")

    # Drive with zero actions (PD holds default pose); we only check the tracking
    # *command/error* tensors are finite and sane — not that it tracks well.
    zero = torch.zeros((env.num_envs, act_dim), device=device)
    cmd = env.command_manager.get_term("motion")

    max_anchor_err = 0.0
    max_body_err = 0.0
    for i in range(args.steps):
        obs, rew, term, trunc, extra = env.step(zero)
        assert torch.isfinite(rew).all(), f"non-finite reward at step {i}"
        for g, t in obs.items():
            assert torch.isfinite(t).all(), f"non-finite obs[{g}] at step {i}"
        # MotionCommandView contract fields (plan §2):
        anchor_err = torch.norm(
            cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1
        ).max().item()
        body_err = torch.norm(
            cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1
        ).max().item()
        max_anchor_err = max(max_anchor_err, anchor_err)
        max_body_err = max(max_body_err, body_err)
        assert np.isfinite(anchor_err) and np.isfinite(body_err)

    print(f"[step] {args.steps} steps OK")
    print(f"[track] max anchor_pos err = {max_anchor_err:.3f} m")
    print(f"[track] max body_pos  err = {max_body_err:.3f} m")
    # Zero-action: robot won't follow, so errors grow — but must stay finite & bounded.
    assert max_body_err < 50.0, "body error implausibly large — body-order scramble?"
    print("\nPHASE A SMOKE: PASS ✅  (real PMT clip loads + steps in mjlab, finite errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
