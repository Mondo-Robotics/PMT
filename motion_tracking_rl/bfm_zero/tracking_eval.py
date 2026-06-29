"""Pragmatic zero-shot tracking evaluation for the BFM-Zero IsaacLab runner.

Mirrors the BFM-Zero zero-shot *tracking* protocol: encode a reference clip into a latent z via
the backward map (windowed mean over seq_length frames, then project), then roll out the actor with
that fixed z and measure how well the robot tracks the reference motion.

Metrics (logged under eval/*):
  - tracking_mpjpe_m        : mean per-body position error in the root/yaw-local frame [m]
  - tracking_success        : fraction of (env, frame) with local MPJPE < success_threshold
  - tracking_root_xy_err_m  : mean global root horizontal position error [m]
  - tracking_fall_rate      : fraction of eval envs that terminated early (fell)

Design (per Codex guidance):
  - encode z from seq_length=8 windowed-mean of backward_map over each env's assigned clip
  - roll out a fixed horizon with the FIXED z, agent in eval mode
  - eval transitions do NOT enter the replay buffer; obs normalizers are not updated
  - compare achieved vs reference body positions from the command each step
"""

from __future__ import annotations

import numpy as np
import torch

from . import obs_math


def _yaw_inverse_quat_xyzw(root_quat_xyzw: torch.Tensor) -> torch.Tensor:
    from ._vendor.torch_utils import calc_heading_quat_inv

    return calc_heading_quat_inv(root_quat_xyzw, w_last=True)


def _to_local(body_pos_w: torch.Tensor, root_pos_w: torch.Tensor, heading_inv_xyzw: torch.Tensor) -> torch.Tensor:
    """Express body positions in the root, yaw-aligned (heading-local) frame. [N,B,3]."""
    from ._vendor.torch_utils import my_quat_rotate

    n, b, _ = body_pos_w.shape
    rel = body_pos_w - root_pos_w.unsqueeze(1)
    hexp = heading_inv_xyzw.unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    out = my_quat_rotate(hexp, rel.reshape(n * b, 3)).reshape(n, b, 3)
    return out


@torch.no_grad()
def run_tracking_eval(
    vec_env,
    agent,
    *,
    horizon: int = 250,
    seq_length: int = 8,
    success_threshold: float = 0.25,
    device: str = "cuda",
) -> dict:
    """Run one zero-shot tracking eval pass on the current resident clips. Returns metric dict.

    Uses each env's command-assigned clip. Encodes a per-env z from the clip's first
    ``horizon`` frames (windowed-mean backward map), then rolls out with fixed z.
    """
    model = agent._model
    was_training = model.training
    model.train(False)

    cmd = vec_env.motion_command
    store = cmd.data_store

    # Reset env -> each env assigned a clip + start frame by the command.
    obs_np, _ = vec_env.reset()
    n = vec_env.num_envs
    motion_ids = cmd.motion_ids.clone()
    start_frames = cmd.frame_ids.clone()

    # --- Encode per-env tracking z from the clip (windowed-mean backward map). ---
    lengths = store.motion_lengths.to(device)[motion_ids]
    enc_len = int(min(horizon, int(lengths.min().item()) - 1)) if lengths.numel() else horizon
    enc_len = max(enc_len, seq_length + 1)
    default_jp = vec_env.robot.data.default_joint_pos[0].detach().float().reshape(1, -1)

    # Build expert obs for frames [start .. start+enc_len) per env, then z via backward map.
    from . import expert_streaming as es

    zs = []
    # Process the encode window frame-by-frame into [enc_len, n, dim] then average windows.
    priv_seq = []
    state_seq = []
    for k in range(enc_len):
        fids = torch.clamp(start_frames + k, max=(store.motion_lengths.to(device)[motion_ids] - 1))
        jp, jv, bp, bq, blv, bav = store.get_motion_data(motion_ids, fids)
        to = lambda x: x.to(device=device, dtype=torch.float32)
        jp = to(jp) - default_jp
        obs = es.build_expert_obs_from_frames(jp, to(jv), to(bp), to(bq), to(blv), to(bav))
        state_seq.append(obs["state"])
        priv_seq.append(obs["privileged_state"])
    # backward_map needs the obs dict; use state+privileged (its input filter).
    # z per frame, then BFM windowed-mean over seq_length, then mean over the clip.
    z_frames = []
    for k in range(enc_len):
        b_in = {"state": state_seq[k], "privileged_state": priv_seq[k],
                "last_action": torch.zeros(n, obs_math.LAST_ACTION_DIM, device=device)}
        z_frames.append(model.backward_map(b_in))
    z_stack = torch.stack(z_frames, dim=1)  # [n, enc_len, z_dim]
    # windowed mean (seq_length) then average
    zt = torch.zeros_like(z_stack)
    for k in range(enc_len):
        end = min(k + seq_length, enc_len)
        zt[:, k] = z_stack[:, k:end].mean(dim=1)
    z = model.project_z(zt.mean(dim=1))  # [n, z_dim]

    # --- Roll out with fixed z, measure tracking each step. ---
    mpjpe_acc = torch.zeros(n, device=device)
    succ_acc = torch.zeros(n, device=device)
    root_xy_acc = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    frames_counted = torch.zeros(n, device=device)
    ever_done = torch.zeros(n, dtype=torch.bool, device=device)

    obs_np, _ = vec_env.reset()
    for _ in range(horizon):
        obs = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
        obs.pop("time", None)
        action = agent.act(obs=obs, z=z, mean=True)
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=device)
        obs_np, _, terminated, truncated, _ = vec_env.step(action)

        # Achieved vs reference body positions (world) over tracked bodies.
        ref_w = cmd.body_pos_w  # [n, B, 3]
        ach_w = cmd.robot_body_pos_w  # [n, B, 3]
        root_ref = ref_w[:, 0]
        root_ach = ach_w[:, 0]
        # Achieved robot root quat for heading-local frame.
        rq = obs_math.wxyz_to_xyzw(cmd.robot_body_quat_w[:, 0])
        hinv = _yaw_inverse_quat_xyzw(rq)
        ref_l = _to_local(ref_w, root_ref, _yaw_inverse_quat_xyzw(obs_math.wxyz_to_xyzw(cmd.body_quat_w[:, 0])))
        ach_l = _to_local(ach_w, root_ach, hinv)
        mpjpe = torch.norm(ach_l - ref_l, dim=-1).mean(dim=1)  # [n]
        root_xy = torch.norm((root_ach - root_ref)[:, :2], dim=-1)  # [n]

        m = alive.float()
        mpjpe_acc += mpjpe * m
        succ_acc += (mpjpe < success_threshold).float() * m
        root_xy_acc += root_xy * m
        frames_counted += m

        done = torch.as_tensor(np.logical_or(terminated, truncated), device=device)
        ever_done |= (torch.as_tensor(terminated, device=device) & alive)  # terminated = fall
        alive = alive & ~done

    fc = frames_counted.clamp(min=1)
    metrics = {
        "eval/tracking_mpjpe_m": (mpjpe_acc / fc).mean().item(),
        "eval/tracking_success": (succ_acc / fc).mean().item(),
        "eval/tracking_root_xy_err_m": (root_xy_acc / fc).mean().item(),
        "eval/tracking_fall_rate": ever_done.float().mean().item(),
    }
    model.train(was_training)
    return metrics
