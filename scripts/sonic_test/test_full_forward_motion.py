"""End-to-end forward of OUR ONNX-loaded SonicActorCritic on a REAL motion frame.

Builds a (motion-derived) 640-D g1 encoder branch + 930-D decoder proprio history from
walk1_subject1.npz, runs encode_robot -> FSQ token -> control_decoder, and reports the
29-D action. This shows the reproduced net runs the full deploy inference path with the
released weights and yields finite, bounded actions (open-loop; closed-loop tracking
needs the MuJoCo/Isaac sim loop).
"""
from __future__ import annotations

import os
import numpy as np
import torch

# Paths are env-driven so this runs on any machine. Point PMT_SONIC_ONNX_DIR at the
# released SONIC ONNX dir (model_encoder.onnx + model_decoder.onnx) and PMT_TEST_NPZ at a
# motion clip (e.g. lafan_walk/walk1_subject1.npz).
RELEASE = os.environ.get("PMT_SONIC_ONNX_DIR", "<PATH_TO_SONIC_ONNX_RELEASE_DIR>")
ENC = os.path.join(RELEASE, "model_encoder.onnx")
DEC = os.path.join(RELEASE, "model_decoder.onnx")
NPZ = os.environ.get("PMT_TEST_NPZ", "<PATH_TO_MOTION_NPZ>")


def build_net():
    from motion_tracking_rl.networks.actor_critic import SonicActorCritic
    from tensordict import TensorDict
    obs = TensorDict({
        "policy": torch.zeros(1, 930),
        "critic": torch.zeros(1, 930),
        "robot_encoder": torch.zeros(1, 640),
        "encoder_mode_4": torch.zeros(1, 1),
    }, batch_size=[1])
    net = SonicActorCritic(
        obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]}, num_actions=29,
        init_noise_std=0.05, min_action_std=0.001, max_action_std=0.5,
        action_encoder_source="mode", encoder_mode_key="encoder_mode_4",
        detach_action_token=True, decoder_proprio_layout="interleaved_step_history",
        control_decoder_input_order="token_proprio", robot_encoder_layout="g1_onnx_repack",
        train_robot_encoder=True, train_human_encoder=False, train_hybrid_encoder=False,
        encoder_hidden_dims=[2048, 1024, 512, 512],
        actor_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        motion_decoder_hidden_dims=[2048, 1024, 512, 512],
        robot_motion_dim=640, latent_dim=64, fsq_levels=[32] * 64, activation="silu",
        pretrained_encoder_onnx_path=ENC, pretrained_decoder_onnx_path=DEC,
        load_pretrained_robot_encoder=True, load_pretrained_control_decoder=True,
        load_pretrained_human_encoder=False, load_pretrained_hybrid_encoder=False,
        strict_pretrained_shapes=True,
    )
    net.eval()
    return net


def quat_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    # q: [...,4] wxyz -> 6D (first two columns of rotation matrix)
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    r00 = 1 - 2 * (y * y + z * z); r10 = 2 * (x * y + w * z); r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z); r11 = 1 - 2 * (x * x + z * z); r21 = 2 * (y * z + w * x)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)


def main():
    d = np.load(NPZ)
    jp = d["joint_pos"]; jv = d["joint_vel"]; bq = d["body_quat_w"]  # [T,29],[T,29],[T,30,4]
    T = jp.shape[0]
    t0 = 100
    # 640 g1 branch: 10 future frames @ step5 of [joint_pos(29)+joint_vel(29)=58] -> 580,
    # + 10 future anchor-orientation 6D (pelvis quat) -> 60. (Approx deploy layout.)
    idx = np.clip(t0 + np.arange(10) * 5, 0, T - 1)
    posvel = np.concatenate([jp[idx], jv[idx]], axis=-1).reshape(-1)  # [580]
    anchor6d = quat_to_rot6d(bq[idx, 0]).reshape(-1)                  # [60]
    g1_branch = np.concatenate([posvel, anchor6d]).astype(np.float32)[None]  # [1,640]

    # 930 proprio = 10 frames * 93 (deploy interleaved history). Use a plausible per-frame
    # vector [base_ang_vel(3), joint_pos(29), joint_vel(29), last_action(29), gravity(3)]=93.
    frame = np.concatenate([
        np.zeros(3, np.float32), jp[t0], jv[t0], np.zeros(29, np.float32),
        np.array([0, 0, -1], np.float32),
    ])  # 93
    proprio = np.tile(frame, 10).astype(np.float32)[None]  # [1,930]

    net = build_net()
    with torch.no_grad():
        g1_t = torch.from_numpy(g1_branch)
        token = net.encode_robot(g1_t)                 # FSQ-quantized token [1,64]
        prop_t = torch.from_numpy(proprio)
        dec_in = torch.cat([token, prop_t], dim=-1)    # token_proprio order
        action = net.control_decoder(dec_in)           # [1,29]
    a = action.numpy()[0]
    print(f"[motion] frame {t0} of {T}")
    print(f"[ours] token[:8]  = {np.round(token.numpy()[0, :8], 3)}")
    print(f"[ours] action[:8] = {np.round(a[:8], 4)}")
    print(f"[ours] action range = [{a.min():.3f}, {a.max():.3f}], finite={np.isfinite(a).all()}")
    ok = bool(np.isfinite(a).all()) and float(np.max(np.abs(a))) < 50.0
    print(f"[RESULT] full encode->FSQ->decode path runs: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
