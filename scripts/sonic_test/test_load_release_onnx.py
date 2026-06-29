"""Standalone test: load the official SONIC release ONNX (encoder + control decoder)
into OUR reproduced SonicActorCritic and run a forward pass.

This verifies our PyTorch reproduction is WEIGHT-COMPATIBLE with the released SONIC ONNX
(point ``SONIC_RELEASE_DIR`` / ``PMT_SONIC_ONNX_DIR`` at the released dir) via the
``g1_onnx_repack`` deploy contract (no Isaac/GPU needed).

It also cross-checks our net's outputs against direct onnxruntime inference of the SAME
ONNX graphs on the SAME random obs, so we can quantify whether our forward path matches
the released policy numerically.
"""
from __future__ import annotations

import os
import numpy as np
import torch

RELEASE = os.environ.get(
    "SONIC_RELEASE_DIR",
    os.environ.get("PMT_SONIC_ONNX_DIR", "<PATH_TO_SONIC_ONNX_RELEASE_DIR>"),
)
ENC = os.path.join(RELEASE, "model_encoder.onnx")
DEC = os.path.join(RELEASE, "model_decoder.onnx")

# Deploy (g1_onnx_repack) contract dims (SonicActorCriticG1DeployCfg).
ROBOT_MOTION_DIM = 640
LATENT_DIM = 64
PROPRIO_DIM = 930
NUM_ACTIONS = 29
ENC_ONNX_IN = 1762   # encoder obs_dict input
DEC_ONNX_IN = 994    # decoder obs_dict input (= 64 token + 930 proprio)


def build_net():
    from motion_tracking_rl.networks.actor_critic import SonicActorCritic
    from tensordict import TensorDict

    # Minimal obs dict to size the net (policy=930 proprio, robot_encoder=640).
    obs = TensorDict(
        {
            "policy": torch.zeros(1, PROPRIO_DIM),
            "critic": torch.zeros(1, PROPRIO_DIM),
            "robot_encoder": torch.zeros(1, ROBOT_MOTION_DIM),
            "encoder_mode_4": torch.zeros(1, 1),
        },
        batch_size=[1],
    )
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    net = SonicActorCritic(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=NUM_ACTIONS,
        init_noise_std=0.05,
        min_action_std=0.001,
        max_action_std=0.5,
        action_encoder_source="mode",
        encoder_mode_key="encoder_mode_4",
        detach_action_token=True,
        decoder_proprio_layout="interleaved_step_history",
        control_decoder_input_order="token_proprio",
        robot_encoder_layout="g1_onnx_repack",
        train_robot_encoder=True,
        train_human_encoder=False,
        train_hybrid_encoder=False,
        encoder_hidden_dims=[2048, 1024, 512, 512],
        actor_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        motion_decoder_hidden_dims=[2048, 1024, 512, 512],
        robot_motion_dim=ROBOT_MOTION_DIM,
        latent_dim=LATENT_DIM,
        fsq_levels=[32] * 64,
        activation="silu",
        pretrained_encoder_onnx_path=ENC,
        pretrained_decoder_onnx_path=DEC,
        load_pretrained_robot_encoder=True,
        load_pretrained_human_encoder=False,
        load_pretrained_hybrid_encoder=False,
        load_pretrained_control_decoder=True,
        strict_pretrained_shapes=True,
    )
    net.eval()
    return net


def main():
    print(f"[test] encoder ONNX: {ENC}")
    print(f"[test] decoder ONNX: {DEC}")
    assert os.path.isfile(ENC) and os.path.isfile(DEC), "release ONNX missing"

    net = build_net()
    print("[test] SonicActorCritic built + ONNX weights loaded OK")

    import onnxruntime as ort

    rng = np.random.default_rng(0)
    # Random encoder obs in the ONNX's native layout (1762) and decoder proprio (930).
    enc_in = rng.standard_normal((1, ENC_ONNX_IN)).astype(np.float32)
    proprio = rng.standard_normal((1, PROPRIO_DIM)).astype(np.float32)

    # --- reference: run the raw ONNX graphs ---
    enc_sess = ort.InferenceSession(ENC, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(DEC, providers=["CPUExecutionProvider"])
    enc_name = enc_sess.get_inputs()[0].name
    tokens_onnx = enc_sess.run(None, {enc_name: enc_in})[0]  # [1,64]
    dec_name = dec_sess.get_inputs()[0].name
    dec_in = np.concatenate([tokens_onnx, proprio], axis=-1).astype(np.float32)  # [1,994]
    assert dec_in.shape[-1] == DEC_ONNX_IN, dec_in.shape
    action_onnx = dec_sess.run(None, {dec_name: dec_in})[0]  # [1,29]
    print(f"[onnx] tokens shape {tokens_onnx.shape}, action shape {action_onnx.shape}")
    print(f"[onnx] action[:6] = {np.round(action_onnx[0, :6], 4)}")

    # --- ours: run the loaded PyTorch net's robot encoder + control decoder ---
    # The g1_onnx_repack robot encoder consumes the 640-D branch; the ONNX encoder
    # consumes the full 1762 obs. We instead validate the CONTROL-DECODER path, which is
    # the action-producing head and is loaded layer-for-layer from the decoder ONNX.
    with torch.no_grad():
        tok_t = torch.from_numpy(tokens_onnx)
        prop_t = torch.from_numpy(proprio)
        # control_decoder input order = token_proprio
        dec_input = torch.cat([tok_t, prop_t], dim=-1)
        action_ours = net.control_decoder(dec_input)
    action_ours_np = action_ours.numpy()
    print(f"[ours] action[:6] = {np.round(action_ours_np[0, :6], 4)}")

    max_abs = float(np.max(np.abs(action_ours_np - action_onnx)))
    print(f"[compare] control_decoder max|Δaction| (ours vs ONNX) = {max_abs:.3e}")
    ok = max_abs < 1e-3
    print(f"[RESULT] control-decoder weight match: {'PASS' if ok else 'FAIL'} (tol 1e-3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
