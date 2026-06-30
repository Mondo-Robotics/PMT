"""Visualize a TCRS motion clip (.npz) on a terrain scene.

Replays an optimized/raw/ghost .npz produced by
`stair_mppi.mppi_foot_planner_smooth` on a MuJoCo scene, reconstructing each
frame's qpos = [root_pos(3), root_quat(4), joint_qpos(29)] from the clip's
`body_pos_w` / `body_quat_w` / `joint_pos` fields (with the npz->MuJoCo joint
reorder), then either:

  * opens the interactive MuJoCo viewer (default), or
  * renders headless to an .mp4 / .gif  (--out, works under MUJOCO_GL=egl).

Usage:
  # interactive (needs a display)
  python -m visualize_npz --npz outputs/my_experiment/optimized/<clip>_optimized.npz

  # headless video
  MUJOCO_GL=egl python -m visualize_npz \
    --npz outputs/my_experiment/optimized/<clip>_optimized.npz \
    --out clip.mp4

The terrain XML defaults to the same scene the planner uses; pass --xml to match
whatever scene the clip was generated on.
"""
from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np

from stair_mppi.minimal_mppi_demo import (
    NPZ_PELVIS,
    build_joint_mapping,
    reorder_to_mujoco,
    update_robot_pose,
)

DEFAULT_XML = "assets/g1/g1_29dof_scene_stairs_ud.xml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="motion clip .npz (optimized/raw/ghost)")
    ap.add_argument("--xml", default=DEFAULT_XML, help="MuJoCo scene XML (must match the clip's terrain)")
    ap.add_argument("--out", default=None, help="if set, render to this .mp4/.gif headless instead of the viewer")
    ap.add_argument("--fps", type=float, default=None, help="override playback fps (default: clip fps)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (viewer only)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    data_npz = np.load(args.npz)
    joint_pos = np.asarray(data_npz["joint_pos"], dtype=np.float64)        # [T, 29]
    body_pos_w = np.asarray(data_npz["body_pos_w"], dtype=np.float64)      # [T, 30, 3]
    body_quat_w = np.asarray(data_npz["body_quat_w"], dtype=np.float64)    # [T, 30, 4] (wxyz)
    fps = float(np.asarray(data_npz["fps"]).reshape(-1)[0])
    if args.fps:
        fps = args.fps
    n_frames = joint_pos.shape[0]
    print(f"[viz] {os.path.basename(args.npz)}: {n_frames} frames @ {fps:.1f} fps")

    model = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
    data = mujoco.MjData(model)
    joint_mapping = build_joint_mapping(model)

    def set_frame(t: int):
        root_pos = body_pos_w[t, NPZ_PELVIS]
        root_quat = body_quat_w[t, NPZ_PELVIS]
        joint_qpos = reorder_to_mujoco(joint_pos[t], joint_mapping)
        update_robot_pose(data, root_pos, root_quat, joint_qpos)
        mujoco.mj_forward(model, data)

    if args.out:
        _render_video(model, data, set_frame, n_frames, fps, args)
    else:
        _run_viewer(model, data, set_frame, n_frames, fps, args)


def _run_viewer(model, data, set_frame, n_frames, fps, args):
    import time

    import mujoco.viewer

    dt = 1.0 / (fps * max(args.speed, 1e-6))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Force every geom group visible. The terrain boxes live on geom group 2,
        # and the viewer can have a group toggled off (then the terrain "disappears"
        # and only the robot shows). Enabling all groups makes the scene robust.
        for g in range(len(viewer.opt.geomgroup)):
            viewer.opt.geomgroup[g] = 1
        t = 0
        while viewer.is_running():
            set_frame(t % n_frames)
            viewer.sync()
            time.sleep(dt)
            t += 1


def _render_video(model, data, set_frame, n_frames, fps, args):
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("--out needs imageio: pip install imageio imageio-ffmpeg")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance *= 1.4

    # Force every geom group visible so the terrain boxes (geom group 2) always render.
    opt = mujoco.MjvOption()
    for g in range(len(opt.geomgroup)):
        opt.geomgroup[g] = 1

    frames = []
    for t in range(n_frames):
        set_frame(t)
        # keep the camera tracking the pelvis
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, cam, scene_option=opt)
        frames.append(renderer.render())
    renderer.close()

    out = args.out
    if out.lower().endswith(".gif"):
        imageio.mimsave(out, frames, fps=fps)
    else:
        imageio.mimsave(out, frames, fps=fps, macro_block_size=None)
    print(f"[viz] wrote {out}  ({n_frames} frames @ {fps:.1f} fps)")


if __name__ == "__main__":
    main()
