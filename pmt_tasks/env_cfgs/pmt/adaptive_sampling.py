from __future__ import annotations

import os
from typing import List, Optional, Union

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import pmt_tasks.mdp as mdp
from pmt_tasks.env_cfgs.multi_motion_flat import (
    TRACKED_BODY_NAMES,
    MultiMotionCommandsCfg,
    MultiMotionFlatSceneCfg,
)
from pmt_tasks.robots.g1 import G1_CYLINDER_CFG
from pmt_tasks.tracking_env_cfg import TrackingEnvCfg
from pmt_tasks.utils.motion_paths import find_motion_files


@configclass
class PMTAdaptiveSamplingObservationsCfg:
    """Observation groups for the PMT adaptive-sampling flat-plane task."""

    @configclass
    class ProprioCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CommandWindowCfg(ObsGroup):
        command_window = ObsTerm(
            func=mdp.command_window,
            params={"command_name": "motion", "half_window": 10, "stride": 1, "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio)
        command = ObsTerm(
            func=mdp.command_window,
            params={"command_name": "motion", "half_window": 0, "stride": 1, "flatten": True},
        )
        reference_base_height = ObsTerm(
            func=mdp.reference_base_height,
            params={"command_name": "motion"},
        )
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = ProprioCfg()
    proprio = ProprioCfg()
    proprio_history = ProprioHistoryCfg()
    command_window = CommandWindowCfg()
    critic = CriticCfg()


@configclass
class PMTAdaptiveSamplingEnvCfg(TrackingEnvCfg):
    """Standalone PMT adaptive-sampling env with the hybrid streaming command."""

    pmt_motion_paths: Optional[Union[str, List[str]]] = None
    pmt_decimation: int = 4
    pmt_sim_dt: float = 0.005
    pmt_sampler_type: str = "bin_adaptive"
    pmt_storage_mode: str = "hybrid"
    pmt_max_working_set: int = 0
    pmt_num_load_workers: int = 16
    pmt_use_process_pool: bool = False
    pmt_history_length: int = 10

    pmt_offline_prior_path: str = ""
    pmt_offline_prior_strength: float = 0.0
    pmt_hybrid_error_weight: float = 0.0
    pmt_hybrid_failure_weight: float = 1.0
    pmt_hybrid_error_good: float = 0.0
    pmt_hybrid_error_bad: float = 1.0
    pmt_hybrid_retention_ratio: float = 0.0
    pmt_hybrid_topk_motion: int = 1
    pmt_hybrid_topk_motion_weight: float = 0.3
    pmt_hybrid_retention_success_thresh: float = 0.85
    pmt_global_age_ratio: float = 0.0
    pmt_global_age_tau: float = 10.0
    pmt_hybrid_uncertainty_weight: float = 0.0
    pmt_hybrid_uncertainty_gate_lo: float = 0.2
    pmt_hybrid_uncertainty_gate_hi: float = 0.8
    pmt_hybrid_uncertainty_norm: float = 1.0
    pmt_hybrid_hard_buffer_ratio: float = 0.0
    pmt_hybrid_hard_buffer_k: int = 64

    scene = MultiMotionFlatSceneCfg(num_envs=64, env_spacing=0.0)
    observations = PMTAdaptiveSamplingObservationsCfg()
    commands = MultiMotionCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = int(self.pmt_decimation)
        self.sim.dt = float(self.pmt_sim_dt)
        self.sim.render_interval = self.decimation

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 0.0

        self.actions.joint_pos.scale = 1.0
        self.commands.motion.anchor_body = "torso_link"
        self.commands.motion.body_names = TRACKED_BODY_NAMES
        self.commands.motion.sampler_type = str(self.pmt_sampler_type)
        self.commands.motion.update_action_offset_with_ref = True
        self.observations.proprio_history.history_length = int(self.pmt_history_length)
        self.observations.proprio_history.flatten_history_dim = False

        if hasattr(self.terminations, "bad_orientation"):
            self.terminations.bad_orientation.params["limit_angle"] = 1.4
        if hasattr(self.terminations, "base_height"):
            self.terminations.base_height.params["minimum_height"] = 0.35

        if self.pmt_motion_paths:
            discovery = find_motion_files(motion_paths=self.pmt_motion_paths, strict=False)
            motion_files = discovery.files
            if motion_files:
                self.commands.motion.motion_files = motion_files
                print(f"[{type(self).__name__}] Loaded {len(motion_files)} motion file(s)")
            else:
                print(
                    f"[{type(self).__name__}] No motion files found from {self.pmt_motion_paths}; "
                    "motion command will resolve paths later."
                )
        else:
            print(f"[{type(self).__name__}] pmt_motion_paths is empty; deferring motion discovery")

        if self.pmt_storage_mode == "hybrid":
            base = self.commands.motion
            self.commands.motion = mdp.AdaptiveSamplingMotionCommandCfg(
                asset_name=base.asset_name,
                resampling_time_range=base.resampling_time_range,
                debug_vis=base.debug_vis,
                motion_files=base.motion_files,
                anchor_body=base.anchor_body,
                body_names=base.body_names,
                storage_device=base.storage_device,
                use_fp16=True,
                chunk_length=0,
                load_human_motion=False,
                sampler_type="hybrid",
                enable_frame_sampling=base.enable_frame_sampling,
                adaptive_beta=base.adaptive_beta,
                adaptive_alpha=base.adaptive_alpha,
                adaptive_uniform_ratio=base.adaptive_uniform_ratio,
                adaptive_update_interval=base.adaptive_update_interval,
                adaptive_kernel_size=base.adaptive_kernel_size,
                adaptive_kernel_lambda=base.adaptive_kernel_lambda,
                motion_fps=base.motion_fps,
                bin_duration=base.bin_duration,
                pose_range=base.pose_range,
                velocity_range=base.velocity_range,
                joint_position_range=base.joint_position_range,
                default_reset_joint_prob=base.default_reset_joint_prob,
                default_reset_joint_lerp_range=base.default_reset_joint_lerp_range,
                default_reset_frame_range=base.default_reset_frame_range,
                default_reset_zero_vel=base.default_reset_zero_vel,
                update_action_offset_with_ref=True,
                action_term_name=base.action_term_name,
                max_working_set=int(self.pmt_max_working_set),
                num_load_workers=int(self.pmt_num_load_workers),
                use_process_pool=bool(self.pmt_use_process_pool),
                offline_prior_path=str(self.pmt_offline_prior_path),
                offline_prior_strength=float(self.pmt_offline_prior_strength),
                hybrid_error_weight=float(self.pmt_hybrid_error_weight),
                hybrid_failure_weight=float(self.pmt_hybrid_failure_weight),
                hybrid_error_good=float(self.pmt_hybrid_error_good),
                hybrid_error_bad=float(self.pmt_hybrid_error_bad),
                hybrid_retention_ratio=float(self.pmt_hybrid_retention_ratio),
                hybrid_topk_motion=int(self.pmt_hybrid_topk_motion),
                hybrid_topk_motion_weight=float(self.pmt_hybrid_topk_motion_weight),
                hybrid_retention_success_thresh=float(self.pmt_hybrid_retention_success_thresh),
                global_age_ratio=float(self.pmt_global_age_ratio),
                global_age_tau=float(self.pmt_global_age_tau),
                hybrid_uncertainty_weight=float(self.pmt_hybrid_uncertainty_weight),
                hybrid_uncertainty_gate_lo=float(self.pmt_hybrid_uncertainty_gate_lo),
                hybrid_uncertainty_gate_hi=float(self.pmt_hybrid_uncertainty_gate_hi),
                hybrid_uncertainty_norm=float(self.pmt_hybrid_uncertainty_norm),
                hybrid_hard_buffer_ratio=float(self.pmt_hybrid_hard_buffer_ratio),
                hybrid_hard_buffer_k=int(self.pmt_hybrid_hard_buffer_k),
            )
            print(
                f"[{type(self).__name__}] HYBRID adaptive sampling enabled "
                f"(working_set={self.pmt_max_working_set}, workers={self.pmt_num_load_workers}, "
                f"prior='{self.pmt_offline_prior_path or None}' strength={self.pmt_offline_prior_strength}, "
                f"err_w={self.pmt_hybrid_error_weight}, retention={self.pmt_hybrid_retention_ratio}, "
                f"age={self.pmt_global_age_ratio}, unc_w={self.pmt_hybrid_uncertainty_weight}, "
                f"hardbuf={self.pmt_hybrid_hard_buffer_ratio})"
            )
        elif self.pmt_storage_mode == "streaming":
            base = self.commands.motion
            self.commands.motion = mdp.StreamingMultiMotionCommandV2Cfg(
                asset_name=base.asset_name,
                resampling_time_range=base.resampling_time_range,
                debug_vis=base.debug_vis,
                motion_files=base.motion_files,
                body_names=base.body_names,
                anchor_body=base.anchor_body,
                ref_body_index=base.ref_body_index,
                joint_names=base.joint_names,
                pose_error_threshold=base.pose_error_threshold,
                velocity_error_threshold=base.velocity_error_threshold,
                ignore_base_height=base.ignore_base_height,
                termination_body_names=base.termination_body_names,
                termination_body_poses=base.termination_body_poses,
                use_heading=base.use_heading,
                generator_type=base.generator_type,
                sampler_type=base.sampler_type,
                update_action_offset_with_ref=True,
                load_mode="streaming",
                cache_mode="none",
                storage_mode="streaming",
                preload=False,
                max_working_set=int(self.pmt_max_working_set),
                num_load_workers=int(self.pmt_num_load_workers),
                use_process_pool=bool(self.pmt_use_process_pool),
                max_refill_per_step=1,
                command_name=base.command_name,
                motion_selection=base.motion_selection,
                interval_steps=base.interval_steps,
                storage_device=getattr(base, "storage_device", "cpu"),
                streaming_device=getattr(base, "streaming_device", "cpu"),
                mmap_dir=getattr(base, "mmap_dir", None),
            )
            print(
                f"[{type(self).__name__}] Streaming motion storage enabled "
                f"(max_working_set={self.pmt_max_working_set}, "
                f"workers={self.pmt_num_load_workers}, "
                f"process_pool={self.pmt_use_process_pool})"
            )

        self.commands.motion.update_action_offset_with_ref = True
        if self.pmt_storage_mode not in ("streaming", "hybrid"):
            self.commands.motion.storage_mode = self.pmt_storage_mode
            self.commands.motion.load_mode = "eager"
            self.commands.motion.cache_mode = "memory"
        self.commands.motion.motion_files = [
            os.path.abspath(path) if isinstance(path, str) else path
            for path in self.commands.motion.motion_files
        ]
