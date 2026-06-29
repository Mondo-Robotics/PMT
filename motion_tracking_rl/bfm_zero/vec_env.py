"""IsaacLab ManagerBasedRLEnv -> BFM-Zero dict-obs adapter (Option B).

This adapter wraps a ``gym.make(...)`` IsaacLab env and produces the BFM observation dict
(``state``/``privileged_state``/``last_action``/``history_actor`` + ``time``) directly from the
robot articulation data, plus per-step ``info["aux_rewards"]`` for the auxiliary critic.

It deliberately does NOT use the env's ObservationManager groups or RewardManager terms for the
learning signal (env reward is zero; FB-CPR + aux critic own the objective). The env still owns
simulation, terrain, the streaming mocap command (used as reset/mocap source + expert store), the
action manager (direct joint targets), contact sensors, resets, and time-out termination.

Conventions:
  - IsaacLab body quaternions are wxyz -> converted to xyzw for the (w_last=True) obs builders.
  - Online ``state.base_ang_vel`` uses base-frame ``root_ang_vel_b`` scaled by 0.25 (env parity);
    the expert buffer uses world-frame root ang vel (upstream asymmetry, intentionally preserved).
  - The policy action contract is the original BFM-Zero normalized range [-1, 1]. HumanoidVerse
    converts that to env action units [-5, 5] before applying the G1 joint-position scale. This
    adapter owns the same conversion: replay stores the raw policy action in the runner, while
    ``last_action`` / history ``actions`` store the post-normalized env action.
  - history_actor follows the flatten-before-push timing contract; reset zeroes done envs.
"""

from __future__ import annotations

import numpy as np
import torch

from . import aux_rewards as auxr
from . import obs_math

HISTORY_LEN = 4
ACTION_NORMALIZE_TO = 5.0
# Bodies penalised for undesired contact = everything except the 4 end-effectors.
_END_EFFECTORS = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
_ANKLE_ROLL_JOINTS = ["left_ankle_roll_joint", "right_ankle_roll_joint"]


class BFMZeroVecEnv:
    """Gymnasium-vector-like wrapper exposing BFM dict obs + aux rewards over an IsaacLab env.

    API (mirrors what ``humanoidverse.train.Workspace`` expects):
      - ``num_envs``
      - ``single_observation_space`` / ``single_action_space``
      - ``reset() -> (obs_dict_np, info)``
      - ``step(action_np) -> (obs_dict_np, reward_np, terminated_np, truncated_np, info)``
        where ``info["aux_rewards"]`` is a dict of 8 [N] numpy arrays.
    """

    def __init__(self, env, tracked_body_names: list[str], device: str | None = None):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = torch.device(device or self.unwrapped.device)
        self.num_envs = self.unwrapped.num_envs

        robot = self.unwrapped.scene["robot"]
        self.robot = robot

        # Resolve PRIVILEGED-body indices (all 30 robot bodies, pelvis first) in the canonical
        # order. The original BFM-Zero used the full body set for ``max_local_self``; this must
        # match the expert side, which the streaming command resolves with the SAME name list.
        self._body_idx, resolved = robot.find_bodies(obs_math.PRIVILEGED_BODY_NAMES, preserve_order=True)
        assert list(resolved) == list(obs_math.PRIVILEGED_BODY_NAMES), (resolved, obs_math.PRIVILEGED_BODY_NAMES)
        self._tracked_body_names = list(obs_math.PRIVILEGED_BODY_NAMES)

        # Aux indices.
        self._feet_idx, _ = robot.find_bodies(_FEET, preserve_order=True)
        self._ankle_roll_joint_idx, _ = robot.find_joints(_ANKLE_ROLL_JOINTS, preserve_order=True)
        # Penalised contact bodies = all bodies except end-effectors.
        all_body_names = robot.body_names
        self._penalised_body_idx = [i for i, n in enumerate(all_body_names) if n not in _END_EFFECTORS]

        # Contact sensor.
        self._contact = self.unwrapped.scene.sensors.get("contact_forces", None)

        # Joint-position SOFT limits for aux ``limits_dof_pos`` (BFM uses soft limits): derive from
        # hard limits with BFM's 0.95 soft factor, mirroring reward_bfm_zero.yaml.
        hard = robot.data.joint_pos_limits  # [N, J, 2]
        mid = (hard[..., 0] + hard[..., 1]) / 2
        half_range = (hard[..., 1] - hard[..., 0]) / 2
        self._dof_pos_soft_low = mid - half_range * auxr.SOFT_DOF_POS_LIMIT
        self._dof_pos_soft_high = mid + half_range * auxr.SOFT_DOF_POS_LIMIT
        self._joint_effort_limits = robot.data.joint_effort_limits  # [N, J]

        self.num_joints = robot.data.joint_pos.shape[1]
        assert self.num_joints == obs_math.NUM_JOINTS, (self.num_joints, obs_math.NUM_JOINTS)

        self.history = obs_math.HistoryActorBuffer(self.num_envs, HISTORY_LEN, device=self.device)
        self._last_action = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.num_joints, device=self.device)

        self._gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self._gravity_vec[:, 2] = -1.0

        self._build_spaces()

    # ------------------------------------------------------------------ spaces
    def _build_spaces(self):
        import gymnasium
        from gymnasium import spaces

        def box(d):
            return spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32)

        self.single_observation_space = gymnasium.spaces.Dict(
            {
                "state": box(obs_math.STATE_DIM),
                "privileged_state": box(obs_math.PRIVILEGED_STATE_DIM),
                "last_action": box(obs_math.LAST_ACTION_DIM),
                "history_actor": box(self.history.dim),
                "time": box(1),
            }
        )
        # Official BFM-Zero exposes normalized policy actions in [-1, 1], then scales them to
        # [-5, 5] inside the env before the robot-specific joint target scale is applied.
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

    # ------------------------------------------------------------------ obs build
    def _raw_proprio(self):
        d = self.robot.data
        joint_pos_rel = d.joint_pos - d.default_joint_pos
        joint_vel = d.joint_vel
        proj_grav = d.projected_gravity_b
        base_ang_vel = d.root_ang_vel_b
        return joint_pos_rel, joint_vel, proj_grav, base_ang_vel

    def _build_obs(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        joint_pos_rel, joint_vel, proj_grav, base_ang_vel = self._raw_proprio()

        state = obs_math.build_state_from_raw(joint_pos_rel, joint_vel, proj_grav, base_ang_vel)

        body_pos_w = d.body_pos_w[:, self._body_idx]
        body_quat_xyzw = obs_math.wxyz_to_xyzw(d.body_quat_w[:, self._body_idx])
        body_lin_vel_w = d.body_lin_vel_w[:, self._body_idx]
        body_ang_vel_w = d.body_ang_vel_w[:, self._body_idx]
        # 30 real bodies + 1 extended head body -> 463-dim privileged_state (official contract).
        privileged = obs_math.build_privileged_state_with_extend(
            body_pos_w, body_quat_xyzw, body_lin_vel_w, body_ang_vel_w
        )

        # history_actor reflects PAST frames (flatten before pushing current frame).
        history_actor = self.history.flatten()
        time = self.unwrapped.episode_length_buf.unsqueeze(-1).float()
        return {
            "state": state,
            "privileged_state": privileged,
            "last_action": self._last_action.clone(),
            "history_actor": history_actor,
            "time": time,
        }

    def _push_history(self):
        """Append the CURRENT proprio frame (raw values, scaled inside the buffer)."""
        joint_pos_rel, joint_vel, proj_grav, base_ang_vel = self._raw_proprio()
        frames = {
            "actions": self._last_action,
            "base_ang_vel": base_ang_vel,
            "dof_pos": joint_pos_rel,
            "dof_vel": joint_vel,
            "projected_gravity": proj_grav,
        }
        self.history.push(frames, apply_scale=True)

    # ------------------------------------------------------------------ aux rewards
    def _compute_aux_rewards(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        torques = d.applied_torque
        dof_pos = d.joint_pos

        out = {}
        out["penalty_torques"] = auxr.aux_penalty_torques(torques)
        out["penalty_action_rate"] = auxr.aux_penalty_action_rate(self._prev_action, self._last_action)
        out["limits_dof_pos"] = auxr.aux_limits_dof_pos(
            dof_pos, self._dof_pos_soft_low, self._dof_pos_soft_high
        )
        out["limits_torque"] = auxr.aux_limits_torque(torques, self._joint_effort_limits)

        if self._contact is not None:
            net_forces = self._contact.data.net_forces_w  # [N, B, 3]
            penalised = net_forces[:, self._penalised_body_idx, :]
            out["penalty_undesired_contact"] = auxr.aux_penalty_undesired_contact(penalised)
            feet_forces = net_forces[:, self._feet_idx, :]
            feet_contact = (feet_forces[:, :, 2] > auxr.CONTACT_FORCE_THRESHOLD).float()
            out["penalty_slippage"] = auxr.aux_penalty_slippage(
                d.body_lin_vel_w[:, self._feet_idx], feet_forces
            )
        else:
            zeros = torch.zeros(self.num_envs, device=self.device)
            out["penalty_undesired_contact"] = zeros.clone()
            out["penalty_slippage"] = zeros.clone()
            feet_contact = torch.zeros(self.num_envs, 2, device=self.device)

        left_q = obs_math.wxyz_to_xyzw(d.body_quat_w[:, self._feet_idx[0]])
        right_q = obs_math.wxyz_to_xyzw(d.body_quat_w[:, self._feet_idx[1]])
        out["penalty_feet_ori"] = auxr.aux_penalty_feet_ori(left_q, right_q, self._gravity_vec, feet_contact)

        lr = dof_pos[:, self._ankle_roll_joint_idx[0:1]]
        rr = dof_pos[:, self._ankle_roll_joint_idx[1:2]]
        out["penalty_ankle_roll"] = auxr.aux_penalty_ankle_roll(lr, rr)
        return out

    # ------------------------------------------------------------------ gym API
    def _to_numpy(self, obs: dict[str, torch.Tensor]):
        return {k: v.detach().cpu().numpy() for k, v in obs.items()}

    def reset(self):
        _, info = self.env.reset()
        self.history.reset()
        self._last_action.zero_()
        self._prev_action.zero_()
        # current frame is pushed AFTER building the (zero-history) obs, matching BFM timing.
        obs = self._build_obs()
        self._push_history()
        return self._to_numpy(obs), info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_t = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        else:
            action_t = action.to(self.device).float()

        # Match HumanoidVerse BaseTask's normalize_action=True path:
        # raw policy action [-1, 1] -> env action [-5, 5]. The runner keeps the raw policy action
        # in replay; the env/control path and last_action observations use env action units.
        env_action_t = action_t.clamp(-1.0, 1.0) * ACTION_NORMALIZE_TO

        self._prev_action = self._last_action.clone()
        self._last_action = env_action_t

        _, _env_reward, terminated, time_outs, extras = self.env.step(env_action_t)
        terminated = terminated.bool()
        time_outs = time_outs.bool()

        # Aux rewards for the transition that just occurred. IsaacLab auto-resets done envs INSIDE
        # step(), so for done envs these reflect the post-reset state — but the runner drops
        # current-done transitions (their next_obs is the reset state, not the true s'), so the
        # bogus done-row aux values are never used for learning. Non-done rows are exact.
        aux = self._compute_aux_rewards()

        # IsaacLab returns POST-reset obs for done envs (same-step autoreset). Zero the history and
        # last_action of done envs BEFORE building the returned obs so each new episode's first obs
        # has zero history / zero last_action (matching BFM-Zero reset semantics).
        done = torch.logical_or(terminated, time_outs)
        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.history.reset(done_ids)
            self._last_action[done_ids] = 0.0
            self._prev_action[done_ids] = 0.0

        # Build next-state obs (history reflects frames up to the previous step), then push the
        # current frame for the next step. For done envs the history was just zeroed, so their
        # returned obs has empty history and the pushed frame is the new episode's first frame.
        obs = self._build_obs()
        self._push_history()

        reward = torch.zeros(self.num_envs, device=self.device)  # FB-CPR owns learning signal
        info = dict(extras) if extras is not None else {}
        info["aux_rewards"] = {k: v.detach().cpu().numpy() for k, v in aux.items()}

        return (
            self._to_numpy(obs),
            reward.detach().cpu().numpy(),
            terminated.detach().cpu().numpy(),
            time_outs.detach().cpu().numpy(),
            info,
        )

    @property
    def motion_command(self):
        """The streaming motion command (expert mocap source)."""
        return self.unwrapped.command_manager.get_term("motion")

    def close(self):
        self.env.close()
