"""
MPPI (Model Predictive Path Integral) foot swing trajectory optimizer.

Sampling-based foot swing planner. Uses cubic-spline-parameterized
control points with a pre-computed basis matrix for fast batch
expansion.

Usage:
    from stair_mppi.mppi_foot import MppiFootOptimizer, MppiFootParams
    opt = MppiFootOptimizer(MppiFootParams())
    traj_x, traj_y, traj_z = opt.optimize_swing(
        liftoff_pos, landing_pos, swing_duration, ref_traj, ref_times, terrain
    )

Also exposes quaternion helpers (``extract_yaw_from_quat``,
``slerp_quaternion``) used throughout the planner stack.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Quaternion utilities (used by mppi_foot_planner_smooth and other planners)
# ---------------------------------------------------------------------------

def extract_yaw_from_quat(q_wxyz: np.ndarray) -> float:
    """Extract yaw angle from a [w,x,y,z] quaternion."""
    w, x, y, z = q_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def slerp_quaternion(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray,
                     t: np.ndarray) -> np.ndarray:
    """SLERP between two [w,x,y,z] quaternions. Returns (N, 4) in [w,x,y,z].

    Aligns q1 to the same hemisphere as q0 (flips if dot < 0) to ensure
    shortest-arc interpolation — avoids unexpected 360 degree foot rotations.
    """
    t = np.asarray(t, dtype=np.float64)
    q0 = np.asarray(q0_wxyz, dtype=np.float64)
    q1 = np.asarray(q1_wxyz, dtype=np.float64)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    rots = Rotation.from_quat(np.stack([q0_xyzw, q1_xyzw]))
    slerp_obj = Slerp([0.0, 1.0], rots)
    interp = slerp_obj(np.clip(t, 0.0, 1.0))
    q_xyzw = interp.as_quat()
    return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MppiFootParams:
    """MPPI foot planner 的参数。"""
    n_knots: int = 0          # 内部控制点数（不含首尾边界点）
    n_samples: int = 128        # 每次迭代的采样数
    n_iterations: int = 10      # MPPI 迭代次数
    n_dense: int = 100          # 密集评估点数
    temperature: float = 0.1    # softmax 温度（越小越贪心）
    noise_std_xy: float = 0.02  # x/y 通道噪声标准差 (m)
    noise_std_z: float = 0.04   # z 通道噪声标准差 (m)
    w_track: float = 10.0       # 参考轨迹跟踪权重
    w_smooth: float = 10.0       # 平滑度权重（二阶差分）
    w_terrain: float = 10000.0   # 地形穿透惩罚权重
    w_ceiling: float = 500.0    # 高度上界惩罚权重
    h_clearance: float = 0.03   # 地形间距裕量 (m)
    h_contact_z: float = 0.0  # midfoot 在平地支撑时的 z 高度
    ceiling_margin: float = 0.3 # 允许超过起落点的最大高度 (m)
    seed: int = 42              # 随机种子
    # Phase-aware cost terms (default 0 = disabled, no behavior change)
    w_stance_ground: float = 0.0        # stance 阶段惩罚脚离地
    w_swing_clearance_reward: float = 0.0  # swing 阶段奖励离地间距


# ---------------------------------------------------------------------------
# Lightweight trajectory wrapper (.eval(t) sampler over dense values)
# ---------------------------------------------------------------------------

class EvalWrapper:
    """包装密集轨迹数组，提供 .eval(t) 接口对任意时刻做线性插值。

    内部存储 (n_dense,) 的时间和值数组，eval() 用线性插值求任意时刻的值。
    """

    def __init__(self, times: np.ndarray, values: np.ndarray):
        self.times = np.asarray(times, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)

    def eval(self, t: np.ndarray) -> np.ndarray:
        """在任意时间点上求值（线性插值）。"""
        return np.interp(t, self.times, self.values)


# ---------------------------------------------------------------------------
# Pre-computed spline basis matrix
# ---------------------------------------------------------------------------

def _build_basis_matrix(knot_times: np.ndarray,
                        dense_times: np.ndarray) -> np.ndarray:
    """预计算 clamped cubic spline 的 basis matrix。

    对于固定的 knot 时间位置，clamped cubic spline 是 knot 值的线性函数：
        dense_values = B @ knot_values
    其中 B: (n_dense, n_knots_total)。

    通过逐个设置 unit knot 值来提取每一列。

    Args:
        knot_times: (n_knots_total,) 包含首尾边界点的 knot 时间。
        dense_times: (n_dense,) 密集评估时间。

    Returns:
        B: (n_dense, n_knots_total) basis matrix。
    """
    n_knots = len(knot_times)
    n_dense = len(dense_times)
    B = np.zeros((n_dense, n_knots), dtype=np.float64)
    for k in range(n_knots):
        unit = np.zeros(n_knots, dtype=np.float64)
        unit[k] = 1.0
        cs = CubicSpline(knot_times, unit, bc_type='clamped')
        B[:, k] = cs(dense_times)
    return B


# ---------------------------------------------------------------------------
# MPPI Foot Swing Optimizer
# ---------------------------------------------------------------------------

class MppiFootOptimizer:
    """用 MPPI 采样优化单脚摆动轨迹。

    返回三个 EvalWrapper (x/y/z)，可在 mppi_foot_planner_smooth.py 的
    precompute_mppi_foot 流程中直接使用。
    """

    def __init__(self, params: Optional[MppiFootParams] = None):
        self.params = params or MppiFootParams()
        self.rng = np.random.default_rng(self.params.seed)

    def optimize_swing(
        self,
        liftoff_pos: np.ndarray,
        landing_pos: np.ndarray,
        swing_duration: float,
        ref_traj: np.ndarray,
        ref_times: np.ndarray,
        terrain,
        v_start: Optional[np.ndarray] = None,
        v_end: Optional[np.ndarray] = None,
        phase_stance_mask: Optional[np.ndarray] = None,
        phase_swing_envelope: Optional[np.ndarray] = None,
    ) -> Tuple[EvalWrapper, EvalWrapper, EvalWrapper]:
        """优化一次摆动的 xyz 轨迹。

        返回 (traj_x, traj_y, traj_z) — 三个 EvalWrapper 对象。
        v_start/v_end 被忽略（MPPI 通过 clamped spline 边界条件隐式处理）。

        Optional phase-aware parameters (default None = disabled):
            phase_stance_mask: (n_dense,) bool mask for stance frames along dense_times.
            phase_swing_envelope: (n_dense,) float [0,1] swing intensity along dense_times.

        Returns:
            (traj_x, traj_y, traj_z) — 三个 EvalWrapper 对象，各自有 .eval(t) 方法。
        """
        p = self.params
        n_knots_inner = p.n_knots
        n_knots_total = n_knots_inner + 2  # 含首尾

        # ---- 时间设置 ----
        dense_times = np.linspace(0.0, swing_duration, p.n_dense)
        knot_times = np.linspace(0.0, swing_duration, n_knots_total)

        # ---- 预计算 basis matrix ----
        B_full = _build_basis_matrix(knot_times, dense_times)
        # 分离边界列和内部列
        B_start = B_full[:, 0]       # (n_dense,)
        B_end = B_full[:, -1]        # (n_dense,)
        B_inner = B_full[:, 1:-1]    # (n_dense, n_knots_inner)

        # 边界贡献: (n_dense, 3) — 首尾固定点对轨迹的贡献
        boundary_contrib = (B_start[:, None] * liftoff_pos[None, :]
                            + B_end[:, None] * landing_pos[None, :])

        # ---- 参考轨迹插值到 dense_times ----
        ref_interp = np.column_stack([
            np.interp(dense_times, ref_times, ref_traj[:, d])
            for d in range(3)
        ])  # (n_dense, 3)

        # ---- clearance envelope: sin²(πφ)，端点为 0，中间最大 ----
        phi = dense_times / max(swing_duration, 1e-12)
        swing_envelope = np.sin(np.pi * phi) ** 2
        clearance = p.h_clearance * 2.0 * swing_envelope  # (n_dense,)

        # ---- 高度上界 ----
        z_ceiling = max(liftoff_pos[2], landing_pos[2]) + p.ceiling_margin

        # ---- 初始化 mu: 线性插值内部点 ----
        mu = np.zeros((n_knots_inner, 3), dtype=np.float64)
        for i in range(n_knots_inner):
            alpha = (i + 1) / (n_knots_inner + 1)
            mu[i] = (1 - alpha) * liftoff_pos + alpha * landing_pos

        # ---- 噪声标准差 per channel ----
        noise_std = np.array([p.noise_std_xy, p.noise_std_xy, p.noise_std_z],
                             dtype=np.float64)

        # ---- MPPI 迭代 ----
        for _iter in range(p.n_iterations):
            # 采样: (n_samples, n_knots_inner, 3)
            noise = self.rng.normal(size=(p.n_samples, n_knots_inner, 3))
            noise *= noise_std[None, None, :]
            samples = mu[None, :, :] + noise
            samples[0] = mu  # 第 0 个样本 = 当前均值（无噪声）

            # 展开为密集轨迹: (n_samples, n_dense, 3)
            # trajs = boundary_contrib + B_inner @ samples_per_channel
            trajs = (boundary_contrib[None, :, :]
                     + np.einsum('hk,ikd->ihd', B_inner, samples))

            # 地形查询: (n_samples, n_dense)
            terrain_z = np.empty((p.n_samples, p.n_dense), dtype=np.float64)
            for i in range(p.n_samples):
                terrain_z[i] = terrain.height_batch(trajs[i, :, 0],
                                                    trajs[i, :, 1])

            # 计算 cost
            costs = self._cost_batch(
                trajs, terrain_z, ref_interp, clearance, z_ceiling,
                phase_stance_mask=phase_stance_mask,
                phase_swing_envelope=phase_swing_envelope,
            )

            # softmax 加权
            beta = costs.min()
            weights = np.exp(-(costs - beta) / p.temperature)
            w_sum = weights.sum()
            if w_sum < 1e-30:
                weights = np.ones(p.n_samples) / p.n_samples
            else:
                weights /= w_sum

            # 更新 mu
            mu = np.einsum('i,ijk->jk', weights, samples)

        # ---- 最终轨迹 ----
        final_traj = boundary_contrib + B_inner @ mu  # (n_dense, 3)

        return (
            EvalWrapper(dense_times, final_traj[:, 0]),
            EvalWrapper(dense_times, final_traj[:, 1]),
            EvalWrapper(dense_times, final_traj[:, 2]),
        )

    def _cost_batch(
        self,
        trajs: np.ndarray,         # (K, N, 3)
        terrain_z: np.ndarray,     # (K, N)
        ref_interp: np.ndarray,    # (N, 3)
        clearance: np.ndarray,     # (N,)
        z_ceiling: float,
        phase_stance_mask: Optional[np.ndarray] = None,   # (N,) bool
        phase_swing_envelope: Optional[np.ndarray] = None, # (N,) float [0,1]
    ) -> np.ndarray:
        """批量计算所有样本的 cost。

        Args:
            trajs: (n_samples, n_dense, 3) 采样轨迹。
            terrain_z: (n_samples, n_dense) 每条轨迹 xy 处的地形高度。
            ref_interp: (n_dense, 3) 参考轨迹（插值到 dense_times）。
            clearance: (n_dense,) clearance envelope。
            z_ceiling: z 方向高度上限。
            phase_stance_mask: (n_dense,) bool — True = stance phase。
            phase_swing_envelope: (n_dense,) float — swing 相位强度 [0,1]。

        Returns:
            costs: (n_samples,) 每条轨迹的总 cost。
        """
        p = self.params
        K = trajs.shape[0]
        costs = np.zeros(K, dtype=np.float64)

        # 1. 跟踪 cost: ||traj - ref||²
        track_err = trajs - ref_interp[None, :, :]  # (K, N, 3)
        costs += p.w_track * np.sum(track_err ** 2, axis=(1, 2))

        # 2. 平滑 cost: 二阶差分
        if trajs.shape[1] > 2:
            d2 = np.diff(trajs, n=2, axis=1)  # (K, N-2, 3)
            costs += p.w_smooth * np.sum(d2 ** 2, axis=(1, 2))

        # 3. z 方向地形穿透
        floor_z = terrain_z + p.h_contact_z + clearance[None, :]  # (K, N)
        pen_z = np.maximum(floor_z - trajs[:, :, 2], 0.0)        # (K, N)
        costs += p.w_terrain * np.sum(pen_z ** 2, axis=1)

        # 4. 高度上界
        over_z = np.maximum(trajs[:, :, 2] - z_ceiling, 0.0)     # (K, N)
        costs += p.w_ceiling * np.sum(over_z ** 2, axis=1)

        # 5. Stance ground penalty: penalize foot leaving ground during stance
        if p.w_stance_ground > 0.0 and phase_stance_mask is not None:
            ground_z = terrain_z + p.h_contact_z  # (K, N)
            stance_err = (trajs[:, :, 2] - ground_z) * phase_stance_mask[None, :]  # (K, N)
            costs += p.w_stance_ground * np.sum(stance_err ** 2, axis=1)

        # 6. Swing clearance reward: reward clearance above terrain during swing
        if p.w_swing_clearance_reward > 0.0 and phase_swing_envelope is not None:
            ground_z = terrain_z + p.h_contact_z  # (K, N)
            swing_clearance = (trajs[:, :, 2] - ground_z) * phase_swing_envelope[None, :]  # (K, N)
            # Negative cost = reward for being above ground
            costs -= p.w_swing_clearance_reward * np.sum(
                np.maximum(swing_clearance, 0.0), axis=1
            )

        return costs


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MPPI Foot Planner Self-Test")
    print("=" * 60)

    # Test 1: EvalWrapper
    print("\n[1] EvalWrapper interpolation...")
    t = np.linspace(0, 1, 10)
    v = np.sin(t)
    ew = EvalWrapper(t, v)
    t_query = np.array([0.0, 0.5, 1.0])
    result = ew.eval(t_query)
    expected = np.interp(t_query, t, v)
    assert np.allclose(result, expected), f"Mismatch: {result} vs {expected}"
    print("  PASS")

    # Test 2: Basis matrix
    print("\n[2] Basis matrix reproduces CubicSpline...")
    knot_t = np.linspace(0, 1, 5)
    dense_t = np.linspace(0, 1, 50)
    B = _build_basis_matrix(knot_t, dense_t)
    knot_vals = np.array([0.0, 0.3, 0.8, 0.5, 1.0])
    dense_basis = B @ knot_vals
    cs = CubicSpline(knot_t, knot_vals, bc_type='clamped')
    dense_cs = cs(dense_t)
    max_err = np.max(np.abs(dense_basis - dense_cs))
    assert max_err < 1e-10, f"Basis matrix error: {max_err}"
    print(f"  Max error: {max_err:.2e} — PASS")

    # Test 3: Flat ground optimization
    print("\n[3] Flat-ground MPPI optimization...")

    class FlatTerrain:
        def height_at(self, x, y=0.0): return 0.0
        def height_batch(self, x, y=None): return np.zeros_like(np.asarray(x))

    h_cz = 0.049
    params = MppiFootParams(n_samples=64, n_iterations=8, n_knots=4)
    opt = MppiFootOptimizer(params)
    liftoff = np.array([0.0, 0.0, h_cz])
    landing = np.array([0.4, 0.0, h_cz])
    duration = 0.5
    ref_t = np.linspace(0, duration, 30)
    ref_z = h_cz + 0.05 * (1.0 - np.cos(2 * np.pi * ref_t / duration))
    ref_x = np.linspace(0.0, 0.4, 30)
    ref_traj = np.column_stack([ref_x, np.zeros(30), ref_z])

    tx, ty, tz = opt.optimize_swing(
        liftoff, landing, duration, ref_traj, ref_t, FlatTerrain()
    )
    z_out = tz.eval(ref_t)
    max_track_err = np.max(np.abs(z_out - ref_z))
    print(f"  Max z tracking error: {max_track_err * 1000:.2f} mm")
    print(f"  Boundary: z(0)={tz.eval(np.array([0.0]))[0]:.4f} "
          f"(target={liftoff[2]:.4f}), "
          f"z(T)={tz.eval(np.array([duration]))[0]:.4f} "
          f"(target={landing[2]:.4f})")
    print(f"  {'PASS' if max_track_err < 0.02 else 'WARN: tracking > 20mm'}")

    # Test 4: Step terrain
    print("\n[4] Step terrain MPPI optimization...")
    from stair_mppi.terrain import StairTerrain, StairStep
    step_terrain = StairTerrain(
        steps=[StairStep(x_lo=0.15, x_hi=0.5, top_z=0.1, name="step_0")],
        half_width_y=1.0,
    )
    landing_on_step = np.array([0.3, 0.0, 0.1 + h_cz])
    params_step = MppiFootParams(
        n_samples=256, n_iterations=15, n_knots=5,
        noise_std_z=0.06, temperature=0.05,
    )
    opt_step = MppiFootOptimizer(params_step)
    tx2, ty2, tz2 = opt_step.optimize_swing(
        liftoff, landing_on_step, duration, ref_traj, ref_t, step_terrain
    )
    z_out2 = tz2.eval(ref_t)
    x_out2 = tx2.eval(ref_t)
    terrain_along = step_terrain.height_batch(x_out2)
    max_pen = np.max(np.maximum(terrain_along + h_cz - z_out2, 0))
    print(f"  Max penetration: {max_pen * 1000:.2f} mm")
    print(f"  Landing z: {z_out2[-1]:.4f} (target: {landing_on_step[2]:.4f})")
    print(f"  {'PASS' if max_pen < 0.005 else 'WARN: penetration > 5mm'}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
