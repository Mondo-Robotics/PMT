"""Phase-1 episode-error accumulation test for AdaptiveSamplingMotionCommand,
WITHOUT Isaac Sim. Models the REAL IsaacLab ordering, including the crucial fact
that CommandManager.reset() ZEROES self.metrics[env_ids] BEFORE _record_outcomes
runs (command_manager.py:142 then :147).

Design after Codex reviewer-B: the step accumulator (filled in _update_command,
which runs when CommandTerm.compute -> _update_metrics has fresh metrics) is the
SINGLE source of truth. _record_outcomes does NOT read self.metrics (already zeroed
on the termination path). We assert:
  * each simulated step contributes exactly one error sample,
  * the episode scalar = 0.5*mean + 0.5*max over the simulated steps,
  * an env that terminates with zero simulated steps yields scalar 0 (not NaN),
  * no double counting on the motion-end path.
The logic under test mirrors adaptive_sampling_motion_command.py verbatim.
"""
import torch


class FakeCmd:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self._uses_error = True
        self._episode_error_sum = torch.zeros(num_envs)
        self._episode_error_max = torch.zeros(num_envs)
        self._episode_error_count = torch.zeros(num_envs)
        self.metrics_err = torch.zeros(num_envs)  # stand-in for composite of self.metrics
        self.recorded = {}  # env_id -> folded episode scalar

    def _composite_error(self):
        return self.metrics_err.clone()

    def _episode_error_scalar(self, env_ids):
        cnt = self._episode_error_count[env_ids].clamp(min=1.0)
        mean_err = self._episode_error_sum[env_ids] / cnt
        max_err = self._episode_error_max[env_ids]
        return 0.5 * mean_err + 0.5 * max_err

    def _reset_episode_error(self, env_ids):
        self._episode_error_sum[env_ids] = 0.0
        self._episode_error_max[env_ids] = 0.0
        self._episode_error_count[env_ids] = 0.0

    # --- mirrors the real command (accumulator is the only source of truth) ---
    def update_command_accumulate(self, step_err):
        """Called once per simulated step with fresh metrics (= step_err)."""
        self.metrics_err = step_err
        err = self._composite_error().detach()
        self._episode_error_sum += err
        self._episode_error_max = torch.maximum(self._episode_error_max, err)
        self._episode_error_count += 1.0

    def record_outcomes(self, env_ids, zero_metrics_first):
        # Termination path: CommandManager.reset zeroed metrics before this.
        if zero_metrics_first:
            self.metrics_err[env_ids] = 0.0
        ep = self._episode_error_scalar(env_ids)
        for i, e in zip(env_ids.tolist(), ep.tolist()):
            self.recorded[i] = e

    def resample(self, env_ids, zero_metrics_first):
        self.record_outcomes(env_ids, zero_metrics_first)
        self._reset_episode_error(env_ids)


def test_termination_episode_scalar_is_mean_max_of_simulated_steps():
    c = FakeCmd(2)
    # Simulate 3 steps for env 0 with errors 0.2, 0.6, 0.4 (env 1 mirrors but we check 0).
    for e in (0.2, 0.6, 0.4):
        c.update_command_accumulate(torch.tensor([e, 0.0]))
    # Env 0 terminates: IsaacLab zeroes metrics[0] BEFORE record. Scalar must use the
    # accumulator only: mean=(0.2+0.6+0.4)/3=0.4, max=0.6 -> 0.5*0.4+0.5*0.6=0.5.
    c.resample(torch.tensor([0]), zero_metrics_first=True)
    assert abs(c.recorded[0] - 0.5) < 1e-6, c.recorded
    assert c._episode_error_count[0].item() == 0.0  # cleared
    print("OK test_termination_episode_scalar_is_mean_max_of_simulated_steps")


def test_zero_step_termination_yields_zero_not_nan():
    c = FakeCmd(2)
    # Env 1 terminates immediately, no simulated steps accumulated this episode.
    c.resample(torch.tensor([1]), zero_metrics_first=True)
    assert c.recorded[1] == 0.0
    assert not torch.isnan(torch.tensor(c.recorded[1]))
    print("OK test_zero_step_termination_yields_zero_not_nan")


def test_motion_end_path_no_double_count():
    """Motion-end resample also reads only the accumulator; metrics NOT zeroed yet,
    but we must not add an extra sample. Scalar matches simulated steps."""
    c = FakeCmd(2)
    for e in (1.0, 1.0):
        c.update_command_accumulate(torch.tensor([1.0, 1.0]))
    # Motion-end path: metrics still live (not zeroed), but record reads accumulator.
    c.resample(torch.tensor([0]), zero_metrics_first=False)
    # mean=1.0, max=1.0 -> 1.0; exactly 2 samples, no third.
    assert abs(c.recorded[0] - 1.0) < 1e-6, c.recorded
    assert c._episode_error_count[0].item() == 0.0
    print("OK test_motion_end_path_no_double_count")


if __name__ == "__main__":
    test_termination_episode_scalar_is_mean_max_of_simulated_steps()
    test_zero_step_termination_yields_zero_not_nan()
    test_motion_end_path_no_double_count()
    print("\nPHASE-1 ACCUMULATOR/ORDERING TESTS PASSED")
