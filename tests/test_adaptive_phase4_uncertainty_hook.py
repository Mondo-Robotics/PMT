"""Phase-4 receive_policy_uncertainty shape/NaN-defensiveness (Isaac-free).

Mirrors the command's hook logic verbatim on a tiny stub to prove malformed pushes
are ignored (never crash, never poison the accumulator) and valid pushes normalize.
"""
import torch


class HookStub:
    def __init__(self, num_envs, unc_norm=0.5):
        self.num_envs = num_envs
        self._uses_uncertainty = True
        self._unc_norm = unc_norm or 1.0
        self.device = torch.device("cpu")
        self._latest_uncertainty = torch.zeros(num_envs)

    # verbatim copy of AdaptiveSamplingMotionCommand.receive_policy_uncertainty body
    def receive_policy_uncertainty(self, uncertainty):
        if not self._uses_uncertainty:
            return
        if not isinstance(uncertainty, torch.Tensor):
            return
        u = uncertainty.detach()
        if u.dim() == 0:
            return
        if u.dim() > 1:
            u = u.reshape(u.shape[0], -1).mean(dim=-1)
        if u.shape[0] != self.num_envs:
            return
        u = u.to(self.device, dtype=torch.float32)
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        self._latest_uncertainty = (u / self._unc_norm).clamp(0.0, 1.0)


def test_valid_2d_push_normalizes_and_clamps():
    h = HookStub(4, unc_norm=0.5)
    h.receive_policy_uncertainty(torch.full((4, 29), 0.5))  # /0.5 = 1.0 -> clamp 1.0
    assert torch.allclose(h._latest_uncertainty, torch.ones(4)), h._latest_uncertainty
    print("OK test_valid_2d_push_normalizes_and_clamps")


def test_3d_push_reduced_per_env():
    h = HookStub(3, unc_norm=1.0)
    h.receive_policy_uncertainty(torch.full((3, 4, 2), 0.2))
    assert torch.allclose(h._latest_uncertainty, torch.full((3,), 0.2), atol=1e-6)
    print("OK test_3d_push_reduced_per_env")


def test_scalar_push_ignored():
    h = HookStub(4); h._latest_uncertainty = torch.full((4,), 0.3)
    h.receive_policy_uncertainty(torch.tensor(0.9))  # scalar -> ignore
    assert torch.allclose(h._latest_uncertainty, torch.full((4,), 0.3))
    print("OK test_scalar_push_ignored")


def test_wrong_env_count_ignored():
    h = HookStub(4); h._latest_uncertainty = torch.full((4,), 0.3)
    h.receive_policy_uncertainty(torch.full((7, 29), 0.5))  # 7 != 4 -> ignore
    assert torch.allclose(h._latest_uncertainty, torch.full((4,), 0.3))
    print("OK test_wrong_env_count_ignored")


def test_nan_push_sanitized():
    h = HookStub(4, unc_norm=1.0)
    u = torch.full((4, 5), 0.5); u[0, 0] = float("nan"); u[1, 1] = float("inf")
    h.receive_policy_uncertainty(u)
    assert torch.isfinite(h._latest_uncertainty).all()
    print("OK test_nan_push_sanitized")


def test_non_tensor_ignored():
    h = HookStub(4); h._latest_uncertainty = torch.full((4,), 0.3)
    h.receive_policy_uncertainty(None)
    h.receive_policy_uncertainty([1, 2, 3])
    assert torch.allclose(h._latest_uncertainty, torch.full((4,), 0.3))
    print("OK test_non_tensor_ignored")


def test_disabled_is_noop():
    h = HookStub(4); h._uses_uncertainty = False; h._latest_uncertainty = torch.zeros(4)
    h.receive_policy_uncertainty(torch.full((4, 5), 0.5))
    assert torch.allclose(h._latest_uncertainty, torch.zeros(4))
    print("OK test_disabled_is_noop")


if __name__ == "__main__":
    test_valid_2d_push_normalizes_and_clamps()
    test_3d_push_reduced_per_env()
    test_scalar_push_ignored()
    test_wrong_env_count_ignored()
    test_nan_push_sanitized()
    test_non_tensor_ignored()
    test_disabled_is_noop()
    print("\nPHASE-4 UNCERTAINTY HOOK TESTS PASSED")
