from __future__ import annotations

import itertools

import torch

from flow_interpolation.training.coupling import pair_batch_exact_ot
from flow_interpolation.training.losses import RectifiedFlowLoss


def test_exact_ot_uses_data_to_noise_permutation_direction() -> None:
    data = torch.tensor([[0.0], [10.0]])
    noise = torch.tensor([[9.0], [1.0]])

    returned_data, paired_noise, permutation = pair_batch_exact_ot(data, noise)

    torch.testing.assert_close(returned_data, data)
    torch.testing.assert_close(permutation, torch.tensor([1, 0]))
    torch.testing.assert_close(paired_noise, noise[permutation])
    torch.testing.assert_close(paired_noise, torch.tensor([[1.0], [9.0]]))


def test_exact_ot_matches_brute_force_minimum_and_preserves_noise_batch() -> None:
    data = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]])
    noise = torch.tensor([[3.5, 4.0], [0.5, 2.5], [2.0, 0.5], [-0.5, 0.0]])
    _, paired_noise, permutation = pair_batch_exact_ot(data, noise)

    cost = torch.cdist(data, noise).square()
    selected_cost = cost[torch.arange(data.shape[0]), permutation].sum()
    brute_force_costs = [
        cost[torch.arange(data.shape[0]), torch.tensor(candidate)].sum()
        for candidate in itertools.permutations(range(data.shape[0]))
    ]

    torch.testing.assert_close(selected_cost, torch.stack(brute_force_costs).min())
    torch.testing.assert_close(
        paired_noise[paired_noise[:, 0].argsort()],
        noise[noise[:, 0].argsort()],
    )


def test_exact_ot_singleton_is_identity_and_shape_mismatch_is_rejected() -> None:
    data = torch.tensor([[2.0, 3.0]])
    _, paired_noise, permutation = pair_batch_exact_ot(data, torch.tensor([[1.0, 4.0]]))
    torch.testing.assert_close(paired_noise, torch.tensor([[1.0, 4.0]]))
    torch.testing.assert_close(permutation, torch.tensor([0]))

    try:
        pair_batch_exact_ot(torch.randn(2, 3), torch.randn(3, 3))
    except ValueError as error:
        assert "matching shapes" in str(error)
    else:
        raise AssertionError("Expected mismatched batches to be rejected")


class _LinearVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 3)

    def forward(self, x, conditioning, time):
        del conditioning, time
        return self.projection(x)


def test_independent_coupling_reports_identity_cost_metrics() -> None:
    criterion = RectifiedFlowLoss(coupling="independent")
    loss, metrics = criterion.loss_and_metrics(_LinearVelocity(), torch.randn(5, 3))

    torch.testing.assert_close(loss.detach(), metrics[0])
    torch.testing.assert_close(metrics[1], metrics[2])
    torch.testing.assert_close(metrics[3], torch.tensor(1.0))
    torch.testing.assert_close(metrics[4], torch.tensor(1.0))


def test_independent_coupling_preserves_baseline_rng_and_loss() -> None:
    model = _LinearVelocity()
    data = torch.randn(5, 3)
    criterion = RectifiedFlowLoss(coupling="independent")

    torch.manual_seed(123)
    actual = criterion(model, data)

    torch.manual_seed(123)
    time = torch.rand(data.shape[0])
    noise = torch.randn_like(data)
    time_view = time.view(data.shape[0], 1)
    interpolated = torch.lerp(data, noise, time_view)
    expected = (model(interpolated, None, time) - noise + data).square().mean()

    torch.testing.assert_close(actual, expected)


def test_minibatch_ot_reduces_cost_and_flow_loss_remains_differentiable() -> None:
    torch.manual_seed(7)
    model = _LinearVelocity()
    criterion = RectifiedFlowLoss(coupling="minibatch-ot")
    loss, metrics = criterion.loss_and_metrics(model, torch.randn(8, 3))

    assert metrics[2] <= metrics[1] + 1e-6
    assert metrics[3] <= 1.0 + 1e-6
    loss.backward()
    assert model.projection.weight.grad is not None
    assert model.projection.weight.grad.norm().item() > 0.0
