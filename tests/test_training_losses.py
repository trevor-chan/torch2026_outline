from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from flow_interpolation.data import OrderedTripletDataset
from flow_interpolation.training.callbacks import center_frame_batch
from flow_interpolation.training.losses import RectifiedFlowLoss


class _ScaledVelocity(torch.nn.Module):
    def __init__(self, scale: float = 0.25) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, x, conditioning, time):
        del conditioning, time
        return self.scale * x


def test_ordered_triplet_dataset_preserves_stride() -> None:
    base = TensorDataset(torch.arange(8, dtype=torch.float32).unsqueeze(1))
    triplets = OrderedTripletDataset(base, frame_stride=2)

    assert len(triplets) == 4
    torch.testing.assert_close(
        triplets[1][0].squeeze(1),
        torch.tensor([1.0, 3.0, 5.0]),
    )


def test_sampler_batch_selection_uses_triplet_center() -> None:
    data = torch.randn(2, 3, 1, 2, 2)
    conditioning = torch.randn(2, 3, 4)

    center_data, center_conditioning = center_frame_batch(data, conditioning)

    torch.testing.assert_close(center_data, data[:, 1])
    torch.testing.assert_close(center_conditioning, conditioning[:, 1])


def test_baseline_flow_loss_still_accepts_independent_frames() -> None:
    criterion = RectifiedFlowLoss()
    loss, metrics = criterion.loss_and_metrics(
        _ScaledVelocity(),
        torch.randn(4, 1, 2, 2),
    )

    torch.testing.assert_close(loss, metrics[0])
    torch.testing.assert_close(metrics[1:], torch.zeros(7))


def test_linear_triplet_has_zero_acceleration_under_linear_encoder() -> None:
    starts = torch.tensor([1.0, 4.0]).view(2, 1, 1, 1, 1)
    increments = torch.tensor([0.5, -0.25]).view(2, 1, 1, 1, 1)
    coordinates = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1, 1)
    triplets = starts + increments * coordinates
    criterion = RectifiedFlowLoss(acceleration_weight=0.1)

    _, metrics = criterion.loss_and_metrics(_ScaledVelocity(), triplets)

    torch.testing.assert_close(metrics[1], torch.tensor(0.0), atol=1e-12, rtol=0.0)


def test_curved_triplet_acceleration_is_weighted_and_differentiable() -> None:
    triplets = torch.tensor([0.0, 1.0, 4.0]).view(1, 3, 1, 1, 1)
    model = _ScaledVelocity()
    criterion = RectifiedFlowLoss(
        acceleration_weight=0.2,
        acceleration_ode_steps=2,
        acceleration_solver="heun",
    )

    total, metrics = criterion.loss_and_metrics(model, triplets)
    torch.testing.assert_close(total, metrics[0] + metrics[2])
    torch.testing.assert_close(metrics[2], 0.2 * metrics[1])
    assert metrics[1].item() > 0.0

    metrics[1].backward()
    assert model.scale.grad is not None
    assert model.scale.grad.abs().item() > 0.0


def test_acceleration_loss_requires_ordered_triplets() -> None:
    criterion = RectifiedFlowLoss(acceleration_weight=0.1)
    try:
        criterion(_ScaledVelocity(), torch.randn(4, 1, 2, 2))
    except ValueError as error:
        assert "[batch, 3, ...]" in str(error)
    else:
        raise AssertionError("Expected independent frames to be rejected")
