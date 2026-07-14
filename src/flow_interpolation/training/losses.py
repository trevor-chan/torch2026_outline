"""Training objectives for rectified-flow models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RectifiedFlowLoss(nn.Module):
    """Rectified-flow matching with optional ordered latent acceleration.

    When acceleration regularization is enabled, ``data`` must have shape
    ``[batch, 3, ...]``. Flow matching is evaluated on the center frame. The
    complete triplet is encoded with a differentiable ODE solve and penalized by
    its raw second finite difference in terminal latent space.
    """

    metric_names = (
        "flow_matching_loss",
        "acceleration_loss",
        "weighted_acceleration_loss",
        "encoded_value_mean",
        "encoded_value_std",
        "encoded_radius_mean",
        "encoded_radius_std",
        "encoded_radius_over_expected",
    )

    def __init__(
        self,
        *,
        acceleration_weight: float = 0.0,
        acceleration_ode_steps: int = 1,
        acceleration_solver: str = "euler",
        data_time: float = 1e-3,
        noise_time: float = 1.0 - 1e-3,
    ) -> None:
        super().__init__()
        if acceleration_weight < 0.0:
            raise ValueError("acceleration_weight must be non-negative")
        if acceleration_ode_steps <= 0:
            raise ValueError("acceleration_ode_steps must be positive")
        if acceleration_solver not in {"euler", "heun"}:
            raise ValueError("acceleration_solver must be 'euler' or 'heun'")
        if not 0.0 <= data_time < noise_time <= 1.0:
            raise ValueError("Expected 0 <= data_time < noise_time <= 1")
        self.acceleration_weight = float(acceleration_weight)
        self.acceleration_ode_steps = int(acceleration_ode_steps)
        self.acceleration_solver = acceleration_solver
        self.data_time = float(data_time)
        self.noise_time = float(noise_time)

    @staticmethod
    def _center_conditioning(
        conditioning: torch.Tensor | None,
        triplet_batch_size: int,
    ) -> torch.Tensor | None:
        if conditioning is None:
            return None
        if conditioning.shape[0] != triplet_batch_size:
            raise ValueError("Conditioning batch size does not match data")
        if conditioning.ndim >= 2 and conditioning.shape[1] == 3:
            return conditioning[:, 1]
        return conditioning

    @staticmethod
    def _triplet_conditioning(
        conditioning: torch.Tensor | None,
        triplet_batch_size: int,
    ) -> torch.Tensor | None:
        if conditioning is None:
            return None
        if conditioning.shape[0] != triplet_batch_size:
            raise ValueError("Conditioning batch size does not match data")
        if conditioning.ndim >= 2 and conditioning.shape[1] == 3:
            return conditioning.flatten(0, 1)
        return conditioning.repeat_interleave(3, dim=0)

    @staticmethod
    def _flow_matching_loss(
        model: nn.Module,
        data: torch.Tensor,
        conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = data.shape[0]
        time = torch.rand(batch_size, device=data.device)
        noise = torch.randn_like(data)
        time_view = time.view(batch_size, *([1] * (data.ndim - 1)))
        interpolated = torch.lerp(data, noise, time_view)
        prediction = model(interpolated, conditioning, time)
        return (prediction - noise + data).square().mean()

    def _encode_triplets(
        self,
        model: nn.Module,
        triplets: torch.Tensor,
        conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = triplets.shape[0]
        shared_boundary_noise = torch.randn(
            batch_size,
            1,
            *triplets.shape[2:],
            device=triplets.device,
            dtype=triplets.dtype,
        )
        state = torch.lerp(triplets, shared_boundary_noise, self.data_time).flatten(0, 1)
        flat_conditioning = self._triplet_conditioning(conditioning, batch_size)
        times = torch.linspace(
            self.data_time,
            self.noise_time,
            self.acceleration_ode_steps + 1,
            device=state.device,
            dtype=state.dtype,
        )
        for current_time, next_time in zip(times[:-1], times[1:]):
            step_size = next_time - current_time
            current_velocity = model(
                state,
                flat_conditioning,
                current_time.expand(state.shape[0]),
            )
            if self.acceleration_solver == "euler":
                state = state + step_size * current_velocity
            else:
                predicted_state = state + step_size * current_velocity
                next_velocity = model(
                    predicted_state,
                    flat_conditioning,
                    next_time.expand(state.shape[0]),
                )
                state = state + 0.5 * step_size * (current_velocity + next_velocity)
        return state.unflatten(0, (batch_size, 3))

    def loss_and_metrics(
        self,
        model: nn.Module,
        data: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.acceleration_weight > 0.0:
            if data.ndim < 3 or data.shape[1] != 3:
                raise ValueError(
                    "Acceleration regularization requires data shaped [batch, 3, ...]"
                )
            center_data = data[:, 1]
            center_conditioning = self._center_conditioning(conditioning, data.shape[0])
        else:
            center_data = data
            center_conditioning = conditioning

        flow_loss = self._flow_matching_loss(model, center_data, center_conditioning)
        if self.acceleration_weight == 0.0:
            zero = flow_loss.new_zeros(())
            metrics = torch.stack((flow_loss, zero, zero, zero, zero, zero, zero, zero))
            return flow_loss, metrics

        encoded = self._encode_triplets(model, data, conditioning).float()
        acceleration = encoded[:, 2] - 2.0 * encoded[:, 1] + encoded[:, 0]
        acceleration_loss = acceleration.square().mean()
        weighted_acceleration = self.acceleration_weight * acceleration_loss
        encoded_flat = encoded.flatten(start_dim=2)
        radii = encoded_flat.norm(dim=2)
        expected_radius = math.sqrt(encoded_flat.shape[2])
        total_loss = flow_loss + weighted_acceleration
        metrics = torch.stack(
            (
                flow_loss.float(),
                acceleration_loss,
                weighted_acceleration,
                encoded.mean(),
                encoded.std(unbiased=False),
                radii.mean(),
                radii.std(unbiased=False),
                radii.mean() / expected_radius,
            )
        )
        return total_loss, metrics

    def forward(
        self,
        model: nn.Module,
        data: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_and_metrics(model, data, conditioning)[0]
