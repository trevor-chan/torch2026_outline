"""Training objectives for rectified-flow models."""

from __future__ import annotations

import torch
import torch.nn as nn

from flow_interpolation.training.coupling import pair_batch_exact_ot


class RectifiedFlowLoss(nn.Module):
    """Rectified-flow matching with configurable source-target coupling."""

    metric_names = (
        "flow_matching_loss",
        "coupling_independent_cost_mse",
        "coupling_paired_cost_mse",
        "coupling_cost_ratio",
        "coupling_fixed_point_fraction",
    )

    def __init__(self, *, coupling: str = "independent") -> None:
        super().__init__()
        if coupling not in {"independent", "minibatch-ot"}:
            raise ValueError(f"Unknown flow coupling: {coupling}")
        self.coupling = coupling

    def _pair_noise(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.coupling == "minibatch-ot":
            _, paired_noise, permutation = pair_batch_exact_ot(data, noise)
            return paired_noise, permutation
        permutation = torch.arange(data.shape[0], device=data.device)
        return noise, permutation

    @staticmethod
    def _coupling_metrics(
        data: torch.Tensor,
        noise: torch.Tensor,
        paired_noise: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        data_float = data.detach().float()
        independent_cost = (data_float - noise.detach().float()).square().mean()
        paired_cost = (data_float - paired_noise.detach().float()).square().mean()
        cost_ratio = paired_cost / independent_cost.clamp_min(1e-12)
        identity = torch.arange(permutation.shape[0], device=permutation.device)
        fixed_point_fraction = (permutation == identity).float().mean()
        return torch.stack(
            (
                independent_cost,
                paired_cost,
                cost_ratio,
                fixed_point_fraction,
            )
        )

    def loss_and_metrics(
        self,
        model: nn.Module,
        data: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = data.shape[0]
        # Preserve the baseline objective's RNG order when coupling is independent.
        time = torch.rand(batch_size, device=data.device)
        noise = torch.randn_like(data)
        paired_noise, permutation = self._pair_noise(data, noise)

        time_view = time.view(batch_size, *([1] * (data.ndim - 1)))
        interpolated = torch.lerp(data, paired_noise, time_view)
        target_velocity = paired_noise - data
        predicted_velocity = model(interpolated, conditioning, time)
        flow_loss = (predicted_velocity - target_velocity).square().mean()
        coupling_metrics = self._coupling_metrics(
            data,
            noise,
            paired_noise,
            permutation,
        )
        return flow_loss, torch.cat((flow_loss.detach().float().reshape(1), coupling_metrics))

    def forward(
        self,
        model: nn.Module,
        data: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_and_metrics(model, data, conditioning)[0]
