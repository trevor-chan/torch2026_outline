import sys
import types

import torch

transformer_stub = types.ModuleType("flow_interpolation.models")
transformer_stub.TransformerDiffusionModel = torch.nn.Module
sys.modules.setdefault("flow_interpolation.models", transformer_stub)

dataset_stub = types.ModuleType("flow_interpolation.data")


class _DatasetStub:
    pass


dataset_stub.BouncingBallVideoDataset = _DatasetStub
sys.modules.setdefault("flow_interpolation.data", dataset_stub)

from flow_interpolation.evaluation.common import FlowSettings  # noqa: E402
from flow_interpolation.evaluation.data import CadenceInfo, SequenceData  # noqa: E402
from flow_interpolation.evaluation.roundtrip import (  # noqa: E402
    _depth_target_time,
    _normalized_error_metrics,
    run_roundtrip_evaluation,
)


class ZeroVelocity(torch.nn.Module):
    def forward(self, x, conditioning, t):
        del conditioning, t
        return torch.zeros_like(x)


def _sequence() -> SequenceData:
    generator = torch.Generator().manual_seed(3)
    frames = torch.randn((9, 3, 4, 4), generator=generator)
    observed_indices = torch.tensor([0, 4, 8])
    return SequenceData(
        frames=frames,
        observed_indices=observed_indices,
        observed_frames=frames[observed_indices],
        cadence=CadenceInfo(
            training_frame_dt=0.25,
            high_frame_dt=0.0625,
            requested_ratio=4.0,
            endpoint_stride=4,
            actual_endpoint_dt=0.25,
            endpoint_dt_error=0.0,
            relative_error=0.0,
            rounding_policy="exact",
        ),
        high_rate_color_walk_std=0.05,
        start_index=0,
    )


def test_depth_target_time() -> None:
    flow = FlowSettings(0.001, 0.999, 8, "heun", 4, 4)
    assert abs(_depth_target_time(flow, 1.0) - 0.001) < 1e-12
    assert abs(_depth_target_time(flow, 0.9) - 0.1008) < 1e-12


def test_normalized_metrics_scale_with_std() -> None:
    target = torch.tensor([[-1.0, 1.0]])
    prediction = target + 0.5
    metrics = _normalized_error_metrics(prediction, target)
    assert abs(metrics["normalized_error"]["rmse_over_target_std"] - 0.5) < 1e-6
    assert abs(metrics["normalized_error"]["mse_over_target_variance"] - 0.25) < 1e-6


def test_zero_velocity_roundtrip_suite(tmp_path) -> None:
    payload = run_roundtrip_evaluation(
        model=ZeroVelocity().eval(),
        device=torch.device("cpu"),
        sequence=_sequence(),
        flow=FlowSettings(0.001, 0.999, 4, "heun", 4, 4),
        num_samples=4,
        boundary_noise_mode="shared",
        seed=7,
        output_json=str(tmp_path / "metrics.json"),
        image_depths=[0.9, 1.0],
        batch_sizes=[1, 2, 4],
        step_counts=None,
    )
    assert payload["data_to_noise_to_data"]["cycle_at_data_eps"]["error"]["rmse"] == 0.0
    assert payload["noise_to_data_to_noise"]["cycle"]["error"]["rmse"] == 0.0
    assert payload["encoded_data_latent_to_data_to_latent"]["cycle"]["error"]["rmse"] == 0.0
    for row in payload["image_boundary_sweep"]["rows"]:
        assert row["noise_endpoint_cycle"]["error"]["rmse"] == 0.0
        assert row["partial_state_cycle"]["error"]["rmse"] == 0.0
    for direction in ("data_to_noise_to_data", "noise_to_data_to_noise"):
        for row in payload["batch_consistency"][direction]["rows"]:
            assert row["cycle_endpoint_difference_vs_batch1"]["max_abs"] == 0.0
