from __future__ import annotations

import torch

from flow_interpolation.evaluation.experiments.hybrid import (
    compose_hybrid_state,
    decode_from_time_in_chunks,
    interpolate_images,
)
from flow_interpolation.utils.flow import FlowSettings


class ZeroVelocityModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor | None,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del conditioning, t
        return torch.zeros_like(x)


def test_image_interpolation_preserves_keyframes() -> None:
    keyframes = torch.tensor([0.0, 1.0, 0.25]).view(3, 1, 1, 1)
    for method in ("linear", "smoothstep", "catmull-rom"):
        path = interpolate_images(keyframes, samples_per_segment=4, method=method)
        assert path.shape == (9, 1, 1, 1)
        torch.testing.assert_close(path[0], keyframes[0])
        torch.testing.assert_close(path[4], keyframes[1])
        torch.testing.assert_close(path[8], keyframes[2])


def test_hybrid_composition_endpoints() -> None:
    images = torch.randn(5, 3, 4, 4)
    noise = torch.randn_like(images)
    torch.testing.assert_close(compose_hybrid_state(images, noise, 0.0), images)
    torch.testing.assert_close(compose_hybrid_state(images, noise, 1.0), noise)
    torch.testing.assert_close(
        compose_hybrid_state(images, noise, 0.25),
        0.75 * images + 0.25 * noise,
    )


def test_partial_decode_uses_remaining_interval() -> None:
    states = torch.randn(4, 3, 4, 4)
    flow = FlowSettings(
        data_time=0.001,
        noise_time=0.999,
        ode_steps=100,
        solver="heun",
        encode_batch_size=4,
        decode_batch_size=2,
    )
    decoded, steps = decode_from_time_in_chunks(
        ZeroVelocityModel(),
        states,
        t_start=0.5,
        flow=flow,
        device=torch.device("cpu"),
        desc="test",
    )
    assert steps == 50
    torch.testing.assert_close(decoded, states)
