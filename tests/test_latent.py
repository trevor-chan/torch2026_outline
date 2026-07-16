from __future__ import annotations

import torch

from flow_interpolation.evaluation.cli import build_parser
from flow_interpolation.evaluation.experiments.latent import (
    decode_from_tau,
    encode_keyframes_to_tau,
    interpolation_time_from_tau,
)
from flow_interpolation.utils.flow import FlowSettings, perturb_to_p_eps


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, conditioning, t):
        del conditioning, t
        return torch.zeros_like(x)


def _flow() -> FlowSettings:
    return FlowSettings(
        data_time=0.001,
        noise_time=0.999,
        ode_steps=100,
        solver="heun",
        encode_batch_size=4,
        decode_batch_size=4,
    )


def test_tau_maps_to_configured_flow_interval() -> None:
    flow = _flow()
    assert interpolation_time_from_tau(flow, 0.0) == flow.data_time
    assert interpolation_time_from_tau(flow, 0.5) == 0.5
    assert interpolation_time_from_tau(flow, 1.0) == flow.noise_time


def test_latent_cli_defaults_tau_to_full_noise_transport() -> None:
    assert build_parser().parse_args(["latent"]).tau == 1.0


def test_tau_zero_interpolates_at_perturbed_data_boundary() -> None:
    samples = torch.randn(3, 1, 2, 2)
    eps_noise = torch.randn(1, 1, 2, 2)
    states, interpolation_time, steps = encode_keyframes_to_tau(
        _ZeroVelocity(),
        samples,
        _flow(),
        torch.device("cpu"),
        tau=0.0,
        eps_noise=eps_noise,
        desc="test",
    )

    assert interpolation_time == _flow().data_time
    assert steps == 0
    torch.testing.assert_close(
        states,
        perturb_to_p_eps(samples, _flow().data_time, eps_noise),
    )


def test_partial_tau_scales_encode_and_decode_steps() -> None:
    samples = torch.randn(3, 1, 2, 2)
    eps_noise = torch.randn(1, 1, 2, 2)
    states, interpolation_time, encode_steps = encode_keyframes_to_tau(
        _ZeroVelocity(),
        samples,
        _flow(),
        torch.device("cpu"),
        tau=0.5,
        eps_noise=eps_noise,
        desc="test",
    )
    decoded, decode_steps = decode_from_tau(
        _ZeroVelocity(),
        states,
        _flow(),
        torch.device("cpu"),
        tau=0.5,
        desc="test",
    )

    assert interpolation_time == 0.5
    assert encode_steps == 50
    assert decode_steps == 50
    torch.testing.assert_close(decoded, states)


def test_tau_one_uses_full_endpoint_and_step_count() -> None:
    samples = torch.randn(3, 1, 2, 2)
    states, interpolation_time, encode_steps = encode_keyframes_to_tau(
        _ZeroVelocity(),
        samples,
        _flow(),
        torch.device("cpu"),
        tau=1.0,
        eps_noise=torch.zeros(1, 1, 2, 2),
        desc="test",
    )
    decoded, decode_steps = decode_from_tau(
        _ZeroVelocity(),
        states,
        _flow(),
        torch.device("cpu"),
        tau=1.0,
        desc="test",
    )

    assert interpolation_time == _flow().noise_time
    assert encode_steps == _flow().ode_steps
    assert decode_steps == _flow().ode_steps
    torch.testing.assert_close(decoded, states)
