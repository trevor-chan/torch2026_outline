import sys
import types

import torch

# The evaluation folder is intended to live beside the repository's transformer.py
# and dataset.py. Provide minimal stubs so these isolated utility tests can run in
# the artifact directory as well.
transformer_stub = types.ModuleType("transformer")
transformer_stub.TransformerDiffusionModel = torch.nn.Module
sys.modules.setdefault("transformer", transformer_stub)

dataset_stub = types.ModuleType("dataset")


class _DatasetStub:
    pass


dataset_stub.BouncingBallVideoDataset = _DatasetStub
sys.modules.setdefault("dataset", dataset_stub)

from eval_endpoint_bridge import (  # noqa: E402
    _batch_radius_slerp,
    _variance_preserving_residual_mix,
    bridge_envelope,
    sample_bridge_innovation,
)


def test_bridge_envelope_vanishes_at_keyframes() -> None:
    observed = torch.tensor([0, 4, 8])
    for kind in ("sine", "brownian", "quadratic"):
        envelope = bridge_envelope(
            9,
            observed,
            kind=kind,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert torch.equal(envelope[observed], torch.zeros(3))
        assert float(envelope.max()) <= 1.0 + 1e-6
        assert float(envelope[2]) > 0.99


def test_piecewise_innovation_shape() -> None:
    generator = torch.Generator().manual_seed(7)
    innovation = sample_bridge_innovation(
        mode="piecewise-slerp",
        num_frames=9,
        frame_shape=(3, 4, 4),
        observed_indices=torch.tensor([0, 4, 8]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=generator,
        slerp_mode="iscs",
    )
    assert innovation.shape == (9, 3, 4, 4)
    assert torch.isfinite(innovation).all()


def test_batch_slerp_endpoints() -> None:
    generator = torch.Generator().manual_seed(11)
    current = torch.randn((5, 3, 4, 4), generator=generator)
    target = torch.randn((5, 3, 4, 4), generator=generator)
    assert torch.allclose(_batch_radius_slerp(current, target, 0.0), current, atol=1e-5)
    assert torch.allclose(_batch_radius_slerp(current, target, 1.0), target, atol=1e-5)


def test_residual_mix_preserves_center_where_amplitude_is_zero() -> None:
    center = torch.randn(5, 3, 4, 4)
    innovation = torch.randn_like(center)
    amplitude = torch.tensor([0.0, 0.1, 0.5, 1.0, 0.0])
    mixed = _variance_preserving_residual_mix(center, innovation, amplitude)
    assert torch.equal(mixed[0], center[0])
    assert torch.equal(mixed[-1], center[-1])
    assert torch.equal(mixed[3], innovation[3])
