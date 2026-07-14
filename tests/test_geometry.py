import torch

from flow_interpolation.evaluation.geometry import interpolate_keyframes, slerp_pair


def test_slerp_endpoints():
    a = torch.randn(3, 4, 4)
    b = torch.randn(3, 4, 4)
    path = slerp_pair(a, b, torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(path[0], a)
    torch.testing.assert_close(path[-1], b)


def test_path_lengths_and_keyframes():
    keyframes = torch.randn(4, 3, 4, 4)
    samples_per_segment = 5
    for method in ("lerp", "slerp", "squad"):
        path = interpolate_keyframes(keyframes, samples_per_segment, method)
        assert path.shape[0] == (keyframes.shape[0] - 1) * samples_per_segment + 1
        for index in range(keyframes.shape[0]):
            torch.testing.assert_close(path[index * samples_per_segment], keyframes[index], atol=2e-5, rtol=2e-5)
