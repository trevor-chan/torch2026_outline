"""Implicit representations of a scene as a function of space and time.

Every model implements ``render(times) -> [B, C, H, W]``: it evaluates itself on
the full pixel grid at each requested time. The forward operator needs a dense
image to FFT, so there is no benefit to the ray-sampling style of evaluation
used in view synthesis, and rendering the whole grid keeps the k-space loss a
single batched transform.

Times are continuous in ``[0, 1]``. Nothing about the interface assumes the
query times coincide with observation times, which is the point: the fitted
scene can be rendered at instants where nothing was measured.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _output_head(hidden_dim: int, channels: int) -> nn.Linear:
    """Final decoder layer, zero-initialized so rendering starts at zero.

    The output is deliberately unbounded. A sigmoid head is the obvious way to
    respect the ``[0, 1]`` range of the images, but this scene is mostly dark
    background, so the decoder drives its logits far negative on the first few
    hundred steps, saturates, and the whole model dies at exactly zero with no
    gradient left to recover. Data consistency is what should constrain the
    range; clamping happens at display time only.
    """
    layer = nn.Linear(hidden_dim, channels)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class SceneModel(nn.Module):
    """Base class holding the pixel grid and the render contract."""

    def __init__(self, *, height: int, width: int, channels: int = 3) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        # Pixel centers on [-1, 1] in both axes, matching grid_sample's
        # align_corners=False convention.
        ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height * 2.0 - 1.0
        xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_y", grid_y, persistent=False)
        self.register_buffer("grid_x", grid_x, persistent=False)

    def _query_coordinates(self, times: torch.Tensor) -> torch.Tensor:
        """Expand ``times`` into ``[B, H, W, 3]`` coordinates ``(x, y, t)``.

        Spatial coordinates land in ``[-1, 1]``; time is rescaled from ``[0, 1]``
        to ``[-1, 1]`` so all three axes share a scale.
        """
        if times.ndim != 1:
            raise ValueError(f"Expected times of shape [B], got {tuple(times.shape)}")
        batch = times.shape[0]
        grid_x = self.grid_x.expand(batch, -1, -1)
        grid_y = self.grid_y.expand(batch, -1, -1)
        grid_t = (times.view(batch, 1, 1) * 2.0 - 1.0).expand(batch, self.height, self.width)
        return torch.stack((grid_x, grid_y, grid_t), dim=-1)

    def render(self, times: torch.Tensor) -> torch.Tensor:
        """Render the scene at each time; returns ``[B, C, H, W]``.

        Values are unbounded reals, nominally in ``[0, 1]``. Clamp for display,
        not before the loss.
        """
        raise NotImplementedError

    def parameter_groups(self, lr: float) -> list[dict]:
        """Optimizer parameter groups. Subclasses may override to retune."""
        return [{"params": list(self.parameters()), "lr": lr}]

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        return self.render(times)


class FourierFeatureScene(SceneModel):
    """Coordinate MLP on random Fourier features of ``(x, y, t)``.

    Space and time get separate frequency scales. Tying them would force one
    bandwidth choice to govern both how sharp the image can be and how fast the
    scene may change, which confounds exactly the trade-off under study. The
    smoothness of this model is set entirely by ``space_scale`` and
    ``time_scale``, making it the cleaner control against which to judge what
    binning contributes.
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        channels: int = 3,
        num_features: int = 128,
        space_scale: float = 8.0,
        time_scale: float = 4.0,
        hidden_dim: int = 256,
        num_layers: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__(height=height, width=width, channels=channels)
        if num_features <= 0:
            raise ValueError("num_features must be positive")

        generator = torch.Generator().manual_seed(seed)
        frequencies = torch.randn(3, num_features, generator=generator)
        frequencies[:2] *= space_scale
        frequencies[2] *= time_scale
        # Fixed, not learned: the feature bank defines the model's smoothness
        # prior, and letting it drift makes that prior a moving target.
        self.register_buffer("frequencies", frequencies, persistent=True)

        layers: list[nn.Module] = [nn.Linear(2 * num_features, hidden_dim), nn.GELU()]
        for _ in range(max(num_layers - 2, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(_output_head(hidden_dim, channels))
        self.mlp = nn.Sequential(*layers)

    def render(self, times: torch.Tensor) -> torch.Tensor:
        coordinates = self._query_coordinates(times)
        projected = 2.0 * math.pi * (coordinates @ self.frequencies)
        features = torch.cat((projected.sin(), projected.cos()), dim=-1)
        return self.mlp(features).permute(0, 3, 1, 2)


class KPlaneScene(SceneModel):
    """Factorized feature planes over ``(x, y)``, ``(x, t)``, and ``(y, t)``.

    The standard K-planes decomposition: features from the three planes are
    multiplied together and decoded by a small MLP. Multiplication rather than
    concatenation is what lets a spatial pattern be modulated by time without
    storing a full 3D grid.

    Note for the binning experiment: the low-rank structure is itself a temporal
    smoother, so this model has a built-in bias toward temporally coherent
    solutions. Whatever effect binning has here is on top of that, which is why
    :class:`FourierFeatureScene` is worth running as a control.
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        channels: int = 3,
        feature_dim: int = 32,
        resolutions: Sequence[int] = (32, 64),
        time_resolution: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        init_std: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(height=height, width=width, channels=channels)
        if not resolutions:
            raise ValueError("resolutions must contain at least one entry")

        self.resolutions = tuple(int(resolution) for resolution in resolutions)
        self.feature_dim = feature_dim
        self.planes = nn.ParameterList()
        generator = torch.Generator().manual_seed(seed)
        for resolution in self.resolutions:
            # Scale the temporal axis with the spatial one so a multiscale stack
            # refines space and time together.
            time_size = max(2, int(round(time_resolution * resolution / self.resolutions[0])))
            for shape in (
                (resolution, resolution),  # (y, x)
                (time_size, resolution),  # (t, x)
                (time_size, resolution),  # (t, y)
            ):
                # Mean-one initialization: three planes are multiplied together,
                # so zero-mean init would leave features at O(init_std^3) and
                # starve the decoder of gradient.
                self.planes.append(
                    nn.Parameter(
                        1.0
                        + init_std
                        * torch.randn(1, feature_dim, *shape, generator=generator)
                    )
                )

        in_dim = feature_dim * len(self.resolutions)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(max(num_layers - 2, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(_output_head(hidden_dim, channels))
        self.decoder = nn.Sequential(*layers)

    def parameter_groups(self, lr: float) -> list[dict]:
        """Feature planes take the full learning rate; the decoder takes 1/10.

        The planes carry the scene content and need to move fast; a decoder
        moving at the same rate reinterprets those features underneath them and
        destabilizes the fit.
        """
        return [
            {"params": list(self.planes), "lr": lr},
            {"params": list(self.decoder.parameters()), "lr": lr * 0.1},
        ]

    @staticmethod
    def _sample_plane(plane: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample ``plane`` at ``[N, 2]`` coordinates ordered ``(u, v)``."""
        grid = coordinates.view(1, -1, 1, 2)
        # "border" rather than "zeros": t = 0 and t = 1 land exactly on the
        # plane boundary, where zero padding would halve the features and fade
        # the first and last frames toward the decoder's bias.
        sampled = F.grid_sample(
            plane, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.view(plane.shape[1], -1).transpose(0, 1)

    def render(self, times: torch.Tensor) -> torch.Tensor:
        coordinates = self._query_coordinates(times)
        batch = coordinates.shape[0]
        flat = coordinates.reshape(-1, 3)
        x, y, t = flat[:, 0:1], flat[:, 1:2], flat[:, 2:3]
        # grid_sample takes (u, v) with u indexing the last plane axis.
        pairs = (
            torch.cat((x, y), dim=-1),  # (y, x) plane
            torch.cat((x, t), dim=-1),  # (t, x) plane
            torch.cat((y, t), dim=-1),  # (t, y) plane
        )

        scales = []
        for scale_index in range(len(self.resolutions)):
            product = None
            for plane_index, pair in enumerate(pairs):
                plane = self.planes[3 * scale_index + plane_index]
                features = self._sample_plane(plane, pair)
                product = features if product is None else product * features
            scales.append(product)

        rgb = self.decoder(torch.cat(scales, dim=-1))
        return rgb.view(batch, self.height, self.width, self.channels).permute(0, 3, 1, 2)


SCENE_MODELS: dict[str, Callable[..., SceneModel]] = {
    "fourier-mlp": FourierFeatureScene,
    "kplanes": KPlaneScene,
}


def build_scene_model(name: str, **kwargs) -> SceneModel:
    if name not in SCENE_MODELS:
        raise ValueError(f"Unknown scene model: {name}. Choose from {sorted(SCENE_MODELS)}")
    return SCENE_MODELS[name](**kwargs)
