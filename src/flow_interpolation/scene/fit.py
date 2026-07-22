"""Optimization loop fitting a scene model to sparse k-space observations.

This is per-scene optimization, not supervised training: there is one "dataset"
(the measurements of a single sequence) and the fitted parameters are the answer
rather than a model to be deployed. The loop is therefore separate from
``training/engine.py``, which is built around dataloader-driven minibatch
training of a generative model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from flow_interpolation.kspace.dataset import DynamicKSpaceData
from flow_interpolation.scene.binning import BinSchedule, bin_window
from flow_interpolation.scene.losses import kspace_consistency_loss, spatial_tv, temporal_tv
from flow_interpolation.scene.models import SceneModel
from flow_interpolation.utils.metrics import image_metrics


@dataclass
class SceneFitter:
    """Fit ``model`` to ``data`` under a progressive temporal binning schedule."""

    model: SceneModel
    data: DynamicKSpaceData
    optimizer: torch.optim.Optimizer
    schedule: BinSchedule
    device: torch.device
    max_steps: int = 20_000
    batch_size: int = 8
    spatial_tv_weight: float = 0.0
    temporal_tv_weight: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    log_interval: int = 200
    eval_interval: int = 2_000
    callbacks: list[Callable[[int, "SceneFitter"], None]] = field(default_factory=list)
    writer: Optional[SummaryWriter] = None
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    seed: int = 0
    _generator: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def _sample_centers(self) -> torch.Tensor:
        """Pick the bin-center frames for one step.

        With a temporal-TV penalty active the batch has to be a consecutive run
        of frames for the finite difference to mean anything; otherwise centers
        are drawn independently for better coverage per step.
        """
        num_frames = self.data.num_frames
        count = min(self.batch_size, num_frames)
        if self.temporal_tv_weight > 0.0:
            start = int(
                torch.randint(
                    0, max(num_frames - count + 1, 1), (1,), generator=self._generator
                ).item()
            )
            centers = torch.arange(start, start + count)
        else:
            centers = torch.randperm(num_frames, generator=self._generator)[:count]
        return centers.to(self.device)

    def step(self, step_index: int) -> dict[str, float]:
        half_width = self.schedule.half_width_at(step_index)
        centers = self._sample_centers()
        window = bin_window(centers, half_width, self.data.num_frames)

        rendered = self.model.render(self.data.times.to(self.device)[centers])
        consistency = kspace_consistency_loss(
            rendered,
            self.data.kspace[window],
            self.data.masks[window],
        )

        loss = consistency
        metrics = {
            "consistency": consistency.item(),
            "bin_width": float(2 * half_width + 1),
        }
        if self.spatial_tv_weight > 0.0:
            penalty = spatial_tv(rendered)
            loss = loss + self.spatial_tv_weight * penalty
            metrics["spatial_tv"] = penalty.item()
        if self.temporal_tv_weight > 0.0:
            penalty = temporal_tv(rendered)
            loss = loss + self.temporal_tv_weight * penalty
            metrics["temporal_tv"] = penalty.item()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
            metrics["grad_norm"] = float(grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics["loss"] = loss.item()
        return metrics

    @torch.no_grad()
    def render_sequence(self, times: Optional[torch.Tensor] = None, chunk: int = 32) -> torch.Tensor:
        """Render the fitted scene at ``times`` (default: the observation times)."""
        was_training = self.model.training
        self.model.eval()
        try:
            if times is None:
                times = self.data.times
            times = times.to(self.device)
            frames = [
                self.model.render(times[start : start + chunk]).cpu()
                for start in range(0, times.shape[0], chunk)
            ]
        finally:
            self.model.train(was_training)
        return torch.cat(frames, dim=0)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Compare the rendered sequence against the held-out ground truth."""
        rendered = self.render_sequence().clamp(0.0, 1.0)
        return image_metrics(rendered, self.data.frames.cpu())

    def __call__(self) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = {}
        for step_index in range(self.max_steps):
            metrics = self.step(step_index)
            for name, value in metrics.items():
                running[name] = running.get(name, 0.0) + value

            step = step_index + 1
            if step % self.log_interval == 0:
                averaged = {name: value / self.log_interval for name, value in running.items()}
                running = {}
                print(
                    f"Step {step}, loss: {averaged['loss']:.5f}, "
                    f"consistency: {averaged['consistency']:.5f}, "
                    f"bin width: {averaged['bin_width']:.1f}"
                )
                if self.writer is not None:
                    for name, value in averaged.items():
                        self.writer.add_scalar(f"fit/{name}", value, step)
                    self.writer.flush()

            if self.eval_interval > 0 and step % self.eval_interval == 0:
                evaluation = self.evaluate()
                print(f"  Reconstruction PSNR: {evaluation['psnr_db']:.2f} dB")
                if self.writer is not None:
                    for name, value in evaluation.items():
                        self.writer.add_scalar(f"reconstruction/{name}", value, step)
                    self.writer.flush()

            for callback in self.callbacks:
                callback(step, self)

        return self.evaluate()


def save_scene(model: SceneModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved scene model to {path}")
