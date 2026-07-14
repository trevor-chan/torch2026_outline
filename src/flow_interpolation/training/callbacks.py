"""Validation, sampling, and checkpoint callbacks for flow training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flow_interpolation.training.checkpoints import save_training_checkpoint
from flow_interpolation.utils.images import denormalize_image, reconstruct_image, save_image_batch
from flow_interpolation.utils.training import EMA


def unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        data = batch[0]
        conditioning = batch[1] if len(batch) > 1 else None
        return data, conditioning
    return batch, None


@dataclass
class ValidationCallback:
    model: nn.Module
    criterion: Callable
    device: torch.device
    num_iterations: int = 100
    call_every: int = 1_000
    amp_dtype: Optional[torch.dtype] = torch.bfloat16
    writer: Optional[SummaryWriter] = None
    log_enabled: bool = True

    @torch.compiler.disable()
    def __call__(self, training_step: int, dataloader: DataLoader | None = None) -> float | None:
        if self.call_every <= 0 or training_step % self.call_every != 0 or dataloader is None:
            return None

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        data_iter = iter(dataloader)
        try:
            with torch.no_grad():
                for _ in range(self.num_iterations):
                    try:
                        data, conditioning = unpack_batch(next(data_iter))
                    except StopIteration:
                        data_iter = iter(dataloader)
                        data, conditioning = unpack_batch(next(data_iter))
                    data = data.to(self.device)
                    if conditioning is not None:
                        conditioning = conditioning.to(self.device)
                    with torch.autocast(
                        self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.amp_dtype is not None,
                    ):
                        loss = self.criterion(self.model, data, conditioning)
                    total_loss += loss.item()
        finally:
            self.model.train(was_training)

        average_loss = total_loss / self.num_iterations
        if self.log_enabled:
            print(f"Validation Loss: {average_loss:.4f}")
        if self.writer is not None:
            self.writer.add_scalar("validation/loss", average_loss, training_step)
            self.writer.flush()
        return average_loss


@dataclass
class RectifiedFlowSampler:
    model: nn.Module
    device: torch.device
    num_steps: int = 32
    call_every: int = 1_000
    output_dir: str | Path = "outputs"
    writer: Optional[SummaryWriter] = None
    tensorboard_tag: str = "samples/model"
    write_enabled: bool = True

    @torch.compiler.disable()
    def __call__(
        self,
        training_step: int,
        dataloader: DataLoader | None = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        if self.call_every <= 0 or training_step % self.call_every != 0 or dataloader is None:
            return

        data, conditioning = unpack_batch(next(iter(dataloader)))
        data = data.to(self.device)
        batch_size = data.shape[0]
        x_t = torch.randn_like(data)
        if labels is None:
            labels = conditioning.to(self.device) if conditioning is not None else None
        time_steps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=self.device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for t_curr, t_next in zip(time_steps[:-1], time_steps[1:]):
                    velocity = self.model(x_t, labels, t_curr.expand(batch_size))
                    x_t += (t_next - t_curr) * velocity
        finally:
            self.model.train(was_training)

        images = self._samples_to_images(x_t)
        if self.write_enabled:
            output_dir = Path(self.output_dir)
            output_path = output_dir / f"images_{training_step:09d}.png"
            save_image_batch(
                images=images,
                filepath=str(output_path),
                nrow=min(8, batch_size),
                normalize=False,
                padding=2,
            )
            print(f"Generated images saved to {output_path}")
        if self.writer is not None:
            self.writer.add_images(self.tensorboard_tag, images.clamp(0.0, 1.0), training_step)
            self.writer.flush()

    @staticmethod
    def _samples_to_images(samples: torch.Tensor) -> torch.Tensor:
        if samples.dim() == 4:
            return samples.clamp(0.0, 1.0)
        return denormalize_image(reconstruct_image(samples))


@dataclass
class EMAFlowSampler(RectifiedFlowSampler):
    ema: Optional[EMA] = None
    output_dir: str | Path = "ema_outputs"
    tensorboard_tag: str = "samples/ema"

    @torch.compiler.disable()
    def __call__(
        self,
        training_step: int,
        dataloader: DataLoader | None = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        if self.call_every <= 0 or training_step % self.call_every != 0 or self.ema is None:
            return
        self.ema.store()
        try:
            super().__call__(training_step, dataloader, labels)
        finally:
            self.ema.restore()


@dataclass
class CheckpointCallback:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    ema: Optional[EMA] = None
    call_every: int = 10_000
    output_dir: str | Path = "outputs/checkpoints"
    enabled: bool = True
    extra: dict = field(default_factory=dict)
    _last_saved_step: Optional[int] = field(default=None, init=False)

    @torch.compiler.disable()
    def __call__(self, training_step: int, dataloader: DataLoader | None = None) -> None:
        del dataloader
        if not self.enabled or self.call_every <= 0 or training_step <= 0:
            return
        if training_step % self.call_every == 0:
            self._save(training_step)

    def _save(self, training_step: int) -> None:
        path = Path(self.output_dir) / f"step_{training_step:09d}.pt"
        save_training_checkpoint(
            path,
            step=training_step,
            model=self.model,
            optimizer=self.optimizer,
            ema=self.ema,
            extra=self.extra,
        )
        self._last_saved_step = training_step

    def on_train_end(self, training_step: int) -> None:
        if self.enabled and training_step > 0 and self._last_saved_step != training_step:
            self._save(training_step)
