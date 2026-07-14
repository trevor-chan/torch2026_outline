"""Core flow-matching training engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flow_interpolation.utils.training import (
    EMA,
    calculate_mfu,
    estimate_training_flops,
    format_memory_stats,
    get_memory_stats,
)


@dataclass
class Trainer:
    """Train a velocity model with AMP, clipping, EMA, callbacks, and telemetry."""

    workdir: Path
    model: nn.Module
    dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device
    callbacks: list[Callable] = field(default_factory=list)
    log_interval: int = 1_000
    profile_memory: bool = True
    ema: Optional[EMA] = None
    peak_flops: float = 91.1e12
    flops_per_batch: Optional[float] = None
    max_steps: Optional[int] = None
    amp_dtype: Optional[torch.dtype] = torch.bfloat16
    grad_clip_norm: Optional[float] = 1.0
    writer: Optional[SummaryWriter] = None
    start_step: int = 0
    is_main_process: bool = True
    current_step: int = field(init=False)

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        self.workdir = Path(self.workdir)
        self.current_step = self.start_step

    @torch.compile
    def train_step(
        self,
        data: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad(set_to_none=True)
        data = data.to(self.device)
        if conditioning is not None:
            conditioning = conditioning.to(self.device)

        with torch.autocast(
            self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            loss = self.criterion(self.model, data, conditioning)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip_norm if self.grad_clip_norm is not None else float("inf"),
        )
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update()
        return loss.detach(), grad_norm.detach()

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (tuple, list)):
            data = batch[0]
            conditioning = batch[1] if len(batch) > 1 else None
            return data, conditioning
        return batch, None

    def _estimate_flops(self) -> None:
        if self.flops_per_batch is not None:
            return
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            data, conditioning = self._unpack_batch(next(iter(self.dataloader)))
            self.flops_per_batch = estimate_training_flops(
                self.model,
                data,
                conditioning,
                device=self.device,
                amp_dtype=self.amp_dtype,
            )
        except Exception as error:
            print(f"Could not estimate training FLOPs: {error}")
            self.flops_per_batch = 0.0
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    def _log_metrics(
        self,
        *,
        step: int,
        loss_sum: torch.Tensor,
        grad_norm_sum: torch.Tensor,
        grad_norm_max: torch.Tensor,
        steps: int,
        elapsed: float,
    ) -> None:
        average_step_time = elapsed / steps
        iterations_per_second = steps / elapsed
        flops = self.flops_per_batch or 0.0
        achieved_tflops = flops / average_step_time / 1e12
        mfu = calculate_mfu(flops, average_step_time, self.peak_flops)
        average_loss = loss_sum.item() / steps
        average_grad_norm = grad_norm_sum.item() / steps
        max_grad_norm = grad_norm_max.item()
        learning_rate = self.optimizer.param_groups[0]["lr"]
        memory_stats = get_memory_stats(self.device) if self.profile_memory else None

        if self.is_main_process:
            message = (
                f"Step {step}, Loss: {average_loss:.4f}, "
                f"Grad norm: {average_grad_norm:.3f} avg / {max_grad_norm:.3f} max, "
                f"Speed: {iterations_per_second:.2f} it/s, "
                f"Throughput: {achieved_tflops:.2f} TFLOP/s, MFU: {mfu:.2f}%"
            )
            if memory_stats:
                message += f", Memory: {format_memory_stats(memory_stats)}"
            print(message)

        if self.writer is not None:
            self.writer.add_scalar("train/loss", average_loss, step)
            self.writer.add_scalar("train/grad_norm_mean", average_grad_norm, step)
            self.writer.add_scalar("train/grad_norm_max", max_grad_norm, step)
            self.writer.add_scalar("train/learning_rate", learning_rate, step)
            self.writer.add_scalar("performance/steps_per_second", iterations_per_second, step)
            self.writer.add_scalar("performance/step_time_ms", average_step_time * 1_000.0, step)
            self.writer.add_scalar("performance/achieved_tflops", achieved_tflops, step)
            self.writer.add_scalar("performance/mfu_percent", mfu, step)
            if memory_stats:
                for name, value in memory_stats.items():
                    self.writer.add_scalar(f"memory/{name}", value, step)
            self.writer.flush()

    def __call__(self) -> int:
        """Run until ``max_steps`` and return the completed global step."""
        self.model.train()
        self._estimate_flops()
        data_iter = iter(self.dataloader)
        step = self.start_step
        self.current_step = step
        steps_since_log = 0
        callback_seconds = 0.0
        loss_sum = None
        grad_norm_sum = None
        grad_norm_max = None
        last_log_time = time.perf_counter()

        while self.max_steps is None or step < self.max_steps:
            try:
                data, conditioning = self._unpack_batch(next(data_iter))
            except StopIteration:
                data_iter = iter(self.dataloader)
                data, conditioning = self._unpack_batch(next(data_iter))

            self.model.train()
            loss, grad_norm = self.train_step(data, conditioning)
            step += 1
            self.current_step = step
            steps_since_log += 1
            loss_sum = loss.clone() if loss_sum is None else loss_sum + loss
            grad_norm_sum = grad_norm.clone() if grad_norm_sum is None else grad_norm_sum + grad_norm
            grad_norm_max = (
                grad_norm.clone()
                if grad_norm_max is None
                else torch.maximum(grad_norm_max, grad_norm)
            )

            should_log = step % self.log_interval == 0
            if should_log:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed = max(time.perf_counter() - last_log_time - callback_seconds, 1e-12)
                self._log_metrics(
                    step=step,
                    loss_sum=loss_sum,
                    grad_norm_sum=grad_norm_sum,
                    grad_norm_max=grad_norm_max,
                    steps=steps_since_log,
                    elapsed=elapsed,
                )

            callback_start = time.perf_counter()
            for callback in self.callbacks:
                callback(step, self.val_dataloader)
            callback_elapsed = time.perf_counter() - callback_start

            if should_log:
                steps_since_log = 0
                callback_seconds = 0.0
                loss_sum = None
                grad_norm_sum = None
                grad_norm_max = None
                last_log_time = time.perf_counter()
            else:
                callback_seconds += callback_elapsed

        for callback in self.callbacks:
            on_train_end = getattr(callback, "on_train_end", None)
            if on_train_end is not None:
                on_train_end(step)
        return step
