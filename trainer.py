import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict
from torch.utils.data import DataLoader
from utils import get_memory_usage, format_memory_stats, EMA, estimate_flops, calculate_mfu


@dataclass
class Trainer:
    """
    A trainer class that handles the training of a PyTorch model.
    """
    model: nn.Module
    dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device
    callbacks: List[Callable] = field(default_factory=list)
    log_interval: int = 100_000
    profile_memory: bool = True
    ema: Optional[EMA] = None  # EMA instance, if None no EMA updates will be done
    peak_flops: float = 91.1e12  # RTX 6000 ADA peak FLOPS (91.1 TFLOPS)
    flops_per_batch: Optional[float] = None  # Will be computed when needed
    max_steps: Optional[int] = None
    amp_dtype: Optional[torch.dtype] = torch.bfloat16  # autocast dtype; None disables mixed precision
    grad_clip_norm: Optional[float] = 1.0  # max gradient norm; None disables clipping

    @torch.compile
    def train_step(self, data: torch.Tensor, conditioning: Optional[torch.Tensor] = None):
        """
        Perform a training step on a batch of data.
        """
        self.optimizer.zero_grad()

        data = data.to(self.device)
        if conditioning is not None:
            conditioning = conditioning.to(self.device)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            loss = self.criterion(self.model, data, conditioning)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip_norm if self.grad_clip_norm is not None else float("inf"),
        )
        if isinstance(grad_norm, torch.distributed.tensor.DTensor):
            grad_norm = grad_norm.full_tensor()
        self.optimizer.step()


        # Update EMA weights if available
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
    
    def __call__(self):
        """
        Run the training loop.
        """
        self.model.train()
        
        data_iter = iter(self.dataloader)
        
        step = 0
        
        # Tracking for iterations per second
        start_time = time.time()
        last_log_time = start_time
        steps_since_log = 0

        # Gradient norm tracking between logs (kept on device to avoid syncs)
        grad_norm_sum = None
        grad_norm_max = None
        
        # Estimate FLOPs on first batch
        try:
            data, conditioning = self._unpack_batch(next(iter(self.dataloader)))
            self.flops_per_batch = estimate_flops(self.model, data, conditioning, device=self.device)
        except Exception as e:
            print(f"Failed to estimate FLOPs: {e}")
            self.flops_per_batch = 0
        
        while self.max_steps is None or step < self.max_steps:
            # Run callbacks
            for callback in self.callbacks:
                callback(step, self.val_dataloader)
            
            # Get data batch
            try:
                data, conditioning = self._unpack_batch(next(data_iter))
            except StopIteration:
                data_iter = iter(self.dataloader)
                data, conditioning = self._unpack_batch(next(data_iter))
            
            # Perform training step
            loss, grad_norm = self.train_step(data, conditioning)

            if grad_norm_sum is None:
                grad_norm_sum = grad_norm.clone()
                grad_norm_max = grad_norm.clone()
            else:
                grad_norm_sum += grad_norm
                torch.maximum(grad_norm_max, grad_norm, out=grad_norm_max)

            # Update step counter and iterations tracking
            step += 1
            steps_since_log += 1
            
            # Log periodically
            if step % self.log_interval == 0:
                # Calculate time stats
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                iterations_per_second = steps_since_log / elapsed_time
                avg_step_time = elapsed_time / steps_since_log
                
                # Calculate MFU
                mfu = calculate_mfu(self.flops_per_batch, avg_step_time, self.peak_flops)
                
                # Build log message (grad norms are pre-clipping)
                log_message = (f"Step {step}, Loss: {loss.item():.4f}, "
                              f"Grad norm: {grad_norm_sum.item() / steps_since_log:.3f} avg / {grad_norm_max.item():.3f} max, "
                              f"Speed: {iterations_per_second:.2f} it/s, "
                              f"[MFU: {mfu:.2f}%]")

                # Add memory stats if profiling is enabled
                if self.profile_memory:
                    log_message += f", Memory: {get_memory_usage(self.device)}"
                
                print(log_message)
                
                # Reset counters
                steps_since_log = 0
                last_log_time = current_time
                grad_norm_sum = None
                grad_norm_max = None
