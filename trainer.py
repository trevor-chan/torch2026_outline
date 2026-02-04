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
    log_interval: int = 1000
    profile_memory: bool = True
    ema: Optional[EMA] = None  # EMA instance, if None no EMA updates will be done
    peak_flops: float = 91.1e12  # RTX 6000 ADA peak FLOPS (91.1 TFLOPS)
    flops_per_batch: Optional[float] = None  # Will be computed when needed
    
    @torch.compile
    def train_step(self, data: torch.Tensor, conditioning: torch.Tensor):
        """
        Perform a training step on a batch of data.
        """
        self.optimizer.zero_grad()
        
        data, conditioning = data.to(self.device), conditioning.to(self.device)
        loss = self.criterion(self.model, data, conditioning)
        
        loss.backward()
        self.optimizer.step()
        
        
        # Update EMA weights if available
        if self.ema is not None:
            self.ema.update()
        
        # Update memory stats after each step if profiling is enabled
        if self.profile_memory:
            self.last_memory_stats = get_memory_usage(self.device)
        
        return loss.item()
    
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
        
        # Estimate FLOPs on first batch
        try:
            first_batch = next(iter(self.dataloader))
            self.flops_per_batch = estimate_flops(self.model, *first_batch, device=self.device)
        except Exception as e:
            print(f"Failed to estimate FLOPs: {e}")
            self.flops_per_batch = 0
        
        while True:
            # Run callbacks
            for callback in self.callbacks:
                callback(step, self.val_dataloader)
            
            # Get data batch
            try:
                data, conditioning = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                data, conditioning = next(data_iter)
            
            # Perform training step
            loss = self.train_step(data, conditioning)
            
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
                
                # Build log message
                log_message = (f"Step {step}, Loss: {loss:.4f}, "
                              f"Speed: {iterations_per_second:.2f} it/s, "
                              f"[MFU: {mfu:.2f}%]")
                
                # Add memory stats if profiling is enabled
                if self.profile_memory:
                    log_message += f", Memory: {self.last_memory_stats}"
                
                print(log_message)
                
                # Reset counters
                steps_since_log = 0
                last_log_time = current_time