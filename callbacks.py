import torch
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List
import torch.nn as nn
from torch.utils.data import DataLoader
from image_utils import reconstruct_image, denormalize_image, one_hot_encode, save_image_batch
import os


def unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        data = batch[0]
        conditioning = batch[1] if len(batch) > 1 else None
        return data, conditioning

    return batch, None

@dataclass
class ValidationCallback:
    """
    A validation callback that evaluates a model on a validation dataset.
    
    Attributes:
        model: The model to evaluate
        criterion: Loss function to use for evaluation
        device: Device to run validation on
        num_iterations: Number of validation batches to evaluate
        call_every: How often to call this callback in training steps
    """
    model: nn.Module
    criterion: Callable
    device: torch.device
    num_iterations: int = 100
    call_every: int = 1000
    
    @torch.compiler.disable()
    def __call__(self, training_step: int, dataloader: DataLoader = None) -> float:
        """
        Run validation loop for the specified number of iterations.
        
        Args:
            training_step: Current training step
            dataloader: DataLoader to use for validation
            
        Returns:
            Average validation loss or None if not called on schedule
        """
        if training_step % self.call_every != 0 or dataloader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        
        # Create iterator for validation data
        data_iter = iter(dataloader)
        
        with torch.no_grad():
            for i in range(self.num_iterations):
                try:
                    data, conditioning = unpack_batch(next(data_iter))
                except StopIteration:
                    # Restart iterator if we run out of data
                    data_iter = iter(dataloader)
                    data, conditioning = unpack_batch(next(data_iter))
                                        
                # Move data to device
                data = data.to(self.device)
                if conditioning is not None:
                    conditioning = conditioning.to(self.device)
                
                # Compute loss
                loss = self.criterion(self.model, data, conditioning)
                total_loss += loss.item()
                
        print(f"Validation Loss: {total_loss / self.num_iterations:.4f}")
        
        # Return average loss
        return total_loss / self.num_iterations


@dataclass
class RectifiedFlowSampler:
    """
    Implements rectified flow sampling to generate images from noise.
    
    Attributes:
        model: The trained model to use for sampling
        num_steps: Number of sampling steps
        device: Device to run sampling on
        call_every: How often to call this callback in training steps
    """
    model: nn.Module
    device: torch.device
    num_steps: int = 32
    call_every: int = 1000
    output_dir: str = "outputs"
    
    @torch.compiler.disable()
    def __call__(
        self, 
        training_step: int,
        dataloader: DataLoader = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Generate images using rectified flow sampling.
        
        Args:
            training_step: Current training step
            dataloader: DataLoader to use for obtaining sample shapes
            labels: Optional tensor of labels to condition on
            
        Returns:
            None, but saves generated images to file
        """
        
        if training_step % self.call_every != 0 or dataloader is None:
            return None
        
        # get a batch of samples from the dataloader
        data, conditioning = unpack_batch(next(iter(dataloader)))
        data = data.to(self.device)
        batch_size = data.shape[0]
        x_t = torch.randn_like(data)
        
        # If no labels provided, use the targets from the batch
        if labels is None:
            labels = conditioning.to(self.device) if conditioning is not None else None
        
        # use a linear time schedule for sampling
        time_steps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=self.device)
        
        # Diffusion sampling loop
        self.model.eval()
        with torch.no_grad():
            for t_curr, t_next in zip(time_steps[:-1], time_steps[1:]):
                # Current time step and delta
                t = t_curr.expand(batch_size)
                
                # Get velocity prediction from model
                v = self.model(x_t, labels, t)
                
                # Update sample
                x_t += (t_next - t_curr) * v
        
        images = self._samples_to_images(x_t)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save images using the new utility function
        output_path = os.path.join(self.output_dir, f"images_{training_step}.png")
        save_image_batch(
            images=images,
            filepath=output_path,
            nrow=min(8, batch_size),  # Maximum of 8 images per row
            normalize=False,  # Already normalized
            padding=2
        )
        
        print(f"Generated images saved to {output_path}")

    @staticmethod
    def _samples_to_images(samples: torch.Tensor) -> torch.Tensor:
        if samples.dim() == 4:
            return samples.clamp(0.0, 1.0)

        return denormalize_image(reconstruct_image(samples))

@dataclass
class EMAFlowSampler(RectifiedFlowSampler):
    """
    Extends RectifiedFlowSampler to use EMA weights for sampling.
    
    Attributes:
        model: The trained model to use for sampling
        ema: The EMA instance containing EMA weights
        num_steps: Number of sampling steps
        device: Device to run sampling on
        call_every: How often to call this callback in training steps
    """
    ema: Optional[Callable] = None
    output_dir: str = "ema_outputs"
    
    @torch.compiler.disable()
    def __call__(
        self, 
        training_step: int,
        dataloader: DataLoader = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if training_step % self.call_every != 0 or dataloader is None or self.ema is None:
            return None
        
        # Store EMA weights
        self.ema.store()
        
        # Run sampling with EMA weights
        try:
            super().__call__(training_step, dataloader, labels)
        finally:
            # Restore original weights, even if an exception occurs
            self.ema.restore()
