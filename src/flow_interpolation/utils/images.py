import torch
import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple


def denormalize_image(image, source_range=(-1, 1)):
    """Convert an image from ``source_range`` to byte-valued pixels."""
    lower, upper = source_range
    return ((image - lower) / (upper - lower) * 255).clamp(0, 255)

def reconstruct_image(flat_image, height=28, width=28, channels=1):
    """
    Reconstruct flattened image back to 2D/3D
    Returns:
        Reconstructed image tensor [C, H, W] or [B, C, H, W]
    """
    if len(flat_image.shape) == 2:  # Batch of flattened images
        batch_size = flat_image.shape[0]
        return flat_image.view(batch_size, channels, height, width)
    else:  # Single flattened image
        return flat_image.view(channels, height, width) 

def save_image_batch(
    images: torch.Tensor, 
    filepath: str, 
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None
) -> None:
    """
    Save a batch of images to a file
    
    Args:
        images: Tensor of images, shape (N, C, H, W)
        filepath: Path to save the file to
        nrow: Number of images displayed in each row of the grid
        padding: Amount of padding between images
        normalize: Flag to normalize images to [0, 1]
        value_range: Tuple of (min, max) used to normalize images if normalize=True
        scale_each: Flag to scale each image independently if normalize=True
        pad_value: Value for the padded pixels
        format: Image format, if None uses the filepath's extension
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Get image dimensions
    if images.dim() == 4:  # N,C,H,W
        N, C, H, W = images.shape
    else:
        raise ValueError(f"Expected 4D tensor of shape [N,C,H,W], got tensor with shape {images.shape}")
    
    # Normalize if needed
    if normalize:
        if value_range is None:
            value_range = (-1, 1)  # Default normalization range
        
        # Apply normalization
        if scale_each:
            # Scale each image independently
            for i in range(N):
                images[i] = denormalize_image(images[i], source_range=value_range)
        else:
            # Scale all images together
            images = denormalize_image(images, source_range=value_range)
    
    # If images are in range [0, 1], scale to [0, 255]
    if images.max() <= 1.0:
        images = images * 255
    
    # Move to CPU and convert to numpy
    images = images.cpu().numpy().astype(np.uint8)
    
    # Calculate grid layout
    nrow = min(nrow, N)
    ncol = (N + nrow - 1) // nrow  # Ceiling division
    
    # Create output grid with padding
    grid_H = H * nrow + padding * (nrow - 1)
    grid_W = W * ncol + padding * (ncol - 1)
    
    # Create grid with padding
    grid = np.full((C, grid_H, grid_W), pad_value, dtype=np.uint8)
    
    # Fill grid with images
    for idx in range(N):
        row = idx // ncol
        col = idx % ncol
        y = row * (H + padding)
        x = col * (W + padding)
        grid[:, y:y+H, x:x+W] = images[idx]
    
    # Convert to PIL format based on channels
    if C == 1:
        # For grayscale
        grid = grid.squeeze(0)
        img = Image.fromarray(grid, mode='L')
    elif C == 3:
        # For RGB, rearrange to HWC
        grid = np.transpose(grid, (1, 2, 0))
        img = Image.fromarray(grid, mode='RGB')
    elif C == 4:
        # For RGBA, rearrange to HWC
        grid = np.transpose(grid, (1, 2, 0))
        img = Image.fromarray(grid, mode='RGBA')
    else:
        raise ValueError(f"Unsupported number of channels: {C}")
    
    # Save the image
    img.save(filepath, format=format) 
