import torch
import matplotlib.pyplot as plt
import psutil
import os

def save_model(model, path='model.pth'):
    """
    Save model to disk
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path='model.pth', device='cpu'):
    """
    Load model from disk
    
    Args:
        model: PyTorch model to load weights into
        path: Path to load the model from
        device: Device to load the model to
        
    Returns:
        model: Loaded model
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def estimate_flops(model, data, conditioning=None, device=None):
    """
    Estimate FLOPs per batch for the model.
    This is a simple estimation using PyTorch's FlopCounterMode.
    
    Args:
        model: PyTorch model to estimate FLOPs for
        data: Input data tensor
        conditioning: Conditioning tensor
        device: Device to run estimation on
        
    Returns:
        float: Estimated FLOPs per batch
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        
        # Move model inputs to the specified device if provided
        if device is not None:
            data = data.to(device)
            if conditioning is not None:
                conditioning = conditioning.to(device)
        
        # Create input tensor for model
        model_input = (data, conditioning, torch.randn(data.shape[0], device=data.device))
        
        # Use FlopCounterMode as a context manager
        with FlopCounterMode(model) as flop_counter:
            _ = model(*model_input)
        
        # Get FLOPs count
        flops = flop_counter.get_total_flops()
        
        print(f"Estimated model FLOPs per batch: {flops/1e9:.2f} GFLOPs")
        return flops
        
    except ImportError:
        print("Warning: torch.utils.flop_counter not available. Make sure you're using a recent PyTorch version.")
        return 0
    except Exception as e:
        print(f"Error estimating FLOPs: {e}")
        return 0

class EMA:
    """
    Exponential Moving Average for model weights
    
    Args:
        model: PyTorch model to track
        decay: EMA decay rate (higher = slower moving average)
        device: Device to store EMA weights on
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.shadow_params = [p.clone().detach().to(device) for p in model.parameters()]
        self.collected_params = []
        
    def update(self):
        """Update EMA weights"""
        for i, param in enumerate(self.model.parameters()):
            # Use in-place operations to update shadow_params
            self.shadow_params[i].copy_(
                self.decay * self.shadow_params[i] + (1 - self.decay) * param.detach()
            )
            
    def store(self):
        """Store current model parameters and replace with EMA weights"""
        self.collected_params = [param.clone().detach() for param in self.model.parameters()]
        for i, param in enumerate(self.model.parameters()):
            param.data.copy_(self.shadow_params[i])
    
    def restore(self):
        """Restore original model parameters"""
        for i, param in enumerate(self.model.parameters()):
            param.data.copy_(self.collected_params[i])
            
    

def get_memory_usage(device=None):
    """
    Get current CPU and GPU memory usage.
    
    Args:
        device: PyTorch device to check GPU memory for. If None, 
               only CPU memory is returned.
    
    Returns:
        String containing formatted memory usage information
    """
    memory_stats = {}
    
    try:
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
        memory_stats['cpu_memory_gb'] = cpu_memory
    except Exception as e:
        memory_stats['cpu_memory_error'] = str(e)
    
    # GPU memory if CUDA is available
    if torch.cuda.is_available():
        try:
            # Get device index if it's a CUDA device
            device_idx = 0  # Default to first GPU
            if device is not None and device.type == 'cuda' and hasattr(device, 'index'):
                device_idx = device.index
                
            gpu_memory_allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024 * 1024)  # Convert to GB
            gpu_memory_reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024 * 1024)  # Convert to GB
            
            memory_stats['gpu_allocated_gb'] = gpu_memory_allocated
            memory_stats['gpu_reserved_gb'] = gpu_memory_reserved
        except Exception as e:
            memory_stats['gpu_memory_error'] = str(e)
    
    return format_memory_stats(memory_stats)

def format_memory_stats(memory_stats):
    """
    Format memory statistics into a readable string.
    
    Args:
        memory_stats: Dictionary containing memory statistics
        
    Returns:
        Formatted string with memory information
    """
    result = []
    
    if 'cpu_memory_gb' in memory_stats:
        result.append(f"CPU: {memory_stats['cpu_memory_gb']:.2f}GB")
    elif 'cpu_memory_error' in memory_stats:
        result.append(f"CPU Error: {memory_stats['cpu_memory_error']}")
        
    if 'gpu_allocated_gb' in memory_stats:
        result.append(f"GPU: {memory_stats['gpu_allocated_gb']:.2f}GB (used) / {memory_stats['gpu_reserved_gb']:.2f}GB (reserved)")
    elif 'gpu_memory_error' in memory_stats:
        result.append(f"GPU Error: {memory_stats['gpu_memory_error']}")
        
    if not result:
        return "Memory stats unavailable"
        
    return " | ".join(result)

def calculate_mfu(flops_per_batch, batch_time, peak_flops):
    """
    Calculate Model FLOP Utilization (MFU).
    
    Args:
        flops_per_batch: FLOPs per batch
        batch_time: Time taken for processing one batch in seconds
        peak_flops: Peak FLOPs capability of the hardware
            
    Returns:
        float: MFU as a percentage
    """
    if flops_per_batch == 0 or batch_time == 0:
        return 0.0
            
    achieved_flops = flops_per_batch / batch_time  # FLOPs/second achieved
    mfu = (achieved_flops / peak_flops) * 100  # Convert to percentage
    return mfu 
