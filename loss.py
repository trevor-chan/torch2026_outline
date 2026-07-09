import torch
import torch.nn as nn


class RectifiedFlowLoss(nn.Module): # This is a flow matching loss function
    """
    Implements a rectified flow diffusion loss function.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, model, data, conditioning):
        """
        Computes the rectified flow diffusion loss.
        
        Args:
            model: The model to evaluate
            data: Input data tensor
            conditioning: Conditioning labels
            
        Returns:
            Computed loss value
        """
        
        batch_size = data.shape[0]
        device = data.device
        
        t = torch.rand(batch_size, device=device) # Generate a random timestep
        
        noise = torch.randn_like(data) # Generate noise vector with the same shape as the data
        
        t_reshaped = t.view(batch_size, *([1] * (len(data.shape) - 1))) # Reshape the timestep to match the data shape
        
        interpolated = torch.lerp(data, noise, t_reshaped) # Interpolate between the data and the noise using the timestep
        
        prediction = model(interpolated, conditioning, t) # Predict the noise

        loss = ((prediction - noise + data) ** 2).mean() # Compute the loss
        
        return loss