import torch
import torch.nn as nn
from image_utils import flatten_image
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=None, output_dim=784, activation=nn.ReLU()):
        """
        Multilayer Perceptron (MLP) with configurable depth and width
        
        Args:
            input_dim (int): Input dimension (flattened 28x28 MNIST image)
            hidden_dims (list of int): List of hidden dimensions for each layer
            output_dim (int): Output dimension (10 for MNIST)
            activation (nn.Module): Activation function to use between layers
        """
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [4096, 4096, 4096, 4096]
        self.activation = activation
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.input = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        
    
    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output(x)
    
    
    def shard(self, mp_policy: bool = True):
        if mp_policy:
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        for layer_idx, layer in enumerate(self.layers):
            reshard_after_forward = layer_idx < len(self.layers) - 1
            fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)

        fully_shard(self, mp_policy=mp_policy, reshard_after_forward=True)


class DiffusionModel(nn.Module):
    """
    Extends the MLP model to work with diffusion model inputs
    (data, optional conditioning, and time).
    """
    def __init__(self, input_dim=795, hidden_dims=None, output_dim=784):
        """
        Initialize the diffusion model
        
        Args:
            input_dim (int): Input dimension (flattened image + conditioning + time)
            hidden_dims (list of int): List of hidden dimensions for each layer
            output_dim (int): Output dimension (flattened image dimension)
        """
        super(DiffusionModel, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    
    def forward(self, x, conditioning=None, t=None):
        """
        Forward pass with conditioning and time inputs.
        
        Args:
            x: Input data tensor
            conditioning: Optional conditioning tensor
            t: Timestep tensor
            
        Returns:
            Model prediction
        """
        original_shape = x.shape
        x = flatten_image(x)

        if t is None:
            raise ValueError("DiffusionModel.forward requires a timestep tensor")

        if conditioning is None:
            conditioning = x.new_empty(x.shape[0], 0)
        else:
            conditioning = conditioning.to(x.device)
            if conditioning.dim() == 1:
                conditioning = conditioning.view(x.shape[0], 1)
            elif conditioning.dim() > 2:
                conditioning = flatten_image(conditioning)

        t_embed = t.view(-1, 1)
        inputs = torch.cat([x, conditioning, t_embed], dim=1)

        output = self.mlp(inputs)
        if len(original_shape) > 2:
            output = output.view(*original_shape)

        return output
    
    def shard(self):
        self.mlp.shard()
