# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# from torch.utils.data import Dataset, DataLoader
# import torch.multiprocessing as mp
# import random
# import numpy as np
# from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard


# class RandomDataset(Dataset):
#     def __init__(self, input_size, num_samples=1000):
#         self.input_size = input_size
#         self.num_samples = num_samples
        
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         # Generate random input tensor and random target
#         x = torch.randn(self.input_size)
#         y = torch.randint(0, 2, (1,)).item()  # Binary classification
#         return x, y


# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(SimpleMLP, self).__init__()
#         layers = []
#         layer_sizes = [input_size] + hidden_sizes + [output_size]
        
#         for i in range(len(layer_sizes) - 1):
#             layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
#             if i < len(layer_sizes) - 2:
#                 layers.append(nn.ReLU())
        
#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.model(x)
    
#     def shard(self):
#         for layer in self.model:
#             fully_shard(layer)
#         fully_shard(self)


# def train_fsdp(rank, world_size):
#     # Initialize process group
    
#     dist.init_process_group("nccl",
#                              rank=rank, 
#                              world_size=world_size)
    
#     # Set device for this process
#     device = torch.device(f"cuda:{rank}")
#     torch.cuda.set_device(device)
    
#     # Model parameters
#     input_size = 128
#     hidden_sizes = [256, 512, 256]
#     output_size = 1
#     batch_size = 32
#     epochs = 5
    
#     # Create model with FSDP wrapping
#     model = SimpleMLP(input_size, hidden_sizes, output_size).to(device)
    
#     # Wrap model with FSDP
#     model.shard()
    
#     # Loss and optimizer
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Dataset and DataLoader
#     dataset = RandomDataset(input_size)
#     sampler = torch.utils.data.distributed.DistributedSampler(
#         dataset, num_replicas=world_size, rank=rank
#     )
#     dataloader = DataLoader(
#         dataset, batch_size=batch_size, sampler=sampler, shuffle=False
#     )
    
#     # Training loop
#     for epoch in range(epochs):
#         sampler.set_epoch(epoch)
#         running_loss = 0.0
        
#         for i, (inputs, targets) in enumerate(dataloader):
#             print(f'here')
#             inputs = inputs.to(device)
#             targets = targets.float().to(device).view(-1, 1)
            
#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             print(f'here2')
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print(f'here3')
            
#             running_loss += loss.item()
            
#             if i % 10 == 0 and rank == 0:
#                 print(f"[FSDP] Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
#                       f"Loss: {loss.item():.4f}")
    
#     # Clean up
#     dist.destroy_process_group()


# def main():
#     # Number of processes to spawn
#     world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
#     print(f"Using {world_size} processes for distributed training")
    
#     # Set random seeds for reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
    
#     # Choose training method
#     train_fsdp(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']))


# if __name__ == "__main__":
#     main()



# import torch
# import os
# import torch.distributed as dist

# # initialize distributed training
# dist.init_process_group("nccl",
#                          rank=int(os.environ['RANK']),
#                          world_size=int(os.environ['WORLD_SIZE']))

# # Check if CUDA is available
# if torch.cuda.is_available():
#     # Get the number of available GPUs
#     num_gpus = torch.cuda.device_count()
#     print(f"Number of available GPUs: {num_gpus}")

#     # Create tensors on different GPUs
#     tensors = []
#     for i in range(num_gpus):
#         device = torch.device(f'cuda:{i}')
#         tensor = torch.randn(3, 3, device=device)
#         tensors.append(tensor)

#     # Attempt communication between GPUs
#     try:
#         # Move tensor from GPU 0 to GPU 1 (if multiple GPUs are available)
#         if num_gpus > 1:
#             tensors[0] = tensors[0].to(torch.device('cuda:1'))
#             print("Successfully moved tensor from GPU 0 to GPU 1.")
#         else:
#             print("Only one GPU is available, skipping cross-GPU communication test.")
#     except Exception as e:
#          print(f"Error during cross-GPU communication: {e}")
# else:
#     print("CUDA is not available. Please ensure you have a CUDA-enabled GPU and drivers installed.")