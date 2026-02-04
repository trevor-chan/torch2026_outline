import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import MNISTDataset
from model import DiffusionModel
# from transformer import TransformerDiffusionModel
from trainer import Trainer
from utils import save_model, EMA
from loss import RectifiedFlowLoss
from callbacks import ValidationCallback, RectifiedFlowSampler, EMAFlowSampler

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST Diffusion Model Training')
    parser.add_argument('--batch-size', type=int, default=1024, help='input batch size for training (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 42)')
    parser.add_argument('--save-model', action='store_true', default=True, help='save model after training')
    parser.add_argument('--val-steps', type=int, default=100, help='number of steps for validation (default: 100)')
    parser.add_argument('--log-interval', type=int, default=500, help='how many steps to wait before logging training status')
    parser.add_argument('--val-interval', type=int, default=5000, help='validation interval in steps (default: 1000)')
    parser.add_argument('--sample-interval', type=int, default=5000, help='sampling interval in steps (default: 1000)')
    parser.add_argument('--sample-steps', type=int, default=32, help='number of steps for sampling (default: 32)')
    parser.add_argument('--profile-memory', action='store_true', default=True, help='enable memory profiling')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay rate (default: 0.999)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Assume CUDA is always available
    device = torch.device("cuda")
    
    # initialize distributed training
    if torch.cuda.device_count() > 1:
        torch.multiprocessing.set_start_method("spawn")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(
            "nccl",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            device_id=torch.device("cuda", index=int(os.environ["LOCAL_RANK"])),
        )
    
    # Create model
    # MNIST images are (28 * 28) + 10 (one-hot labels) + 1 (time) = 795
    model = DiffusionModel(input_dim=784 + 10 + 1, hidden_dims=[4096, 4096,], output_dim=784)
    # model = TransformerDiffusionModel(dim=64, num_heads=1, num_layers=2, max_seq_len=1024, output_dim=784)
    model = model.to(torch.device(device))
    model = torch.compile(model)
    
    # Print model summary
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create datasets
    if torch.cuda.device_count() > 1:
        train_dataset = MNISTDataset(train=True, seed=args.seed + int(os.environ["RANK"]))
        val_dataset = MNISTDataset(train=False, seed=args.seed + int(os.environ["RANK"]))
        model.shard()
    else:
        train_dataset = MNISTDataset(train=True, seed=args.seed)
        val_dataset = MNISTDataset(train=False, seed=args.seed)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8)    
    
    # Define loss and optimizer
    criterion = RectifiedFlowLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create EMA for model weights
    ema = EMA(model, decay=args.ema_decay, device=device)
    
    # Initialize callbacks
    validation_callback = ValidationCallback(
        model=model,
        criterion=criterion,
        device=device,
        num_iterations=args.val_steps,
        call_every=args.val_interval
    )
    
    # Regular sampling callback
    sampling_callback = RectifiedFlowSampler(
        model=model,
        device=device,
        num_steps=args.sample_steps,
        call_every=args.sample_interval
    )
    
    # EMA sampling callback
    ema_sampling_callback = EMAFlowSampler(
        model=model,
        ema=ema,
        device=device,
        num_steps=args.sample_steps,
        call_every=args.sample_interval
    )    
    
    # Create trainer with callbacks and their respective dataloaders
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=[validation_callback, sampling_callback, ema_sampling_callback],
        log_interval=args.log_interval,
        profile_memory=args.profile_memory,
        ema=ema
    )
    
    # Start training
    print("Training...")
    trainer()
    
    # Save model
    if args.save_model:
        save_model(model, 'mnist_diffusion.pth')
        print("Model saved to mnist_diffusion.pth")
        
        # Save EMA model
        ema.store()  # Apply EMA weights
        save_model(model, 'mnist_diffusion_ema.pth')
        ema.restore()  # Restore original weights
        print("EMA model saved to mnist_diffusion_ema.pth")


if __name__ == '__main__':
    main() 