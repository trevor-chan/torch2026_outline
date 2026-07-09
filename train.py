import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import BouncingBallVideoDataset
from transformer import TransformerDiffusionModel
from trainer import Trainer
from utils import save_model, EMA
from loss import RectifiedFlowLoss
from callbacks import ValidationCallback, RectifiedFlowSampler, EMAFlowSampler

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bouncing Ball Frame Diffusion Model Training')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.00005)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--save-model', action='store_true', default=True, help='save model after training')
    parser.add_argument('--val-steps', type=int, default=128, help='number of steps for validation (default: 100)')
    parser.add_argument('--log-interval', type=int, default=2_000, help='how many steps to wait before logging training status')
    parser.add_argument('--val-interval', type=int, default=10_000, help='validation interval in steps (default: 50000)')
    parser.add_argument('--sample-interval', type=int, default=10_000, help='sampling interval in steps (default: 50000)')
    parser.add_argument('--sample-steps', type=int, default=128, help='number of steps for sampling (default: 32)')
    parser.add_argument('--max-steps', type=int, default=None, help='optional number of training steps before stopping')
    parser.add_argument('--profile-memory', action='store_true', default=True, help='enable memory profiling')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay rate (default: 0.999)')
    parser.add_argument('--dataset-size', type=int, default=500_000, help='number of synthetic training frames (default: 10000)')
    parser.add_argument('--val-size', type=int, default=2_000, help='number of synthetic validation frames (default: 2000)')
    parser.add_argument('--image-size', type=int, default=16, help='synthetic frame size in pixels (default: 16)')
    parser.add_argument('--model-dim', '--hidden-dim', dest='model_dim', type=int, default=512, help='transformer width (default: 256)')
    parser.add_argument('--num-layers', type=int, default=4, help='number of transformer blocks (default: 4)')
    parser.add_argument('--num-heads', type=int, default=4, help='number of attention heads (default: 4)')
    parser.add_argument('--time-embed-dim', type=int, default=None, help='time embedding width (default: model dim)')
    parser.add_argument('--drop-rate', type=float, default=0.0, help='transformer residual dropout (default: 0.0)')
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, help='attention dropout (default: 0.0)')
    parser.add_argument('--rope-theta', type=float, default=100.0, help='2D RoPE frequency base (default: 100.0)')
    parser.add_argument('--num-workers', type=int, default=24, help='training dataloader workers (default: 12)')
    parser.add_argument('--val-workers', type=int, default=4, help='validation dataloader workers (default: 2)')
    parser.add_argument(
        '--dataset-video',
        type=str,
        default='outputs/bouncing_ball_dataset.mp4',
        help='path for the generated dataset preview video',
    )
    parser.add_argument(
        '--skip-dataset-video',
        action='store_true',
        help='do not write the dataset preview video during initialization',
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize distributed training
    distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ
    if distributed:
        torch.multiprocessing.set_start_method("spawn")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(
            "nccl",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            device_id=torch.device("cuda", index=int(os.environ["LOCAL_RANK"])),
        )
        device = torch.device("cuda", index=int(os.environ["LOCAL_RANK"]))
    
    model = TransformerDiffusionModel(
        in_channels=3,
        dim=args.model_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        time_embed_dim=args.time_embed_dim,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        rope_theta=args.rope_theta,
    )
    model = model.to(device)
    
    # Print model summary
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create datasets
    if distributed:
        rank = int(os.environ["RANK"])
        train_seed = args.seed + rank
        val_seed = args.seed + 10_000 + rank
        write_dataset_video = rank == 0 and not args.skip_dataset_video
        model.shard()
    else:
        train_seed = args.seed
        val_seed = args.seed + 10_000
        write_dataset_video = not args.skip_dataset_video

    model = torch.compile(model)

    train_dataset = BouncingBallVideoDataset(
        num_samples=args.dataset_size,
        image_size=args.image_size,
        seed=train_seed,
        normalize=False,
        return_conditioning=False,
        video_path=args.dataset_video,
        write_video=write_dataset_video,
    )
    val_dataset = BouncingBallVideoDataset(
        num_samples=args.val_size,
        image_size=args.image_size,
        seed=val_seed,
        normalize=False,
        return_conditioning=False,
        write_video=False,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=device.type == "cuda",
    )
    
    # Define loss and optimizer
    criterion = RectifiedFlowLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = optim.Muon(model.parameters(), lr=args.lr * 10)
    
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
        call_every=args.sample_interval,
        output_dir="outputs/bouncing_ball_samples",
    )
    
    # EMA sampling callback
    ema_sampling_callback = EMAFlowSampler(
        model=model,
        ema=ema,
        device=device,
        num_steps=args.sample_steps,
        call_every=args.sample_interval,
        output_dir="outputs/bouncing_ball_ema_samples",
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
        ema=ema,
        max_steps=args.max_steps,
    )
    
    # Start training
    print("Training...")
    trainer()
    
    # Save model
    if args.save_model:
        save_model(model, 'bouncing_ball_diffusion.pth')
        print("Model saved to bouncing_ball_diffusion.pth")
        
        # Save EMA model
        ema.store()  # Apply EMA weights
        save_model(model, 'bouncing_ball_diffusion_ema.pth')
        ema.restore()  # Restore original weights
        print("EMA model saved to bouncing_ball_diffusion_ema.pth")


if __name__ == '__main__':
    main() 
