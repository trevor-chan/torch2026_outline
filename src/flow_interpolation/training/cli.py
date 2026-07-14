"""Command-line entry point for rectified-flow training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flow_interpolation.data import BouncingBallVideoDataset, OrderedTripletDataset
from flow_interpolation.models import TransformerDiffusionModel
from flow_interpolation.training.callbacks import (
    CheckpointCallback,
    EMAFlowSampler,
    RectifiedFlowSampler,
    ValidationCallback,
)
from flow_interpolation.training.checkpoints import (
    find_latest_checkpoint,
    load_training_checkpoint,
    workdir_from_checkpoint,
)
from flow_interpolation.training.engine import Trainer
from flow_interpolation.training.losses import RectifiedFlowLoss
from flow_interpolation.training.runs import (
    create_workdir,
    load_training_arguments,
    save_training_config,
)
from flow_interpolation.utils.training import EMA, save_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flow_interpolation train",
        description="Train a rectified-flow model on the bouncing-ball sequence.",
    )
    run = parser.add_argument_group("run management")
    run.add_argument("--runs-dir", default="outputs/runs", help="parent directory for new runs")
    run.add_argument("--run-name", help="optional name for a new run; collisions get numeric suffixes")
    run.add_argument("--workdir", help="exact directory for a new run; must not already exist")
    run.add_argument(
        "--resume",
        help="checkpoint file, checkpoint directory, or run directory to resume",
    )

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-steps", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=2_000)
    parser.add_argument("--val-interval", type=int, default=10_000, help="set 0 to disable")
    parser.add_argument("--sample-interval", type=int, default=10_000, help="set 0 to disable")
    parser.add_argument("--checkpoint-interval", type=int, default=10_000, help="set 0 to save only at normal exit")
    parser.add_argument("--sample-steps", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--profile-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--peak-tflops", type=float, default=91.1, help="per-device peak FLOP/s used for MFU")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="set <=0 to disable clipping")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-amp", action="store_true", help="disable BF16 autocast")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")

    acceleration = parser.add_argument_group("latent acceleration regularization")
    acceleration.add_argument(
        "--acceleration-weight",
        type=float,
        default=0.0,
        help="weight on the raw terminal-latent second-difference loss; 0 disables it",
    )
    acceleration.add_argument(
        "--acceleration-frame-stride",
        type=int,
        default=1,
        help="spacing between frames in each ordered training triplet",
    )
    acceleration.add_argument(
        "--acceleration-ode-steps",
        type=int,
        default=1,
        help="differentiable data-to-noise steps; 1 Euler step is the cheap endpoint proxy",
    )
    acceleration.add_argument(
        "--acceleration-solver",
        choices=("euler", "heun"),
        default="euler",
    )
    acceleration.add_argument("--acceleration-data-eps", type=float, default=1e-3)
    acceleration.add_argument("--acceleration-noise-eps", type=float, default=1e-3)

    parser.add_argument("--dataset-size", type=int, default=50_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--model-dim", "--hidden-dim", dest="model_dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--time-embed-dim", type=int, default=None)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--attn-drop-rate", type=float, default=0.0)
    parser.add_argument("--rope-theta", type=float, default=100.0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument(
        "--dataset-video",
        help="optional preview path; defaults to <workdir>/artifacts/bouncing_ball_dataset.mp4",
    )
    parser.add_argument("--skip-dataset-video", action="store_true")
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _apply_resume_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str],
) -> argparse.Namespace:
    if not args.resume:
        return args
    checkpoint = find_latest_checkpoint(args.resume)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found at {args.resume}")
    saved_arguments = load_training_arguments(workdir_from_checkpoint(checkpoint))
    explicit_destinations = set()
    for token in argv:
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            explicit_destinations.add(action.dest)

    excluded = {"resume", "runs_dir", "run_name", "workdir"}
    for destination, value in saved_arguments.items():
        if (
            destination not in excluded
            and destination not in explicit_destinations
            and hasattr(args, destination)
        ):
            setattr(args, destination, value)
    return args


def _initialize_distributed(device: torch.device) -> tuple[torch.device, bool, int]:
    distributed = device.type == "cuda" and "LOCAL_RANK" in os.environ
    if not distributed:
        return device, False, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        "nccl",
        rank=rank,
        world_size=int(os.environ["WORLD_SIZE"]),
        device_id=torch.device("cuda", index=local_rank),
    )
    return torch.device("cuda", index=local_rank), True, rank


def _select_run(
    args: argparse.Namespace,
    *,
    distributed: bool,
    rank: int,
) -> tuple[Path, Path | None]:
    selection: list[str | None] = [None, None]
    if rank == 0:
        if args.resume:
            checkpoint = find_latest_checkpoint(args.resume)
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint found at {args.resume}")
            selection = [str(workdir_from_checkpoint(checkpoint)), str(checkpoint)]
        else:
            workdir = create_workdir(
                args.runs_dir,
                run_name=args.run_name,
                workdir=args.workdir,
            )
            selection = [str(workdir), None]
    if distributed:
        torch.distributed.broadcast_object_list(selection, src=0)
        torch.distributed.barrier()
    return Path(selection[0]), Path(selection[1]) if selection[1] is not None else None


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _apply_resume_defaults(parser, parser.parse_args(raw_argv), raw_argv)
    if args.resume and (args.workdir or args.run_name):
        parser.error("--resume cannot be combined with --workdir or --run-name")
    if args.val_steps <= 0:
        parser.error("--val-steps must be positive")
    if args.peak_tflops <= 0:
        parser.error("--peak-tflops must be positive")
    if args.acceleration_weight < 0.0:
        parser.error("--acceleration-weight must be non-negative")
    if args.acceleration_frame_stride <= 0:
        parser.error("--acceleration-frame-stride must be positive")
    if args.acceleration_ode_steps <= 0:
        parser.error("--acceleration-ode-steps must be positive")
    if args.acceleration_weight > 0.0 and min(args.dataset_size, args.val_size) <= (
        2 * args.acceleration_frame_stride
    ):
        parser.error(
            "training and validation datasets must each contain more than "
            "2 * acceleration-frame-stride frames"
        )
    if args.acceleration_weight > 0.0 and (
        args.drop_rate > 0.0 or args.attn_drop_rate > 0.0
    ):
        parser.error(
            "acceleration regularization requires --drop-rate 0 and "
            "--attn-drop-rate 0 so stochastic masks do not create false curvature"
        )
    if not 0.0 <= args.acceleration_data_eps < 1.0 - args.acceleration_noise_eps <= 1.0:
        parser.error(
            "acceleration endpoint epsilons must satisfy "
            "0 <= data_eps < 1 - noise_eps <= 1"
        )

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    amp_dtype = None if args.no_amp else torch.bfloat16
    device, distributed, rank = _initialize_distributed(_resolve_device(args.device))
    is_main_process = rank == 0
    workdir, resume_checkpoint = _select_run(args, distributed=distributed, rank=rank)
    if is_main_process:
        print(f"Training work directory: {workdir}")

    model = TransformerDiffusionModel(
        in_channels=3,
        dim=args.model_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        time_embed_dim=args.time_embed_dim,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        rope_theta=args.rope_theta,
    ).to(device)
    if distributed:
        model.shard()
    if args.compile:
        model = torch.compile(model)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if is_main_process:
        print(f"Total parameters: {total_parameters:,d} ({total_parameters / 1e6:.2f} M)")

    train_seed = args.seed + rank
    val_seed = args.seed + 10_000 + rank
    dataset_video = Path(args.dataset_video) if args.dataset_video else workdir / "artifacts/bouncing_ball_dataset.mp4"
    write_dataset_video = (
        is_main_process
        and resume_checkpoint is None
        and not args.skip_dataset_video
    )
    train_dataset = BouncingBallVideoDataset(
        num_samples=args.dataset_size,
        image_size=args.image_size,
        seed=train_seed,
        normalize=False,
        return_conditioning=False,
        video_path=str(dataset_video),
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
    if args.acceleration_weight > 0.0:
        train_dataset = OrderedTripletDataset(
            train_dataset,
            frame_stride=args.acceleration_frame_stride,
        )
        val_dataset = OrderedTripletDataset(
            val_dataset,
            frame_stride=args.acceleration_frame_stride,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = RectifiedFlowLoss(
        acceleration_weight=args.acceleration_weight,
        acceleration_ode_steps=args.acceleration_ode_steps,
        acceleration_solver=args.acceleration_solver,
        data_time=args.acceleration_data_eps,
        noise_time=1.0 - args.acceleration_noise_eps,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay, device=device)
    start_step = 0
    resume_result = None
    if resume_checkpoint is not None:
        resume_result = load_training_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            ema=ema,
            device=device,
        )
        start_step = resume_result.step

    writer = None
    if is_main_process:
        writer = SummaryWriter(
            log_dir=str(workdir / "tensorboard"),
            purge_step=start_step + 1 if resume_checkpoint is not None else None,
            filename_suffix=(
                f".resume_{start_step:09d}" if resume_checkpoint is not None else ".train"
            ),
        )
        print(f"TensorBoard logs: {workdir / 'tensorboard'}")
        save_training_config(
            workdir,
            args,
            resolved={
                "device": str(device),
                "distributed": distributed,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "total_parameters": total_parameters,
                "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
                "dataset_video": str(dataset_video),
                "start_step": start_step,
                "full_state_resume": resume_result.full_state if resume_result else None,
                "training_batch_semantics": (
                    "ordered_triplets_with_center_frame_flow_matching"
                    if args.acceleration_weight > 0.0
                    else "independent_frames"
                ),
                "model_evaluations_per_training_item": (
                    1
                    + 3
                    * args.acceleration_ode_steps
                    * (2 if args.acceleration_solver == "heun" else 1)
                    if args.acceleration_weight > 0.0
                    else 1
                ),
            },
            resumed_from=resume_checkpoint,
        )

    validation_callback = ValidationCallback(
        model=model,
        criterion=criterion,
        device=device,
        num_iterations=args.val_steps,
        call_every=args.val_interval,
        amp_dtype=amp_dtype,
        writer=writer,
        log_enabled=is_main_process,
    )
    sampling_callback = RectifiedFlowSampler(
        model=model,
        device=device,
        num_steps=args.sample_steps,
        call_every=args.sample_interval,
        output_dir=workdir / "samples/model",
        writer=writer,
        write_enabled=is_main_process,
    )
    ema_sampling_callback = EMAFlowSampler(
        model=model,
        ema=ema,
        device=device,
        num_steps=args.sample_steps,
        call_every=args.sample_interval,
        output_dir=workdir / "samples/ema",
        writer=writer,
        write_enabled=is_main_process,
    )
    checkpoint_callback = CheckpointCallback(
        model=model,
        optimizer=optimizer,
        ema=ema,
        call_every=args.checkpoint_interval,
        output_dir=workdir / "checkpoints",
        enabled=is_main_process,
        extra={"workdir": str(workdir)},
    )

    trainer = Trainer(
        workdir=workdir,
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=[
            validation_callback,
            sampling_callback,
            ema_sampling_callback,
            checkpoint_callback,
        ],
        log_interval=args.log_interval,
        profile_memory=args.profile_memory,
        ema=ema,
        peak_flops=args.peak_tflops * 1e12,
        max_steps=args.max_steps,
        amp_dtype=amp_dtype,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        writer=writer,
        start_step=start_step,
        is_main_process=is_main_process,
    )

    try:
        print("Training...")
        final_step = trainer()
        if args.save_model and is_main_process:
            save_model(model, workdir / f"model_final_step_{final_step:09d}.pth")
            ema.store()
            try:
                save_model(model, workdir / f"model_ema_final_step_{final_step:09d}.pth")
            finally:
                ema.restore()
            print(f"Saved final model weights at step {final_step}")
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()
