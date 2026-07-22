"""Command-line entry point for fitting a scene to sparse k-space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from flow_interpolation.data import BouncingBallVideoDataset
from flow_interpolation.kspace import (
    MASK_FAMILIES,
    build_dynamic_kspace,
    temporal_average_reconstruction,
    zero_filled_reconstruction,
)
from flow_interpolation.scene.binning import build_bin_schedule
from flow_interpolation.scene.fit import SceneFitter, save_scene
from flow_interpolation.scene.models import SCENE_MODELS, build_scene_model
from flow_interpolation.scene.visualization import ReconstructionVisualizer
from flow_interpolation.training.runs import create_workdir, save_training_config
from flow_interpolation.utils.metrics import image_metrics, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flow_interpolation fit",
        description="Fit an implicit scene model to sparse dynamic k-space observations.",
    )
    run = parser.add_argument_group("run management")
    run.add_argument("--runs-dir", default="outputs/fits", help="parent directory for new runs")
    run.add_argument("--run-name", help="optional name for a new run")
    run.add_argument("--workdir", help="exact directory for a new run; must not already exist")

    data = parser.add_argument_group("measurements")
    data.add_argument("--num-frames", type=int, default=200, help="dense frames to simulate")
    data.add_argument("--image-size", type=int, default=32)
    data.add_argument("--sampling-rate", type=float, default=0.1)
    data.add_argument("--mask-family", choices=sorted(MASK_FAMILIES), default="variable-density")
    data.add_argument(
        "--center-fraction",
        type=float,
        default=0.0,
        help="fraction of the k-space area around DC always sampled",
    )
    data.add_argument("--noise-std", type=float, default=0.0, help="complex measurement noise")
    data.add_argument("--frame-dt", type=float, default=0.05, help="simulation seconds per frame")
    data.add_argument("--data-seed", type=int, default=0)

    binning = parser.add_argument_group("temporal binning")
    binning.add_argument(
        "--condition",
        choices=("wide", "narrow", "curriculum"),
        default="curriculum",
        help="fixed wide bins, fixed narrow bins, or an annealed curriculum",
    )
    binning.add_argument("--start-width", type=int, default=25, help="bin width in frames")
    binning.add_argument("--end-width", type=int, default=1)
    binning.add_argument("--anneal-fraction", type=float, default=0.5)
    binning.add_argument(
        "--anneal-kind",
        choices=("linear", "exponential", "step"),
        default="exponential",
    )

    model = parser.add_argument_group("scene model")
    model.add_argument("--scene-model", choices=sorted(SCENE_MODELS), default="kplanes")
    model.add_argument("--feature-dim", type=int, default=32)
    model.add_argument("--resolutions", default="32,64", help="comma-separated plane resolutions")
    model.add_argument("--time-resolution", type=int, default=32)
    model.add_argument("--num-features", type=int, default=128, help="fourier-mlp feature count")
    model.add_argument("--space-scale", type=float, default=8.0, help="fourier-mlp spatial bandwidth")
    model.add_argument("--time-scale", type=float, default=8.0, help="fourier-mlp temporal bandwidth")
    model.add_argument("--hidden-dim", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=3)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--max-steps", type=int, default=20_000)
    optimization.add_argument("--batch-size", type=int, default=8, help="bin centers per step")
    optimization.add_argument("--lr", type=float, default=1e-2)
    optimization.add_argument("--weight-decay", type=float, default=0.0)
    optimization.add_argument("--spatial-tv-weight", type=float, default=0.0)
    optimization.add_argument("--temporal-tv-weight", type=float, default=0.0)
    optimization.add_argument("--grad-clip-norm", type=float, default=1.0, help="<=0 disables")
    optimization.add_argument("--log-interval", type=int, default=200)
    optimization.add_argument("--eval-interval", type=int, default=2_000, help="0 disables")
    optimization.add_argument("--seed", type=int, default=0)
    optimization.add_argument("--device", default="auto")

    visualization = parser.add_argument_group("visualization")
    visualization.add_argument(
        "--panel-interval",
        type=int,
        default=2_000,
        help="steps between progress snippet videos; 0 disables",
    )
    visualization.add_argument(
        "--snippet-frames",
        type=int,
        default=48,
        help="consecutive frames in each progress video",
    )
    visualization.add_argument(
        "--snippet-start",
        type=int,
        default=None,
        help="first frame of the snippet; defaults to the middle of the sequence",
    )
    visualization.add_argument(
        "--snippet-upsample",
        type=int,
        default=1,
        help=(
            "render this many query times per observation interval; above 1 the "
            "measured columns hold their nearest observation"
        ),
    )
    visualization.add_argument(
        "--residual-scale",
        type=float,
        default=4.0,
        help="gain on the image residual panel",
    )
    visualization.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.num_frames < 2:
        parser.error("--num-frames must be at least 2")

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    workdir = create_workdir(
        args.runs_dir,
        run_name=args.run_name,
        workdir=args.workdir,
        subdirectories=("artifacts", "samples", "tensorboard"),
    )
    print(f"Fit work directory: {workdir}")

    dataset = BouncingBallVideoDataset(
        num_samples=args.num_frames,
        image_size=args.image_size,
        seed=args.data_seed,
        frame_dt=args.frame_dt,
        normalize=False,
        write_video=False,
    )
    data = build_dynamic_kspace(
        dataset.samples,
        sampling_rate=args.sampling_rate,
        family=args.mask_family,
        center_fraction=args.center_fraction,
        noise_std=args.noise_std,
        seed=args.data_seed,
    )
    schedule = build_bin_schedule(
        args.condition,
        num_frames=data.num_frames,
        max_steps=args.max_steps,
        start_width=args.start_width,
        end_width=args.end_width,
        anneal_fraction=args.anneal_fraction,
        kind=args.anneal_kind,
    )
    print(
        f"Per-frame sampling rate: {data.sampling_rate:.3f}, "
        f"union coverage: {data.union_coverage:.3f}, "
        f"coverage at the initial bin: "
        f"{data.coverage_for_window(schedule.half_width_at(0)):.3f}"
    )

    if args.scene_model == "kplanes":
        model_kwargs = dict(
            feature_dim=args.feature_dim,
            resolutions=tuple(int(value) for value in args.resolutions.split(",")),
            time_resolution=args.time_resolution,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        model_kwargs = dict(
            num_features=args.num_features,
            space_scale=args.space_scale,
            time_scale=args.time_scale,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            seed=args.seed,
        )
    model = build_scene_model(
        args.scene_model,
        height=data.image_shape[1],
        width=data.image_shape[2],
        channels=data.image_shape[0],
        **model_kwargs,
    ).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Scene parameters: {total_parameters:,d}")

    data = data.to(device)
    optimizer = torch.optim.Adam(
        model.parameter_groups(args.lr), lr=args.lr, weight_decay=args.weight_decay
    )
    writer = SummaryWriter(log_dir=str(workdir / "tensorboard"))
    save_training_config(
        workdir,
        args,
        resolved={
            "device": str(device),
            "total_parameters": total_parameters,
            "measured_sampling_rate": data.sampling_rate,
            "union_coverage": data.union_coverage,
            "initial_bin_width": schedule.width_at(0),
            "final_bin_width": schedule.width_at(args.max_steps),
        },
    )

    visualizer = ReconstructionVisualizer(
        data=data,
        call_every=args.panel_interval,
        output_dir=workdir / "samples",
        writer=writer,
        snippet_frames=args.snippet_frames,
        snippet_start=args.snippet_start,
        snippet_upsample=args.snippet_upsample,
        fps=1.0 / args.frame_dt,
        residual_scale=args.residual_scale,
        display_scale=max(1, 96 // args.image_size),
    )
    fitter = SceneFitter(
        model=model,
        data=data,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        spatial_tv_weight=args.spatial_tv_weight,
        temporal_tv_weight=args.temporal_tv_weight,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        callbacks=[visualizer],
        writer=writer,
        seed=args.seed,
    )

    try:
        print("Fitting...")
        final_metrics = fitter()
        ground_truth = data.frames.cpu()
        zero_filled = zero_filled_reconstruction(data).cpu()
        temporal_average = temporal_average_reconstruction(
            data, half_width=schedule.half_width_at(0)
        ).cpu()

        results = {
            "scene": final_metrics,
            "zero_filled": image_metrics(zero_filled, ground_truth),
            "temporal_average": image_metrics(temporal_average, ground_truth),
            "condition": args.condition,
            "sampling_rate": data.sampling_rate,
            "union_coverage": data.union_coverage,
        }
        save_json(results, workdir / "results.json")
        print(
            f"PSNR - scene: {results['scene']['psnr_db']:.2f} dB, "
            f"zero-filled: {results['zero_filled']['psnr_db']:.2f} dB, "
            f"temporal average: {results['temporal_average']['psnr_db']:.2f} dB"
        )

        save_scene(model, workdir / "scene_final.pth")
        if args.save_video:
            visualizer.write_sequence_video(fitter, workdir / "artifacts/reconstruction.mp4")
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
