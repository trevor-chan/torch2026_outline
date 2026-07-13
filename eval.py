from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eval_common import (
    FlowSettings,
    ModelSettings,
    build_model,
    resolve_device,
    seed_everything,
    validate_flow_settings,
)
from eval_data import DEFAULT_TRAINING_COLOR_WALK_STD, SequenceData, build_sequence
from eval_data_consistency import run_data_consistency_evaluation
from eval_geodesic import run_latent_geodesic_evaluation
from eval_latent_interpolation import run_latent_interpolation_evaluation
from eval_roundtrip import run_roundtrip_evaluation


def csv_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default="bouncing_ball_diffusion_ema.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--non-strict-load", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--time-embed-dim", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--num-intervals", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=64)
    parser.add_argument("--training-frame-dt", type=float, default=0.25)
    parser.add_argument("--high-frame-dt", type=float, default=0.02)
    parser.add_argument(
        "--training-color-walk-std",
        type=float,
        default=DEFAULT_TRAINING_COLOR_WALK_STD,
        help="Dataset training default is 0.1.",
    )
    parser.add_argument(
        "--color-walk-std",
        type=float,
        default=None,
        help="Override the high-rate per-frame std. By default it is variance-scaled from training.",
    )
    parser.add_argument(
        "--stride-rounding",
        choices=("nearest", "floor", "ceil", "exact"),
        default="nearest",
        help="Nearest uses explicit half-up rounding and reports the resulting timing error.",
    )

    parser.add_argument("--ode-steps", type=int, default=128)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--data-eps", type=float, default=1e-3)
    parser.add_argument("--t-eps", type=float, default=1e-3)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs/eval")


def add_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods",
        type=csv_list,
        default=["slerp", "squad"],
        help="Comma-separated subset of slerp,squad,lerp.",
    )
    parser.add_argument(
        "--slerp-mode",
        choices=("iscs", "radius-lerp"),
        default="iscs",
    )
    parser.add_argument(
        "--boundary-noise-mode",
        choices=("shared", "independent"),
        default="shared",
        help="Shared preserves the existing common epsilon-boundary perturbation across a timeline.",
    )


def add_video_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--display-scale", type=int, default=8)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=4.0)
    parser.add_argument("--save-tensors", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_arguments(common)

    parser = argparse.ArgumentParser(
        description="Evaluation suite for rectified-flow temporal/through-plane interpolation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    roundtrip = subparsers.add_parser("roundtrip", parents=[common])
    roundtrip.add_argument("--roundtrip-samples", type=int, default=32)
    roundtrip.add_argument(
        "--boundary-noise-mode", choices=("shared", "independent"), default="shared"
    )
    roundtrip.add_argument("--save-tensors", action="store_true")

    geodesic = subparsers.add_parser("geodesic", parents=[common])
    add_geometry_arguments(geodesic)
    geodesic.add_argument("--save-tensors", action="store_true")

    latent = subparsers.add_parser("latent", parents=[common])
    add_geometry_arguments(latent)
    add_video_arguments(latent)

    dc = subparsers.add_parser("dc", parents=[common])
    dc.add_argument(
        "--noise-controls",
        type=csv_list,
        default=["independent", "slerp"],
        help="Comma-separated subset of independent,slerp.",
    )
    dc.add_argument("--renoise-mode", choices=("dds", "ddpm"), default="dds")
    dc.add_argument("--eta", type=float, default=0.85)
    dc.add_argument("--dc-strength", type=float, default=1.0)
    dc.add_argument("--slerp-mode", choices=("iscs", "radius-lerp"), default="iscs")
    dc.add_argument("--clip-x0", action="store_true")
    add_video_arguments(dc)

    all_parser = subparsers.add_parser("all", parents=[common])
    add_geometry_arguments(all_parser)
    add_video_arguments(all_parser)
    all_parser.add_argument("--roundtrip-samples", type=int, default=32)
    all_parser.add_argument(
        "--noise-controls", type=csv_list, default=["independent", "slerp"]
    )
    all_parser.add_argument("--renoise-mode", choices=("dds", "ddpm"), default="dds")
    all_parser.add_argument("--eta", type=float, default=0.85)
    all_parser.add_argument("--dc-strength", type=float, default=1.0)
    all_parser.add_argument("--clip-x0", action="store_true")
    return parser


def make_flow(args: argparse.Namespace) -> FlowSettings:
    settings = FlowSettings(
        data_time=args.data_eps,
        noise_time=1.0 - args.t_eps,
        ode_steps=args.ode_steps,
        solver=args.solver,
        encode_batch_size=args.encode_batch_size,
        decode_batch_size=args.decode_batch_size,
    )
    validate_flow_settings(settings)
    return settings


def make_model_settings(args: argparse.Namespace) -> ModelSettings:
    return ModelSettings(
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        time_embed_dim=args.time_embed_dim,
        rope_theta=args.rope_theta,
        strict_load=not args.non_strict_load,
        compile_model=args.compile,
    )


def make_sequence(args: argparse.Namespace) -> SequenceData:
    return build_sequence(
        image_size=args.image_size,
        seed=args.seed,
        start_index=args.start_index,
        num_intervals=args.num_intervals,
        training_frame_dt=args.training_frame_dt,
        high_frame_dt=args.high_frame_dt,
        training_color_walk_std=args.training_color_walk_std,
        color_walk_std=args.color_walk_std,
        stride_rounding=args.stride_rounding,
    )


def validate_methods(methods: list[str]) -> None:
    unknown = set(methods) - {"slerp", "squad", "lerp"}
    if unknown:
        raise ValueError(f"Unknown interpolation method(s): {sorted(unknown)}")


def validate_noise_controls(noise_controls: list[str]) -> None:
    unknown = set(noise_controls) - {"independent", "slerp"}
    if unknown:
        raise ValueError(f"Unknown noise control(s): {sorted(unknown)}")


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    flow = make_flow(args)
    sequence = make_sequence(args)
    model = build_model(make_model_settings(args), device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.command in {"geodesic", "latent", "all"}:
        validate_methods(args.methods)
    if args.command in {"dc", "all"}:
        validate_noise_controls(args.noise_controls)

    if args.command in {"roundtrip", "all"}:
        roundtrip_dir = output_root / "roundtrip"
        run_roundtrip_evaluation(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            num_samples=args.roundtrip_samples,
            boundary_noise_mode=args.boundary_noise_mode,
            seed=args.seed + 101,
            output_json=str(roundtrip_dir / "metrics.json"),
            output_tensors=str(roundtrip_dir / "tensors.pt") if args.save_tensors else None,
        )

    if args.command in {"geodesic", "all"}:
        geodesic_dir = output_root / "geodesic"
        run_latent_geodesic_evaluation(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            methods=args.methods,
            slerp_mode=args.slerp_mode,
            boundary_noise_mode=args.boundary_noise_mode,
            seed=args.seed + 202,
            output_json=str(geodesic_dir / "metrics.json"),
            output_csv=str(geodesic_dir / "per_frame.csv"),
            output_tensors=str(geodesic_dir / "tensors.pt") if args.save_tensors else None,
        )

    if args.command in {"latent", "all"}:
        run_latent_interpolation_evaluation(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            methods=args.methods,
            slerp_mode=args.slerp_mode,
            boundary_noise_mode=args.boundary_noise_mode,
            seed=args.seed + 303,
            output_dir=str(output_root / "latent"),
            video_fps=args.video_fps,
            display_scale=args.display_scale,
            gap=args.gap,
            residual_scale=args.residual_scale,
            save_tensors=args.save_tensors,
        )

    if args.command in {"dc", "all"}:
        run_data_consistency_evaluation(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            noise_controls=args.noise_controls,
            renoise_mode=args.renoise_mode,
            eta=args.eta,
            dc_strength=args.dc_strength,
            slerp_mode=args.slerp_mode,
            seed=args.seed + 404,
            clip_x0=args.clip_x0,
            output_dir=str(output_root / "data_consistency"),
            video_fps=args.video_fps,
            display_scale=args.display_scale,
            gap=args.gap,
            residual_scale=args.residual_scale,
            save_tensors=args.save_tensors,
        )


if __name__ == "__main__":
    main()
