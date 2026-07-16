from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flow_interpolation.data import DEFAULT_TRAINING_COLOR_WALK_STD, SequenceData, build_sequence
from flow_interpolation.evaluation.experiments.data_consistency import run_data_consistency_evaluation
from flow_interpolation.evaluation.experiments.endpoint_bridge import run_endpoint_bridge_evaluation
from flow_interpolation.evaluation.experiments.epsilon import run_epsilon_ablation
from flow_interpolation.evaluation.experiments.trajectory import run_trajectory_analysis
from flow_interpolation.evaluation.experiments.hybrid import (
    IMAGE_INTERPOLATION_METHODS,
    run_hybrid_latent_interpolation_evaluation,
)
from flow_interpolation.evaluation.experiments.latent import run_latent_interpolation_evaluation
from flow_interpolation.evaluation.experiments.roundtrip import run_roundtrip_evaluation
from flow_interpolation.utils.flow import (
    FlowSettings,
    ModelSettings,
    build_model,
    resolve_device,
    seed_everything,
    validate_flow_settings,
)


def csv_float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floating-point values.") from error
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def csv_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def csv_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integer values.") from error
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default="outputs/checkpoints/bouncing_ball_diffusion_ema_step_150000.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--non-strict-load", action="store_true")
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--time-embed-dim", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--num-intervals", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=512)
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
        "--background-noise-std",
        type=float,
        default=0.0,
        help=(
            "Match the model's training-time Gaussian background noise in image-space "
            "units. Noise is independent per rendered frame."
        ),
    )
    parser.add_argument(
        "--stride-rounding",
        choices=("nearest", "floor", "ceil", "exact"),
        default="nearest",
        help="Nearest uses explicit half-up rounding and reports the resulting timing error.",
    )

    parser.add_argument("--ode-steps", type=int, default=256)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--data-eps", type=float, default=1e-4)
    parser.add_argument("--t-eps", type=float, default=1e-5)
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


def add_trajectory_arguments(parser: argparse.ArgumentParser) -> None:
    add_geometry_arguments(parser)
    parser.add_argument(
        "--keyframe-strides",
        type=csv_int_list,
        default=None,
        help=(
            "Dense-frame strides for the keyframe-density sweep. By default uses "
            "half, equal, and twice the configured endpoint stride."
        ),
    )
    parser.add_argument(
        "--plot-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save PCA path, residual, geometry, and density-summary plots.",
    )
    parser.add_argument(
        "--decode-paths",
        action="store_true",
        help="Decode each latent path and save image-space comparison videos and metrics.",
    )
    add_render_arguments(parser)
    parser.add_argument("--save-tensors", action="store_true")


def add_render_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--display-scale", type=int, default=8)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=4.0)


def add_video_arguments(parser: argparse.ArgumentParser) -> None:
    add_render_arguments(parser)
    parser.add_argument("--save-tensors", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_arguments(common)

    parser = argparse.ArgumentParser(
        prog="python -m flow_interpolation eval",
        description="Evaluation suite for rectified-flow temporal/through-plane interpolation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    roundtrip = subparsers.add_parser("roundtrip", parents=[common])
    roundtrip.add_argument("--roundtrip-samples", type=int, default=32)
    roundtrip.add_argument(
        "--roundtrip-image-depths",
        type=csv_float_list,
        default=[0.9, 0.99, 0.999, 1.0],
        help=(
            "Comma-separated fractions of the noise->data integration path for the "
            "image-boundary sweep. Defaults to 90%,99%,99.9%,100%."
        ),
    )
    roundtrip.add_argument(
        "--roundtrip-batch-sizes",
        type=csv_int_list,
        default=[1, 2, 4, 8, 16, 32],
        help="Comma-separated batch sizes for the batch-composition sanity check.",
    )
    roundtrip.add_argument(
        "--roundtrip-step-counts",
        type=csv_int_list,
        default=None,
        help=(
            "Optional fixed-step convergence sweep, for example 64,128,256,512. "
            "Omitted by default because it adds substantial runtime."
        ),
    )
    roundtrip.add_argument(
        "--boundary-noise-mode", choices=("shared", "independent"), default="shared"
    )
    roundtrip.add_argument("--save-tensors", action="store_true")

    epsilon = subparsers.add_parser(
        "epsilon",
        parents=[common],
        help="Ablate the image-boundary epsilon used for data-to-noise inversion.",
    )
    epsilon.add_argument(
        "--epsilons",
        type=csv_float_list,
        default=[1e-5, 1e-4, 1e-3, 1e-2],
        help=(
            "Comma-separated boundary epsilons. The configured --data-eps is "
            "automatically included and used as the comparison reference."
        ),
    )
    epsilon.add_argument(
        "--epsilon-boundary-samples",
        type=int,
        default=8,
        help="Number of common boundary-noise draws used at every epsilon.",
    )
    epsilon.add_argument(
        "--epsilon-frame-source",
        choices=("observed", "dense"),
        default="observed",
        help="Encode sparse observed keyframes or the full dense generated sequence.",
    )
    epsilon.add_argument(
        "--boundary-noise-mode",
        choices=("shared", "independent"),
        default="shared",
        help="Shared uses one boundary perturbation field for every frame in a draw.",
    )
    epsilon.add_argument("--save-tensors", action="store_true")

    trajectory = subparsers.add_parser(
        "trajectory",
        aliases=["geodesic"],
        parents=[common],
        help="Compare dense encoded trajectories with sparse latent interpolators.",
    )
    add_trajectory_arguments(trajectory)

    latent = subparsers.add_parser("latent", parents=[common])
    add_geometry_arguments(latent)
    latent.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help=(
            "Fraction of the configured data-to-noise ODE interval traversed before "
            "latent interpolation. 0 interpolates at data_eps; 1 uses noise_time."
        ),
    )
    add_video_arguments(latent)

    hybrid = subparsers.add_parser(
        "hybrid",
        parents=[common],
        help="Mix image interpolation and a SQUAD noise path at an intermediate RF time.",
    )
    hybrid.add_argument(
        "--mix-times",
        type=csv_float_list,
        default=[0.25, 0.5, 0.75, 0.9],
        help=(
            "Comma-separated RF times. Each start state is "
            "x_t=(1-t)*x_image+t*z_squad, then decoded from t to data_time."
        ),
    )
    hybrid.add_argument(
        "--image-methods",
        type=csv_list,
        default=["linear"],
        help="Comma-separated subset of linear,smoothstep,catmull-rom.",
    )
    hybrid.add_argument("--slerp-mode", choices=("iscs", "radius-lerp"), default="iscs")
    hybrid.add_argument(
        "--boundary-noise-mode", choices=("shared", "independent"), default="shared"
    )
    hybrid.add_argument(
        "--hard-keyframes",
        action="store_true",
        help="Replace generated keyframe outputs with the exact observed images.",
    )
    hybrid.add_argument(
        "--allow-image-overshoot",
        action="store_true",
        help="Do not clamp cubic image interpolation to [0,1] before hybrid composition.",
    )
    hybrid.add_argument(
        "--no-start-state-comparison",
        action="store_true",
        help="Skip tracing the ordinary SQUAD ODE path to each mix time.",
    )
    add_video_arguments(hybrid)

    bridge = subparsers.add_parser("bridge", parents=[common])
    bridge.add_argument(
        "--samplers",
        type=csv_list,
        default=["init", "iterative"],
        help="Comma-separated subset of init,iterative.",
    )
    bridge.add_argument(
        "--stochasticities",
        type=csv_float_list,
        default=[0.1],
        help="Comma-separated bridge residual amplitudes in [0,1].",
    )
    bridge.add_argument("--num-samples", type=int, default=4)
    bridge.add_argument(
        "--innovation-mode",
        choices=("independent", "piecewise-slerp", "global-slerp", "segment-shared"),
        default="piecewise-slerp",
    )
    bridge.add_argument(
        "--bridge-envelope",
        choices=("sine", "brownian", "quadratic"),
        default="sine",
    )
    bridge.add_argument("--bridge-strength", type=float, default=0.25)
    bridge.add_argument("--bridge-power", type=float, default=1.0)
    bridge.add_argument("--noise-power", type=float, default=1.0)
    bridge.add_argument("--bridge-blend", choices=("slerp", "lerp"), default="slerp")
    bridge.add_argument("--noise-refresh", choices=("fixed", "fresh"), default="fixed")
    bridge.add_argument("--dc-strength", type=float, default=1.0)
    bridge.add_argument("--slerp-mode", choices=("iscs", "radius-lerp"), default="iscs")
    bridge.add_argument(
        "--boundary-noise-mode", choices=("shared", "independent"), default="shared"
    )
    bridge.add_argument("--clip-x0", action="store_true")
    add_video_arguments(bridge)

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
    all_parser.add_argument("--tau", type=float, default=1.0)
    all_parser.add_argument("--keyframe-strides", type=csv_int_list, default=None)
    add_video_arguments(all_parser)
    all_parser.add_argument(
        "--plot-paths", action=argparse.BooleanOptionalAction, default=True
    )
    all_parser.add_argument("--decode-paths", action="store_true")
    all_parser.add_argument("--roundtrip-samples", type=int, default=32)
    all_parser.add_argument(
        "--roundtrip-image-depths",
        type=csv_float_list,
        default=[0.9, 0.99, 0.999, 1.0],
    )
    all_parser.add_argument(
        "--roundtrip-batch-sizes",
        type=csv_int_list,
        default=[1, 2, 4, 8, 16, 32],
    )
    all_parser.add_argument(
        "--roundtrip-step-counts",
        type=csv_int_list,
        default=None,
    )
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
        background_noise_std=args.background_noise_std,
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.background_noise_std < 0.0:
        raise ValueError("--background-noise-std must be non-negative")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    flow = make_flow(args)
    sequence = make_sequence(args)
    model = build_model(make_model_settings(args), device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.command in {"trajectory", "geodesic", "latent", "all"}:
        validate_methods(args.methods)
    if args.command in {"latent", "all"} and not 0.0 <= args.tau <= 1.0:
        raise ValueError("--tau must lie in [0, 1]")
    if args.command == "hybrid":
        unknown_image_methods = set(args.image_methods) - IMAGE_INTERPOLATION_METHODS
        if unknown_image_methods:
            raise ValueError(
                f"Unknown image interpolation method(s): {sorted(unknown_image_methods)}"
            )
        if any(value < flow.data_time or value > flow.noise_time for value in args.mix_times):
            raise ValueError(
                "Every hybrid mix time must be inside the configured flow interval "
                f"[{flow.data_time}, {flow.noise_time}]"
            )
    if args.command in {"dc", "all"}:
        validate_noise_controls(args.noise_controls)
    if args.command == "bridge":
        unknown_samplers = set(args.samplers) - {"init", "iterative"}
        if unknown_samplers:
            raise ValueError(f"Unknown bridge sampler(s): {sorted(unknown_samplers)}")
        if any(value < 0.0 or value > 1.0 for value in args.stochasticities):
            raise ValueError("All stochasticities must be in [0, 1]")

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
            image_depths=args.roundtrip_image_depths,
            batch_sizes=args.roundtrip_batch_sizes,
            step_counts=args.roundtrip_step_counts,
        )

    if args.command == "epsilon":
        run_epsilon_ablation(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            epsilons=args.epsilons,
            num_boundary_samples=args.epsilon_boundary_samples,
            boundary_noise_mode=args.boundary_noise_mode,
            frame_source=args.epsilon_frame_source,
            seed=args.seed + 151,
            output_dir=str(output_root / "epsilon_ablation"),
            save_tensors=args.save_tensors,
        )

    if args.command in {"trajectory", "geodesic", "all"}:
        trajectory_dir = output_root / "trajectory"
        run_trajectory_analysis(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            methods=args.methods,
            slerp_mode=args.slerp_mode,
            boundary_noise_mode=args.boundary_noise_mode,
            seed=args.seed + 202,
            output_json=str(trajectory_dir / "metrics.json"),
            output_csv=str(trajectory_dir / "per_frame.csv"),
            output_tensors=str(trajectory_dir / "tensors.pt") if args.save_tensors else None,
            keyframe_strides=args.keyframe_strides,
            plot_paths=args.plot_paths,
            decode_paths=args.decode_paths,
            video_fps=args.video_fps,
            display_scale=args.display_scale,
            gap=args.gap,
            residual_scale=args.residual_scale,
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
            tau=args.tau,
            seed=args.seed + 303,
            output_dir=str(output_root / "latent"),
            video_fps=args.video_fps,
            display_scale=args.display_scale,
            gap=args.gap,
            residual_scale=args.residual_scale,
            save_tensors=args.save_tensors,
        )

    if args.command == "hybrid":
        run_hybrid_latent_interpolation_evaluation(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            mix_times=args.mix_times,
            image_methods=args.image_methods,
            slerp_mode=args.slerp_mode,
            boundary_noise_mode=args.boundary_noise_mode,
            seed=args.seed + 353,
            hard_keyframes=args.hard_keyframes,
            clamp_image_interpolation=not args.allow_image_overshoot,
            compare_start_states=not args.no_start_state_comparison,
            output_dir=str(output_root / "hybrid_latent_interpolation"),
            video_fps=args.video_fps,
            display_scale=args.display_scale,
            gap=args.gap,
            residual_scale=args.residual_scale,
            save_tensors=args.save_tensors,
        )

    if args.command == "bridge":
        run_endpoint_bridge_evaluation(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            samplers=args.samplers,
            stochasticities=args.stochasticities,
            num_samples=args.num_samples,
            innovation_mode=args.innovation_mode,
            envelope_kind=args.bridge_envelope,
            bridge_strength=args.bridge_strength,
            bridge_power=args.bridge_power,
            noise_power=args.noise_power,
            bridge_blend=args.bridge_blend,
            noise_refresh=args.noise_refresh,
            dc_strength=args.dc_strength,
            slerp_mode=args.slerp_mode,
            boundary_noise_mode=args.boundary_noise_mode,
            seed=args.seed + 505,
            clip_x0=args.clip_x0,
            output_dir=str(output_root / "endpoint_bridge"),
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
