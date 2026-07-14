# Hybrid latent interpolation evaluation

This experiment combines a dense image-space interpolation with the existing
SQUAD path through terminal-noise space. For each requested rectified-flow time
`t_mix`, it constructs

```text
x_t = (1 - t_mix) * x_image_interpolated + t_mix * z_squad
```

and integrates the learned ODE only from `t_mix` to the configured data boundary.

The experiment is intentionally an ablation rather than a theoretically exact
ODE inversion: the linearly composed `x_t` need not lie on the model's ordinary
probability-flow characteristic. The sweep tests whether injecting endpoint-derived
image structure at an intermediate noise level improves temporal consistency or
fidelity relative to pure SQUAD decoding.

## Installation

Copy these files into the evaluation directory:

- `eval_hybrid_latent_interpolation.py`
- the updated `eval.py`, or apply `hybrid_latent_interpolation.patch`
- optionally `test_eval_hybrid_latent_interpolation.py`

## Recommended first run

```bash
python eval.py hybrid \
  --mix-times 0.1,0.25,0.5,0.75,0.9,0.99 \
  --image-methods linear \
  --solver heun \
  --ode-steps 128 \
  --save-tensors
```

Interpretation of `t_mix`:

- Small `t_mix`: image interpolation dominates and only a short generative decode
  remains.
- Large `t_mix`: the state is close to the SQUAD noise path and the model has more
  of the trajectory over which to transform it.
- `t_mix` must lie inside `[data_eps, 1 - t_eps]`.

For a compact sweep around the likely transition region:

```bash
python eval.py hybrid \
  --mix-times 0.5,0.7,0.8,0.9,0.95 \
  --image-methods linear,smoothstep \
  --hard-keyframes
```

`--hard-keyframes` affects only the final reported/video output. It replaces the
sampled keyframe images with their exact observations; missing-frame metrics are
unchanged.

## Image interpolation choices

- `linear`: ordinary pixelwise linear interpolation.
- `smoothstep`: pixelwise interpolation with `u^2(3-2u)`, reducing segment-end
  velocity but not providing cross-segment derivative continuity.
- `catmull-rom`: four-keyframe cubic interpolation. It is clamped to `[0,1]` by
  default; pass `--allow-image-overshoot` to disable clamping.

## Start-state comparison

By default, the evaluator also follows the ordinary pure-SQUAD ODE trajectory to
each `t_mix` and reports the difference between that state and the constructed
hybrid state. This helps distinguish a small correction to the normal SQUAD path
from a large off-trajectory intervention.

This adds one partial SQUAD integration per unique mix time. Disable it with:

```bash
--no-start-state-comparison
```

## Outputs

The command writes under `outputs/eval/hybrid_latent_interpolation/`:

- `hybrid_latent_interpolation.mp4`
- `hybrid_latent_interpolation_metrics.json`
- `hybrid_latent_interpolation_tensors.pt` when `--save-tensors` is used

The video includes:

1. Ground truth
2. Nearest observed frame
3. Deterministic SQUAD baseline
4. Each direct image interpolation baseline
5. Each image/noise hybrid configuration
6. Absolute residual panels for every prediction

Metrics include all-, missing-, and observed-frame fidelity, temporal step and
acceleration statistics, deviation from deterministic SQUAD, the image/noise
weights, remaining ODE step count, and optional starting-state deviation from the
ordinary SQUAD trajectory.
