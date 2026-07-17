# Flow Interpolation Experiments

This repository studies generative interpolation between sparse observations using a
2D rectified-flow model. The observations may be keyframes from a video or slices
from a 3D volume. The working hypothesis is that paths between encoded noise states
have more predictable geometry than paths constrained directly to an unknown data
manifold.

The current experiments use a synthetic bouncing-ball sequence to compare latent
linear, spherical, and spherical-cubic paths; hybrid image/latent paths; stochastic
endpoint bridges; and data-consistent sampling. The same interfaces are intended to
support volumetric slice interpolation without coupling dataset details to the model
or evaluation implementations.

## Setup

Python 3.10 or newer is required. Install the package and development tools in an
environment with the appropriate PyTorch/CUDA build:

```bash
python -m pip install -e ".[dev]"
```

Package dependencies and test configuration live in `pyproject.toml`; there is no
second requirements file to keep synchronized.

## Commands

Use the package module as the primary entry point:

```bash
python -m flow_interpolation --help
python -m flow_interpolation train --help
python -m flow_interpolation eval --help
```

Each fresh training command creates a new timestamped work directory under
`outputs/runs/`. Use `--run-name` for a readable name; an existing name receives a
numeric suffix instead of being overwritten:

```bash
python -m flow_interpolation train \
  --run-name baseline \
  --max-steps 150000 \
  --checkpoint-interval 5000
```

Each run contains its configuration, TensorBoard events, checkpoints, samples,
dataset preview, and final model exports:

```text
outputs/runs/baseline/
|-- config.json
|-- artifacts/
|-- checkpoints/step_000005000.pt
|-- samples/model/
|-- samples/ema/
|-- tensorboard/
`-- model_ema_final_step_000150000.pth
```

Checkpoints contain the model, optimizer, EMA, completed step, and PyTorch RNG
state. Resume from either a run directory or a specific checkpoint; saved arguments
become defaults, while explicitly supplied options override them:

```bash
python -m flow_interpolation train \
  --resume outputs/runs/baseline \
  --max-steps 200000
```

To train on images with a small independent Gaussian perturbation in the rendered
background, use:

```bash
python -m flow_interpolation train \
  --run-name bouncing_ball_background_noise_std_0p01 \
  --background-noise-std 0.01 \
  --max-steps 150000
```

The standard deviation is expressed in the same `[0,1]` image units as the generated
frames. Noise is sampled independently per frame before the ball and trail are
composited. It uses a separate deterministic random stream, so changing the noise
amplitude does not change the underlying trajectory or color walk. The default is
zero for compatibility with existing runs. For the epsilon diagnostic, evaluate a
noise-trained model once with the default clean inputs and once with the matching
`--background-noise-std`; use separate `--output-dir` values so the runs do not
overwrite one another.

Inspect all runs with:

```bash
tensorboard --logdir outputs/runs
```

MFU uses measured forward-plus-backward model FLOPs and a per-device peak supplied
by `--peak-tflops` (91.1 by default). TensorBoard records loss, validation loss,
gradient norms, learning rate, step time, throughput, MFU, memory, and sample images.

### Minibatch optimal-transport coupling

Independent Gaussian coupling remains the default. Exact quadratic-cost assignment
within each sampled minibatch can be enabled with:

```bash
python -m flow_interpolation train \
  --run-name minibatch-ot \
  --coupling minibatch-ot
```

For data samples `x_i` and one Gaussian minibatch `z_j`, this mode minimizes
`sum_i ||x_i - z_perm(i)||^2` with SciPy's Hungarian solver, then uses the permuted
noise in the otherwise unchanged rectified-flow objective. The solve is exact for
the two empirical minibatches, while repeated minibatch sampling makes it an online
approximation to population optimal transport. The permutation preserves the sampled
Gaussian marginal exactly.

TensorBoard records independent and paired mean-squared transport costs, their ratio,
and the permutation's fixed-point fraction. A ratio below one verifies that assignment
reduced the minibatch transport cost. Pairing is local to each process in distributed
training; it does not gather a global batch. Cost construction is quadratic in batch
size and the Hungarian solve is cubic, so batch-size scaling should be measured.

Evaluation is organized by experiment. For example:

```bash
python -m flow_interpolation eval trajectory \
  --checkpoint outputs/runs/baseline/model_ema_final_step_000150000.pth \
  --methods lerp,slerp,squad \
  --keyframe-strides 6,13,26 \
  --decode-paths \
  --save-tensors
python -m flow_interpolation eval nuisance \
  --checkpoint outputs/runs/baseline/model_ema_final_step_000150000.pth \
  --svd-ranks 0,1,2,4,8,16,32,64 \
  --fourier-harmonics 0,1,2,4,8,16,32,64 \
  --save-tensors
python -m flow_interpolation eval latent --checkpoint outputs/runs/baseline/model_ema_final_step_000150000.pth
python -m flow_interpolation eval hybrid --checkpoint outputs/runs/baseline/model_ema_final_step_000150000.pth
python -m flow_interpolation eval roundtrip --checkpoint outputs/runs/baseline/model_ema_final_step_000150000.pth
python -m flow_interpolation eval epsilon \
  --checkpoint outputs/runs/bouncing_ball_background_noise_std_0p01/model_ema_final_step_000150000.pth \
  --background-noise-std 0.01 \
  --epsilons 1e-5,1e-4,1e-3,1e-2 \
  --epsilon-boundary-samples 8 \
  --save-tensors
```

The trajectory diagnostic encodes the dense generated sequence once and treats that
encoded path as an empirical oracle. It compares sparse LERP, SLERP, and SQUAD paths
across keyframe densities, then reports interpolation error, endpoint-plane and local
four-keyframe subspace residuals, radial/angular motion, speed, acceleration,
curvature, and tangent alignment under `outputs/eval/trajectory/`. Shared boundary
noise is the default so framewise epsilon perturbations do not dominate the measured
trajectory. Each run saves reference-geometry, keyframe-density, and per-stride path
and residual plots under `outputs/eval/trajectory/plots/`. The path view is a shared
2D PCA projection for visualization only; residuals and metrics are calculated in the
full latent space. `--decode-paths` additionally writes decoded comparison videos and
image-space metrics, while `--no-plot-paths` disables static plotting.

The `nuisance` diagnostic asks whether the dense terminal-latent trajectory contains
a low-dimensional or low-frequency physical signal embedded in visually unimportant
variation. It forms the adjacent-difference matrix, applies a mean-preserving centered
SVD and temporal Fourier low-pass sweep, integrates each reconstructed difference
sequence from the original first latent, and decodes every result. Outputs under
`outputs/eval/nuisance/` compare retained latent difference energy with total,
foreground-weighted, and background-weighted decoded image loss. A soft activity
centroid measures whether ball motion survives even when pixel-level detail changes.
The compact `sweep.csv` has one row per rank or cutoff rather than one row per frame.
Full rank and full bandwidth are added automatically as identity controls. Shared
boundary noise is strongly recommended so the test measures trajectory structure
rather than independent epsilon-boundary perturbations.

The `epsilon` diagnostic re-encodes the same images with identical boundary-noise
draws at every requested epsilon. It separates terminal-latent variance across
images from variance across repeated boundary perturbations, saves channel-averaged
variance maps, and reports absolute, trajectory-centered, and trajectory-step RMSE
relative to `--data-eps`. Use `--epsilon-frame-source dense` to estimate the maps
from the full generated sequence rather than only the observed keyframes. Outputs
are written under `outputs/eval/epsilon_ablation/`. Summary plots include the
standard-normal target variance and scale of one. Raw displacement RMSE is therefore
read directly in prior standard-deviation units. The trajectory-SNR plot compares
within-frame variability over boundary draws,
`V_enc = E[k,r] ||z_k^(r) - mean_r z_k^(r)||^2`, with adjacent motion of the
draw-averaged trajectory,
`V_time = E[k] ||mean_r z_(k+1)^(r) - mean_r z_k^(r)||^2`, and reports
`V_time / V_enc`.

Run the focused unit tests with `pytest`.

## Organization

```text
src/flow_interpolation/
|-- data/          datasets, sparse sequences, and observation masks
|-- models/        flow velocity model architectures
|-- training/      training CLI, loop, losses, and callbacks
|-- evaluation/    evaluation CLI and targeted experiment implementations
`-- utils/         flow, interpolation, metrics, rendering, and training utilities
tests/             focused tests for interpolation and evaluation behavior
docs/              experiment rationale and design notes
outputs/           generated checkpoints, tensors, images, and videos (ignored)
```

Training and evaluation share models, datasets, and reusable utilities through
their package APIs, but neither imports the other's CLI. New targeted diagnostics
belong in `evaluation/experiments/`. General flow integration, interpolation math,
metrics, and rendering belong in `utils/`; reusable model components belong in
`models/`.

## Current Scope

- The trained velocity field is a 2D transformer model using rectified-flow loss.
- Evaluation assumes ordered sparse observations and reports reconstruction,
  temporal, path-geometry, and round-trip diagnostics.
- The bouncing-ball dataset is an experimental fixture, not a domain assumption.
- Multi-GPU sharding and MFU reporting remain experimental and hardware-dependent.
