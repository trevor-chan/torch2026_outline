# Evaluation architecture

The evaluation CLI and targeted diagnostics live in
`src/flow_interpolation/evaluation/`. Reusable mechanics live in `utils/`, while
sequence construction and observation masks live in `data/`.

## Evaluation files

- `evaluation/cli.py`: command parsing and experiment dispatch.
- `evaluation/experiments/roundtrip.py`: cycle, boundary, and batch diagnostics.
- `evaluation/experiments/trajectory.py`: dense encoded-trajectory and density diagnostics.
- `evaluation/experiments/latent.py`: endpoint inversion and latent interpolation.
- `evaluation/experiments/data_consistency.py`: sampling with hard temporal measurements.
- `evaluation/experiments/endpoint_bridge.py`: stochastic endpoint-conditioned bridges.
- `evaluation/experiments/hybrid.py`: mixed image/latent interpolation.

## Shared files

- `utils/flow.py`: model loading, ODE integration, endpoint prediction, and chunking.
- `utils/interpolation.py`: LERP, SLERP, and generalized hyperspherical SQUAD.
- `utils/metrics.py`: image/latent metrics and JSON result serialization.
- `utils/trajectory.py`: subspace, differential-geometry, and path-comparison metrics.
- `utils/trajectory_visualization.py`: static path, residual, and metric plots.
- `utils/visualization.py`: comparison panels and MP4 writing.
- `data/sequences.py`: sparse sequence construction, cadence, and observation masks.

Experiment-specific samplers remain beside the experiments that define their
behavior. Only generally reusable flow integration and interpolation primitives are
shared through `utils`.

## Important behavior changes

### Color walk

The training default now matches `BouncingBallVideoDataset`: `training_color_walk_std=0.1`. For a high-rate sequence, the default per-frame standard deviation is

```text
training_std * sqrt(high_frame_dt / training_frame_dt)
```

so the random-walk variance per unit time stays fixed.

### Background noise

Background noise is controlled with `--background-noise-std`. It is zero-mean
Gaussian noise in `[0,1]` image units, sampled independently for every rendered
frame before compositing the foreground. Unlike the color walk, it models a
per-frame observation perturbation and is therefore not scaled with frame spacing.
Training and evaluation should use the same value. A separate deterministic RNG
stream ensures that a clean/noisy dataset comparison retains the same physical
trajectory and color evolution.

### Endpoint cadence

`training_frame_dt / high_frame_dt` is no longer passed through Python's banker's `round`. The default is explicit half-up nearest-integer rounding. Every run prints:

- requested ratio,
- selected integer stride,
- requested endpoint spacing,
- actual endpoint spacing,
- signed and relative timing error.

Use `--stride-rounding exact` to reject non-integral ratios instead.

For the defaults `0.25 / 0.02 = 12.5`, nearest selects a stride of 13 and reports an actual endpoint spacing of 0.26 s. To avoid approximation entirely, use a divisible high-rate spacing such as `--high-frame-dt 0.025`.

### Epsilon boundary

The implementation preserves the existing epsilon-boundary perturbation. Clean images are moved to

```text
x_eps = (1 - eps) x_0 + eps z
```

before ordinary data-to-noise inversion. Round-trip cycle tests do **not** inject a second perturbation halfway through a cycle.

The focused epsilon ablation uses common random boundary-noise draws to compare
several values without adding Monte Carlo differences between configurations:

```bash
python -m flow_interpolation eval epsilon \
  --epsilons 1e-5,1e-4,1e-3,1e-2 \
  --epsilon-boundary-samples 8 \
  --epsilon-frame-source observed \
  --save-tensors
```

For each epsilon it reports coordinate-wise variance across images, variance across
boundary-noise draws for each fixed image, and their combined variance. The static
maps average these CxHxW tensors over channels and include the coordinate second
moment. For the standard-normal target, coordinate variance, second moment, latent
standard deviation, and radius divided by `sqrt(d)` should all be near one. Paired
latent RMSE measures the total map displacement relative to `--data-eps`;
trajectory-centered RMSE removes the per-draw mean latent, and trajectory-step RMSE
compares adjacent encoded-frame differences. Their raw values are already measured
in prior standard-deviation units, while JSON and CSV also retain normalization by
the corresponding reference trajectory scale. The latter two distinguish a harmless
common shift from a change in the geometry used for interpolation.
`--epsilon-frame-source dense` uses the full high-rate sequence when a stronger
population estimate is worth the extra ODE solves.

The same run compares encoding uncertainty with the temporal latent signal:

```text
V_enc  = E_k E_r ||z_k^(r) - mean_r[z_k^(r)]||_2^2
V_time = E_k ||mean_r[z_(k+1)^(r)] - mean_r[z_k^(r)]||_2^2
SNR_trajectory = V_time / V_enc
```

Both full-dimensional squared-L2 energies and per-coordinate values are saved. Their
ratio is identical. Values above one mean the average adjacent-frame latent motion
is larger than the within-frame ambiguity induced by the boundary perturbation;
values below one mean boundary-seed variability dominates. `V_time` depends on the
frame spacing, so dense and observed-frame runs should not be compared without
accounting for their recorded cadence.

Across-image and boundary-draw variance are conditional summaries:
`E_draw[Var_image(z | draw)]` and `E_image[Var_draw(z | image)]`. They are not
additive components of the combined variance. Also, a dense timeline contains
strongly correlated frames from one process realization, so agreement with the
global Gaussian prior should be judged most directly from per-latent scale and
second moments; estimating full prior calibration requires a broader independent
image sample.

## Commands

```bash
# Both cycle directions
python -m flow_interpolation eval roundtrip --checkpoint bouncing_ball_diffusion_ema.pth

# Compare the dense encoded reference with sparse interpolation paths
python -m flow_interpolation eval trajectory \
  --methods lerp,slerp,squad \
  --keyframe-strides 6,13,26

# Existing endpoint-latent interpolation, now with optional SQUAD
python -m flow_interpolation eval latent --methods slerp,squad

# ISCS-inspired temporal inverse problem: compare iid and SLERP innovations
python -m flow_interpolation eval dc --noise-controls independent,slerp --renoise-mode dds --eta 0.85

# Run all evaluations with one model/sequence load
python -m flow_interpolation eval all --methods slerp,squad --noise-controls independent,slerp
```

Outputs are written under `outputs/eval` by default.

## Interpretation of round-trip metrics

For data → noise → data′, two errors are reported:

1. `cycle_at_data_eps`: compares the decoded result to the exact perturbed input at `data_eps`; this isolates ODE cycle fidelity.
2. `clean_endpoint_estimate`: applies the rectified-flow endpoint identity `x0_hat = x_t - t v_theta(x_t,t)` and compares with the original clean image.

For noise → data → noise′, the decoded `data_eps` state is re-encoded directly, without a fresh epsilon perturbation.

The suite now also reports:

- `encoded_data_latent_to_data_to_latent`: the noise-first composition evaluated on latents that are known to lie in the numerical image of the encoder. Comparing this with a fresh Gaussian cycle distinguishes a generic integrator failure from a latent-distribution/conditioning effect.
- `image_boundary_sweep`: cycles random Gaussian anchors through 90%, 99%, 99.9%, and 100% of the configured noise→image path. A depth of 100% reaches `data_time`; it reaches mathematical `t=0` only when `--data-eps 0` is used.
- `batch_consistency`: evaluates the same anchor alone and inside larger fixed batches, reporting differences in the initial vector field, forward endpoint, and completed cycle.
- `solver_step_sweep`: optional convergence measurements when `--roundtrip-step-counts` is supplied.

Every principal error now includes scale-normalized quantities:

```text
rmse_over_target_std
mse_over_target_variance
rmse_over_target_rms
mae_over_target_mean_abs
```

`RMSE/std` and `MSE/variance` test whether absolute cycle errors simply track the scale of the state distribution. `RMSE/RMS` is also included because it remains well defined for non-zero-mean states and is the global relative-L2 error. Target statistics include the fraction of channel/pixel locations whose standard deviation across the evaluated frames is below `1e-4`, `1e-3`, and `1e-2`, which helps identify nearly deterministic image regions.

Example:

```bash
python -m flow_interpolation eval roundtrip \
  --roundtrip-image-depths 0.9,0.99,0.999,1.0 \
  --roundtrip-batch-sizes 1,2,4,8,16,32 \
  --roundtrip-step-counts 64,128,256,512 \
  --save-tensors
```

The step sweep is omitted by default because it substantially increases runtime.

## Encoded-trajectory diagnostics

The full high-rate ground-truth sequence is encoded with one shared epsilon-boundary
draw by default. This dense encoded path is an empirical oracle for the observed
process, not a claim about the globally optimal path through noise space. The
`trajectory` command, with `geodesic` retained as an alias, reports:

- LERP, SLERP, and SQUAD error by frame, missing frame, and interpolation coordinate;
- residual energy outside each two-endpoint plane;
- residual energy outside the local four-keyframe subspace available to SQUAD;
- radius, radial speed, angular speed, and radial/angular step fractions;
- latent speed, acceleration, turning angle, and discrete curvature;
- interpolator tangent alignment and speed error;
- closest-point timing error `|alpha* - s|` and equivalent frame/time displacement;
- closest-curve orthogonal L2, RMSE, and relative-L2 geometric error;
- a keyframe-density sweep controlled by `--keyframe-strides`.

The JSON contains aggregate and interpolation-coordinate summaries. The CSV contains
all per-frame measurements, and `--save-tensors` retains the encoded reference,
interpolated paths, and raw diagnostic tensors. When no stride list is supplied, the
command uses half, equal, and twice the configured endpoint stride where possible.

Static plotting is enabled by default and writes the following under `trajectory/plots/`:

- `reference_geometry.png`, showing radius, step decomposition, derivatives, and curvature;
- `density_summary.png`, comparing interpolation, tangent, speed, and subspace metrics;
- `timing_geometry_stride_*.png`, comparing the inferred time warp with closest-curve error;
- one `paths_and_residuals_stride_XXXX.png` per requested stride.

The plotted latent paths use one shared two-dimensional PCA basis fitted to the dense
reference and compared methods. This projection is descriptive only: every residual
and summary statistic is still evaluated in the original latent dimension. Use
`--no-plot-paths` for metric-only runs. Use `--decode-paths` to decode the empirical
reference and each interpolated path, add image-space metrics to the JSON, and save
per-stride MP4 comparisons under `trajectory/videos/`. Decoding is opt-in because it
adds one reverse ODE solve for the reference and one for every method and stride.

## Data-consistency sampler

The observed low-rate frames define a temporal masking operator. At every reverse step, the sampler:

1. predicts `x0_hat` and `z_hat` from the current RF state;
2. replaces the observed frames in `x0_hat` with their exact measurements;
3. samples a fresh innovation field, either independent per frame or generated from two Gaussian anchors by one global SLERP path;
4. re-noises to the next time.

The default DDS-like RF adaptation uses

```text
z_mix = sqrt(1 - eta^2) * z_hat + eta * innovation
x_next = (1 - t_next) * x0_dc + t_next * z_mix
```

`--renoise-mode ddpm` instead uses entirely fresh innovations at every step. This is deliberately labeled an ISCS-inspired RF adaptation because ISCS is formulated for score/SDE samplers, not this linear rectified-flow model.

## Suggested first runs

Use Heun and run the boundary/batch diagnostics first:

```bash
python -m flow_interpolation eval roundtrip --solver heun --ode-steps 128
```

Then request a solver convergence sweep in a single run:

```bash
python -m flow_interpolation eval roundtrip --solver heun --ode-steps 128 \
  --roundtrip-step-counts 64,128,256,512
```

After cycle fidelity is acceptable, compare:

```bash
python -m flow_interpolation eval trajectory --methods lerp,slerp,squad --solver heun --ode-steps 256
python -m flow_interpolation eval dc --noise-controls independent,slerp --eta 0.0
python -m flow_interpolation eval dc --noise-controls independent,slerp --eta 0.5
python -m flow_interpolation eval dc --noise-controls independent,slerp --eta 0.85
python -m flow_interpolation eval dc --noise-controls independent,slerp --renoise-mode ddpm
```

## Endpoint-conditioned stochastic SQUAD bridge

The `bridge` command treats the deterministic SQUAD latent path as the central
trajectory and adds uncertainty only inside intervals between observed frames.
The residual envelope is exactly zero at every keyframe.

```bash
python -m flow_interpolation eval bridge \
  --samplers init,iterative \
  --stochasticities 0.05,0.10,0.20 \
  --num-samples 4 \
  --innovation-mode piecewise-slerp \
  --bridge-envelope sine \
  --bridge-strength 0.25 \
  --noise-refresh fixed
```

`init` perturbs the SQUAD terminal bridge once and then performs the normal
deterministic ODE decode. `iterative` additionally guides the inferred terminal
noise back toward the SQUAD bridge during every reverse step. Both use a
variance-preserving residual mix, and both hard-project observed frames at the
output; iterative mode also projects them at every step.

Useful first ablations:

```bash
# Check that eta=0 reproduces (or nearly reproduces) deterministic SQUAD.
python -m flow_interpolation eval bridge --samplers init --stochasticities 0 --num-samples 1

# One-shot uncertainty around the working latent interpolation.
python -m flow_interpolation eval bridge --samplers init --stochasticities 0.02,0.05,0.10

# Test whether repeated guidance helps or over-constrains the trajectory.
python -m flow_interpolation eval bridge --samplers iterative --stochasticities 0,0.05,0.10 \
  --bridge-strength 0.10
```

The outputs are written to `outputs/eval/endpoint_bridge/`. The MP4 shows the
first stochastic sample from each configuration. The JSON reports aggregate
fidelity over all samples plus diversity on missing frames. With
`--save-tensors`, every sample is retained for further analysis.
