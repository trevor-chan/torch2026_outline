# Evaluation refactor

Place these files in the repository root on the `slerp_interp` branch. They import the existing `dataset.py` and `transformer.py` directly.

## Files

- `eval.py`: unified command-line interface.
- `eval_common.py`: checkpoint loading, ODE integration, epsilon-boundary handling, encoding/decoding, and metrics.
- `eval_data.py`: synthetic sequence creation, color-walk scaling, and explicit cadence resolution.
- `eval_geometry.py`: LERP, ISCS-style SLERP, radius-preserving SLERP, and generalized hyperspherical SQUAD.
- `eval_roundtrip.py`: data → noise → data′ and noise → data → noise′ cycle tests.
- `eval_geodesic.py`: latent-geodesic errors against the encoded high-rate ground-truth trajectory.
- `eval_latent_interpolation.py`: endpoint inversion, SLERP/SQUAD interpolation, and deterministic decoding.
- `eval_data_consistency.py`: ISCS-inspired rectified-flow posterior sampling with hard temporal measurement consistency.
- `eval_visualization.py`: MP4 comparison panels.

## Important behavior changes

### Color walk

The training default now matches `BouncingBallVideoDataset`: `training_color_walk_std=0.1`. For a high-rate sequence, the default per-frame standard deviation is

```text
training_std * sqrt(high_frame_dt / training_frame_dt)
```

so the random-walk variance per unit time stays fixed.

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

## Commands

```bash
# Both cycle directions
python eval.py roundtrip --checkpoint bouncing_ball_diffusion_ema.pth

# Compare encoded ground-truth latents with SLERP and SQUAD paths
python eval.py geodesic --methods slerp,squad

# Existing endpoint-latent interpolation, now with optional SQUAD
python eval.py latent --methods slerp,squad

# ISCS-inspired temporal inverse problem: compare iid and SLERP innovations
python eval.py dc --noise-controls independent,slerp --renoise-mode dds --eta 0.85

# Run all evaluations with one model/sequence load
python eval.py all --methods slerp,squad --noise-controls independent,slerp
```

Outputs are written under `outputs/eval` by default.

## Interpretation of round-trip metrics

For data → noise → data′, two errors are reported:

1. `cycle_at_data_eps`: compares the decoded result to the exact perturbed input at `data_eps`; this isolates ODE/model cycle fidelity.
2. `clean_endpoint_estimate`: applies the rectified-flow endpoint identity `x0_hat = x_t - t v_theta(x_t,t)` and compares with the original clean image.

For noise → data → noise′, the decoded `data_eps` state is re-encoded directly, without a fresh epsilon perturbation.

## Latent-geodesic metrics

The full high-rate ground-truth sequence is encoded with one shared epsilon-boundary draw by default. For every frame, the CSV records:

- relative L2 error,
- latent RMSE,
- cosine similarity,
- angular error in degrees,
- relative radius error.

Summaries are reported separately for all frames and missing/intermediate frames.

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

Use Heun, then sweep integration resolution:

```bash
python eval.py roundtrip --solver heun --ode-steps 128
python eval.py roundtrip --solver heun --ode-steps 256
python eval.py roundtrip --solver heun --ode-steps 512
```

After cycle fidelity is acceptable, compare:

```bash
python eval.py geodesic --methods lerp,slerp,squad --solver heun --ode-steps 256
python eval.py dc --noise-controls independent,slerp --eta 0.0
python eval.py dc --noise-controls independent,slerp --eta 0.5
python eval.py dc --noise-controls independent,slerp --eta 0.85
python eval.py dc --noise-controls independent,slerp --renoise-mode ddpm
```
