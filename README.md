# Continuous-Time Reconstruction from Sparse Dynamic k-Space

This repository fits an implicit scene model `s_theta(x, y, t)` directly to
sparse, temporally-localized k-space observations of a moving scene, so that the
scene can be rendered at any continuous time — including instants where nothing
was measured. The motivating setting is dynamic MRI, where filling a full
Cartesian grid takes longer than the motion allows.

See [PLAN.md](PLAN.md) for the problem statement, the progressive temporal
binning hypothesis, and the experiment plan.

## Setup

Python 3.10 or newer is required. Install the package and development tools in an
environment with the appropriate PyTorch/CUDA build:

```bash
python -m pip install -e ".[dev]"
```

## Commands

```bash
python -m flow_interpolation --help
python -m flow_interpolation fit --help     # scene fitting (the current work)
python -m flow_interpolation train --help   # rectified-flow training (retained)
```

### Fitting a scene

Each `fit` command simulates a bouncing-ball sequence, subsamples its k-space,
and fits a scene model to the result under a temporal binning schedule. Runs
land in a fresh timestamped directory under `outputs/fits/`:

```bash
python -m flow_interpolation fit \
  --run-name curriculum \
  --num-frames 200 --image-size 32 --sampling-rate 0.1 \
  --condition curriculum --start-width 25 --end-width 1 \
  --max-steps 20000
```

The three conditions of the main experiment:

```bash
python -m flow_interpolation fit --run-name wide       --condition wide       --start-width 25
python -m flow_interpolation fit --run-name narrow     --condition narrow
python -m flow_interpolation fit --run-name curriculum --condition curriculum --start-width 25
```

`wide` and `narrow` hold the bin width fixed; `curriculum` anneals from
`--start-width` to `--end-width` over `--anneal-fraction` of the run and then
holds, so all three finish on comparable objectives. `--anneal-kind` selects
linear, exponential (default), or stepped narrowing.

Each run writes `config.json`, `results.json` (scene PSNR alongside zero-filled
and temporal-average baselines), TensorBoard events, the fitted scene weights,
progress videos under `samples/`, and a full-sequence panel video under
`artifacts/`.

```bash
tensorboard --logdir outputs/fits
```

### Progress videos

Every `--panel-interval` steps the fit writes a short comparison video to
`samples/snippet_<step>.mp4`. Every video frame is a 2x4 panel:

```text
             ground truth      measured         fitted           residual
image    |  dense frame    |  zero-filled   |  s_theta(t)    |  |fit - truth| |
k-space  |  full spectrum  |  sparse M . k  |  F[s_theta(t)] |  |dK|          |
```

Reading both rows together is the point. A reconstruction that looks plausible
but is quietly ignoring the measurements shows up as structure in the k-space
residual at *observed* frequencies; over-smoothing shows up as missing energy at
high radius. Neither is visible from the image row alone.

Motion is what is being reconstructed, so the output is a video rather than
stills: jitter, a lagging or smeared ball, and the trail detaching are obvious
in motion and nearly invisible frame by frame. `--snippet-frames` sets the
length of the window and `--snippet-start` its position; the window is fixed
across the run (centered by default, away from the one-sided bins at the
sequence boundaries) so successive videos are directly comparable.

`--snippet-upsample N` renders `N` query times per observation interval, which
is the only way to actually see what a continuous-time model claims — the fit
column moves smoothly while the measured columns step. Playback speeds up to
match, so the snippet still runs at the acquisition's real rate. Between
observation times the measured columns can only hold their nearest observation,
so the residual column there mixes reconstruction error with real motion; read
it strictly on the frames that were sampled.

TensorBoard gets the middle frame of each snippet under `reconstruction/panels`
rather than the video itself, since `add_video` requires moviepy and this
project depends only on `imageio-ffmpeg`.

All k-space panels share one log-magnitude map, pinned to the ground-truth peak
and spanning roughly 60 dB, so brightness is comparable across columns and
stable across steps — per-panel autoscaling would make a fit appear to converge
merely because its own dynamic range shifted. Unobserved frequencies are exactly
black, so column 2 doubles as a view of the sampling mask. `--residual-scale`
sets the gain on the image residual only; the k-space residual uses the shared
map.

### Measurement simulation

`--sampling-rate` sets the per-frame fraction of the k-space grid retained.
`--center-fraction` forces a fully sampled block around DC, and `--noise-std`
adds complex measurement noise. `--mask-family` picks the sampling pattern:

| Family | Pattern |
|---|---|
| `uniform` | Uniformly random points. |
| `variable-density` | Density falls off as `(1 + r)^-decay` from DC. |
| `poisson` | Radial density follows a Poisson pmf; a much harder center bias. |
| `cartesian` | Fully sampled parallel lines (phase-encode undersampling). |
| `radial` | Golden-angle spokes through DC. |
| `without-replacement` | Points dealt across frames so coverage over time is even. |

The first five are drawn independently per frame. `without-replacement` is
generated at the sequence level instead: points are dealt from a shuffled pool,
so every frame still gets the same number of samples but no frequency repeats
until all the others have been used.

That distinction matters because independent draws waste samples. By the
coupon-collector argument, a window of `1 / sampling_rate` independent frames
covers only about `1 - 1/e` of k-space; the same window of dealt frames covers
all of it. Measured over 60 frames at a 10% rate:

| Family | Mean radius | Window coverage, half-width 0 / 2 / 5 / 10 |
|---|---|---|
| `uniform` | 0.749 | 0.100 / 0.402 / 0.670 / 0.865 |
| `variable-density` | 0.599 | 0.100 / 0.383 / 0.620 / 0.801 |
| `poisson` | 0.362 | 0.100 / 0.252 / 0.339 / 0.412 |
| `without-replacement` | 0.749 | 0.100 / 0.472 / 0.866 / **1.000** |

`without-replacement` has per-frame statistics identical to `uniform` and
differs only in how the samples are distributed over time, which makes the pair
a clean A/B on temporal sampling structure alone. Its `decay` biases the *order*
of the deal toward low frequencies without weakening the full-cycle coverage
guarantee, since every point is still dealt exactly once per cycle.

`poisson` trades the opposite way: its window coverage saturates low because it
keeps resampling the center, but the frequencies it does sample carry most of
the energy. Its `lam` sets where the density peaks — the Poisson pmf is
maximized at `k = floor(lam)`, so `lam <= 1` (the default) decays monotonically
from DC, and larger values deliberately peak on an annulus instead.

### Scene models

`--scene-model kplanes` (default) uses factorized xy/xt/yt feature planes;
`--scene-model fourier-mlp` uses a coordinate MLP on random Fourier features
with separate spatial and temporal bandwidths. The K-planes factorization is
faster to fit but its low-rank structure is itself a temporal smoother, so the
Fourier MLP is the cleaner control for isolating what binning contributes.

Both render unbounded values rather than passing through a sigmoid: this scene
is mostly dark background, and a saturating head collapses to zero early in
optimization with no gradient left to recover.

Run the unit tests with `pytest`.

## Organization

```text
src/flow_interpolation/
|-- data/          the bouncing-ball sequence used as the dynamic scene
|-- kspace/        Fourier forward model, sampling masks, simulated observations
|-- scene/         implicit scene models, binning schedules, losses, fitting CLI
|-- models/        transformer velocity model (retained for a future prior)
|-- training/      rectified-flow training CLI, loop, losses, and callbacks
`-- utils/         metrics, rendering, and training utilities
tests/             focused tests for the forward model and scene fitting
outputs/           generated checkpoints, tensors, images, and videos (ignored)
```

The rectified-flow training stack is retained but not on the current path: a
learned generative prior is a plausible later addition once the binning question
is settled, so the model, loss, and training scaffolding stay.

## Current Scope

- Reconstruction is per-scene optimization, not supervised learning; there is no
  train/test split over scenes and no generalization claim.
- The fixture assumes zero image phase, one coil, independent grid-point
  sampling, and no field inhomogeneity. Each is a place it diverges from a real
  acquisition; the forward operator does not hard-code the real-valued
  assumption, so a complex scene model can be swapped in.
- The bouncing-ball dataset is an experimental fixture, not a domain assumption.
