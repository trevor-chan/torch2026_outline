# Continuous-Time Scene Reconstruction from Sparse Dynamic k-Space

## Problem

Given a series of temporally-localized, spatially-sparse k-space observations
`{k_t}` of a moving scene, reconstruct the continuous-time image `f(x, y, t)`.
The acquisition is assumed too slow to fill a full Cartesian grid inside the
motion-limited time window, so no single time point has a complete measurement.
Every frame is individually under-determined; the reconstruction has to borrow
information across time.

## Approach

Fit an implicit scene model `s_theta(x, y, t) -> RGB` directly to the
measurements. There is no training set and no generalization requirement: this
is per-scene optimization (a NeRF-style inverse problem, not supervised
learning). Each optimization step renders the scene on the pixel grid at one or
more query times, applies a 2D FFT, and compares the result to the observed
k-space samples at the spatial frequencies where the two overlap.

```
loss(theta) = E_t || M_t . F[ s_theta(., ., t) ] - k_t ||^2
```

where `M_t` is the binary sampling mask for the observation at time `t`, `F` is
the centered 2D DFT, and the norm is over complex residuals at sampled
frequencies only.

The implicit model supplies the regularization that the measurements lack. Its
architecture (smooth coordinate MLP, factorized feature planes) determines what
kind of spatiotemporal continuity is assumed, and explicit priors (spatial and
temporal total variation, later a learned generative prior) can be layered on.

### Progressive temporal binning

The core methodological question of the first experiment. Naively matching
`s_theta(t)` against only `k_t` gives each rendered frame ~10% of the
frequencies it needs, so early optimization is badly ill-posed and the model can
settle into a temporally-inconsistent solution.

Instead, compare the model at time `t` against the pool of observations falling
in a window `[t - w/2, t + w/2]`:

```
loss_bin(theta, t, w) = (1/|B|) sum_{j in B(t, w)} || M_j . F[ s_theta(t) ] - k_j ||^2
```

A wide bin asks a single rendered frame to explain many frames' worth of
frequencies. The union of the masks over the bin covers a much larger fraction
of k-space, so the problem is well-posed, but the target is the time-average of
the scene over the window: correct geometry, blurred motion. Narrowing `w`
during optimization then trades that coverage back for temporal resolution,
starting from a good initialization rather than from noise.

This is a coarse-to-fine curriculum in *time* rather than in frequency or
resolution. It should be compared against the two static baselines:

| Condition | Bin width | Expectation |
|---|---|---|
| `wide` | fixed, wide (e.g. 25 frames) | Sharp spatial structure, temporally over-smoothed; motion blur / trail smearing. |
| `narrow` | fixed, `w = 1` (exact) | Per-frame problem under-determined; expect aliasing, streaking, unstable geometry. |
| `curriculum` | annealed wide -> narrow | Hypothesis: geometry from the wide phase survives the anneal, giving both sharp space and sharp time. |

Secondary axes once the main comparison is settled: sampling rate (5/10/25% per
frame), mask family (uniform random vs. variable-density vs. Cartesian lines vs.
radial spokes), and scene-model family.

## Data (initial experiment)

The existing `BouncingBallVideoDataset` provides the dynamic scene. It is a good
fixture here for the same reason it was for the interpolation work: known
ground truth, controllable motion rate, non-trivial temporal structure (the
trail and the color walk mean frames are not simply translations of each other).

Simulation pipeline:

1. Render a dense frame sequence `f_i`, `i = 0..N-1` at times `t_i = i * dt`.
2. Assume zero phase — the image is real-valued — and take `K_i = F[f_i]`,
   a centered orthonormal 2D FFT applied per color channel.
3. Draw a per-frame sampling mask `M_i` retaining ~10% of the grid, with a
   different realization per frame so the union over time covers k-space.
4. Store `(M_i, M_i . K_i, t_i)`. Optionally add complex Gaussian measurement
   noise at a specified SNR.

The dense frames `f_i` are retained only as ground truth for evaluation; the
optimizer sees k-space alone.

Caveats to keep visible, since they are places the fixture diverges from MRI:
the phase-zero assumption makes k-space Hermitian, so a mask and its conjugate
mirror carry the same information (the mask generator should be able to
enforce/avoid conjugate pairs); real acquisitions sample along continuous
trajectories rather than independent grid points; there is one coil, no field
inhomogeneity, and no off-resonance.

## Evaluation

Against held-out dense ground-truth frames:

- **Spatial fidelity**: PSNR / SSIM per frame, plus the same restricted to
  unobserved frequencies (fitting the observed ones is trivially achievable).
- **Temporal fidelity**: error of the ball centroid trajectory versus ground
  truth; temporal frequency content of a pixel time-series; error at
  *intermediate* times not corresponding to any observation frame, which is the
  actual claim of a continuous-time model.
- **Data consistency**: residual at observed frequencies, to confirm the model
  is not simply ignoring the measurements.
- **Qualitative**: side-by-side ground-truth / reconstruction / error videos,
  and a zero-filled inverse-FFT baseline for reference.

Baselines worth having before claiming anything: zero-filled IFFT per frame,
temporal-average IFFT (all frames pooled, one static image), and a low-rank or
TV-regularized per-frame reconstruction.

## Implementation Plan

### Phase 0 — prune (this branch)

Remove the interpolation-experiment evaluation stack, which is specific to the
previous question. Keep the training scaffolding (trainer loop, EMA,
checkpoints, run directories, TensorBoard callbacks), the transformer velocity
model, the rectified-flow loss, and the bouncing-ball dataset. The generative
prior is a plausible Phase 3 component, so the diffusion machinery stays.

Deleted: `evaluation/`, `utils/interpolation.py`, `utils/trajectory.py`,
`utils/trajectory_visualization.py`, `utils/flow.py`, `data/sequences.py`, the
corresponding tests, and the interpolation design docs.

### Phase 1 — forward model and data

```
kspace/transforms.py   centered orthonormal fft2/ifft2, zero-filled recon
kspace/sampling.py     per-frame mask generators (uniform, variable-density,
                       Cartesian lines, radial spokes) with center-block option
kspace/dataset.py      DynamicKSpaceDataset: dense frames -> (mask, k, t)
```

Correctness tests: FFT round-trip, Parseval/orthonormality, full-mask
reconstruction is exact, mask sampling rates are as requested, masks vary
across frames.

### Phase 2 — scene model and fitting

```
scene/models.py        FourierFeatureMLP (SIREN-ish baseline) and KPlaneScene
                       (factorized xy/xt/yt feature planes); both expose
                       render(t) -> [B, 3, H, W] on the pixel grid
scene/binning.py       bin-width schedules (constant, linear, exponential,
                       step) and the bin -> frame-index gather
scene/losses.py        masked complex k-space data-consistency loss, spatial
                       and temporal TV regularizers
scene/fit.py           optimization loop reusing the existing Trainer telemetry
scene/cli.py           `python -m flow_interpolation fit`
```

### Phase 3 — experiments

Run the three binning conditions at a fixed sampling rate and seed, then sweep
sampling rate and mask family. Add evaluation metrics and comparison videos.
Only after the binning question is answered does the learned prior (using the
retained rectified-flow model as a plug-and-play denoiser or score prior)
become worth attaching.

## Open questions

- Best scene parameterization for this regime. K-planes-style factorization is
  the obvious efficient choice, but its low-rank temporal structure may itself
  act as the temporal smoother, confounding the binning comparison. The plain
  coordinate MLP is a cleaner control.
- Whether the anneal schedule needs to be tied to the motion rate (bin width in
  units of "pixels of ball displacement" rather than frames) to transfer across
  scenes.
- Whether binning should weight bin members by distance from the center time
  (a soft kernel) rather than a hard window. A hard window is the stated
  starting point; a Gaussian kernel is a one-line generalization.
- Whether to model phase explicitly. Zero phase is fine for the fixture but a
  real acquisition needs a complex-valued scene model; the forward operator
  should not hard-code the real-valued assumption.
