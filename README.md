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

Training writes previews, samples, and checkpoints under `outputs/` by default:

```bash
python -m flow_interpolation train \
  --max-steps 150000 \
  --checkpoint-interval 5000
```

Evaluation is organized by experiment. For example:

```bash
python -m flow_interpolation eval latent --checkpoint outputs/checkpoints/model.pth
python -m flow_interpolation eval hybrid --checkpoint outputs/checkpoints/model.pth
python -m flow_interpolation eval roundtrip --checkpoint outputs/checkpoints/model.pth
```

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
