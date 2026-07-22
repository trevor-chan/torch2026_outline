"""Training run directory and configuration management."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_SUBDIRECTORIES = (
    "artifacts",
    "checkpoints",
    "samples/model",
    "samples/ema",
    "tensorboard",
)


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _validate_run_name(run_name: str) -> str:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be a non-empty directory name, not a path")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run_name may contain only letters, numbers, dots, dashes, and underscores")
    return run_name


def create_workdir(
    runs_dir: str | os.PathLike[str],
    *,
    run_name: str | None = None,
    workdir: str | os.PathLike[str] | None = None,
    subdirectories: tuple[str, ...] = RUN_SUBDIRECTORIES,
) -> Path:
    """Create a fresh run directory without reusing an existing path.

    ``subdirectories`` defaults to the training layout; scene fits pass their
    own so a fit directory does not sprout empty checkpoint and EMA folders.
    """
    if workdir is not None:
        candidate = Path(workdir).expanduser()
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"Work directory already exists: {candidate}. Use --resume to continue it."
            ) from error
    else:
        root = Path(runs_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        base_name = _validate_run_name(run_name) if run_name is not None else _timestamp()
        candidate = root / base_name
        suffix = 1
        while True:
            try:
                candidate.mkdir(exist_ok=False)
                break
            except FileExistsError:
                candidate = root / f"{base_name}-{suffix:02d}"
                suffix += 1

    for relative_path in subdirectories:
        (candidate / relative_path).mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def save_training_config(
    workdir: str | os.PathLike[str],
    args: argparse.Namespace | dict[str, Any],
    *,
    resolved: dict[str, Any] | None = None,
    resumed_from: str | os.PathLike[str] | None = None,
) -> Path:
    """Write an immutable JSON snapshot of the effective training configuration."""
    workdir = Path(workdir)
    arguments = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, "-m", "flow_interpolation", *sys.argv[1:]],
        "arguments": arguments,
        "resolved": resolved or {},
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
    }

    filename = "config.json" if resumed_from is None else f"config_resume_{_timestamp()}.json"
    path = workdir / filename
    suffix = 1
    while path.exists():
        stem = Path(filename).stem
        path = workdir / f"{stem}-{suffix:02d}.json"
        suffix += 1

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary_path, path)
    print(f"Training configuration saved to {path}")
    return path


def load_training_arguments(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the original run arguments used as resume defaults."""
    path = Path(workdir) / "config.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise TypeError(f"Invalid arguments payload in {path}")
    return arguments
