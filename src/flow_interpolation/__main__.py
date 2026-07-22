"""Unified module entry point for training and evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flow_interpolation",
        description="Train flow-matching models and fit implicit scenes to sparse k-space.",
    )
    parser.add_argument("command", nargs="?", choices=("train", "fit"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command, *command_arguments = arguments
    if command == "train":
        from flow_interpolation.training.cli import main as command_main
    elif command == "fit":
        from flow_interpolation.scene.cli import main as command_main
    else:
        parser.error(f"unknown command: {command}")

    command_main(command_arguments)


if __name__ == "__main__":
    main()
