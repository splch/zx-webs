#!/usr/bin/env python
"""Run a single ZX-Webs pipeline stage.

Usage
-----
    python scripts/run_stage.py corpus
    python scripts/run_stage.py zx --config configs/small_run.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from zx_webs.config import load_config
from zx_webs.pipeline import Pipeline


def main() -> None:
    """CLI entry point for running a single pipeline stage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run a single ZX-Webs pipeline stage")
    parser.add_argument("stage", type=str, help="Stage name to run")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the pipeline YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = Pipeline(config)
    pipeline.run_stage(args.stage)


if __name__ == "__main__":
    main()
