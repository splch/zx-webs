#!/usr/bin/env python
"""Run the full ZX-Webs pipeline (or a range of stages).

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config configs/small_run.yaml
    python scripts/run_pipeline.py --start corpus --end zx
"""
from zx_webs.pipeline import main

if __name__ == "__main__":
    main()
