"""Stage 6 -- Benchmarking surviving candidates against baselines."""
from __future__ import annotations

from zx_webs.stage6_bench.comparator import ComparisonResult, compare_candidate_to_baselines
from zx_webs.stage6_bench.metrics import CircuitMetrics, SupermarQFeatures
from zx_webs.stage6_bench.runner import run_stage6
from zx_webs.stage6_bench.tasks import BENCHMARK_TASKS

__all__ = [
    "BENCHMARK_TASKS",
    "CircuitMetrics",
    "ComparisonResult",
    "SupermarQFeatures",
    "compare_candidate_to_baselines",
    "run_stage6",
]
