"""Stage 6 -- Functional benchmarking of surviving candidates."""
from __future__ import annotations

from zx_webs.stage6_bench.comparator import (
    ComparisonResult,
    TaskMatch,
    compare_candidate_to_baselines,
    match_candidate_to_tasks,
)
from zx_webs.stage6_bench.metrics import (
    CircuitMetrics,
    SupermarQFeatures,
    compute_unitary,
    entanglement_capacity,
    is_clifford_unitary,
)
from zx_webs.stage6_bench.runner import run_stage6
from zx_webs.stage6_bench.tasks import BenchmarkTask, build_benchmark_tasks

__all__ = [
    "BenchmarkTask",
    "CircuitMetrics",
    "ComparisonResult",
    "SupermarQFeatures",
    "TaskMatch",
    "build_benchmark_tasks",
    "compare_candidate_to_baselines",
    "compute_unitary",
    "entanglement_capacity",
    "is_clifford_unitary",
    "match_candidate_to_tasks",
    "run_stage6",
]
