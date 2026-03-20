"""Benchmark task definitions.

Maps task names to descriptions and metric focus areas.  These are used by
:func:`run_stage6` to annotate results and by future reporting to highlight
the most relevant metrics per task family.
"""
from __future__ import annotations

BENCHMARK_TASKS: dict[str, dict] = {
    "grover_oracle": {
        "description": "Grover's search oracle",
        "metric_focus": ["t_count", "depth"],
    },
    "maxcut": {
        "description": "MaxCut QAOA circuit",
        "metric_focus": ["cnot_count", "depth"],
    },
    "qpe": {
        "description": "Quantum Phase Estimation",
        "metric_focus": ["t_count", "total_gates"],
    },
    "ghz": {
        "description": "GHZ state preparation",
        "metric_focus": ["cnot_count", "depth"],
    },
    "arithmetic": {
        "description": "Arithmetic circuits (adders, etc.)",
        "metric_focus": ["total_gates", "depth"],
    },
}
