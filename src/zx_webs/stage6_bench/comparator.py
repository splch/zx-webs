"""Unitary-aware comparison of candidates against benchmark tasks.

Replaces the old gate-count-only comparison with a system that:

1. Computes each candidate's unitary matrix.
2. Matches candidates against tasks with the same qubit count.
3. Evaluates process fidelity.
4. Only reports gate improvements when fidelity exceeds a threshold.

:func:`match_candidate_to_tasks` is the main entry point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyzx as zx

from zx_webs.stage6_bench.metrics import CircuitMetrics
from zx_webs.stage6_bench.tasks import BenchmarkTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskMatch:
    """Result of matching a candidate against a single benchmark task.

    Attributes
    ----------
    candidate_id:
        Identifier of the candidate circuit.
    task_name:
        Name of the benchmark task.
    fidelity:
        Process fidelity in [0, 1].
    candidate_metrics:
        Gate-count and depth metrics of the candidate circuit.
    baseline_metrics:
        Gate-count and depth metrics of the original baseline for this task.
    gate_improvement:
        Per-metric percentage improvement of candidate over baseline.
        Positive values mean the candidate is *better* (fewer gates/depth).
        Only meaningful when ``fidelity >= threshold``.
    is_improvement:
        ``True`` if fidelity >= threshold **and** the candidate has strictly
        fewer total gates than the baseline.
    """

    candidate_id: str = ""
    task_name: str = ""
    fidelity: float = 0.0
    candidate_metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    baseline_metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    gate_improvement: dict[str, float] = field(default_factory=dict)
    is_improvement: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "candidate_id": self.candidate_id,
            "task_name": self.task_name,
            "fidelity": self.fidelity,
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "gate_improvement": self.gate_improvement,
            "is_improvement": self.is_improvement,
        }


# ---------------------------------------------------------------------------
# Backward-compatible ComparisonResult alias
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Legacy comparison result -- kept for backward compatibility.

    New code should use :class:`TaskMatch` and
    :func:`match_candidate_to_tasks` instead.
    """

    candidate_id: str = ""
    candidate_metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    baseline_id: str = ""
    baseline_metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    candidate_dominates: bool = False
    baseline_dominates: bool = False
    improvements: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pct_improvement(candidate_val: int, baseline_val: int) -> float:
    """Return percentage improvement (positive = candidate is better).

    ``100 * (baseline - candidate) / baseline`` when baseline > 0;
    0.0 when both are zero; -100.0 when only baseline is zero but
    candidate is not.
    """
    if baseline_val == 0:
        return 0.0 if candidate_val == 0 else -100.0
    return 100.0 * (baseline_val - candidate_val) / baseline_val


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------


def match_candidate_to_tasks(
    candidate_id: str,
    candidate_qasm: str,
    tasks: list[BenchmarkTask],
    fidelity_threshold: float = 0.99,
) -> list[TaskMatch]:
    """Match a candidate against all benchmark tasks of the same qubit count.

    For each task whose ``n_qubits`` matches the candidate's qubit count:

    1. Compute the candidate's unitary matrix.
    2. Evaluate process fidelity against the task's target unitary.
    3. Compute gate-count improvements (meaningful only when fidelity is high).
    4. Mark as an improvement if fidelity >= threshold **and** total gates
       are strictly fewer.

    Parameters
    ----------
    candidate_id:
        Identifier for the candidate circuit.
    candidate_qasm:
        OPENQASM 2.0 string of the candidate.
    tasks:
        List of benchmark tasks to match against.
    fidelity_threshold:
        Minimum fidelity to consider a candidate as functionally equivalent.

    Returns
    -------
    list[TaskMatch]
        One result per task with matching qubit count.  Empty if the
        candidate cannot be parsed or has no matching tasks.
    """
    # Parse the candidate circuit.
    try:
        cand_circuit = zx.Circuit.from_qasm(candidate_qasm)
    except Exception:
        logger.warning("Failed to parse candidate %s; skipping.", candidate_id)
        return []

    cand_qubits = cand_circuit.qubits

    # Filter tasks to matching qubit count.
    matching_tasks = [t for t in tasks if t.n_qubits == cand_qubits]
    if not matching_tasks:
        return []

    # Compute candidate metrics.
    try:
        cand_metrics = CircuitMetrics.from_pyzx_circuit(cand_circuit)
    except Exception:
        logger.warning("Failed to compute metrics for candidate %s.", candidate_id)
        return []

    # Compute candidate unitary (may fail for very large circuits).
    cand_unitary: np.ndarray | None = None
    if cand_qubits <= 10:
        try:
            cand_unitary = np.array(cand_circuit.to_matrix())
        except Exception:
            logger.debug(
                "Failed to compute unitary for candidate %s (%d qubits).",
                candidate_id, cand_qubits,
            )

    results: list[TaskMatch] = []
    compared_metrics = ("t_count", "cnot_count", "total_gates", "depth")

    for task in matching_tasks:
        # Compute fidelity.
        if cand_unitary is not None:
            fidelity = task.fidelity(cand_unitary)
        else:
            # Cannot compute fidelity -- report 0.
            fidelity = 0.0

        # Build baseline metrics from the task.
        baseline_metrics = CircuitMetrics(
            t_count=task.baseline_t_count,
            cnot_count=task.baseline_cnot_count,
            total_gates=task.baseline_gate_count,
            depth=task.baseline_depth,
            qubit_count=task.n_qubits,
        )

        # Compute gate improvements.
        improvements: dict[str, float] = {}
        for m in compared_metrics:
            cand_val = getattr(cand_metrics, m)
            bl_val = getattr(baseline_metrics, m)
            improvements[m] = _pct_improvement(cand_val, bl_val)

        # Determine if this is a real improvement.
        is_improvement = (
            fidelity >= fidelity_threshold
            and cand_metrics.total_gates < baseline_metrics.total_gates
        )

        results.append(TaskMatch(
            candidate_id=candidate_id,
            task_name=task.name,
            fidelity=fidelity,
            candidate_metrics=cand_metrics,
            baseline_metrics=baseline_metrics,
            gate_improvement=improvements,
            is_improvement=is_improvement,
        ))

    return results


# ---------------------------------------------------------------------------
# Legacy comparator (backward compatibility)
# ---------------------------------------------------------------------------


def compare_candidate_to_baselines(
    candidate_id: str,
    candidate_qasm: str,
    baselines: list[dict[str, Any]],
) -> list[ComparisonResult]:
    """Compare a candidate against baselines using gate-count metrics.

    .. deprecated::
        Use :func:`match_candidate_to_tasks` for meaningful functional
        comparison.  This function is kept for backward compatibility with
        Stage 7 reporting and other code that reads ``ComparisonResult``.

    Parameters
    ----------
    candidate_id:
        Unique identifier for the candidate.
    candidate_qasm:
        OPENQASM 2.0 string of the candidate circuit.
    baselines:
        List of dicts, each with ``"id"`` and ``"qasm"`` keys.

    Returns
    -------
    list[ComparisonResult]
        One result per baseline.
    """
    try:
        cand_metrics = CircuitMetrics.from_qasm(candidate_qasm)
    except Exception:
        logger.warning("Failed to parse candidate %s; skipping comparisons.", candidate_id)
        return []

    results: list[ComparisonResult] = []
    compared_metrics = ("t_count", "cnot_count", "depth")

    for bl in baselines:
        bl_id = bl.get("id", "unknown")
        bl_qasm = bl.get("qasm", "")
        if not bl_qasm:
            continue

        try:
            bl_metrics = CircuitMetrics.from_qasm(bl_qasm)
        except Exception:
            logger.warning("Failed to parse baseline %s; skipping.", bl_id)
            continue

        improvements = {
            m: _pct_improvement(getattr(cand_metrics, m), getattr(bl_metrics, m))
            for m in compared_metrics
        }

        results.append(
            ComparisonResult(
                candidate_id=candidate_id,
                candidate_metrics=cand_metrics,
                baseline_id=bl_id,
                baseline_metrics=bl_metrics,
                candidate_dominates=cand_metrics.dominates(bl_metrics),
                baseline_dominates=bl_metrics.dominates(cand_metrics),
                improvements=improvements,
            )
        )

    return results
