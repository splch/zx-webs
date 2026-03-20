"""Compare candidate circuits against baseline algorithms.

:func:`compare_candidate_to_baselines` computes :class:`CircuitMetrics` for
a candidate and each baseline, determines Pareto dominance, and records
per-metric percentage improvements.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from zx_webs.stage6_bench.metrics import CircuitMetrics

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing one candidate against one baseline.

    Attributes
    ----------
    candidate_id:
        Identifier of the candidate circuit.
    candidate_metrics:
        Metrics computed for the candidate.
    baseline_id:
        Identifier of the baseline circuit.
    baseline_metrics:
        Metrics computed for the baseline.
    candidate_dominates:
        ``True`` if the candidate Pareto-dominates the baseline.
    baseline_dominates:
        ``True`` if the baseline Pareto-dominates the candidate.
    improvements:
        Per-metric percentage improvement of the candidate over the baseline.
        Positive values mean the candidate is *better* (lower count/depth).
    """

    candidate_id: str = ""
    candidate_metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    baseline_id: str = ""
    baseline_metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    candidate_dominates: bool = False
    baseline_dominates: bool = False
    improvements: dict[str, float] = field(default_factory=dict)


def _pct_improvement(candidate_val: int, baseline_val: int) -> float:
    """Return percentage improvement (positive = candidate is better).

    ``100 * (baseline - candidate) / baseline`` when baseline > 0;
    0.0 when both are zero; -100.0 when only baseline is zero but
    candidate is not (candidate is strictly worse on a trivial baseline).
    """
    if baseline_val == 0:
        return 0.0 if candidate_val == 0 else -100.0
    return 100.0 * (baseline_val - candidate_val) / baseline_val


def compare_candidate_to_baselines(
    candidate_id: str,
    candidate_qasm: str,
    baselines: list[dict[str, Any]],
) -> list[ComparisonResult]:
    """Compare a candidate circuit against all baselines.

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
    except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
