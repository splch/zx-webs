"""Concrete benchmark tasks with target unitaries and evaluation functions.

Each :class:`BenchmarkTask` encapsulates a target unitary matrix derived from
a known quantum algorithm, along with methods to evaluate how well a candidate
circuit approximates that target.  Tasks are built automatically from the
algorithm corpus via :func:`build_benchmark_tasks`.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyzx as zx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BenchmarkTask dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkTask:
    """A concrete benchmarking task with a target unitary and evaluation.

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"qft_4q"``.
    description:
        Human-readable description, e.g. ``"arithmetic/qft on 4 qubits"``.
    n_qubits:
        Number of qubits the target unitary operates on.
    target_unitary:
        The ``2^n x 2^n`` complex unitary matrix for the target algorithm.
    baseline_gate_count:
        Total gate count of the original (baseline) circuit for this task.
    baseline_t_count:
        T-gate count of the original baseline circuit.
    baseline_cnot_count:
        CNOT count of the original baseline circuit.
    baseline_depth:
        Circuit depth of the original baseline circuit.
    metric_focus:
        Which metrics are most relevant for this task family.
    """

    name: str = ""
    description: str = ""
    n_qubits: int = 0
    target_unitary: np.ndarray = field(default_factory=lambda: np.eye(1))
    baseline_gate_count: int = 0
    baseline_t_count: int = 0
    baseline_cnot_count: int = 0
    baseline_depth: int = 0
    metric_focus: list[str] = field(default_factory=lambda: ["fidelity", "gate_count"])

    def fidelity(self, candidate_unitary: np.ndarray) -> float:
        """Compute process fidelity between this task's target and a candidate.

        Process fidelity is defined as::

            F = |Tr(U_target^dagger @ U_candidate)|^2 / d^2

        where ``d = 2^n_qubits`` is the Hilbert space dimension.

        Parameters
        ----------
        candidate_unitary:
            The unitary matrix of the candidate circuit.

        Returns
        -------
        float
            Fidelity in [0, 1].  Returns 0.0 if shapes don't match.
        """
        if candidate_unitary.shape != self.target_unitary.shape:
            return 0.0
        d = self.target_unitary.shape[0]
        overlap = np.trace(self.target_unitary.conj().T @ candidate_unitary)
        return float(np.abs(overlap) ** 2 / (d ** 2))

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary (excludes the unitary matrix)."""
        return {
            "name": self.name,
            "description": self.description,
            "n_qubits": self.n_qubits,
            "baseline_gate_count": self.baseline_gate_count,
            "baseline_t_count": self.baseline_t_count,
            "baseline_cnot_count": self.baseline_cnot_count,
            "baseline_depth": self.baseline_depth,
            "metric_focus": self.metric_focus,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_target_unitary(qasm_str: str) -> np.ndarray:
    """Build a unitary matrix from a QASM string via PyZX.

    Parameters
    ----------
    qasm_str:
        An OPENQASM 2.0 string.

    Returns
    -------
    np.ndarray
        The unitary matrix as a complex NumPy array.
    """
    c = zx.Circuit.from_qasm(qasm_str)
    return np.array(c.to_matrix())


def _baseline_metrics_from_qasm(qasm_str: str) -> dict[str, int]:
    """Extract baseline gate-count metrics from a QASM string.

    Returns
    -------
    dict with keys ``gate_count``, ``t_count``, ``cnot_count``, ``depth``.
    """
    c = zx.Circuit.from_qasm(qasm_str)
    sd = c.stats_dict()
    return {
        "gate_count": sd.get("gates", 0),
        "t_count": sd.get("tcount", 0),
        "cnot_count": sd.get("cnot", 0),
        "depth": c.depth(),
    }


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------


def build_benchmark_tasks(
    qubit_counts: list[int] | None = None,
    max_unitary_qubits: int = 10,
) -> list[BenchmarkTask]:
    """Build benchmark tasks from the algorithm corpus.

    For each algorithm in the registry at each qubit count, creates a
    :class:`BenchmarkTask` whose target unitary is the algorithm's unitary
    matrix.  This allows us to check whether any candidate circuit
    approximates a known algorithm more efficiently.

    Parameters
    ----------
    qubit_counts:
        Qubit counts to generate tasks for.  When *None*, uses
        ``[3, 4, 5]`` as a fallback.  In production, callers should
        derive these from the actual corpus/survivor qubit counts.
    max_unitary_qubits:
        Maximum qubit count for unitary matrix computation.

    Returns
    -------
    list[BenchmarkTask]
        All successfully built benchmark tasks.
    """
    if qubit_counts is None:
        qubit_counts = [3, 4, 5]

    from zx_webs.stage1_corpus.algorithms import ALGORITHM_REGISTRY
    from zx_webs.stage1_corpus.qasm_bridge import circuit_to_pyzx_qasm

    tasks: list[BenchmarkTask] = []

    for key, fn in sorted(ALGORITHM_REGISTRY.items()):
        family, name = key.split("/")

        for n_qubits in qubit_counts:
            try:
                min_q: int = getattr(fn, "min_qubits", 2)
                if n_qubits < min_q:
                    continue

                # Determine the correct first-parameter name.
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                first_param = params[0] if params else "n_qubits"

                qc = fn(**{first_param: n_qubits})
                qasm = circuit_to_pyzx_qasm(qc)

                # Only build unitaries for circuits we can actually compute.
                pyzx_circ = zx.Circuit.from_qasm(qasm)
                actual_qubits = pyzx_circ.qubits
                if actual_qubits > max_unitary_qubits:
                    logger.debug(
                        "Skipping %s/%s at %d qubits (too large for unitary).",
                        family, name, actual_qubits,
                    )
                    continue

                target = np.array(pyzx_circ.to_matrix())
                bl = _baseline_metrics_from_qasm(qasm)

                tasks.append(BenchmarkTask(
                    name=f"{name}_{actual_qubits}q",
                    description=f"{family}/{name} on {actual_qubits} qubits",
                    n_qubits=actual_qubits,
                    target_unitary=target,
                    baseline_gate_count=bl["gate_count"],
                    baseline_t_count=bl["t_count"],
                    baseline_cnot_count=bl["cnot_count"],
                    baseline_depth=bl["depth"],
                    metric_focus=["fidelity", "gate_count"],
                ))
                logger.debug(
                    "Built task %s/%s on %d qubits (%d gates).",
                    family, name, actual_qubits, bl["gate_count"],
                )

            except Exception:
                logger.debug(
                    "Failed to build task for %s at n=%d.",
                    key, n_qubits,
                    exc_info=True,
                )
                continue

    logger.info("Built %d benchmark tasks.", len(tasks))
    return tasks
