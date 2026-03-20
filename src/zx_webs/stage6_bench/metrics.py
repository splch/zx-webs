"""Circuit-level metrics and SupermarQ-style feature vectors.

:class:`CircuitMetrics` captures standard gate-count and depth statistics
for a quantum circuit.  :class:`SupermarQFeatures` computes normalised
feature vectors inspired by the SupermarQ benchmark suite.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pyzx as zx


# ---------------------------------------------------------------------------
# Circuit-level metrics
# ---------------------------------------------------------------------------


@dataclass
class CircuitMetrics:
    """Circuit-level performance metrics."""

    t_count: int = 0
    cnot_count: int = 0
    total_two_qubit: int = 0
    total_gates: int = 0
    depth: int = 0
    qubit_count: int = 0

    @staticmethod
    def from_pyzx_circuit(c: zx.Circuit) -> CircuitMetrics:
        """Extract metrics from a PyZX circuit.

        Uses ``stats_dict()`` for gate counts and ``depth()`` for circuit
        depth (``stats_dict`` does not populate depth reliably).
        """
        sd = c.stats_dict()
        return CircuitMetrics(
            t_count=sd.get("tcount", 0),
            cnot_count=sd.get("cnot", 0),
            total_two_qubit=sd.get("twoqubit", 0),
            total_gates=sd.get("gates", 0),
            depth=c.depth(),
            qubit_count=c.qubits,
        )

    @staticmethod
    def from_qasm(qasm_str: str) -> CircuitMetrics:
        """Extract metrics from an OPENQASM 2.0 string via PyZX."""
        c = zx.Circuit.from_qasm(qasm_str)
        return CircuitMetrics.from_pyzx_circuit(c)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)

    def dominates(self, other: CircuitMetrics) -> bool:
        """Return *True* if *self* Pareto-dominates *other*.

        Pareto dominance: *self* is strictly better on **at least one**
        metric and no worse on any.  The compared metrics are ``t_count``,
        ``cnot_count``, and ``depth``.
        """
        metrics = ("t_count", "cnot_count", "depth")
        at_least_one_better = False
        for m in metrics:
            self_val = getattr(self, m)
            other_val = getattr(other, m)
            if self_val > other_val:
                return False
            if self_val < other_val:
                at_least_one_better = True
        return at_least_one_better


# ---------------------------------------------------------------------------
# SupermarQ-style feature vectors
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    """Clamp *x* to the interval [0, 1]."""
    return max(0.0, min(1.0, x))


@dataclass
class SupermarQFeatures:
    """SupermarQ-style feature vectors (all normalised to [0, 1]).

    These are lightweight approximations computed purely from circuit
    structure, without executing the circuit on a backend.

    Attributes
    ----------
    program_communication:
        Fraction of gates that are two-qubit gates.
    critical_depth:
        ``depth / total_gates`` -- how serialised the circuit is.
    entanglement_ratio:
        Alias for *program_communication* (two-qubit fraction).
    parallelism:
        ``(gates_per_layer - 1) / (n_qubits - 1)`` for multi-qubit
        circuits; 0 for single-qubit circuits.
    liveness:
        Rough estimate of qubit utilisation: ``total_gates / (depth * n_qubits)``.
    """

    program_communication: float = 0.0
    critical_depth: float = 0.0
    entanglement_ratio: float = 0.0
    parallelism: float = 0.0
    liveness: float = 0.0

    @staticmethod
    def from_qasm(qasm_str: str) -> SupermarQFeatures:
        """Compute SupermarQ features from an OPENQASM 2.0 string."""
        c = zx.Circuit.from_qasm(qasm_str)
        total_gates = len(c.gates)
        two_q = c.twoqubitcount()
        depth = c.depth()
        n_qubits = c.qubits

        if total_gates == 0:
            return SupermarQFeatures()

        program_communication = _clamp01(two_q / total_gates)
        critical_depth = _clamp01(depth / total_gates) if total_gates > 0 else 0.0
        entanglement_ratio = program_communication

        if n_qubits > 1 and depth > 0:
            gates_per_layer = total_gates / depth
            parallelism = _clamp01((gates_per_layer - 1) / (n_qubits - 1))
        else:
            parallelism = 0.0

        if depth > 0 and n_qubits > 0:
            liveness = _clamp01(total_gates / (depth * n_qubits))
        else:
            liveness = 0.0

        return SupermarQFeatures(
            program_communication=program_communication,
            critical_depth=critical_depth,
            entanglement_ratio=entanglement_ratio,
            parallelism=parallelism,
            liveness=liveness,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict."""
        return asdict(self)
