"""Circuit-level metrics, SupermarQ-style features, and unitary analysis.

:class:`CircuitMetrics` captures standard gate-count and depth statistics
for a quantum circuit.  :class:`SupermarQFeatures` computes normalised
feature vectors inspired by the SupermarQ benchmark suite.

The unitary analysis functions (:func:`compute_unitary`,
:func:`is_clifford_unitary`, :func:`entanglement_capacity`) provide
functional characterisation of circuits beyond gate counts.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pyzx as zx

logger = logging.getLogger(__name__)

# Default maximum qubit count for which we compute full unitary matrices.
# Overridable via function parameters.
_DEFAULT_MAX_UNITARY_QUBITS = 10


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


# ---------------------------------------------------------------------------
# Unitary analysis functions
# ---------------------------------------------------------------------------


def compute_unitary(
    qasm_str: str, max_unitary_qubits: int | None = None
) -> np.ndarray | None:
    """Compute the unitary matrix from a QASM string.

    Returns ``None`` if the circuit has more than *max_unitary_qubits*
    qubits (the ``2^n x 2^n`` matrix would be too large) or if parsing fails.

    Parameters
    ----------
    qasm_str:
        An OPENQASM 2.0 string.
    max_unitary_qubits:
        Maximum qubit count.  Defaults to ``_DEFAULT_MAX_UNITARY_QUBITS``.

    Returns
    -------
    np.ndarray or None
        The unitary as a complex NumPy array, or ``None``.
    """
    if max_unitary_qubits is None:
        max_unitary_qubits = _DEFAULT_MAX_UNITARY_QUBITS

    try:
        c = zx.Circuit.from_qasm(qasm_str)
    except Exception:
        logger.debug("Failed to parse QASM for unitary computation.")
        return None

    if c.qubits > max_unitary_qubits:
        logger.debug(
            "Circuit has %d qubits (> %d); skipping unitary.",
            c.qubits, max_unitary_qubits,
        )
        return None

    try:
        return np.array(c.to_matrix())
    except Exception:
        logger.debug("Failed to compute unitary matrix.", exc_info=True)
        return None


def is_clifford_unitary(unitary: np.ndarray, atol: float = 1e-8) -> bool:
    """Check if a unitary is in the Clifford group (heuristic).

    A unitary is Clifford if it maps every Pauli operator to another
    Pauli operator under conjugation.  For efficiency, we only check the
    single-qubit Pauli generators (X, Y, Z on each qubit) and verify
    that the result is also a tensor product of Paulis (up to phase).

    For single-qubit unitaries, this is exact.  For multi-qubit unitaries,
    this is a necessary condition that is sufficient in practice for
    circuits built from standard gate sets.

    Parameters
    ----------
    unitary:
        A ``2^n x 2^n`` unitary matrix.
    atol:
        Absolute tolerance for the Pauli check.

    Returns
    -------
    bool
    """
    n = int(np.log2(unitary.shape[0]))
    if 2**n != unitary.shape[0]:
        return False

    # Single-qubit Paulis.
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    single_paulis = [pauli_x, pauli_y, pauli_z]

    u_dag = unitary.conj().T

    for qubit in range(n):
        for pauli in single_paulis:
            # Build the n-qubit Pauli: I x ... x P x ... x I
            op = np.array([[1.0]], dtype=complex)
            for q in range(n):
                if q == qubit:
                    op = np.kron(op, pauli)
                else:
                    op = np.kron(op, identity)

            # Conjugate: U P U^dagger
            conjugated = unitary @ op @ u_dag

            # Check if the result is a Pauli (up to phase):
            # A matrix is a Pauli (up to phase) if it is unitary and
            # has exactly one non-zero element per row and column,
            # with all non-zero elements having absolute value 1.
            if not _is_n_qubit_pauli(conjugated, n, atol):
                return False

    return True


def _is_n_qubit_pauli(mat: np.ndarray, n: int, atol: float = 1e-8) -> bool:
    """Check if a matrix is an n-qubit Pauli operator (up to global phase).

    Decomposes the matrix into a tensor product of single-qubit factors
    and checks if each factor is one of {I, X, Y, Z} up to a phase.

    For n=1, checks against the 4 single-qubit Paulis directly.
    For n>1, uses a recursive decomposition.
    """
    d = 2**n

    if mat.shape != (d, d):
        return False

    # Single-qubit case: check directly.
    if n == 1:
        paulis_1q = [
            np.eye(2, dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]
        for p in paulis_1q:
            # Check if mat = alpha * p for some scalar alpha.
            # Find alpha from the first nonzero of p.
            for i in range(2):
                for j in range(2):
                    if abs(p[i, j]) > atol:
                        alpha = mat[i, j] / p[i, j]
                        if abs(abs(alpha) - 0.0) < atol:
                            break  # mat has zero where p doesn't; try next p
                        if np.allclose(mat, alpha * p, atol=atol):
                            return True
                        break
                else:
                    continue
                break
        return False

    # Multi-qubit case: decompose as (A tensor B) where A is 2x2 and B is 2^{n-1} x 2^{n-1}.
    # If mat = phase * (P_1 tensor P_rest), then the 2x2 blocks form the tensor product structure.
    half = 2**(n - 1)
    # Extract the four blocks:
    blocks = [[mat[i * half:(i + 1) * half, j * half:(j + 1) * half] for j in range(2)] for i in range(2)]

    # Find a nonzero block and use it to determine the first Pauli factor.
    ref_block = None
    ref_i, ref_j = 0, 0
    for i in range(2):
        for j in range(2):
            if np.linalg.norm(blocks[i][j]) > atol:
                ref_block = blocks[i][j]
                ref_i, ref_j = i, j
                break
        if ref_block is not None:
            break

    if ref_block is None:
        return False

    # Every other block should be proportional to ref_block.
    coeffs = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            block = blocks[i][j]
            norm_block = np.linalg.norm(block)
            norm_ref = np.linalg.norm(ref_block)
            if norm_block < atol:
                coeffs[i, j] = 0.0
            else:
                # Find the ratio.
                # block = coeffs[i,j] * ref_block
                # Use least squares for robustness.
                ref_flat = ref_block.ravel()
                block_flat = block.ravel()
                idx = np.argmax(np.abs(ref_flat))
                if abs(ref_flat[idx]) < atol:
                    return False
                ratio = block_flat[idx] / ref_flat[idx]
                if not np.allclose(block, ratio * ref_block, atol=atol * max(1, norm_ref)):
                    return False
                coeffs[i, j] = ratio

    # coeffs should be a single-qubit Pauli (up to phase),
    # and ref_block should be an (n-1)-qubit Pauli (up to phase).
    return _is_n_qubit_pauli(coeffs, 1, atol) and _is_n_qubit_pauli(ref_block, n - 1, atol)


def process_fidelity(u1: np.ndarray, u2: np.ndarray) -> float:
    """Compute the process fidelity between two unitary matrices.

    ``|Tr(U1^dag @ U2)|^2 / d^2`` where ``d`` is the matrix dimension.

    Parameters
    ----------
    u1, u2:
        Unitary matrices of the same shape.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    if u1.shape != u2.shape:
        return 0.0
    d = u1.shape[0]
    tr = np.trace(u1.conj().T @ u2)
    return float(abs(tr) ** 2) / (d * d)


def novelty_score(
    candidate_unitary: np.ndarray,
    corpus_unitaries: list[np.ndarray],
) -> float:
    """Compute how novel a candidate unitary is relative to all known unitaries.

    Returns ``1 - max_fidelity`` where ``max_fidelity`` is the highest process
    fidelity to any corpus unitary of the same dimension.  A score of 1.0 means
    the candidate is maximally unlike anything in the corpus; 0.0 means it's
    identical to a known algorithm.

    Parameters
    ----------
    candidate_unitary:
        The candidate's unitary matrix.
    corpus_unitaries:
        List of known algorithm unitaries.

    Returns
    -------
    float
        Novelty score in [0, 1].
    """
    if not corpus_unitaries:
        return 1.0

    max_fidelity = 0.0
    d = candidate_unitary.shape[0]

    for corpus_u in corpus_unitaries:
        if corpus_u.shape != candidate_unitary.shape:
            continue
        fid = process_fidelity(candidate_unitary, corpus_u)
        if fid > max_fidelity:
            max_fidelity = fid
            if max_fidelity >= 1.0 - 1e-10:
                break  # exact match, no need to continue

    return 1.0 - max_fidelity


def entanglement_capacity(unitary: np.ndarray) -> float:
    """Estimate the entanglement-generating capacity of a unitary.

    Uses the operator entanglement entropy: reshape the unitary as a
    bipartite operator (splitting qubits into two halves), compute the
    singular values, and return the normalised entropy of the squared
    singular values.

    For a product unitary (no entanglement), the capacity is 0.
    For maximally entangling unitaries, it approaches 1.

    Parameters
    ----------
    unitary:
        A ``2^n x 2^n`` unitary matrix with ``n >= 2``.

    Returns
    -------
    float
        Normalised entanglement capacity in [0, 1].
        Returns 0.0 for single-qubit unitaries or on error.
    """
    n = int(np.log2(unitary.shape[0]))
    if n < 2:
        return 0.0

    # Split qubits into two halves.
    n_a = n // 2
    n_b = n - n_a
    d_a = 2**n_a
    d_b = 2**n_b
    d = d_a * d_b

    try:
        # Reshape U as a d_a^2 x d_b^2 matrix for the operator Schmidt decomposition.
        # U_{(i,j),(k,l)} -> M_{(i,k),(j,l)}
        u_reshaped = unitary.reshape(d_a, d_b, d_a, d_b)
        u_reshaped = u_reshaped.transpose(0, 2, 1, 3)
        u_reshaped = u_reshaped.reshape(d_a * d_a, d_b * d_b)

        svs = np.linalg.svd(u_reshaped, compute_uv=False)
        # Normalise singular values squared to form a probability distribution.
        probs = svs**2
        probs = probs / probs.sum()

        # Compute entropy.
        nonzero = probs > 1e-15
        entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))

        # Normalise by maximum entropy.
        max_entropy = 2 * n_a * np.log2(2)  # = 2 * n_a (in bits)
        if max_entropy == 0:
            return 0.0

        return float(min(1.0, entropy / max_entropy))
    except Exception:
        logger.debug("Failed to compute entanglement capacity.", exc_info=True)
        return 0.0
