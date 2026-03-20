"""Circuit deduplication for Stage 5 filtering.

Two circuits are considered duplicates if they implement the same unitary
transformation (up to a global phase).  For circuits that are too large to
compare via full unitary matrices, a fallback QASM-string comparison is used.

When CuPy is available, GPU-accelerated matrix operations are used for the
unitary comparison.  Otherwise, NumPy is used as a fallback.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyzx as zx

logger = logging.getLogger(__name__)

# Maximum qubit count for unitary comparison.  Beyond this, the 2^n x 2^n
# matrix becomes impractical.
_MAX_UNITARY_QUBITS = 8

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

try:
    import cupy as cp
    _HAS_CUPY = True
    logger.debug("CuPy available -- GPU-accelerated deduplication enabled.")
except ImportError:
    _HAS_CUPY = False


# ---------------------------------------------------------------------------
# Unitary equivalence
# ---------------------------------------------------------------------------


def _matrices_equal_up_to_phase(u1: np.ndarray, u2: np.ndarray, atol: float = 1e-8) -> bool:
    """Check if u1 == e^(i*theta) * u2 using GPU if available.

    Finds the first non-zero element in both matrices, computes the phase
    difference, and checks whether the entire matrices match under that
    global phase.

    Parameters
    ----------
    u1, u2:
        Unitary matrices as NumPy arrays.
    atol:
        Absolute tolerance for ``allclose`` comparison.

    Returns
    -------
    bool
    """
    if _HAS_CUPY:
        u1_gpu = cp.asarray(u1)
        u2_gpu = cp.asarray(u2)
        flat1 = u1_gpu.ravel()
        flat2 = u2_gpu.ravel()
        for i in range(len(flat1)):
            if cp.abs(flat2[i]) > atol:
                phase = flat1[i] / flat2[i]
                return bool(cp.allclose(u1_gpu, phase * u2_gpu, atol=atol))
        # All elements of u2 are near-zero -- compare directly.
        return bool(cp.allclose(u1_gpu, u2_gpu, atol=atol))
    else:
        # NumPy fallback
        flat1 = u1.ravel()
        flat2 = u2.ravel()
        for i in range(len(flat1)):
            if abs(flat2[i]) > atol:
                phase = flat1[i] / flat2[i]
                return bool(np.allclose(u1, phase * u2, atol=atol))
        return bool(np.allclose(u1, u2, atol=atol))


def _normalise_global_phase(mat: np.ndarray) -> np.ndarray:
    """Remove the global phase from a unitary matrix.

    The first non-zero element is rotated to be real and positive.
    """
    flat = mat.flatten()
    for elem in flat:
        if abs(elem) > 1e-9:
            phase = elem / abs(elem)
            return mat / phase
    return mat


def circuits_equivalent(
    qasm1: str,
    qasm2: str,
    method: str = "unitary",
) -> bool:
    """Check whether two QASM circuits implement the same unitary.

    Parameters
    ----------
    qasm1, qasm2:
        OPENQASM 2.0 strings.
    method:
        ``"unitary"`` -- full unitary comparison (exact up to global phase).
        ``"qasm"`` -- literal QASM string equality (fast but fragile).

    Returns
    -------
    bool
    """
    if method == "qasm":
        return qasm1.strip() == qasm2.strip()

    # -- Unitary comparison ---------------------------------------------------
    try:
        c1 = zx.Circuit.from_qasm(qasm1)
        c2 = zx.Circuit.from_qasm(qasm2)
    except Exception:  # noqa: BLE001
        # Fall back to string comparison if parsing fails.
        return qasm1.strip() == qasm2.strip()

    if c1.qubits != c2.qubits:
        return False

    if c1.qubits > _MAX_UNITARY_QUBITS:
        # Too large for unitary comparison; fall back to QASM equality.
        return qasm1.strip() == qasm2.strip()

    try:
        m1 = c1.to_matrix()
        m2 = c2.to_matrix()
    except Exception:  # noqa: BLE001
        return qasm1.strip() == qasm2.strip()

    return _matrices_equal_up_to_phase(m1, m2, atol=1e-8)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_circuits(
    circuits: list[dict[str, Any]],
    method: str = "unitary",
) -> list[dict[str, Any]]:
    """Remove duplicate circuits from a list.

    Each element in *circuits* is a dict that must contain a
    ``"circuit_qasm"`` key.

    The first occurrence of each unique circuit is kept; later duplicates
    are dropped.

    Parameters
    ----------
    circuits:
        List of dicts, each with at least ``"circuit_qasm"`` and ``"stats"``.
    method:
        Equivalence method passed to :func:`circuits_equivalent`.

    Returns
    -------
    list[dict]
        The deduplicated list (order preserved, first occurrence kept).
    """
    if not circuits:
        return []

    unique: list[dict[str, Any]] = []
    for candidate in circuits:
        qasm = candidate.get("circuit_qasm", "")
        is_dup = False
        for kept in unique:
            kept_qasm = kept.get("circuit_qasm", "")
            if circuits_equivalent(qasm, kept_qasm, method=method):
                is_dup = True
                break
        if not is_dup:
            unique.append(candidate)

    n_removed = len(circuits) - len(unique)
    if n_removed > 0:
        logger.info("Deduplication removed %d circuits.", n_removed)

    return unique
