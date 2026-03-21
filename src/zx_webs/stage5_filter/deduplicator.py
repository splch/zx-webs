"""Circuit deduplication for Stage 5 filtering.

Two circuits are considered duplicates if they implement the same unitary
transformation (up to a global phase).  For circuits that are too large to
compare via full unitary matrices, a fallback QASM-string comparison is used.

This module provides two deduplication strategies:

- **Hash-based (O(n))**: :func:`deduplicate_circuits` computes a
  phase-normalised unitary hash for each circuit and uses a dict for
  constant-time lookups.
- **Pairwise (legacy)**: :func:`circuits_equivalent` compares two circuits
  directly.

When CuPy is available, GPU-accelerated matrix operations are used for the
pairwise comparison.  Otherwise, NumPy is used as a fallback.
"""
from __future__ import annotations

import hashlib
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
# Unitary equivalence (pairwise)
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
# Hash-based deduplication (O(n))
# ---------------------------------------------------------------------------


def _unitary_hash(qasm_str: str, atol: float = 1e-8) -> str | None:
    """Compute a hash of the phase-normalised unitary for fast dedup.

    The unitary matrix is normalised to remove global phase (first non-zero
    element rotated to real positive), then rounded and hashed.

    Parameters
    ----------
    qasm_str:
        An OPENQASM 2.0 string.
    atol:
        Tolerance used during global-phase normalisation.

    Returns
    -------
    str or None
        A hex digest string, or ``None`` if the unitary cannot be computed
        (too many qubits, parse error, etc.).
    """
    try:
        c = zx.Circuit.from_qasm(qasm_str)
    except Exception:
        return None

    if c.qubits > _MAX_UNITARY_QUBITS:
        return None

    try:
        mat = np.array(c.to_matrix(), dtype=np.complex128)
    except Exception:
        return None

    # Normalise global phase: rotate so first significant element is real+positive.
    flat = mat.ravel()
    for val in flat:
        if abs(val) > atol:
            phase = val / abs(val)
            mat = mat / phase
            break

    # Round to avoid floating-point noise, then hash.
    rounded = np.round(mat, decimals=6)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_circuits(
    circuits: list[dict[str, Any]],
    method: str = "unitary",
) -> list[dict[str, Any]]:
    """Remove duplicate circuits from a list using hash-based O(n) dedup.

    Each element in *circuits* is a dict that must contain a
    ``"circuit_qasm"`` key.

    The first occurrence of each unique circuit is kept; later duplicates
    are dropped.

    For circuits small enough to compute unitaries (<= 8 qubits), a
    phase-normalised unitary hash is used for O(1) lookup.  For larger
    circuits, QASM string equality is used as a fallback.

    Parameters
    ----------
    circuits:
        List of dicts, each with at least ``"circuit_qasm"`` and ``"stats"``.
    method:
        Equivalence method: ``"unitary"`` (hash-based, default) or
        ``"qasm"`` (literal string equality).

    Returns
    -------
    list[dict]
        The deduplicated list (order preserved, first occurrence kept).
    """
    if not circuits:
        return []

    seen_hashes: dict[str, bool] = {}
    seen_qasm: set[str] = set()
    unique: list[dict[str, Any]] = []

    for candidate in circuits:
        qasm = candidate.get("circuit_qasm", "")

        if method == "unitary":
            h = _unitary_hash(qasm)
            if h is not None:
                if h in seen_hashes:
                    continue
                seen_hashes[h] = True
                unique.append(candidate)
            else:
                # Fallback: QASM string comparison for large circuits.
                stripped = qasm.strip()
                if stripped in seen_qasm:
                    continue
                seen_qasm.add(stripped)
                unique.append(candidate)
        else:
            # Pure QASM string comparison.
            stripped = qasm.strip()
            if stripped in seen_qasm:
                continue
            seen_qasm.add(stripped)
            unique.append(candidate)

    n_removed = len(circuits) - len(unique)
    if n_removed > 0:
        logger.info("Deduplication removed %d circuits.", n_removed)

    return unique
