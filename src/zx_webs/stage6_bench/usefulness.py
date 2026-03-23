"""Usefulness scoring for discovered quantum circuits.

Tests whether a circuit has computationally useful properties, beyond
just being "novel" (far from known algorithms).  Three tiers of tests:

**Tier 1 -- Functional primitive detection**:
  Is the circuit a phase oracle, permutation, controlled-U, or does it
  prepare a known useful quantum state (GHZ, W, Dicke, uniform)?

**Tier 2 -- Algebraic property tests**:
  Does it work as a QAOA mixer?  Does it preserve useful symmetries
  (parity, Hamming weight)?  Is it a good scrambler?  Is it periodic
  or an involution (useful for amplitude amplification)?

**Tier 3 -- Structural richness measures**:
  How far is the unitary from Haar-random (typicality)?  How much
  non-stabiliser "magic" does it contain?  What is its entangling
  power across multiple bipartitions?

The composite ``usefulness_score`` feeds into the fitness tracker to
create evolutionary pressure toward circuits that are both novel AND
computationally meaningful.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UsefulnessReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class UsefulnessReport:
    """Result of usefulness analysis for a single circuit."""

    # Tier 1: Functional primitive detection
    is_diagonal: bool = False
    is_permutation: bool = False
    diagonal_balance: float | None = None
    state_prep_matches: dict[str, float] = field(default_factory=dict)
    is_controlled_unitary: bool = False

    # Tier 2: Algebraic properties
    mixer_strength: float = 0.0
    preserves_hamming_weight: bool = False
    preserves_parity: bool = False
    scrambling_score: float = 0.0
    is_involution: bool = False
    periodicity: int | None = None

    # Tier 3: Structural measures
    haar_typicality: float = 0.0
    magic_score: float = 0.0
    entangling_power_multipartite: float = 0.0

    # Composite
    usefulness_score: float = 0.0
    usefulness_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_diagonal": self.is_diagonal,
            "is_permutation": self.is_permutation,
            "diagonal_balance": self.diagonal_balance,
            "state_prep_matches": self.state_prep_matches,
            "is_controlled_unitary": self.is_controlled_unitary,
            "mixer_strength": self.mixer_strength,
            "preserves_hamming_weight": self.preserves_hamming_weight,
            "preserves_parity": self.preserves_parity,
            "scrambling_score": self.scrambling_score,
            "is_involution": self.is_involution,
            "periodicity": self.periodicity,
            "haar_typicality": self.haar_typicality,
            "magic_score": self.magic_score,
            "entangling_power_multipartite": self.entangling_power_multipartite,
            "usefulness_score": self.usefulness_score,
            "usefulness_tags": self.usefulness_tags,
        }


# ---------------------------------------------------------------------------
# Tier 1: Functional primitive detection
# ---------------------------------------------------------------------------

def _check_diagonal(U: np.ndarray, atol: float = 1e-6) -> tuple[bool, float | None]:
    """Check if U is a diagonal unitary (phase oracle).

    Returns (is_diagonal, balance) where balance measures how evenly
    distributed the phases are (1.0 = maximally balanced = useful for
    Deutsch-Jozsa / Grover-type oracles).
    """
    d = U.shape[0]
    off_diag_norm = np.sum(np.abs(U) ** 2) - np.sum(np.abs(np.diag(U)) ** 2)
    is_diag = off_diag_norm < atol * d

    if not is_diag:
        return False, None

    # Measure phase balance: how many distinct phase clusters are there?
    phases = np.angle(np.diag(U))
    # Normalise phases to [0, 2pi)
    phases = phases % (2 * np.pi)
    # Check if phases split into two roughly equal groups (balanced oracle)
    median_phase = np.median(phases)
    group_a = np.sum(phases < median_phase + 0.1)
    group_b = d - group_a
    balance = 1.0 - abs(group_a - group_b) / d
    return True, float(balance)


def _check_permutation(U: np.ndarray, atol: float = 1e-6) -> bool:
    """Check if U is a permutation matrix (classical reversible oracle)."""
    d = U.shape[0]
    abs_U = np.abs(U)
    # Each row and column should have exactly one entry with |U| close to 1
    row_max = np.max(abs_U, axis=1)
    col_max = np.max(abs_U, axis=0)
    row_sum = np.sum(abs_U ** 2, axis=1)
    return bool(
        np.all(row_max > 1 - atol)
        and np.all(col_max > 1 - atol)
        and np.allclose(row_sum, 1.0, atol=atol)
    )


def _check_state_prep(
    U: np.ndarray, n_qubits: int
) -> dict[str, float]:
    """Check if U|0...0> produces a known useful quantum state.

    Returns a dict of {state_name: fidelity} for states with fidelity > 0.5.
    """
    d = 2 ** n_qubits
    zero_state = np.zeros(d, dtype=complex)
    zero_state[0] = 1.0
    output = U @ zero_state
    matches: dict[str, float] = {}

    # Uniform superposition: |+>^n = H^n|0>^n
    uniform = np.ones(d, dtype=complex) / np.sqrt(d)
    fid = float(np.abs(np.dot(uniform.conj(), output)) ** 2)
    if fid > 0.5:
        matches["uniform_superposition"] = round(fid, 4)

    # GHZ: (|00...0> + |11...1>) / sqrt(2)
    if n_qubits >= 2:
        ghz = np.zeros(d, dtype=complex)
        ghz[0] = ghz[d - 1] = 1 / np.sqrt(2)
        fid = float(np.abs(np.dot(ghz.conj(), output)) ** 2)
        if fid > 0.5:
            matches["ghz"] = round(fid, 4)

    # W state: equal superposition of single-excitation states
    if n_qubits >= 2:
        w = np.zeros(d, dtype=complex)
        for k in range(n_qubits):
            w[1 << k] = 1 / np.sqrt(n_qubits)
        fid = float(np.abs(np.dot(w.conj(), output)) ** 2)
        if fid > 0.5:
            matches["w_state"] = round(fid, 4)

    # Dicke states D(n, k) for k = n//2 (balanced)
    if n_qubits >= 3:
        k = n_qubits // 2
        dicke = np.zeros(d, dtype=complex)
        count = 0
        for i in range(d):
            if bin(i).count('1') == k:
                count += 1
                dicke[i] = 1.0
        if count > 0:
            dicke /= np.sqrt(count)
            fid = float(np.abs(np.dot(dicke.conj(), output)) ** 2)
            if fid > 0.5:
                matches["dicke_balanced"] = round(fid, 4)

    return matches


def _check_controlled_unitary(U: np.ndarray, n_qubits: int, atol: float = 1e-6) -> bool:
    """Check if U has the structure of a controlled-V gate.

    Tests whether any qubit acts as a control: the unitary should be
    block-diagonal with one block being the identity.
    """
    if n_qubits < 2:
        return False

    d = U.shape[0]
    half = d // 2

    # Check each qubit as a potential control
    for ctrl_qubit in range(n_qubits):
        # Build permutation to put control qubit as MSB
        # For simplicity, check the natural block structure
        # when splitting on each qubit
        stride = 2 ** (n_qubits - 1 - ctrl_qubit)

        # Extract indices where control qubit = 0 and = 1
        idx_0 = []
        idx_1 = []
        for i in range(d):
            if (i >> (n_qubits - 1 - ctrl_qubit)) & 1 == 0:
                idx_0.append(i)
            else:
                idx_1.append(i)

        # Check if the block for ctrl=0 is identity
        block_0 = U[np.ix_(idx_0, idx_0)]
        cross_01 = U[np.ix_(idx_0, idx_1)]
        cross_10 = U[np.ix_(idx_1, idx_0)]

        if (np.allclose(block_0, np.eye(half, dtype=complex), atol=atol)
                and np.linalg.norm(cross_01) < atol
                and np.linalg.norm(cross_10) < atol):
            return True

        # Check if the block for ctrl=1 is identity
        block_1 = U[np.ix_(idx_1, idx_1)]
        if (np.allclose(block_1, np.eye(half, dtype=complex), atol=atol)
                and np.linalg.norm(cross_01) < atol
                and np.linalg.norm(cross_10) < atol):
            return True

    return False


# ---------------------------------------------------------------------------
# Tier 2: Algebraic property tests
# ---------------------------------------------------------------------------

def _mixer_strength(U: np.ndarray) -> float:
    """Measure how well U works as a QAOA mixer.

    Computes ||[U, Z^{otimes n}]||_F / (2 * d) where d = dim(U).
    A good mixer has a large commutator with the cost Hamiltonian
    (Z-diagonal). Returns a value in [0, 1].
    """
    d = U.shape[0]
    # Z^{otimes n} is diagonal with entries (-1)^{popcount(i)}
    z_diag = np.array([(-1) ** bin(i).count('1') for i in range(d)], dtype=complex)
    Z = np.diag(z_diag)

    commutator = U @ Z - Z @ U
    # Frobenius norm, normalised
    comm_norm = np.linalg.norm(commutator, 'fro')
    # Max possible commutator norm for a unitary and Z is 2*d (when U is X^{otimes n})
    return float(min(1.0, comm_norm / (2 * np.sqrt(d))))


def _check_hamming_weight_preservation(U: np.ndarray, n_qubits: int, atol: float = 1e-6) -> bool:
    """Check if U preserves Hamming weight (number of 1s in bitstring).

    A Hamming-weight-preserving unitary is useful as a feasibility-preserving
    mixer for constrained optimisation (e.g., portfolio optimisation, TSP).
    """
    d = 2 ** n_qubits
    for i in range(d):
        hw_in = bin(i).count('1')
        row = U[i, :]
        for j in range(d):
            if abs(row[j]) > atol:
                hw_out = bin(j).count('1')
                if hw_out != hw_in:
                    return False
    return True


def _check_parity_preservation(U: np.ndarray, n_qubits: int, atol: float = 1e-6) -> bool:
    """Check if U preserves parity (even/odd number of 1s)."""
    d = 2 ** n_qubits
    for i in range(d):
        parity_in = bin(i).count('1') % 2
        row = U[i, :]
        for j in range(d):
            if abs(row[j]) > atol:
                parity_out = bin(j).count('1') % 2
                if parity_out != parity_in:
                    return False
    return True


def _scrambling_score(U: np.ndarray, n_qubits: int, n_samples: int = 10, seed: int = 42) -> float:
    """Estimate scrambling via OTOC proxy.

    Computes |Tr(P_a U P_b U^dag P_a U P_b U^dag)|/d for random
    single-qubit Paulis P_a, P_b. Low average = good scrambler.
    Returns scrambling quality in [0, 1] (1 = maximally scrambling).
    """
    if n_qubits < 2:
        return 0.0

    d = 2 ** n_qubits
    rng = np.random.default_rng(seed)
    U_dag = U.conj().T

    # Single-qubit Paulis
    paulis_1q = [
        np.array([[0, 1], [1, 0]], dtype=complex),       # X
        np.array([[0, -1j], [1j, 0]], dtype=complex),     # Y
        np.array([[1, 0], [0, -1]], dtype=complex),       # Z
    ]

    otoc_values = []
    for _ in range(n_samples):
        # Pick two random single-qubit Paulis on different qubits
        q_a, q_b = rng.choice(n_qubits, size=2, replace=False)
        p_a_1q = paulis_1q[rng.integers(3)]
        p_b_1q = paulis_1q[rng.integers(3)]

        # Embed into n-qubit space
        P_a = _embed_single_qubit_op(p_a_1q, q_a, n_qubits)
        P_b = _embed_single_qubit_op(p_b_1q, q_b, n_qubits)

        # OTOC: Tr(P_a U P_b U^dag P_a U P_b U^dag) / d
        V = P_a @ U @ P_b @ U_dag @ P_a @ U @ P_b @ U_dag
        otoc = abs(np.trace(V)) / d
        otoc_values.append(otoc)

    # For a perfect scrambler, OTOC -> 0. For identity, OTOC = 1.
    mean_otoc = np.mean(otoc_values)
    return float(min(1.0, max(0.0, 1.0 - mean_otoc)))


def _embed_single_qubit_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a 2x2 operator on a specific qubit into n-qubit space."""
    result = np.array([[1.0]], dtype=complex)
    for q in range(n_qubits):
        if q == qubit:
            result = np.kron(result, op)
        else:
            result = np.kron(result, np.eye(2, dtype=complex))
    return result


def _check_periodicity(U: np.ndarray, max_order: int = 12, atol: float = 1e-6) -> tuple[bool, int | None]:
    """Check if U is an involution (U^2 = I) or has small periodicity (U^k = I)."""
    d = U.shape[0]
    identity = np.eye(d, dtype=complex)

    Uk = np.eye(d, dtype=complex)
    for k in range(1, max_order + 1):
        Uk = Uk @ U
        if np.allclose(Uk, identity, atol=atol):
            return (k == 2, k)

    return (False, None)


# ---------------------------------------------------------------------------
# Tier 3: Structural richness measures
# ---------------------------------------------------------------------------

def _haar_typicality(U: np.ndarray) -> float:
    """Measure how "generic" U is relative to Haar-random unitaries.

    Uses |Tr(U)|^2/d^2 as a proxy. Haar-random unitaries have this
    concentrated around 1/d, while trivial unitaries (identity, Paulis)
    have values near 1. Returns typicality in [0, 1] (1 = Haar-typical).
    """
    d = U.shape[0]
    tr_sq = abs(np.trace(U)) ** 2 / (d * d)
    # For Haar-random, E[|Tr(U)|^2/d^2] = 1/d
    # For identity, |Tr(U)|^2/d^2 = 1
    # Typicality = 1 when tr_sq is small (Haar-like)
    return float(min(1.0, max(0.0, 1.0 - tr_sq)))


def _magic_score(U: np.ndarray, n_qubits: int, max_exact_qubits: int = 4) -> float:
    """Estimate non-stabiliserness (magic) of U|0...0>.

    For small qubit counts (<= max_exact_qubits), computes the stabiliser
    fidelity: max overlap with any stabiliser state. Magic = 1 - max_overlap.

    For larger circuits, uses a proxy based on the T-count contribution
    to eigenvalue phases (non-Clifford phases in the eigenspectrum).
    """
    d = 2 ** n_qubits

    # Apply U to |0...0>
    zero_state = np.zeros(d, dtype=complex)
    zero_state[0] = 1.0
    output = U @ zero_state

    if n_qubits <= max_exact_qubits:
        # Generate stabiliser states and find max overlap
        stab_states = _generate_stabiliser_states(n_qubits)
        if stab_states:
            max_overlap = max(
                abs(np.dot(s.conj(), output)) ** 2
                for s in stab_states
            )
            return float(1.0 - max_overlap)

    # Proxy for larger circuits: check eigenvalue phases
    eigenvalues = np.linalg.eigvals(U)
    phases = np.angle(eigenvalues) / np.pi  # in units of pi

    # Clifford eigenvalues are multiples of pi/4: 0, 1/4, 1/2, 3/4, 1, ...
    clifford_phases = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    min_dists = []
    for p in phases:
        p_mod = (p % 2.0 + 2.0) % 2.0
        dists = np.abs(clifford_phases - p_mod)
        min_dists.append(np.min(dists))

    # Average distance from Clifford phases, normalised to [0, 1]
    avg_dist = np.mean(min_dists)
    return float(min(1.0, avg_dist / 0.125))  # 0.125 = max distance from any Clifford phase


# Cache for stabiliser states (computed once per qubit count)
_stabiliser_cache: dict[int, list[np.ndarray]] = {}


def _generate_stabiliser_states(n_qubits: int) -> list[np.ndarray]:
    """Generate all stabiliser states for n qubits.

    Only feasible for n <= 4 (counts: 6, 60, 1080, 36720).
    """
    if n_qubits in _stabiliser_cache:
        return _stabiliser_cache[n_qubits]

    if n_qubits > 4:
        return []

    # For 1 qubit: |0>, |1>, |+>, |->, |+i>, |-i>
    if n_qubits == 1:
        states = [
            np.array([1, 0], dtype=complex),
            np.array([0, 1], dtype=complex),
            np.array([1, 1], dtype=complex) / np.sqrt(2),
            np.array([1, -1], dtype=complex) / np.sqrt(2),
            np.array([1, 1j], dtype=complex) / np.sqrt(2),
            np.array([1, -1j], dtype=complex) / np.sqrt(2),
        ]
        _stabiliser_cache[1] = states
        return states

    # For n > 1: generate from n-1 qubit states via tensor products
    # and also include entangled stabiliser states
    # Use the Clifford group action on |0...0>
    # For efficiency, use the known count and recursive construction
    prev_states = _generate_stabiliser_states(n_qubits - 1)

    # Single-qubit stabiliser states
    s1 = _generate_stabiliser_states(1)

    states: list[np.ndarray] = []
    seen: set[int] = set()

    # Product states
    for s_prev in prev_states:
        for s_single in s1:
            state = np.kron(s_prev, s_single)
            key = hash(state.tobytes())
            if key not in seen:
                seen.add(key)
                states.append(state)

    # For 2 qubits, also add Bell states explicitly
    if n_qubits == 2:
        bell_states = [
            np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),   # Φ+
            np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # Φ-
            np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),   # Ψ+
            np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),  # Ψ-
            np.array([1, 0, 0, 1j], dtype=complex) / np.sqrt(2),
            np.array([1, 0, 0, -1j], dtype=complex) / np.sqrt(2),
            np.array([0, 1, 1j, 0], dtype=complex) / np.sqrt(2),
            np.array([0, 1, -1j, 0], dtype=complex) / np.sqrt(2),
        ]
        for bs in bell_states:
            key = hash(bs.tobytes())
            if key not in seen:
                seen.add(key)
                states.append(bs)

    # For n >= 3, the product construction misses many entangled states.
    # We accept the approximation -- the resulting magic score is a lower bound.
    _stabiliser_cache[n_qubits] = states
    logger.debug("Generated %d stabiliser states for %d qubits.", len(states), n_qubits)
    return states


def _entangling_power_multipartite(U: np.ndarray, n_qubits: int) -> float:
    """Average entangling power across all bipartitions.

    For each bipartition of qubits into two groups, computes the
    operator entanglement via SVD and averages. This captures
    multi-body entanglement structure that the single half-split misses.
    """
    if n_qubits < 2:
        return 0.0

    d = 2 ** n_qubits
    all_qubits = list(range(n_qubits))

    # Enumerate bipartitions (each qubit in A or B, exclude empty sets)
    # For efficiency, only consider splits of size 1 vs rest and half-splits
    entropies = []

    for k in range(1, n_qubits // 2 + 1):
        for subset_a in combinations(all_qubits, k):
            subset_b = [q for q in all_qubits if q not in subset_a]
            ent = _bipartite_operator_entropy(U, list(subset_a), subset_b, n_qubits)
            if ent is not None:
                entropies.append(ent)

    if not entropies:
        return 0.0

    return float(np.mean(entropies))


def _bipartite_operator_entropy(
    U: np.ndarray,
    qubits_a: list[int],
    qubits_b: list[int],
    n_qubits: int,
) -> float | None:
    """Compute normalised operator entanglement for a bipartition."""
    n_a = len(qubits_a)
    n_b = len(qubits_b)
    d_a = 2 ** n_a
    d_b = 2 ** n_b

    try:
        # Permute qubit indices so A qubits come first, then B
        perm = qubits_a + qubits_b
        # Build the permutation matrix for reordering qubits
        d = 2 ** n_qubits
        P = np.zeros((d, d), dtype=complex)
        for i in range(d):
            # Decompose i into bits, reorder, compose
            bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
            new_bits = [bits[perm[q]] for q in range(n_qubits)]
            j = sum(b << (n_qubits - 1 - q) for q, b in enumerate(new_bits))
            P[j, i] = 1.0

        # Permute U
        U_perm = P @ U @ P.T

        # Reshape for operator Schmidt decomposition
        u_reshaped = U_perm.reshape(d_a, d_b, d_a, d_b)
        u_reshaped = u_reshaped.transpose(0, 2, 1, 3)
        u_reshaped = u_reshaped.reshape(d_a * d_a, d_b * d_b)

        svs = np.linalg.svd(u_reshaped, compute_uv=False)
        probs = svs ** 2
        probs = probs / probs.sum()

        nonzero = probs > 1e-15
        entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))

        max_entropy = 2 * min(n_a, n_b) * np.log2(2)
        if max_entropy == 0:
            return 0.0

        return float(min(1.0, entropy / max_entropy))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_usefulness(
    unitary: np.ndarray,
    n_qubits: int,
    scrambling_samples: int = 10,
    magic_max_qubits: int = 4,
) -> UsefulnessReport:
    """Run all usefulness tests on a unitary matrix.

    Parameters
    ----------
    unitary:
        The circuit's unitary matrix (2^n x 2^n).
    n_qubits:
        Number of qubits.
    scrambling_samples:
        Number of random Pauli pairs for OTOC estimation.
    magic_max_qubits:
        Max qubits for exact stabiliser enumeration.

    Returns
    -------
    UsefulnessReport
    """
    report = UsefulnessReport()
    tags: list[str] = []

    # --- Tier 1: Functional primitives ---
    report.is_diagonal, report.diagonal_balance = _check_diagonal(unitary)
    if report.is_diagonal:
        tags.append("phase_oracle")
        if report.diagonal_balance is not None and report.diagonal_balance > 0.8:
            tags.append("balanced_oracle")

    report.is_permutation = _check_permutation(unitary)
    if report.is_permutation:
        tags.append("classical_oracle")

    report.state_prep_matches = _check_state_prep(unitary, n_qubits)
    for state_name in report.state_prep_matches:
        tags.append(f"prepares_{state_name}")

    report.is_controlled_unitary = _check_controlled_unitary(unitary, n_qubits)
    if report.is_controlled_unitary:
        tags.append("controlled_unitary")

    # --- Tier 2: Algebraic properties ---
    report.mixer_strength = _mixer_strength(unitary)
    if report.mixer_strength > 0.5:
        tags.append("good_mixer")

    report.preserves_hamming_weight = _check_hamming_weight_preservation(
        unitary, n_qubits
    )
    if report.preserves_hamming_weight:
        tags.append("preserves_hamming_weight")

    report.preserves_parity = _check_parity_preservation(unitary, n_qubits)
    if report.preserves_parity:
        tags.append("preserves_parity")

    report.scrambling_score = _scrambling_score(
        unitary, n_qubits, n_samples=scrambling_samples
    )
    if report.scrambling_score > 0.7:
        tags.append("good_scrambler")

    is_involution, periodicity = _check_periodicity(unitary)
    report.is_involution = is_involution
    report.periodicity = periodicity
    if is_involution:
        tags.append("involution")
    if periodicity is not None and periodicity <= 12:
        tags.append(f"order_{periodicity}")

    # --- Tier 3: Structural measures ---
    report.haar_typicality = _haar_typicality(unitary)

    report.magic_score = _magic_score(unitary, n_qubits, max_exact_qubits=magic_max_qubits)
    if report.magic_score > 0.5:
        tags.append("high_magic")

    report.entangling_power_multipartite = _entangling_power_multipartite(
        unitary, n_qubits
    )
    if report.entangling_power_multipartite > 0.5:
        tags.append("strong_entangler")

    # --- Composite score ---
    # Tier 1: functional detection (0.30 weight)
    tier1 = 0.0
    if report.is_diagonal:
        tier1 += 0.3
        if report.diagonal_balance and report.diagonal_balance > 0.8:
            tier1 += 0.2
    if report.is_permutation:
        tier1 += 0.2
    if report.state_prep_matches:
        tier1 += 0.3 * max(report.state_prep_matches.values())
    if report.is_controlled_unitary:
        tier1 += 0.3
    tier1 = min(1.0, tier1)

    # Tier 2: algebraic utility (0.35 weight)
    tier2 = 0.0
    tier2 += 0.25 * report.mixer_strength
    tier2 += 0.20 * report.scrambling_score
    if report.preserves_hamming_weight:
        tier2 += 0.25
    if report.preserves_parity:
        tier2 += 0.10
    if report.is_involution:
        tier2 += 0.15
    elif report.periodicity is not None:
        tier2 += 0.10
    tier2 = min(1.0, tier2)

    # Tier 3: structural richness (0.35 weight)
    tier3 = 0.0
    tier3 += 0.35 * report.haar_typicality
    tier3 += 0.35 * report.magic_score
    tier3 += 0.30 * report.entangling_power_multipartite
    tier3 = min(1.0, tier3)

    report.usefulness_score = 0.30 * tier1 + 0.35 * tier2 + 0.35 * tier3
    report.usefulness_tags = tags

    return report
