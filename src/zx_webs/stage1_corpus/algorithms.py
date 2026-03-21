"""Quantum algorithm implementations and corpus builder.

Every public algorithm is registered via the ``@register`` decorator and stored
in :data:`ALGORITHM_REGISTRY`.  Each function accepts keyword arguments (most
commonly ``n_qubits``) and returns a :class:`~qiskit.circuit.QuantumCircuit`.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable

from qiskit import QuantumCircuit

from zx_webs.config import CorpusConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: dict[str, Callable[..., QuantumCircuit]] = {}


def register(family: str, name: str) -> Callable[[Callable[..., QuantumCircuit]], Callable[..., QuantumCircuit]]:
    """Decorator that adds a circuit-builder function to the global registry."""

    def decorator(fn: Callable[..., QuantumCircuit]) -> Callable[..., QuantumCircuit]:
        key = f"{family}/{name}"
        ALGORITHM_REGISTRY[key] = fn
        fn.family = family  # type: ignore[attr-defined]
        fn.algo_name = name  # type: ignore[attr-defined]
        return fn

    return decorator


# ===================================================================
# Oracular family
# ===================================================================


@register("oracular", "deutsch_jozsa")
def build_deutsch_jozsa(n_qubits: int = 3) -> QuantumCircuit:
    """Deutsch-Jozsa algorithm with a balanced oracle.

    Uses *n_qubits* input qubits plus one ancilla qubit initialised to |1>.
    The balanced oracle applies CX from each input qubit to the ancilla.
    """
    total = n_qubits + 1
    qc = QuantumCircuit(total)

    # Initialise ancilla to |1>
    qc.x(n_qubits)

    # Apply Hadamard to all qubits
    for q in range(total):
        qc.h(q)

    # Balanced oracle: CX from every input qubit to ancilla
    for q in range(n_qubits):
        qc.cx(q, n_qubits)

    # Apply Hadamard to input qubits
    for q in range(n_qubits):
        qc.h(q)

    # Measurement layer omitted -- we export pure unitary circuits
    return qc


build_deutsch_jozsa.min_qubits = 2  # type: ignore[attr-defined]


@register("oracular", "bernstein_vazirani")
def build_bernstein_vazirani(
    n_qubits: int = 4,
    secret: int | None = None,
) -> QuantumCircuit:
    """Bernstein-Vazirani algorithm.

    Recovers a secret bit-string *s* encoded in the oracle U_s|x>|y> = |x>|y + s.x>.
    If *secret* is ``None``, defaults to ``2**n_qubits - 1`` (all ones).
    """
    if secret is None:
        secret = (1 << n_qubits) - 1  # all-ones string

    total = n_qubits + 1
    qc = QuantumCircuit(total)

    # Ancilla to |1>
    qc.x(n_qubits)

    # Hadamard on all
    for q in range(total):
        qc.h(q)

    # Oracle: CX from qubit i to ancilla when bit i of secret is set
    for i in range(n_qubits):
        if (secret >> i) & 1:
            qc.cx(i, n_qubits)

    # Hadamard on input qubits
    for q in range(n_qubits):
        qc.h(q)

    return qc


build_bernstein_vazirani.min_qubits = 2  # type: ignore[attr-defined]


@register("oracular", "grover")
def build_grover(
    n_qubits: int = 3,
    iterations: int = 1,
    marked_state: int | None = None,
) -> QuantumCircuit:
    """Grover's search marking a target state.

    Uses *n_qubits* search qubits plus one ancilla qubit.
    If *marked_state* is ``None``, defaults to the all-ones state.
    """
    if marked_state is None:
        marked_state = (1 << n_qubits) - 1  # all-ones

    total = n_qubits + 1
    qc = QuantumCircuit(total)

    # Ancilla to |1>
    qc.x(n_qubits)

    # Initial superposition
    for q in range(total):
        qc.h(q)

    for _ in range(iterations):
        # --- Oracle: flip phase of |marked_state> via X gates + MCX ---
        # Apply X to qubits where marked_state has a 0 bit.
        for i in range(n_qubits):
            if not ((marked_state >> i) & 1):
                qc.x(i)

        if n_qubits == 1:
            qc.cx(0, n_qubits)
        elif n_qubits == 2:
            qc.ccx(0, 1, n_qubits)
        else:
            qc.mcx(list(range(n_qubits)), n_qubits)

        # Undo the X gates.
        for i in range(n_qubits):
            if not ((marked_state >> i) & 1):
                qc.x(i)

        # --- Diffusion operator ---
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits):
            qc.x(q)

        # Multi-controlled Z = H on last search qubit, MCX, H
        qc.h(n_qubits - 1)
        if n_qubits == 1:
            pass  # trivial -- no controls needed besides identity
        elif n_qubits == 2:
            qc.cx(0, n_qubits - 1)
        else:
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

        for q in range(n_qubits):
            qc.x(q)
        for q in range(n_qubits):
            qc.h(q)

    return qc


build_grover.min_qubits = 2  # type: ignore[attr-defined]


@register("oracular", "simon")
def build_simon(n_qubits: int = 4, period: int | None = None) -> QuantumCircuit:
    """Simon's algorithm for a secret period *s*.

    Uses *n_qubits* input qubits and *n_qubits* output qubits.
    The oracle maps |x>|0> -> |x>|x XOR (s * f(x))> where f is chosen so
    that f(x) = f(x XOR s).  Here we use the simple oracle that copies x
    to the output register and then XORs by s based on the period bits.

    If *period* is ``None``, defaults to ``3`` (bits 0 and 1 set).
    """
    if n_qubits < 2:
        raise ValueError("Simon's algorithm requires at least 2 input qubits")

    if period is None:
        period = 3  # bits 0 and 1 set in little-endian

    total = 2 * n_qubits
    qc = QuantumCircuit(total)

    # Hadamard on input register
    for q in range(n_qubits):
        qc.h(q)

    # Oracle: copy input to output register
    for i in range(n_qubits):
        qc.cx(i, n_qubits + i)

    # Oracle: XOR output by s when first bit of input is 1
    for i in range(n_qubits):
        if (period >> i) & 1:
            qc.cx(0, n_qubits + i)

    # Hadamard on input register
    for q in range(n_qubits):
        qc.h(q)

    return qc


build_simon.min_qubits = 2  # type: ignore[attr-defined]


@register("oracular", "deutsch")
def build_deutsch(n_qubits: int = 2) -> QuantumCircuit:
    """Deutsch's algorithm -- the simplest oracle-based quantum algorithm.

    Determines whether a function f:{0,1}->{0,1} is constant or balanced
    using a single query.  Uses one input qubit plus one ancilla.
    We implement the balanced oracle f(x)=x (CX gate).
    *n_qubits* is accepted for interface consistency but the circuit always
    uses 2 qubits (1 input + 1 ancilla).
    """
    qc = QuantumCircuit(2)

    # Ancilla to |1>
    qc.x(1)

    # Hadamard on both
    qc.h(0)
    qc.h(1)

    # Balanced oracle: f(x) = x => CX
    qc.cx(0, 1)

    # Hadamard on input qubit
    qc.h(0)

    return qc


build_deutsch.min_qubits = 2  # type: ignore[attr-defined]


@register("oracular", "quantum_counting")
def build_quantum_counting(n_qubits: int = 3) -> QuantumCircuit:
    """Quantum counting algorithm (Brassard, Hoyer, Mosca, Tapp 1998).

    Combines Grover iterate with phase estimation to estimate the number
    of marked items.  Uses *n_qubits* counting qubits plus 2 qubits for
    a simple 1-qubit Grover search space (1 search + 1 ancilla).

    The Grover oracle marks the |1> state on the search qubit.
    """
    n_count = n_qubits
    # Search space: 1 search qubit + 1 ancilla
    total = n_count + 2
    search_q = n_count
    ancilla = n_count + 1
    qc = QuantumCircuit(total)

    # Prepare ancilla in |1> for phase kickback
    qc.x(ancilla)

    # Hadamard on counting register
    for q in range(n_count):
        qc.h(q)

    # Prepare search qubit in superposition
    qc.h(search_q)
    qc.h(ancilla)

    # Controlled Grover iterates: controlled-G^{2^k}
    for k in range(n_count):
        reps = 1 << k
        for _ in range(reps):
            # Controlled oracle: mark |1> on search qubit
            # Oracle: flip ancilla when search_q = |1>
            qc.ccx(k, search_q, ancilla)

            # Controlled diffusion on search qubit
            qc.ch(k, search_q)
            qc.cz(k, search_q)
            qc.ch(k, search_q)

    # Inverse QFT on counting register
    for i in range(n_count // 2):
        qc.swap(i, n_count - 1 - i)

    for i in range(n_count):
        for j in range(i):
            angle = -math.pi / (1 << (i - j))
            qc.cp(angle, j, i)
        qc.h(i)

    return qc


build_quantum_counting.min_qubits = 2  # type: ignore[attr-defined]


# ===================================================================
# Arithmetic family
# ===================================================================


@register("arithmetic", "qft")
def build_qft(n_qubits: int = 4) -> QuantumCircuit:
    """Standard Quantum Fourier Transform.

    Applies Hadamard and controlled-phase rotations in the textbook order,
    followed by SWAP gates to reverse qubit ordering.
    """
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            angle = math.pi / (1 << (j - i))
            qc.cp(angle, j, i)

    # Reverse qubit order
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)

    return qc


build_qft.min_qubits = 2  # type: ignore[attr-defined]


@register("arithmetic", "inverse_qft")
def build_inverse_qft(n_qubits: int = 4) -> QuantumCircuit:
    """Inverse Quantum Fourier Transform.

    Applies the inverse of the standard QFT: reverse qubit ordering first,
    then inverse controlled-phase rotations and Hadamards in reverse order.
    """
    qc = QuantumCircuit(n_qubits)

    # Reverse qubit order first
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)

    # Inverse QFT operations
    for i in range(n_qubits - 1, -1, -1):
        for j in range(n_qubits - 1, i, -1):
            angle = -math.pi / (1 << (j - i))
            qc.cp(angle, j, i)
        qc.h(i)

    return qc


build_inverse_qft.min_qubits = 2  # type: ignore[attr-defined]


@register("arithmetic", "qpe")
def build_qpe(n_precision: int = 3) -> QuantumCircuit:
    """Quantum Phase Estimation for a T-gate unitary (eigenvalue e^{i pi/4}).

    Uses *n_precision* counting qubits plus one eigenstate qubit initialised
    to |1> (eigenstate of T).
    """
    total = n_precision + 1
    target = n_precision  # index of the eigenstate qubit
    qc = QuantumCircuit(total)

    # Initialise eigenstate to |1>
    qc.x(target)

    # Hadamard on counting register
    for q in range(n_precision):
        qc.h(q)

    # Controlled-U^{2^k} applications
    for k in range(n_precision):
        # T^{2^k} = Phase(pi/4 * 2^k)
        angle = math.pi / 4 * (1 << k)
        qc.cp(angle, k, target)

    # Inverse QFT on counting register
    for i in range(n_precision // 2):
        qc.swap(i, n_precision - 1 - i)

    for i in range(n_precision):
        for j in range(i):
            angle = -math.pi / (1 << (i - j))
            qc.cp(angle, j, i)
        qc.h(i)

    return qc


build_qpe.min_qubits = 2  # type: ignore[attr-defined]


@register("arithmetic", "ripple_adder")
def build_ripple_adder(n_bits: int = 2) -> QuantumCircuit:
    """Ripple-carry adder (Cuccaro et al. style).

    Adds two *n_bits*-bit numbers stored in registers A and B, using one
    ancilla carry qubit.  Layout: [a_0 .. a_{n-1}, b_0 .. b_{n-1}, carry].
    The sum overwrites register B and the final carry is in the carry qubit.
    """
    total = 2 * n_bits + 1
    qc = QuantumCircuit(total)

    a = list(range(n_bits))
    b = list(range(n_bits, 2 * n_bits))
    carry = 2 * n_bits

    # Propagate carry forward
    for i in range(n_bits):
        qc.ccx(a[i], b[i], carry if i == n_bits - 1 else b[i + 1] if i + 1 < n_bits else carry)
        qc.cx(a[i], b[i])

    # For a clean ripple-carry, we do the reverse pass
    # This computes a[i] + b[i] + c[i-1] properly
    # Re-implement using the standard MAJ-UMA decomposition
    return _ripple_adder_maj_uma(n_bits)


def _ripple_adder_maj_uma(n_bits: int) -> QuantumCircuit:
    """Ripple-carry adder using MAJ (majority) and UMA (unmajority-and-add) gates.

    Qubit layout: [c_0, a_0, b_0, a_1, b_1, ..., a_{n-1}, b_{n-1}]
    where c_0 is the input carry.  After execution b_i holds the sum bits
    and c_0 holds the output carry (for n_bits=1) or the carry propagates
    to the last a qubit.
    """
    # Layout: carry, then pairs (a_i, b_i)
    total = 1 + 2 * n_bits
    qc = QuantumCircuit(total)

    carry = 0

    def a(i: int) -> int:
        return 1 + 2 * i

    def b(i: int) -> int:
        return 2 + 2 * i

    def maj(qc: QuantumCircuit, x: int, y: int, z: int) -> None:
        """MAJ gate: majority of three bits."""
        qc.cx(z, y)
        qc.cx(z, x)
        qc.ccx(x, y, z)

    def uma(qc: QuantumCircuit, x: int, y: int, z: int) -> None:
        """UMA gate: unmajority and add."""
        qc.ccx(x, y, z)
        qc.cx(z, x)
        qc.cx(x, y)

    # Forward pass: MAJ gates
    # First MAJ uses carry
    maj(qc, carry, b(0), a(0))
    for i in range(1, n_bits):
        maj(qc, a(i - 1), b(i), a(i))

    # Reverse pass: UMA gates
    for i in range(n_bits - 1, 0, -1):
        uma(qc, a(i - 1), b(i), a(i))
    uma(qc, carry, b(0), a(0))

    return qc


build_ripple_adder.min_qubits = 2  # type: ignore[attr-defined]


@register("arithmetic", "draper_adder")
def build_draper_adder(n_bits: int = 3) -> QuantumCircuit:
    """QFT-based adder (Draper 2000).

    Adds two *n_bits*-bit numbers using the Quantum Fourier Transform.
    Register A is qubits [0..n_bits-1], register B is [n_bits..2*n_bits-1].
    The sum A+B is stored in register B.
    """
    total = 2 * n_bits
    qc = QuantumCircuit(total)

    a = list(range(n_bits))
    b = list(range(n_bits, 2 * n_bits))

    # QFT on register B
    for i in range(n_bits):
        qc.h(b[i])
        for j in range(i + 1, n_bits):
            angle = math.pi / (1 << (j - i))
            qc.cp(angle, b[j], b[i])

    # Controlled phase rotations: add A to B in Fourier space
    for i in range(n_bits):
        for j in range(i, n_bits):
            angle = math.pi / (1 << (j - i))
            qc.cp(angle, a[j], b[i])

    # Inverse QFT on register B
    for i in range(n_bits - 1, -1, -1):
        for j in range(n_bits - 1, i, -1):
            angle = -math.pi / (1 << (j - i))
            qc.cp(angle, b[j], b[i])
        qc.h(b[i])

    return qc


build_draper_adder.min_qubits = 2  # type: ignore[attr-defined]


@register("arithmetic", "quantum_multiplier")
def build_quantum_multiplier(n_bits: int = 2) -> QuantumCircuit:
    """Simple quantum multiplier circuit using controlled additions.

    Multiplies two *n_bits*-bit numbers A and B.  Layout:
    [a_0..a_{n-1}, b_0..b_{n-1}, result_0..result_{2n-1}].
    Uses schoolbook multiplication: for each bit of A, conditionally
    add a shifted copy of B to the result register.
    """
    n_result = 2 * n_bits
    total = 2 * n_bits + n_result
    qc = QuantumCircuit(total)

    a = list(range(n_bits))
    b = list(range(n_bits, 2 * n_bits))
    result = list(range(2 * n_bits, total))

    # Schoolbook multiplication: for each bit a[i], add B << i to result
    for i in range(n_bits):
        for j in range(n_bits):
            if i + j < n_result:
                # Controlled add: if a[i]=1, XOR b[j] into result[i+j]
                qc.ccx(a[i], b[j], result[i + j])

    return qc


build_quantum_multiplier.min_qubits = 2  # type: ignore[attr-defined]


@register("arithmetic", "quantum_comparator")
def build_quantum_comparator(n_bits: int = 2) -> QuantumCircuit:
    """Quantum comparator circuit.

    Compares two *n_bits*-bit numbers A and B using a cascade of Toffoli
    gates.  Layout: [a_0..a_{n-1}, b_0..b_{n-1}, ancilla, result].
    After execution, result qubit is |1> if A >= B.
    """
    total = 2 * n_bits + 2  # ancilla + result
    qc = QuantumCircuit(total)

    a = list(range(n_bits))
    b = list(range(n_bits, 2 * n_bits))
    anc = 2 * n_bits
    res = 2 * n_bits + 1

    # Compare bit by bit from MSB to LSB
    # Simple comparator: compute A XOR B, propagate borrow
    for i in range(n_bits - 1, -1, -1):
        # XOR a[i] into ancilla
        qc.cx(a[i], anc)
        qc.cx(b[i], anc)
        # Conditional propagation to result
        qc.ccx(a[i], anc, res)
        # Reset ancilla
        qc.cx(b[i], anc)
        qc.cx(a[i], anc)

    return qc


build_quantum_comparator.min_qubits = 2  # type: ignore[attr-defined]


# ===================================================================
# Variational family
# ===================================================================


@register("variational", "qaoa_maxcut")
def build_qaoa_maxcut(
    n_qubits: int = 4,
    layers: int = 1,
    edges: list[tuple[int, int]] | None = None,
) -> QuantumCircuit:
    """QAOA ansatz for MaxCut on a given graph.

    Uses fixed parameters gamma=pi/4 and beta=pi/8 for each layer.
    If *edges* is ``None``, defaults to a ring graph.
    """
    gamma = math.pi / 4
    beta = math.pi / 8

    if edges is None:
        # Default: ring graph
        edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(layers):
        # Cost unitary: exp(-i * gamma * Z_i Z_j) for each edge (i, j)
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)

        # Mixer unitary: exp(-i * beta * X_i)
        for q in range(n_qubits):
            qc.rx(2 * beta, q)

    return qc


build_qaoa_maxcut.min_qubits = 3  # type: ignore[attr-defined]


@register("variational", "vqe_hardware_efficient")
def build_vqe_hardware_efficient(
    n_qubits: int = 4,
    layers: int = 1,
    angle_offset: int = 0,
) -> QuantumCircuit:
    """Hardware-efficient variational ansatz for VQE.

    Each layer applies Ry rotations on all qubits followed by a linear
    chain of CNOT gates.  Uses fixed parameter values (multiples of pi/7)
    for reproducibility.  *angle_offset* shifts the parameter counter
    to produce different angle sets.
    """
    qc = QuantumCircuit(n_qubits)

    param_counter = angle_offset
    for layer in range(layers):
        # Ry rotation layer
        for q in range(n_qubits):
            angle = math.pi * (param_counter + 1) / 7
            qc.ry(angle, q)
            param_counter += 1

        # CNOT entangling layer (linear chain)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        # Second Ry rotation layer
        for q in range(n_qubits):
            angle = math.pi * (param_counter + 1) / 7
            qc.ry(angle, q)
            param_counter += 1

    return qc


build_vqe_hardware_efficient.min_qubits = 2  # type: ignore[attr-defined]


@register("variational", "excitation_preserving")
def build_excitation_preserving(
    n_qubits: int = 4,
    layers: int = 1,
    angle_offset: int = 0,
) -> QuantumCircuit:
    """Excitation-preserving variational ansatz.

    Preserves the total number of excitations (Hamming weight).  Each layer
    applies RZ rotations and then excitation-preserving two-qubit gates
    (RXX + RYY combinations) on nearest-neighbour pairs.

    Uses fixed parameter values for reproducibility.
    """
    qc = QuantumCircuit(n_qubits)

    param_counter = angle_offset
    for _ in range(layers):
        # Single-qubit RZ rotations
        for q in range(n_qubits):
            angle = math.pi * (param_counter + 1) / 9
            qc.rz(angle, q)
            param_counter += 1

        # Excitation-preserving two-qubit gates (nearest-neighbour)
        # iSWAP-like: RXX(theta) RYY(theta) preserves excitation count
        for q in range(n_qubits - 1):
            theta = math.pi * (param_counter + 1) / 11
            param_counter += 1

            # RXX(theta): exp(-i * theta/2 * XX)
            qc.h(q)
            qc.h(q + 1)
            qc.cx(q, q + 1)
            qc.rz(theta, q + 1)
            qc.cx(q, q + 1)
            qc.h(q)
            qc.h(q + 1)

            # RYY(theta): exp(-i * theta/2 * YY)
            qc.rx(math.pi / 2, q)
            qc.rx(math.pi / 2, q + 1)
            qc.cx(q, q + 1)
            qc.rz(theta, q + 1)
            qc.cx(q, q + 1)
            qc.rx(-math.pi / 2, q)
            qc.rx(-math.pi / 2, q + 1)

    return qc


build_excitation_preserving.min_qubits = 2  # type: ignore[attr-defined]


@register("variational", "vqe_uccsd_ansatz")
def build_vqe_uccsd_ansatz(
    n_qubits: int = 4,
    layers: int = 1,
) -> QuantumCircuit:
    """Simplified UCCSD-inspired variational ansatz.

    Implements single and double excitation operators commonly used in
    Unitary Coupled-Cluster (UCC) chemistry ansatze.

    Uses *n_qubits* qubits with Hartree-Fock initial state (first n//2
    qubits set to |1>), followed by single and double excitation gates.
    """
    qc = QuantumCircuit(n_qubits)
    n_occ = n_qubits // 2  # occupied orbitals
    n_virt = n_qubits - n_occ  # virtual orbitals

    # Hartree-Fock initial state: first n_occ qubits to |1>
    for q in range(n_occ):
        qc.x(q)

    param_idx = 0
    for _ in range(layers):
        # Single excitations: occupied -> virtual
        for i in range(n_occ):
            for a in range(n_occ, n_qubits):
                theta = math.pi * (param_idx + 1) / 13
                param_idx += 1
                # Givens rotation: exp(theta * (a†_a a_i - a†_i a_a))
                # Implemented as controlled-Ry
                qc.cx(i, a)
                qc.ry(theta, a)
                qc.cx(i, a)
                qc.ry(-theta, a)

        # Double excitations: pairs of occupied -> pairs of virtual
        for i in range(n_occ):
            for j in range(i + 1, n_occ):
                for a in range(n_occ, n_qubits):
                    for b_idx in range(a + 1, n_qubits):
                        theta = math.pi * (param_idx + 1) / 17
                        param_idx += 1
                        # Simplified double excitation via entangling block
                        qc.cx(i, a)
                        qc.cx(j, b_idx)
                        qc.rz(theta, b_idx)
                        qc.cx(j, b_idx)
                        qc.cx(i, a)

    return qc


build_vqe_uccsd_ansatz.min_qubits = 4  # type: ignore[attr-defined]


@register("variational", "qaoa_sk_model")
def build_qaoa_sk_model(
    n_qubits: int = 4,
    layers: int = 1,
    seed: int = 42,
) -> QuantumCircuit:
    """QAOA ansatz for the Sherrington-Kirkpatrick (SK) spin glass model.

    All-to-all random ZZ couplings with fixed parameters.
    The SK model has H = sum_{i<j} J_{ij} Z_i Z_j where J_{ij} are
    random +/- 1 couplings.
    """
    gamma = math.pi / 4
    beta = math.pi / 8

    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(layers):
        # Cost unitary with random J_{ij} couplings
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                j_ij = rng.choice([-1.0, 1.0])
                qc.cx(i, j)
                qc.rz(2 * gamma * j_ij, j)
                qc.cx(i, j)

        # Mixer unitary
        for q in range(n_qubits):
            qc.rx(2 * beta, q)

    return qc


build_qaoa_sk_model.min_qubits = 3  # type: ignore[attr-defined]


# ===================================================================
# Simulation family
# ===================================================================


@register("simulation", "trotter_ising")
def build_trotter_ising(
    n_qubits: int = 4,
    steps: int = 1,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
) -> QuantumCircuit:
    """First-order Trotter decomposition for the transverse-field Ising model.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    The evolution e^{-iHt} (t=1) is split into *steps* Trotter steps.
    """
    dt = 1.0 / steps  # time per step

    qc = QuantumCircuit(n_qubits)

    for _ in range(steps):
        # ZZ interaction terms: exp(i * dt * J * Z_i Z_{i+1})
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * j_coupling * dt, i + 1)
            qc.cx(i, i + 1)

        # Transverse field terms: exp(i * dt * h * X_i)
        for i in range(n_qubits):
            qc.rx(2 * h_field * dt, i)

    return qc


build_trotter_ising.min_qubits = 2  # type: ignore[attr-defined]


@register("simulation", "hamiltonian_sim")
def build_hamiltonian_sim(
    n_qubits: int = 4,
    j_coupling: float = 0.5,
    h_field: float = 0.3,
) -> QuantumCircuit:
    """Simple Hamiltonian simulation with ZZ coupling and X field terms.

    H = sum_{i} j * Z_i Z_{i+1} + h * X_i

    Single Trotter step with t=1.
    """
    t = 1.0

    qc = QuantumCircuit(n_qubits)

    # ZZ terms: exp(-i * t * j * Z_i Z_{i+1})
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(2 * j_coupling * t, i + 1)
        qc.cx(i, i + 1)

    # X field terms: exp(-i * t * h * X_i)
    for i in range(n_qubits):
        qc.rx(2 * h_field * t, i)

    return qc


build_hamiltonian_sim.min_qubits = 2  # type: ignore[attr-defined]


@register("simulation", "trotter_heisenberg")
def build_trotter_heisenberg(
    n_qubits: int = 4,
    steps: int = 1,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    h_field: float = 0.5,
) -> QuantumCircuit:
    """First-order Trotter decomposition for the Heisenberg XXZ model.

    H = sum_i [ Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1} ] + h sum_i Z_i

    Setting Jx=Jy=Jz gives the isotropic Heisenberg (XXX) model.
    Setting Jz=0 gives the XX model.
    """
    dt = 1.0 / steps

    qc = QuantumCircuit(n_qubits)

    for _ in range(steps):
        # XX interaction: exp(-i * dt * Jx * X_i X_{i+1})
        for i in range(n_qubits - 1):
            qc.h(i)
            qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * jx * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i)
            qc.h(i + 1)

        # YY interaction: exp(-i * dt * Jy * Y_i Y_{i+1})
        for i in range(n_qubits - 1):
            qc.rx(math.pi / 2, i)
            qc.rx(math.pi / 2, i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * jy * dt, i + 1)
            qc.cx(i, i + 1)
            qc.rx(-math.pi / 2, i)
            qc.rx(-math.pi / 2, i + 1)

        # ZZ interaction: exp(-i * dt * Jz * Z_i Z_{i+1})
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * jz * dt, i + 1)
            qc.cx(i, i + 1)

        # Magnetic field: exp(-i * dt * h * Z_i)
        for i in range(n_qubits):
            qc.rz(2 * h_field * dt, i)

    return qc


build_trotter_heisenberg.min_qubits = 2  # type: ignore[attr-defined]


@register("simulation", "trotter_xy")
def build_trotter_xy(
    n_qubits: int = 4,
    steps: int = 1,
    j_coupling: float = 1.0,
) -> QuantumCircuit:
    """First-order Trotter decomposition for the XY model.

    H = J sum_i [ X_i X_{i+1} + Y_i Y_{i+1} ]

    This model conserves the total number of excitations and is relevant
    to quantum state transfer and spin chain dynamics.
    """
    dt = 1.0 / steps

    qc = QuantumCircuit(n_qubits)

    for _ in range(steps):
        # XX interaction
        for i in range(n_qubits - 1):
            qc.h(i)
            qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * j_coupling * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i)
            qc.h(i + 1)

        # YY interaction
        for i in range(n_qubits - 1):
            qc.rx(math.pi / 2, i)
            qc.rx(math.pi / 2, i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * j_coupling * dt, i + 1)
            qc.cx(i, i + 1)
            qc.rx(-math.pi / 2, i)
            qc.rx(-math.pi / 2, i + 1)

    return qc


build_trotter_xy.min_qubits = 2  # type: ignore[attr-defined]


@register("simulation", "suzuki_trotter_ising")
def build_suzuki_trotter_ising(
    n_qubits: int = 4,
    steps: int = 1,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
) -> QuantumCircuit:
    """Second-order Suzuki-Trotter decomposition for the transverse-field Ising model.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    Second-order formula: e^{-iHdt} ~ e^{-iH_1 dt/2} e^{-iH_2 dt} e^{-iH_1 dt/2}
    which gives better accuracy than first-order Trotter at the same step count.
    """
    dt = 1.0 / steps

    qc = QuantumCircuit(n_qubits)

    for _ in range(steps):
        # Half-step ZZ interactions
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(j_coupling * dt, i + 1)  # half of 2*J*dt
            qc.cx(i, i + 1)

        # Full-step X field terms
        for i in range(n_qubits):
            qc.rx(2 * h_field * dt, i)

        # Half-step ZZ interactions again
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(j_coupling * dt, i + 1)  # half of 2*J*dt
            qc.cx(i, i + 1)

    return qc


build_suzuki_trotter_ising.min_qubits = 2  # type: ignore[attr-defined]


@register("simulation", "trotter_hubbard")
def build_trotter_hubbard(
    n_qubits: int = 4,
    steps: int = 1,
    t_hop: float = 1.0,
    u_onsite: float = 0.5,
) -> QuantumCircuit:
    """First-order Trotter for the 1D Fermi-Hubbard model (Jordan-Wigner).

    H = -t sum_i (c†_i c_{i+1} + h.c.) + U sum_i n_{i,up} n_{i,down}

    Uses Jordan-Wigner mapping where even qubits are spin-up and odd
    qubits are spin-down on the same site.  *n_qubits* must be even.
    """
    if n_qubits % 2 != 0:
        n_qubits = n_qubits + 1  # ensure even

    n_sites = n_qubits // 2
    qc = QuantumCircuit(n_qubits)
    dt = 1.0 / steps

    for _ in range(steps):
        # Hopping terms: -t (c†_i c_{i+1} + h.c.) for each spin species
        # In JW: XX + YY interaction between adjacent same-spin qubits
        for spin in range(2):  # 0=up, 1=down
            for site in range(n_sites - 1):
                q1 = 2 * site + spin
                q2 = 2 * (site + 1) + spin

                # XX interaction
                qc.h(q1)
                qc.h(q2)
                qc.cx(q1, q2)
                qc.rz(2 * t_hop * dt, q2)
                qc.cx(q1, q2)
                qc.h(q1)
                qc.h(q2)

                # YY interaction
                qc.rx(math.pi / 2, q1)
                qc.rx(math.pi / 2, q2)
                qc.cx(q1, q2)
                qc.rz(2 * t_hop * dt, q2)
                qc.cx(q1, q2)
                qc.rx(-math.pi / 2, q1)
                qc.rx(-math.pi / 2, q2)

        # On-site interaction: U * n_up * n_down = U/4 * (I - Z_up)(I - Z_down)
        for site in range(n_sites):
            q_up = 2 * site
            q_down = 2 * site + 1
            # ZZ term
            qc.cx(q_up, q_down)
            qc.rz(u_onsite * dt / 2, q_down)
            qc.cx(q_up, q_down)
            # Single Z terms
            qc.rz(-u_onsite * dt / 4, q_up)
            qc.rz(-u_onsite * dt / 4, q_down)

    return qc


build_trotter_hubbard.min_qubits = 4  # type: ignore[attr-defined]


# ===================================================================
# Entanglement family
# ===================================================================


@register("entanglement", "ghz")
def build_ghz(n_qubits: int = 5) -> QuantumCircuit:
    """GHZ state preparation: (|00...0> + |11...1>) / sqrt(2)."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


build_ghz.min_qubits = 2  # type: ignore[attr-defined]


@register("entanglement", "w_state")
def build_w_state(n_qubits: int = 4) -> QuantumCircuit:
    """W state preparation: equal superposition of all single-excitation states.

    Produces (|100...0> + |010...0> + ... + |000...1>) / sqrt(n).

    Uses a sequence of controlled rotations to distribute amplitude evenly.
    The k-th qubit receives amplitude sqrt(1/(n-k)) from the remaining
    un-excited amplitude, starting from qubit 0.
    """
    if n_qubits < 2:
        raise ValueError("W state requires at least 2 qubits")

    qc = QuantumCircuit(n_qubits)

    # Put the first qubit into a state that will give 1/n probability
    # Ry(2 * arccos(sqrt(1/n))) |0> = sqrt(1/n)|1> + sqrt((n-1)/n)|0>
    # But we want |1> to be the excited state, so we use arcsin.
    # Actually: Ry(2*theta)|0> = cos(theta)|0> + sin(theta)|1>
    # We want sin(theta) = sqrt(1/n), so theta = arcsin(sqrt(1/n))
    theta = math.asin(math.sqrt(1.0 / n_qubits))
    qc.ry(2 * theta, 0)

    for k in range(1, n_qubits - 1):
        # Rotate qubit k conditioned on qubit k-1 being |0>.
        # We want to split the remaining amplitude equally among (n-k) qubits.
        # The conditional rotation angle: sin(theta_k) = sqrt(1/(n-k))
        theta_k = math.asin(math.sqrt(1.0 / (n_qubits - k)))

        # Controlled-Ry: apply Ry on qubit k controlled on qubit k-1 = |0>
        # |0>-controlled = X on control, then CRy, then X on control
        qc.x(k - 1)
        qc.cry(2 * theta_k, k - 1, k)
        qc.x(k - 1)

    # The last qubit gets a CNOT from the second-to-last qubit (|0>-controlled)
    qc.x(n_qubits - 2)
    qc.cx(n_qubits - 2, n_qubits - 1)
    qc.x(n_qubits - 2)

    return qc


build_w_state.min_qubits = 2  # type: ignore[attr-defined]


@register("entanglement", "bell_state")
def build_bell_state(n_qubits: int = 2) -> QuantumCircuit:
    """Bell state preparation: (|00> + |11>) / sqrt(2).

    Creates the canonical Bell pair on the first two qubits.
    *n_qubits* is accepted for interface consistency but the circuit
    always uses exactly 2 qubits.
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


build_bell_state.min_qubits = 2  # type: ignore[attr-defined]


@register("entanglement", "cluster_state")
def build_cluster_state(n_qubits: int = 4) -> QuantumCircuit:
    """1D linear cluster state preparation.

    Creates a cluster state by applying Hadamard on all qubits followed
    by CZ gates on all nearest-neighbour pairs.  Cluster states are the
    resource states for measurement-based quantum computation (MBQC).
    """
    qc = QuantumCircuit(n_qubits)

    # All qubits into |+>
    for q in range(n_qubits):
        qc.h(q)

    # CZ on nearest-neighbour pairs
    for q in range(n_qubits - 1):
        qc.cz(q, q + 1)

    return qc


build_cluster_state.min_qubits = 2  # type: ignore[attr-defined]


@register("entanglement", "graph_state")
def build_graph_state(n_qubits: int = 4) -> QuantumCircuit:
    """Ring graph state preparation.

    Creates a graph state on a ring topology (all qubits connected to their
    neighbours in a cycle).  Graph states generalise cluster states and are
    fundamental resources in MBQC and quantum error correction.
    """
    qc = QuantumCircuit(n_qubits)

    # All qubits into |+>
    for q in range(n_qubits):
        qc.h(q)

    # CZ on ring edges
    for q in range(n_qubits):
        qc.cz(q, (q + 1) % n_qubits)

    return qc


build_graph_state.min_qubits = 3  # type: ignore[attr-defined]


@register("entanglement", "dicke_state")
def build_dicke_state(n_qubits: int = 4) -> QuantumCircuit:
    """Dicke state D(n, n//2) preparation.

    Prepares the symmetric Dicke state with n//2 excitations, which is
    the equal superposition of all n-qubit states with exactly n//2 ones.
    Uses a cascade of partial Ry rotations and CNOT gates.

    For n=4, k=2: prepares D(4,2) = (|0011> + |0101> + |0110> + |1001> + |1010> + |1100>) / sqrt(6).
    """
    k = n_qubits // 2  # number of excitations

    qc = QuantumCircuit(n_qubits)

    # Start with the first k qubits in |1>
    for q in range(k):
        qc.x(q)

    # Spread excitations using split-and-cyclic-shift pattern
    # For each position from right to left, split excitation probability
    for i in range(n_qubits - 1, 0, -1):
        # Number of excitations still to distribute among qubits [0..i]
        exc_remaining = min(k, i + 1)
        if exc_remaining == 0:
            continue

        # Split the excitation with the right probability
        exc_above = min(k, i)
        if exc_above <= 0 or exc_above >= i + 1:
            continue

        theta = math.acos(math.sqrt(exc_above / (i + 1)))
        qc.cry(2 * theta, i - 1, i)

        # Entangle with previous qubits
        for j in range(i - 1, max(i - exc_above - 1, -1), -1):
            if j >= 0:
                qc.cx(j, i)

    return qc


build_dicke_state.min_qubits = 4  # type: ignore[attr-defined]


# ===================================================================
# Error correction family
# ===================================================================


@register("error_correction", "bit_flip_code")
def build_bit_flip_code(n_qubits: int = 3) -> QuantumCircuit:
    """3-qubit bit-flip repetition code encoder.

    Encodes a single logical qubit into 3 physical qubits to protect
    against single bit-flip (X) errors.  The encoding maps:
    |0> -> |000>, |1> -> |111>.

    Always uses exactly 3 qubits regardless of the n_qubits parameter.
    """
    qc = QuantumCircuit(3)

    # Encode: qubit 0 is the data qubit, qubits 1-2 are ancillae
    qc.cx(0, 1)
    qc.cx(0, 2)

    return qc


build_bit_flip_code.min_qubits = 3  # type: ignore[attr-defined]


@register("error_correction", "phase_flip_code")
def build_phase_flip_code(n_qubits: int = 3) -> QuantumCircuit:
    """3-qubit phase-flip repetition code encoder.

    Encodes a single logical qubit into 3 physical qubits to protect
    against single phase-flip (Z) errors.  Works in the Hadamard basis:
    |+> -> |+++>, |-> -> |--->.

    Always uses exactly 3 qubits.
    """
    qc = QuantumCircuit(3)

    # Encode in the Hadamard basis
    qc.cx(0, 1)
    qc.cx(0, 2)

    # Rotate to the Hadamard basis for phase-flip protection
    qc.h(0)
    qc.h(1)
    qc.h(2)

    return qc


build_phase_flip_code.min_qubits = 3  # type: ignore[attr-defined]


@register("error_correction", "shor_code")
def build_shor_code(n_qubits: int = 9) -> QuantumCircuit:
    """Shor [[9,1,3]] code encoder.

    Encodes one logical qubit into 9 physical qubits by concatenating
    a 3-qubit phase-flip code with 3-qubit bit-flip codes.  This code
    can correct any single-qubit error (bit-flip, phase-flip, or both).

    Always uses exactly 9 qubits.
    """
    qc = QuantumCircuit(9)

    # Phase-flip encoding: spread data qubit across qubits 0, 3, 6
    qc.cx(0, 3)
    qc.cx(0, 6)

    # Move to Hadamard basis for phase-flip protection
    qc.h(0)
    qc.h(3)
    qc.h(6)

    # Bit-flip encoding for each of the three blocks
    # Block 1: qubits 0, 1, 2
    qc.cx(0, 1)
    qc.cx(0, 2)

    # Block 2: qubits 3, 4, 5
    qc.cx(3, 4)
    qc.cx(3, 5)

    # Block 3: qubits 6, 7, 8
    qc.cx(6, 7)
    qc.cx(6, 8)

    return qc


build_shor_code.min_qubits = 9  # type: ignore[attr-defined]


@register("error_correction", "steane_code")
def build_steane_code(n_qubits: int = 7) -> QuantumCircuit:
    """Steane [[7,1,3]] code encoder.

    Encodes one logical qubit into 7 physical qubits using the classical
    [7,4] Hamming code structure.  This is a CSS code that can correct
    any arbitrary single-qubit error.

    The encoding circuit uses Hadamard and CNOT gates following the
    parity check matrix of the Hamming code.

    Always uses exactly 7 qubits.
    """
    qc = QuantumCircuit(7)

    # Data qubit is q0; ancillae are q1-q6
    # Encoding following the Steane code structure:
    # The Steane code is based on the classical [7,4] Hamming code
    # with parity check matrix rows: (1110000), (1001100), (0101010)

    # Create superposition on ancilla qubits
    qc.h(1)
    qc.h(2)
    qc.h(3)

    # Propagate from ancillae to other qubits via CNOT
    # These implement the encoding based on the Hamming code structure
    qc.cx(1, 4)
    qc.cx(1, 5)

    qc.cx(2, 4)
    qc.cx(2, 6)

    qc.cx(3, 5)
    qc.cx(3, 6)

    # Entangle with the data qubit
    qc.cx(0, 4)
    qc.cx(0, 5)
    qc.cx(0, 6)

    return qc


build_steane_code.min_qubits = 7  # type: ignore[attr-defined]


@register("error_correction", "repetition_code")
def build_repetition_code(n_qubits: int = 5) -> QuantumCircuit:
    """n-qubit repetition code encoder.

    Encodes a single logical qubit into *n_qubits* physical qubits.
    Protects against up to (n-1)/2 bit-flip errors.
    Maps |0> -> |00...0>, |1> -> |11...1>.
    """
    if n_qubits < 3:
        n_qubits = 3

    qc = QuantumCircuit(n_qubits)

    # Encode: CNOT from data qubit (0) to all others
    for q in range(1, n_qubits):
        qc.cx(0, q)

    return qc


build_repetition_code.min_qubits = 3  # type: ignore[attr-defined]


@register("error_correction", "surface_code_plaquette")
def build_surface_code_plaquette(n_qubits: int = 5) -> QuantumCircuit:
    """Surface code single plaquette stabilizer circuit.

    Implements one Z-stabilizer and one X-stabilizer measurement cycle
    on a minimal surface code plaquette (4 data + 1 syndrome qubit).

    The circuit entangles 4 data qubits with 1 ancilla qubit to measure
    the weight-4 stabilizer.  Always uses exactly 5 qubits.
    """
    qc = QuantumCircuit(5)

    # Qubit layout: 0-3 are data qubits, 4 is the syndrome ancilla

    # Z-type stabilizer: ZZZZ measurement
    # Prepare ancilla in |+>
    qc.h(4)

    # CNOT from ancilla to each data qubit
    qc.cx(4, 0)
    qc.cx(4, 1)
    qc.cx(4, 2)
    qc.cx(4, 3)

    # Return ancilla to computational basis
    qc.h(4)

    # X-type stabilizer: XXXX measurement
    # Prepare ancilla in |0>
    qc.h(4)

    # CNOT from each data qubit to ancilla
    qc.cx(0, 4)
    qc.cx(1, 4)
    qc.cx(2, 4)
    qc.cx(3, 4)

    qc.h(4)

    return qc


build_surface_code_plaquette.min_qubits = 5  # type: ignore[attr-defined]


# ===================================================================
# Linear algebra family
# ===================================================================


@register("linear_algebra", "swap_test")
def build_swap_test(n_qubits: int = 3) -> QuantumCircuit:
    """SWAP test circuit.

    Uses one ancilla qubit and two input registers of size (n_qubits-1)//2 each.
    The SWAP test estimates the overlap |<psi|phi>|^2 between two quantum
    states by performing a controlled-SWAP and measuring the ancilla.

    For minimum case (n_qubits=3): 1 ancilla + 1 qubit per state = 3 qubits.
    """
    # At least 3 qubits: 1 ancilla + 1 per register
    if n_qubits < 3:
        n_qubits = 3

    # Register sizes
    reg_size = (n_qubits - 1) // 2
    total = 1 + 2 * reg_size
    qc = QuantumCircuit(total)

    ancilla = 0
    reg_a = list(range(1, 1 + reg_size))
    reg_b = list(range(1 + reg_size, 1 + 2 * reg_size))

    # Hadamard on ancilla
    qc.h(ancilla)

    # Controlled-SWAP between registers A and B
    for i in range(reg_size):
        # Fredkin gate = controlled-SWAP decomposition
        qc.cx(reg_b[i], reg_a[i])
        qc.ccx(ancilla, reg_a[i], reg_b[i])
        qc.cx(reg_b[i], reg_a[i])

    # Hadamard on ancilla
    qc.h(ancilla)

    return qc


build_swap_test.min_qubits = 3  # type: ignore[attr-defined]


@register("linear_algebra", "hadamard_test")
def build_hadamard_test(n_qubits: int = 3) -> QuantumCircuit:
    """Hadamard test circuit.

    Estimates <psi|U|psi> for a unitary U by using one ancilla qubit
    and *n_qubits - 1* target qubits.  The ancilla controls the
    application of U.

    We use U = a sequence of controlled-Z and controlled-T gates as
    a representative non-trivial unitary.
    """
    if n_qubits < 2:
        n_qubits = 2

    qc = QuantumCircuit(n_qubits)
    ancilla = 0
    targets = list(range(1, n_qubits))

    # Hadamard on ancilla
    qc.h(ancilla)

    # Controlled-U: apply controlled gates from ancilla to target qubits
    for t in targets:
        qc.cz(ancilla, t)

    # Add controlled-T for non-trivial phase structure
    for t in targets:
        # Controlled-T = controlled phase(pi/4)
        qc.cp(math.pi / 4, ancilla, t)

    # Hadamard on ancilla
    qc.h(ancilla)

    return qc


build_hadamard_test.min_qubits = 2  # type: ignore[attr-defined]


@register("linear_algebra", "inner_product")
def build_inner_product(n_qubits: int = 4) -> QuantumCircuit:
    """Quantum inner product estimation circuit.

    Computes the inner product of two quantum states encoded in
    registers of size n_qubits//2 using the SWAP-based interference
    technique.

    Layout: [reg_a (n//2 qubits), reg_b (n//2 qubits)].
    """
    if n_qubits < 4:
        n_qubits = 4

    half = n_qubits // 2
    total = 2 * half
    qc = QuantumCircuit(total)

    reg_a = list(range(half))
    reg_b = list(range(half, total))

    # Prepare both registers in interesting states using Ry rotations
    for i, q in enumerate(reg_a):
        angle = math.pi * (i + 1) / (half + 1)
        qc.ry(angle, q)

    for i, q in enumerate(reg_b):
        angle = math.pi * (i + 2) / (half + 2)
        qc.ry(angle, q)

    # Bell-basis measurement-like interference
    for i in range(half):
        qc.cx(reg_a[i], reg_b[i])
        qc.h(reg_a[i])

    return qc


build_inner_product.min_qubits = 4  # type: ignore[attr-defined]


@register("linear_algebra", "hhl_rotation")
def build_hhl_rotation(n_qubits: int = 4) -> QuantumCircuit:
    """HHL-style controlled rotation subroutine.

    Implements the key rotation step of the HHL algorithm for solving
    linear systems Ax=b.  Uses a clock register (QPE output) and an
    ancilla qubit.

    Layout: *n_qubits - 1* clock qubits + 1 ancilla for the rotation.
    The rotation angle is proportional to 1/eigenvalue.
    """
    if n_qubits < 3:
        n_qubits = 3

    n_clock = n_qubits - 1
    qc = QuantumCircuit(n_qubits)
    ancilla = n_qubits - 1
    clock = list(range(n_clock))

    # The rotation step: controlled rotations on ancilla
    # For eigenvalue lambda encoded in the clock register,
    # we rotate the ancilla by arcsin(C/lambda)
    for k in range(n_clock):
        # Rotation angle for eigenvalue 2^k
        eigenvalue = 1 << (k + 1)
        c_param = 1.0  # normalization constant
        ratio = min(c_param / eigenvalue, 1.0)
        angle = 2 * math.asin(ratio)
        qc.cry(angle, clock[k], ancilla)

    return qc


build_hhl_rotation.min_qubits = 3  # type: ignore[attr-defined]


@register("linear_algebra", "quantum_walk")
def build_quantum_walk(n_qubits: int = 4) -> QuantumCircuit:
    """Discrete-time quantum walk on a line.

    Implements a coined quantum walk with a Hadamard coin on a 1D position
    space.  Uses 1 qubit for the coin and *n_qubits - 1* qubits for the
    position register (encoding 2^(n-1) positions).

    Each step applies the coin operator (Hadamard on coin qubit) followed
    by the conditional shift operator.
    """
    if n_qubits < 3:
        n_qubits = 3

    qc = QuantumCircuit(n_qubits)
    coin = 0
    pos = list(range(1, n_qubits))
    n_pos = len(pos)

    # Number of walk steps
    n_steps = min(3, n_pos)

    for _ in range(n_steps):
        # Coin flip: Hadamard on coin qubit
        qc.h(coin)

        # Conditional increment (shift right when coin=|1>)
        # Controlled increment on position register
        for i in range(n_pos - 1, 0, -1):
            # Multi-controlled increment
            controls = [coin] + pos[:i]
            target = pos[i]
            if len(controls) == 1:
                qc.cx(controls[0], target)
            elif len(controls) == 2:
                qc.ccx(controls[0], controls[1], target)
            else:
                qc.mcx(controls, target)

        qc.cx(coin, pos[0])

        # Conditional decrement (shift left when coin=|0>)
        qc.x(coin)
        for i in range(n_pos - 1, 0, -1):
            controls = [coin] + pos[:i]
            target = pos[i]
            if len(controls) == 1:
                qc.cx(controls[0], target)
            elif len(controls) == 2:
                qc.ccx(controls[0], controls[1], target)
            else:
                qc.mcx(controls, target)

        qc.cx(coin, pos[0])
        qc.x(coin)

    return qc


build_quantum_walk.min_qubits = 3  # type: ignore[attr-defined]


# ===================================================================
# Communication family
# ===================================================================


@register("communication", "teleportation")
def build_teleportation(n_qubits: int = 3) -> QuantumCircuit:
    """Quantum teleportation protocol (unitary part only).

    Implements the unitary gates of the teleportation protocol without
    measurements (since PyZX cannot handle them).  The circuit:
    1. Creates a Bell pair between qubits 1 and 2.
    2. Applies the Bell-basis transform on qubits 0 and 1.
    3. Applies correction gates (CX, CZ) conditioned on qubits 0 and 1.

    After execution, the state of qubit 0 has been "teleported" to qubit 2
    (in the coherent/deferred-measurement model).

    Always uses exactly 3 qubits.
    """
    qc = QuantumCircuit(3)

    # Create Bell pair between qubits 1 and 2
    qc.h(1)
    qc.cx(1, 2)

    # Bell-basis measurement on qubits 0 and 1 (unitary part)
    qc.cx(0, 1)
    qc.h(0)

    # Coherent corrections (deferred measurement principle)
    qc.cx(1, 2)
    qc.cz(0, 2)

    return qc


build_teleportation.min_qubits = 3  # type: ignore[attr-defined]


@register("communication", "superdense_coding")
def build_superdense_coding(n_qubits: int = 2) -> QuantumCircuit:
    """Superdense coding protocol (unitary part).

    Transmits 2 classical bits using 1 qubit by exploiting a shared
    Bell pair.  The circuit:
    1. Creates a Bell pair.
    2. Applies encoding operations (X and Z) on one qubit to encode 2 bits.
    3. Decodes via CNOT and Hadamard.

    Always uses exactly 2 qubits.
    """
    qc = QuantumCircuit(2)

    # Create Bell pair
    qc.h(0)
    qc.cx(0, 1)

    # Encode two bits: here we encode "11" => apply Z then X
    qc.z(0)
    qc.x(0)

    # Decode
    qc.cx(0, 1)
    qc.h(0)

    return qc


build_superdense_coding.min_qubits = 2  # type: ignore[attr-defined]


@register("communication", "entanglement_swapping")
def build_entanglement_swapping(n_qubits: int = 4) -> QuantumCircuit:
    """Entanglement swapping protocol (coherent version).

    Creates entanglement between qubits 0 and 3 without them ever
    interacting directly, using two Bell pairs and a Bell measurement
    (implemented coherently).

    Layout: qubits 0-1 form one Bell pair, qubits 2-3 form another.
    Bell measurement on qubits 1-2 swaps the entanglement.

    Always uses exactly 4 qubits.
    """
    qc = QuantumCircuit(4)

    # Create Bell pair 1: qubits 0 and 1
    qc.h(0)
    qc.cx(0, 1)

    # Create Bell pair 2: qubits 2 and 3
    qc.h(2)
    qc.cx(2, 3)

    # Bell measurement on qubits 1 and 2 (coherent)
    qc.cx(1, 2)
    qc.h(1)

    # Coherent corrections
    qc.cx(2, 3)
    qc.cz(1, 3)

    return qc


build_entanglement_swapping.min_qubits = 4  # type: ignore[attr-defined]


@register("communication", "ghz_distribution")
def build_ghz_distribution(n_qubits: int = 4) -> QuantumCircuit:
    """GHZ state distribution protocol.

    Distributes a GHZ-like entangled state across *n_qubits* parties
    using a cascade of Bell pairs and entanglement swapping.
    This models a quantum network distributing multipartite entanglement.
    """
    if n_qubits < 3:
        n_qubits = 3

    qc = QuantumCircuit(n_qubits)

    # Create initial Bell pair between first two qubits
    qc.h(0)
    qc.cx(0, 1)

    # Extend entanglement to each subsequent party
    for q in range(2, n_qubits):
        # Create fresh Bell pair using qubit q
        qc.h(q)

        # Entangle with the existing chain via CNOT
        qc.cx(q - 1, q)

        # Apply Hadamard to merge into GHZ-like state
        qc.h(q - 1)

        # Correction
        qc.cx(q - 1, q)

    return qc


build_ghz_distribution.min_qubits = 3  # type: ignore[attr-defined]


@register("communication", "quantum_secret_sharing")
def build_quantum_secret_sharing(n_qubits: int = 3) -> QuantumCircuit:
    """Quantum secret sharing protocol.

    Distributes a quantum secret among *n_qubits* parties such that
    all parties must cooperate to reconstruct the secret.  Based on
    the GHZ-state approach to quantum secret sharing (Hillery et al.).

    The circuit prepares a GHZ state and then applies local rotations
    to encode the secret.
    """
    if n_qubits < 3:
        n_qubits = 3

    qc = QuantumCircuit(n_qubits)

    # Prepare GHZ state
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)

    # Encode secret as a rotation on the first qubit
    # Using a fixed angle for reproducibility
    secret_angle = math.pi / 5
    qc.rz(secret_angle, 0)

    # Apply complementary rotations to enable reconstruction
    for q in range(1, n_qubits):
        qc.rz(secret_angle / (n_qubits - 1), q)

    return qc


build_quantum_secret_sharing.min_qubits = 3  # type: ignore[attr-defined]


@register("communication", "bb84_encoding")
def build_bb84_encoding(n_qubits: int = 4) -> QuantumCircuit:
    """BB84 QKD encoding circuit (unitary part).

    Implements the encoding stage of the BB84 quantum key distribution
    protocol.  Each qubit is encoded in either the Z basis (identity)
    or the X basis (Hadamard) based on a fixed basis choice string,
    and the bit value determines whether X is applied.

    Uses *n_qubits* qubits, each representing one key bit.
    """
    qc = QuantumCircuit(n_qubits)

    # Fixed basis and bit choices for reproducibility
    # Basis: 0=Z, 1=X; Bits: 0 or 1
    for q in range(n_qubits):
        bit_value = q % 2  # alternating 0, 1
        basis = (q // 2) % 2  # alternating Z, X in pairs

        # Encode bit value
        if bit_value:
            qc.x(q)

        # Choose basis
        if basis:
            qc.h(q)

    return qc


build_bb84_encoding.min_qubits = 2  # type: ignore[attr-defined]


# ===================================================================
# Corpus builder
# ===================================================================


def _generate_variants(
    key: str,
    fn: Callable[..., QuantumCircuit],
    name: str,
    n: int,
    first_param: str,
    rng: random.Random,
) -> list[tuple[str, QuantumCircuit]]:
    """Generate multiple variant instances for parameterized algorithms.

    Returns a list of ``(variant_suffix, circuit)`` pairs.
    """
    variants: list[tuple[str, QuantumCircuit]] = []

    if name == "bernstein_vazirani":
        # 3 random secrets per qubit count.
        max_secret = (1 << n) - 1
        secrets = set()
        secrets.add(max_secret)  # all-ones (original)
        while len(secrets) < min(3, max_secret + 1):
            secrets.add(rng.randint(1, max_secret))
        for idx, secret in enumerate(sorted(secrets)):
            qc = fn(**{first_param: n}, secret=secret)
            variants.append((f"_s{secret}", qc))

    elif name == "grover":
        # 3 different marked states (not just all-ones).
        max_state = (1 << n) - 1
        states = set()
        states.add(max_state)  # all-ones (original)
        while len(states) < min(3, max_state + 1):
            states.add(rng.randint(0, max_state))
        for idx, ms in enumerate(sorted(states)):
            qc = fn(**{first_param: n}, marked_state=ms)
            variants.append((f"_m{ms}", qc))

    elif name == "simon":
        # 3 different periods.
        periods = set()
        periods.add(3)  # original
        max_period = (1 << n) - 1
        while len(periods) < min(3, max_period):
            p = rng.randint(1, max_period)
            if p != 0:
                periods.add(p)
        for idx, period in enumerate(sorted(periods)):
            qc = fn(**{first_param: n}, period=period)
            variants.append((f"_p{period}", qc))

    elif name == "qaoa_maxcut":
        # Ring + star + random graph topologies.
        topologies: list[tuple[str, list[tuple[int, int]]]] = []

        # Ring graph (original default).
        ring_edges = [(i, (i + 1) % n) for i in range(n)]
        topologies.append(("ring", ring_edges))

        # Star graph (vertex 0 connected to all others).
        if n >= 3:
            star_edges = [(0, i) for i in range(1, n)]
            topologies.append(("star", star_edges))

        # Random graph (Erdos-Renyi p=0.5).
        if n >= 3:
            rand_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        rand_edges.append((i, j))
            if not rand_edges:
                rand_edges = [(0, 1)]  # ensure at least one edge
            topologies.append(("random", rand_edges))

        for topo_name, edges in topologies:
            qc = fn(**{first_param: n}, edges=edges)
            variants.append((f"_{topo_name}", qc))

    elif name == "vqe_hardware_efficient":
        # 3 different angle sets (offset by 0, 5, 11).
        for offset in [0, 5, 11]:
            qc = fn(**{first_param: n}, angle_offset=offset)
            variants.append((f"_ao{offset}", qc))

    elif name == "excitation_preserving":
        # 3 different angle offsets.
        for offset in [0, 7, 13]:
            qc = fn(**{first_param: n}, angle_offset=offset)
            variants.append((f"_ao{offset}", qc))

    elif name == "trotter_ising":
        # 3 different coupling strengths.
        params_list = [
            (1.0, 1.0, "J1h1"),
            (0.5, 1.5, "J05h15"),
            (2.0, 0.5, "J2h05"),
        ]
        for j, h, label in params_list:
            qc = fn(**{first_param: n}, j_coupling=j, h_field=h)
            variants.append((f"_{label}", qc))

    elif name == "hamiltonian_sim":
        # 3 different coupling strengths.
        params_list = [
            (0.5, 0.3, "J05h03"),
            (1.0, 0.5, "J1h05"),
            (0.3, 1.0, "J03h1"),
        ]
        for j, h, label in params_list:
            qc = fn(**{first_param: n}, j_coupling=j, h_field=h)
            variants.append((f"_{label}", qc))

    elif name == "trotter_heisenberg":
        # 3 different coupling strengths.
        params_list = [
            (1.0, 1.0, 1.0, 0.5, "iso"),      # isotropic XXX
            (1.0, 1.0, 0.0, 0.0, "xy"),        # XY model
            (1.0, 1.0, 2.0, 0.5, "xxz"),       # XXZ anisotropic
        ]
        for jx, jy, jz, h, label in params_list:
            qc = fn(**{first_param: n}, jx=jx, jy=jy, jz=jz, h_field=h)
            variants.append((f"_{label}", qc))

    elif name == "trotter_xy":
        # 3 different coupling strengths.
        for j_val, label in [(0.5, "J05"), (1.0, "J1"), (2.0, "J2")]:
            qc = fn(**{first_param: n}, j_coupling=j_val)
            variants.append((f"_{label}", qc))

    elif name == "suzuki_trotter_ising":
        # 3 different coupling strengths.
        params_list = [
            (1.0, 1.0, "J1h1"),
            (0.5, 1.5, "J05h15"),
            (2.0, 0.5, "J2h05"),
        ]
        for j, h, label in params_list:
            qc = fn(**{first_param: n}, j_coupling=j, h_field=h)
            variants.append((f"_{label}", qc))

    elif name == "trotter_hubbard":
        # 3 different parameter sets.
        params_list = [
            (1.0, 0.5, "t1u05"),
            (0.5, 2.0, "t05u2"),
            (1.0, 1.0, "t1u1"),
        ]
        for t_h, u, label in params_list:
            qc = fn(**{first_param: n}, t_hop=t_h, u_onsite=u)
            variants.append((f"_{label}", qc))

    elif name == "qaoa_sk_model":
        # 3 different random seeds for coupling matrices.
        for seed_val in [42, 137, 271]:
            qc = fn(**{first_param: n}, seed=seed_val)
            variants.append((f"_seed{seed_val}", qc))

    elif name == "superdense_coding":
        # 1 variant -- fixed protocol, no meaningful parameterization
        qc = fn(**{first_param: n})
        variants.append(("", qc))

    else:
        # No multi-instance generation for this algorithm.
        qc = fn(**{first_param: n})
        variants.append(("", qc))

    return variants


def build_corpus(config: CorpusConfig) -> list[dict[str, Any]]:
    """Build corpus entries from the algorithm registry.

    Each entry is a dict with keys:
        ``algorithm_id``, ``family``, ``name``, ``n_qubits``,
        ``circuit`` (:class:`~qiskit.circuit.QuantumCircuit`).

    Instantiates each algorithm at each qubit count in
    ``config.qubit_counts`` that is <= ``config.max_qubits``.
    For parameterized algorithms, generates multiple instances with
    different parameter values for better coverage.
    Skips algorithms whose minimum qubit count exceeds the target or
    whose family is not in the config's family list.
    """
    corpus: list[dict[str, Any]] = []
    rng = random.Random(config.seed)

    for key, fn in sorted(ALGORITHM_REGISTRY.items()):
        family: str = fn.family  # type: ignore[attr-defined]
        name: str = fn.algo_name  # type: ignore[attr-defined]

        # Filter by configured families
        if config.families and family not in config.families:
            continue

        min_q: int = getattr(fn, "min_qubits", 2)

        for n in config.qubit_counts:
            if n > config.max_qubits:
                continue
            if n < min_q:
                continue

            try:
                # Some algorithms use n_qubits, some use n_precision, etc.
                # Inspect the function signature to determine the right kwarg.
                import inspect

                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                first_param = params[0] if params else "n_qubits"

                variants = _generate_variants(key, fn, name, n, first_param, rng)

                for suffix, qc in variants:
                    corpus.append(
                        {
                            "algorithm_id": f"{key}_q{n}{suffix}",
                            "family": family,
                            "name": name,
                            "n_qubits": qc.num_qubits,
                            "circuit": qc,
                        }
                    )
                    logger.debug("Built %s%s with %d qubits", key, suffix, qc.num_qubits)
            except Exception:
                logger.warning("Failed to build %s at n=%d", key, n, exc_info=True)

    logger.info("Corpus built: %d circuits", len(corpus))
    return corpus
