"""Quantum algorithm implementations and corpus builder.

Every public algorithm is registered via the ``@register`` decorator and stored
in :data:`ALGORITHM_REGISTRY`.  Each function accepts keyword arguments (most
commonly ``n_qubits``) and returns a :class:`~qiskit.circuit.QuantumCircuit`.
"""
from __future__ import annotations

import logging
import math
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
def build_grover(n_qubits: int = 3, iterations: int = 1) -> QuantumCircuit:
    """Grover's search marking the all-ones state |111...1>.

    Uses *n_qubits* search qubits plus one ancilla qubit.
    """
    total = n_qubits + 1
    qc = QuantumCircuit(total)

    # Ancilla to |1>
    qc.x(n_qubits)

    # Initial superposition
    for q in range(total):
        qc.h(q)

    for _ in range(iterations):
        # --- Oracle: flip phase of |11...1> via multi-controlled X on ancilla ---
        # For the all-ones target, we simply use MCX (decomposes to Toffolis).
        if n_qubits == 1:
            qc.cx(0, n_qubits)
        elif n_qubits == 2:
            qc.ccx(0, 1, n_qubits)
        else:
            qc.mcx(list(range(n_qubits)), n_qubits)

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
def build_simon(n_qubits: int = 4) -> QuantumCircuit:
    """Simon's algorithm for secret period s = 110...0 (MSB pattern).

    Uses *n_qubits* input qubits and *n_qubits* output qubits.
    The oracle maps |x>|0> -> |x>|x XOR (s * f(x))> where f is chosen so
    that f(x) = f(x XOR s).  Here we use the simple oracle that copies x
    to the output register and then XORs by s on the first two output bits.
    """
    if n_qubits < 2:
        raise ValueError("Simon's algorithm requires at least 2 input qubits")

    total = 2 * n_qubits
    qc = QuantumCircuit(total)

    # Hadamard on input register
    for q in range(n_qubits):
        qc.h(q)

    # Oracle: copy input to output register
    for i in range(n_qubits):
        qc.cx(i, n_qubits + i)

    # Oracle: XOR output by s when first bit of input is 1
    # Secret s = 110...0 (bits 0 and 1 set in little-endian)
    qc.cx(0, n_qubits + 0)
    qc.cx(0, n_qubits + 1)

    # Hadamard on input register
    for q in range(n_qubits):
        qc.h(q)

    return qc


build_simon.min_qubits = 2  # type: ignore[attr-defined]


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


# ===================================================================
# Variational family
# ===================================================================


@register("variational", "qaoa_maxcut")
def build_qaoa_maxcut(
    n_qubits: int = 4,
    layers: int = 1,
) -> QuantumCircuit:
    """QAOA ansatz for MaxCut on a ring graph.

    Uses fixed parameters gamma=pi/4 and beta=pi/8 for each layer.
    Edges connect qubit i to qubit (i+1) mod n_qubits.
    """
    gamma = math.pi / 4
    beta = math.pi / 8

    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(layers):
        # Cost unitary: exp(-i * gamma * Z_i Z_j) for each edge (i, j)
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
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
) -> QuantumCircuit:
    """Hardware-efficient variational ansatz for VQE.

    Each layer applies Ry rotations on all qubits followed by a linear
    chain of CNOT gates.  Uses fixed parameter values (multiples of pi/7)
    for reproducibility.
    """
    qc = QuantumCircuit(n_qubits)

    param_counter = 0
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


# ===================================================================
# Simulation family
# ===================================================================


@register("simulation", "trotter_ising")
def build_trotter_ising(n_qubits: int = 4, steps: int = 1) -> QuantumCircuit:
    """First-order Trotter decomposition for the transverse-field Ising model.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    with J=1, h=1, and total time t=1.  The evolution e^{-iHt} is split
    into *steps* Trotter steps, each applying:
        prod_i exp(i*dt*Z_i Z_{i+1})  *  prod_i exp(i*dt*X_i)
    """
    dt = 1.0 / steps  # time per step

    qc = QuantumCircuit(n_qubits)

    for _ in range(steps):
        # ZZ interaction terms: exp(i * dt * J * Z_i Z_{i+1})
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * dt, i + 1)
            qc.cx(i, i + 1)

        # Transverse field terms: exp(i * dt * h * X_i)
        for i in range(n_qubits):
            qc.rx(2 * dt, i)

    return qc


build_trotter_ising.min_qubits = 2  # type: ignore[attr-defined]


@register("simulation", "hamiltonian_sim")
def build_hamiltonian_sim(n_qubits: int = 4) -> QuantumCircuit:
    """Simple Hamiltonian simulation with ZZ coupling and X field terms.

    H = sum_{i} 0.5 * Z_i Z_{i+1} + 0.3 * X_i

    Single Trotter step with t=1.
    """
    t = 1.0
    j_coupling = 0.5
    h_field = 0.3

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


# ===================================================================
# Corpus builder
# ===================================================================


def build_corpus(config: CorpusConfig) -> list[dict[str, Any]]:
    """Build corpus entries from the algorithm registry.

    Each entry is a dict with keys:
        ``algorithm_id``, ``family``, ``name``, ``n_qubits``,
        ``circuit`` (:class:`~qiskit.circuit.QuantumCircuit`).

    Instantiates each algorithm at each qubit count in
    ``config.qubit_counts`` that is <= ``config.max_qubits``.
    Skips algorithms whose minimum qubit count exceeds the target or
    whose family is not in the config's family list.
    """
    corpus: list[dict[str, Any]] = []

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

                qc = fn(**{first_param: n})
                corpus.append(
                    {
                        "algorithm_id": f"{key}_q{n}",
                        "family": family,
                        "name": name,
                        "n_qubits": qc.num_qubits,
                        "circuit": qc,
                    }
                )
                logger.debug("Built %s with %d qubits", key, qc.num_qubits)
            except Exception:
                logger.warning("Failed to build %s at n=%d", key, n, exc_info=True)

    logger.info("Corpus built: %d circuits", len(corpus))
    return corpus
