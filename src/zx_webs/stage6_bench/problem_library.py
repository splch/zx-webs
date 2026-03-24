"""Problem library for screening composed circuits against useful computational tasks.

Instead of only checking whether composed circuits match the 47 known corpus
algorithms, this module generates target unitaries and states from a broad
library of quantum computational problems.  A "hit" -- a composed circuit with
high fidelity to a problem target -- means the pipeline discovered a novel way
to solve a known problem.

Target categories:
    - State preparation (GHZ, W, Dicke, graph states, cluster states)
    - Hamiltonian simulation (TFIM, Heisenberg, XY, Hubbard)
    - Multi-controlled gates (Toffoli, Fredkin, C^n-Z, etc.)
    - Quantum arithmetic (adders, multipliers via Qiskit circuit library)
    - QEC encoding circuits ([[4,2,2]], [[5,1,3]], [[7,1,3]])
"""
from __future__ import annotations

import logging
from functools import reduce
from typing import Any

import numpy as np
from scipy.linalg import expm

from zx_webs.stage6_bench.tasks import BenchmarkTask

logger = logging.getLogger(__name__)

# Baseline metrics for problem library tasks.  Set high so any real circuit
# Pareto-dominates them, making is_improvement = True whenever fidelity
# exceeds the threshold.  This signals "hit against problem library" without
# requiring a known baseline algorithm.
_NO_BASELINE = 999_999

# ---------------------------------------------------------------------------
# Pauli matrices (reused across Hamiltonian builders)
# ---------------------------------------------------------------------------

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(*mats: np.ndarray) -> np.ndarray:
    """Multi-argument Kronecker product."""
    return reduce(np.kron, mats)


def _pauli_on_sites(op: np.ndarray, sites: list[int], n: int) -> np.ndarray:
    """Place a single-qubit Pauli `op` on each site in `sites`, identity elsewhere."""
    terms = [op if i in sites else I for i in range(n)]
    return _kron(*terms)


def _two_site_op(op_a: np.ndarray, op_b: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    """Place `op_a` on site `i` and `op_b` on site `j`, identity elsewhere."""
    terms = []
    for k in range(n):
        if k == i:
            terms.append(op_a)
        elif k == j:
            terms.append(op_b)
        else:
            terms.append(I)
    return _kron(*terms)


# ---------------------------------------------------------------------------
# State preparation targets
# ---------------------------------------------------------------------------


def _ghz_state(n: int) -> np.ndarray:
    """GHZ state (|00...0> + |11...1>) / sqrt(2)."""
    d = 2**n
    state = np.zeros(d, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)
    state[-1] = 1.0 / np.sqrt(2)
    return state


def _w_state(n: int) -> np.ndarray:
    """W state: uniform superposition over single-excitation states."""
    d = 2**n
    state = np.zeros(d, dtype=complex)
    for k in range(n):
        idx = 1 << (n - 1 - k)
        state[idx] = 1.0 / np.sqrt(n)
    return state


def _dicke_state(n: int, k: int) -> np.ndarray:
    """Dicke state D(n,k): uniform superposition over Hamming weight k."""
    d = 2**n
    state = np.zeros(d, dtype=complex)
    count = 0
    for idx in range(d):
        if bin(idx).count("1") == k:
            count += 1
    amp = 1.0 / np.sqrt(count) if count > 0 else 0.0
    for idx in range(d):
        if bin(idx).count("1") == k:
            state[idx] = amp
    return state


def _graph_state(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Graph state: apply CZ to edges starting from |+>^n."""
    d = 2**n
    # Start with |+>^n
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    state = reduce(np.kron, [plus] * n)
    # Apply CZ for each edge
    for i, j in edges:
        cz = np.eye(d, dtype=complex)
        for basis in range(d):
            bit_i = (basis >> (n - 1 - i)) & 1
            bit_j = (basis >> (n - 1 - j)) & 1
            if bit_i == 1 and bit_j == 1:
                cz[basis, basis] = -1.0
        state = cz @ state
    return state


def _line_graph_edges(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _cycle_graph_edges(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def _star_graph_edges(n: int) -> list[tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def _complete_graph_edges(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _state_prep_tasks(qubit_counts: list[int]) -> list[BenchmarkTask]:
    """Generate state preparation benchmark tasks."""
    tasks: list[BenchmarkTask] = []

    for n in qubit_counts:
        if n < 3:
            continue

        state_targets: list[tuple[str, str, np.ndarray]] = []

        # GHZ
        state_targets.append(("ghz_prep", f"Prepare {n}-qubit GHZ state", _ghz_state(n)))

        # W state
        state_targets.append(("w_prep", f"Prepare {n}-qubit W state", _w_state(n)))

        # Dicke states D(n,k) for k=1..n-1 (k=1 is W state, skip)
        for k in range(2, n):
            state_targets.append((
                f"dicke_{k}_prep",
                f"Prepare Dicke state D({n},{k})",
                _dicke_state(n, k),
            ))

        # Graph states: line, cycle, star, complete
        for graph_name, edge_fn in [
            ("line", _line_graph_edges),
            ("cycle", _cycle_graph_edges),
            ("star", _star_graph_edges),
            ("complete", _complete_graph_edges),
        ]:
            state_targets.append((
                f"graph_{graph_name}_prep",
                f"Prepare {graph_name} graph state on {n} qubits",
                _graph_state(n, edge_fn(n)),
            ))

        # Cluster state (line graph = 1D cluster)
        state_targets.append((
            "cluster_1d_prep",
            f"Prepare 1D cluster state on {n} qubits",
            _graph_state(n, _line_graph_edges(n)),
        ))

        for short_name, desc, target_state in state_targets:
            tasks.append(BenchmarkTask(
                name=f"{short_name}_{n}q",
                description=f"problem_library/state_prep: {desc}",
                n_qubits=n,
                target_unitary=np.eye(1),  # unused for state prep
                target_state=target_state,
                target_type="state_prep",
                baseline_gate_count=_NO_BASELINE,
                baseline_t_count=_NO_BASELINE,
                baseline_cnot_count=_NO_BASELINE,
                baseline_depth=_NO_BASELINE,
                metric_focus=["fidelity"],
            ))

    logger.info("Built %d state preparation tasks.", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Hamiltonian simulation targets
# ---------------------------------------------------------------------------


def _tfim_hamiltonian(n: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:
    """Transverse-field Ising model: H = -J sum Z_i Z_{i+1} - h sum X_i."""
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        H -= J * _two_site_op(Z, Z, i, i + 1, n)
    for i in range(n):
        H -= h * _pauli_on_sites(X, [i], n)
    return H


def _heisenberg_hamiltonian(n: int, J: float = 1.0, h: float = 0.0) -> np.ndarray:
    """Heisenberg XXX model: H = J sum (XX + YY + ZZ) - h sum Z_i."""
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        H += J * (_two_site_op(X, X, i, i + 1, n)
                   + _two_site_op(Y, Y, i, i + 1, n)
                   + _two_site_op(Z, Z, i, i + 1, n))
    for i in range(n):
        H -= h * _pauli_on_sites(Z, [i], n)
    return H


def _xy_hamiltonian(n: int, J: float = 1.0) -> np.ndarray:
    """XY model: H = J sum (XX + YY)."""
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        H += J * (_two_site_op(X, X, i, i + 1, n)
                   + _two_site_op(Y, Y, i, i + 1, n))
    return H


def _hubbard_hamiltonian(n_sites: int, t: float = 1.0, U: float = 2.0) -> np.ndarray:
    """1D Fermi-Hubbard model via Jordan-Wigner.

    Uses 2 * n_sites qubits (spin-up on first n_sites, spin-down on next).
    H = -t sum_sigma (c†_i c_{i+1} + h.c.) + U sum_i n_{i,up} n_{i,down}

    Jordan-Wigner mapping:
        c†_i c_{i+1} = (X_i X_{i+1} + Y_i Y_{i+1}) / 4  (adjacent in JW ordering)
        n_i = (I - Z_i) / 2
    """
    n_qubits = 2 * n_sites
    d = 2**n_qubits
    H = np.zeros((d, d), dtype=complex)

    # Hopping: spin-up sites 0..n_sites-1, spin-down sites n_sites..2*n_sites-1
    for spin_offset in [0, n_sites]:
        for i in range(n_sites - 1):
            site_a = spin_offset + i
            site_b = spin_offset + i + 1
            H -= t * 0.5 * (_two_site_op(X, X, site_a, site_b, n_qubits)
                             + _two_site_op(Y, Y, site_a, site_b, n_qubits))

    # On-site interaction: n_{i,up} * n_{i,down}
    for i in range(n_sites):
        n_up = 0.5 * (np.eye(d, dtype=complex) - _pauli_on_sites(Z, [i], n_qubits))
        n_down = 0.5 * (np.eye(d, dtype=complex) - _pauli_on_sites(Z, [n_sites + i], n_qubits))
        H += U * (n_up @ n_down)

    return H


def _hamiltonian_tasks(
    qubit_counts: list[int],
    times: list[float] | None = None,
    h_values: list[float] | None = None,
) -> list[BenchmarkTask]:
    """Generate Hamiltonian simulation benchmark tasks: e^{-iHt}."""
    if times is None:
        times = [0.1, 0.5, 1.0]
    if h_values is None:
        h_values = [0.5, 1.0, 2.0]

    tasks: list[BenchmarkTask] = []

    for n in qubit_counts:
        if n < 3:
            continue

        hamiltonians: list[tuple[str, str, np.ndarray]] = []

        # TFIM at various field strengths
        for h in h_values:
            hamiltonians.append((
                f"tfim_h{h}",
                f"TFIM {n}q J=1 h={h}",
                _tfim_hamiltonian(n, J=1.0, h=h),
            ))

        # Heisenberg XXX
        hamiltonians.append((
            "heisenberg",
            f"Heisenberg XXX {n}q J=1",
            _heisenberg_hamiltonian(n, J=1.0),
        ))

        # XY model
        hamiltonians.append((
            "xy",
            f"XY model {n}q J=1",
            _xy_hamiltonian(n, J=1.0),
        ))

        # Fermi-Hubbard (needs even qubit count, 2 qubits per site)
        if n >= 4 and n % 2 == 0:
            n_sites = n // 2
            for u_val in [2.0, 4.0]:
                hamiltonians.append((
                    f"hubbard_U{u_val}",
                    f"Fermi-Hubbard {n_sites}-site (={n}q) t=1 U={u_val}",
                    _hubbard_hamiltonian(n_sites, t=1.0, U=u_val),
                ))

        for ham_name, ham_desc, H in hamiltonians:
            for t_val in times:
                U_target = expm(-1j * H * t_val)
                tasks.append(BenchmarkTask(
                    name=f"{ham_name}_t{t_val}_{n}q",
                    description=f"problem_library/hamiltonian: {ham_desc} at t={t_val}",
                    n_qubits=n,
                    target_unitary=U_target,
                    target_type="unitary",
                    baseline_gate_count=_NO_BASELINE,
                    baseline_t_count=_NO_BASELINE,
                    baseline_cnot_count=_NO_BASELINE,
                    baseline_depth=_NO_BASELINE,
                    metric_focus=["fidelity"],
                ))

    logger.info("Built %d Hamiltonian simulation tasks.", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Multi-controlled gate targets
# ---------------------------------------------------------------------------


def _controlled_gate_tasks(qubit_counts: list[int]) -> list[BenchmarkTask]:
    """Generate multi-controlled gate benchmark tasks via Qiskit."""
    tasks: list[BenchmarkTask] = []

    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
    except ImportError:
        logger.warning("Qiskit not available; skipping controlled gate tasks.")
        return tasks

    for n in qubit_counts:
        if n < 3:
            continue

        gate_circuits: list[tuple[str, str, QuantumCircuit]] = []

        # Toffoli (3 qubits)
        if n == 3:
            qc = QuantumCircuit(3)
            qc.ccx(0, 1, 2)
            gate_circuits.append(("toffoli", "Toffoli (CCX) gate", qc))

            qc = QuantumCircuit(3)
            qc.cswap(0, 1, 2)
            gate_circuits.append(("fredkin", "Fredkin (CSWAP) gate", qc))

            qc = QuantumCircuit(3)
            qc.ccz(0, 1, 2)
            gate_circuits.append(("ccz", "CCZ gate", qc))

        # C^(n-1)-X (n qubits, 1 target, n-1 controls)
        if n >= 3:
            qc = QuantumCircuit(n)
            qc.mcx(list(range(n - 1)), n - 1)
            gate_circuits.append((
                f"mc{n-1}x",
                f"Multi-controlled X with {n-1} controls",
                qc,
            ))

        # C^(n-1)-Z
        if n >= 3:
            qc = QuantumCircuit(n)
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
            gate_circuits.append((
                f"mc{n-1}z",
                f"Multi-controlled Z with {n-1} controls",
                qc,
            ))

        for gate_name, desc, qc in gate_circuits:
            try:
                U = Operator(qc).data
                tasks.append(BenchmarkTask(
                    name=f"{gate_name}_{n}q",
                    description=f"problem_library/controlled_gate: {desc}",
                    n_qubits=n,
                    target_unitary=np.array(U, dtype=complex),
                    target_type="unitary",
                    baseline_gate_count=_NO_BASELINE,
                    baseline_t_count=_NO_BASELINE,
                    baseline_cnot_count=_NO_BASELINE,
                    baseline_depth=_NO_BASELINE,
                    metric_focus=["fidelity"],
                ))
            except Exception:
                logger.debug("Failed to build gate target %s at %dq.", gate_name, n)

    logger.info("Built %d controlled gate tasks.", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Quantum arithmetic targets
# ---------------------------------------------------------------------------


def _arithmetic_tasks(qubit_counts: list[int]) -> list[BenchmarkTask]:
    """Generate quantum arithmetic benchmark tasks via Qiskit circuit library."""
    tasks: list[BenchmarkTask] = []

    try:
        from qiskit.quantum_info import Operator
    except ImportError:
        logger.warning("Qiskit not available; skipping arithmetic tasks.")
        return tasks

    # Try importing specific circuits; each may or may not exist in this Qiskit version
    arithmetic_generators: list[tuple[str, str, Any]] = []

    try:
        from qiskit.circuit.library import QFTGate
        from qiskit import QuantumCircuit as _QC
        for n in qubit_counts:
            if n < 3:
                continue
            qc = _QC(n)
            qc.append(QFTGate(n), range(n))
            arithmetic_generators.append((f"qft_{n}q", f"QFT on {n} qubits", qc))
    except (ImportError, Exception) as e:
        logger.debug("QFT generation failed: %s", e)

    try:
        from qiskit.circuit.library import QFTGate
        from qiskit import QuantumCircuit as _QC
        for n in qubit_counts:
            if n < 3:
                continue
            qc = _QC(n)
            qc.append(QFTGate(n).inverse(), range(n))
            arithmetic_generators.append((f"iqft_{n}q", f"Inverse QFT on {n} qubits", qc))
    except (ImportError, Exception) as e:
        logger.debug("Inverse QFT generation failed: %s", e)

    # Draper QFT adder
    try:
        from qiskit.circuit.library import DraperQFTAdder
        for num_bits in [1, 2]:
            n = 2 * num_bits + 1  # a(num_bits) + b(num_bits) + carry
            if n in qubit_counts or any(q >= n for q in qubit_counts):
                qc = DraperQFTAdder(num_bits)
                actual_n = qc.num_qubits
                if actual_n in qubit_counts:
                    arithmetic_generators.append((
                        f"draper_adder_{num_bits}bit_{actual_n}q",
                        f"Draper QFT adder ({num_bits}-bit) on {actual_n} qubits",
                        qc,
                    ))
    except (ImportError, Exception) as e:
        logger.debug("Draper adder generation failed: %s", e)

    # CDKM ripple-carry adder
    try:
        from qiskit.circuit.library import CDKMRippleCarryAdder
        for num_bits in [1, 2]:
            qc = CDKMRippleCarryAdder(num_bits)
            actual_n = qc.num_qubits
            if actual_n in qubit_counts:
                arithmetic_generators.append((
                    f"cdkm_adder_{num_bits}bit_{actual_n}q",
                    f"CDKM ripple-carry adder ({num_bits}-bit) on {actual_n} qubits",
                    qc,
                ))
    except (ImportError, Exception) as e:
        logger.debug("CDKM adder generation failed: %s", e)

    for name, desc, qc in arithmetic_generators:
        try:
            U = Operator(qc).data
            n = qc.num_qubits
            tasks.append(BenchmarkTask(
                name=name,
                description=f"problem_library/arithmetic: {desc}",
                n_qubits=n,
                target_unitary=np.array(U, dtype=complex),
                target_type="unitary",
                baseline_gate_count=_NO_BASELINE,
                baseline_t_count=_NO_BASELINE,
                baseline_cnot_count=_NO_BASELINE,
                baseline_depth=_NO_BASELINE,
                metric_focus=["fidelity"],
            ))
        except Exception:
            logger.debug("Failed to build arithmetic target %s.", name)

    logger.info("Built %d arithmetic tasks.", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# QEC encoding targets
# ---------------------------------------------------------------------------


def _qec_tasks(qubit_counts: list[int]) -> list[BenchmarkTask]:
    """Generate QEC encoding unitary targets.

    Builds encoding circuits for small stabilizer codes and extracts their
    unitary matrices.  The encoding maps k logical qubits into n physical
    qubits, so the target unitary is 2^n x 2^n.
    """
    tasks: list[BenchmarkTask] = []

    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
    except ImportError:
        logger.warning("Qiskit not available; skipping QEC tasks.")
        return tasks

    code_circuits: list[tuple[str, str, QuantumCircuit]] = []

    # [[4,2,2]] code encoding circuit
    # Encodes 2 logical qubits into 4 physical qubits.
    # |psi_L> = CNOT(0,2) CNOT(1,3) H(0) H(1) |psi>|00>
    if 4 in qubit_counts:
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        code_circuits.append(("qec_4_2_2", "[[4,2,2]] error-detecting code encoding", qc))

    # [[5,1,3]] perfect code encoding circuit
    # Standard encoder from Nielsen & Chuang / Laflamme et al.
    if 5 in qubit_counts:
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.cz(0, 3)
        qc.cz(1, 2)
        qc.cz(1, 4)
        qc.cz(2, 3)
        qc.cz(3, 4)
        code_circuits.append(("qec_5_1_3", "[[5,1,3]] perfect code encoding", qc))

    # [[7,1,3]] Steane code encoding circuit
    if 7 in qubit_counts:
        qc = QuantumCircuit(7)
        # Standard Steane encoder: logical qubit on q0, ancillae on q1-q6
        qc.h(3)
        qc.h(4)
        qc.h(5)
        qc.cx(0, 6)
        qc.cx(0, 1)
        qc.cx(4, 0)
        qc.cx(4, 2)
        qc.cx(4, 6)
        qc.cx(5, 0)
        qc.cx(5, 1)
        qc.cx(5, 6)
        qc.cx(3, 2)
        qc.cx(3, 1)
        qc.cx(3, 6)
        code_circuits.append(("qec_7_1_3", "[[7,1,3]] Steane code encoding", qc))

    for name, desc, qc in code_circuits:
        try:
            U = Operator(qc).data
            n = qc.num_qubits
            tasks.append(BenchmarkTask(
                name=f"{name}_{n}q",
                description=f"problem_library/qec: {desc}",
                n_qubits=n,
                target_unitary=np.array(U, dtype=complex),
                target_type="unitary",
                baseline_gate_count=_NO_BASELINE,
                baseline_t_count=_NO_BASELINE,
                baseline_cnot_count=_NO_BASELINE,
                baseline_depth=_NO_BASELINE,
                metric_focus=["fidelity"],
            ))
        except Exception:
            logger.debug("Failed to build QEC target %s.", name)

    logger.info("Built %d QEC encoding tasks.", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_problem_library_tasks(
    qubit_counts: list[int] | None = None,
    categories: list[str] | None = None,
    hamiltonian_times: list[float] | None = None,
    hamiltonian_h_values: list[float] | None = None,
) -> list[BenchmarkTask]:
    """Build benchmark tasks from the problem library.

    Parameters
    ----------
    qubit_counts:
        Which qubit counts to generate targets for.  Defaults to [3, 4, 5].
    categories:
        Which problem categories to include.  Defaults to all:
        ``["state_prep", "hamiltonian", "controlled_gates", "arithmetic", "qec"]``.
    hamiltonian_times:
        Time values for Hamiltonian simulation targets e^{-iHt}.
    hamiltonian_h_values:
        Transverse field strengths for TFIM.

    Returns
    -------
    list[BenchmarkTask]
        Problem library tasks, each with ``target_type`` set to either
        ``"unitary"`` or ``"state_prep"``.
    """
    if qubit_counts is None:
        qubit_counts = [3, 4, 5]
    if categories is None:
        categories = ["state_prep", "hamiltonian", "controlled_gates", "arithmetic", "qec"]

    tasks: list[BenchmarkTask] = []

    builders = {
        "state_prep": lambda: _state_prep_tasks(qubit_counts),
        "hamiltonian": lambda: _hamiltonian_tasks(qubit_counts, hamiltonian_times, hamiltonian_h_values),
        "controlled_gates": lambda: _controlled_gate_tasks(qubit_counts),
        "arithmetic": lambda: _arithmetic_tasks(qubit_counts),
        "qec": lambda: _qec_tasks(qubit_counts),
    }

    for cat in categories:
        builder = builders.get(cat)
        if builder is None:
            logger.warning("Unknown problem library category: %s", cat)
            continue
        try:
            cat_tasks = builder()
            tasks.extend(cat_tasks)
        except Exception:
            logger.warning("Failed to build %s tasks.", cat, exc_info=True)

    logger.info(
        "Problem library: %d total tasks across categories %s for qubit counts %s.",
        len(tasks), categories, qubit_counts,
    )
    return tasks
