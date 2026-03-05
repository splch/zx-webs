#!/usr/bin/env python3
"""
ZXEA: ZX-Irreducible Entangling Ansatz — Design and Benchmark
==============================================================

Combines ZX-irreducible entangling layers (cluster_chain, hadamard_sandwich
from TVH analysis) with per-qubit variational parameters (from HEA) to get
both expressibility AND trainability.

Benchmarks:
  1. VQE on 4-qubit Heisenberg model
  2. VQE on 4-qubit Transverse-Field Ising model
  3. VQE on 6-qubit Heisenberg model
  4. Expressibility (mean pairwise fidelity)
  5. Gradient variance (trainability / barren plateau detection)
  6. Noise resilience (depolarising noise model)
  7. Convergence speed (energy vs. function evaluations)
  8. Gate efficiency (error per gate, error per 2-qubit gate)

Outputs (scripts/output/discovery/):
  - zxea_report.md     Exhaustive comparison report
  - zxea_results.json  Machine-readable results
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "discovery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Ansatz Definitions
# ═══════════════════════════════════════════════════════════════════════


def make_zxea(params: np.ndarray, n_qubits: int = 4, n_layers: int = 2) -> QuantumCircuit:
    """
    ZX-Irreducible Entangling Ansatz (ZXEA).

    Per layer:
      1. RY(θ_i), RZ(φ_i) on each qubit       (variational)
      2. H on all → CZ chain → H on all        (fixed cluster_chain)
      3. RY(θ'_i) on each qubit                 (variational)

    Parameters per layer: 3 * n_qubits.
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        # Variational rotation block
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits):
            qc.rz(params[idx], q)
            idx += 1
        # Fixed cluster_chain entangling layer
        qc.h(range(n_qubits))
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
        qc.h(range(n_qubits))
        # Post-entangling variational rotation
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
    return qc


def n_params_zxea(n_qubits: int = 4, n_layers: int = 2) -> int:
    return 3 * n_qubits * n_layers


def make_zxea_h(params: np.ndarray, n_qubits: int = 4, n_layers: int = 2) -> QuantumCircuit:
    """
    ZXEA with Hadamard Sandwich (ZXEA-H).

    Per layer:
      1. RY(θ_i), RZ(φ_i) on each qubit       (variational)
      2. H on all → CZ chain → H on all        (fixed cluster_chain)
      3. RY(θ'_i) on each qubit                 (variational)
      4. H-S-H on all qubits                    (fixed hadamard_sandwich = Rx(pi/2))

    Parameters per layer: 3 * n_qubits (same count, one extra fixed layer).
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        # Variational rotation block
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits):
            qc.rz(params[idx], q)
            idx += 1
        # Fixed cluster_chain entangling layer
        qc.h(range(n_qubits))
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
        qc.h(range(n_qubits))
        # Post-entangling variational rotation
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        # Fixed hadamard_sandwich (H-S-H = Rx(pi/2), ZX-irreducible)
        for q in range(n_qubits):
            qc.h(q)
            qc.s(q)
            qc.h(q)
    return qc


def n_params_zxea_h(n_qubits: int = 4, n_layers: int = 2) -> int:
    return 3 * n_qubits * n_layers


def make_hea(params: np.ndarray, n_qubits: int = 4, n_layers: int = 2) -> QuantumCircuit:
    """
    Hardware-Efficient Ansatz (HEA) baseline.

    Per layer:
      1. RY(θ_i), RZ(φ_i) on each qubit
      2. CX chain
      3. RY(θ'_i) on each qubit

    Parameters per layer: 3 * n_qubits.
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits):
            qc.rz(params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
    return qc


def n_params_hea(n_qubits: int = 4, n_layers: int = 2) -> int:
    return 3 * n_qubits * n_layers


def make_qaoa_flex(params: np.ndarray, n_qubits: int = 4, n_layers: int = 2) -> QuantumCircuit:
    """
    Flexible QAOA ansatz with per-layer gamma/beta.

    Per layer:
      1. ZZ interaction: CX-RZ(gamma_l)-CX chain
      2. RX mixer: RX(beta_l) on each qubit

    Parameters per layer: 2 (gamma_l, beta_l).
    """
    qc = QuantumCircuit(n_qubits)
    # Initial superposition
    qc.h(range(n_qubits))
    idx = 0
    for _ in range(n_layers):
        gamma = params[idx]
        beta = params[idx + 1]
        idx += 2
        # ZZ interaction
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            qc.rz(gamma, q + 1)
            qc.cx(q, q + 1)
        # Mixer
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
    return qc


def n_params_qaoa_flex(n_qubits: int = 4, n_layers: int = 2) -> int:
    return 2 * n_layers


def make_tvh_original(params: np.ndarray, n_qubits: int = 4, n_layers: int = 2) -> QuantumCircuit:
    """
    Original TVH with 2 global parameters (gamma, beta).
    """
    gamma, beta = params[0], params[1]
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        # ZZ interaction backbone
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            qc.rz(gamma, q + 1)
            qc.cx(q, q + 1)
        # Cluster chain
        qc.h(range(n_qubits))
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
        qc.h(range(n_qubits))
        # Hadamard sandwich
        for q in range(0, n_qubits - 1, 2):
            qc.h(q)
            qc.rz(beta, q)
            qc.h(q)
        # Mixer
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
    return qc


def n_params_tvh_original(n_qubits: int = 4, n_layers: int = 2) -> int:
    return 2


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Hamiltonians
# ═══════════════════════════════════════════════════════════════════════


def heisenberg_hamiltonian(n_qubits: int) -> tuple[np.ndarray, float]:
    """
    Heisenberg XXX chain: H = Σ_{<i,j>} (XX + YY + ZZ).
    Returns (matrix, exact_ground_energy).
    """
    terms = []
    for i in range(n_qubits - 1):
        for pauli in ["XX", "YY", "ZZ"]:
            label = "I" * i + pauli + "I" * (n_qubits - i - 2)
            terms.append((label, 1.0))
    op = SparsePauliOp.from_list(terms)
    mat = np.asarray(op.to_matrix())
    eigvals = np.linalg.eigvalsh(mat)
    return mat, float(eigvals[0])


def ising_hamiltonian(n_qubits: int, h: float = 1.0) -> tuple[np.ndarray, float]:
    """
    Transverse-field Ising: H = -Σ_{<i,j>} ZZ - h·Σ_i X.
    Returns (matrix, exact_ground_energy).
    """
    terms = []
    for i in range(n_qubits - 1):
        label = "I" * i + "ZZ" + "I" * (n_qubits - i - 2)
        terms.append((label, -1.0))
    for i in range(n_qubits):
        label = "I" * i + "X" + "I" * (n_qubits - i - 1)
        terms.append((label, -h))
    op = SparsePauliOp.from_list(terms)
    mat = np.asarray(op.to_matrix())
    eigvals = np.linalg.eigvalsh(mat)
    return mat, float(eigvals[0])


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: VQE Engine
# ═══════════════════════════════════════════════════════════════════════


def evaluate_energy(params: np.ndarray, make_fn, n_qubits: int, H_mat: np.ndarray) -> float:
    """Compute <ψ(θ)|H|ψ(θ)> via statevector simulation."""
    qc = make_fn(params, n_qubits=n_qubits)
    sv = Statevector.from_instruction(qc)
    psi = sv.data
    return float(np.real(psi.conj() @ H_mat @ psi))


def run_vqe(
    make_fn,
    n_params: int,
    H_mat: np.ndarray,
    n_qubits: int = 4,
    n_restarts: int = 40,
    maxiter: int = 800,
) -> dict:
    """Run multi-restart COBYLA VQE. Returns best energy and statistics."""
    best_energy = np.inf
    best_params = None
    energies = []

    for _ in range(n_restarts):
        x0 = RNG.uniform(0, 2 * np.pi, size=n_params)
        try:
            result = minimize(
                evaluate_energy, x0,
                args=(make_fn, n_qubits, H_mat),
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.5},
            )
            energies.append(result.fun)
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            continue

    return {
        "best_energy": float(best_energy),
        "best_params": best_params.tolist() if best_params is not None else None,
        "mean_energy": float(np.mean(energies)) if energies else None,
        "std_energy": float(np.std(energies)) if energies else None,
        "n_converged": len(energies),
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Expressibility
# ═══════════════════════════════════════════════════════════════════════


def measure_expressibility(
    make_fn, n_params: int, n_qubits: int = 4, n_samples: int = 800,
) -> dict:
    """
    Estimate expressibility via mean pairwise fidelity.
    Lower mean fidelity = more expressible (Haar random ~ 1/(2^n + 1)).
    """
    states = []
    for _ in range(n_samples):
        params = RNG.uniform(0, 2 * np.pi, size=n_params)
        qc = make_fn(params, n_qubits=n_qubits)
        sv = Statevector.from_instruction(qc)
        states.append(sv)

    # Sample pairwise fidelities (not all pairs — too expensive)
    n_pairs = min(2000, n_samples * (n_samples - 1) // 2)
    fidelities = []
    for _ in range(n_pairs):
        i, j = RNG.choice(n_samples, size=2, replace=False)
        f = state_fidelity(states[i], states[j])
        fidelities.append(f)

    haar_mean = 1.0 / (2**n_qubits + 1)
    mean_fid = float(np.mean(fidelities))

    return {
        "mean_fidelity": mean_fid,
        "std_fidelity": float(np.std(fidelities)),
        "haar_reference": haar_mean,
        "expressibility_ratio": mean_fid / haar_mean if haar_mean > 0 else None,
        "n_samples": n_samples,
        "n_pairs": n_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Gradient Variance (Trainability)
# ═══════════════════════════════════════════════════════════════════════


def measure_gradient_variance(
    make_fn,
    n_params: int,
    H_mat: np.ndarray,
    n_qubits: int = 4,
    n_points: int = 200,
    epsilon: float = 0.01,
) -> dict:
    """
    Estimate Var(∂E/∂θ_i) via finite differences at random parameter points.
    Higher variance = more trainable (not in barren plateau).
    """
    all_grads = []

    for _ in range(n_points):
        params = RNG.uniform(0, 2 * np.pi, size=n_params)
        grads = np.zeros(n_params)
        e0 = evaluate_energy(params, make_fn, n_qubits, H_mat)
        for k in range(n_params):
            params_plus = params.copy()
            params_plus[k] += epsilon
            e_plus = evaluate_energy(params_plus, make_fn, n_qubits, H_mat)
            grads[k] = (e_plus - e0) / epsilon
        all_grads.append(grads)

    all_grads = np.array(all_grads)  # (n_points, n_params)
    var_per_param = np.var(all_grads, axis=0)
    mean_grad_var = float(np.mean(var_per_param))

    return {
        "mean_gradient_variance": mean_grad_var,
        "max_gradient_variance": float(np.max(var_per_param)),
        "min_gradient_variance": float(np.min(var_per_param)),
        "n_points": n_points,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Noise Resilience
# ═══════════════════════════════════════════════════════════════════════


def noisy_energy(
    make_fn, params: np.ndarray, n_qubits: int, H_mat: np.ndarray, p_gate: float,
) -> float:
    """
    Simplified depolarising noise model.

    For each gate, the density matrix mixes with the maximally mixed state:
      ρ' = (1 - p_eff) |ψ><ψ| + p_eff · I/2^n

    where p_eff = 1 - (1 - p_gate)^n_gates.
    """
    qc = make_fn(params, n_qubits=n_qubits)
    n_gates = qc.size()
    sv = Statevector.from_instruction(qc)
    psi = sv.data

    # Effective noise
    p_eff = 1.0 - (1.0 - p_gate) ** n_gates
    dim = 2**n_qubits

    # E_noisy = (1 - p_eff) * <ψ|H|ψ> + p_eff * Tr(H) / dim
    e_ideal = float(np.real(psi.conj() @ H_mat @ psi))
    e_mixed = float(np.real(np.trace(H_mat))) / dim

    return (1.0 - p_eff) * e_ideal + p_eff * e_mixed


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Convergence Tracking
# ═══════════════════════════════════════════════════════════════════════


def run_vqe_tracked(
    make_fn,
    n_params: int,
    H_mat: np.ndarray,
    n_qubits: int = 4,
    maxiter: int = 800,
    n_restarts: int = 5,
) -> dict:
    """Run VQE and track best energy vs. function evaluations."""
    best_global_energy = np.inf
    best_trace = None

    for _ in range(n_restarts):
        trace = []
        call_count = [0]
        best_so_far = [np.inf]

        def objective(params):
            e = evaluate_energy(params, make_fn, n_qubits, H_mat)
            call_count[0] += 1
            if e < best_so_far[0]:
                best_so_far[0] = e
            trace.append((call_count[0], best_so_far[0]))
            return e

        x0 = RNG.uniform(0, 2 * np.pi, size=n_params)
        try:
            result = minimize(
                objective, x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.5},
            )
            if result.fun < best_global_energy:
                best_global_energy = result.fun
                best_trace = trace
        except Exception:
            continue

    # Subsample trace for report (at most 50 points)
    if best_trace and len(best_trace) > 50:
        step = len(best_trace) // 50
        best_trace = best_trace[::step]

    return {
        "best_energy": float(best_global_energy),
        "convergence_trace": [(int(n), float(e)) for n, e in (best_trace or [])],
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 8: Gate Count Analysis
# ═══════════════════════════════════════════════════════════════════════


def count_gates(make_fn, n_params: int, n_qubits: int = 4) -> dict:
    """Count total gates and 2-qubit gates."""
    params = np.zeros(n_params)
    qc = make_fn(params, n_qubits=n_qubits)

    total = qc.size()
    two_qubit = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)

    return {
        "total_gates": total,
        "two_qubit_gates": two_qubit,
        "depth": qc.depth(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════


def generate_report(results: dict) -> str:
    """Generate comprehensive markdown report."""
    lines = []

    def add(s=""):
        lines.append(s)

    add("# ZXEA: ZX-Irreducible Entangling Ansatz — Benchmark Report")
    add()
    add("## Hypothesis")
    add()
    add("Combine ZX-irreducible entangling layers (cluster_chain, hadamard_sandwich")
    add("from TVH analysis) with per-qubit variational parameters (from HEA) to achieve")
    add("both high expressibility AND trainability.")
    add()
    add("TVH achieved near-Haar expressibility from just 2 parameters but failed at VQE")
    add("(48% error) due to flat energy landscapes. HEA with 24 per-qubit parameters")
    add("achieves 3.6% error. ZXEA should combine the best of both.")
    add()

    # ── Ansatz Summary ──
    add("## Ansatz Summary")
    add()
    add("| Ansatz | Params (4q, 2L) | Entangling | Key Feature |")
    add("|--------|-----------------|------------|-------------|")
    ansatze = results.get("ansatze", {})
    for name, info in ansatze.items():
        gc = info.get("gate_counts", {})
        add(f"| {name} | {info['n_params']} | "
            f"{gc.get('two_qubit_gates', '?')} 2q-gates | {info.get('description', '')} |")
    add()

    # ── Gate Counts ──
    add("## Gate Counts")
    add()
    add("| Ansatz | Total Gates | 2-Qubit Gates | Depth |")
    add("|--------|-------------|---------------|-------|")
    for name, info in ansatze.items():
        gc = info.get("gate_counts", {})
        add(f"| {name} | {gc.get('total_gates', '?')} | "
            f"{gc.get('two_qubit_gates', '?')} | {gc.get('depth', '?')} |")
    add()

    # ── VQE Results ──
    add("## VQE Results")
    add()
    vqe = results.get("vqe", {})
    for ham_name, ham_results in vqe.items():
        e_exact = ham_results.get("exact_energy", 0)
        add(f"### {ham_name}")
        add(f"Exact ground-state energy: **{e_exact:.4f}**")
        add()
        add("| Ansatz | Best Energy | Error (%) | Mean Energy | Std |")
        add("|--------|-------------|-----------|-------------|-----|")
        for ans_name, r in ham_results.get("results", {}).items():
            best_e = r.get("best_energy", np.inf)
            err_pct = abs(best_e - e_exact) / abs(e_exact) * 100 if e_exact != 0 else 0
            mean_e = r.get("mean_energy")
            std_e = r.get("std_energy")
            mean_str = f"{mean_e:.4f}" if mean_e is not None else "N/A"
            std_str = f"{std_e:.4f}" if std_e is not None else "N/A"
            add(f"| {ans_name} | {best_e:.4f} | {err_pct:.1f}% | {mean_str} | {std_str} |")
        add()

    # ── Expressibility ──
    add("## Expressibility")
    add()
    expr = results.get("expressibility", {})
    if expr:
        haar_ref = list(expr.values())[0].get("haar_reference", 0)
        add(f"Haar random reference (4 qubits): **{haar_ref:.4f}**")
        add()
        add("| Ansatz | Mean Fidelity | Ratio to Haar | Interpretation |")
        add("|--------|---------------|---------------|----------------|")
        for name, r in expr.items():
            mf = r.get("mean_fidelity", 0)
            ratio = r.get("expressibility_ratio", 0)
            if ratio and ratio < 1.5:
                interp = "Near-Haar (excellent)"
            elif ratio and ratio < 3.0:
                interp = "Good"
            elif ratio and ratio < 10.0:
                interp = "Moderate"
            else:
                interp = "Limited"
            add(f"| {name} | {mf:.4f} | {ratio:.2f}x | {interp} |")
        add()

    # ── Gradient Variance ──
    add("## Gradient Variance (Trainability)")
    add()
    grad = results.get("gradient_variance", {})
    if grad:
        add("Higher variance = more trainable (further from barren plateau).")
        add()
        add("| Ansatz | Mean Var(∂E/∂θ) | Max Var | Min Var |")
        add("|--------|-----------------|---------|---------|")
        for name, r in grad.items():
            mv = r.get("mean_gradient_variance", 0)
            maxv = r.get("max_gradient_variance", 0)
            minv = r.get("min_gradient_variance", 0)
            add(f"| {name} | {mv:.6f} | {maxv:.6f} | {minv:.6f} |")
        add()

    # ── Noise Resilience ──
    add("## Noise Resilience")
    add()
    noise = results.get("noise", {})
    if noise:
        add("Energy at optimal noiseless parameters under depolarising noise.")
        add()
        noise_levels = sorted({p for v in noise.values() for p in v})
        header = "| Ansatz | " + " | ".join(f"p={p}" for p in noise_levels) + " |"
        sep = "|--------| " + " | ".join("------" for _ in noise_levels) + " |"
        add(header)
        add(sep)
        for name, nr in noise.items():
            vals = " | ".join(f"{nr.get(p, 'N/A'):.4f}" if isinstance(nr.get(p), (int, float)) else "N/A"
                              for p in noise_levels)
            add(f"| {name} | {vals} |")
        add()

    # ── Convergence ──
    add("## Convergence Speed")
    add()
    conv = results.get("convergence", {})
    if conv:
        add("Best energy found at selected function evaluation counts:")
        add()
        # Find common evaluation milestones
        milestones = [50, 100, 200, 400, 800]
        header = "| Ansatz | " + " | ".join(f"@{m} evals" for m in milestones) + " |"
        sep = "|--------| " + " | ".join("------" for _ in milestones) + " |"
        add(header)
        add(sep)
        for name, cr in conv.items():
            trace = cr.get("convergence_trace", [])
            vals = []
            for m in milestones:
                # Find energy at or before this milestone
                e = None
                for n_eval, energy in trace:
                    if n_eval <= m:
                        e = energy
                vals.append(f"{e:.4f}" if e is not None else "N/A")
            add(f"| {name} | {' | '.join(vals)} |")
        add()

    # ── Gate Efficiency ──
    add("## Gate Efficiency")
    add()
    eff = results.get("gate_efficiency", {})
    if eff:
        add("| Ansatz | Error/Gate | Error/2q-Gate | Best Error (%) |")
        add("|--------|-----------|---------------|----------------|")
        for name, r in eff.items():
            eg = r.get("error_per_gate", 0)
            e2q = r.get("error_per_2q_gate", 0)
            ep = r.get("error_pct", 0)
            add(f"| {name} | {eg:.4f} | {e2q:.4f} | {ep:.1f}% |")
        add()

    # ── Conclusions ──
    add("## Conclusions")
    add()

    # Auto-generate conclusions from data
    if vqe and expr and grad:
        # Find best VQE ansatz on 4q Heisenberg
        heis_4q = vqe.get("4q_Heisenberg", {})
        if heis_4q:
            e_exact = heis_4q.get("exact_energy", 0)
            vqe_results = heis_4q.get("results", {})
            if vqe_results:
                best_ansatz = min(vqe_results.items(), key=lambda x: abs(x[1].get("best_energy", np.inf) - e_exact))
                best_err = abs(best_ansatz[1]["best_energy"] - e_exact) / abs(e_exact) * 100
                add(f"1. **Best VQE performer (4q Heisenberg)**: {best_ansatz[0]} "
                    f"with {best_err:.1f}% error")

        # Expressibility winner
        if expr:
            expr_best = min(expr.items(), key=lambda x: x[1].get("mean_fidelity", np.inf))
            add(f"2. **Most expressive**: {expr_best[0]} "
                f"(mean fidelity {expr_best[1]['mean_fidelity']:.4f})")

        # Trainability winner
        if grad:
            grad_best = max(grad.items(), key=lambda x: x[1].get("mean_gradient_variance", 0))
            add(f"3. **Most trainable**: {grad_best[0]} "
                f"(mean grad var {grad_best[1]['mean_gradient_variance']:.6f})")

        add()
        add("### ZXEA Assessment")
        add()

        zxea_4q = vqe.get("4q_Heisenberg", {}).get("results", {}).get("ZXEA", {})
        hea_4q = vqe.get("4q_Heisenberg", {}).get("results", {}).get("HEA", {})
        tvh_4q = vqe.get("4q_Heisenberg", {}).get("results", {}).get("TVH", {})

        if zxea_4q and hea_4q:
            e_exact = vqe["4q_Heisenberg"]["exact_energy"]
            zxea_err = abs(zxea_4q["best_energy"] - e_exact) / abs(e_exact) * 100
            hea_err = abs(hea_4q["best_energy"] - e_exact) / abs(e_exact) * 100
            tvh_err = abs(tvh_4q["best_energy"] - e_exact) / abs(e_exact) * 100 if tvh_4q else None

            add(f"- ZXEA VQE error: {zxea_err:.1f}% vs HEA: {hea_err:.1f}%"
                + (f" vs TVH: {tvh_err:.1f}%" if tvh_err is not None else ""))

        zxea_expr = expr.get("ZXEA", {})
        hea_expr = expr.get("HEA", {})
        if zxea_expr and hea_expr:
            add(f"- ZXEA expressibility: {zxea_expr.get('mean_fidelity', 0):.4f} "
                f"vs HEA: {hea_expr.get('mean_fidelity', 0):.4f} "
                f"(lower = more expressive)")

        zxea_grad = grad.get("ZXEA", {})
        hea_grad = grad.get("HEA", {})
        if zxea_grad and hea_grad:
            add(f"- ZXEA trainability: {zxea_grad.get('mean_gradient_variance', 0):.6f} "
                f"vs HEA: {hea_grad.get('mean_gradient_variance', 0):.6f}")

        add()
        add("### Hypothesis Verdict")
        add()
        add("The results above show whether ZX-irreducible entangling layers combined")
        add("with per-qubit parameters successfully bridge the expressibility-trainability")
        add("gap between TVH and HEA.")

    add()
    add("---")
    add("*Generated by benchmark_zxea.py*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


ANSATZE = {
    "ZXEA": {
        "make_fn": make_zxea,
        "n_params_fn": n_params_zxea,
        "description": "ZX-irreducible cluster_chain entangling",
    },
    "ZXEA-H": {
        "make_fn": make_zxea_h,
        "n_params_fn": n_params_zxea_h,
        "description": "ZXEA + hadamard_sandwich (H-S-H)",
    },
    "HEA": {
        "make_fn": make_hea,
        "n_params_fn": n_params_hea,
        "description": "Standard CX-chain entangling",
    },
    "QAOA-flex": {
        "make_fn": make_qaoa_flex,
        "n_params_fn": n_params_qaoa_flex,
        "description": "Per-layer gamma/beta ZZ+mixer",
    },
    "TVH": {
        "make_fn": make_tvh_original,
        "n_params_fn": n_params_tvh_original,
        "description": "Original 2-param global TVH",
    },
}


def main():
    t_start = time.time()
    results = {"ansatze": {}, "vqe": {}, "expressibility": {}, "gradient_variance": {},
               "noise": {}, "convergence": {}, "gate_efficiency": {}}

    # ── Setup ──
    print("=" * 70)
    print("ZXEA: ZX-Irreducible Entangling Ansatz — Benchmark")
    print("=" * 70)

    n_qubits_4 = 4
    n_layers = 2

    # Collect ansatz info
    for name, spec in ANSATZE.items():
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        gc = count_gates(spec["make_fn"], np_val, n_qubits_4)
        results["ansatze"][name] = {
            "n_params": np_val,
            "gate_counts": gc,
            "description": spec["description"],
        }
        print(f"  {name:12s}: {np_val:3d} params, {gc['total_gates']:3d} gates "
              f"({gc['two_qubit_gates']:2d} 2q), depth {gc['depth']}")

    # ── Benchmark 1: VQE 4-qubit Heisenberg ──
    print("\n[1/8] VQE: 4-qubit Heisenberg...")
    H_heis4, e_exact_heis4 = heisenberg_hamiltonian(4)
    print(f"  Exact energy: {e_exact_heis4:.4f}")
    results["vqe"]["4q_Heisenberg"] = {"exact_energy": e_exact_heis4, "results": {}}

    for name, spec in ANSATZE.items():
        t0 = time.time()
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        vqe_result = run_vqe(spec["make_fn"], np_val, H_heis4, n_qubits_4)
        elapsed = time.time() - t0
        err_pct = abs(vqe_result["best_energy"] - e_exact_heis4) / abs(e_exact_heis4) * 100
        print(f"  {name:12s}: E={vqe_result['best_energy']:.4f} "
              f"(err={err_pct:.1f}%) [{elapsed:.1f}s]")
        results["vqe"]["4q_Heisenberg"]["results"][name] = vqe_result

    # ── Benchmark 2: VQE 4-qubit TFIM ──
    print("\n[2/8] VQE: 4-qubit Transverse-Field Ising...")
    H_ising4, e_exact_ising4 = ising_hamiltonian(4, h=1.0)
    print(f"  Exact energy: {e_exact_ising4:.4f}")
    results["vqe"]["4q_TFIM"] = {"exact_energy": e_exact_ising4, "results": {}}

    for name, spec in ANSATZE.items():
        t0 = time.time()
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        vqe_result = run_vqe(spec["make_fn"], np_val, H_ising4, n_qubits_4)
        elapsed = time.time() - t0
        err_pct = abs(vqe_result["best_energy"] - e_exact_ising4) / abs(e_exact_ising4) * 100
        print(f"  {name:12s}: E={vqe_result['best_energy']:.4f} "
              f"(err={err_pct:.1f}%) [{elapsed:.1f}s]")
        results["vqe"]["4q_TFIM"]["results"][name] = vqe_result

    # ── Benchmark 3: VQE 6-qubit Heisenberg ──
    print("\n[3/8] VQE: 6-qubit Heisenberg...")
    H_heis6, e_exact_heis6 = heisenberg_hamiltonian(6)
    print(f"  Exact energy: {e_exact_heis6:.4f}")
    results["vqe"]["6q_Heisenberg"] = {"exact_energy": e_exact_heis6, "results": {}}

    n_qubits_6 = 6
    for name, spec in ANSATZE.items():
        t0 = time.time()
        np_val = spec["n_params_fn"](n_qubits_6, n_layers)
        vqe_result = run_vqe(spec["make_fn"], np_val, H_heis6, n_qubits_6,
                             n_restarts=20, maxiter=600)
        elapsed = time.time() - t0
        err_pct = abs(vqe_result["best_energy"] - e_exact_heis6) / abs(e_exact_heis6) * 100
        print(f"  {name:12s}: E={vqe_result['best_energy']:.4f} "
              f"(err={err_pct:.1f}%) [{elapsed:.1f}s]")
        results["vqe"]["6q_Heisenberg"]["results"][name] = vqe_result

    # ── Benchmark 4: Expressibility ──
    print("\n[4/8] Expressibility (4-qubit)...")
    for name, spec in ANSATZE.items():
        t0 = time.time()
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        expr_result = measure_expressibility(spec["make_fn"], np_val, n_qubits_4)
        elapsed = time.time() - t0
        print(f"  {name:12s}: mean_fid={expr_result['mean_fidelity']:.4f} "
              f"({expr_result['expressibility_ratio']:.2f}x Haar) [{elapsed:.1f}s]")
        results["expressibility"][name] = expr_result

    # ── Benchmark 5: Gradient Variance ──
    print("\n[5/8] Gradient variance (4-qubit Heisenberg)...")
    for name, spec in ANSATZE.items():
        t0 = time.time()
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        grad_result = measure_gradient_variance(spec["make_fn"], np_val, H_heis4, n_qubits_4)
        elapsed = time.time() - t0
        print(f"  {name:12s}: mean_var={grad_result['mean_gradient_variance']:.6f} [{elapsed:.1f}s]")
        results["gradient_variance"][name] = grad_result

    # ── Benchmark 6: Noise Resilience ──
    print("\n[6/8] Noise resilience...")
    noise_levels = [0.001, 0.005, 0.01]
    for name, spec in ANSATZE.items():
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        # Use best params from 4q Heisenberg VQE
        best_params = results["vqe"]["4q_Heisenberg"]["results"][name].get("best_params")
        if best_params is None:
            results["noise"][name] = {p: None for p in noise_levels}
            continue
        best_params = np.array(best_params)
        noise_results = {}
        for p in noise_levels:
            e_noisy = noisy_energy(spec["make_fn"], best_params, n_qubits_4, H_heis4, p)
            noise_results[p] = e_noisy
        results["noise"][name] = noise_results
        print(f"  {name:12s}: " + ", ".join(f"p={p}:{noise_results[p]:.4f}" for p in noise_levels))

    # ── Benchmark 7: Convergence Speed ──
    print("\n[7/8] Convergence tracking (4-qubit Heisenberg)...")
    for name, spec in ANSATZE.items():
        t0 = time.time()
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        conv_result = run_vqe_tracked(spec["make_fn"], np_val, H_heis4, n_qubits_4)
        elapsed = time.time() - t0
        n_trace = len(conv_result.get("convergence_trace", []))
        print(f"  {name:12s}: best={conv_result['best_energy']:.4f}, "
              f"{n_trace} trace points [{elapsed:.1f}s]")
        results["convergence"][name] = conv_result

    # ── Benchmark 8: Gate Efficiency ──
    print("\n[8/8] Gate efficiency...")
    for name, spec in ANSATZE.items():
        np_val = spec["n_params_fn"](n_qubits_4, n_layers)
        gc = results["ansatze"][name]["gate_counts"]
        best_e = results["vqe"]["4q_Heisenberg"]["results"][name]["best_energy"]
        err = abs(best_e - e_exact_heis4)
        err_pct = err / abs(e_exact_heis4) * 100
        total = gc["total_gates"]
        two_q = gc["two_qubit_gates"]
        results["gate_efficiency"][name] = {
            "error_pct": err_pct,
            "error_per_gate": err / total if total > 0 else 0,
            "error_per_2q_gate": err / two_q if two_q > 0 else 0,
            "total_gates": total,
            "two_qubit_gates": two_q,
        }
        print(f"  {name:12s}: {err_pct:.1f}% err, "
              f"{err/total:.4f}/gate, {err/two_q:.4f}/2q-gate" if two_q > 0 else
              f"  {name:12s}: {err_pct:.1f}% err, {err/total:.4f}/gate")

    # ── Generate outputs ──
    print("\nGenerating report...")
    report = generate_report(results)
    report_path = OUTPUT_DIR / "zxea_report.md"
    report_path.write_text(report)
    print(f"  Saved {report_path}")

    json_path = OUTPUT_DIR / "zxea_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved {json_path}")

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
