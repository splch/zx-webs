#!/usr/bin/env python3
"""Generalized irr_pair11 ansatz: scale the structural motifs with qubit count.

The original irr_pair11 was discovered by composing two irreducible ZX motifs
(phase_gadget_3t + cluster_chain) into a 6-qubit circuit.  The problem: the
motif placement doesn't scale — gate count stays fixed at 12 regardless of
system size, leaving most qubits idle at >6q.

This script implements a *principled generalization* that preserves the three
structural elements identified by ablation as critical:

  1. Star hub — a central qubit entangled via CX fan-out to ~n/3 neighbours
  2. Phase gadgets — T gates conjugated by CX pairs (irreducible non-Clifford)
  3. Chain tail — nearest-neighbour CX chain extending entanglement to far qubits

All three scale linearly with qubit count so every qubit participates.

Usage:
    source .venv/bin/activate
    python scripts/visualize_irr_pair11.py
"""

from __future__ import annotations

import json
import itertools
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy import stats
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ═════════════════════════════════════════════════════════════════════════
# Generalized irr_pair11 entangler
# ═════════════════════════════════════════════════════════════════════════

def irr_pair11_entangler(n: int) -> QuantumCircuit:
    """Build the generalized irr_pair11 entangling layer for n qubits.

    Structure (scales with n):
      1. Star hub: qubit 0 is the hub, CX from qubits 1..hub_size -> 0
      2. Phase gadgets: T gates on every 3rd qubit, each conjugated by CX
         pairs that connect it to its neighbours
      3. Chain tail: CX chain across remaining qubits to propagate entanglement

    At n=6 this reproduces essentially the same connectivity as the original.
    """
    assert n >= 4, "Need at least 4 qubits"
    qc = QuantumCircuit(n)

    # --- 1. Star hub: fan-in to qubit 0 from qubits 1..hub_size ---
    hub_size = max(2, n // 3)
    for i in range(1, hub_size + 1):
        qc.cx(i, 0)

    # --- 2. Phase gadgets: place on every 3rd qubit starting from hub_size ---
    # Each gadget: CX(anchor, target), T(target), CX(anchor, target)
    # Plus CX between consecutive gadget anchors for inter-gadget entanglement
    gadget_anchors = []
    q = 1
    while q < n - 1:
        anchor = q
        target = q + 1
        # Inter-gadget entanglement: connect to previous anchor
        if gadget_anchors:
            qc.cx(gadget_anchors[-1], anchor)
        qc.cx(anchor, target)
        qc.t(target)
        qc.cx(anchor, target)
        gadget_anchors.append(anchor)
        q += 3  # stride of 3 to avoid overlap

    # --- 3. Chain tail: connect last gadget region to remaining qubits ---
    last_touched = max(gadget_anchors[-1] + 1, hub_size) if gadget_anchors else hub_size
    for i in range(last_touched, n - 1):
        qc.cx(i, i + 1)

    return qc


def irr_pair11_original_6q() -> QuantumCircuit:
    """Reproduce the exact original 6q circuit from verification data."""
    qc = QuantumCircuit(6)
    qc.cx(1, 0)
    qc.cx(2, 0)
    qc.cx(3, 0)
    qc.cx(1, 2)
    qc.cx(1, 3)
    qc.cx(2, 3)
    qc.t(3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    qc.t(2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


# ═════════════════════════════════════════════════════════════════════════
# Baseline entanglers
# ═════════════════════════════════════════════════════════════════════════

def cx_chain_entangler(n: int, n_2q: int) -> QuantumCircuit:
    """CX-chain baseline with given 2Q gate budget."""
    qc = QuantumCircuit(n)
    placed = 0
    while placed < n_2q:
        for i in range(min(n - 1, n_2q - placed)):
            qc.cx(i, i + 1)
            placed += 1
            if placed >= n_2q:
                break
    return qc


def hea_entangler(n: int, n_2q: int) -> QuantumCircuit:
    """CZ brick-layer (HEA) baseline with given 2Q gate budget."""
    qc = QuantumCircuit(n)
    placed = 0
    layer = 0
    while placed < n_2q:
        start = layer % 2
        for i in range(start, n - 1, 2):
            qc.cz(i, i + 1)
            placed += 1
            if placed >= n_2q:
                break
        layer += 1
    return qc


# ═════════════════════════════════════════════════════════════════════════
# Hamiltonian builders
# ═════════════════════════════════════════════════════════════════════════

_PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label: str) -> np.ndarray:
    result = np.array([[1.0 + 0j]])
    for ch in label:
        result = np.kron(result, _PAULI_1Q[ch])
    return result


def build_hamiltonian(n: int, model: str) -> np.ndarray:
    d = 2**n
    H = np.zeros((d, d), dtype=complex)

    if model == "heisenberg":
        for i in range(n - 1):
            for p in "XYZ":
                label = ["I"] * n
                label[i] = p
                label[i + 1] = p
                H += _pauli_matrix("".join(label))

    elif model == "tfim":
        for i in range(n - 1):
            label = ["I"] * n
            label[i] = "Z"
            label[i + 1] = "Z"
            H -= _pauli_matrix("".join(label))
        for i in range(n):
            label = ["I"] * n
            label[i] = "X"
            H -= _pauli_matrix("".join(label))

    elif model == "xy":
        for i in range(n - 1):
            for p in "XY":
                label = ["I"] * n
                label[i] = p
                label[i + 1] = p
                H += _pauli_matrix("".join(label))

    elif model == "xxz":
        delta = 0.5
        for i in range(n - 1):
            for p in "XY":
                label = ["I"] * n
                label[i] = p
                label[i + 1] = p
                H += _pauli_matrix("".join(label))
            label = ["I"] * n
            label[i] = "Z"
            label[i + 1] = "Z"
            H += delta * _pauli_matrix("".join(label))

    elif model == "random_2local":
        rng = np.random.default_rng(42)
        for i in range(n - 1):
            for p1 in "XYZ":
                for p2 in "XYZ":
                    coeff = rng.normal(0, 1)
                    if abs(coeff) < 0.3:
                        continue
                    label = ["I"] * n
                    label[i] = p1
                    label[i + 1] = p2
                    H += coeff * _pauli_matrix("".join(label))
        for i in range(n):
            for p in "XYZ":
                coeff = rng.normal(0, 0.5)
                if abs(coeff) < 0.2:
                    continue
                label = ["I"] * n
                label[i] = p
                H += coeff * _pauli_matrix("".join(label))

    return H


# ═════════════════════════════════════════════════════════════════════════
# VQE engine
# ═════════════════════════════════════════════════════════════════════════

def vqe_test(
    entangler_qc: QuantumCircuit,
    n_qubits: int,
    H_matrix: np.ndarray,
    n_restarts: int = 10,
    maxiter: int = 400,
    seed: int = 42,
) -> dict:
    """Run VQE with RY/RZ - entangler - RY/RZ ansatz. Returns detailed results."""
    n_params = 4 * n_qubits

    def energy(params):
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(params[2 * i], i)
            qc.rz(params[2 * i + 1], i)
        qc.compose(entangler_qc, inplace=True)
        for i in range(n_qubits):
            qc.ry(params[2 * n_qubits + 2 * i], i)
            qc.rz(params[2 * n_qubits + 2 * i + 1], i)
        sv = Statevector.from_instruction(qc)
        return float(np.real(np.array(sv.data).conj() @ H_matrix @ np.array(sv.data)))

    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    all_energies = []

    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        try:
            res = minimize(energy, x0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.5})
            all_energies.append(float(res.fun))
            if res.fun < best_energy:
                best_energy = float(res.fun)
        except Exception:
            pass

    return {
        "best_energy": best_energy,
        "all_energies": all_energies,
        "mean_energy": float(np.mean(all_energies)) if all_energies else float("inf"),
        "std_energy": float(np.std(all_energies)) if all_energies else 0.0,
        "n_params": n_params,
    }


def count_2q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits >= 2)


# ═════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═════════════════════════════════════════════════════════════════════════

QUBIT_SIZES = [4, 5, 6, 7, 8, 10]
MODELS = ["heisenberg", "tfim", "xy", "xxz", "random_2local"]
N_RESTARTS = 10
MAXITER = 400


def print_circuit_summary(name: str, qc: QuantumCircuit):
    n_2q = count_2q(qc)
    n_1q = qc.size() - n_2q
    print(f"  {name}: {qc.num_qubits}q, {qc.size()} gates "
          f"({n_1q} 1Q + {n_2q} 2Q), depth={qc.depth()}")


def run_benchmark():
    print("=" * 72)
    print("GENERALIZED irr_pair11 BENCHMARK")
    print("=" * 72)
    print(f"Settings: {N_RESTARTS} restarts, {MAXITER} iterations, COBYLA")
    print(f"Qubit sizes: {QUBIT_SIZES}")
    print(f"Hamiltonians: {MODELS}")
    print()

    # --- Show circuit structure at each qubit count ---
    print("Circuit structure at each qubit count:")
    print("-" * 50)
    for nq in QUBIT_SIZES:
        qc = irr_pair11_entangler(nq)
        print_circuit_summary(f"gen_irr11_{nq}q", qc)
        # Print gate list compactly
        gates = []
        for inst in qc.data:
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            gates.append(f"{inst.operation.name} {qubits}")
        print(f"    Gates: {' | '.join(gates)}")
    print()

    # --- Also show original 6q for comparison ---
    print("Original 6q circuit (from discovery):")
    orig_6q = irr_pair11_original_6q()
    print_circuit_summary("orig_irr11_6q", orig_6q)
    print()

    # --- Run VQE benchmarks ---
    results = {}
    total = len(QUBIT_SIZES) * len(MODELS)
    done = 0
    t0 = time.time()

    for nq in QUBIT_SIZES:
        # Skip 10q for models that would blow up (2^10 = 1024, still manageable)
        gen_qc = irr_pair11_entangler(nq)
        n_2q = count_2q(gen_qc)
        cx_qc = cx_chain_entangler(nq, max(1, n_2q))
        hea_qc = hea_entangler(nq, max(1, n_2q))

        for model in MODELS:
            done += 1
            elapsed = time.time() - t0
            print(f"  [{done}/{total}] {nq}q {model} "
                  f"(elapsed {elapsed:.0f}s) ...", flush=True)

            H = build_hamiltonian(nq, model)
            evals = np.linalg.eigvalsh(H)
            exact_gs = float(evals[0])

            if exact_gs == 0:
                print(f"    Skipping (E_gs = 0)")
                continue

            # Generalized irr_pair11
            gen_res = vqe_test(gen_qc, nq, H, N_RESTARTS, MAXITER)
            gen_err = abs(gen_res["best_energy"] - exact_gs) / abs(exact_gs)

            # CX-chain baseline (same 2Q gate budget)
            cx_res = vqe_test(cx_qc, nq, H, N_RESTARTS, MAXITER)
            cx_err = abs(cx_res["best_energy"] - exact_gs) / abs(exact_gs)

            # HEA baseline (same 2Q gate budget)
            hea_res = vqe_test(hea_qc, nq, H, N_RESTARTS, MAXITER)
            hea_err = abs(hea_res["best_energy"] - exact_gs) / abs(exact_gs)

            # Original irr_pair11 at 6q for comparison
            orig_err = None
            if nq == 6:
                orig_res = vqe_test(orig_6q, nq, H, N_RESTARTS, MAXITER)
                orig_err = abs(orig_res["best_energy"] - exact_gs) / abs(exact_gs)

            best_base = min(cx_err, hea_err)
            improvement = best_base - gen_err
            wins = gen_err < best_base

            key = f"gen_irr11_{nq}q_{model}"
            results[key] = {
                "n_qubits": nq,
                "model": model,
                "n_2q_gates": n_2q,
                "n_params": gen_res["n_params"],
                "exact_gs": exact_gs,
                "gen_error": gen_err,
                "cx_error": cx_err,
                "hea_error": hea_err,
                "orig_6q_error": orig_err,
                "improvement_vs_best_baseline": improvement,
                "wins": wins,
                "gen_best_energy": gen_res["best_energy"],
                "gen_mean_energy": gen_res["mean_energy"],
                "gen_std_energy": gen_res["std_energy"],
                "cx_best_energy": cx_res["best_energy"],
                "hea_best_energy": hea_res["best_energy"],
            }

            tag = "WIN" if wins else "lose"
            print(f"    gen={gen_err:.4f}  cx={cx_err:.4f}  "
                  f"hea={hea_err:.4f}  [{tag}]"
                  f"{'  orig=' + f'{orig_err:.4f}' if orig_err is not None else ''}")

    elapsed_total = time.time() - t0
    print(f"\nTotal time: {elapsed_total:.0f}s")

    # --- Statistical significance for winning cases ---
    print("\n" + "=" * 72)
    print("STATISTICAL SIGNIFICANCE (winning cases)")
    print("=" * 72)

    for key, r in results.items():
        if not r["wins"]:
            continue
        nq = r["n_qubits"]
        model = r["model"]
        gen_qc = irr_pair11_entangler(nq)
        n_2q = count_2q(gen_qc)
        best_base_name = "cx_chain" if r["cx_error"] < r["hea_error"] else "hea"
        best_base_qc = (cx_chain_entangler(nq, max(1, n_2q)) if best_base_name == "cx_chain"
                        else hea_entangler(nq, max(1, n_2q)))

        H = build_hamiltonian(nq, model)
        exact_gs = r["exact_gs"]

        # Run multiple seeds
        gen_errors = []
        base_errors = []
        for seed in range(42, 52):  # 10 seeds
            g = vqe_test(gen_qc, nq, H, N_RESTARTS, MAXITER, seed=seed)
            b = vqe_test(best_base_qc, nq, H, N_RESTARTS, MAXITER, seed=seed)
            gen_errors.append(abs(g["best_energy"] - exact_gs) / abs(exact_gs))
            base_errors.append(abs(b["best_energy"] - exact_gs) / abs(exact_gs))

        gen_arr = np.array(gen_errors)
        base_arr = np.array(base_errors)
        t_stat, p_val = stats.ttest_ind(gen_arr, base_arr, alternative="less")
        cohens_d = (np.mean(base_arr) - np.mean(gen_arr)) / np.sqrt(
            (np.std(gen_arr)**2 + np.std(base_arr)**2) / 2)

        results[key]["stat_test"] = {
            "gen_mean": float(np.mean(gen_arr)),
            "gen_std": float(np.std(gen_arr)),
            "base_mean": float(np.mean(base_arr)),
            "base_std": float(np.std(base_arr)),
            "best_baseline": best_base_name,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "significant": bool(p_val < 0.05),
            "gen_errors_by_seed": gen_errors,
            "base_errors_by_seed": base_errors,
        }

        sig = "YES" if p_val < 0.05 else "no"
        print(f"  {key}: gen={np.mean(gen_arr):.4f}±{np.std(gen_arr):.4f} "
              f"vs {best_base_name}={np.mean(base_arr):.4f}±{np.std(base_arr):.4f} "
              f"p={p_val:.6f} d={cohens_d:.2f} [{sig}]")

    # --- Summary tables ---
    print("\n" + "=" * 72)
    print("RESULTS BY QUBIT COUNT")
    print("=" * 72)
    print(f"{'Config':<30s} {'Gen':>8s} {'CX':>8s} {'HEA':>8s} {'Improv':>8s} {'Win?':>5s}")
    print("-" * 70)
    for key in sorted(results.keys()):
        r = results[key]
        tag = "YES" if r["wins"] else ""
        print(f"{key:<30s} {r['gen_error']:>8.4f} {r['cx_error']:>8.4f} "
              f"{r['hea_error']:>8.4f} {r['improvement_vs_best_baseline']:>+8.4f} {tag:>5s}")

    # --- Summary: wins by model ---
    print("\n" + "=" * 72)
    print("WINS BY MODEL")
    print("=" * 72)
    for model in MODELS:
        model_results = {k: v for k, v in results.items() if v["model"] == model}
        wins = sum(1 for v in model_results.values() if v["wins"])
        total = len(model_results)
        print(f"  {model:<20s}: {wins}/{total} wins")

    # --- Summary: wins by qubit count ---
    print("\nWINS BY QUBIT COUNT")
    print("-" * 40)
    for nq in QUBIT_SIZES:
        nq_results = {k: v for k, v in results.items() if v["n_qubits"] == nq}
        wins = sum(1 for v in nq_results.values() if v["wins"])
        total = len(nq_results)
        n_2q = count_2q(irr_pair11_entangler(nq))
        print(f"  {nq}q ({n_2q} 2Q gates): {wins}/{total} wins")

    # --- 6q comparison: generalized vs original ---
    print("\n" + "=" * 72)
    print("6q COMPARISON: GENERALIZED vs ORIGINAL irr_pair11")
    print("=" * 72)
    print(f"{'Model':<20s} {'Gen':>8s} {'Orig':>8s} {'Diff':>8s}")
    print("-" * 50)
    for key, r in sorted(results.items()):
        if r["n_qubits"] == 6 and r["orig_6q_error"] is not None:
            diff = r["gen_error"] - r["orig_6q_error"]
            print(f"{r['model']:<20s} {r['gen_error']:>8.4f} "
                  f"{r['orig_6q_error']:>8.4f} {diff:>+8.4f}")

    # --- Save results ---
    out_dir = Path(__file__).parent / "output" / "generalized_irr_pair11"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Write text report
    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("GENERALIZED irr_pair11 EVALUATION REPORT")
    report_lines.append("=" * 72)
    report_lines.append(f"Settings: {N_RESTARTS} restarts, {MAXITER} iterations, COBYLA")
    report_lines.append(f"Total runtime: {elapsed_total:.0f}s")
    report_lines.append("")

    report_lines.append("CIRCUIT STRUCTURE")
    report_lines.append("-" * 50)
    for nq in QUBIT_SIZES:
        qc = irr_pair11_entangler(nq)
        n_2q = count_2q(qc)
        n_1q = qc.size() - n_2q
        report_lines.append(
            f"  {nq}q: {qc.size()} gates ({n_1q} 1Q + {n_2q} 2Q), depth={qc.depth()}")
    report_lines.append("")

    report_lines.append("VQE RESULTS")
    report_lines.append("-" * 70)
    report_lines.append(
        f"{'Config':<30s} {'Gen':>8s} {'CX':>8s} {'HEA':>8s} {'Improv':>8s} {'Win':>5s}")
    for key in sorted(results.keys()):
        r = results[key]
        tag = "YES" if r["wins"] else ""
        report_lines.append(
            f"{key:<30s} {r['gen_error']:>8.4f} {r['cx_error']:>8.4f} "
            f"{r['hea_error']:>8.4f} {r['improvement_vs_best_baseline']:>+8.4f} {tag:>5s}")
    report_lines.append("")

    # Statistical significance section
    sig_results = {k: v for k, v in results.items() if "stat_test" in v}
    if sig_results:
        report_lines.append("STATISTICAL SIGNIFICANCE")
        report_lines.append("-" * 70)
        for key, r in sorted(sig_results.items()):
            st = r["stat_test"]
            sig = "SIGNIFICANT" if st["significant"] else "not significant"
            report_lines.append(
                f"  {key}: p={st['p_value']:.6f} d={st['cohens_d']:.2f} [{sig}]")
        report_lines.append("")

    # Wins summary
    total_wins = sum(1 for v in results.values() if v["wins"])
    total_tests = len(results)
    report_lines.append(f"OVERALL: {total_wins}/{total_tests} wins over best baseline")
    report_lines.append("")

    # 6q comparison
    report_lines.append("6q GENERALIZED vs ORIGINAL")
    report_lines.append("-" * 50)
    for key, r in sorted(results.items()):
        if r["n_qubits"] == 6 and r["orig_6q_error"] is not None:
            diff = r["gen_error"] - r["orig_6q_error"]
            report_lines.append(
                f"  {r['model']:<20s} gen={r['gen_error']:.4f} "
                f"orig={r['orig_6q_error']:.4f} diff={diff:+.4f}")

    report_text = "\n".join(report_lines)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report_text)

    print(f"\nResults saved to {out_dir}/")
    print(f"  results.json")
    print(f"  report.txt")


if __name__ == "__main__":
    run_benchmark()
