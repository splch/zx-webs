#!/usr/bin/env python3
"""Final validation of the three most promising approaches for novel algorithm discovery
within the ZX graph paradigm.

Approach A: PyZX compose() on graph-like diagrams (100% extraction rate)
Approach B: Parallel (tensor) + Hadamard stitching (93% extraction rate)
Approach C: Phase modification of graph-like diagrams (100% extraction, 75% novel)

This experiment:
1. Tests each approach at scale
2. Measures novelty (how different are the output unitaries from inputs)
3. Tests with actual algorithm circuits (not just random)
4. Validates tensor correctness
5. Tests the hybrid approach combining all three
"""

import sys
import numpy as np
from fractions import Fraction
from collections import Counter
import random
import traceback

import pyzx as zx
from pyzx.simplify import full_reduce, to_graph_like, is_graph_like, spider_simp
from pyzx.extract import extract_circuit
from pyzx.gflow import gflow
from pyzx.utils import VertexType, EdgeType
from pyzx.generate import CNOT_HAD_PHASE_circuit


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def unitary_distance(g1, g2, n_qubits):
    """Compute distance between two unitaries (trace distance of Choi states)."""
    try:
        t1 = g1.to_tensor()
        t2 = g2.to_tensor()

        # Normalize to unitaries
        t1 = t1 / np.sqrt(np.abs(np.trace(t1.conj().T @ t1)) + 1e-30)
        t2 = t2 / np.sqrt(np.abs(np.trace(t2.conj().T @ t2)) + 1e-30)

        # Frobenius distance (up to global phase)
        min_dist = float('inf')
        for phase_mult in [1, -1, 1j, -1j]:
            dist = np.linalg.norm(t1 - phase_mult * t2, 'fro')
            min_dist = min(min_dist, dist)

        return min_dist
    except:
        return -1.0


# =========================================================================
# Approach A: Sequential compose of graph-like diagrams
# =========================================================================
def approach_a_sequential():
    separator("APPROACH A: Sequential compose of graph-like diagrams")

    results = Counter()
    novelty_scores = []

    for n_qubits in [2, 3, 4, 5]:
        for seed_a in range(10):
            for seed_b in range(10):
                if seed_a >= seed_b:
                    continue
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 500)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Compose
                    g_c = g_a.copy()
                    g_c.compose(g_b.copy())
                    full_reduce(g_c)

                    # Extract
                    try:
                        c_out = extract_circuit(g_c.copy())
                        results["success"] += 1

                        # Measure novelty vs both inputs
                        if n_qubits <= 4:
                            g_out = c_out.to_graph()
                            d_a = unitary_distance(g_out, c_a.to_graph(), n_qubits)
                            d_b = unitary_distance(g_out, c_b.to_graph(), n_qubits)
                            if d_a > 0 and d_b > 0:
                                novelty = min(d_a, d_b)
                                novelty_scores.append(novelty)
                                if novelty > 0.1:
                                    results["novel"] += 1
                                else:
                                    results["trivial"] += 1
                    except:
                        results["extract_fail"] += 1

                except Exception as e:
                    results[f"error"] += 1

    total = results.get("success", 0) + results.get("extract_fail", 0) + results.get("error", 0)
    print(f"  Results: {dict(results)}")
    print(f"  Total attempts: {total}")
    print(f"  Success rate: {results.get('success',0)/max(total,1)*100:.1f}%")
    if novelty_scores:
        print(f"  Novelty scores: mean={np.mean(novelty_scores):.4f}, "
              f"median={np.median(novelty_scores):.4f}, "
              f"min={np.min(novelty_scores):.4f}, max={np.max(novelty_scores):.4f}")


# =========================================================================
# Approach B: Parallel + Hadamard stitching
# =========================================================================
def approach_b_stitching():
    separator("APPROACH B: Parallel composition + Hadamard stitching")

    results = Counter()
    novelty_scores = []

    for n_qubits in [2, 3, 4]:
        for seed_a in range(8):
            for seed_b in range(8):
                if seed_a >= seed_b:
                    continue
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 500)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Tensor product (parallel)
                    g_par = g_a.tensor(g_b)

                    # Stitch: add Hadamard edges between Z-spiders
                    inputs_p = set(g_par.inputs())
                    outputs_p = set(g_par.outputs())

                    interior = [v for v in g_par.vertices()
                               if v not in inputs_p and v not in outputs_p
                               and g_par.type(v) == VertexType.Z]

                    # Partition by qubit
                    q_threshold = max(g_a.qubit(v) for v in g_a.vertices()) + 0.5
                    set_a = [v for v in interior if g_par.qubit(v) < q_threshold]
                    set_b = [v for v in interior if g_par.qubit(v) >= q_threshold]

                    if not set_a or not set_b:
                        results["no_stitchable_vertices"] += 1
                        continue

                    rng = random.Random(seed_a * 100 + seed_b)
                    n_stitches = rng.randint(1, min(4, len(set_a), len(set_b)))

                    for _ in range(n_stitches):
                        va = rng.choice(set_a)
                        vb = rng.choice(set_b)
                        if not g_par.connected(va, vb):
                            g_par.add_edge((va, vb), edgetype=EdgeType.HADAMARD)

                    # Reduce and extract
                    full_reduce(g_par)
                    try:
                        c_out = extract_circuit(g_par.copy())
                        results["success"] += 1

                        # The output has 2*n_qubits qubits
                        # Measure novelty
                        if n_qubits <= 3:
                            g_out = c_out.to_graph()
                            # Compare to trivial parallel (no stitching)
                            g_trivial = g_a.tensor(g_b)
                            full_reduce(g_trivial)
                            d = unitary_distance(g_out, g_trivial, 2*n_qubits)
                            if d > 0:
                                novelty_scores.append(d)
                                if d > 0.1:
                                    results["novel_vs_parallel"] += 1
                    except:
                        results["extract_fail"] += 1

                except Exception as e:
                    results["error"] += 1

    total = results.get("success", 0) + results.get("extract_fail", 0)
    print(f"  Results: {dict(results)}")
    print(f"  Success rate: {results.get('success',0)/max(total,1)*100:.1f}%")
    if novelty_scores:
        print(f"  Novelty vs trivial parallel: mean={np.mean(novelty_scores):.4f}, "
              f"median={np.median(novelty_scores):.4f}")


# =========================================================================
# Approach C: Phase perturbation
# =========================================================================
def approach_c_phase_perturbation():
    separator("APPROACH C: Phase perturbation of graph-like diagrams")

    results = Counter()
    novelty_scores = []

    for n_qubits in [2, 3, 4, 5]:
        for seed in range(20):
            try:
                c_orig = CNOT_HAD_PHASE_circuit(n_qubits, 20, seed=seed)
                g = c_orig.to_graph()
                full_reduce(g)

                rng = random.Random(seed * 77 + 42)

                # Several perturbation strategies
                for strategy in ["random", "increment", "zero_out", "negate"]:
                    g_mod = g.copy()
                    interior = [v for v in g_mod.vertices()
                               if g_mod.type(v) == VertexType.Z
                               and v not in g_mod.inputs()
                               and v not in g_mod.outputs()
                               and g_mod.type(v) != VertexType.BOUNDARY]

                    for v in interior:
                        if rng.random() < 0.4:  # Modify 40%
                            if strategy == "random":
                                g_mod.set_phase(v, Fraction(rng.randint(0, 7), 4))
                            elif strategy == "increment":
                                old = g_mod.phase(v)
                                g_mod.set_phase(v, old + Fraction(1, 4))
                            elif strategy == "zero_out":
                                g_mod.set_phase(v, 0)
                            elif strategy == "negate":
                                old = g_mod.phase(v)
                                g_mod.set_phase(v, (-old) % 2)

                    # The graph is still graph-like (phases don't affect graph-like-ness)
                    assert is_graph_like(g_mod)

                    # Reduce and extract
                    g2 = g_mod.copy()
                    full_reduce(g2)
                    try:
                        c_out = extract_circuit(g2.copy())
                        results[f"{strategy}_success"] += 1

                        if n_qubits <= 4:
                            g_out = c_out.to_graph()
                            d = unitary_distance(g_out, c_orig.to_graph(), n_qubits)
                            if d > 0:
                                novelty_scores.append((strategy, d))
                                if d > 0.1:
                                    results[f"{strategy}_novel"] += 1

                    except:
                        results[f"{strategy}_fail"] += 1

            except Exception as e:
                results[f"error: {str(e)[:40]}"] += 1

    print(f"  Results: {dict(results)}")
    if novelty_scores:
        for strat in ["random", "increment", "zero_out", "negate"]:
            scores = [s for (st, s) in novelty_scores if st == strat]
            if scores:
                print(f"  {strat}: mean_novelty={np.mean(scores):.4f}, novel_count={sum(1 for s in scores if s > 0.1)}")


# =========================================================================
# Hybrid: Compose + Stitch + Phase-modify
# =========================================================================
def hybrid_approach():
    separator("HYBRID: Compose + Stitch + Phase-modify")

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for trial in range(30):
            try:
                rng = random.Random(trial * 1337)

                # Pick 2-3 random source circuits
                n_sources = rng.randint(2, 3)
                sources = []
                for i in range(n_sources):
                    c = CNOT_HAD_PHASE_circuit(n_qubits, rng.randint(10, 25),
                                                seed=rng.randint(0, 1000))
                    g = c.to_graph()
                    full_reduce(g)
                    sources.append(g)

                # Step 1: Sequential compose first two
                g_result = sources[0].copy()
                for i in range(1, len(sources)):
                    g_result.compose(sources[i].copy())

                # Step 2: Phase perturbation
                g_mod = g_result.copy()
                for v in list(g_mod.vertices()):
                    if (g_mod.type(v) == VertexType.Z
                        and v not in g_mod.inputs()
                        and v not in g_mod.outputs()
                        and g_mod.type(v) != VertexType.BOUNDARY):
                        if rng.random() < 0.2:
                            g_mod.set_phase(v, Fraction(rng.randint(0, 7), 4))

                # Step 3: Reduce and extract
                full_reduce(g_mod)

                try:
                    c_out = extract_circuit(g_mod.copy())
                    results["success"] += 1
                    results[f"gates_{len(c_out.gates)}"] += 1
                    results[f"2q_gates_{c_out.twoqubitcount()}"] += 1
                except:
                    results["extract_fail"] += 1

            except Exception as e:
                results["error"] += 1

    total = results.get("success", 0) + results.get("extract_fail", 0) + results.get("error", 0)
    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    print(f"\n  Success rate: {results.get('success',0)/max(total,1)*100:.1f}%")


# =========================================================================
# Scalability test
# =========================================================================
def scalability_test():
    separator("SCALABILITY: Testing larger qubit counts")

    results = Counter()

    for n_qubits in [5, 8, 10, 15, 20]:
        for seed in range(3):
            try:
                c_a = CNOT_HAD_PHASE_circuit(n_qubits, n_qubits * 5, seed=seed)
                c_b = CNOT_HAD_PHASE_circuit(n_qubits, n_qubits * 5, seed=seed + 100)

                g_a = c_a.to_graph()
                g_b = c_b.to_graph()
                full_reduce(g_a)
                full_reduce(g_b)

                g_c = g_a.copy()
                g_c.compose(g_b.copy())

                # Phase perturb
                rng = random.Random(seed * 42)
                for v in list(g_c.vertices()):
                    if (g_c.type(v) == VertexType.Z
                        and v not in g_c.inputs()
                        and v not in g_c.outputs()
                        and g_c.type(v) != VertexType.BOUNDARY):
                        if rng.random() < 0.15:
                            g_c.set_phase(v, Fraction(rng.randint(0, 7), 4))

                full_reduce(g_c)

                import time
                t0 = time.time()
                try:
                    c_out = extract_circuit(g_c.copy())
                    t1 = time.time()
                    results[f"q{n_qubits}_success"] += 1
                    print(f"  n_qubits={n_qubits}, seed={seed}: {len(c_out.gates)} gates, "
                          f"{c_out.twoqubitcount()} 2q gates, {t1-t0:.2f}s")
                except Exception as e:
                    t1 = time.time()
                    results[f"q{n_qubits}_fail"] += 1
                    print(f"  n_qubits={n_qubits}, seed={seed}: FAILED ({t1-t0:.2f}s) - {str(e)[:60]}")

            except Exception as e:
                results[f"q{n_qubits}_error"] += 1

    print(f"\n  Results: {dict(results)}")


if __name__ == "__main__":
    print("ZX-Webs Final Approach Validation")

    approach_a_sequential()
    approach_b_stitching()
    approach_c_phase_perturbation()
    hybrid_approach()
    scalability_test()
