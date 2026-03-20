#!/usr/bin/env python
"""Diagnostic experiments: why does circuit extraction fail on composed ZX-diagrams?"""
import logging
import pyzx as zx
from fractions import Fraction
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from zx_webs.config import load_config
from zx_webs.persistence import load_manifest
from zx_webs.stage1_corpus.algorithms import ALGORITHM_REGISTRY
from zx_webs.stage1_corpus.qasm_bridge import circuit_to_pyzx_qasm
from zx_webs.stage3_mining.zx_web import ZXWeb
from zx_webs.pipeline import Pipeline


def ensure_stages_1_3():
    data_dir = Path("data")
    if not (data_dir / "mined_webs" / "manifest.json").exists():
        config = load_config("configs/tiny_run.yaml")
        pipeline = Pipeline(config)
        pipeline.run(start_stage="corpus", end_stage="mining")


def experiment_1():
    print("=== EXPERIMENT 1: Inspect mined webs ===")
    ensure_stages_1_3()
    webs_manifest = load_manifest(Path("data/mined_webs"))
    print(f"Total mined webs: {len(webs_manifest)}")

    webs = [ZXWeb.from_dict(item) for item in webs_manifest[:50]]
    sizes = {}
    for w in webs:
        sizes[w.n_spiders] = sizes.get(w.n_spiders, 0) + 1
    print(f"Web sizes (n_spiders: count): {dict(sorted(sizes.items()))}")

    for w in webs[:5]:
        print(f"  {w.web_id}: {w.n_spiders} spiders, {len(w.boundary_wires)} boundary wires, "
              f"inputs={w.n_inputs}, outputs={w.n_outputs}")
    print()


def experiment_2():
    print("=== EXPERIMENT 2: Extraction on single algorithms (unreduced vs reduced) ===")
    algos = ["entanglement/ghz", "oracular/deutsch_jozsa", "oracular/bernstein_vazirani",
             "arithmetic/qft", "entanglement/w_state"]

    for algo_key in algos:
        fn = ALGORITHM_REGISTRY[algo_key]
        qc = fn(3)
        qasm = circuit_to_pyzx_qasm(qc)
        c = zx.Circuit.from_qasm(qasm)
        g = c.to_graph()

        for method_name, reduce_fn in [
            ("unreduced", lambda g: g),
            ("teleport_reduce", lambda g: (zx.teleport_reduce(g), g)[-1]),
            ("full_reduce", lambda g: (zx.full_reduce(g), g)[-1]),
        ]:
            try:
                g_copy = g.copy()
                reduce_fn(g_copy)
                extracted = zx.extract_circuit(g_copy.copy())
                print(f"  {algo_key} ({method_name}): OK - {extracted.stats()}")
            except Exception as e:
                print(f"  {algo_key} ({method_name}): FAIL - {type(e).__name__}: {str(e)[:100]}")
        print()


def experiment_3():
    print("=== EXPERIMENT 3: Compose ZX diagrams via PyZX compose() ===")
    pairs = [
        ("entanglement/ghz", "entanglement/ghz"),
        ("entanglement/ghz", "entanglement/w_state"),
        ("oracular/deutsch_jozsa", "entanglement/ghz"),
        ("arithmetic/qft", "arithmetic/qft"),
    ]

    for a_key, b_key in pairs:
        fn_a = ALGORITHM_REGISTRY[a_key]
        fn_b = ALGORITHM_REGISTRY[b_key]
        qc_a, qc_b = fn_a(3), fn_b(3)
        qasm_a = circuit_to_pyzx_qasm(qc_a)
        qasm_b = circuit_to_pyzx_qasm(qc_b)
        c_a = zx.Circuit.from_qasm(qasm_a)
        c_b = zx.Circuit.from_qasm(qasm_b)
        g_a = c_a.to_graph()
        g_b = c_b.to_graph()

        try:
            g_combined = g_a.copy()
            g_combined.compose(g_b.copy())
        except Exception as e:
            print(f"  {a_key} + {b_key}: COMPOSE FAIL - {type(e).__name__}: {str(e)[:100]}")
            print()
            continue

        # Try extract without reduction
        try:
            extracted = zx.extract_circuit(g_combined.copy())
            print(f"  {a_key} + {b_key} (no reduce): OK - {extracted.stats()}")
        except Exception as e:
            print(f"  {a_key} + {b_key} (no reduce): FAIL - {type(e).__name__}: {str(e)[:100]}")

        # Try full_reduce then extract
        try:
            g_reduced = g_combined.copy()
            zx.full_reduce(g_reduced)
            extracted = zx.extract_circuit(g_reduced.copy())
            print(f"  {a_key} + {b_key} (full_reduce): OK - {extracted.stats()}")
        except Exception as e:
            print(f"  {a_key} + {b_key} (full_reduce): FAIL - {type(e).__name__}: {str(e)[:100]}")
        print()


def experiment_4():
    print("=== EXPERIMENT 4: Compose at Qiskit circuit level, then ZX-optimize ===")
    # Only pair algorithms with matching qubit counts
    pairs = [
        ("entanglement/ghz", "entanglement/w_state"),
        ("entanglement/ghz", "arithmetic/qft"),
        ("arithmetic/qft", "entanglement/ghz"),
        ("entanglement/w_state", "entanglement/ghz"),
    ]

    for a_key, b_key in pairs:
        fn_a = ALGORITHM_REGISTRY[a_key]
        fn_b = ALGORITHM_REGISTRY[b_key]
        qc_a, qc_b = fn_a(3), fn_b(3)

        # Match qubit counts by padding
        n = max(qc_a.num_qubits, qc_b.num_qubits)
        from qiskit import QuantumCircuit
        combined = QuantumCircuit(n)
        combined.compose(qc_a, qubits=list(range(qc_a.num_qubits)), inplace=True)
        combined.compose(qc_b, qubits=list(range(qc_b.num_qubits)), inplace=True)
        qasm = circuit_to_pyzx_qasm(combined)
        c = zx.Circuit.from_qasm(qasm)
        g = c.to_graph()

        orig_stats = c.stats()

        try:
            g_copy = g.copy()
            zx.full_reduce(g_copy)
            extracted = zx.extract_circuit(g_copy.copy())
            print(f"  {a_key} >> {b_key}: OK")
            print(f"    Original:  {orig_stats}")
            print(f"    Optimized: {extracted.stats()}")
        except Exception as e:
            print(f"  {a_key} >> {b_key}: FAIL - {type(e).__name__}: {str(e)[:100]}")
        print()


def experiment_5():
    print("=== EXPERIMENT 5: Apply random ZX rewrites to existing circuits ===")
    algos = ["entanglement/ghz", "oracular/deutsch_jozsa", "arithmetic/qft"]
    import random
    rng = random.Random(42)

    for algo_key in algos:
        fn = ALGORITHM_REGISTRY[algo_key]
        qc = fn(3)
        qasm = circuit_to_pyzx_qasm(qc)
        c = zx.Circuit.from_qasm(qasm)
        g = c.to_graph()
        orig_stats = c.stats()

        successes = 0
        improvements = 0
        for trial in range(20):
            g_copy = g.copy()

            # Apply a random sequence of simplification steps
            steps = rng.sample([
                "spider_simp", "id_simp", "pivot_simp", "lcomp_simp",
                "bialg_simp", "gadget_simp",
            ], k=rng.randint(1, 4))

            for step in steps:
                try:
                    getattr(zx.simplify, step)(g_copy)
                except Exception:
                    pass

            try:
                extracted = zx.extract_circuit(g_copy.copy())
                successes += 1
                # Check if it's an improvement
                new_stats = extracted.stats()
                # Simple check: fewer 2-qubit gates
            except Exception:
                pass

        print(f"  {algo_key}: {successes}/20 random rewrite sequences extractable")
    print()


def experiment_6():
    print("=== EXPERIMENT 6: Phase gadget insertion/removal ===")
    print("Testing if modifying phases on existing circuits yields extractable variants...")

    algos = ["entanglement/ghz", "oracular/deutsch_jozsa", "arithmetic/qft"]
    import random
    rng = random.Random(42)

    for algo_key in algos:
        fn = ALGORITHM_REGISTRY[algo_key]
        qc = fn(3)
        qasm = circuit_to_pyzx_qasm(qc)
        c = zx.Circuit.from_qasm(qasm)
        g = c.to_graph()
        orig_stats = c.stats()

        successes = 0
        novel = 0
        for trial in range(20):
            g_copy = g.copy()
            # Randomly modify some spider phases
            vertices = list(g_copy.vertices())
            for v in rng.sample(vertices, min(3, len(vertices))):
                if g_copy.type(v) in (1, 2):  # Z or X spider
                    new_phase = Fraction(rng.randint(0, 7), 4)
                    g_copy.set_phase(v, new_phase)

            try:
                extracted = zx.extract_circuit(g_copy.copy())
                successes += 1
                if extracted.stats() != orig_stats:
                    novel += 1
            except Exception:
                pass

        print(f"  {algo_key}: {successes}/20 phase-modified extractable, {novel} novel")
    print()


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6()
    print("=== ALL EXPERIMENTS DONE ===")
