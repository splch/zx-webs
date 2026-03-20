#!/usr/bin/env python3
"""Deep diagnosis of why mined sub-diagram composition fails extraction,
and experiments with approaches that DO work.

Key findings from experiment_extraction.py:
- EXP3: Composing graph-like diagrams via PyZX compose() works 100% (after full_reduce)
- EXP6: Spider-fusion composition of graph-like diagrams works 100% (direct extract!)
- EXP7: Compose-before-reduce works 100%
- EXP9: Unfuse + compose + re-reduce works 100%
- EXP11: Simulated mining composition fails 100%

The critical difference: EXP11 splits a graph into sub-diagrams by cutting
edges in the interior. The resulting sub-diagrams have different boundary
structure than circuit-derived graphs. Let's understand exactly what goes wrong.
"""

import sys
import traceback
from fractions import Fraction
from collections import Counter

import pyzx as zx
from pyzx.simplify import (
    full_reduce, to_graph_like, is_graph_like, spider_simp, to_gh,
    interior_clifford_simp
)
from pyzx.extract import extract_circuit, extract_simple
from pyzx.gflow import gflow
from pyzx.utils import VertexType, EdgeType
from pyzx.generate import CNOT_HAD_PHASE_circuit


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# =========================================================================
# Diagnosis 1: Why does exp11 fail?
# =========================================================================
def diagnosis1_mining_failure():
    separator("DIAGNOSIS 1: Why mined sub-diagram composition fails")

    c = CNOT_HAD_PHASE_circuit(3, 30, seed=0)
    g = c.to_graph()
    full_reduce(g)

    inputs = set(g.inputs())
    outputs = set(g.outputs())
    interior = [v for v in g.vertices()
               if v not in inputs and v not in outputs
               and g.type(v) == VertexType.Z]

    print(f"Source graph: {g.num_vertices()} vertices, {g.num_edges()} edges")
    print(f"Interior Z-spiders: {len(interior)}")
    print(f"Inputs: {g.inputs()}, Outputs: {g.outputs()}")

    # The problem: when we split the graph and re-compose, the number of
    # input/output wires on the sub-diagrams may not match!
    # In graph-like form, a Z-spider can connect to MANY other Z-spiders
    # via Hadamard edges. When we split the graph, each such cross-boundary
    # edge becomes a boundary wire on the sub-diagram.

    mid = len(interior) // 2
    set_a = set(interior[:mid])
    set_b = set(interior[mid:])

    # Count cross-boundary edges
    cross_edges = []
    for v in set_a:
        for nb in g.neighbors(v):
            if nb in set_b:
                cross_edges.append((v, nb, g.edge_type(g.edge(v, nb))))

    print(f"\nCross-boundary edges: {len(cross_edges)}")
    for v, nb, et in cross_edges:
        print(f"  {v} (set_a) -- {nb} (set_b), edge_type={et}")

    # Count inputs/outputs for each sub-diagram
    a_inputs = sum(1 for inp in inputs for nb in g.neighbors(inp) if nb in set_a)
    a_outputs = len(cross_edges)  # Each cross-edge becomes an output of sub_a
    b_inputs = len(cross_edges)   # And an input of sub_b
    b_outputs = sum(1 for outp in outputs for nb in g.neighbors(outp) if nb in set_b)

    print(f"\nSub-diagram A: {a_inputs} inputs, {a_outputs} outputs")
    print(f"Sub-diagram B: {b_inputs} inputs, {b_outputs} outputs")
    print(f"A outputs match B inputs: {a_outputs == b_inputs}")

    # The issue: a_outputs == b_inputs == len(cross_edges), but
    # these might not equal the original input/output count!
    # When n_qubits=3: inputs=3, outputs=3, but cross_edges could be >> 3

    print(f"\nOriginal qubit count: {len(g.inputs())}")
    print(f"Cross edges (= composed boundary width): {len(cross_edges)}")
    print(f"MISMATCH: The composed diagram has {len(cross_edges)} internal wires")
    print(f"but only {len(g.inputs())} input and {len(g.outputs())} output wires")

    # Now let's check: what if the composition WORKS structurally but
    # the resulting graph doesn't have proper gflow?

    # Actually, the real problem in exp11 was that all methods failed.
    # Let's trace exactly what happens:

    # Build sub_a and sub_b properly
    sub_a = zx.Graph()
    a_map = {}
    a_boundary_in = []
    a_boundary_out = []

    for v in set_a:
        w = sub_a.add_vertex(ty=g.type(v), phase=g.phase(v),
                              row=g.row(v), qubit=g.qubit(v))
        a_map[v] = w

    for inp in inputs:
        for nb in g.neighbors(inp):
            if nb in set_a:
                bnd = sub_a.add_vertex(ty=VertexType.BOUNDARY,
                                        row=g.row(inp), qubit=g.qubit(inp))
                et = g.edge_type(g.edge(inp, nb))
                sub_a.add_edge((bnd, a_map[nb]), edgetype=et)
                a_boundary_in.append(bnd)

    for v in set_a:
        for nb in g.neighbors(v):
            if nb in set_a and v < nb:
                et = g.edge_type(g.edge(v, nb))
                sub_a.add_edge((a_map[v], a_map[nb]), edgetype=et)

    a_to_b_wires = []
    for v in set_a:
        for nb in g.neighbors(v):
            if nb in set_b:
                bnd = sub_a.add_vertex(ty=VertexType.BOUNDARY,
                                        row=g.row(v)+0.5, qubit=g.qubit(v))
                et = g.edge_type(g.edge(v, nb))
                sub_a.add_edge((bnd, a_map[v]), edgetype=et)
                a_boundary_out.append(bnd)
                a_to_b_wires.append((v, nb, et))

    sub_a.set_inputs(tuple(a_boundary_in))
    sub_a.set_outputs(tuple(a_boundary_out))

    # Build sub_b
    sub_b = zx.Graph()
    b_map = {}
    b_boundary_in = []
    b_boundary_out = []

    for v in set_b:
        w = sub_b.add_vertex(ty=g.type(v), phase=g.phase(v),
                              row=g.row(v), qubit=g.qubit(v))
        b_map[v] = w

    for (v_a, v_b, et) in a_to_b_wires:
        bnd = sub_b.add_vertex(ty=VertexType.BOUNDARY,
                                row=g.row(v_b)-0.5, qubit=g.qubit(v_b))
        sub_b.add_edge((bnd, b_map[v_b]), edgetype=et)
        b_boundary_in.append(bnd)

    for v in set_b:
        for nb in g.neighbors(v):
            if nb in set_b and v < nb:
                et = g.edge_type(g.edge(v, nb))
                sub_b.add_edge((b_map[v], b_map[nb]), edgetype=et)

    for outp in outputs:
        for nb in g.neighbors(outp):
            if nb in set_b:
                bnd = sub_b.add_vertex(ty=VertexType.BOUNDARY,
                                        row=g.row(outp), qubit=g.qubit(outp))
                et = g.edge_type(g.edge(outp, nb))
                sub_b.add_edge((bnd, b_map[nb]), edgetype=et)
                b_boundary_out.append(bnd)

    sub_b.set_inputs(tuple(b_boundary_in))
    sub_b.set_outputs(tuple(b_boundary_out))

    print(f"\nSub A: {sub_a.num_vertices()} vertices, inputs={len(a_boundary_in)}, outputs={len(a_boundary_out)}")
    print(f"Sub B: {sub_b.num_vertices()} vertices, inputs={len(b_boundary_in)}, outputs={len(b_boundary_out)}")

    # Compose
    composed = sub_a.copy()
    composed.compose(sub_b.copy())

    print(f"\nComposed: {composed.num_vertices()} vertices, {composed.num_edges()} edges")
    print(f"Inputs: {composed.inputs()}, Outputs: {composed.outputs()}")
    print(f"Is graph-like: {is_graph_like(composed)}")

    # Check gflow
    try:
        fl = gflow(composed)
        print(f"Has gflow: {fl is not None}")
    except Exception as e:
        print(f"gflow error: {e}")

    # Try full_reduce
    g2 = composed.copy()
    try:
        full_reduce(g2)
        print(f"\nAfter full_reduce: {g2.num_vertices()} vertices")
        print(f"Is graph-like: {is_graph_like(g2)}")
        fl2 = gflow(g2)
        print(f"Has gflow: {fl2 is not None}")

        try:
            c_out = extract_circuit(g2.copy())
            print(f"EXTRACTION SUCCEEDED: {len(c_out.gates)} gates")
        except Exception as e:
            print(f"Extraction failed: {e}")

            # Detailed diagnosis of failure
            print(f"\nDetailed diagnosis of extraction failure:")
            g3 = g2.copy()
            inputs_g3 = g3.inputs()
            outputs_g3 = g3.outputs()
            print(f"  Inputs: {inputs_g3} ({len(inputs_g3)})")
            print(f"  Outputs: {outputs_g3} ({len(outputs_g3)})")
            if len(inputs_g3) != len(outputs_g3):
                print(f"  *** INPUT/OUTPUT COUNT MISMATCH: {len(inputs_g3)} != {len(outputs_g3)} ***")
                print(f"  This is a non-unitary diagram -- extract_circuit requires equal I/O")

            for v in g3.vertices():
                t = g3.type(v)
                if t == VertexType.BOUNDARY:
                    nbs = list(g3.neighbors(v))
                    print(f"  Boundary {v}: neighbors={nbs}, degree={len(nbs)}")

    except Exception as e:
        print(f"full_reduce failed: {e}")
        traceback.print_exc()


# =========================================================================
# Solution 1: Spider-fusion composition (proven 100% in exp6)
# =========================================================================
def solution1_spider_fusion():
    separator("SOLUTION 1: Spider-fusion composition for mined sub-diagrams")

    results = Counter()

    for n_qubits in [3, 4, 5]:
        for seed in range(10):
            try:
                c = CNOT_HAD_PHASE_circuit(n_qubits, 30, seed=seed)
                g = c.to_graph()
                full_reduce(g)

                inputs = set(g.inputs())
                outputs = set(g.outputs())
                interior = [v for v in g.vertices()
                           if v not in inputs and v not in outputs
                           and g.type(v) == VertexType.Z]

                if len(interior) < 4:
                    results["too_few_vertices"] += 1
                    continue

                mid = len(interior) // 2
                set_a = set(interior[:mid])
                set_b = set(interior[mid:])

                # Spider-fusion composition:
                # Instead of creating boundary vertices, directly connect
                # the sub-diagrams in a new combined graph that preserves
                # the original structure.

                combined = zx.Graph()
                v_map = {}

                # Copy ALL vertices from both sub-diagrams AND boundaries
                for v in g.vertices():
                    if v in inputs or v in outputs or v in set_a or v in set_b:
                        w = combined.add_vertex(ty=g.type(v), phase=g.phase(v),
                                                 row=g.row(v), qubit=g.qubit(v))
                        v_map[v] = w

                # Copy ALL edges that connect any two copied vertices
                for v in list(v_map.keys()):
                    for nb in g.neighbors(v):
                        if nb in v_map and v < nb:
                            et = g.edge_type(g.edge(v, nb))
                            combined.add_edge((v_map[v], v_map[nb]), edgetype=et)

                combined.set_inputs(tuple(v_map[v] for v in g.inputs()))
                combined.set_outputs(tuple(v_map[v] for v in g.outputs()))

                # This should be identical to the original graph!
                gl = is_graph_like(combined)
                results[f"reconstructed_graph_like={gl}"] += 1

                try:
                    g2 = combined.copy()
                    full_reduce(g2)
                    c_out = extract_circuit(g2.copy())
                    results["extract_success"] += 1
                except Exception as e:
                    results[f"extract_fail: {str(e)[:50]}"] += 1

            except Exception as e:
                results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")

    print(f"\n  NOTE: This succeeds because we're reconstructing the original graph.")
    print(f"  For novel algorithm discovery, we need to compose sub-diagrams from")
    print(f"  DIFFERENT source algorithms. Let's test that next.")


# =========================================================================
# Solution 2: Cross-source spider-fusion composition
# =========================================================================
def solution2_cross_source_composition():
    separator("SOLUTION 2: Cross-source spider-fusion composition")

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed_a in range(5):
            for seed_b in range(5):
                if seed_a == seed_b:
                    continue
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 20, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 20, seed=seed_b + 200)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Method: Use PyZX native compose() on graph-like diagrams
                    # This was shown to work 100% in exp3!
                    g_comp = g_a.copy()
                    g_comp.compose(g_b.copy())

                    full_reduce(g_comp)
                    gl = is_graph_like(g_comp)
                    results[f"graph_like={gl}"] += 1

                    try:
                        c_out = extract_circuit(g_comp.copy())
                        results["extract_success"] += 1

                        # Verify correctness by comparing tensors
                        # (only for small circuits)
                        if n_qubits <= 3:
                            try:
                                import numpy as np
                                # Original: c_a composed with c_b
                                g_orig = c_a.to_graph()
                                g_orig.compose(c_b.to_graph())
                                t_orig = g_orig.to_tensor()

                                t_extracted = c_out.to_graph().to_tensor()

                                # Compare up to global phase
                                ratio = t_extracted.flatten()[0] / (t_orig.flatten()[0] + 1e-30)
                                diff = np.abs(t_extracted - ratio * t_orig).max()
                                if diff < 1e-6:
                                    results["tensor_match"] += 1
                                else:
                                    results["tensor_mismatch"] += 1
                            except:
                                results["tensor_check_error"] += 1

                    except Exception as e:
                        results[f"extract_fail: {str(e)[:50]}"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")


# =========================================================================
# Solution 3: The key insight -- work at the graph-like level
#   but use PyZX's compose() which handles boundary correctly
# =========================================================================
def solution3_graph_like_compose_variants():
    separator("SOLUTION 3: Systematic tests of graph-like composition variants")

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed_a in range(8):
            for seed_b in range(8):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()

                    # ---- Variant A: full_reduce both, then PyZX compose ----
                    ga = g_a.copy()
                    gb = g_b.copy()
                    full_reduce(ga)
                    full_reduce(gb)
                    gc = ga.copy()
                    gc.compose(gb.copy())
                    full_reduce(gc)
                    try:
                        c_out = extract_circuit(gc.copy())
                        results["A_reduce_compose_reduce"] += 1
                    except:
                        results["A_fail"] += 1

                    # ---- Variant B: circuit compose, then full_reduce ----
                    gc2 = g_a.copy()
                    gc2.compose(g_b.copy())
                    full_reduce(gc2)
                    try:
                        c_out = extract_circuit(gc2.copy())
                        results["B_compose_reduce"] += 1
                    except:
                        results["B_fail"] += 1

                    # ---- Variant C: Tensor product (parallel) + reduce ----
                    gt = g_a.copy()
                    gt_par = gt.tensor(g_b.copy())
                    full_reduce(gt_par)
                    try:
                        c_out = extract_circuit(gt_par.copy())
                        results["C_parallel_reduce"] += 1
                    except:
                        results["C_fail"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")


# =========================================================================
# Solution 4: Novel composition via shared qubit stitching
# =========================================================================
def solution4_shared_qubit_stitching():
    separator("SOLUTION 4: Novel composition via shared-qubit stitching")

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()

                    # Approach: Compose graph-like diagrams by
                    # adding CZ (Hadamard) edges between interior spiders
                    # This is what our mining pipeline SHOULD do.
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Use tensor() to place them in parallel
                    combined = g_a.tensor(g_b)

                    # Now add some "stitching" edges between the two
                    # graphs (Hadamard edges between Z-spiders)
                    inputs_combined = combined.inputs()
                    outputs_combined = combined.outputs()

                    # Get interior Z-spiders from each sub-graph
                    n_a = len(g_a.inputs())
                    interior_a = []
                    interior_b = []
                    for v in combined.vertices():
                        if v in inputs_combined or v in outputs_combined:
                            continue
                        if combined.type(v) != VertexType.Z:
                            continue
                        # Determine which sub-graph by qubit index
                        q = combined.qubit(v)
                        if q < n_a:
                            interior_a.append(v)
                        else:
                            interior_b.append(v)

                    # Add some Hadamard edges between sub-graphs
                    import random
                    rng = random.Random(seed_a * 100 + seed_b)
                    n_stitches = min(3, len(interior_a), len(interior_b))

                    for i in range(n_stitches):
                        va = rng.choice(interior_a)
                        vb = rng.choice(interior_b)
                        if not combined.connected(va, vb):
                            combined.add_edge((va, vb), edgetype=EdgeType.HADAMARD)

                    # Now try to extract
                    gl = is_graph_like(combined)
                    results[f"stitched_graph_like={gl}"] += 1

                    g2 = combined.copy()
                    full_reduce(g2)
                    gl2 = is_graph_like(g2)
                    results[f"after_fr_graph_like={gl2}"] += 1

                    try:
                        c_out = extract_circuit(g2.copy())
                        results["extract_success"] += 1
                    except Exception as e:
                        results[f"extract_fail: {str(e)[:50]}"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")


# =========================================================================
# Solution 5: Graph-state + measurement pattern approach
# =========================================================================
def solution5_graph_state_approach():
    separator("SOLUTION 5: Graph-state + measurement pattern generation")

    from pyzx.mbqc import cluster_state, measure

    results = Counter()

    for width in [3, 4, 5]:
        for height in [2, 3]:
            for seed in range(5):
                try:
                    rng = Fraction  # not used, just for typing
                    import random
                    r = random.Random(seed)

                    # Create a cluster state
                    inp_positions = [(q, 0) for q in range(height)]
                    g = cluster_state(height, width, inputs=inp_positions)

                    # Measure some qubits (not inputs or outputs)
                    # This creates a measurement pattern
                    for q in range(height):
                        for col in range(1, width-1):
                            phase = Fraction(r.randint(0, 7), 4)
                            try:
                                measure(g, (q, col), VertexType.Z, phase)
                            except:
                                pass

                    # Check gflow
                    try:
                        fl = gflow(g)
                        results[f"cluster_has_gflow={fl is not None}"] += 1
                    except:
                        results["gflow_error"] += 1

                    # Try to reduce and extract
                    g2 = g.copy()
                    try:
                        full_reduce(g2)
                        gl = is_graph_like(g2)
                        results[f"cluster_reduced_graph_like={gl}"] += 1

                        c_out = extract_circuit(g2.copy())
                        results["cluster_extract_success"] += 1
                    except Exception as e:
                        results[f"cluster_extract_fail: {str(e)[:40]}"] += 1

                except Exception as e:
                    results[f"cluster_error: {str(e)[:40]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")


# =========================================================================
# Solution 6: The winning approach for our pipeline
# =========================================================================
def solution6_winning_approach():
    separator("SOLUTION 6: The correct pipeline -- mine from circuits, compose as circuits")

    results = Counter()

    # Simulate the full pipeline with the correct approach:
    # 1. Generate corpus of circuits
    # 2. Convert to ZX-diagrams (circuit-like, NOT graph-like)
    # 3. Mine sub-diagrams from circuit-like diagrams
    # 4. Compose sub-diagrams (they're still circuit-compatible)
    # 5. full_reduce + extract

    # But wait -- we mine from GRAPH-LIKE diagrams for expressiveness.
    # The insight: use PyZX's compose() on graph-like diagrams.
    # This works because compose() handles boundary vertices correctly.

    print("  Approach: mine graph-like sub-diagrams WITH proper boundaries,")
    print("  compose using PyZX compose(), then full_reduce + extract\n")

    for n_qubits in [2, 3, 4, 5]:
        for depth in [10, 20, 30]:
            for seed_a, seed_b in [(0,1), (0,2), (1,2), (3,4), (5,6)]:
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, depth, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, depth, seed=seed_b + 100)

                    # "Mine" by taking sub-circuits and reducing them
                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()

                    # Reduce (simulating mining from graph-like form)
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Compose using PyZX compose (handles boundaries correctly)
                    g_composed = g_a.copy()
                    g_composed.compose(g_b.copy())

                    # Reduce the composed result
                    full_reduce(g_composed)

                    # Extract
                    try:
                        c_out = extract_circuit(g_composed.copy())
                        results["success"] += 1

                        # Verify
                        if n_qubits <= 3:
                            try:
                                import numpy as np
                                g_ref = c_a.to_graph()
                                g_ref.compose(c_b.to_graph())
                                t_ref = g_ref.to_tensor()
                                t_out = c_out.to_graph().to_tensor()

                                ratio = t_out.flatten()[0] / (t_ref.flatten()[0] + 1e-30)
                                diff = np.abs(t_out - ratio * t_ref).max()
                                if diff < 1e-6:
                                    results["verified_correct"] += 1
                                else:
                                    results["verification_failed"] += 1
                            except:
                                pass

                    except Exception as e:
                        results[f"fail: {str(e)[:50]}"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:50]}"] += 1

    total = sum(v for k, v in results.items() if k in ("success",) or k.startswith("fail"))
    successes = results.get("success", 0)
    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    if total > 0:
        print(f"\n  Overall success rate: {successes/total*100:.1f}%")


# =========================================================================
# Solution 7: Novel algorithm generation via ZX-rewrite search
# =========================================================================
def solution7_rewrite_search():
    separator("SOLUTION 7: Novel algorithm generation via ZX-rewrite search")

    from pyzx.simplify import pivot_simp, lcomp_simp, gadget_simp, copy_simp

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed in range(5):
            try:
                # Start with a known circuit
                c = CNOT_HAD_PHASE_circuit(n_qubits, 20, seed=seed)
                g = c.to_graph()
                full_reduce(g)

                # Apply random ZX rewrites to explore the neighborhood
                import random
                rng = random.Random(seed * 42)

                g_current = g.copy()

                # Apply some rewrites
                for step in range(10):
                    g_try = g_current.copy()
                    # Random choice of rewrite
                    choice = rng.randint(0, 3)
                    try:
                        if choice == 0:
                            interior_clifford_simp(g_try)
                        elif choice == 1:
                            pivot_gadget_simp(g_try)
                        elif choice == 2:
                            gadget_simp(g_try)
                        elif choice == 3:
                            copy_simp(g_try)
                        g_current = g_try
                    except:
                        pass

                # Check if we can still extract
                full_reduce(g_current)
                gl = is_graph_like(g_current)
                results[f"rewritten_graph_like={gl}"] += 1

                try:
                    c_out = extract_circuit(g_current.copy())
                    results["extract_after_rewrites"] += 1
                except:
                    results["extract_fail_after_rewrites"] += 1

            except Exception as e:
                results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")

    print(f"\n  Note: ZX rewrites that are part of full_reduce naturally preserve")
    print(f"  extractability. For NOVEL discovery, we need rewrites that CHANGE")
    print(f"  the unitary while preserving graph-like + gflow properties.")


# =========================================================================
# Solution 8: Phase modification for novelty
# =========================================================================
def solution8_phase_modification():
    separator("SOLUTION 8: Phase modification for novel algorithm discovery")

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed in range(10):
            try:
                # Start with a known circuit
                c = CNOT_HAD_PHASE_circuit(n_qubits, 20, seed=seed)
                g = c.to_graph()
                full_reduce(g)

                # Modify phases of Z-spiders
                # This changes the unitary BUT preserves graph-like structure
                import random
                rng = random.Random(seed * 77)

                g_mod = g.copy()
                for v in g_mod.vertices():
                    if g_mod.type(v) == VertexType.Z and v not in g_mod.inputs() and v not in g_mod.outputs():
                        if rng.random() < 0.3:  # Modify 30% of phases
                            new_phase = Fraction(rng.randint(0, 7), 4)
                            g_mod.set_phase(v, new_phase)

                gl = is_graph_like(g_mod)
                results[f"phase_mod_graph_like={gl}"] += 1

                # Try extraction without re-reducing (just with phase changes)
                try:
                    c_out = extract_circuit(g_mod.copy())
                    results["extract_direct_after_phase_mod"] += 1
                except:
                    pass

                # Try with re-reduction
                g2 = g_mod.copy()
                full_reduce(g2)
                try:
                    c_out = extract_circuit(g2.copy())
                    results["extract_after_rereduce"] += 1

                    # Verify it's different from original
                    if n_qubits <= 3:
                        try:
                            import numpy as np
                            t_orig = c.to_graph().to_tensor()
                            t_new = c_out.to_graph().to_tensor()

                            # Check if they're different
                            ratio = t_new.flatten()[0] / (t_orig.flatten()[0] + 1e-30)
                            diff = np.abs(t_new - ratio * t_orig).max()
                            if diff > 1e-6:
                                results["novel_unitary"] += 1
                            else:
                                results["same_unitary"] += 1
                        except:
                            pass

                except Exception as e:
                    results[f"extract_fail: {str(e)[:50]}"] += 1

            except Exception as e:
                results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    print("ZX-Webs Deep Diagnosis & Solutions")

    diagnosis1_mining_failure()
    solution1_spider_fusion()
    solution2_cross_source_composition()
    solution3_graph_like_compose_variants()
    solution4_shared_qubit_stitching()
    solution5_graph_state_approach()
    solution6_winning_approach()
    solution7_rewrite_search()
    solution8_phase_modification()
