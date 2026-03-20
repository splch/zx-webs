#!/usr/bin/env python3
"""Comprehensive experiments on ZX-diagram composition and circuit extraction.

Tests multiple approaches for making composed ZX sub-diagrams extractable:
1. full_reduce on composed diagrams
2. to_graph_like + full_reduce
3. PyZX native compose() on circuit-derived graphs
4. Compose graph-like diagrams with proper boundary handling
5. gflow checking
6. Random graph-like generation and extraction
7. Causal-flow-preserving composition
"""

import sys
import traceback
from fractions import Fraction
from collections import Counter

import pyzx as zx
from pyzx.simplify import (
    full_reduce, to_graph_like, is_graph_like, spider_simp, to_gh,
    clifford_simp, interior_clifford_simp, gadget_simp, pivot_gadget_simp
)
from pyzx.extract import extract_circuit, extract_simple
from pyzx.gflow import gflow
from pyzx.utils import VertexType, EdgeType
from pyzx.generate import CNOT_HAD_PHASE_circuit, identity


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# =========================================================================
# Experiment 1: Baseline -- circuits -> full_reduce -> extract
# =========================================================================
def experiment1_baseline():
    separator("EXP 1: Baseline -- circuit -> full_reduce -> extract")

    successes = 0
    failures = 0

    for n_qubits in [2, 3, 4, 5]:
        for depth in [10, 20, 50]:
            for seed in range(5):
                try:
                    c = CNOT_HAD_PHASE_circuit(n_qubits, depth, seed=seed)
                    g = c.to_graph()

                    # Check graph-like before
                    gl_before = is_graph_like(g)

                    full_reduce(g)
                    gl_after = is_graph_like(g)

                    c2 = extract_circuit(g.copy())
                    successes += 1
                except Exception as e:
                    failures += 1

    print(f"  Baseline: {successes} successes, {failures} failures out of {successes+failures}")
    print(f"  Success rate: {successes/(successes+failures)*100:.1f}%")
    return successes, failures


# =========================================================================
# Experiment 2: Compose two circuits via PyZX compose(), then extract
# =========================================================================
def experiment2_pyzx_compose():
    separator("EXP 2: PyZX native compose() on circuit-derived graphs")

    results = {"direct_extract": 0, "after_full_reduce": 0, "fail": 0}

    for n_qubits in [2, 3, 4]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()

                    # Native compose
                    g_composed = g_a.copy()
                    g_composed.compose(g_b.copy())

                    # Try direct extract (should work -- it's still circuit-like)
                    try:
                        c_out = extract_circuit(g_composed.copy())
                        results["direct_extract"] += 1
                        continue
                    except:
                        pass

                    # Try full_reduce then extract
                    try:
                        g2 = g_composed.copy()
                        full_reduce(g2)
                        c_out = extract_circuit(g2.copy())
                        results["after_full_reduce"] += 1
                        continue
                    except:
                        pass

                    results["fail"] += 1
                except Exception as e:
                    results["fail"] += 1

    total = sum(results.values())
    print(f"  Results: {results}")
    print(f"  Total success rate: {(results['direct_extract']+results['after_full_reduce'])/total*100:.1f}%")
    return results


# =========================================================================
# Experiment 3: Compose REDUCED graphs, then try to extract
# =========================================================================
def experiment3_compose_reduced():
    separator("EXP 3: Compose fully-reduced (graph-like) diagrams, then extract")

    results = {
        "compose_gl_direct": 0,
        "compose_gl_fullreduce": 0,
        "compose_gl_to_gl_fullreduce": 0,
        "fail": 0,
        "compose_failed": 0,
    }

    diagnostics = Counter()

    for n_qubits in [2, 3, 4]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()

                    # Fully reduce both
                    full_reduce(g_a)
                    full_reduce(g_b)

                    assert is_graph_like(g_a), "g_a not graph-like after full_reduce"
                    assert is_graph_like(g_b), "g_b not graph-like after full_reduce"

                    # Try to compose using native compose
                    try:
                        g_composed = g_a.copy()
                        g_composed.compose(g_b.copy())
                    except Exception as e:
                        results["compose_failed"] += 1
                        diagnostics[f"compose_error: {type(e).__name__}"] += 1
                        continue

                    # Check if result is graph-like
                    gl = is_graph_like(g_composed)
                    diagnostics[f"composed_is_graph_like={gl}"] += 1

                    # Try direct extract
                    try:
                        c_out = extract_circuit(g_composed.copy())
                        results["compose_gl_direct"] += 1
                        continue
                    except Exception as e:
                        diagnostics[f"direct_extract_error: {str(e)[:60]}"] += 1

                    # Try full_reduce then extract
                    try:
                        g2 = g_composed.copy()
                        full_reduce(g2)
                        gl2 = is_graph_like(g2)
                        diagnostics[f"after_fullreduce_is_graph_like={gl2}"] += 1
                        c_out = extract_circuit(g2.copy())
                        results["compose_gl_fullreduce"] += 1
                        continue
                    except Exception as e:
                        diagnostics[f"fullreduce_extract_error: {str(e)[:60]}"] += 1

                    # Try to_graph_like + full_reduce then extract
                    try:
                        g3 = g_composed.copy()
                        to_graph_like(g3)
                        full_reduce(g3)
                        c_out = extract_circuit(g3.copy())
                        results["compose_gl_to_gl_fullreduce"] += 1
                        continue
                    except Exception as e:
                        diagnostics[f"to_gl_extract_error: {str(e)[:60]}"] += 1

                    results["fail"] += 1

                except Exception as e:
                    results["fail"] += 1
                    diagnostics[f"outer_error: {str(e)[:60]}"] += 1

    total = sum(results.values())
    print(f"  Results: {results}")
    print(f"  Diagnostics:")
    for k, v in sorted(diagnostics.items()):
        print(f"    {k}: {v}")
    return results


# =========================================================================
# Experiment 4: Manual composition with proper boundary handling
# =========================================================================
def experiment4_manual_compose_with_boundaries():
    separator("EXP 4: Manual composition with proper boundary vertices")

    results = {
        "extract_after_to_gl_fr": 0,
        "extract_after_fr": 0,
        "fail": 0,
    }
    diagnostics = Counter()

    for n_qubits in [2, 3]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()

                    # Fully reduce both
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Manual composition:
                    # 1. Build a new graph
                    # 2. Copy all vertices from g_a and g_b
                    # 3. For each output of g_a and corresponding input of g_b,
                    #    find the Z-spider connected to the boundary, and connect
                    #    them with a Hadamard edge (graph-like style)
                    # 4. Remove intermediate boundary vertices
                    # 5. Set new inputs from g_a, outputs from g_b

                    combined = zx.Graph()

                    # Copy g_a
                    a_map = {}
                    for v in g_a.vertices():
                        w = combined.add_vertex(
                            ty=g_a.type(v), phase=g_a.phase(v),
                            row=g_a.row(v), qubit=g_a.qubit(v)
                        )
                        a_map[v] = w
                    for e in g_a.edges():
                        s, t = g_a.edge_st(e)
                        combined.add_edge((a_map[s], a_map[t]), edgetype=g_a.edge_type(e))

                    # Copy g_b with row offset
                    max_row_a = max((g_a.row(v) for v in g_a.vertices()), default=0)
                    b_map = {}
                    for v in g_b.vertices():
                        w = combined.add_vertex(
                            ty=g_b.type(v), phase=g_b.phase(v),
                            row=g_b.row(v) + max_row_a + 1, qubit=g_b.qubit(v)
                        )
                        b_map[v] = w
                    for e in g_b.edges():
                        s, t = g_b.edge_st(e)
                        combined.add_edge((b_map[s], b_map[t]), edgetype=g_b.edge_type(e))

                    # Connect: for each output boundary of A, find the Z-spider
                    # and for the corresponding input boundary of B, find the Z-spider
                    # Then connect the two Z-spiders with a Hadamard edge
                    outputs_a = g_a.outputs()
                    inputs_b = g_b.inputs()

                    if len(outputs_a) != len(inputs_b):
                        results["fail"] += 1
                        continue

                    for i in range(len(outputs_a)):
                        out_bnd = a_map[outputs_a[i]]
                        in_bnd = b_map[inputs_b[i]]

                        # Find Z-spider neighbor of output boundary in A
                        z_a = None
                        for nb in combined.neighbors(out_bnd):
                            if combined.type(nb) == VertexType.Z:
                                z_a = nb
                                break

                        # Find Z-spider neighbor of input boundary in B
                        z_b = None
                        for nb in combined.neighbors(in_bnd):
                            if combined.type(nb) == VertexType.Z:
                                z_b = nb
                                break

                        if z_a is None or z_b is None:
                            diagnostics["missing_z_spider"] += 1
                            continue

                        # Get edge types from boundaries to spiders
                        e_a = combined.edge(out_bnd, z_a)
                        et_a = combined.edge_type(e_a)
                        e_b = combined.edge(in_bnd, z_b)
                        et_b = combined.edge_type(e_b)

                        # Remove boundary vertices and their edges
                        combined.remove_vertex(out_bnd)
                        combined.remove_vertex(in_bnd)

                        # Determine correct edge type between z_a and z_b
                        # If both edges were SIMPLE: z_a -- boundary -- boundary -- z_b
                        # This is like an identity wire, so z_a and z_b should fuse (SIMPLE edge)
                        # If one is HADAMARD: we get a HADAMARD edge
                        # If both HADAMARD: they cancel, giving SIMPLE
                        if et_a == et_b:
                            new_et = EdgeType.SIMPLE
                        else:
                            new_et = EdgeType.HADAMARD

                        combined.add_edge((z_a, z_b), edgetype=new_et)

                    # Set inputs/outputs
                    new_inputs = tuple(a_map[v] for v in g_a.inputs())
                    new_outputs = tuple(b_map[v] for v in g_b.outputs())
                    combined.set_inputs(new_inputs)
                    combined.set_outputs(new_outputs)

                    # Check graph-like
                    gl = is_graph_like(combined)
                    diagnostics[f"manual_composed_graph_like={gl}"] += 1

                    # Try to_graph_like + full_reduce + extract
                    try:
                        g2 = combined.copy()
                        to_graph_like(g2)
                        full_reduce(g2)
                        gl2 = is_graph_like(g2)
                        diagnostics[f"after_to_gl_fr_graph_like={gl2}"] += 1
                        c_out = extract_circuit(g2.copy())
                        results["extract_after_to_gl_fr"] += 1
                        continue
                    except Exception as e:
                        diagnostics[f"to_gl_fr_error: {str(e)[:60]}"] += 1

                    # Try just full_reduce + extract
                    try:
                        g3 = combined.copy()
                        full_reduce(g3)
                        c_out = extract_circuit(g3.copy())
                        results["extract_after_fr"] += 1
                        continue
                    except Exception as e:
                        diagnostics[f"fr_error: {str(e)[:60]}"] += 1

                    results["fail"] += 1

                except Exception as e:
                    results["fail"] += 1
                    diagnostics[f"outer_error: {str(e)[:60]}"] += 1

    total = sum(results.values())
    print(f"  Results: {results}")
    success = results["extract_after_to_gl_fr"] + results["extract_after_fr"]
    print(f"  Success rate: {success/total*100:.1f}%")
    print(f"  Diagnostics:")
    for k, v in sorted(diagnostics.items()):
        print(f"    {k}: {v}")
    return results


# =========================================================================
# Experiment 5: Check gflow on composed diagrams
# =========================================================================
def experiment5_gflow_check():
    separator("EXP 5: gflow analysis on composed diagrams")

    results = Counter()

    for n_qubits in [2, 3]:
        for seed_a in range(3):
            for seed_b in range(3):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()

                    # Check gflow on individual circuits
                    g_a_gl = g_a.copy()
                    full_reduce(g_a_gl)
                    flow_a = gflow(g_a_gl)
                    results[f"individual_has_gflow={flow_a is not None}"] += 1

                    # Compose via native compose (circuit-like)
                    g_circuit = c_a.to_graph()
                    g_circuit.compose(c_b.to_graph())

                    # Check gflow on circuit-composed
                    g_cc = g_circuit.copy()
                    full_reduce(g_cc)
                    flow_cc = gflow(g_cc)
                    results[f"circuit_composed_has_gflow={flow_cc is not None}"] += 1

                    # Compose reduced forms
                    full_reduce(g_a)
                    full_reduce(g_b)
                    g_reduced = g_a.copy()
                    try:
                        g_reduced.compose(g_b.copy())
                        full_reduce(g_reduced)
                        flow_rc = gflow(g_reduced)
                        results[f"reduced_composed_has_gflow={flow_rc is not None}"] += 1
                    except Exception as e:
                        results[f"reduced_compose_error"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:40]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    return results


# =========================================================================
# Experiment 6: Spider fusion approach -- connect Z-spiders directly
# =========================================================================
def experiment6_spider_fusion_compose():
    separator("EXP 6: Spider-fusion composition (fuse boundary Z-spiders)")

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Build combined graph by FUSING boundary spiders
                    # In graph-like form after full_reduce:
                    #   output boundary --SIMPLE-- Z-spider (in g_a)
                    #   input boundary --SIMPLE-- Z-spider (in g_b)
                    # We want to FUSE the two Z-spiders into one
                    # (spider fusion: same-color spiders connected by simple edge fuse, phases add)

                    combined = zx.Graph()

                    # Copy g_a
                    a_map = {}
                    for v in g_a.vertices():
                        w = combined.add_vertex(
                            ty=g_a.type(v), phase=g_a.phase(v),
                            row=g_a.row(v), qubit=g_a.qubit(v)
                        )
                        a_map[v] = w
                    for e in g_a.edges():
                        s, t = g_a.edge_st(e)
                        combined.add_edge((a_map[s], a_map[t]), edgetype=g_a.edge_type(e))

                    max_row_a = max((g_a.row(v) for v in g_a.vertices()), default=0)
                    b_map = {}
                    for v in g_b.vertices():
                        w = combined.add_vertex(
                            ty=g_b.type(v), phase=g_b.phase(v),
                            row=g_b.row(v) + max_row_a + 2, qubit=g_b.qubit(v)
                        )
                        b_map[v] = w
                    for e in g_b.edges():
                        s, t = g_b.edge_st(e)
                        combined.add_edge((b_map[s], b_map[t]), edgetype=g_b.edge_type(e))

                    outputs_a = g_a.outputs()
                    inputs_b = g_b.inputs()

                    if len(outputs_a) != len(inputs_b):
                        results["size_mismatch"] += 1
                        continue

                    # For each wire: find the Z-spiders on each side
                    # Fuse them: keep z_a, redirect all edges from z_b to z_a, add phases
                    for i in range(len(outputs_a)):
                        out_bnd = a_map[outputs_a[i]]
                        in_bnd = b_map[inputs_b[i]]

                        # Find Z-spider neighbors
                        z_a = None
                        for nb in combined.neighbors(out_bnd):
                            if combined.type(nb) == VertexType.Z:
                                z_a = nb
                                break
                        z_b = None
                        for nb in combined.neighbors(in_bnd):
                            if combined.type(nb) == VertexType.Z:
                                z_b = nb
                                break

                        if z_a is None or z_b is None:
                            results["no_z_spider"] += 1
                            continue

                        # Get edge types
                        et_a = combined.edge_type(combined.edge(out_bnd, z_a))
                        et_b = combined.edge_type(combined.edge(in_bnd, z_b))

                        # Remove boundary vertices
                        combined.remove_vertex(out_bnd)
                        combined.remove_vertex(in_bnd)

                        # If both edges simple: z_a and z_b are connected by identity wire
                        # Spider fusion: merge z_b into z_a
                        # Redirect all neighbors of z_b to z_a
                        if et_a == EdgeType.SIMPLE and et_b == EdgeType.SIMPLE:
                            # Direct fusion: same as simple edge between them, then fuse
                            phase_b = combined.phase(z_b)
                            combined.add_to_phase(z_a, phase_b)

                            for nb in list(combined.neighbors(z_b)):
                                if nb == z_a:
                                    continue
                                et = combined.edge_type(combined.edge(z_b, nb))
                                # Check if z_a already connected to nb
                                if combined.connected(z_a, nb):
                                    # Parallel edge -- for Hadamard edges they cancel
                                    existing_et = combined.edge_type(combined.edge(z_a, nb))
                                    if existing_et == et:
                                        # Same type parallel edges: remove both (for Hadamard)
                                        combined.remove_edge(combined.edge(z_a, nb))
                                    # else: different types, complex case
                                else:
                                    combined.add_edge((z_a, nb), edgetype=et)

                            combined.remove_vertex(z_b)
                        else:
                            # Hadamard involved: connect with appropriate edge
                            if et_a != et_b:
                                combined.add_edge((z_a, z_b), edgetype=EdgeType.HADAMARD)
                            else:
                                combined.add_edge((z_a, z_b), edgetype=EdgeType.SIMPLE)

                    # Set inputs/outputs
                    new_inputs = tuple(a_map[v] for v in g_a.inputs())
                    new_outputs = tuple(b_map[v] for v in g_b.outputs())
                    combined.set_inputs(new_inputs)
                    combined.set_outputs(new_outputs)

                    # Clean up
                    spider_simp(combined)

                    gl = is_graph_like(combined)
                    results[f"fused_graph_like={gl}"] += 1

                    # Check gflow
                    try:
                        fl = gflow(combined)
                        results[f"fused_has_gflow={fl is not None}"] += 1
                    except:
                        results["gflow_error"] += 1

                    # Try extraction approaches
                    extracted = False

                    # Approach A: direct extract
                    try:
                        c_out = extract_circuit(combined.copy())
                        results["extract_direct"] += 1
                        extracted = True
                    except:
                        pass

                    if not extracted:
                        # Approach B: full_reduce + extract
                        try:
                            g2 = combined.copy()
                            full_reduce(g2)
                            c_out = extract_circuit(g2.copy())
                            results["extract_after_fr"] += 1
                            extracted = True
                        except:
                            pass

                    if not extracted:
                        # Approach C: to_graph_like + full_reduce + extract
                        try:
                            g3 = combined.copy()
                            to_graph_like(g3)
                            full_reduce(g3)
                            c_out = extract_circuit(g3.copy())
                            results["extract_after_togl_fr"] += 1
                            extracted = True
                        except Exception as e:
                            results[f"final_fail: {str(e)[:50]}"] += 1

                    if not extracted:
                        results["all_failed"] += 1

                except Exception as e:
                    results[f"outer_error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    return results


# =========================================================================
# Experiment 7: Compose circuits BEFORE reducing -- the key insight
# =========================================================================
def experiment7_compose_before_reduce():
    separator("EXP 7: Compose circuit graphs BEFORE reducing (the standard path)")

    results = Counter()

    for n_qubits in [2, 3, 4, 5]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 100)

                    # Compose as circuits first
                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()
                    g_a.compose(g_b)

                    # NOW reduce and extract
                    full_reduce(g_a)

                    gl = is_graph_like(g_a)
                    results[f"graph_like={gl}"] += 1

                    fl = gflow(g_a)
                    results[f"has_gflow={fl is not None}"] += 1

                    try:
                        c_out = extract_circuit(g_a.copy())
                        results["extract_success"] += 1
                    except Exception as e:
                        results[f"extract_fail: {str(e)[:50]}"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    return results


# =========================================================================
# Experiment 8: What exactly breaks in composed graph-like diagrams?
# =========================================================================
def experiment8_diagnose_failures():
    separator("EXP 8: Detailed diagnosis of why composed graph-like diagrams fail")

    for n_qubits in [2, 3]:
        for seed_a in [0, 1]:
            for seed_b in [0, 1]:
                c_a = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_a)
                c_b = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_b + 100)

                g_a = c_a.to_graph()
                g_b = c_b.to_graph()
                full_reduce(g_a)
                full_reduce(g_b)

                print(f"\n  --- n_q={n_qubits}, seed_a={seed_a}, seed_b={seed_b} ---")
                print(f"  g_a: {g_a.num_vertices()} vertices, {g_a.num_edges()} edges")
                print(f"  g_b: {g_b.num_vertices()} vertices, {g_b.num_edges()} edges")
                print(f"  g_a inputs: {g_a.inputs()}, outputs: {g_a.outputs()}")
                print(f"  g_b inputs: {g_b.inputs()}, outputs: {g_b.outputs()}")

                # Examine boundary structure
                for label, g in [("g_a", g_a), ("g_b", g_b)]:
                    for o in g.outputs():
                        nbs = list(g.neighbors(o))
                        for nb in nbs:
                            et = g.edge_type(g.edge(o, nb))
                            print(f"    {label} output {o} -> vertex {nb} (type={g.type(nb)}, phase={g.phase(nb)}, edge_type={et})")
                    for inp in g.inputs():
                        nbs = list(g.neighbors(inp))
                        for nb in nbs:
                            et = g.edge_type(g.edge(inp, nb))
                            print(f"    {label} input {inp} -> vertex {nb} (type={g.type(nb)}, phase={g.phase(nb)}, edge_type={et})")

                # Try native compose
                g_comp = g_a.copy()
                try:
                    g_comp.compose(g_b.copy())
                    print(f"  Composed: {g_comp.num_vertices()} vertices, {g_comp.num_edges()} edges")
                    gl = is_graph_like(g_comp)
                    print(f"  Is graph-like: {gl}")

                    # Check what specific properties fail
                    for v in g_comp.vertices():
                        if g_comp.type(v) not in [VertexType.Z, VertexType.BOUNDARY]:
                            print(f"    NON-ZX vertex: {v} has type {g_comp.type(v)}")

                    for v1 in g_comp.vertices():
                        for v2 in g_comp.neighbors(v1):
                            if v1 < v2:
                                if g_comp.type(v1) == VertexType.Z and g_comp.type(v2) == VertexType.Z:
                                    et = g_comp.edge_type(g_comp.edge(v1, v2))
                                    if et != EdgeType.HADAMARD:
                                        print(f"    NON-HAD Z-Z edge: {v1}-{v2} has type {et}")

                    # Check gflow
                    fl = gflow(g_comp)
                    print(f"  Has gflow: {fl is not None}")

                    # Try full_reduce
                    g2 = g_comp.copy()
                    full_reduce(g2)
                    gl2 = is_graph_like(g2)
                    print(f"  After full_reduce: graph-like={gl2}, vertices={g2.num_vertices()}")

                    fl2 = gflow(g2)
                    print(f"  After full_reduce: has_gflow={fl2 is not None}")

                    try:
                        c_out = extract_circuit(g2.copy())
                        print(f"  EXTRACTION SUCCEEDED: {len(c_out.gates)} gates")
                    except Exception as e:
                        print(f"  Extraction failed: {e}")

                except Exception as e:
                    print(f"  Compose failed: {e}")


# =========================================================================
# Experiment 9: The "unfuse + re-reduce" approach
# =========================================================================
def experiment9_unfuse_rereduce():
    separator("EXP 9: Unfuse non-Cliffords then compose and re-reduce")

    from pyzx.simplify import unfuse_non_cliffords

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 15, seed=seed_b + 100)

                    g_a = c_a.to_graph()
                    g_b = c_b.to_graph()
                    full_reduce(g_a)
                    full_reduce(g_b)

                    # Unfuse non-Clifford phases into gadgets
                    # This ensures all interior Z-spiders have Clifford phases
                    unfuse_non_cliffords(g_a)
                    unfuse_non_cliffords(g_b)

                    # Compose using native compose
                    g_comp = g_a.copy()
                    g_comp.compose(g_b.copy())

                    # Run spider_simp to clean up
                    spider_simp(g_comp)

                    # Check and fix graph-like property
                    try:
                        to_graph_like(g_comp)
                    except:
                        results["to_graph_like_failed"] += 1

                    # Full reduce
                    full_reduce(g_comp)

                    gl = is_graph_like(g_comp)
                    results[f"graph_like={gl}"] += 1

                    # Extract
                    try:
                        c_out = extract_circuit(g_comp.copy())
                        results["extract_success"] += 1
                    except Exception as e:
                        results[f"extract_fail: {str(e)[:50]}"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:50]}"] += 1

    total_attempts = sum(v for k, v in results.items() if "extract" in k or "fail" in k.lower())
    successes = results.get("extract_success", 0)
    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    if total_attempts > 0:
        print(f"  Success rate: {successes/total_attempts*100:.1f}%")
    return results


# =========================================================================
# Experiment 10: Use extract_simple for causal-flow graphs
# =========================================================================
def experiment10_extract_simple():
    separator("EXP 10: extract_simple on spider-fused (not fully reduced) composed graphs")

    results = Counter()

    for n_qubits in [2, 3, 4]:
        for seed_a in range(5):
            for seed_b in range(5):
                try:
                    c_a = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_a)
                    c_b = CNOT_HAD_PHASE_circuit(n_qubits, 10, seed=seed_b + 100)

                    # Compose as circuits (circuit-like, has causal flow)
                    g = c_a.to_graph()
                    g.compose(c_b.to_graph())

                    # Only do spider fusion (preserves causal flow)
                    spider_simp(g)

                    # Try extract_simple
                    try:
                        c_out = extract_simple(g.copy())
                        results["extract_simple_success"] += 1
                    except Exception as e:
                        results[f"extract_simple_fail: {str(e)[:50]}"] += 1

                    # Also try the standard path for comparison
                    g2 = c_a.to_graph()
                    g2.compose(c_b.to_graph())
                    full_reduce(g2)
                    try:
                        c_out2 = extract_circuit(g2.copy())
                        results["standard_success"] += 1
                    except:
                        results["standard_fail"] += 1

                except Exception as e:
                    results[f"error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    return results


# =========================================================================
# Experiment 11: The critical test -- compose sub-diagrams from mining
#   (simulating what our pipeline does)
# =========================================================================
def experiment11_simulated_mining_compose():
    separator("EXP 11: Simulated mining composition (extract sub-diagrams, compose, extract)")

    results = Counter()

    for n_qubits in [3, 4]:
        for seed in range(5):
            try:
                # Create a "source" circuit
                c = CNOT_HAD_PHASE_circuit(n_qubits, 30, seed=seed)
                g = c.to_graph()
                full_reduce(g)

                # Simulate "mining" by splitting the graph
                # We'll take the first half of interior vertices as sub-diagram A
                # and the rest as sub-diagram B

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

                # Build sub-diagram A: vertices in set_a plus their boundary connections
                sub_a = zx.Graph()
                a_map = {}
                a_boundary_in = []
                a_boundary_out = []

                # Add set_a vertices
                for v in set_a:
                    w = sub_a.add_vertex(ty=g.type(v), phase=g.phase(v),
                                         row=g.row(v), qubit=g.qubit(v))
                    a_map[v] = w

                # Add boundary vertices for connections to inputs
                for inp in inputs:
                    for nb in g.neighbors(inp):
                        if nb in set_a:
                            bnd = sub_a.add_vertex(ty=VertexType.BOUNDARY,
                                                    row=g.row(inp), qubit=g.qubit(inp))
                            et = g.edge_type(g.edge(inp, nb))
                            sub_a.add_edge((bnd, a_map[nb]), edgetype=et)
                            a_boundary_in.append(bnd)

                # Add edges between set_a vertices
                for v in set_a:
                    for nb in g.neighbors(v):
                        if nb in set_a and v < nb:
                            et = g.edge_type(g.edge(v, nb))
                            sub_a.add_edge((a_map[v], a_map[nb]), edgetype=et)

                # Add boundary vertices for connections to set_b
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

                if not a_boundary_in or not a_boundary_out:
                    results["no_boundary_wires"] += 1
                    continue

                sub_a.set_inputs(tuple(a_boundary_in))
                sub_a.set_outputs(tuple(a_boundary_out))

                # Similarly build sub_b
                sub_b = zx.Graph()
                b_map = {}
                b_boundary_in = []
                b_boundary_out = []

                for v in set_b:
                    w = sub_b.add_vertex(ty=g.type(v), phase=g.phase(v),
                                         row=g.row(v), qubit=g.qubit(v))
                    b_map[v] = w

                # Boundary vertices for connections from set_a (inputs to sub_b)
                for (v_a, v_b, et) in a_to_b_wires:
                    bnd = sub_b.add_vertex(ty=VertexType.BOUNDARY,
                                            row=g.row(v_b)-0.5, qubit=g.qubit(v_b))
                    sub_b.add_edge((bnd, b_map[v_b]), edgetype=et)
                    b_boundary_in.append(bnd)

                # Edges between set_b vertices
                for v in set_b:
                    for nb in g.neighbors(v):
                        if nb in set_b and v < nb:
                            et = g.edge_type(g.edge(v, nb))
                            sub_b.add_edge((b_map[v], b_map[nb]), edgetype=et)

                # Boundary vertices for connections to outputs
                for outp in outputs:
                    for nb in g.neighbors(outp):
                        if nb in set_b:
                            bnd = sub_b.add_vertex(ty=VertexType.BOUNDARY,
                                                    row=g.row(outp), qubit=g.qubit(outp))
                            et = g.edge_type(g.edge(outp, nb))
                            sub_b.add_edge((bnd, b_map[nb]), edgetype=et)
                            b_boundary_out.append(bnd)

                if not b_boundary_in or not b_boundary_out:
                    results["no_boundary_wires_b"] += 1
                    continue

                sub_b.set_inputs(tuple(b_boundary_in))
                sub_b.set_outputs(tuple(b_boundary_out))

                results[f"sub_a_graph_like={is_graph_like(sub_a)}"] += 1
                results[f"sub_b_graph_like={is_graph_like(sub_b)}"] += 1

                # Now compose sub_a and sub_b
                try:
                    composed = sub_a.copy()
                    composed.compose(sub_b.copy())
                    results["compose_ok"] += 1
                except Exception as e:
                    results[f"compose_error: {str(e)[:50]}"] += 1
                    continue

                # Try various extraction methods
                extracted = False

                # Method 1: full_reduce + extract
                try:
                    g2 = composed.copy()
                    full_reduce(g2)
                    c_out = extract_circuit(g2.copy())
                    results["method1_fr_extract"] += 1
                    extracted = True
                except:
                    pass

                # Method 2: to_graph_like + full_reduce + extract
                if not extracted:
                    try:
                        g3 = composed.copy()
                        to_graph_like(g3)
                        full_reduce(g3)
                        c_out = extract_circuit(g3.copy())
                        results["method2_togl_fr_extract"] += 1
                        extracted = True
                    except:
                        pass

                # Method 3: spider_simp + extract_simple
                if not extracted:
                    try:
                        g4 = composed.copy()
                        spider_simp(g4)
                        c_out = extract_simple(g4.copy())
                        results["method3_extract_simple"] += 1
                        extracted = True
                    except:
                        pass

                if not extracted:
                    results["all_methods_failed"] += 1

            except Exception as e:
                results[f"outer_error: {str(e)[:50]}"] += 1

    print(f"  Results:")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v}")
    return results


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("ZX-Webs Extraction Experiments")
    print(f"PyZX version: {zx.__version__ if hasattr(zx, '__version__') else 'unknown'}")

    all_results = {}

    all_results["exp1"] = experiment1_baseline()
    all_results["exp2"] = experiment2_pyzx_compose()
    all_results["exp3"] = experiment3_compose_reduced()
    all_results["exp4"] = experiment4_manual_compose_with_boundaries()
    all_results["exp5"] = experiment5_gflow_check()
    all_results["exp6"] = experiment6_spider_fusion_compose()
    all_results["exp7"] = experiment7_compose_before_reduce()
    all_results["exp8"] = experiment8_diagnose_failures()
    all_results["exp9"] = experiment9_unfuse_rereduce()
    all_results["exp10"] = experiment10_extract_simple()
    all_results["exp11"] = experiment11_simulated_mining_compose()

    separator("SUMMARY")
    print("All experiments completed.")
    for name, result in all_results.items():
        print(f"  {name}: {result}")
