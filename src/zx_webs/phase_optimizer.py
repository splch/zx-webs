"""Phase optimisation for near-miss candidates via gradient-free search.

When a candidate circuit has high fidelity (e.g. 0.80-0.99) to a known
benchmark task but isn't an exact match, targeted phase optimisation can
nudge it into an exact functional match while potentially reducing gate
count.

The optimiser treats each interior spider's phase as a free variable and
uses Nelder-Mead (scipy if available, otherwise a pure-NumPy fallback)
to maximise process fidelity against the target unitary.

This is cheap: unitary computation for <=10 qubits takes <1ms, so
100-200 iterations of Nelder-Mead complete in well under a second per
candidate.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any

import numpy as np
import pyzx as zx

from zx_webs.stage6_bench.metrics import compute_unitary, process_fidelity

logger = logging.getLogger(__name__)

_VT_BOUNDARY = 0
_VT_Z = 1
_VT_X = 2


def _get_interior_spiders(graph: zx.Graph) -> list[int]:
    """Return vertex IDs of interior Z/X spiders (not boundaries)."""
    boundary_set = set(graph.inputs() or []) | set(graph.outputs() or [])
    return [
        v for v in graph.vertices()
        if graph.type(v) in (_VT_Z, _VT_X) and v not in boundary_set
    ]


def _set_phases(graph: zx.Graph, spider_ids: list[int], phases: np.ndarray) -> None:
    """Set the phases of the given spiders from a numpy array.

    Phases are in units of pi (matching PyZX convention).
    Values are taken modulo 2 to stay in [0, 2pi).
    """
    for vid, phase_val in zip(spider_ids, phases):
        graph.set_phase(vid, Fraction(phase_val % 2.0).limit_denominator(1024))


def _graph_to_qasm(graph: zx.Graph) -> str | None:
    """Try to extract a QASM string from a ZX graph."""
    try:
        g = graph.copy()
        zx.full_reduce(g)
        c = zx.extract_circuit(g)
        return c.to_qasm()
    except Exception:
        return None


def _fidelity_from_phases(
    base_graph: zx.Graph,
    spider_ids: list[int],
    phases: np.ndarray,
    target_unitary: np.ndarray,
    max_qubits: int = 10,
) -> float:
    """Compute process fidelity for a given phase assignment.

    Returns negative fidelity (for minimisation).
    """
    g = base_graph.copy()
    _set_phases(g, spider_ids, phases)
    qasm = _graph_to_qasm(g)
    if qasm is None:
        return -0.0  # extraction failed
    u = compute_unitary(qasm, max_unitary_qubits=max_qubits)
    if u is None:
        return -0.0
    return -process_fidelity(u, target_unitary)


def _nelder_mead_minimize(
    func,
    x0: np.ndarray,
    maxiter: int = 200,
    tol: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Minimise *func* using Nelder-Mead (pure NumPy fallback).

    Returns (best_x, best_value).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(x0)
    # Build initial simplex: x0 + small perturbations.
    simplex = np.empty((n + 1, n), dtype=np.float64)
    simplex[0] = x0
    for i in range(n):
        point = x0.copy()
        point[i] += 0.1 + rng.uniform(-0.02, 0.02)
        simplex[i + 1] = point

    values = np.array([func(simplex[i]) for i in range(n + 1)])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(maxiter):
        # Sort
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        # Check convergence
        if np.max(np.abs(values - values[0])) < tol:
            break

        # Centroid of all but worst
        centroid = simplex[:-1].mean(axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)

        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
            continue

        # Expansion
        if fr < values[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
            continue

        # Contraction
        xc = centroid + rho * (simplex[-1] - centroid)
        fc = func(xc)
        if fc < values[-1]:
            simplex[-1] = xc
            values[-1] = fc
            continue

        # Shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            values[i] = func(simplex[i])

    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


def optimize_phases(
    graph_json: str,
    target_unitary: np.ndarray,
    max_iterations: int = 200,
    max_qubits: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Optimise spider phases to maximise fidelity to a target unitary.

    Parameters
    ----------
    graph_json:
        PyZX graph serialised as JSON.
    target_unitary:
        The target unitary matrix.
    max_iterations:
        Maximum Nelder-Mead iterations.
    max_qubits:
        Maximum qubit count for unitary computation.
    seed:
        Random seed.

    Returns
    -------
    dict with keys:
        - ``"success"``: bool — whether fidelity improved
        - ``"initial_fidelity"``: float
        - ``"optimized_fidelity"``: float
        - ``"optimized_qasm"``: str — QASM of the optimised circuit (empty if failed)
        - ``"optimized_graph_json"``: str — optimised graph JSON (empty if failed)
        - ``"n_phases_optimized"``: int
    """
    rng = np.random.default_rng(seed)

    try:
        base_graph = zx.Graph.from_json(graph_json)
    except Exception:
        return {
            "success": False, "initial_fidelity": 0.0,
            "optimized_fidelity": 0.0, "optimized_qasm": "",
            "optimized_graph_json": "", "n_phases_optimized": 0,
        }

    spider_ids = _get_interior_spiders(base_graph)
    if not spider_ids:
        return {
            "success": False, "initial_fidelity": 0.0,
            "optimized_fidelity": 0.0, "optimized_qasm": "",
            "optimized_graph_json": "", "n_phases_optimized": 0,
        }

    # Initial phases.
    x0 = np.array([float(base_graph.phase(v)) for v in spider_ids], dtype=np.float64)

    # Compute initial fidelity.
    initial_neg_fid = _fidelity_from_phases(
        base_graph, spider_ids, x0, target_unitary, max_qubits,
    )
    initial_fidelity = -initial_neg_fid

    # Try scipy first, fall back to pure NumPy.
    try:
        from scipy.optimize import minimize as scipy_minimize

        result = scipy_minimize(
            lambda phases: _fidelity_from_phases(
                base_graph, spider_ids, phases, target_unitary, max_qubits,
            ),
            x0,
            method="Nelder-Mead",
            options={"maxiter": max_iterations, "xatol": 1e-6, "fatol": 1e-8},
        )
        best_phases = result.x
        best_neg_fid = result.fun
    except ImportError:
        best_phases, best_neg_fid = _nelder_mead_minimize(
            lambda phases: _fidelity_from_phases(
                base_graph, spider_ids, phases, target_unitary, max_qubits,
            ),
            x0,
            maxiter=max_iterations,
            rng=rng,
        )

    optimized_fidelity = -best_neg_fid

    if optimized_fidelity <= initial_fidelity + 1e-10:
        return {
            "success": False,
            "initial_fidelity": initial_fidelity,
            "optimized_fidelity": optimized_fidelity,
            "optimized_qasm": "",
            "optimized_graph_json": "",
            "n_phases_optimized": len(spider_ids),
        }

    # Build the optimised graph and extract QASM.
    opt_graph = base_graph.copy()
    _set_phases(opt_graph, spider_ids, best_phases)
    opt_qasm = _graph_to_qasm(opt_graph) or ""
    opt_graph_json = opt_graph.to_json() if opt_qasm else ""

    return {
        "success": True,
        "initial_fidelity": initial_fidelity,
        "optimized_fidelity": optimized_fidelity,
        "optimized_qasm": opt_qasm,
        "optimized_graph_json": opt_graph_json,
        "n_phases_optimized": len(spider_ids),
    }


def optimize_near_misses(
    near_miss_candidates: list[dict[str, Any]],
    filtered_dir,
    corpus_dir,
    max_iterations: int = 200,
    fidelity_threshold: float = 0.99,
    max_qubits: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Optimise phases for all near-miss candidates.

    Loads each near-miss candidate's circuit, finds the target task's
    unitary, and runs phase optimisation.

    Parameters
    ----------
    near_miss_candidates:
        From FitnessProfile.near_miss_candidates.
    filtered_dir:
        Path to Stage 5 filtered output.
    corpus_dir:
        Path to Stage 1 corpus (for loading target unitaries).
    max_iterations:
        Nelder-Mead iterations per candidate.
    fidelity_threshold:
        Fidelity above which we consider the optimisation a success.
    max_qubits:
        Max qubits for unitary computation.
    seed:
        Random seed.

    Returns
    -------
    list[dict]
        Optimisation results for each near-miss candidate.
    """
    from pathlib import Path

    from zx_webs.persistence import load_json, load_manifest
    from zx_webs.stage6_bench.tasks import build_benchmark_tasks

    filtered_dir = Path(filtered_dir)
    corpus_dir = Path(corpus_dir)

    if not near_miss_candidates:
        return []

    # Build target unitaries.
    qubit_counts = sorted(set(
        nm.get("n_qubits", 0) for nm in near_miss_candidates if nm.get("n_qubits", 0) > 0
    ))
    try:
        tasks = build_benchmark_tasks(
            qubit_counts=qubit_counts, max_unitary_qubits=max_qubits,
        )
    except Exception:
        logger.warning("Failed to build benchmark tasks for phase optimisation.")
        return []

    task_lookup: dict[str, Any] = {t.name: t for t in tasks}

    # Load survivor data.
    filtered_manifest = load_manifest(filtered_dir)
    surv_lookup: dict[str, dict] = {}
    for entry in filtered_manifest:
        sid = entry.get("survivor_id", "")
        surv_lookup[sid] = entry

    results = []
    for nm in near_miss_candidates:
        sid = nm.get("survivor_id", "")
        target_name = nm.get("target_task", "")

        if target_name not in task_lookup:
            continue

        entry = surv_lookup.get(sid)
        if not entry:
            continue

        # Load graph JSON.
        circuit_path = entry.get("circuit_path", "")
        if not circuit_path or not Path(circuit_path).exists():
            continue

        try:
            circ_data = load_json(Path(circuit_path))
        except Exception:
            continue

        graph_json = circ_data.get("graph_json", "")
        if not graph_json:
            # Fall back: try to get QASM and parse back to graph.
            continue

        target_task = task_lookup[target_name]
        target_unitary = target_task.target_unitary

        logger.info(
            "Phase-optimising %s toward task %s (initial fidelity %.3f)",
            sid, target_name, nm.get("best_fidelity", 0.0),
        )

        opt_result = optimize_phases(
            graph_json=graph_json,
            target_unitary=target_unitary,
            max_iterations=max_iterations,
            max_qubits=max_qubits,
            seed=seed,
        )

        opt_result["survivor_id"] = sid
        opt_result["target_task"] = target_name
        results.append(opt_result)

        if opt_result["success"]:
            logger.info(
                "  -> Improved: fidelity %.4f -> %.4f (%s threshold %.2f)",
                opt_result["initial_fidelity"],
                opt_result["optimized_fidelity"],
                "ABOVE" if opt_result["optimized_fidelity"] >= fidelity_threshold else "below",
                fidelity_threshold,
            )

    n_improved = sum(1 for r in results if r["success"])
    n_above_threshold = sum(
        1 for r in results
        if r["success"] and r["optimized_fidelity"] >= fidelity_threshold
    )
    logger.info(
        "Phase optimisation: %d/%d improved, %d above fidelity threshold %.2f.",
        n_improved, len(results), n_above_threshold, fidelity_threshold,
    )

    return results
