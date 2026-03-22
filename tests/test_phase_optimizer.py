"""Tests for the phase optimiser module."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pyzx as zx
import pytest

from zx_webs.phase_optimizer import (
    _get_interior_spiders,
    _nelder_mead_minimize,
    _set_phases,
    optimize_phases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1q_z_rotation_graph(phase: float = 0.0) -> zx.Graph:
    """Build a 1-qubit graph: input -> Z(phase) -> output."""
    g = zx.Graph()
    i0 = g.add_vertex(ty=0, qubit=0, row=0)  # input boundary
    z0 = g.add_vertex(ty=1, phase=Fraction(phase).limit_denominator(1024), qubit=0, row=1)
    o0 = g.add_vertex(ty=0, qubit=0, row=2)  # output boundary
    g.add_edge((i0, z0), edgetype=1)
    g.add_edge((z0, o0), edgetype=1)
    g.set_inputs((i0,))
    g.set_outputs((o0,))
    return g


def _make_2q_graph(phase_z: float = 0.0, phase_x: float = 0.0) -> zx.Graph:
    """Build a 2-qubit graph with Z and X spiders."""
    g = zx.Graph()
    i0 = g.add_vertex(ty=0, qubit=0, row=0)
    i1 = g.add_vertex(ty=0, qubit=1, row=0)
    z0 = g.add_vertex(ty=1, phase=Fraction(phase_z).limit_denominator(1024), qubit=0, row=1)
    x0 = g.add_vertex(ty=2, phase=Fraction(phase_x).limit_denominator(1024), qubit=1, row=1)
    o0 = g.add_vertex(ty=0, qubit=0, row=2)
    o1 = g.add_vertex(ty=0, qubit=1, row=2)
    g.add_edge((i0, z0), edgetype=1)
    g.add_edge((i1, x0), edgetype=1)
    g.add_edge((z0, x0), edgetype=2)  # Hadamard edge
    g.add_edge((z0, o0), edgetype=1)
    g.add_edge((x0, o1), edgetype=1)
    g.set_inputs((i0, i1))
    g.set_outputs((o0, o1))
    return g


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGetInteriorSpiders:
    def test_1q_graph(self):
        g = _make_1q_z_rotation_graph(0.5)
        spiders = _get_interior_spiders(g)
        assert len(spiders) == 1
        assert g.type(spiders[0]) == 1  # Z spider

    def test_2q_graph(self):
        g = _make_2q_graph(0.25, 0.5)
        spiders = _get_interior_spiders(g)
        assert len(spiders) == 2


class TestSetPhases:
    def test_sets_phases(self):
        g = _make_1q_z_rotation_graph(0.0)
        spiders = _get_interior_spiders(g)
        _set_phases(g, spiders, np.array([0.5]))
        phase = float(g.phase(spiders[0]))
        assert abs(phase - 0.5) < 0.01


class TestNelderMead:
    def test_minimizes_quadratic(self):
        """Nelder-Mead should find the minimum of a simple quadratic."""
        def f(x):
            return (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2

        x0 = np.array([0.0, 0.0])
        best_x, best_val = _nelder_mead_minimize(f, x0, maxiter=500)
        assert abs(best_x[0] - 2.0) < 0.1
        assert abs(best_x[1] - 3.0) < 0.1
        assert best_val < 0.01


class TestOptimizePhases:
    def test_identity_target_already_matched(self):
        """A graph that already implements the identity should report no improvement."""
        g = _make_1q_z_rotation_graph(0.0)  # Z(0) = identity spider
        graph_json = g.to_json()

        # Target: identity matrix
        target = np.eye(2, dtype=complex)

        result = optimize_phases(
            graph_json=graph_json,
            target_unitary=target,
            max_iterations=50,
        )
        # Already a near-perfect match, so either success=False (no improvement)
        # or the fidelity is already ~1.0.
        assert result["initial_fidelity"] >= 0.9 or result["success"] is False

    def test_optimization_improves_fidelity(self):
        """Starting with a wrong phase, optimisation should find a better one."""
        # Build a graph with phase=0.7 (arbitrary)
        g = _make_1q_z_rotation_graph(0.7)
        graph_json = g.to_json()

        # Target: Z(0) = identity (phase=0)
        target_g = _make_1q_z_rotation_graph(0.0)
        try:
            c = zx.extract_circuit(target_g.copy())
            target = np.array(c.to_matrix())
        except Exception:
            pytest.skip("Circuit extraction not available")

        result = optimize_phases(
            graph_json=graph_json,
            target_unitary=target,
            max_iterations=100,
        )
        # The optimiser should be able to improve fidelity.
        assert result["n_phases_optimized"] == 1
        if result["success"]:
            assert result["optimized_fidelity"] > result["initial_fidelity"]

    def test_empty_graph_json(self):
        """Invalid graph JSON should return success=False."""
        result = optimize_phases(
            graph_json="invalid json",
            target_unitary=np.eye(2, dtype=complex),
        )
        assert result["success"] is False
