"""Tests for Stage 2 -- ZX-diagram conversion and simplification."""
from __future__ import annotations

from pathlib import Path

import pyzx as zx
from pyzx.graph.base import BaseGraph
import pytest

from zx_webs.config import ZXConfig
from zx_webs.persistence import save_manifest
from zx_webs.stage2_zx import compute_graph_stats, qasm_to_zx_graph, run_stage2, simplify_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bell_qasm() -> str:
    """Return a minimal QASM 2.0 string for a Bell-state preparation circuit."""
    return (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
    )


def _ghz3_qasm() -> str:
    """Return QASM for a 3-qubit GHZ state preparation."""
    return (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[3];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "cx q[1],q[2];\n"
    )


# ---------------------------------------------------------------------------
# qasm_to_zx_graph tests
# ---------------------------------------------------------------------------


class TestQasmToZxGraph:
    """Tests for the qasm_to_zx_graph converter."""

    def test_bell_state(self) -> None:
        """Converting a Bell-state QASM should return a graph with vertices."""
        g, info = qasm_to_zx_graph(_bell_qasm())

        assert isinstance(g, BaseGraph)
        assert g.num_vertices() > 0
        assert "pre_stats" in info
        assert "post_stats" in info
        assert "reduction_method" in info

    def test_custom_config(self) -> None:
        """qasm_to_zx_graph should respect the provided ZXConfig."""
        config = ZXConfig(reduction="none", normalize=False)
        g, info = qasm_to_zx_graph(_bell_qasm(), config)

        assert info["reduction_method"] == "none"
        assert g.num_vertices() > 0


# ---------------------------------------------------------------------------
# Simplification tests
# ---------------------------------------------------------------------------


class TestSimplifyMethods:
    """Test each simplification method produces a valid graph."""

    @pytest.fixture
    def base_graph(self) -> BaseGraph:
        """Parse a GHZ-3 circuit into an unsimplified ZX graph."""
        circuit = zx.Circuit.from_qasm(_ghz3_qasm())
        return circuit.to_graph()

    @pytest.mark.parametrize(
        "method",
        ["full_reduce", "teleport_reduce", "clifford_simp", "none"],
    )
    def test_simplify_method(self, base_graph: BaseGraph, method: str) -> None:
        """simplify_graph with method={method} should return a valid graph."""
        g_out = simplify_graph(base_graph, method=method)

        assert isinstance(g_out, BaseGraph)
        assert g_out.num_vertices() > 0

        # The original graph should be unmodified (deep copy inside)
        assert base_graph.num_vertices() > 0

    def test_invalid_method_raises(self, base_graph: BaseGraph) -> None:
        """simplify_graph should raise ValueError for unknown methods."""
        with pytest.raises(ValueError, match="Unknown simplification method"):
            simplify_graph(base_graph, method="bogus_method")


# ---------------------------------------------------------------------------
# Graph stats tests
# ---------------------------------------------------------------------------


class TestGraphStats:
    """Tests for compute_graph_stats."""

    def test_stats_on_known_graph(self) -> None:
        """Stats on a Bell-state graph should have expected structure and positive counts."""
        circuit = zx.Circuit.from_qasm(_bell_qasm())
        g = circuit.to_graph()
        stats = compute_graph_stats(g)

        expected_keys = {
            "n_vertices",
            "n_edges",
            "n_z_spiders",
            "n_x_spiders",
            "n_boundary",
            "n_h_boxes",
            "n_simple_edges",
            "n_hadamard_edges",
            "n_inputs",
            "n_outputs",
        }
        assert expected_keys.issubset(stats.keys())
        assert stats["n_vertices"] > 0
        assert stats["n_edges"] > 0
        assert stats["n_boundary"] >= 4  # 2 inputs + 2 outputs for 2-qubit circuit
        assert stats["n_inputs"] == 2
        assert stats["n_outputs"] == 2


# ---------------------------------------------------------------------------
# run_stage2 end-to-end tests
# ---------------------------------------------------------------------------


class TestRunStage2EndToEnd:
    """End-to-end test for the run_stage2 function."""

    def test_run_stage2_creates_outputs(self, tmp_path: Path) -> None:
        """run_stage2 should read a corpus manifest and produce graph files."""
        # Set up a minimal corpus directory with two QASM files
        corpus_dir = tmp_path / "corpus"
        algorithms_dir = corpus_dir / "algorithms" / "entanglement"
        algorithms_dir.mkdir(parents=True)

        bell_path = algorithms_dir / "bell_2q.qasm"
        bell_path.write_text(_bell_qasm())

        ghz_path = algorithms_dir / "ghz_3q.qasm"
        ghz_path.write_text(_ghz3_qasm())

        # Write a corpus manifest
        corpus_manifest = [
            {
                "algorithm_id": "entanglement/bell_q2",
                "family": "entanglement",
                "name": "bell",
                "n_qubits": 2,
                "qasm_path": str(bell_path),
            },
            {
                "algorithm_id": "entanglement/ghz_q3",
                "family": "entanglement",
                "name": "ghz",
                "n_qubits": 3,
                "qasm_path": str(ghz_path),
            },
        ]
        save_manifest(corpus_manifest, corpus_dir)

        # Run Stage 2
        output_dir = tmp_path / "zx_diagrams"
        config = ZXConfig(reduction="full_reduce", normalize=True)
        entries = run_stage2(corpus_dir, output_dir, config)

        # Verify outputs
        assert len(entries) == 2

        # Check output manifest exists
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

        # Check graph files exist
        for entry in entries:
            graph_path = Path(entry["graph_path"])
            assert graph_path.exists(), f"Graph file missing: {graph_path}"
            assert entry["pre_stats"]["n_vertices"] > 0
            assert entry["post_stats"]["n_vertices"] > 0
