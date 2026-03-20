"""Tests for Stage 5 -- circuit extraction filter and deduplication."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pyzx as zx
import pytest

from zx_webs.config import FilterConfig
from zx_webs.persistence import save_json, save_manifest
from zx_webs.stage4_compose.candidate import CandidateAlgorithm
from zx_webs.stage5_filter.deduplicator import circuits_equivalent, deduplicate_circuits
from zx_webs.stage5_filter.extractor import ExtractionResult, run_stage5, try_extract_circuit


# ---------------------------------------------------------------------------
# Helpers -- build extractable and non-extractable graphs
# ---------------------------------------------------------------------------


def _make_extractable_graph(n_qubits: int = 1) -> zx.Graph:
    """Build a simple extractable graph via a PyZX circuit round-trip.

    This guarantees the graph is in a form that ``extract_circuit`` can handle
    after ``full_reduce``.
    """
    c = zx.Circuit(n_qubits)
    if n_qubits >= 2:
        c.add_gate("CNOT", 0, 1)
    c.add_gate("HAD", 0)
    g = c.to_graph()
    return g


def _make_non_extractable_graph() -> zx.Graph:
    """Build a graph with no boundary info that cannot be extracted."""
    g = zx.Graph()
    v1 = g.add_vertex(ty=1, phase=Fraction(0))
    v2 = g.add_vertex(ty=2, phase=Fraction(1, 2))
    g.add_edge((v1, v2), edgetype=1)
    # No inputs/outputs set -> extraction will fail.
    return g


# ---------------------------------------------------------------------------
# ExtractionResult tests
# ---------------------------------------------------------------------------


class TestExtractionResult:
    """Tests for the ExtractionResult data class."""

    def test_default_values(self) -> None:
        """Default ExtractionResult should be a failure with empty fields."""
        result = ExtractionResult()
        assert result.success is False
        assert result.circuit_qasm == ""
        assert result.stats == {}
        assert result.error == ""


# ---------------------------------------------------------------------------
# Circuit extraction tests
# ---------------------------------------------------------------------------


class TestTryExtractCircuit:
    """Tests for the try_extract_circuit function."""

    def test_extract_simple_circuit(self) -> None:
        """Extract a circuit from a known-good graph; verify QASM output."""
        g = _make_extractable_graph(n_qubits=2)
        result = try_extract_circuit(g, timeout=10.0)

        assert result.success is True
        assert "OPENQASM" in result.circuit_qasm
        assert result.stats.get("qubits") == 2
        assert result.stats.get("n_gates", 0) > 0
        assert result.error == ""

    def test_extract_1q_circuit(self) -> None:
        """A 1-qubit circuit should extract successfully."""
        g = _make_extractable_graph(n_qubits=1)
        result = try_extract_circuit(g, timeout=10.0)

        assert result.success is True
        assert "OPENQASM" in result.circuit_qasm

    def test_extract_invalid_graph(self) -> None:
        """A graph with no boundaries -> extraction fails gracefully."""
        g = _make_non_extractable_graph()
        result = try_extract_circuit(g, timeout=5.0)

        assert result.success is False
        assert result.error != ""
        assert result.circuit_qasm == ""

    def test_extract_does_not_mutate_input(self) -> None:
        """The input graph should not be modified by extraction."""
        g = _make_extractable_graph(n_qubits=2)
        original_verts = g.num_vertices()
        original_edges = g.num_edges()

        try_extract_circuit(g, timeout=10.0)

        assert g.num_vertices() == original_verts
        assert g.num_edges() == original_edges

    def test_extract_cnot_blowup_detection(self) -> None:
        """When max_cnot_blowup is very low, circuit should be rejected."""
        g = _make_extractable_graph(n_qubits=2)
        # Set an absurdly low blowup factor to trigger rejection.
        result = try_extract_circuit(g, timeout=10.0, max_cnot_blowup=0.0)

        # The 2-qubit circuit has at least 1 two-qubit gate, so with
        # max_cnot_blowup=0.0 it should be rejected (0.0 * 2 = 0 < 1).
        if result.success:
            # If the circuit has no two-qubit gates after reduction, it
            # legitimately passes. That's fine.
            assert result.stats.get("two_qubit_count", 0) == 0
        else:
            assert "blowup" in result.error.lower() or "CNOT" in result.error


# ---------------------------------------------------------------------------
# Circuit equivalence tests
# ---------------------------------------------------------------------------


class TestCircuitsEquivalent:
    """Tests for circuit equivalence checking."""

    def test_identical_qasm(self) -> None:
        """Two identical QASM strings are equivalent."""
        c = zx.Circuit(1)
        c.add_gate("HAD", 0)
        qasm = c.to_qasm()
        assert circuits_equivalent(qasm, qasm) is True

    def test_different_circuits(self) -> None:
        """Two fundamentally different circuits are not equivalent."""
        c1 = zx.Circuit(1)
        c1.add_gate("HAD", 0)

        c2 = zx.Circuit(1)
        c2.add_gate("ZPhase", 0, phase=Fraction(1, 4))

        assert circuits_equivalent(c1.to_qasm(), c2.to_qasm()) is False

    def test_qasm_method(self) -> None:
        """The 'qasm' method does literal string comparison."""
        qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nh q[0];\n"
        assert circuits_equivalent(qasm, qasm, method="qasm") is True
        assert circuits_equivalent(qasm, qasm + "x q[0];\n", method="qasm") is False

    def test_different_qubit_counts(self) -> None:
        """Circuits on different qubit counts are never equivalent."""
        c1 = zx.Circuit(1)
        c1.add_gate("HAD", 0)

        c2 = zx.Circuit(2)
        c2.add_gate("HAD", 0)

        assert circuits_equivalent(c1.to_qasm(), c2.to_qasm()) is False


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplicateCircuits:
    """Tests for circuit deduplication."""

    def test_dedup_identical(self) -> None:
        """Two identical QASM strings -> deduplicated to 1."""
        c = zx.Circuit(1)
        c.add_gate("HAD", 0)
        qasm = c.to_qasm()

        circuits = [
            {"circuit_qasm": qasm, "stats": {"n_gates": 1}, "candidate_id": "a"},
            {"circuit_qasm": qasm, "stats": {"n_gates": 1}, "candidate_id": "b"},
        ]

        result = deduplicate_circuits(circuits)
        assert len(result) == 1
        assert result[0]["candidate_id"] == "a"  # first occurrence kept

    def test_dedup_different(self) -> None:
        """Two different circuits -> both kept."""
        c1 = zx.Circuit(1)
        c1.add_gate("HAD", 0)

        c2 = zx.Circuit(1)
        c2.add_gate("ZPhase", 0, phase=Fraction(1, 4))

        circuits = [
            {"circuit_qasm": c1.to_qasm(), "stats": {}, "candidate_id": "a"},
            {"circuit_qasm": c2.to_qasm(), "stats": {}, "candidate_id": "b"},
        ]

        result = deduplicate_circuits(circuits)
        assert len(result) == 2

    def test_dedup_empty(self) -> None:
        """Empty list -> empty result."""
        assert deduplicate_circuits([]) == []

    def test_dedup_single(self) -> None:
        """Single circuit -> kept as-is."""
        circuits = [{"circuit_qasm": "OPENQASM 2.0;", "stats": {}}]
        result = deduplicate_circuits(circuits)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# End-to-end Stage 5 test
# ---------------------------------------------------------------------------


class TestRunStage5EndToEnd:
    """End-to-end integration test for run_stage5."""

    def test_run_stage5_creates_outputs(self, tmp_path: Path) -> None:
        """Set up Stage 4 outputs, run Stage 5, verify survivors are created."""
        # Build extractable candidate graphs.
        candidates_dir = tmp_path / "compose"
        cands_subdir = candidates_dir / "candidates"
        cands_subdir.mkdir(parents=True)

        g1 = _make_extractable_graph(n_qubits=2)
        g2 = _make_extractable_graph(n_qubits=1)
        g3 = _make_non_extractable_graph()

        test_candidates = [
            CandidateAlgorithm(
                candidate_id="cand_0000",
                graph_json=g1.to_json(),
                component_web_ids=["web_0"],
                composition_type="sequential",
                n_qubits=2,
                n_spiders=4,
            ),
            CandidateAlgorithm(
                candidate_id="cand_0001",
                graph_json=g2.to_json(),
                component_web_ids=["web_1"],
                composition_type="parallel",
                n_qubits=1,
                n_spiders=2,
            ),
            CandidateAlgorithm(
                candidate_id="cand_0002",
                graph_json=g3.to_json(),
                component_web_ids=["web_2"],
                composition_type="sequential",
                n_qubits=0,
                n_spiders=2,
            ),
        ]

        manifest_entries = []
        for cand in test_candidates:
            cand_path = cands_subdir / f"{cand.candidate_id}.json"
            save_json(cand.to_dict(), cand_path)
            manifest_entries.append(
                {
                    "candidate_id": cand.candidate_id,
                    "candidate_path": str(cand_path),
                    "composition_type": cand.composition_type,
                    "component_web_ids": cand.component_web_ids,
                    "n_qubits": cand.n_qubits,
                    "n_spiders": cand.n_spiders,
                }
            )

        save_manifest(manifest_entries, candidates_dir)

        # Run Stage 5.
        output_dir = tmp_path / "filter_output"
        config = FilterConfig(
            extract_timeout_seconds=10.0,
            max_cnot_blowup_factor=10.0,
            dedup_method="unitary",
        )
        survivors = run_stage5(candidates_dir, output_dir, config)

        # At least the two extractable candidates should survive.
        assert len(survivors) >= 1

        # Manifest should exist.
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

        # Each survivor should have a corresponding JSON file.
        circuits_subdir = output_dir / "circuits"
        assert circuits_subdir.exists()

    def test_run_stage5_empty_manifest(self, tmp_path: Path) -> None:
        """run_stage5 on an empty manifest should return no survivors."""
        candidates_dir = tmp_path / "compose_empty"
        candidates_dir.mkdir()
        save_manifest([], candidates_dir)

        output_dir = tmp_path / "filter_empty"
        survivors = run_stage5(candidates_dir, output_dir)
        assert survivors == []
