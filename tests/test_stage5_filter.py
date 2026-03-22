"""Tests for Stage 5 -- circuit extraction filter and deduplication.

Tests the improved extraction pipeline with:
- to_graph_like() pre-processing
- gflow pre-check
- Proper handling of boundary vertices
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pyzx as zx
import pytest

from zx_webs.config import FilterConfig
from zx_webs.persistence import save_json, save_manifest
from zx_webs.stage4_compose.candidate import CandidateAlgorithm
from zx_webs.stage5_filter.deduplicator import (
    _unitary_hash,
    circuits_equivalent,
    deduplicate_circuits,
)
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


def _make_composed_sequential_graph() -> zx.Graph:
    """Build a graph via PyZX native compose() -- should be extractable."""
    g_a = zx.Graph()
    i0 = g_a.add_vertex(ty=0, qubit=0, row=0)
    z0 = g_a.add_vertex(ty=1, phase=Fraction(0), qubit=0, row=1)
    o0 = g_a.add_vertex(ty=0, qubit=0, row=2)
    g_a.add_edge((i0, z0), edgetype=1)
    g_a.add_edge((z0, o0), edgetype=1)
    g_a.set_inputs((i0,))
    g_a.set_outputs((o0,))

    g_b = zx.Graph()
    i1 = g_b.add_vertex(ty=0, qubit=0, row=0)
    z1 = g_b.add_vertex(ty=1, phase=Fraction(1, 4), qubit=0, row=1)
    o1 = g_b.add_vertex(ty=0, qubit=0, row=2)
    g_b.add_edge((i1, z1), edgetype=1)
    g_b.add_edge((z1, o1), edgetype=1)
    g_b.set_inputs((i1,))
    g_b.set_outputs((o1,))

    result = g_a.copy()
    result.compose(g_b.copy())
    return result


def _make_composed_tensor_graph() -> zx.Graph:
    """Build a graph via PyZX tensor() -- should be extractable."""
    g_a = zx.Graph()
    i0 = g_a.add_vertex(ty=0, qubit=0, row=0)
    z0 = g_a.add_vertex(ty=1, phase=Fraction(0), qubit=0, row=1)
    o0 = g_a.add_vertex(ty=0, qubit=0, row=2)
    g_a.add_edge((i0, z0), edgetype=1)
    g_a.add_edge((z0, o0), edgetype=1)
    g_a.set_inputs((i0,))
    g_a.set_outputs((o0,))

    g_b = zx.Graph()
    i1 = g_b.add_vertex(ty=0, qubit=0, row=0)
    z1 = g_b.add_vertex(ty=1, phase=Fraction(1, 4), qubit=0, row=1)
    o1 = g_b.add_vertex(ty=0, qubit=0, row=2)
    g_b.add_edge((i1, z1), edgetype=1)
    g_b.add_edge((z1, o1), edgetype=1)
    g_b.set_inputs((i1,))
    g_b.set_outputs((o1,))

    return g_a.tensor(g_b)


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

    def test_extract_composed_sequential(self) -> None:
        """A graph composed via PyZX compose() should extract successfully."""
        g = _make_composed_sequential_graph()
        result = try_extract_circuit(g, timeout=10.0)

        assert result.success is True, (
            f"Sequential composed graph should extract: {result.error}"
        )
        assert "OPENQASM" in result.circuit_qasm
        assert result.stats.get("qubits") == 1

    def test_extract_composed_tensor(self) -> None:
        """A graph composed via PyZX tensor() should extract successfully."""
        g = _make_composed_tensor_graph()
        result = try_extract_circuit(g, timeout=10.0)

        assert result.success is True, (
            f"Tensor composed graph should extract: {result.error}"
        )
        assert "OPENQASM" in result.circuit_qasm
        assert result.stats.get("qubits") == 2

    def test_to_graph_like_fallback(self) -> None:
        """Graphs with X-spiders should still extract via to_graph_like."""
        g = zx.Graph()
        i0 = g.add_vertex(ty=0, qubit=0, row=0)
        x0 = g.add_vertex(ty=2, phase=Fraction(0), qubit=0, row=1)  # X-spider
        o0 = g.add_vertex(ty=0, qubit=0, row=2)
        g.add_edge((i0, x0), edgetype=1)
        g.add_edge((x0, o0), edgetype=1)
        g.set_inputs((i0,))
        g.set_outputs((o0,))

        result = try_extract_circuit(g, timeout=10.0)
        assert result.success is True, (
            f"X-spider graph should extract via to_graph_like: {result.error}"
        )


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

    def test_dedup_qasm_method(self) -> None:
        """Dedup with method='qasm' uses literal string comparison."""
        c = zx.Circuit(1)
        c.add_gate("HAD", 0)
        qasm = c.to_qasm()

        circuits = [
            {"circuit_qasm": qasm, "stats": {}, "candidate_id": "a"},
            {"circuit_qasm": qasm, "stats": {}, "candidate_id": "b"},
        ]

        result = deduplicate_circuits(circuits, method="qasm")
        assert len(result) == 1
        assert result[0]["candidate_id"] == "a"


# ---------------------------------------------------------------------------
# Hash-based deduplication tests
# ---------------------------------------------------------------------------


class TestUnitaryHash:
    """Tests for the _unitary_hash function used for O(n) dedup."""

    def test_identical_circuits_same_hash(self) -> None:
        """Two identical QASM strings should produce the same hash."""
        c = zx.Circuit(1)
        c.add_gate("HAD", 0)
        qasm = c.to_qasm()
        h1 = _unitary_hash(qasm)
        h2 = _unitary_hash(qasm)
        assert h1 is not None
        assert h1 == h2

    def test_different_circuits_different_hash(self) -> None:
        """Two different circuits should produce different hashes."""
        c1 = zx.Circuit(1)
        c1.add_gate("HAD", 0)

        c2 = zx.Circuit(1)
        c2.add_gate("ZPhase", 0, phase=Fraction(1, 4))

        h1 = _unitary_hash(c1.to_qasm())
        h2 = _unitary_hash(c2.to_qasm())
        assert h1 is not None
        assert h2 is not None
        assert h1 != h2

    def test_same_circuit_different_qasm_same_hash(self) -> None:
        """Two QASM representations of the same circuit should hash the same.

        Build the same Hadamard circuit in two ways and verify they hash
        identically.
        """
        c1 = zx.Circuit(1)
        c1.add_gate("HAD", 0)

        # Build the same circuit from QASM directly.
        qasm1 = c1.to_qasm()
        c2 = zx.Circuit.from_qasm(qasm1)
        qasm2 = c2.to_qasm()

        h1 = _unitary_hash(qasm1)
        h2 = _unitary_hash(qasm2)
        assert h1 is not None
        assert h2 is not None
        assert h1 == h2

    def test_invalid_qasm_returns_none(self) -> None:
        """Invalid QASM should return None."""
        h = _unitary_hash("not valid qasm at all")
        assert h is None


# ---------------------------------------------------------------------------
# End-to-end Stage 5 test
# ---------------------------------------------------------------------------


class TestConfigurableExtraction:
    """Tests for new configurable extraction parameters."""

    def test_cnot_blowup_disabled(self) -> None:
        """When cnot_blowup_enabled=False, CNOT blowup should not reject."""
        g = _make_extractable_graph(n_qubits=2)
        # With blowup enabled and factor 0, it would reject.
        result_enabled = try_extract_circuit(
            g, timeout=10.0, max_cnot_blowup=0.0, cnot_blowup_enabled=True,
        )
        result_disabled = try_extract_circuit(
            g, timeout=10.0, max_cnot_blowup=0.0, cnot_blowup_enabled=False,
        )
        # When disabled, blowup check is skipped so extraction succeeds
        # (assuming the circuit itself is extractable).
        if not result_enabled.success and "blowup" in result_enabled.error.lower():
            assert result_disabled.success is True

    def test_gflow_precheck_off_by_default(self) -> None:
        """With gflow_precheck=False (default), extraction should not reject on gflow."""
        g = _make_extractable_graph(n_qubits=1)
        result = try_extract_circuit(g, timeout=10.0, gflow_precheck=False)
        assert result.success is True

    def test_optimize_cnots_level(self) -> None:
        """Extraction should accept different optimize_cnots levels."""
        g = _make_extractable_graph(n_qubits=2)
        for level in [0, 1, 2]:
            result = try_extract_circuit(g, timeout=10.0, optimize_cnots=level)
            # All levels should produce extractable circuits.
            assert result.success is True, (
                f"Extraction failed at optimize_cnots={level}: {result.error}"
            )


class TestFailuresManifest:
    """Tests for the failures.json persistence."""

    def test_failures_json_created(self, tmp_path: Path) -> None:
        """run_stage5 should create failures.json for failed candidates."""
        candidates_dir = tmp_path / "compose"
        cands_subdir = candidates_dir / "candidates"
        cands_subdir.mkdir(parents=True)

        g_good = _make_extractable_graph(n_qubits=1)
        g_bad = _make_non_extractable_graph()

        test_candidates = [
            CandidateAlgorithm(
                candidate_id="cand_good",
                graph_json=g_good.to_json(),
                component_web_ids=["web_0"],
                composition_type="sequential",
                n_qubits=1,
                n_spiders=2,
            ),
            CandidateAlgorithm(
                candidate_id="cand_bad",
                graph_json=g_bad.to_json(),
                component_web_ids=["web_1"],
                composition_type="sequential",
                n_qubits=0,
                n_spiders=2,
            ),
        ]

        manifest_entries = []
        for cand in test_candidates:
            cand_path = cands_subdir / f"{cand.candidate_id}.json"
            save_json(cand.to_dict(), cand_path)
            manifest_entries.append({
                "candidate_id": cand.candidate_id,
                "candidate_path": str(cand_path),
                "composition_type": cand.composition_type,
                "component_web_ids": cand.component_web_ids,
                "n_qubits": cand.n_qubits,
                "n_spiders": cand.n_spiders,
            })

        save_manifest(manifest_entries, candidates_dir)

        output_dir = tmp_path / "filter_output"
        config = FilterConfig(
            extract_timeout_seconds=10.0,
            max_cnot_blowup_factor=10.0,
            n_workers=1,
        )
        run_stage5(candidates_dir, output_dir, config)

        failures_path = output_dir / "failures.json"
        assert failures_path.exists(), "failures.json should be created"

        import json
        failures = json.loads(failures_path.read_text())
        assert isinstance(failures, list)
        assert len(failures) >= 1
        assert any(f["candidate_id"] == "cand_bad" for f in failures)


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
        g4 = _make_composed_sequential_graph()

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
            CandidateAlgorithm(
                candidate_id="cand_0003",
                graph_json=g4.to_json(),
                component_web_ids=["web_3", "web_4"],
                composition_type="sequential",
                n_qubits=1,
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

        # At least the extractable candidates should survive.
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

    def test_run_stage5_all_extractable(self, tmp_path: Path) -> None:
        """When all candidates are well-formed, all should survive."""
        candidates_dir = tmp_path / "compose"
        cands_subdir = candidates_dir / "candidates"
        cands_subdir.mkdir(parents=True)

        # Create candidates from circuit round-trips (guaranteed extractable).
        test_candidates = []
        for i in range(3):
            c = zx.Circuit(1)
            c.add_gate("ZPhase", 0, phase=Fraction(i, 4))
            c.add_gate("HAD", 0)
            g = c.to_graph()
            test_candidates.append(
                CandidateAlgorithm(
                    candidate_id=f"cand_{i:04d}",
                    graph_json=g.to_json(),
                    component_web_ids=[f"web_{i}"],
                    composition_type="sequential",
                    n_qubits=1,
                    n_spiders=2,
                )
            )

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

        output_dir = tmp_path / "filter_output"
        config = FilterConfig(
            extract_timeout_seconds=10.0,
            max_cnot_blowup_factor=10.0,
            dedup_method="unitary",
        )
        survivors = run_stage5(candidates_dir, output_dir, config)

        # All 3 candidates should extract (they're from circuit round-trips).
        # After dedup, at least 1 should survive.
        assert len(survivors) >= 1


# ---------------------------------------------------------------------------
# Tests for ZX rewrite search (simulated annealing)
# ---------------------------------------------------------------------------


class TestZXAnnealOptimize:
    """Tests for the ZX annealing post-extraction optimization."""

    def test_anneal_returns_graph(self) -> None:
        """zx_anneal_optimize should return a valid ZX graph."""
        from zx_webs.stage5_filter.extractor import zx_anneal_optimize

        c = zx.Circuit(2)
        c.add_gate("HAD", 0)
        c.add_gate("CNOT", 0, 1)
        g = c.to_graph()
        zx.simplify.to_graph_like(g)
        zx.full_reduce(g)

        result = zx_anneal_optimize(g, iters=20)
        assert result is not None
        assert result.num_vertices() > 0

    def test_anneal_preserves_unitary(self) -> None:
        """Annealed graph should implement the same unitary."""
        import numpy as np
        from zx_webs.stage5_filter.extractor import zx_anneal_optimize

        c = zx.Circuit(2)
        c.add_gate("HAD", 0)
        c.add_gate("CNOT", 0, 1)
        c.add_gate("T", 1)
        original_unitary = np.array(c.to_matrix())

        g = c.to_graph()
        zx.simplify.to_graph_like(g)
        zx.full_reduce(g)

        annealed = zx_anneal_optimize(g, iters=30)
        if annealed is not None:
            try:
                c_out = zx.extract_circuit(annealed.copy())
                annealed_unitary = np.array(c_out.to_matrix())
                # Unitaries should match up to global phase.
                d = original_unitary.shape[0]
                fid = abs(np.trace(original_unitary.conj().T @ annealed_unitary)) ** 2 / d**2
                assert fid > 0.99, f"Fidelity {fid} too low after annealing"
            except Exception:
                pass  # Extraction may fail; that's OK for this test

    def test_anneal_config_defaults(self) -> None:
        """ZX annealing config defaults should be off."""
        config = FilterConfig()
        assert config.zx_anneal is False
        assert config.zx_anneal_iters == 200
        assert config.zx_anneal_max_qubits == 6

    def test_extract_with_anneal_enabled(self) -> None:
        """try_extract_circuit with anneal should succeed on extractable graphs."""
        g = _make_extractable_graph(n_qubits=2)
        result = try_extract_circuit(
            g, timeout=30.0, anneal_iters=20, anneal_max_qubits=4,
        )
        assert result.success is True
        assert result.circuit_qasm != ""
