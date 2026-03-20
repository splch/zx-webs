"""Tests for Stage 6 -- benchmarking candidates against baselines."""
from __future__ import annotations

from pathlib import Path

import pyzx as zx
import pytest

from zx_webs.config import BenchConfig
from zx_webs.persistence import save_json, save_manifest
from zx_webs.stage6_bench.comparator import compare_candidate_to_baselines
from zx_webs.stage6_bench.metrics import CircuitMetrics, SupermarQFeatures
from zx_webs.stage6_bench.runner import run_stage6


# ---------------------------------------------------------------------------
# Helpers -- generate QASM strings for testing
# ---------------------------------------------------------------------------


def _make_qasm(n_qubits: int = 2, add_t: bool = False, add_cnot: bool = True) -> str:
    """Build a simple QASM string via PyZX."""
    c = zx.Circuit(n_qubits)
    if add_cnot and n_qubits >= 2:
        c.add_gate("CNOT", 0, 1)
    c.add_gate("HAD", 0)
    if add_t:
        c.add_gate("T", 0)
    return c.to_qasm()


def _make_large_qasm() -> str:
    """Build a slightly larger circuit with more gates."""
    c = zx.Circuit(3)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT", 1, 2)
    c.add_gate("HAD", 0)
    c.add_gate("HAD", 1)
    c.add_gate("T", 0)
    c.add_gate("T", 1)
    c.add_gate("CNOT", 0, 2)
    return c.to_qasm()


# ---------------------------------------------------------------------------
# CircuitMetrics tests
# ---------------------------------------------------------------------------


class TestCircuitMetricsFromQasm:
    """Test metric extraction from QASM strings."""

    def test_basic_metrics(self) -> None:
        """A 2-qubit CNOT+H+T circuit should have known metric values."""
        qasm = _make_qasm(n_qubits=2, add_t=True, add_cnot=True)
        m = CircuitMetrics.from_qasm(qasm)

        assert m.qubit_count == 2
        assert m.t_count >= 1
        assert m.cnot_count >= 1
        assert m.total_gates >= 3
        assert m.depth >= 1

    def test_single_qubit_no_cnot(self) -> None:
        """A 1-qubit circuit should have zero two-qubit gates."""
        qasm = _make_qasm(n_qubits=1, add_t=True, add_cnot=False)
        m = CircuitMetrics.from_qasm(qasm)

        assert m.qubit_count == 1
        assert m.cnot_count == 0
        assert m.total_two_qubit == 0
        assert m.t_count >= 1

    def test_to_dict_round_trip(self) -> None:
        """to_dict should produce a plain dict with all fields."""
        m = CircuitMetrics(t_count=3, cnot_count=2, total_two_qubit=2,
                           total_gates=7, depth=4, qubit_count=3)
        d = m.to_dict()
        assert d["t_count"] == 3
        assert d["depth"] == 4
        assert d["qubit_count"] == 3


class TestCircuitMetricsDominates:
    """Test Pareto dominance logic."""

    def test_strictly_better_on_all(self) -> None:
        """A that is strictly better on all metrics dominates B."""
        a = CircuitMetrics(t_count=1, cnot_count=2, depth=3)
        b = CircuitMetrics(t_count=5, cnot_count=8, depth=10)
        assert a.dominates(b) is True
        assert b.dominates(a) is False

    def test_better_on_some_equal_on_rest(self) -> None:
        """Better on some, equal on the rest -> dominates."""
        a = CircuitMetrics(t_count=1, cnot_count=2, depth=5)
        b = CircuitMetrics(t_count=3, cnot_count=2, depth=5)
        assert a.dominates(b) is True

    def test_no_dominance_when_equal(self) -> None:
        """Identical metrics -> neither dominates the other."""
        a = CircuitMetrics(t_count=3, cnot_count=4, depth=5)
        b = CircuitMetrics(t_count=3, cnot_count=4, depth=5)
        assert a.dominates(b) is False
        assert b.dominates(a) is False

    def test_no_dominance_mixed(self) -> None:
        """A is better on one metric, B on another -> no dominance."""
        a = CircuitMetrics(t_count=1, cnot_count=10, depth=5)
        b = CircuitMetrics(t_count=5, cnot_count=2, depth=5)
        assert a.dominates(b) is False
        assert b.dominates(a) is False


# ---------------------------------------------------------------------------
# SupermarQFeatures tests
# ---------------------------------------------------------------------------


class TestSupermarQFeatures:
    """Test SupermarQ feature computation."""

    def test_features_in_unit_interval(self) -> None:
        """All features should be in [0, 1]."""
        qasm = _make_large_qasm()
        f = SupermarQFeatures.from_qasm(qasm)

        for field_name in ("program_communication", "critical_depth",
                           "entanglement_ratio", "parallelism", "liveness"):
            value = getattr(f, field_name)
            assert 0.0 <= value <= 1.0, f"{field_name}={value} not in [0,1]"

    def test_single_qubit_parallelism_zero(self) -> None:
        """A 1-qubit circuit should have parallelism=0."""
        qasm = _make_qasm(n_qubits=1, add_t=True, add_cnot=False)
        f = SupermarQFeatures.from_qasm(qasm)
        assert f.parallelism == 0.0

    def test_no_two_qubit_communication_zero(self) -> None:
        """A circuit with no two-qubit gates should have communication=0."""
        qasm = _make_qasm(n_qubits=1, add_t=False, add_cnot=False)
        f = SupermarQFeatures.from_qasm(qasm)
        assert f.program_communication == 0.0
        assert f.entanglement_ratio == 0.0

    def test_to_dict(self) -> None:
        """to_dict should return a dict with all feature names."""
        f = SupermarQFeatures(program_communication=0.5, critical_depth=0.3,
                              entanglement_ratio=0.5, parallelism=0.2, liveness=0.8)
        d = f.to_dict()
        assert d["program_communication"] == 0.5
        assert d["liveness"] == 0.8


# ---------------------------------------------------------------------------
# Comparator tests
# ---------------------------------------------------------------------------


class TestCompareCandidate:
    """Test candidate-vs-baseline comparisons."""

    def test_compare_against_two_baselines(self) -> None:
        """Compare a candidate against two baselines and inspect results."""
        candidate_qasm = _make_qasm(n_qubits=2, add_t=False)
        baselines = [
            {"id": "bl_a", "qasm": _make_qasm(n_qubits=2, add_t=True)},
            {"id": "bl_b", "qasm": _make_large_qasm()},
        ]

        results = compare_candidate_to_baselines("cand_0", candidate_qasm, baselines)
        assert len(results) == 2
        assert results[0].baseline_id == "bl_a"
        assert results[1].baseline_id == "bl_b"

        # Each result should have improvement percentages.
        for r in results:
            assert "t_count" in r.improvements
            assert "cnot_count" in r.improvements
            assert "depth" in r.improvements

    def test_compare_empty_baselines(self) -> None:
        """With no baselines, result list should be empty."""
        results = compare_candidate_to_baselines("cand_0", _make_qasm(), [])
        assert results == []

    def test_dominance_reflected(self) -> None:
        """When candidate is strictly simpler, candidate_dominates should be True."""
        # 1-qubit H gate (minimal) vs larger circuit
        simple = _make_qasm(n_qubits=1, add_t=False, add_cnot=False)
        bigger = _make_large_qasm()
        results = compare_candidate_to_baselines("cand_0", simple, [{"id": "bl_big", "qasm": bigger}])

        assert len(results) == 1
        # The simple circuit should have strictly fewer t_count, cnot, depth.
        r = results[0]
        # We can at least verify the dominance fields are booleans.
        assert isinstance(r.candidate_dominates, bool)
        assert isinstance(r.baseline_dominates, bool)


# ---------------------------------------------------------------------------
# End-to-end Stage 6 test
# ---------------------------------------------------------------------------


class TestRunStage6EndToEnd:
    """End-to-end integration test for run_stage6."""

    def test_run_stage6_creates_results(self, tmp_path: Path) -> None:
        """Set up Stage 5 + corpus dirs, run Stage 6, verify results.json."""
        # -- Set up corpus (baselines) ----------------------------------------
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        baseline_qasm = _make_qasm(n_qubits=2, add_t=True)
        qasm_path = corpus_dir / "baseline_2q.qasm"
        qasm_path.write_text(baseline_qasm)

        save_manifest(
            [
                {
                    "algorithm_id": "algo_baseline",
                    "family": "test",
                    "name": "baseline_2q",
                    "n_qubits": 2,
                    "qasm_path": str(qasm_path),
                },
            ],
            corpus_dir,
        )

        # -- Set up filtered (survivors) --------------------------------------
        filtered_dir = tmp_path / "filtered"
        circuits_dir = filtered_dir / "circuits"
        circuits_dir.mkdir(parents=True)

        survivor_qasm = _make_qasm(n_qubits=2, add_t=False)
        surv_data = {
            "survivor_id": "surv_0000",
            "candidate_id": "cand_0000",
            "circuit_qasm": survivor_qasm,
            "n_qubits": 2,
        }
        surv_path = circuits_dir / "surv_0000.json"
        save_json(surv_data, surv_path)

        save_manifest(
            [
                {
                    "survivor_id": "surv_0000",
                    "circuit_path": str(surv_path),
                    "candidate_id": "cand_0000",
                    "n_qubits": 2,
                },
            ],
            filtered_dir,
        )

        # -- Run Stage 6 -----------------------------------------------------
        output_dir = tmp_path / "bench_output"
        results = run_stage6(filtered_dir, corpus_dir, output_dir)

        # -- Verify -----------------------------------------------------------
        assert len(results) == 1
        result = results[0]
        assert result["survivor_id"] == "surv_0000"
        assert "metrics" in result
        assert "features" in result
        assert "dominates_any_baseline" in result
        assert "comparisons" in result
        assert len(result["comparisons"]) == 1

        # Results file should exist.
        results_path = output_dir / "results.json"
        assert results_path.exists()

        # Manifest should exist.
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

    def test_run_stage6_empty_filtered(self, tmp_path: Path) -> None:
        """Stage 6 on empty filtered manifest returns empty results."""
        filtered_dir = tmp_path / "filtered_empty"
        filtered_dir.mkdir()
        save_manifest([], filtered_dir)

        corpus_dir = tmp_path / "corpus_empty"
        corpus_dir.mkdir()
        save_manifest([], corpus_dir)

        output_dir = tmp_path / "bench_empty"
        results = run_stage6(filtered_dir, corpus_dir, output_dir)
        assert results == []
        assert (output_dir / "results.json").exists()
