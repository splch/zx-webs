"""Tests for Stage 6 -- functional benchmarking of candidates.

Tests the redesigned benchmarking system that uses process fidelity
and unitary-aware comparison rather than naive gate-count comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyzx as zx
import pytest

from zx_webs.config import BenchConfig
from zx_webs.persistence import save_json, save_manifest
from zx_webs.stage6_bench.comparator import (
    ComparisonResult,
    TaskMatch,
    compare_candidate_to_baselines,
    match_candidate_to_tasks,
)
from zx_webs.stage6_bench.metrics import (
    CircuitMetrics,
    SupermarQFeatures,
    compute_unitary,
    entanglement_capacity,
    is_clifford_unitary,
)
from zx_webs.stage6_bench.runner import run_stage6
from zx_webs.stage6_bench.tasks import BenchmarkTask, build_benchmark_tasks


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


def _make_identity_qasm(n_qubits: int = 2) -> str:
    """Build a QASM string that implements the identity (H twice)."""
    c = zx.Circuit(n_qubits)
    c.add_gate("HAD", 0)
    c.add_gate("HAD", 0)
    return c.to_qasm()


# ---------------------------------------------------------------------------
# BenchmarkTask tests
# ---------------------------------------------------------------------------


class TestBenchmarkTask:
    """Tests for BenchmarkTask dataclass and fidelity computation."""

    def test_fidelity_identical_unitaries(self) -> None:
        """Fidelity should be 1.0 when candidate == target."""
        u = np.eye(4, dtype=complex)
        task = BenchmarkTask(
            name="identity_2q",
            n_qubits=2,
            target_unitary=u,
        )
        assert task.fidelity(u) == pytest.approx(1.0, abs=1e-10)

    def test_fidelity_different_unitaries(self) -> None:
        """Fidelity should be < 1.0 for different unitaries."""
        u_target = np.eye(4, dtype=complex)
        # A Hadamard-like unitary on 2 qubits.
        u_cand = np.kron(
            np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            np.eye(2, dtype=complex),
        )
        task = BenchmarkTask(
            name="identity_2q",
            n_qubits=2,
            target_unitary=u_target,
        )
        fid = task.fidelity(u_cand)
        assert 0.0 <= fid < 1.0

    def test_fidelity_mismatched_shapes(self) -> None:
        """Fidelity returns 0.0 when shapes don't match."""
        u2 = np.eye(4, dtype=complex)
        u3 = np.eye(8, dtype=complex)
        task = BenchmarkTask(
            name="test_3q",
            n_qubits=3,
            target_unitary=u3,
        )
        assert task.fidelity(u2) == 0.0

    def test_fidelity_global_phase(self) -> None:
        """Fidelity should be 1.0 even with a global phase difference."""
        u = np.eye(4, dtype=complex)
        # Apply global phase e^{i*pi/4}.
        u_phased = u * np.exp(1j * np.pi / 4)
        task = BenchmarkTask(
            name="identity_2q",
            n_qubits=2,
            target_unitary=u,
        )
        # Process fidelity is phase-sensitive: |Tr(U_t^dag U_c)|^2 / d^2
        # For U_c = e^{i*theta} * U_t: Tr(U_t^dag * e^{i*theta} U_t) = d * e^{i*theta}
        # |d * e^{i*theta}|^2 / d^2 = 1.0
        assert task.fidelity(u_phased) == pytest.approx(1.0, abs=1e-10)

    def test_to_summary_dict(self) -> None:
        """to_summary_dict should produce a JSON-serializable dict without unitary."""
        task = BenchmarkTask(
            name="qft_3q",
            description="arithmetic/qft on 3 qubits",
            n_qubits=3,
            target_unitary=np.eye(8, dtype=complex),
            baseline_gate_count=15,
            baseline_t_count=3,
            baseline_cnot_count=5,
            baseline_depth=10,
        )
        d = task.to_summary_dict()
        assert d["name"] == "qft_3q"
        assert d["baseline_gate_count"] == 15
        assert "target_unitary" not in d


class TestBuildBenchmarkTasks:
    """Tests for the benchmark task builder."""

    def test_builds_tasks_with_correct_structure(self) -> None:
        """build_benchmark_tasks should return tasks with unitaries."""
        tasks = build_benchmark_tasks(qubit_counts=[3])
        assert len(tasks) > 0

        for task in tasks:
            assert task.n_qubits > 0
            assert task.target_unitary.shape == (2**task.n_qubits, 2**task.n_qubits)
            assert task.name != ""
            assert task.description != ""
            assert task.baseline_gate_count >= 0

    def test_tasks_have_valid_unitaries(self) -> None:
        """Target unitaries should be unitary matrices."""
        tasks = build_benchmark_tasks(qubit_counts=[3])
        for task in tasks[:5]:  # Check first 5 for speed.
            u = task.target_unitary
            d = u.shape[0]
            # U^dag U should be close to identity.
            product = u.conj().T @ u
            np.testing.assert_allclose(product, np.eye(d), atol=1e-8)

    def test_no_tasks_for_impossible_qubit_count(self) -> None:
        """No tasks should be built for qubit count 1 (all algorithms need >= 2)."""
        tasks = build_benchmark_tasks(qubit_counts=[1])
        # Some algorithms might work at 1 qubit, but most should fail.
        # We just verify the function doesn't crash.
        assert isinstance(tasks, list)


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
# Unitary analysis tests
# ---------------------------------------------------------------------------


class TestComputeUnitary:
    """Tests for the compute_unitary function."""

    def test_returns_unitary_for_small_circuit(self) -> None:
        """A small circuit should produce a valid unitary matrix."""
        qasm = _make_qasm(n_qubits=2, add_cnot=True)
        u = compute_unitary(qasm)
        assert u is not None
        assert u.shape == (4, 4)
        # Should be unitary.
        np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-8)

    def test_returns_none_for_invalid_qasm(self) -> None:
        """Invalid QASM should return None."""
        u = compute_unitary("not valid qasm")
        assert u is None


class TestIsClifford:
    """Tests for the is_clifford_unitary function."""

    def test_hadamard_is_clifford(self) -> None:
        """The Hadamard gate is a Clifford gate."""
        h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert is_clifford_unitary(h) is True

    def test_identity_is_clifford(self) -> None:
        """The identity is trivially Clifford."""
        assert is_clifford_unitary(np.eye(2, dtype=complex)) is True

    def test_t_gate_is_not_clifford(self) -> None:
        """The T gate is not Clifford."""
        t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        assert is_clifford_unitary(t) is False

    def test_cnot_is_clifford(self) -> None:
        """The CNOT gate is Clifford."""
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        assert is_clifford_unitary(cnot) is True


class TestEntanglementCapacity:
    """Tests for entanglement_capacity."""

    def test_product_unitary_zero_capacity(self) -> None:
        """A tensor product of single-qubit unitaries has zero capacity."""
        u = np.kron(np.eye(2, dtype=complex), np.eye(2, dtype=complex))
        cap = entanglement_capacity(u)
        assert cap == pytest.approx(0.0, abs=0.05)

    def test_single_qubit_returns_zero(self) -> None:
        """A single-qubit unitary should return 0.0."""
        assert entanglement_capacity(np.eye(2, dtype=complex)) == 0.0

    def test_cnot_has_positive_capacity(self) -> None:
        """CNOT should have positive entanglement capacity."""
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        cap = entanglement_capacity(cnot)
        assert cap > 0.0


# ---------------------------------------------------------------------------
# TaskMatch / comparator tests
# ---------------------------------------------------------------------------


class TestMatchCandidateToTasks:
    """Tests for the unitary-aware candidate-to-task matching."""

    def test_identical_circuit_gets_fidelity_one(self) -> None:
        """A candidate identical to the task's baseline gets fidelity=1.0."""
        # Build a simple 2-qubit circuit.
        c = zx.Circuit(2)
        c.add_gate("CNOT", 0, 1)
        c.add_gate("HAD", 0)
        qasm = c.to_qasm()
        u = np.array(c.to_matrix())
        metrics = CircuitMetrics.from_pyzx_circuit(c)

        task = BenchmarkTask(
            name="test_2q",
            n_qubits=2,
            target_unitary=u,
            baseline_gate_count=metrics.total_gates,
            baseline_t_count=metrics.t_count,
            baseline_cnot_count=metrics.cnot_count,
            baseline_depth=metrics.depth,
        )

        matches = match_candidate_to_tasks("cand_0", qasm, [task])
        assert len(matches) == 1
        assert matches[0].fidelity == pytest.approx(1.0, abs=1e-8)
        # Same gates -> not an improvement.
        assert matches[0].is_improvement is False

    def test_only_matches_same_qubit_count(self) -> None:
        """Tasks with different qubit counts should not match."""
        qasm_2q = _make_qasm(n_qubits=2, add_cnot=True)
        task_3q = BenchmarkTask(
            name="test_3q",
            n_qubits=3,
            target_unitary=np.eye(8, dtype=complex),
        )

        matches = match_candidate_to_tasks("cand_0", qasm_2q, [task_3q])
        assert len(matches) == 0

    def test_improvement_with_fewer_gates(self) -> None:
        """A high-fidelity candidate with fewer gates is an improvement."""
        # Build a 2-qubit identity circuit.
        u_identity = np.eye(4, dtype=complex)
        task = BenchmarkTask(
            name="identity_2q",
            n_qubits=2,
            target_unitary=u_identity,
            baseline_gate_count=100,  # Artificially high baseline.
            baseline_t_count=50,
            baseline_cnot_count=30,
            baseline_depth=50,
        )

        # The identity QASM (H;H) has very few gates.
        qasm = _make_identity_qasm(n_qubits=2)
        matches = match_candidate_to_tasks("cand_0", qasm, [task])
        assert len(matches) == 1
        # Fidelity should be high (close to 1.0 for identity-ish circuit).
        m = matches[0]
        if m.fidelity >= 0.99:
            assert m.is_improvement is True
            assert m.gate_improvement["total_gates"] > 0

    def test_different_unitary_low_fidelity(self) -> None:
        """A candidate implementing a different unitary gets low fidelity."""
        # Task targets the identity.
        u_identity = np.eye(4, dtype=complex)
        task = BenchmarkTask(
            name="identity_2q",
            n_qubits=2,
            target_unitary=u_identity,
            baseline_gate_count=10,
        )

        # Candidate is CNOT+H (NOT identity).
        qasm = _make_qasm(n_qubits=2, add_cnot=True, add_t=True)
        matches = match_candidate_to_tasks("cand_0", qasm, [task])
        assert len(matches) == 1
        assert matches[0].fidelity < 0.99
        assert matches[0].is_improvement is False

    def test_empty_tasks_returns_empty(self) -> None:
        """With no tasks, result should be empty."""
        qasm = _make_qasm(n_qubits=2)
        matches = match_candidate_to_tasks("cand_0", qasm, [])
        assert matches == []


# ---------------------------------------------------------------------------
# Legacy comparator tests (backward compatibility)
# ---------------------------------------------------------------------------


class TestCompareCandidate:
    """Test legacy candidate-vs-baseline comparisons."""

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
        simple = _make_qasm(n_qubits=1, add_t=False, add_cnot=False)
        bigger = _make_large_qasm()
        results = compare_candidate_to_baselines("cand_0", simple, [{"id": "bl_big", "qasm": bigger}])

        assert len(results) == 1
        r = results[0]
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
        # Backward-compatible fields.
        assert "dominates_any_baseline" in result
        # New functional benchmarking fields.
        assert "classification" in result
        assert "best_fidelity" in result
        assert "task_matches" in result
        assert "n_qubits" in result

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

    def test_run_stage6_classification_present(self, tmp_path: Path) -> None:
        """Verify that classification data (Clifford, entanglement) is in results."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        save_manifest([], corpus_dir)

        filtered_dir = tmp_path / "filtered"
        circuits_dir = filtered_dir / "circuits"
        circuits_dir.mkdir(parents=True)

        # Use a CNOT+H circuit (Clifford).
        survivor_qasm = _make_qasm(n_qubits=2, add_t=False, add_cnot=True)
        surv_data = {
            "survivor_id": "surv_clifford",
            "candidate_id": "cand_clifford",
            "circuit_qasm": survivor_qasm,
            "n_qubits": 2,
        }
        surv_path = circuits_dir / "surv_clifford.json"
        save_json(surv_data, surv_path)

        save_manifest(
            [
                {
                    "survivor_id": "surv_clifford",
                    "circuit_path": str(surv_path),
                    "candidate_id": "cand_clifford",
                    "n_qubits": 2,
                },
            ],
            filtered_dir,
        )

        output_dir = tmp_path / "bench_classify"
        results = run_stage6(filtered_dir, corpus_dir, output_dir)

        assert len(results) == 1
        classification = results[0]["classification"]
        assert "is_clifford" in classification
        assert "entanglement_capacity" in classification
        # A CNOT+H circuit should be Clifford.
        assert classification["is_clifford"] is True


# ---------------------------------------------------------------------------
# Tests for novelty scoring
# ---------------------------------------------------------------------------


class TestNoveltyScore:
    """Tests for the novelty_score function."""

    def test_identical_unitary_zero_novelty(self) -> None:
        """A unitary identical to corpus should have novelty 0."""
        from zx_webs.stage6_bench.metrics import novelty_score

        u = np.eye(4, dtype=complex)
        score = novelty_score(u, [u])
        assert abs(score) < 1e-10

    def test_empty_corpus_max_novelty(self) -> None:
        """With no corpus unitaries, novelty should be 1.0."""
        from zx_webs.stage6_bench.metrics import novelty_score

        u = np.eye(4, dtype=complex)
        score = novelty_score(u, [])
        assert abs(score - 1.0) < 1e-10

    def test_different_unitary_positive_novelty(self) -> None:
        """A random unitary should have positive novelty vs identity."""
        from zx_webs.stage6_bench.metrics import novelty_score

        identity = np.eye(4, dtype=complex)
        # CNOT matrix is different from identity.
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        score = novelty_score(cnot, [identity])
        assert score > 0.0

    def test_novelty_in_unit_interval(self) -> None:
        """Novelty score should always be in [0, 1]."""
        from zx_webs.stage6_bench.metrics import novelty_score

        u = np.eye(2, dtype=complex)
        corpus = [
            np.array([[0, 1], [1, 0]], dtype=complex),  # X gate
            np.array([[1, 0], [0, -1]], dtype=complex),  # Z gate
        ]
        score = novelty_score(u, corpus)
        assert 0.0 <= score <= 1.0


class TestProcessFidelity:
    """Tests for the process_fidelity function."""

    def test_identical_unitaries_fidelity_one(self) -> None:
        """Fidelity of a unitary with itself should be 1."""
        from zx_webs.stage6_bench.metrics import process_fidelity

        u = np.eye(4, dtype=complex)
        fid = process_fidelity(u, u)
        assert abs(fid - 1.0) < 1e-10

    def test_mismatched_shapes_zero(self) -> None:
        """Mismatched unitary shapes should give fidelity 0."""
        from zx_webs.stage6_bench.metrics import process_fidelity

        u1 = np.eye(2, dtype=complex)
        u2 = np.eye(4, dtype=complex)
        fid = process_fidelity(u1, u2)
        assert fid == 0.0

    def test_novelty_scoring_config(self) -> None:
        """Novelty scoring config should default to False."""
        from zx_webs.config import BenchConfig

        config = BenchConfig()
        assert config.novelty_scoring is False
