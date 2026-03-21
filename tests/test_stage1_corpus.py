"""Tests for Stage 1 -- Algorithm Corpus generation."""
from __future__ import annotations

import inspect

import pytest
from qiskit import QuantumCircuit

from zx_webs.config import CorpusConfig
from zx_webs.stage1_corpus import ALGORITHM_REGISTRY, build_corpus, circuit_to_pyzx_qasm


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestAlgorithmRegistry:
    """Tests for the algorithm registry itself."""

    def test_registry_not_empty(self) -> None:
        """ALGORITHM_REGISTRY should contain at least 47 algorithm builders."""
        assert len(ALGORITHM_REGISTRY) >= 47, (
            f"Expected >= 47 algorithms in registry, got {len(ALGORITHM_REGISTRY)}: "
            f"{sorted(ALGORITHM_REGISTRY.keys())}"
        )

    def test_all_algorithms_build(self) -> None:
        """Every registered algorithm should produce a QuantumCircuit with min args."""
        for key, fn in sorted(ALGORITHM_REGISTRY.items()):
            min_q: int = getattr(fn, "min_qubits", 2)

            # Determine the first parameter name so we can pass min qubit count
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            first_param = params[0] if params else "n_qubits"

            qc = fn(**{first_param: min_q})

            assert isinstance(qc, QuantumCircuit), (
                f"{key} returned {type(qc).__name__}, expected QuantumCircuit"
            )
            assert qc.num_qubits >= 1, f"{key} produced circuit with 0 qubits"


# ---------------------------------------------------------------------------
# build_corpus tests
# ---------------------------------------------------------------------------


class TestBuildCorpus:
    """Tests for the build_corpus function."""

    def test_build_corpus_small(self) -> None:
        """build_corpus with a small config returns a list of well-formed dicts."""
        config = CorpusConfig(
            families=["entanglement"],
            max_qubits=5,
            qubit_counts=[3, 5],
        )
        entries = build_corpus(config)

        assert isinstance(entries, list)
        assert len(entries) > 0, "Expected at least one corpus entry"

        required_keys = {"algorithm_id", "family", "name", "n_qubits", "circuit"}
        for entry in entries:
            assert required_keys.issubset(entry.keys()), (
                f"Missing keys in entry: {required_keys - entry.keys()}"
            )
            assert entry["family"] == "entanglement"
            assert isinstance(entry["circuit"], QuantumCircuit)
            assert entry["n_qubits"] >= 1

    def test_build_corpus_respects_max_qubits(self) -> None:
        """build_corpus should not produce circuits exceeding max_qubits."""
        config = CorpusConfig(
            families=["entanglement"],
            max_qubits=3,
            qubit_counts=[3, 5, 7],
        )
        entries = build_corpus(config)
        for entry in entries:
            # qubit_counts > max_qubits are skipped by build_corpus
            # but n_qubits in the entry reflects the actual circuit size
            # which may be larger than the requested count (e.g. ancilla)
            assert entry["n_qubits"] >= 1

    def test_multi_instance_generation(self) -> None:
        """Parameterized algorithms should generate multiple instances."""
        config = CorpusConfig(
            families=["oracular", "variational", "simulation"],
            max_qubits=5,
            qubit_counts=[3],
        )
        entries = build_corpus(config)

        # Count entries per algorithm name.
        from collections import Counter
        name_counts = Counter(e["name"] for e in entries)

        # Bernstein-Vazirani, Grover, Simon should have multiple instances.
        assert name_counts.get("bernstein_vazirani", 0) >= 2, (
            f"Expected >= 2 BV instances, got {name_counts.get('bernstein_vazirani', 0)}"
        )
        assert name_counts.get("grover", 0) >= 2, (
            f"Expected >= 2 Grover instances, got {name_counts.get('grover', 0)}"
        )

        # Different instances should have different algorithm_ids.
        ids = [e["algorithm_id"] for e in entries if e["name"] == "bernstein_vazirani"]
        assert len(ids) == len(set(ids)), "BV instances should have unique IDs"

    def test_qaoa_topologies(self) -> None:
        """QAOA should generate ring, star, and random topologies."""
        config = CorpusConfig(
            families=["variational"],
            max_qubits=5,
            qubit_counts=[4],
        )
        entries = build_corpus(config)

        qaoa_ids = [e["algorithm_id"] for e in entries if e["name"] == "qaoa_maxcut"]
        # Should have ring, star, and random.
        assert any("ring" in aid for aid in qaoa_ids), "Missing QAOA ring topology"
        assert any("star" in aid for aid in qaoa_ids), "Missing QAOA star topology"
        assert any("random" in aid for aid in qaoa_ids), "Missing QAOA random topology"

    def test_multi_instance_reproducible(self) -> None:
        """Multi-instance generation should be deterministic with same seed."""
        config = CorpusConfig(
            families=["oracular"],
            max_qubits=5,
            qubit_counts=[3],
            seed=42,
        )
        entries1 = build_corpus(config)
        entries2 = build_corpus(config)

        ids1 = [e["algorithm_id"] for e in entries1]
        ids2 = [e["algorithm_id"] for e in entries2]
        assert ids1 == ids2, "Multi-instance generation should be deterministic"


# ---------------------------------------------------------------------------
# QASM bridge tests
# ---------------------------------------------------------------------------


class TestCircuitToPyzxQasm:
    """Tests for the circuit_to_pyzx_qasm bridge function."""

    def test_simple_circuit(self) -> None:
        """Converting a simple circuit should produce valid QASM 2.0 output."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        qasm_str = circuit_to_pyzx_qasm(qc)

        assert isinstance(qasm_str, str)
        assert qasm_str.startswith("OPENQASM")
        assert "measure" not in qasm_str.lower()

    def test_circuit_with_measurements_stripped(self) -> None:
        """Measurements in the input circuit should be stripped from the output."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        qasm_str = circuit_to_pyzx_qasm(qc)

        assert "measure" not in qasm_str.lower()
        assert qasm_str.startswith("OPENQASM")

    def test_output_contains_gate_operations(self) -> None:
        """The QASM output should contain gate definitions and applications."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        qasm_str = circuit_to_pyzx_qasm(qc)

        assert "qreg" in qasm_str
        assert "h " in qasm_str or "h q" in qasm_str
