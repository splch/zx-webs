"""Integration tests for multi-stage pipeline execution."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pyzx as zx
import pytest

from zx_webs.config import (
    ComposeConfig,
    CorpusConfig,
    FilterConfig,
    MiningConfig,
    PipelineConfig,
    ZXConfig,
)
from zx_webs.persistence import load_manifest, save_json, save_manifest
from zx_webs.pipeline import Pipeline, run_stage1
from zx_webs.stage3_mining.zx_web import BoundaryWire, ZXWeb
from zx_webs.stage4_compose.candidate import CandidateAlgorithm
from zx_webs.stage4_compose.stitcher import Stitcher
from zx_webs.stage5_filter.extractor import try_extract_circuit


# ---------------------------------------------------------------------------
# Stage 1 standalone
# ---------------------------------------------------------------------------


class TestRunStage1:
    """Integration tests for the run_stage1 function."""

    def test_run_stage1_creates_manifest_and_qasm(self, tmp_path: Path) -> None:
        """run_stage1 should create a manifest and QASM files on disk."""
        config = PipelineConfig(
            data_dir=tmp_path / "data",
            corpus=CorpusConfig(
                families=["entanglement"],
                max_qubits=4,
                qubit_counts=[3],
            ),
        )

        manifest = run_stage1(config)

        assert len(manifest) > 0

        # Manifest file should exist on disk
        corpus_dir = Path(config.data_dir) / "corpus"
        disk_manifest = load_manifest(corpus_dir)
        assert len(disk_manifest) == len(manifest)

        # Every QASM path in the manifest should point to an existing file
        for entry in manifest:
            qasm_path = Path(entry["qasm_path"])
            assert qasm_path.exists(), f"QASM file missing: {qasm_path}"
            contents = qasm_path.read_text()
            assert contents.startswith("OPENQASM"), (
                f"QASM file does not start with OPENQASM: {qasm_path}"
            )


# ---------------------------------------------------------------------------
# Stages 1-2 end-to-end
# ---------------------------------------------------------------------------


class TestStages1And2EndToEnd:
    """Integration test running corpus generation through ZX conversion."""

    @pytest.mark.timeout(120)
    def test_pipeline_corpus_through_zx(self, small_config: PipelineConfig) -> None:
        """Pipeline.run(start='corpus', end='zx') should produce both manifests."""
        pipeline = Pipeline(small_config)
        pipeline.run(start_stage="corpus", end_stage="zx")

        data_dir = Path(small_config.data_dir)

        # Stage 1 outputs
        corpus_dir = data_dir / "corpus"
        corpus_manifest = load_manifest(corpus_dir)
        assert len(corpus_manifest) > 0, "Corpus manifest should not be empty"

        # Every QASM file referenced in the manifest should exist
        for entry in corpus_manifest:
            assert Path(entry["qasm_path"]).exists()

        # Stage 2 outputs
        zx_dir = data_dir / "zx_diagrams"
        zx_manifest = load_manifest(zx_dir)
        assert len(zx_manifest) > 0, "ZX manifest should not be empty"
        assert len(zx_manifest) == len(corpus_manifest), (
            "ZX manifest should have one entry per corpus entry"
        )

        # Every graph file referenced in the ZX manifest should exist
        for entry in zx_manifest:
            graph_path = Path(entry["graph_path"])
            assert graph_path.exists(), f"Graph file missing: {graph_path}"

    def test_pipeline_single_stage(self, small_config: PipelineConfig) -> None:
        """Pipeline.run_stage('corpus') should produce only corpus outputs."""
        pipeline = Pipeline(small_config)
        pipeline.run_stage("corpus")

        data_dir = Path(small_config.data_dir)

        corpus_manifest = load_manifest(data_dir / "corpus")
        assert len(corpus_manifest) > 0

        # ZX stage should NOT have run
        zx_dir = data_dir / "zx_diagrams"
        assert not zx_dir.exists() or len(load_manifest(zx_dir)) == 0


# ---------------------------------------------------------------------------
# Composition -> Extraction integration
# ---------------------------------------------------------------------------


class TestCompositionExtraction:
    """Integration test: compose webs and verify circuit extraction succeeds."""

    def test_sequential_compose_extracts(self) -> None:
        """Sequentially composed 1-qubit webs should produce extractable circuits."""
        # Build two 1-qubit webs.
        def _make_1q_web(web_id: str, phase: Fraction) -> ZXWeb:
            g = zx.Graph()
            i0 = g.add_vertex(ty=0, qubit=0, row=0)
            z0 = g.add_vertex(ty=1, phase=phase, qubit=0, row=1)
            o0 = g.add_vertex(ty=0, qubit=0, row=2)
            g.add_edge((i0, z0), edgetype=1)
            g.add_edge((z0, o0), edgetype=1)
            g.set_inputs((i0,))
            g.set_outputs((o0,))
            return ZXWeb(
                web_id=web_id,
                graph_json=g.to_json(),
                n_inputs=1,
                n_outputs=1,
            )

        web_a = _make_1q_web("web_a", Fraction(0))
        web_b = _make_1q_web("web_b", Fraction(1, 4))

        config = ComposeConfig(seed=42)
        stitcher = Stitcher(config)
        composed = stitcher.compose_sequential(web_a, web_b)

        assert composed is not None, "Sequential compose should succeed"

        result = try_extract_circuit(composed, timeout=10.0)
        assert result.success is True, (
            f"Extraction should succeed: {result.error}"
        )

    def test_parallel_compose_extracts(self) -> None:
        """Parallel composed webs should produce extractable circuits."""
        def _make_1q_web(web_id: str, phase: Fraction) -> ZXWeb:
            g = zx.Graph()
            i0 = g.add_vertex(ty=0, qubit=0, row=0)
            z0 = g.add_vertex(ty=1, phase=phase, qubit=0, row=1)
            o0 = g.add_vertex(ty=0, qubit=0, row=2)
            g.add_edge((i0, z0), edgetype=1)
            g.add_edge((z0, o0), edgetype=1)
            g.set_inputs((i0,))
            g.set_outputs((o0,))
            return ZXWeb(
                web_id=web_id,
                graph_json=g.to_json(),
                n_inputs=1,
                n_outputs=1,
            )

        web_a = _make_1q_web("web_a", Fraction(0))
        web_b = _make_1q_web("web_b", Fraction(1, 4))

        config = ComposeConfig(seed=42)
        stitcher = Stitcher(config)
        composed = stitcher.compose_parallel(web_a, web_b)

        assert composed is not None, "Parallel compose should succeed"

        result = try_extract_circuit(composed, timeout=10.0)
        assert result.success is True, (
            f"Extraction should succeed: {result.error}"
        )

    def test_phase_perturb_extracts(self) -> None:
        """Phase-perturbed graphs should produce extractable circuits."""
        c = zx.Circuit(1)
        c.add_gate("HAD", 0)
        c.add_gate("ZPhase", 0, phase=Fraction(1, 4))
        g = c.to_graph()

        config = ComposeConfig(seed=42)
        stitcher = Stitcher(config)
        perturbed = stitcher.perturb_phases(g, rate=1.0)

        result = try_extract_circuit(perturbed, timeout=10.0)
        assert result.success is True, (
            f"Phase-perturbed extraction should succeed: {result.error}"
        )

    def test_full_pipeline_produces_survivors(self, tmp_path: Path) -> None:
        """End-to-end: compose webs -> filter -> survivors should be non-empty."""
        # Build webs from circuit round-trips (guaranteed extractable).
        webs = []
        for i in range(3):
            g = zx.Graph()
            i0 = g.add_vertex(ty=0, qubit=0, row=0)
            z0 = g.add_vertex(ty=1, phase=Fraction(i, 4), qubit=0, row=1)
            o0 = g.add_vertex(ty=0, qubit=0, row=2)
            g.add_edge((i0, z0), edgetype=1)
            g.add_edge((z0, o0), edgetype=1)
            g.set_inputs((i0,))
            g.set_outputs((o0,))
            webs.append(ZXWeb(
                web_id=f"web_{i:04d}",
                graph_json=g.to_json(),
                n_inputs=1,
                n_outputs=1,
            ))

        # Stage 4: compose
        compose_config = ComposeConfig(
            max_candidates=20,
            composition_modes=["sequential", "parallel"],
            seed=42,
        )
        stitcher = Stitcher(compose_config)
        candidates = stitcher.generate_candidates(webs)

        assert len(candidates) > 0, "Should generate at least one candidate"

        # Stage 5: filter
        n_success = 0
        for cand in candidates:
            g = zx.Graph.from_json(cand.graph_json)
            result = try_extract_circuit(g, timeout=10.0, max_cnot_blowup=10.0)
            if result.success:
                n_success += 1

        assert n_success > 0, (
            f"At least one candidate should extract successfully "
            f"(out of {len(candidates)} candidates)"
        )


# ---------------------------------------------------------------------------
# Pipeline validation
# ---------------------------------------------------------------------------


class TestPipelineValidation:
    """Tests for Pipeline argument validation."""

    def test_invalid_stage_name_raises(self, small_config: PipelineConfig) -> None:
        """run_stage with an unknown name should raise ValueError."""
        pipeline = Pipeline(small_config)
        with pytest.raises(ValueError, match="Unknown stage"):
            pipeline.run_stage("nonexistent_stage")

    def test_invalid_start_end_order_raises(self, small_config: PipelineConfig) -> None:
        """run() with start after end should raise ValueError."""
        pipeline = Pipeline(small_config)
        with pytest.raises(ValueError, match="comes after"):
            pipeline.run(start_stage="zx", end_stage="corpus")
