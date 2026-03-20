"""Integration tests for multi-stage pipeline execution."""
from __future__ import annotations

from pathlib import Path

import pytest

from zx_webs.config import CorpusConfig, PipelineConfig, ZXConfig
from zx_webs.persistence import load_manifest
from zx_webs.pipeline import Pipeline, run_stage1


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
