"""Tests for Stage 7 -- report generation."""
from __future__ import annotations

from pathlib import Path

import pytest

from zx_webs.config import ReportConfig
from zx_webs.persistence import save_json, save_manifest
from zx_webs.stage7_report.reporter import generate_report_html, generate_summary, run_stage7


# ---------------------------------------------------------------------------
# Helpers -- set up a complete data directory
# ---------------------------------------------------------------------------


def _setup_data_dir(tmp_path: Path) -> Path:
    """Create a data directory with manifests for all stages.

    Returns the path to the data directory.
    """
    data_dir = tmp_path / "data"

    # Stage 1: corpus
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    save_manifest(
        [
            {"algorithm_id": "algo_0", "family": "test", "name": "ghz", "n_qubits": 3},
            {"algorithm_id": "algo_1", "family": "test", "name": "qpe", "n_qubits": 5},
        ],
        corpus_dir,
    )

    # Stage 2: ZX diagrams
    zx_dir = data_dir / "zx_diagrams"
    zx_dir.mkdir()
    save_manifest(
        [
            {"diagram_id": "diag_0"},
            {"diagram_id": "diag_1"},
            {"diagram_id": "diag_2"},
        ],
        zx_dir,
    )

    # Stage 3: mined webs
    webs_dir = data_dir / "mined_webs"
    webs_dir.mkdir()
    save_manifest(
        [{"web_id": "web_0"}, {"web_id": "web_1"}],
        webs_dir,
    )

    # Stage 4: candidates
    cand_dir = data_dir / "candidates"
    cand_dir.mkdir()
    save_manifest(
        [
            {"candidate_id": "cand_0"},
            {"candidate_id": "cand_1"},
            {"candidate_id": "cand_2"},
            {"candidate_id": "cand_3"},
        ],
        cand_dir,
    )

    # Stage 5: filtered
    filt_dir = data_dir / "filtered"
    filt_dir.mkdir()
    save_manifest(
        [{"survivor_id": "surv_0"}, {"survivor_id": "surv_1"}],
        filt_dir,
    )

    # Stage 6: benchmarks
    bench_dir = data_dir / "benchmarks"
    bench_dir.mkdir()
    save_json(
        [
            {"survivor_id": "surv_0", "dominates_any_baseline": True, "metrics": {}},
            {"survivor_id": "surv_1", "dominates_any_baseline": False, "metrics": {}},
        ],
        bench_dir / "results.json",
    )

    return data_dir


# ---------------------------------------------------------------------------
# generate_summary tests
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    """Test summary generation from a multi-stage data directory."""

    def test_summary_structure(self, tmp_path: Path) -> None:
        """Summary should contain expected stage keys and counts."""
        data_dir = _setup_data_dir(tmp_path)
        summary = generate_summary(data_dir)

        assert "generated_at" in summary
        stages = summary["stages"]

        assert stages["corpus"]["n_algorithms"] == 2
        assert stages["zx_diagrams"]["n_diagrams"] == 3
        assert stages["mining"]["n_webs"] == 2
        assert stages["compose"]["n_candidates"] == 4
        assert stages["filter"]["n_survivors"] == 2
        assert stages["bench"]["n_benchmarked"] == 2
        assert stages["bench"]["n_dominating"] == 1

    def test_summary_missing_stages(self, tmp_path: Path) -> None:
        """Summary should handle missing stage directories gracefully."""
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        summary = generate_summary(data_dir)

        stages = summary["stages"]
        assert stages["corpus"]["n_algorithms"] == 0
        assert stages["zx_diagrams"]["n_diagrams"] == 0
        assert "bench" not in stages  # No results.json -> key absent

    def test_summary_generated_at_is_iso(self, tmp_path: Path) -> None:
        """generated_at should be an ISO-format timestamp."""
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        summary = generate_summary(data_dir)
        # Should not raise.
        from datetime import datetime
        datetime.fromisoformat(summary["generated_at"])


# ---------------------------------------------------------------------------
# generate_report_html tests
# ---------------------------------------------------------------------------


class TestGenerateReportHtml:
    """Test HTML report generation."""

    def test_html_file_created(self, tmp_path: Path) -> None:
        """An HTML file should be created with correct content."""
        data_dir = _setup_data_dir(tmp_path)
        summary = generate_summary(data_dir)

        output_path = tmp_path / "report.html"
        generate_report_html(summary, output_path)

        assert output_path.exists()
        html = output_path.read_text()
        assert "<!DOCTYPE html>" in html
        assert "ZX-Webs Pipeline Report" in html
        assert "n_algorithms" in html
        assert "n_survivors" in html

    def test_html_with_empty_summary(self, tmp_path: Path) -> None:
        """HTML generation should not fail on an empty summary."""
        summary = {"generated_at": "2025-01-01T00:00:00+00:00", "stages": {}}
        output_path = tmp_path / "report_empty.html"
        generate_report_html(summary, output_path)

        assert output_path.exists()
        html = output_path.read_text()
        assert "ZX-Webs Pipeline Report" in html


# ---------------------------------------------------------------------------
# End-to-end Stage 7 test
# ---------------------------------------------------------------------------


class TestRunStage7:
    """End-to-end integration test for run_stage7."""

    def test_run_stage7_creates_outputs(self, tmp_path: Path) -> None:
        """run_stage7 should create summary.json and report.html."""
        data_dir = _setup_data_dir(tmp_path)
        output_dir = tmp_path / "report_output"

        summary = run_stage7(data_dir, output_dir)

        # Summary dict should be returned.
        assert "stages" in summary
        assert summary["stages"]["corpus"]["n_algorithms"] == 2

        # Files should exist.
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "report.html").exists()

    def test_run_stage7_json_only(self, tmp_path: Path) -> None:
        """When output_format excludes html, only summary.json is created."""
        data_dir = _setup_data_dir(tmp_path)
        output_dir = tmp_path / "report_json_only"

        config = ReportConfig(output_format=["json"])
        summary = run_stage7(data_dir, output_dir, config)

        assert (output_dir / "summary.json").exists()
        assert not (output_dir / "report.html").exists()
        assert "stages" in summary

    def test_run_stage7_empty_data(self, tmp_path: Path) -> None:
        """run_stage7 on an empty data directory should not fail."""
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        output_dir = tmp_path / "report_empty"

        summary = run_stage7(data_dir, output_dir)
        assert summary["stages"]["corpus"]["n_algorithms"] == 0
        assert (output_dir / "summary.json").exists()
