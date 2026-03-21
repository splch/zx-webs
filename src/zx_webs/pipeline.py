"""Pipeline orchestrator for the ZX-Webs quantum algorithm discovery system.

Provides :func:`run_stage1` to build the algorithm corpus and :class:`Pipeline`
to orchestrate multi-stage execution from corpus generation through reporting.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from zx_webs.config import PipelineConfig, load_config
from zx_webs.persistence import save_manifest
from zx_webs.stage1_corpus import build_corpus, circuit_to_pyzx_qasm
from zx_webs.stage2_zx import run_stage2
from zx_webs.stage3_mining.miner import run_stage3
from zx_webs.stage4_compose.stitcher import run_stage4
from zx_webs.stage5_filter.extractor import run_stage5
from zx_webs.stage6_bench.runner import run_stage6
from zx_webs.stage7_report.reporter import run_stage7

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 runner
# ---------------------------------------------------------------------------


def run_stage1(config: PipelineConfig) -> list[dict]:
    """Run Stage 1: build algorithm corpus and save QASM files.

    Creates ``data/corpus/manifest.json`` and per-algorithm QASM files under
    ``data/corpus/algorithms/{family}/{name}_{n}q.qasm``.

    Parameters
    ----------
    config:
        The full pipeline configuration.  Only ``config.corpus`` and
        ``config.data_dir`` are used.

    Returns
    -------
    list[dict]
        The manifest entries written to ``corpus/manifest.json``.
    """
    corpus_dir = Path(config.data_dir) / "corpus"
    algorithms_dir = corpus_dir / "algorithms"

    corpus_entries = build_corpus(config.corpus)

    manifest: list[dict] = []
    for entry in tqdm(corpus_entries, desc="Stage 1: Building corpus", unit="circuit"):
        family = entry["family"]
        name = entry["name"]
        n_qubits = entry["n_qubits"]
        circuit = entry["circuit"]
        algorithm_id = entry["algorithm_id"]

        # Convert to PyZX-compatible QASM
        qasm_str = circuit_to_pyzx_qasm(circuit)

        # Persist QASM file
        family_dir = algorithms_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        qasm_path = family_dir / f"{name}_{n_qubits}q.qasm"
        qasm_path.write_text(qasm_str)

        manifest.append({
            "algorithm_id": algorithm_id,
            "family": family,
            "name": name,
            "n_qubits": n_qubits,
            "qasm_path": str(qasm_path),
        })

    save_manifest(manifest, corpus_dir)
    logger.info("Stage 1 complete: %d circuits saved to %s", len(manifest), corpus_dir)
    return manifest


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class Pipeline:
    """Orchestrates the ZX-Webs pipeline stages.

    Parameters
    ----------
    config:
        The full pipeline configuration.
    """

    STAGES: list[str] = [
        "corpus",
        "zx",
        "mining",
        "compose",
        "filter",
        "bench",
        "report",
    ]

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache for passing data between stages without disk I/O.
        self._stage_cache: dict[str, object] = {}

    def run(self, start_stage: str = "corpus", end_stage: str = "report") -> None:
        """Run pipeline stages from *start_stage* through *end_stage* (inclusive).

        When running consecutive stages (e.g. mining -> compose), intermediate
        results are passed in-memory to avoid redundant disk I/O.

        Raises
        ------
        ValueError
            If either stage name is not in :attr:`STAGES`.
        """
        start_idx = self._stage_index(start_stage)
        end_idx = self._stage_index(end_stage)

        if start_idx > end_idx:
            raise ValueError(
                f"start_stage {start_stage!r} (index {start_idx}) comes after "
                f"end_stage {end_stage!r} (index {end_idx})"
            )

        for stage_name in self.STAGES[start_idx : end_idx + 1]:
            logger.info("Running stage: %s", stage_name)
            self.run_stage(stage_name)

    def run_stage(self, stage_name: str) -> None:
        """Run a single pipeline stage by name.

        When stages are run sequentially via :meth:`run`, intermediate
        results are cached in ``self._stage_cache`` and passed directly
        to the next stage (e.g. Stage 3 webs -> Stage 4) to avoid the
        I/O overhead of writing and reading thousands of individual files.

        Raises
        ------
        ValueError
            If *stage_name* is not in :attr:`STAGES`.
        """
        self._stage_index(stage_name)  # validate

        if stage_name == "corpus":
            run_stage1(self.config)
        elif stage_name == "zx":
            run_stage2(
                self.data_dir / "corpus",
                self.data_dir / "zx_diagrams",
                self.config.zx,
            )
        elif stage_name == "mining":
            webs = run_stage3(
                self.data_dir / "zx_diagrams",
                self.data_dir / "mined_webs",
                self.config.mining,
                corpus_dir=self.data_dir / "corpus",
                # Skip the expensive bulk JSON write when webs will be
                # passed in-memory to Stage 4.  This avoids materializing
                # graph_json for all ~3M webs (the #1 post-mining bottleneck).
                # The lightweight manifest is still written for diagnostics.
                skip_bulk_write=True,
            )
            # Cache webs for the next stage (compose) to consume in-memory.
            self._stage_cache["mining_webs"] = webs
        elif stage_name == "compose":
            # Use in-memory webs from Stage 3 if available (avoids 224K file reads).
            webs_in_memory = self._stage_cache.pop("mining_webs", None)
            run_stage4(
                self.data_dir / "mined_webs",
                self.data_dir / "candidates",
                self.config.compose,
                webs_in_memory=webs_in_memory,
            )
        elif stage_name == "filter":
            run_stage5(
                self.data_dir / "candidates",
                self.data_dir / "filtered",
                self.config.filter,
            )
        elif stage_name == "bench":
            run_stage6(
                self.data_dir / "filtered",
                self.data_dir / "corpus",
                self.data_dir / "benchmarks",
                self.config.bench,
            )
        elif stage_name == "report":
            run_stage7(
                self.data_dir,
                self.data_dir / "report",
                self.config.report,
            )
        else:
            logger.warning("Stage %r not yet implemented", stage_name)

    # -- helpers -------------------------------------------------------------

    def _stage_index(self, name: str) -> int:
        """Return the index of *name* in :attr:`STAGES`, or raise ValueError."""
        try:
            return self.STAGES.index(name)
        except ValueError:
            raise ValueError(
                f"Unknown stage {name!r}. Valid stages: {self.STAGES}"
            ) from None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for the ZX-Webs pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="ZX-Webs Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the pipeline YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Run a single stage by name",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="corpus",
        help="First stage to run (default: corpus)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="report",
        help="Last stage to run (default: report)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = Pipeline(config)

    if args.stage:
        pipeline.run_stage(args.stage)
    else:
        pipeline.run(args.start, args.end)
