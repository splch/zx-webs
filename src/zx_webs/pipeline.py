"""Pipeline orchestrator for the ZX-Webs quantum algorithm discovery system.

Provides :func:`run_stage1` to build the algorithm corpus and :class:`Pipeline`
to orchestrate multi-stage execution from corpus generation through reporting.

Iterative refinement
--------------------
When ``refinement_rounds > 0``, the pipeline runs multiple passes.  After each
pass, the top survivors (by novelty score) are converted to QASM files and
injected into the corpus as a new ``"discovered"`` family.  The next round
re-mines the augmented corpus, allowing the pipeline to discover sub-patterns
*within its own discoveries* and recombine them with the original corpus
patterns.  This creates evolutionary pressure toward increasingly complex
and novel algorithms.

Fitness-guided evolutionary search
-----------------------------------
When ``fitness_guided`` is enabled (default when refinement_rounds > 0), the
pipeline tracks which web IDs, composition strategies, and family combinations
produced high-fitness survivors.  This fitness profile is used to bias web
selection in subsequent rounds via fitness-weighted FPS -- webs that
contributed to novel or near-miss candidates are prioritised.

Near-miss phase optimisation
----------------------------
Candidates with fidelity 0.80-0.99 to a known task are "near misses" that
almost implement something useful.  When ``phase_optimize_near_misses`` is
enabled, Nelder-Mead optimisation is applied to the spider phases of these
candidates, nudging them toward exact functional matches.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from zx_webs.config import PipelineConfig, load_config
from zx_webs.fitness_tracker import (
    FitnessProfile,
    build_fitness_profile,
    load_fitness_profile,
    merge_profiles,
    save_fitness_profile,
)
from zx_webs.persistence import load_json, load_manifest, save_manifest
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
        # Fitness profile accumulated across refinement rounds.
        self._fitness_profile: FitnessProfile | None = load_fitness_profile(self.data_dir)

    def run(self, start_stage: str = "corpus", end_stage: str = "report") -> None:
        """Run pipeline stages from *start_stage* through *end_stage* (inclusive).

        When running consecutive stages (e.g. mining -> compose), intermediate
        results are passed in-memory to avoid redundant disk I/O.

        When ``config.refinement_rounds > 0`` and start/end span the full
        pipeline (corpus through report), iterative refinement is enabled:
        after each complete pass, top survivors are injected into the corpus
        and the pipeline re-runs from Stage 2 (ZX conversion) onward.

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

        # -- Initial pass (round 0) -------------------------------------------
        for stage_name in self.STAGES[start_idx : end_idx + 1]:
            logger.info("Running stage: %s", stage_name)
            self.run_stage(stage_name)

        # -- Iterative refinement (rounds 1..N) --------------------------------
        n_rounds = self.config.refinement_rounds
        if n_rounds <= 0:
            return

        # Refinement only makes sense when we ran the full pipeline.
        bench_idx = self._stage_index("bench")
        if end_idx < bench_idx:
            logger.info("Refinement skipped: pipeline did not reach 'bench' stage.")
            return

        for round_num in range(1, n_rounds + 1):
            logger.info(
                "=== Iterative refinement round %d/%d ===", round_num, n_rounds,
            )

            # Build fitness profile from benchmarking results before injecting
            # survivors. This captures recipe attribution and near-miss info.
            if self.config.fitness_guided:
                self._build_and_save_fitness_profile(round_num)

            # Phase-optimise near-miss candidates before injection.
            if (
                self.config.phase_optimize_near_misses
                and self._fitness_profile
                and self._fitness_profile.near_miss_candidates
            ):
                self._run_phase_optimisation()

            n_injected = self._inject_survivors_into_corpus(round_num)
            if n_injected == 0:
                logger.info("No survivors to inject; stopping refinement.")
                break

            # Clear data dirs for stages 2-7 (keep corpus).
            for subdir in ["zx_diagrams", "mined_webs", "candidates",
                           "filtered", "benchmarks", "report"]:
                d = self.data_dir / subdir
                if d.exists():
                    shutil.rmtree(d)

            # Re-run from ZX conversion through reporting.
            refine_start = max(start_idx, self._stage_index("zx"))
            for stage_name in self.STAGES[refine_start : end_idx + 1]:
                logger.info("Running stage: %s (round %d)", stage_name, round_num)
                self.run_stage(stage_name)

    def _inject_survivors_into_corpus(self, round_num: int) -> int:
        """Inject top survivors from benchmarking back into the corpus.

        Reads the benchmark results, selects the top-K survivors by
        novelty score (or by highest best_fidelity if novelty is unavailable),
        writes their QASM as new corpus entries under a ``discovered_rN``
        family, and updates the corpus manifest.

        Returns the number of survivors injected.
        """
        top_k = self.config.refinement_top_k
        bench_dir = self.data_dir / "benchmarks"
        results_path = bench_dir / "results.json"
        if not results_path.exists():
            return 0

        results_data = json.loads(results_path.read_text())
        if not results_data:
            return 0

        # Sort by novelty (descending), falling back to 1 - best_fidelity.
        for r in results_data:
            if "novelty_score" not in r:
                r["novelty_score"] = 1.0 - r.get("best_fidelity", 1.0)
        results_data.sort(key=lambda r: r["novelty_score"], reverse=True)

        # Select top K.
        top_survivors = results_data[:top_k]

        # Load QASM for each survivor from the filtered directory.
        filtered_dir = self.data_dir / "filtered"
        filtered_manifest = load_manifest(filtered_dir)
        if not filtered_manifest:
            return 0

        # Build a lookup from survivor_id to circuit data.
        surv_lookup: dict[str, dict] = {}
        for entry in filtered_manifest:
            sid = entry.get("survivor_id", "")
            surv_lookup[sid] = entry

        # Inject into corpus.
        corpus_dir = self.data_dir / "corpus"
        corpus_manifest = load_manifest(corpus_dir) or []
        family_name = f"discovered_r{round_num}"
        family_dir = corpus_dir / "algorithms" / family_name
        family_dir.mkdir(parents=True, exist_ok=True)

        injected = 0
        for r in top_survivors:
            sid = r.get("survivor_id", "")
            entry = surv_lookup.get(sid)
            if not entry:
                continue

            # Load QASM from the circuit JSON file.
            circuit_path = entry.get("circuit_path", "")
            if not circuit_path or not Path(circuit_path).exists():
                continue
            circ_data = json.loads(Path(circuit_path).read_text())
            qasm = circ_data.get("circuit_qasm", "")
            if not qasm:
                continue

            n_qubits = r.get("n_qubits", entry.get("n_qubits", 0))
            alg_id = f"{family_name}/{sid}"

            # Write QASM file.
            qasm_path = family_dir / f"{sid}_{n_qubits}q.qasm"
            qasm_path.write_text(qasm)

            # Add to corpus manifest.
            corpus_manifest.append({
                "algorithm_id": alg_id,
                "family": family_name,
                "name": sid,
                "n_qubits": n_qubits,
                "qasm_path": str(qasm_path),
            })
            injected += 1

        save_manifest(corpus_manifest, corpus_dir)
        logger.info(
            "Refinement round %d: injected %d survivors into corpus as '%s' "
            "(corpus now has %d entries).",
            round_num, injected, family_name, len(corpus_manifest),
        )
        return injected

    def _build_and_save_fitness_profile(self, round_num: int) -> None:
        """Build a fitness profile from the latest benchmark results.

        Analyses Stage 6 results to attribute fitness to web IDs,
        composition strategies, and family combinations.  Merges with
        the existing profile (decaying older signals) and persists to disk.
        """
        bench_dir = self.data_dir / "benchmarks"
        results_path = bench_dir / "results.json"
        if not results_path.exists():
            return

        bench_results = json.loads(results_path.read_text())
        if not bench_results:
            return

        # Load candidate manifest for provenance info.
        cand_dir = self.data_dir / "candidates"
        cand_manifest = load_manifest(cand_dir)

        # Also include filtered manifest (which links survivor_id to candidate_id).
        filtered_manifest = load_manifest(self.data_dir / "filtered")
        combined_manifest = cand_manifest + filtered_manifest

        new_profile = build_fitness_profile(
            bench_results=bench_results,
            candidate_manifest=combined_manifest,
            round_num=round_num,
            near_miss_lo=self.config.near_miss_fidelity_lo,
            near_miss_hi=self.config.near_miss_fidelity_hi,
        )

        self._fitness_profile = merge_profiles(
            self._fitness_profile, new_profile,
            decay=self.config.fitness_decay,
        )
        save_fitness_profile(self._fitness_profile, self.data_dir)
        logger.info(
            "Fitness profile updated (round %d): %d webs, %d near-misses.",
            round_num,
            len(self._fitness_profile.web_fitness),
            len(self._fitness_profile.near_miss_candidates),
        )

    def _run_phase_optimisation(self) -> None:
        """Run Nelder-Mead phase optimisation on near-miss candidates.

        Near-miss candidates (fidelity 0.80-0.99 to a known task) are
        optimised to maximise fidelity.  Successful optimisations are
        injected into the filtered directory as new survivors.
        """
        from zx_webs.phase_optimizer import optimize_near_misses

        if not self._fitness_profile:
            return

        near_misses = self._fitness_profile.near_miss_candidates
        if not near_misses:
            return

        logger.info(
            "Phase optimisation: processing %d near-miss candidates.",
            len(near_misses),
        )

        results = optimize_near_misses(
            near_miss_candidates=near_misses,
            filtered_dir=self.data_dir / "filtered",
            corpus_dir=self.data_dir / "corpus",
            max_iterations=self.config.phase_optimize_max_iters,
            fidelity_threshold=self.config.near_miss_fidelity_hi,
            max_qubits=self.config.filter.max_unitary_qubits,
        )

        # Inject successful phase-optimised circuits into filtered output.
        if not results:
            return

        filtered_dir = self.data_dir / "filtered"
        filtered_manifest = load_manifest(filtered_dir)
        circuits_dir = filtered_dir / "circuits"
        circuits_dir.mkdir(parents=True, exist_ok=True)

        injected = 0
        for r in results:
            if not r.get("success") or not r.get("optimized_qasm"):
                continue
            if r["optimized_fidelity"] < self.config.near_miss_fidelity_hi:
                continue

            sid = f"phaseopt_{r['survivor_id']}"
            circuit_path = circuits_dir / f"{sid}.json"
            circuit_data = {
                "circuit_qasm": r["optimized_qasm"],
                "graph_json": r.get("optimized_graph_json", ""),
                "phase_optimized": True,
                "original_survivor_id": r["survivor_id"],
                "target_task": r.get("target_task", ""),
                "initial_fidelity": r["initial_fidelity"],
                "optimized_fidelity": r["optimized_fidelity"],
            }
            circuit_path.write_text(json.dumps(circuit_data, indent=2))

            filtered_manifest.append({
                "survivor_id": sid,
                "circuit_path": str(circuit_path),
                "n_qubits": 0,  # will be recomputed in Stage 6
                "phase_optimized": True,
            })
            injected += 1

        if injected > 0:
            save_manifest(filtered_manifest, filtered_dir)
            logger.info(
                "Phase optimisation: injected %d optimised circuits.", injected,
            )

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
            # Pass fitness weights from previous rounds to bias web selection.
            fw = None
            if self.config.fitness_guided and self._fitness_profile:
                fw = self._fitness_profile.web_fitness or None
            run_stage4(
                self.data_dir / "mined_webs",
                self.data_dir / "candidates",
                self.config.compose,
                webs_in_memory=webs_in_memory,
                fitness_weights=fw,
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
