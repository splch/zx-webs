#!/usr/bin/env python3
"""Run the ZX-Webs pipeline with BYOL-Explore curiosity-driven composition.

This script:
1. Runs Stages 1-3 (corpus, ZX conversion, mining) normally
2. Replaces Stage 4 with BYOL-Explore curiosity-driven composition
3. Also runs standard Stage 4 for comparison
4. Runs Stages 5-7 (filter, benchmark, report) on both result sets
5. Compares the discoveries from curiosity-driven vs standard exploration

Usage:
    python scripts/run_byol_explore.py [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zx_webs.config import PipelineConfig, load_config
from zx_webs.persistence import load_manifest, save_json, save_manifest
from zx_webs.pipeline import Pipeline, run_stage1
from zx_webs.stage2_zx import run_stage2
from zx_webs.stage3_mining.miner import run_stage3
from zx_webs.stage4_compose.stitcher import Stitcher, run_stage4
from zx_webs.stage5_filter.extractor import run_stage5
from zx_webs.stage6_bench.runner import run_stage6
from zx_webs.stage7_report.reporter import run_stage7
from zx_webs.byol_explore import (
    BYOLExploreAgent,
    extract_graph_features,
    run_curiosity_exploration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy Qiskit transpiler logs
logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def make_config() -> PipelineConfig:
    """Create a configuration tuned for BYOL-Explore discovery.

    Uses multiple algorithm families for cross-family recombination,
    moderate mining parameters, and enables novelty scoring.
    """
    return PipelineConfig.model_validate({
        "data_dir": "data_byol",
        "corpus": {
            "families": [
                "oracular",
                "arithmetic",
                "variational",
                "simulation",
                "entanglement",
                "error_correction",
                "linear_algebra",
                "communication",
            ],
            "max_qubits": 7,
            "qubit_counts": [3, 5, 7],
            "seed": 42,
        },
        "zx": {
            "reduction": "full_reduce",
            "mining_reduction": "teleport_reduce",
            "normalize": True,
        },
        "mining": {
            "min_support": 2,
            "min_vertices": 2,
            "max_vertices": 10,
            "phase_discretization": 8,
            "include_phase_in_label": True,
            "mining_reduction": "teleport_reduce",
            "mining_timeout": 300,
            "discriminative_mining": True,
            "discriminative_min_support": 2,
            "discriminative_max_family_ratio": 0.5,
        },
        "compose": {
            "max_webs_per_candidate": 3,
            "max_candidates": 2000,
            "max_webs_loaded": 50000,
            "composition_modes": ["sequential", "parallel"],
            "min_compose_qubits": 2,
            "max_compose_qubits": 10,
            "seed": 42,
            "prefer_cross_family": True,
            "guided": True,
            "target_qubit_counts": [3, 5, 7],
            "phase_perturbation_resolution": 8,
            "phase_perturbation_rate": 0.3,
            "continuous_phase_perturbation": True,
        },
        "filter": {
            "extract_timeout_seconds": 30.0,
            "max_cnot_blowup_factor": 5.0,
            "cnot_blowup_enabled": True,
            "dedup_method": "unitary",
            "optimize_cnots_level": 2,
            "gflow_precheck": False,
            "max_unitary_qubits": 10,
        },
        "bench": {
            "tasks": ["grover_oracle", "maxcut", "qpe", "ghz", "arithmetic"],
            "qasmbench_path": "data/qasmbench",
            "fidelity_shots": 8192,
            "fidelity_threshold": 0.99,
            "novelty_scoring": True,
            "max_unitary_qubits": 10,
        },
        "report": {
            "output_format": ["json"],
            "figure_dpi": 150,
        },
    })


def run_stages_1_to_3(config: PipelineConfig) -> list:
    """Run the standard pipeline stages 1-3 and return mined webs."""
    data_dir = Path(config.data_dir)

    # Stage 1: Corpus
    logger.info("=" * 60)
    logger.info("STAGE 1: Building algorithm corpus")
    logger.info("=" * 60)
    run_stage1(config)

    # Stage 2: ZX conversion
    logger.info("=" * 60)
    logger.info("STAGE 2: ZX-diagram conversion")
    logger.info("=" * 60)
    run_stage2(
        data_dir / "corpus",
        data_dir / "zx_diagrams",
        config.zx,
    )

    # Stage 3: Mining
    logger.info("=" * 60)
    logger.info("STAGE 3: Frequent sub-diagram mining")
    logger.info("=" * 60)
    webs = run_stage3(
        data_dir / "zx_diagrams",
        data_dir / "mined_webs",
        config.mining,
        corpus_dir=data_dir / "corpus",
        skip_bulk_write=True,
    )

    logger.info("Mining produced %d webs", len(webs))
    return webs


def run_byol_composition(
    webs: list,
    config: PipelineConfig,
    n_episodes: int = 5,
    steps_per_episode: int = 300,
) -> list:
    """Run BYOL-Explore curiosity-driven composition."""
    logger.info("=" * 60)
    logger.info("STAGE 4a: BYOL-Explore curiosity-driven composition")
    logger.info("=" * 60)

    stitcher = Stitcher(config.compose)

    candidates, exploration_log = run_curiosity_exploration(
        webs=webs,
        stitcher=stitcher,
        config=config.compose,
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
        seed=config.compose.seed,
    )

    # Save exploration log
    data_dir = Path(config.data_dir)
    log_dir = data_dir / "byol_exploration"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_json(exploration_log, log_dir / "exploration_log.json")

    # Save candidates
    output_dir = data_dir / "candidates_byol"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for cand in candidates:
        cand_path = candidates_dir / f"{cand.candidate_id}.json"
        save_json(cand.to_dict(), cand_path)
        manifest_entries.append({
            "candidate_id": cand.candidate_id,
            "candidate_path": str(cand_path),
            "composition_type": cand.composition_type,
            "component_web_ids": cand.component_web_ids,
            "n_qubits": cand.n_qubits,
            "n_spiders": cand.n_spiders,
            "source_families": cand.source_families,
            "is_cross_family": cand.is_cross_family,
        })

    save_manifest(manifest_entries, output_dir)
    logger.info("BYOL-Explore: saved %d candidates", len(candidates))

    return candidates


def run_standard_composition(webs: list, config: PipelineConfig) -> list:
    """Run standard (brute-force) composition for comparison."""
    logger.info("=" * 60)
    logger.info("STAGE 4b: Standard composition (baseline)")
    logger.info("=" * 60)

    data_dir = Path(config.data_dir)
    return run_stage4(
        data_dir / "mined_webs",
        data_dir / "candidates_standard",
        config.compose,
        webs_in_memory=webs,
    )


def run_filter_and_bench(
    config: PipelineConfig,
    candidates_dir: str,
    label: str,
) -> list:
    """Run stages 5-6 on a set of candidates."""
    data_dir = Path(config.data_dir)
    filtered_dir = data_dir / f"filtered_{label}"
    bench_dir = data_dir / f"benchmarks_{label}"

    logger.info("STAGE 5 (%s): Filtering candidates", label)
    run_stage5(
        data_dir / candidates_dir,
        filtered_dir,
        config.filter,
    )

    logger.info("STAGE 6 (%s): Benchmarking survivors", label)
    results = run_stage6(
        filtered_dir,
        data_dir / "corpus",
        bench_dir,
        config.bench,
    )

    return results


def analyze_results(
    byol_results: list,
    standard_results: list,
    config: PipelineConfig,
) -> dict:
    """Compare BYOL-Explore vs standard composition results."""
    data_dir = Path(config.data_dir)

    def summarize(results: list, label: str) -> dict:
        if not results:
            return {"label": label, "n_survivors": 0}

        n_survivors = len(results)
        n_with_matches = sum(1 for r in results if r.get("n_task_matches", 0) > 0)
        n_improvements = sum(1 for r in results if r.get("n_improvements", 0) > 0)

        novelty_scores = [r.get("novelty_score", 0) for r in results if "novelty_score" in r]
        mean_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
        max_novelty = max(novelty_scores) if novelty_scores else 0
        high_novelty = sum(1 for s in novelty_scores if s >= 0.5)

        fidelities = [r.get("best_fidelity", 0) for r in results]
        mean_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0

        # Entanglement capacity
        ent_caps = [
            r.get("classification", {}).get("entanglement_capacity", 0)
            for r in results
            if r.get("classification", {}).get("entanglement_capacity") is not None
        ]
        mean_ent = sum(ent_caps) / len(ent_caps) if ent_caps else 0

        # Clifford classification
        n_clifford = sum(
            1 for r in results
            if r.get("classification", {}).get("is_clifford") is True
        )
        n_non_clifford = sum(
            1 for r in results
            if r.get("classification", {}).get("is_clifford") is False
        )

        # Gate count stats
        gate_counts = [r.get("metrics", {}).get("total_gates", 0) for r in results]
        t_counts = [r.get("metrics", {}).get("t_count", 0) for r in results]
        cnot_counts = [r.get("metrics", {}).get("cnot_count", 0) for r in results]

        # Qubit distribution
        qubit_counts = [r.get("n_qubits", 0) for r in results]
        qubit_dist = {}
        for q in qubit_counts:
            qubit_dist[str(q)] = qubit_dist.get(str(q), 0) + 1

        # Improvement details
        improvement_details = []
        for r in results:
            for imp in r.get("improvements", []):
                improvement_details.append({
                    "survivor_id": r["survivor_id"],
                    "task": imp["task_name"],
                    "fidelity": imp["fidelity"],
                    "gate_improvement": imp["gate_improvement"],
                    "candidate_gates": imp["candidate_metrics"]["total_gates"],
                    "baseline_gates": imp["baseline_metrics"]["total_gates"],
                    "candidate_t_count": imp["candidate_metrics"]["t_count"],
                    "baseline_t_count": imp["baseline_metrics"]["t_count"],
                })

        return {
            "label": label,
            "n_survivors": n_survivors,
            "n_with_task_matches": n_with_matches,
            "n_improvements": n_improvements,
            "mean_novelty": mean_novelty,
            "max_novelty": max_novelty,
            "high_novelty_count": high_novelty,
            "mean_fidelity": mean_fidelity,
            "mean_entanglement_capacity": mean_ent,
            "n_clifford": n_clifford,
            "n_non_clifford": n_non_clifford,
            "mean_total_gates": sum(gate_counts) / len(gate_counts) if gate_counts else 0,
            "mean_t_count": sum(t_counts) / len(t_counts) if t_counts else 0,
            "mean_cnot_count": sum(cnot_counts) / len(cnot_counts) if cnot_counts else 0,
            "qubit_distribution": qubit_dist,
            "improvements": improvement_details,
        }

    byol_summary = summarize(byol_results, "byol_explore")
    std_summary = summarize(standard_results, "standard")

    comparison = {
        "byol_explore": byol_summary,
        "standard": std_summary,
        "comparison": {
            "byol_novelty_advantage": (
                byol_summary.get("mean_novelty", 0) - std_summary.get("mean_novelty", 0)
            ),
            "byol_more_improvements": (
                byol_summary.get("n_improvements", 0) > std_summary.get("n_improvements", 0)
            ),
            "byol_survivors": byol_summary.get("n_survivors", 0),
            "std_survivors": std_summary.get("n_survivors", 0),
        },
    }

    # Save comparison
    save_json(comparison, data_dir / "byol_vs_standard_comparison.json")

    return comparison


def print_report(comparison: dict) -> None:
    """Print a human-readable comparison report."""
    byol = comparison["byol_explore"]
    std = comparison["standard"]

    print("\n" + "=" * 70)
    print("  BYOL-Explore vs Standard Composition: Results")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'BYOL-Explore':>15} {'Standard':>15}")
    print("-" * 65)

    metrics = [
        ("Survivors (extracted circuits)", "n_survivors"),
        ("Task matches", "n_with_task_matches"),
        ("Real improvements", "n_improvements"),
        ("Mean novelty score", "mean_novelty"),
        ("Max novelty score", "max_novelty"),
        ("High novelty (>= 0.5)", "high_novelty_count"),
        ("Mean fidelity", "mean_fidelity"),
        ("Mean entanglement capacity", "mean_entanglement_capacity"),
        ("Clifford circuits", "n_clifford"),
        ("Non-Clifford circuits", "n_non_clifford"),
        ("Mean total gates", "mean_total_gates"),
        ("Mean T-count", "mean_t_count"),
        ("Mean CNOT count", "mean_cnot_count"),
    ]

    for label, key in metrics:
        b_val = byol.get(key, 0)
        s_val = std.get(key, 0)
        if isinstance(b_val, float):
            print(f"{label:<35} {b_val:>15.4f} {s_val:>15.4f}")
        else:
            print(f"{label:<35} {b_val:>15} {s_val:>15}")

    print(f"\n{'Qubit distribution':}")
    for label_str, summary in [("BYOL", byol), ("Standard", std)]:
        dist = summary.get("qubit_distribution", {})
        dist_str = ", ".join(f"{k}q:{v}" for k, v in sorted(dist.items()))
        print(f"  {label_str}: {dist_str}")

    # Print improvements
    for label_str, summary in [("BYOL-Explore", byol), ("Standard", std)]:
        imps = summary.get("improvements", [])
        if imps:
            print(f"\n{label_str} improvements:")
            for imp in imps[:10]:  # Show top 10
                gate_imps = imp.get("gate_improvement", {})
                print(
                    f"  {imp['survivor_id']}: matched {imp['task']} "
                    f"(fidelity={imp['fidelity']:.4f}, "
                    f"gates {imp['baseline_gates']}->{imp['candidate_gates']}, "
                    f"T-count {imp['baseline_t_count']}->{imp['candidate_t_count']})"
                )

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ZX-Webs with BYOL-Explore curiosity-driven composition"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config (default: use built-in discovery config)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of BYOL-Explore episodes (default: 5)",
    )
    parser.add_argument(
        "--steps", type=int, default=300,
        help="Steps per episode (default: 300)",
    )
    parser.add_argument(
        "--skip-standard", action="store_true",
        help="Skip standard composition (only run BYOL-Explore)",
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = make_config()

    start_time = time.time()

    # Run stages 1-3 (shared between both methods)
    webs = run_stages_1_to_3(config)

    if not webs:
        logger.error("No webs produced by mining. Exiting.")
        return

    # Run BYOL-Explore composition
    byol_candidates = run_byol_composition(
        webs, config,
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
    )

    # Run standard composition for comparison
    if not args.skip_standard:
        std_candidates = run_standard_composition(webs, config)
    else:
        std_candidates = []

    # Filter and benchmark both sets
    byol_results = run_filter_and_bench(config, "candidates_byol", "byol")

    if not args.skip_standard:
        std_results = run_filter_and_bench(config, "candidates_standard", "standard")
    else:
        std_results = []

    # Analyze and compare
    comparison = analyze_results(byol_results, std_results, config)
    print_report(comparison)

    elapsed = time.time() - start_time
    logger.info("Total elapsed time: %.1f seconds", elapsed)

    # Save final report
    save_json(comparison, Path(config.data_dir) / "final_report.json")
    logger.info(
        "Results saved to %s/final_report.json",
        config.data_dir,
    )


if __name__ == "__main__":
    main()
