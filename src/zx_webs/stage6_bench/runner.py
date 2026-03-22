"""Stage 6 orchestrator -- functional benchmarking of surviving candidates.

:func:`run_stage6` loads the circuits that survived Stage 5 filtering,
builds benchmark tasks from the algorithm corpus, and evaluates each
survivor by:

1. Computing its unitary matrix.
2. Matching it against tasks of the same qubit count via process fidelity.
3. Identifying real improvements: high fidelity (>= threshold) AND fewer gates.
4. Persisting detailed results.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from zx_webs.config import BenchConfig
from zx_webs.persistence import load_manifest, save_json, save_manifest
from zx_webs.stage6_bench.comparator import match_candidate_to_tasks
from zx_webs.stage6_bench.metrics import (
    CircuitMetrics,
    SupermarQFeatures,
    compute_unitary,
    entanglement_capacity,
    is_clifford_unitary,
    novelty_score,
)
from zx_webs.stage6_bench.tasks import build_benchmark_tasks

logger = logging.getLogger(__name__)


def _load_qasm_text(entry: dict[str, Any]) -> str:
    """Return QASM text for a manifest entry.

    Stage 5 survivors store QASM either inline (``circuit_qasm`` key in the
    circuit JSON file) or at an external path.  Corpus entries use
    ``qasm_path``.  This helper tries each strategy in turn.
    """
    # Direct QASM content (used by Stage 5 circuit JSON files).
    if entry.get("circuit_qasm"):
        return entry["circuit_qasm"]

    # Path to a QASM file on disk.
    for key in ("qasm_path", "circuit_path"):
        raw = entry.get(key, "")
        if raw:
            p = Path(raw)
            if p.exists():
                text = p.read_text()
                # circuit_path may point to a JSON wrapper; check.
                if text.lstrip().startswith("{"):
                    import json

                    data = json.loads(text)
                    if "circuit_qasm" in data:
                        return data["circuit_qasm"]
                else:
                    return text
    return ""


def run_stage6(
    filtered_dir: Path,
    corpus_dir: Path,
    output_dir: Path,
    config: BenchConfig | None = None,
) -> list[dict[str, Any]]:
    """Run Stage 6: functional benchmarking of surviving candidates.

    Workflow
    -------
    1. Build benchmark tasks from the algorithm corpus (target unitaries).
    2. Load filtered circuits from Stage 5.
    3. For each survivor:
       a. Compute metrics and SupermarQ features.
       b. Compute unitary and classify (Clifford, entanglement capacity).
       c. Match against all tasks of the same qubit count via fidelity.
       d. Identify real improvements (high fidelity + fewer gates).
    4. Persist results to *output_dir*.

    Parameters
    ----------
    filtered_dir:
        Directory containing Stage 5 output (``manifest.json`` and
        ``circuits/*.json``).
    corpus_dir:
        Directory containing Stage 1 corpus (``manifest.json`` and QASM
        files).
    output_dir:
        Where Stage 6 artefacts will be written.
    config:
        Benchmarking parameters.  Falls back to ``BenchConfig()`` defaults
        when *None*.

    Returns
    -------
    list[dict]
        Benchmark results, one entry per survivor.
    """
    if config is None:
        config = BenchConfig()

    fidelity_threshold = getattr(config, "fidelity_threshold", 0.99)
    max_unitary_qubits = getattr(config, "max_unitary_qubits", 10)
    novelty_enabled = getattr(config, "novelty_scoring", False)

    # -- 1. Build benchmark tasks from the corpus ----------------------------
    # Derive qubit counts from the survivor manifest so tasks match actual data.
    filtered_manifest = load_manifest(filtered_dir)
    survivor_qubit_counts: list[int] = sorted(set(
        entry.get("n_qubits", 0) for entry in (filtered_manifest or [])
        if entry.get("n_qubits", 0) > 0
    ))

    try:
        tasks = build_benchmark_tasks(
            qubit_counts=survivor_qubit_counts if survivor_qubit_counts else None,
            max_unitary_qubits=max_unitary_qubits,
        )
    except Exception:
        logger.warning(
            "Failed to build benchmark tasks from corpus; "
            "proceeding with empty task list.",
            exc_info=True,
        )
        tasks = []

    task_qubit_counts = sorted(set(t.n_qubits for t in tasks))
    logger.info(
        "Stage 6: built %d benchmark tasks for qubit counts %s.",
        len(tasks),
        task_qubit_counts,
    )

    # -- 2. Load Stage 5 survivors -------------------------------------------
    # Re-use the manifest we already loaded above.
    if not filtered_manifest:
        logger.warning(
            "Stage 5 manifest at %s is empty -- nothing to benchmark.",
            filtered_dir,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json([], output_dir / "results.json")
        save_manifest([], output_dir)
        return []

    logger.info(
        "Stage 6: loaded %d survivors, matching against %d tasks.",
        len(filtered_manifest),
        len(tasks),
    )

    # -- 2b. Collect corpus unitaries for novelty scoring --------------------
    import numpy as np

    corpus_unitaries_by_qubits: dict[int, list[np.ndarray]] = {}
    if novelty_enabled:
        for task in tasks:
            nq = task.n_qubits
            if nq not in corpus_unitaries_by_qubits:
                corpus_unitaries_by_qubits[nq] = []
            corpus_unitaries_by_qubits[nq].append(task.target_unitary)

    # -- 3. Benchmark each survivor ------------------------------------------
    results: list[dict[str, Any]] = []

    for survivor in tqdm(
        filtered_manifest,
        desc="Stage 6: Benchmarking",
        unit="survivor",
    ):
        survivor_id = survivor.get("survivor_id", "unknown")
        qasm_str = _load_qasm_text(survivor)
        if not qasm_str:
            logger.debug("Survivor %s has no readable QASM -- skipping.", survivor_id)
            continue

        # 3a. Compute circuit metrics.
        try:
            metrics = CircuitMetrics.from_qasm(qasm_str)
        except Exception:
            logger.warning("Failed to compute metrics for %s.", survivor_id)
            continue

        # 3a. Compute SupermarQ features.
        try:
            features = SupermarQFeatures.from_qasm(qasm_str)
        except Exception:
            features = SupermarQFeatures()

        # 3b. Compute unitary and classify.
        unitary = compute_unitary(qasm_str, max_unitary_qubits=max_unitary_qubits)
        classification: dict[str, Any] = {}
        if unitary is not None:
            classification["is_clifford"] = is_clifford_unitary(unitary)
            classification["entanglement_capacity"] = entanglement_capacity(unitary)
        else:
            classification["is_clifford"] = None
            classification["entanglement_capacity"] = None

        # 3c. Compute novelty score.
        nov_score: float | None = None
        if novelty_enabled and unitary is not None:
            corpus_us = corpus_unitaries_by_qubits.get(metrics.qubit_count, [])
            nov_score = novelty_score(unitary, corpus_us)

        # 3d. Match against benchmark tasks.
        matches = match_candidate_to_tasks(
            candidate_id=survivor_id,
            candidate_qasm=qasm_str,
            tasks=tasks,
            fidelity_threshold=fidelity_threshold,
            max_unitary_qubits=max_unitary_qubits,
        )

        # 3e. Identify real improvements.
        real_improvements = [m for m in matches if m.is_improvement]
        best_fidelity = max((m.fidelity for m in matches), default=0.0)
        dominates_any = len(real_improvements) > 0

        result_entry: dict[str, Any] = {
            "survivor_id": survivor_id,
            "metrics": metrics.to_dict(),
            "features": features.to_dict(),
            "classification": classification,
            "n_qubits": metrics.qubit_count,
            # Backward-compatible fields.
            "dominates_any_baseline": dominates_any,
            "n_baselines_dominated": len(real_improvements),
            # New functional benchmarking fields.
            "best_fidelity": best_fidelity,
            "n_task_matches": len(matches),
            "n_improvements": len(real_improvements),
            "task_matches": [m.to_dict() for m in matches],
            "improvements": [m.to_dict() for m in real_improvements],
        }
        if nov_score is not None:
            result_entry["novelty_score"] = nov_score
        results.append(result_entry)

    # -- 4. Persist results --------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, output_dir / "results.json")
    save_manifest(results, output_dir)

    n_with_matches = sum(1 for r in results if r.get("n_task_matches", 0) > 0)
    n_improvements = sum(1 for r in results if r.get("n_improvements", 0) > 0)
    logger.info(
        "Stage 6 complete: %d/%d survivors matched tasks, "
        "%d have real improvements (fidelity >= %.2f + fewer gates).",
        n_with_matches,
        len(results),
        n_improvements,
        fidelity_threshold,
    )
    if novelty_enabled:
        novelty_scores = [r["novelty_score"] for r in results if "novelty_score" in r]
        if novelty_scores:
            high_novelty = sum(1 for s in novelty_scores if s >= 0.5)
            logger.info(
                "Novelty: %d/%d survivors scored (mean=%.3f, max=%.3f, "
                "%d with novelty >= 0.5).",
                len(novelty_scores), len(results),
                sum(novelty_scores) / len(novelty_scores),
                max(novelty_scores),
                high_novelty,
            )
    return results
