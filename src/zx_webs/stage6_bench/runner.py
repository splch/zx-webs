"""Stage 6 orchestrator -- benchmark surviving candidates.

:func:`run_stage6` loads the circuits that survived Stage 5 filtering,
compares them against the original corpus baselines using circuit-level
metrics and (optionally) SupermarQ features, and persists the results.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from zx_webs.config import BenchConfig
from zx_webs.persistence import load_manifest, save_json, save_manifest
from zx_webs.stage6_bench.comparator import compare_candidate_to_baselines
from zx_webs.stage6_bench.metrics import CircuitMetrics, SupermarQFeatures

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
    """Run Stage 6: benchmark surviving candidates.

    Workflow
    -------
    1. Load filtered circuits from Stage 5.
    2. Load original corpus circuits as baselines.
    3. Compute metrics and SupermarQ features for each survivor.
    4. Compare candidates against baselines.
    5. Identify candidates that Pareto-dominate at least one baseline.
    6. Persist results to *output_dir*.

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

    # -- 1. Load Stage 5 survivors -------------------------------------------
    filtered_manifest = load_manifest(filtered_dir)
    if not filtered_manifest:
        logger.warning("Stage 5 manifest at %s is empty -- nothing to benchmark.", filtered_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json([], output_dir / "results.json")
        save_manifest([], output_dir)
        return []

    # -- 2. Load baselines from corpus ---------------------------------------
    corpus_manifest = load_manifest(corpus_dir)
    baselines: list[dict[str, Any]] = []
    for item in corpus_manifest:
        qasm_text = _load_qasm_text(item)
        if qasm_text:
            baselines.append(
                {
                    "id": item.get("algorithm_id", item.get("name", "unknown")),
                    "qasm": qasm_text,
                    "family": item.get("family", ""),
                }
            )

    logger.info(
        "Stage 6: loaded %d survivors and %d baselines.",
        len(filtered_manifest),
        len(baselines),
    )

    # -- 3-5. Compute metrics & compare --------------------------------------
    results: list[dict[str, Any]] = []

    for survivor in tqdm(filtered_manifest, desc="Stage 6: Benchmarking", unit="survivor"):
        survivor_id = survivor.get("survivor_id", "unknown")
        qasm_str = _load_qasm_text(survivor)
        if not qasm_str:
            logger.debug("Survivor %s has no readable QASM -- skipping.", survivor_id)
            continue

        # Compute metrics.
        try:
            metrics = CircuitMetrics.from_qasm(qasm_str)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to compute metrics for %s.", survivor_id)
            continue

        # Compute SupermarQ features.
        try:
            features = SupermarQFeatures.from_qasm(qasm_str)
        except Exception:  # noqa: BLE001
            features = SupermarQFeatures()

        # Compare against baselines.
        comparisons = compare_candidate_to_baselines(survivor_id, qasm_str, baselines)

        dominates_any = any(c.candidate_dominates for c in comparisons)

        results.append(
            {
                "survivor_id": survivor_id,
                "metrics": metrics.to_dict(),
                "features": features.to_dict(),
                "dominates_any_baseline": dominates_any,
                "n_baselines_dominated": sum(1 for c in comparisons if c.candidate_dominates),
                "comparisons": [
                    {
                        "baseline_id": c.baseline_id,
                        "candidate_dominates": c.candidate_dominates,
                        "baseline_dominates": c.baseline_dominates,
                        "improvements": c.improvements,
                    }
                    for c in comparisons
                ],
            }
        )

    # -- 6. Persist results --------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, output_dir / "results.json")
    save_manifest(results, output_dir)

    n_dominant = sum(1 for r in results if r.get("dominates_any_baseline"))
    logger.info(
        "Stage 6 complete: %d/%d candidates dominate at least one baseline.",
        n_dominant,
        len(results),
    )
    return results
