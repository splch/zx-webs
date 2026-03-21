"""Stage 2 -- convert QASM strings to simplified ZX-diagrams."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyzx as zx
from tqdm import tqdm

from zx_webs.config import ZXConfig
from zx_webs.persistence import load_manifest, save_graph_json, save_manifest
from zx_webs.stage2_zx.graph_stats import compute_graph_stats
from zx_webs.stage2_zx.simplifier import simplify_graph

logger = logging.getLogger(__name__)


def qasm_to_zx_graph(
    qasm_str: str,
    config: ZXConfig | None = None,
) -> tuple[zx.Graph, dict[str, Any]]:
    """Convert a QASM string to a simplified ZX-diagram.

    Parameters
    ----------
    qasm_str:
        An OpenQASM 2.0 source string.
    config:
        ZX conversion / simplification parameters.  Falls back to
        ``ZXConfig()`` defaults when *None*.

    Returns
    -------
    (simplified_graph, info)
        *info* is a dict with:

        - ``pre_stats``  -- graph statistics **before** simplification
        - ``post_stats`` -- graph statistics **after** simplification
        - ``reduction_method`` -- the simplification method that was applied
    """
    if config is None:
        config = ZXConfig()

    circuit = zx.Circuit.from_qasm(qasm_str)
    g: zx.Graph = circuit.to_graph()

    pre_stats = compute_graph_stats(g)

    g_simplified = simplify_graph(
        g,
        method=config.reduction,
        normalize=config.normalize,
    )

    post_stats = compute_graph_stats(g_simplified)

    info: dict[str, Any] = {
        "pre_stats": pre_stats,
        "post_stats": post_stats,
        "reduction_method": config.reduction,
    }

    return g_simplified, info


def run_stage2(
    corpus_dir: Path,
    output_dir: Path,
    config: ZXConfig | None = None,
) -> list[dict[str, Any]]:
    """Run Stage 2 on all circuits produced by Stage 1.

    Reads ``corpus_dir/manifest.json``, converts every QASM file to a
    simplified ZX-diagram, and persists the results under *output_dir*.

    Outputs
    -------
    - ``output_dir/graphs/{algorithm_id}.zxg.json`` -- one file per circuit
    - ``output_dir/manifest.json`` -- index of all converted graphs

    Parameters
    ----------
    corpus_dir:
        Directory that contains the Stage 1 corpus manifest and QASM files.
    output_dir:
        Where Stage 2 artefacts will be written.
    config:
        ZX conversion / simplification parameters.

    Returns
    -------
    list[dict]
        The entries written to the output manifest.
    """
    if config is None:
        config = ZXConfig()

    corpus_manifest = load_manifest(corpus_dir)
    if not corpus_manifest:
        logger.warning("Stage 1 manifest at %s is empty -- nothing to convert.", corpus_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    for item in tqdm(corpus_manifest, desc="Stage 2: Converting to ZX", unit="graph"):
        algorithm_id: str = item["algorithm_id"]
        qasm_path = Path(item["qasm_path"])

        logger.info("Converting %s (%s) ...", algorithm_id, qasm_path.name)

        qasm_str = qasm_path.read_text()
        g, info = qasm_to_zx_graph(qasm_str, config)

        # Persist the simplified graph as JSON.
        graph_path = graphs_dir / f"{algorithm_id}.zxg.json"
        graph_json: dict[str, Any] = json.loads(g.to_json())
        save_graph_json(graph_json, graph_path)

        entries.append(
            {
                "algorithm_id": algorithm_id,
                "source_algorithm": item.get("name", ""),
                "family": item.get("family", ""),
                "n_qubits": item.get("n_qubits", 0),
                "graph_path": str(graph_path),
                **info,
            }
        )

        logger.info(
            "  %s: %d -> %d vertices (%s)",
            algorithm_id,
            info["pre_stats"]["n_vertices"],
            info["post_stats"]["n_vertices"],
            config.reduction,
        )

    save_manifest(entries, output_dir)
    logger.info("Stage 2 complete -- %d graphs written to %s", len(entries), output_dir)

    return entries
