"""Stage 3 orchestrator -- mine frequent sub-diagrams from ZX-diagram corpus.

Usage::

    from pathlib import Path
    from zx_webs.stage3_mining.miner import run_stage3

    webs = run_stage3(
        zx_dir=Path("data/zx_diagrams"),
        output_dir=Path("data/webs"),
    )
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyzx as zx

from zx_webs.config import MiningConfig
from zx_webs.persistence import load_manifest, save_json, save_manifest
from zx_webs.stage3_mining.gspan_adapter import GSpanAdapter, GSpanResult
from zx_webs.stage3_mining.graph_encoder import ZXLabelEncoder
from zx_webs.stage3_mining.zx_web import BoundaryWire, ZXWeb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boundary wire recovery
# ---------------------------------------------------------------------------

# PyZX vertex type constants.
_VT_BOUNDARY = 0


def _identify_boundary_wires(
    pyzx_graph: zx.Graph,
) -> list[BoundaryWire]:
    """Identify boundary wires on a sub-diagram's PyZX graph.

    A "boundary wire" exists wherever a vertex has degree 1 -- i.e. it was
    connected to an external vertex in the original larger diagram that is
    not part of this sub-pattern.

    For boundary vertices (type 0), we look at their single neighbour to
    determine the wire direction if the graph carries input/output info.

    For non-boundary vertices with fewer edges than they would have in a
    connected interior, we record a boundary wire for each "missing" edge
    slot.  In practice, with the gSpan sub-graph the simplest heuristic is
    to flag leaf vertices (degree 1) that are not boundary type.

    Parameters
    ----------
    pyzx_graph:
        The sub-diagram as a PyZX ``Graph``.

    Returns
    -------
    list[BoundaryWire]
        One record per identified boundary wire.
    """
    wires: list[BoundaryWire] = []
    input_set = set(pyzx_graph.inputs()) if pyzx_graph.inputs() else set()
    output_set = set(pyzx_graph.outputs()) if pyzx_graph.outputs() else set()

    for v in pyzx_graph.vertices():
        vtype = pyzx_graph.type(v)
        neighbors = list(pyzx_graph.neighbors(v))

        if vtype == _VT_BOUNDARY:
            # Boundary vertices represent connection points to the outside.
            direction = "unknown"
            if v in input_set:
                direction = "input"
            elif v in output_set:
                direction = "output"

            for nb in neighbors:
                nb_type = pyzx_graph.type(nb)
                nb_phase = float(pyzx_graph.phase(nb))
                # Determine edge type between v and nb.
                edge = pyzx_graph.edge(v, nb)
                etype = pyzx_graph.edge_type(edge)
                wires.append(
                    BoundaryWire(
                        internal_vertex=nb,
                        spider_type=nb_type,
                        spider_phase=nb_phase,
                        edge_type=etype,
                        direction=direction,
                    )
                )
        elif len(neighbors) == 1:
            # A leaf non-boundary vertex -- likely had more connections in the
            # original graph.  Record it as a potential boundary wire.
            nb = neighbors[0]
            edge = pyzx_graph.edge(v, nb)
            etype = pyzx_graph.edge_type(edge)
            wires.append(
                BoundaryWire(
                    internal_vertex=v,
                    spider_type=vtype,
                    spider_phase=float(pyzx_graph.phase(v)),
                    edge_type=etype,
                    direction="unknown",
                )
            )

    return wires


# ---------------------------------------------------------------------------
# Result -> ZXWeb conversion
# ---------------------------------------------------------------------------


def _result_to_zx_web(
    result: GSpanResult,
    web_id: str,
    adapter: GSpanAdapter,
) -> ZXWeb:
    """Convert a :class:`GSpanResult` to a :class:`ZXWeb`.

    Parameters
    ----------
    result:
        gSpan mining result.
    web_id:
        Unique identifier for the web.
    adapter:
        The adapter used for mining (carries the label encoder).

    Returns
    -------
    ZXWeb
    """
    pyzx_graph = adapter.result_to_pyzx(result)

    # Count spiders (non-boundary vertices).
    n_spiders = sum(
        1 for v in pyzx_graph.vertices() if pyzx_graph.type(v) != _VT_BOUNDARY
    )

    graph_json = pyzx_graph.to_json()

    # Identify boundary wires.
    boundary_wires = _identify_boundary_wires(pyzx_graph)

    n_inputs = sum(1 for bw in boundary_wires if bw.direction == "input")
    n_outputs = sum(1 for bw in boundary_wires if bw.direction == "output")

    return ZXWeb(
        web_id=web_id,
        graph_json=graph_json,
        boundary_wires=boundary_wires,
        support=result.support,
        source_graph_ids=result.source_graph_ids,
        n_spiders=n_spiders,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )


# ---------------------------------------------------------------------------
# Stage 3 entry point
# ---------------------------------------------------------------------------


def run_stage3(
    zx_dir: Path,
    output_dir: Path,
    config: MiningConfig | None = None,
) -> list[ZXWeb]:
    """Run Stage 3: mine frequent sub-diagrams from ZX-diagram corpus.

    Workflow
    --------
    1. Load ZX-diagrams from the Stage 2 output directory.
    2. Run gSpan mining to discover frequent sub-graphs.
    3. Convert results to :class:`ZXWeb` objects with boundary info.
    4. Persist the webs and a manifest under *output_dir*.

    Parameters
    ----------
    zx_dir:
        Directory containing Stage 2 outputs (``manifest.json`` and
        ``graphs/*.zxg.json``).
    output_dir:
        Where Stage 3 artefacts will be written.
    config:
        Mining parameters.  Falls back to ``MiningConfig()`` defaults when
        *None*.

    Returns
    -------
    list[ZXWeb]
        The discovered frequent sub-diagrams.
    """
    if config is None:
        config = MiningConfig()

    # -- 1. Load ZX-diagrams from Stage 2 manifest ---------------------------
    manifest = load_manifest(zx_dir)
    if not manifest:
        logger.warning("Stage 2 manifest at %s is empty -- nothing to mine.", zx_dir)
        return []

    graphs: list[zx.Graph] = []
    algorithm_ids: list[str] = []

    for entry in manifest:
        graph_path = Path(entry["graph_path"])
        if not graph_path.exists():
            logger.warning("Graph file not found: %s -- skipping.", graph_path)
            continue

        graph_json_str = graph_path.read_text(encoding="utf-8")
        g = zx.Graph.from_json(graph_json_str)
        graphs.append(g)
        algorithm_ids.append(entry.get("algorithm_id", ""))

    if not graphs:
        logger.warning("No valid graphs loaded from %s.", zx_dir)
        return []

    logger.info("Loaded %d ZX-diagrams for mining.", len(graphs))

    # -- 2. Run gSpan mining --------------------------------------------------
    adapter = GSpanAdapter(config)
    results = adapter.mine(graphs)

    logger.info("gSpan found %d frequent sub-graphs.", len(results))

    # -- 3. Convert to ZXWeb objects ------------------------------------------
    webs: list[ZXWeb] = []
    for idx, result in enumerate(results):
        web_id = f"web_{idx:04d}"
        web = _result_to_zx_web(result, web_id, adapter)
        webs.append(web)

    # -- 4. Persist results ---------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    webs_dir = output_dir / "webs"
    webs_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    for web in webs:
        web_path = webs_dir / f"{web.web_id}.json"
        save_json(web.to_dict(), web_path)

        manifest_entries.append(
            {
                "web_id": web.web_id,
                "web_path": str(web_path),
                "support": web.support,
                "source_graph_ids": web.source_graph_ids,
                "n_spiders": web.n_spiders,
                "n_boundary_wires": len(web.boundary_wires),
                "n_inputs": web.n_inputs,
                "n_outputs": web.n_outputs,
            }
        )

    save_manifest(manifest_entries, output_dir)

    logger.info(
        "Stage 3 complete -- %d webs written to %s", len(webs), output_dir
    )

    return webs
