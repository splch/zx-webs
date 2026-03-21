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

import copy
import json
import logging
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyzx as zx

from zx_webs.config import MiningConfig
from zx_webs.persistence import load_manifest, save_manifest, save_webs_bulk
from zx_webs.stage2_zx.simplifier import simplify_graph
from zx_webs.stage3_mining.gspan_adapter import GSpanAdapter, GSpanResult
from zx_webs.stage3_mining.graph_encoder import ZXLabelEncoder
from zx_webs.stage3_mining.zx_web import BoundaryWire, ZXWeb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boundary wire recovery
# ---------------------------------------------------------------------------

# PyZX vertex type constants.
_VT_BOUNDARY = 0


def _ensure_proper_boundaries(pyzx_graph: zx.Graph) -> zx.Graph:
    """Ensure a sub-diagram has proper PyZX boundary vertices.

    When gSpan mines a sub-graph that includes boundary (type 0) vertices
    from the original circuit, those vertices naturally represent the
    circuit's qubit wires.  We use the ``row`` attribute to distinguish
    inputs (lower row) from outputs (higher row), which preserves the
    multi-qubit structure of the source circuit.

    When no boundary vertices are present (the mined pattern is purely
    interior), we attach new boundary vertices at leaf positions and use
    ``row`` to assign input/output direction.

    Parameters
    ----------
    pyzx_graph:
        The sub-diagram as a PyZX ``Graph``.

    Returns
    -------
    zx.Graph
        The graph with proper boundary vertices set.
    """
    # If the graph already has inputs and outputs, return as-is.
    if pyzx_graph.inputs() and pyzx_graph.outputs():
        return pyzx_graph

    g = pyzx_graph

    # Collect existing boundary vertices (type 0).
    existing_boundaries = [
        v for v in g.vertices() if g.type(v) == _VT_BOUNDARY
    ]

    # If there are existing boundary vertices but no inputs/outputs set,
    # use the row attribute to determine direction: lower row = input,
    # higher row = output.  This preserves the original circuit's
    # multi-qubit structure.
    if existing_boundaries:
        if len(existing_boundaries) == 1:
            # Single boundary: create a second one connected to the same
            # neighbour so we have at least one input and one output.
            bv = existing_boundaries[0]
            neighbors = list(g.neighbors(bv))
            if neighbors:
                nb = neighbors[0]
                bv2 = g.add_vertex(
                    ty=_VT_BOUNDARY,
                    qubit=g.qubit(bv),
                    row=g.row(bv) + 1.0,
                )
                g.add_edge((bv2, nb), edgetype=1)
                g.set_inputs(tuple([bv]))
                g.set_outputs(tuple([bv2]))
            else:
                g.set_inputs(tuple([bv]))
                g.set_outputs(tuple([bv]))
            return g

        # Multiple boundary vertices: partition by row.
        # Sort by row to get a clean input/output split.
        sorted_boundaries = sorted(existing_boundaries, key=lambda v: g.row(v))

        # Split evenly: first half = inputs, second half = outputs.
        # For valid quantum circuits, we need n_inputs == n_outputs.
        n_total = len(sorted_boundaries)
        half = n_total // 2

        if n_total % 2 == 0:
            # Even number: clean split.
            inputs = sorted_boundaries[:half]
            outputs = sorted_boundaries[half:]
        else:
            # Odd number: take floor(n/2) for each side and add a
            # matching boundary vertex to balance.
            inputs = sorted_boundaries[:half]
            outputs = sorted_boundaries[half:]
            # outputs has one more than inputs; add a boundary vertex
            # to balance on the input side by connecting to the same
            # interior neighbour as the last output vertex.
            extra_out = outputs[-1]
            neighbors = list(g.neighbors(extra_out))
            interior_nbs = [nb for nb in neighbors if g.type(nb) != _VT_BOUNDARY]
            if interior_nbs:
                nb = interior_nbs[0]
                bv_new = g.add_vertex(
                    ty=_VT_BOUNDARY,
                    qubit=g.qubit(extra_out),
                    row=0,
                )
                g.add_edge((bv_new, nb), edgetype=1)
                inputs.append(bv_new)
            else:
                # Fallback: just move the extra to inputs.
                inputs.append(outputs.pop())

        g.set_inputs(tuple(inputs))
        g.set_outputs(tuple(outputs))
        return g

    # No boundary vertices exist.  Identify leaf vertices (degree 1)
    # that are interior spiders -- these were cut from the parent graph.
    leaf_vertices = []
    for v in sorted(g.vertices()):
        if g.type(v) == _VT_BOUNDARY:
            continue
        if g.vertex_degree(v) <= 1:
            leaf_vertices.append(v)

    if not leaf_vertices:
        # No leaves found -- pick the two vertices with lowest/highest row
        # as boundary attachment points.
        all_verts = sorted(g.vertices(), key=lambda v: (g.row(v), v))
        if len(all_verts) >= 2:
            leaf_vertices = [all_verts[0], all_verts[-1]]
        elif len(all_verts) == 1:
            leaf_vertices = [all_verts[0]]
        else:
            return g  # empty graph, nothing to do

    # Add boundary vertices for each leaf, using row to determine
    # input vs output direction.
    rows = [g.row(lv) for lv in leaf_vertices]
    median_row = sorted(rows)[len(rows) // 2]

    input_boundaries = []
    output_boundaries = []
    for lv in leaf_vertices:
        is_input = g.row(lv) <= median_row and len(input_boundaries) <= len(output_boundaries)
        if is_input:
            row_offset = -0.5
        else:
            row_offset = 0.5

        bv = g.add_vertex(
            ty=_VT_BOUNDARY,
            qubit=g.qubit(lv),
            row=g.row(lv) + row_offset,
        )
        g.add_edge((bv, lv), edgetype=1)  # simple edge

        if is_input:
            input_boundaries.append(bv)
        else:
            output_boundaries.append(bv)

    # Ensure both sides are non-empty.
    if not input_boundaries and output_boundaries:
        input_boundaries = [output_boundaries.pop(0)]
    elif not output_boundaries and input_boundaries:
        output_boundaries = [input_boundaries.pop()]

    # If we ended up with only one boundary total, add a second one.
    all_bv = input_boundaries + output_boundaries
    if len(all_bv) == 1:
        lv = leaf_vertices[0]
        bv2 = g.add_vertex(
            ty=_VT_BOUNDARY,
            qubit=g.qubit(lv),
            row=g.row(lv) + 0.5,
        )
        g.add_edge((bv2, lv), edgetype=1)
        if not output_boundaries:
            output_boundaries = [bv2]
        else:
            input_boundaries = [bv2]

    g.set_inputs(tuple(input_boundaries))
    g.set_outputs(tuple(output_boundaries))

    return g


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
        elif len(neighbors) == 1 and v not in input_set and v not in output_set:
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


def _build_graph_json(adapter: GSpanAdapter, result: GSpanResult) -> str:
    """Build the full graph_json for a mining result (expensive).

    This is the deferred builder called lazily when ``graph_json`` is first
    needed on a ZXWeb.  It constructs the full PyZX graph, ensures proper
    boundary vertices, and serializes to JSON.
    """
    pyzx_graph = adapter.result_to_pyzx(result)
    pyzx_graph = _ensure_proper_boundaries(pyzx_graph)
    return pyzx_graph.to_json()


def _result_to_zx_web(
    result: GSpanResult,
    web_id: str,
    adapter: GSpanAdapter,
    family_lookup: dict[int, str] | None = None,
) -> ZXWeb:
    """Convert a :class:`GSpanResult` to a :class:`ZXWeb`.

    Uses **lazy evaluation**: only lightweight metadata (n_spiders,
    n_inputs, n_outputs, boundary_wires) is computed eagerly from the
    raw mining result.  The expensive PyZX graph construction and JSON
    serialization are deferred until ``to_pyzx_graph()`` or ``to_dict()``
    is called.

    This is critical for performance: with 1.5M mining results, eager
    construction would take 5+ minutes building PyZX graphs that are
    mostly discarded (Stage 4 FPS selects only ~50K webs).

    Parameters
    ----------
    result:
        gSpan mining result.
    web_id:
        Unique identifier for the web.
    adapter:
        The adapter used for mining (carries the label encoder).
    family_lookup:
        Optional mapping from source graph index to algorithm family name.
        When provided, the web's ``source_families`` field is populated.

    Returns
    -------
    ZXWeb
    """
    # Extract lightweight metadata without constructing a PyZX graph.
    meta = adapter.extract_metadata(result)

    n_spiders = meta["n_spiders"]
    n_inputs = meta["n_inputs"]
    n_outputs = meta["n_outputs"]

    # Convert boundary wire dicts to BoundaryWire objects.
    boundary_wires = [
        BoundaryWire(
            internal_vertex=bw["internal_vertex"],
            spider_type=bw["spider_type"],
            spider_phase=bw["spider_phase"],
            edge_type=bw["edge_type"],
            direction=bw["direction"],
        )
        for bw in meta["boundary_wires"]
    ]

    # Derive source families from the family lookup.
    source_families: list[str] = []
    if family_lookup:
        seen: set[str] = set()
        for gid in result.source_graph_ids:
            fam = family_lookup.get(gid, "")
            if fam and fam not in seen:
                source_families.append(fam)
                seen.add(fam)

    # Create a deferred builder that captures the adapter and result.
    # The lambda captures by reference -- result and adapter must remain
    # alive as long as any un-materialized ZXWeb exists.
    graph_builder = lambda: _build_graph_json(adapter, result)

    return ZXWeb(
        web_id=web_id,
        graph_json="",  # deferred -- will be built lazily
        boundary_wires=boundary_wires,
        support=result.support,
        source_graph_ids=result.source_graph_ids,
        source_families=source_families,
        n_spiders=n_spiders,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        _graph_builder=graph_builder,
    )


# ---------------------------------------------------------------------------
# Reversed boundary generation
# ---------------------------------------------------------------------------


def _build_reversed_graph_json(original_web: ZXWeb) -> str:
    """Build graph_json for a reversed web (deferred builder).

    Materializes the original web's graph, swaps inputs/outputs, and
    serializes to JSON.
    """
    g = original_web.to_pyzx_graph()
    if not g.inputs() or not g.outputs():
        return original_web.get_graph_json()  # fallback: return unchanged
    original_inputs = tuple(g.inputs())
    original_outputs = tuple(g.outputs())
    g.set_inputs(original_outputs)
    g.set_outputs(original_inputs)
    return g.to_json()


def _make_reversed_web(web: ZXWeb, new_web_id: str) -> ZXWeb | None:
    """Create a reversed version of a web (inputs and outputs swapped).

    Uses **lazy evaluation**: the reversed graph_json is not built until
    actually needed.  Only metadata (boundary wires, n_inputs/n_outputs)
    is computed eagerly by swapping directions.

    This ensures no valid composition is missed due to arbitrary boundary
    orientation.

    Parameters
    ----------
    web:
        The original web.
    new_web_id:
        Identifier for the reversed web.

    Returns
    -------
    ZXWeb or None
        The reversed web, or None if the web cannot be reversed
        (e.g. no inputs or outputs).
    """
    if web.n_inputs == 0 or web.n_outputs == 0:
        return None

    # Reverse the boundary wires (metadata only -- no graph construction).
    reversed_wires: list[BoundaryWire] = []
    for bw in web.boundary_wires:
        new_direction = bw.direction
        if bw.direction == "input":
            new_direction = "output"
        elif bw.direction == "output":
            new_direction = "input"
        reversed_wires.append(
            BoundaryWire(
                internal_vertex=bw.internal_vertex,
                spider_type=bw.spider_type,
                spider_phase=bw.spider_phase,
                edge_type=bw.edge_type,
                direction=new_direction,
            )
        )

    # Deferred builder that will materialize the reversed graph on demand.
    graph_builder = lambda: _build_reversed_graph_json(web)

    return ZXWeb(
        web_id=new_web_id,
        graph_json="",  # deferred
        boundary_wires=reversed_wires,
        support=web.support,
        source_graph_ids=web.source_graph_ids,
        source_families=web.source_families,
        n_spiders=web.n_spiders,
        n_inputs=web.n_outputs,  # swapped
        n_outputs=web.n_inputs,  # swapped
        _graph_builder=graph_builder,
    )


# ---------------------------------------------------------------------------
# Stage 3 entry point
# ---------------------------------------------------------------------------


def run_stage3(
    zx_dir: Path,
    output_dir: Path,
    config: MiningConfig | None = None,
    corpus_dir: Path | None = None,
    skip_bulk_write: bool = False,
) -> list[ZXWeb]:
    """Run Stage 3: mine frequent sub-diagrams from ZX-diagram corpus.

    Workflow
    --------
    1. Load ZX-diagrams from the Stage 2 output directory.
    2. Optionally re-reduce graphs with a structure-preserving method
       (e.g. ``teleport_reduce``) before mining.
    3. Run gSpan mining to discover frequent sub-graphs.
    4. Convert results to :class:`ZXWeb` objects with boundary info.
    5. Persist the webs and a manifest under *output_dir*.

    Parameters
    ----------
    zx_dir:
        Directory containing Stage 2 outputs (``manifest.json`` and
        ``graphs/*.zxg.json``).
    output_dir:
        Where Stage 3 artefacts will be written.
    skip_bulk_write:
        When ``True``, skip writing the bulk JSON file (``webs_bulk.json``).
        Set this when webs will be passed in-memory to Stage 4, avoiding
        the expensive materialization of graph_json for all webs.  The
        manifest (lightweight metadata) is always written.
    config:
        Mining parameters.  Falls back to ``MiningConfig()`` defaults when
        *None*.
    corpus_dir:
        Optional path to the Stage 1 corpus directory.  When provided, the
        corpus manifest is used to map graph indices to algorithm families
        so that each :class:`ZXWeb` records which families it came from.

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
    families: list[str] = []

    for entry in manifest:
        graph_path = Path(entry["graph_path"])
        if not graph_path.exists():
            logger.warning("Graph file not found: %s -- skipping.", graph_path)
            continue

        graph_json_str = graph_path.read_text(encoding="utf-8")
        g = zx.Graph.from_json(graph_json_str)
        graphs.append(g)
        algorithm_ids.append(entry.get("algorithm_id", ""))
        families.append(entry.get("family", ""))

    if not graphs:
        logger.warning("No valid graphs loaded from %s.", zx_dir)
        return []

    # -- 1b. Optionally re-reduce for structure-preserving mining -------------
    stage2_reduction = manifest[0].get("reduction_method", "full_reduce") if manifest else "full_reduce"
    mining_reduction = config.mining_reduction

    if mining_reduction != "full_reduce" and mining_reduction != stage2_reduction:
        # Re-reduce from the original QASM files if we have corpus_dir,
        # otherwise apply the mining reduction on top of what we have.
        if corpus_dir is not None:
            corpus_manifest = load_manifest(corpus_dir)
            if corpus_manifest:
                qasm_lookup: dict[str, str] = {
                    e.get("algorithm_id", ""): e.get("qasm_path", "")
                    for e in corpus_manifest
                }
                re_reduced: list[zx.Graph] = []
                re_ids: list[str] = []
                re_families: list[str] = []
                for aid, fam in zip(algorithm_ids, families):
                    qasm_path_str = qasm_lookup.get(aid, "")
                    if not qasm_path_str:
                        continue
                    qasm_path = Path(qasm_path_str)
                    if not qasm_path.exists():
                        continue
                    try:
                        qasm_str = qasm_path.read_text(encoding="utf-8")
                        circuit = zx.Circuit.from_qasm(qasm_str)
                        g_raw: zx.Graph = circuit.to_graph()
                        g_reduced = simplify_graph(g_raw, method=mining_reduction)
                        re_reduced.append(g_reduced)
                        re_ids.append(aid)
                        re_families.append(fam)
                    except Exception as exc:
                        logger.debug(
                            "Failed to re-reduce %s with %s: %s",
                            aid, mining_reduction, exc,
                        )
                if re_reduced:
                    logger.info(
                        "Re-reduced %d graphs with %s for mining (was %s).",
                        len(re_reduced), mining_reduction, stage2_reduction,
                    )
                    graphs = re_reduced
                    algorithm_ids = re_ids
                    families = re_families
        else:
            logger.info(
                "mining_reduction=%s requested but no corpus_dir provided; "
                "using Stage 2 graphs as-is.",
                mining_reduction,
            )

    # Build family lookup: graph index -> family name.
    family_lookup: dict[int, str] = {}
    for idx, fam in enumerate(families):
        if fam:
            family_lookup[idx] = fam

    # Split graphs into small (for gSpan mining) and all (for validation).
    max_v = config.max_input_vertices
    small_graphs = [g for g in graphs if g.num_vertices() <= max_v]
    small_ids = [a for g, a in zip(graphs, algorithm_ids) if g.num_vertices() <= max_v]

    # Build a mapping from small_graphs index to original index for family lookup.
    small_to_original: dict[int, int] = {}
    si = 0
    for oi, g in enumerate(graphs):
        if g.num_vertices() <= max_v:
            small_to_original[si] = oi
            si += 1

    # Remap family_lookup to use small graph indices.
    small_family_lookup: dict[int, str] = {}
    for small_idx, orig_idx in small_to_original.items():
        if orig_idx in family_lookup:
            small_family_lookup[small_idx] = family_lookup[orig_idx]

    n_skipped = len(graphs) - len(small_graphs)

    if n_skipped:
        logger.info(
            "Mining from %d graphs with <=%d vertices (skipped %d large graphs; "
            "patterns will still be validated across all %d).",
            len(small_graphs), max_v, n_skipped, len(graphs),
        )

    if not small_graphs:
        logger.warning("All graphs exceeded max_input_vertices=%d.", max_v)
        return []

    logger.info("Loaded %d ZX-diagrams for mining.", len(small_graphs))

    # -- 2. Run gSpan mining on small graphs -----------------------------------
    sizes = [g.num_vertices() for g in small_graphs]
    sizes_summary = "min=%d, max=%d, mean=%.1f" % (
        min(sizes), max(sizes), sum(sizes) / len(sizes),
    )
    logger.info(
        "Starting gSpan mining on %d graphs (sizes: %s)...",
        len(small_graphs), sizes_summary,
    )

    adapter = GSpanAdapter(config)
    t0 = time.time()
    results = adapter.mine(small_graphs)
    logger.info(
        "gSpan completed in %.1fs: found %d frequent sub-graphs.",
        time.time() - t0, len(results),
    )

    # -- 3. Convert to ZXWeb objects (lazy -- no PyZX graph construction) -----
    t_convert = time.time()
    webs: list[ZXWeb] = []
    web_counter = 0
    for idx, result in enumerate(results):
        web_id = f"web_{web_counter:04d}"
        web = _result_to_zx_web(
            result, web_id, adapter,
            family_lookup=small_family_lookup,
        )
        webs.append(web)
        web_counter += 1

        # Generate a reversed version (inputs and outputs swapped) to ensure
        # no valid composition is missed due to arbitrary boundary orientation.
        reversed_web = _make_reversed_web(web, f"web_{web_counter:04d}")
        if reversed_web is not None:
            webs.append(reversed_web)
            web_counter += 1

    logger.info(
        "Created %d ZXWeb objects (lazy) in %.1fs from %d mining results.",
        len(webs), time.time() - t_convert, len(results),
    )

    # -- 4. Persist results ---------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    t_persist = time.time()

    if skip_bulk_write:
        logger.info(
            "Skipping bulk JSON write (webs will be passed in-memory to Stage 4)."
        )
    else:
        # Write all webs to a single bulk JSON file.
        # NOTE: this triggers lazy materialization of graph_json for ALL webs,
        # which is expensive (O(minutes) for 1M+ webs).  Use skip_bulk_write=True
        # in sequential pipelines where webs are passed in-memory.
        webs_data = [web.to_dict() for web in webs]
        save_webs_bulk(webs_data, output_dir)

    # Build the manifest (lightweight metadata only -- no graph_json needed).
    # This is always written as it's cheap and useful for diagnostics.
    manifest_entries: list[dict[str, Any]] = []

    for web in webs:
        manifest_entries.append(
            {
                "web_id": web.web_id,
                "support": web.support,
                "source_graph_ids": web.source_graph_ids,
                "source_families": web.source_families,
                "n_spiders": web.n_spiders,
                "n_boundary_wires": len(web.boundary_wires),
                "n_inputs": web.n_inputs,
                "n_outputs": web.n_outputs,
            }
        )

    save_manifest(manifest_entries, output_dir)

    logger.info(
        "Stage 3 complete -- %d webs (%s bulk write) persisted in %.1fs to %s",
        len(webs),
        "with" if not skip_bulk_write else "without",
        time.time() - t_persist,
        output_dir,
    )

    return webs
