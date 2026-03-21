"""Adapter that wraps gSpan mining for use with PyZX graphs.

This module provides :class:`GSpanAdapter`, which:

1. Encodes a corpus of PyZX graphs into gSpan's integer-labelled text format.
2. Runs the gSpan frequent sub-graph mining algorithm.
3. Converts the results back into PyZX ``Graph`` objects wrapped as
   :class:`GSpanResult` records.

The primary backend is **submine** (a C++ gSpan implementation).  When
submine is not installed, we fall back to the pure-Python ``gspan-mining``
library with a silent subclass that captures results without printing.
"""
from __future__ import annotations

import collections
import copy
import logging
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyzx as zx
from tqdm import tqdm

from zx_webs.config import MiningConfig
from zx_webs.stage3_mining.graph_encoder import (
    ZXLabelEncoder,
    pyzx_graphs_to_gspan_file,
)

logger = logging.getLogger(__name__)

# Try to import the C++ backend from submine.
try:
    from submine.algorithms import gspan_cpp as _gspan_cpp

    _HAS_SUBMINE = True
except ImportError:
    _gspan_cpp = None  # type: ignore[assignment]
    _HAS_SUBMINE = False

# Import the pure-Python fallback (gspan-mining).
try:
    from gspan_mining.gspan import DFScode, DFSedge, PDFS, Projected, gSpan as _GSpanAlgo
    from gspan_mining.gspan import record_timestamp

    _HAS_GSPAN_MINING = True
except ImportError:
    _HAS_GSPAN_MINING = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GSpanResult:
    """A single frequent sub-graph discovered by gSpan.

    Attributes
    ----------
    dfscode:
        The canonical DFS code (a ``DFScode`` object from gspan-mining, or
        *None* when the submine C++ backend is used).
    gspan_graph:
        The sub-graph as a ``gspan_mining.graph.Graph`` (integer/string
        labels), or *None* when the submine backend is used.
    support:
        Number of input graphs in which this sub-graph appears.
    source_graph_ids:
        Indices of the input graphs that contain an embedding of this
        sub-graph.
    submine_dict:
        When the submine backend is used, the raw result dict containing
        ``nodes``, ``node_labels``, ``edges``, ``edge_labels``, ``support``,
        and ``graph_ids``.
    """

    dfscode: Any = None  # DFScode or None
    gspan_graph: Any = None  # gspan_mining.graph.Graph or None
    support: int = 0
    source_graph_ids: list[int] = field(default_factory=list)
    submine_dict: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Silent gSpan subclass -- captures results without printing
# (only defined when the pure-Python gspan-mining library is available)
# ---------------------------------------------------------------------------

if _HAS_GSPAN_MINING:

    class _SilentGSpan(_GSpanAlgo):
        """gSpan subclass that silently captures mining results.

        The upstream ``_report`` method prints to stdout and relies on the
        deprecated ``DataFrame.append``.  We override it to store each frequent
        sub-graph with its support and source graph IDs in ``captured_results``.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.captured_results: list[GSpanResult] = []

        @record_timestamp
        def run(self) -> None:
            """Override to add tqdm progress over root edge labels."""
            self._read_graphs()
            self._generate_1edge_frequent_subgraphs()
            if self._max_num_vertices < 2:
                return
            root = collections.defaultdict(Projected)
            for g in self.graphs.values():
                for vid, v in g.vertices.items():
                    edges = self._get_forward_root_edges(g, vid)
                    for e in edges:
                        root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                            PDFS(g.gid, e, None)
                        )

            for vevlb, projected in tqdm(
                root.items(), desc="Stage 3: gSpan mining", unit="edge-label"
            ):
                self._DFScode.append(DFSedge(0, 1, vevlb))
                self._subgraph_mining(projected)
                self._DFScode.pop()

        def _report(self, projected: Any) -> None:  # type: ignore[override]
            """Override to silently record the current DFScode + metadata."""
            dfscode = copy.copy(self._DFScode)
            self._frequent_subgraphs.append(dfscode)

            if dfscode.get_num_vertices() < self._min_num_vertices:
                return

            graph = dfscode.to_graph(
                gid=next(self._counter),
                is_undirected=self._is_undirected,
            )

            source_ids = sorted(set(p.gid for p in projected))

            self.captured_results.append(
                GSpanResult(
                    dfscode=dfscode,
                    gspan_graph=graph,
                    support=self._support,
                    source_graph_ids=source_ids,
                )
            )


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class GSpanAdapter:
    """Wrap the gspan-mining library for use with PyZX graphs.

    Parameters
    ----------
    config:
        Mining parameters (min support, vertex bounds, phase discretisation).
    """

    def __init__(self, config: MiningConfig) -> None:
        self.config = config
        self.encoder = ZXLabelEncoder(
            phase_bins=config.phase_discretization,
            include_phase=config.include_phase_in_label,
        )

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def mine(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Run gSpan mining on a corpus of ZX-diagrams.

        Steps
        -----
        1. Encode all PyZX graphs into the gSpan text format.
        2. Write to a temporary file / string.
        3. Run the gSpan algorithm (submine C++ backend preferred,
           pure-Python ``gspan-mining`` as fallback).
        4. Return :class:`GSpanResult` objects for every frequent sub-graph.

        Parameters
        ----------
        graphs:
            List of simplified PyZX ``Graph`` instances.

        Returns
        -------
        list[GSpanResult]
            Frequent sub-graphs sorted by descending support.
        """
        if not graphs:
            return []

        if _HAS_SUBMINE:
            return self._mine_submine(graphs)
        if _HAS_GSPAN_MINING:
            return self._mine_python(graphs)
        raise RuntimeError(
            "No gSpan backend available. Install submine (recommended) or gspan-mining."
        )

    def _mine_submine(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Mine using the submine C++ backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gspan_path = Path(tmpdir) / "corpus.gspan"
            pyzx_graphs_to_gspan_file(graphs, gspan_path, self.encoder)
            gspan_data = gspan_path.read_text(encoding="utf-8")

        logger.debug("Running submine C++ gSpan backend.")
        raw_results = _gspan_cpp.mine_from_string(
            gspan_data,
            minsup=self.config.min_support,
            maxpat_min=self.config.min_vertices,
            maxpat_max=self.config.max_vertices,
            directed=False,
            where=True,
        )

        results: list[GSpanResult] = []
        for rd in raw_results:
            graph_ids = sorted(set(rd["graph_ids"]))
            results.append(
                GSpanResult(
                    support=rd["support"],
                    source_graph_ids=graph_ids,
                    submine_dict=rd,
                )
            )

        # Sort by descending support, then ascending number of nodes.
        results.sort(key=lambda r: (-r.support, len(r.submine_dict["nodes"])))
        return results

    def _mine_python(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Mine using the pure-Python gspan-mining fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gspan_path = Path(tmpdir) / "corpus.gspan"
            pyzx_graphs_to_gspan_file(graphs, gspan_path, self.encoder)

            gs = _SilentGSpan(
                database_file_name=str(gspan_path),
                min_support=self.config.min_support,
                min_num_vertices=self.config.min_vertices,
                max_num_vertices=self.config.max_vertices,
                is_undirected=True,
                verbose=False,
                visualize=False,
                where=False,
            )
            gs.run()

        # Sort by descending support, then ascending number of vertices.
        results = sorted(
            gs.captured_results,
            key=lambda r: (-r.support, r.dfscode.get_num_vertices()),
        )
        return results

    # ------------------------------------------------------------------
    # Result -> PyZX conversion
    # ------------------------------------------------------------------

    def result_to_pyzx(self, result: GSpanResult) -> zx.Graph:
        """Convert a :class:`GSpanResult` to a PyZX ``Graph``.

        Vertex and edge labels are decoded using the same
        :class:`ZXLabelEncoder` that was used for encoding.

        Supports results from both the submine C++ backend (via
        ``submine_dict``) and the pure-Python gspan-mining library (via
        ``gspan_graph``).

        Parameters
        ----------
        result:
            A single gSpan mining result.

        Returns
        -------
        zx.Graph
            A new PyZX graph with correct spider types, phases, and edge types.
        """
        if result.submine_dict is not None:
            return self._submine_dict_to_pyzx(result.submine_dict)
        return self._gspan_graph_to_pyzx(result.gspan_graph)

    def _submine_dict_to_pyzx(self, rd: dict[str, Any]) -> zx.Graph:
        """Build a PyZX graph from a submine result dict.

        The dict contains:
        - ``nodes``: list of node ids (0-based)
        - ``node_labels``: list of integer labels (parallel to ``nodes``)
        - ``edges``: list of (src, dst) tuples (each undirected edge appears twice)
        - ``edge_labels``: list of integer labels (parallel to ``edges``)
        """
        _VT_BOUNDARY = 0
        g = zx.Graph()

        nodes = rd["nodes"]
        node_labels = rd["node_labels"]

        # Separate boundary and interior vertices.
        boundary_vids: list[int] = []
        interior_vids: list[int] = []
        vlabel_map: dict[int, int] = {}

        for vid, vlabel in zip(nodes, node_labels):
            vlabel_map[vid] = vlabel
            vtype, _ = self.encoder.decode_vertex(vlabel)
            if vtype == _VT_BOUNDARY:
                boundary_vids.append(vid)
            else:
                interior_vids.append(vid)

        # Map submine node id -> PyZX vertex id.
        sm_to_pyzx: dict[int, int] = {}

        # Add interior vertices first (row=1, sequential qubit assignment).
        for idx, vid in enumerate(interior_vids):
            vlabel = vlabel_map[vid]
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase = Fraction(2 * phase_bin, self.encoder.phase_bins)
            pv = g.add_vertex(ty=vtype, phase=phase, qubit=idx, row=1)
            sm_to_pyzx[vid] = pv

        # Add boundary vertices using the label to determine direction.
        input_qubit = 0
        output_qubit = 0
        input_pvs: list[int] = []
        output_pvs: list[int] = []

        for vid in boundary_vids:
            vlabel = vlabel_map[vid]
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase_val: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase_val = Fraction(2 * phase_bin, self.encoder.phase_bins)

            if self.encoder.is_output_boundary(vlabel):
                row = 2
                qubit = output_qubit
                output_qubit += 1
            else:
                row = 0
                qubit = input_qubit
                input_qubit += 1

            pv = g.add_vertex(ty=vtype, phase=phase_val, qubit=qubit, row=row)
            sm_to_pyzx[vid] = pv

            if self.encoder.is_output_boundary(vlabel):
                output_pvs.append(pv)
            else:
                input_pvs.append(pv)

        # Collect edges (submine returns each undirected edge twice, so dedupe).
        edges = rd["edges"]
        edge_labels = rd["edge_labels"]
        seen: set[tuple[int, int]] = set()
        for (src, dst), elabel in zip(edges, edge_labels):
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            seen.add(key)
            etype = self.encoder.decode_edge(elabel)
            pv_from = sm_to_pyzx[src]
            pv_to = sm_to_pyzx[dst]
            g.add_edge((pv_from, pv_to), edgetype=etype)

        # Set inputs/outputs if we identified any from the labels.
        if input_pvs:
            g.set_inputs(tuple(input_pvs))
        if output_pvs:
            g.set_outputs(tuple(output_pvs))

        return g

    def _gspan_graph_to_pyzx(self, gg: Any) -> zx.Graph:
        """Build a PyZX graph from a gspan-mining ``Graph`` object."""
        _VT_BOUNDARY = 0
        g = zx.Graph()

        # Separate boundary and non-boundary vertices so we can assign
        # meaningful row/qubit positions that enable correct input/output
        # assignment later in _ensure_proper_boundaries.
        boundary_vids: list[Any] = []
        interior_vids: list[Any] = []

        for vid, vertex in gg.vertices.items():
            vlabel = int(vertex.vlb) if isinstance(vertex.vlb, str) else vertex.vlb
            vtype, _ = self.encoder.decode_vertex(vlabel)
            if vtype == _VT_BOUNDARY:
                boundary_vids.append(vid)
            else:
                interior_vids.append(vid)

        # Map gspan vertex ids -> pyzx vertex ids.
        gspan_to_pyzx: dict[Any, int] = {}

        # Add interior vertices first (row=1, sequential qubit assignment).
        for idx, vid in enumerate(interior_vids):
            vertex = gg.vertices[vid]
            vlabel = int(vertex.vlb) if isinstance(vertex.vlb, str) else vertex.vlb
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase = Fraction(2 * phase_bin, self.encoder.phase_bins)
            pv = g.add_vertex(ty=vtype, phase=phase, qubit=idx, row=1)
            gspan_to_pyzx[vid] = pv

        # Add boundary vertices using the label to determine direction.
        input_qubit = 0
        output_qubit = 0
        input_pvs: list[int] = []
        output_pvs: list[int] = []

        for vid in boundary_vids:
            vertex = gg.vertices[vid]
            vlabel = int(vertex.vlb) if isinstance(vertex.vlb, str) else vertex.vlb
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase_val: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase_val = Fraction(2 * phase_bin, self.encoder.phase_bins)

            if self.encoder.is_output_boundary(vlabel):
                row = 2
                qubit = output_qubit
                output_qubit += 1
            else:
                row = 0
                qubit = input_qubit
                input_qubit += 1

            pv = g.add_vertex(ty=vtype, phase=phase_val, qubit=qubit, row=row)
            gspan_to_pyzx[vid] = pv

            if self.encoder.is_output_boundary(vlabel):
                output_pvs.append(pv)
            else:
                input_pvs.append(pv)

        # Collect edges (avoid duplicates in undirected graph).
        seen: set[tuple[int, int]] = set()
        for vid, vertex in gg.vertices.items():
            for to_vid, edge in vertex.edges.items():
                key = (min(vid, to_vid), max(vid, to_vid))
                if key in seen:
                    continue
                seen.add(key)

                elabel = int(edge.elb) if isinstance(edge.elb, str) else edge.elb
                etype = self.encoder.decode_edge(elabel)
                pv_from = gspan_to_pyzx[vid]
                pv_to = gspan_to_pyzx[to_vid]
                g.add_edge((pv_from, pv_to), edgetype=etype)

        # Set inputs/outputs if we identified any from the labels.
        if input_pvs:
            g.set_inputs(tuple(input_pvs))
        if output_pvs:
            g.set_outputs(tuple(output_pvs))

        return g
