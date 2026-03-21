"""Adapter that wraps the gspan-mining library for use with PyZX graphs.

This module provides :class:`GSpanAdapter`, which:

1. Encodes a corpus of PyZX graphs into gSpan's integer-labelled text format.
2. Runs the gSpan frequent sub-graph mining algorithm.
3. Converts the results back into PyZX ``Graph`` objects wrapped as
   :class:`GSpanResult` records.

The adapter works around several quirks of the ``gspan-mining`` library:

* Labels are stored as **strings** internally (they are read via
  ``str.split`` and never cast).
* The ``_report`` method prints to stdout and uses the deprecated
  ``DataFrame.append`` API.  We override it to silently capture results.
* Source graph IDs are only available through the ``projected`` parameter
  passed to ``_report``.
"""
from __future__ import annotations

import collections
import copy
import itertools
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyzx as zx
from gspan_mining.gspan import DFScode, DFSedge, PDFS, Projected, gSpan as _GSpanAlgo
from gspan_mining.gspan import record_timestamp
from tqdm import tqdm

from zx_webs.config import MiningConfig
from zx_webs.stage3_mining.graph_encoder import (
    ZXLabelEncoder,
    pyzx_graphs_to_gspan_file,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GSpanResult:
    """A single frequent sub-graph discovered by gSpan.

    Attributes
    ----------
    dfscode:
        The canonical DFS code (a ``DFScode`` object, which is a list of
        ``DFSedge``).
    gspan_graph:
        The sub-graph as a ``gspan_mining.graph.Graph`` (integer/string
        labels).
    support:
        Number of input graphs in which this sub-graph appears.
    source_graph_ids:
        Indices of the input graphs that contain an embedding of this
        sub-graph.
    """

    dfscode: DFScode
    gspan_graph: Any  # gspan_mining.graph.Graph
    support: int = 0
    source_graph_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Silent gSpan subclass -- captures results without printing
# ---------------------------------------------------------------------------


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
        2. Write to a temporary file.
        3. Run the gSpan algorithm.
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

        Parameters
        ----------
        result:
            A single gSpan mining result.

        Returns
        -------
        zx.Graph
            A new PyZX graph with correct spider types, phases, and edge types.
        """
        gg = result.gspan_graph
        g = zx.Graph()

        # Separate boundary and non-boundary vertices so we can assign
        # meaningful row/qubit positions that enable correct input/output
        # assignment later in _ensure_proper_boundaries.
        _VT_BOUNDARY = 0
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
        # The encoder distinguishes input boundaries (label 0) from
        # output boundaries (label 4), so we can recover the correct
        # input/output assignment that was present in the source graph.
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
