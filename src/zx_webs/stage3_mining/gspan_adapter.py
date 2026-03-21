"""Adapter that wraps the gspan-mining library for use with PyZX graphs.

This module provides :class:`GSpanAdapter`, which:

1. Encodes a corpus of PyZX graphs into gSpan's integer-labelled text format.
2. Runs the gSpan frequent sub-graph mining algorithm.
3. Converts the results back into PyZX ``Graph`` objects wrapped as
   :class:`GSpanResult` records.

Two backends are supported:

* **C++ gSpan binary** (preferred): A fast compiled binary at
  ``vendor/gspan-cpp/gSpan``.  Used automatically when the binary exists.
* **Python gspan-mining** (fallback): The pure-Python ``gspan-mining``
  library, used when the binary is not available.

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
import logging
import subprocess
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

logger = logging.getLogger(__name__)


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
# C++ gSpan output parsing helpers
# ---------------------------------------------------------------------------


def _build_input_filter_reverse_maps(
    gspan_file_path: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    """Replicate the C++ InputFilter label remapping and return reverse maps.

    The C++ gSpan binary remaps vertex and edge labels internally: it
    collects all unique labels, sorts them by descending frequency, and
    assigns new labels starting from 2.  The output DFSCode uses these
    remapped labels.

    This function parses the same gSpan input file and replicates that
    logic so we can reverse the mapping: ``remapped_label -> original_label``.

    Returns
    -------
    (vertex_reverse_map, edge_reverse_map)
        Dictionaries from remapped label to original label.
    """
    vertex_counts: dict[int, int] = {}
    edge_counts: dict[int, int] = {}
    vertex_order: list[int] = []  # insertion order (first occurrence)
    edge_order: list[int] = []

    with open(gspan_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("t "):
                continue
            parts = line.split()
            if parts[0] == "v":
                label = int(parts[2])
                if label not in vertex_counts:
                    vertex_order.append(label)
                    vertex_counts[label] = 0
                vertex_counts[label] += 1
            elif parts[0] == "e":
                label = int(parts[3])
                if label not in edge_counts:
                    edge_order.append(label)
                    edge_counts[label] = 0
                edge_counts[label] += 1

    # Sort by descending frequency (matching C++ InputFilter::filterV/filterE).
    # C++ uses std::sort with operator< defined as "cnt > o.cnt" (descending).
    # Python's sort is stable; C++ std::sort is not.  To match C++ behaviour
    # we only sort by -count.  For equal counts, the relative order depends on
    # the C++ std::sort implementation, which typically preserves input order
    # for introsort on most platforms.  We use the insertion order as a
    # secondary key -- this matches the most common C++ stdlib behaviour and
    # is correct as long as the counts differ (which they usually do).
    sorted_vlabels = sorted(
        vertex_order, key=lambda lb: -vertex_counts[lb]
    )
    sorted_elabels = sorted(
        edge_order, key=lambda lb: -edge_counts[lb]
    )

    # Build reverse maps: remapped (rank + 2) -> original
    vertex_rev: dict[int, int] = {}
    for rank, orig_label in enumerate(sorted_vlabels):
        vertex_rev[rank + 2] = orig_label

    edge_rev: dict[int, int] = {}
    for rank, orig_label in enumerate(sorted_elabels):
        edge_rev[rank + 2] = orig_label

    return vertex_rev, edge_rev


def _parse_cpp_dfscode_output(
    stdout: str,
) -> list[list[tuple[int, int, int, int, int]]]:
    """Parse the C++ gSpan DFSCode output from stdout.

    Each pattern starts with ``t # N`` and is followed by lines of
    ``a b la lab lb``.  A trailing ``Running Time: ...`` line is ignored.

    Returns
    -------
    list of patterns, where each pattern is a list of (a, b, la, lab, lb) tuples.
    """
    patterns: list[list[tuple[int, int, int, int, int]]] = []
    current: list[tuple[int, int, int, int, int]] | None = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Running Time:"):
            continue
        if line.startswith("t # "):
            current = []
            patterns.append(current)
            continue
        if current is None:
            continue
        parts = line.split()
        if len(parts) == 5:
            a, b, la, lab, lb = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            current.append((a, b, la, lab, lb))

    return patterns


def _parse_gspan_input_file(
    gspan_file_path: Path,
) -> list[tuple[dict[int, int], list[tuple[int, int, int]]]]:
    """Parse a gSpan input file into per-graph labeled structures.

    Returns a list of ``(vertex_labels, edges)`` tuples, one per graph,
    where ``vertex_labels`` maps vertex id to label and ``edges`` is a list
    of ``(from_id, to_id, edge_label)`` tuples.
    """
    graphs: list[tuple[dict[int, int], list[tuple[int, int, int]]]] = []
    current_vlabels: dict[int, int] = {}
    current_edges: list[tuple[int, int, int]] = []
    in_graph = False

    with open(gspan_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("t # -1"):
                if in_graph:
                    graphs.append((current_vlabels, current_edges))
                break
            if line.startswith("t # "):
                if in_graph:
                    graphs.append((current_vlabels, current_edges))
                current_vlabels = {}
                current_edges = []
                in_graph = True
                continue
            parts = line.split()
            if parts[0] == "v":
                vid = int(parts[1])
                label = int(parts[2])
                current_vlabels[vid] = label
            elif parts[0] == "e":
                u = int(parts[1])
                v = int(parts[2])
                label = int(parts[3])
                current_edges.append((u, v, label))

    return graphs


def _compute_support(
    pattern_vlabels: dict[int, int],
    pattern_edges: list[tuple[int, int, int]],
    input_graphs: list[tuple[dict[int, int], list[tuple[int, int, int]]]],
) -> tuple[int, list[int]]:
    """Count how many input graphs contain the pattern as a subgraph.

    Uses recursive backtracking for label-constrained subgraph isomorphism
    on the integer-labeled gSpan graphs.

    Returns ``(support_count, list_of_graph_ids)`` where graph_ids are
    0-based indices into ``input_graphs``.
    """
    pattern_vids = sorted(pattern_vlabels)
    # Build pattern adjacency: for each vertex, set of (neighbor, edge_label)
    pattern_adj: dict[int, set[tuple[int, int]]] = {v: set() for v in pattern_vids}
    for u, v, el in pattern_edges:
        pattern_adj[u].add((v, el))
        pattern_adj[v].add((u, el))

    source_ids: list[int] = []

    for gid, (g_vlabels, g_edges) in enumerate(input_graphs):
        # Build graph adjacency
        g_adj: dict[int, set[tuple[int, int]]] = {v: set() for v in g_vlabels}
        for u, v, el in g_edges:
            g_adj[u].add((v, el))
            g_adj[v].add((u, el))

        # Try to find a mapping from pattern vertices to graph vertices.
        if _subgraph_match(
            pattern_vids, 0, pattern_vlabels, pattern_adj,
            g_vlabels, g_adj, {}
        ):
            source_ids.append(gid)

    return len(source_ids), source_ids


def _subgraph_match(
    pattern_vids: list[int],
    idx: int,
    p_vlabels: dict[int, int],
    p_adj: dict[int, set[tuple[int, int]]],
    g_vlabels: dict[int, int],
    g_adj: dict[int, set[tuple[int, int]]],
    mapping: dict[int, int],
) -> bool:
    """Recursive backtracking subgraph isomorphism check.

    Tries to extend ``mapping`` (pattern_vid -> graph_vid) one vertex at
    a time.  Returns True as soon as a valid complete mapping is found.
    """
    if idx == len(pattern_vids):
        return True

    pv = pattern_vids[idx]
    pv_label = p_vlabels[pv]
    used = set(mapping.values())

    # Candidate graph vertices: same label, not already mapped.
    for gv, gv_label in g_vlabels.items():
        if gv_label != pv_label or gv in used:
            continue

        # Check edge constraints: for every already-mapped neighbor of pv,
        # the corresponding edge must exist in the graph.
        ok = True
        for neighbor, edge_label in p_adj[pv]:
            if neighbor in mapping:
                gn = mapping[neighbor]
                if (gn, edge_label) not in g_adj[gv]:
                    ok = False
                    break
        if not ok:
            continue

        mapping[pv] = gv
        if _subgraph_match(
            pattern_vids, idx + 1, p_vlabels, p_adj,
            g_vlabels, g_adj, mapping,
        ):
            return True
        del mapping[pv]

    return False


def _dfscode_edges_to_pyzx(
    edges: list[tuple[int, int, int, int, int]],
    encoder: ZXLabelEncoder,
) -> zx.Graph:
    """Build a PyZX graph from a list of DFSCode edges with original labels.

    Each edge is ``(a, b, la, lab, lb)`` using original (un-remapped)
    vertex/edge labels from the ZXLabelEncoder.

    Parameters
    ----------
    edges:
        List of DFSCode edges with original labels.
    encoder:
        The label encoder used for the corpus.

    Returns
    -------
    zx.Graph
        A new PyZX graph.
    """
    _VT_BOUNDARY = 0

    # Collect vertex labels from the DFSCode edges.
    # Each DFSCode node (a, b, la, lab, lb) tells us vertex a has label la
    # and vertex b has label lb.
    vertex_labels: dict[int, int] = {}
    for a, b, la, lab, lb in edges:
        if a not in vertex_labels:
            vertex_labels[a] = la
        if b not in vertex_labels:
            vertex_labels[b] = lb

    # Separate boundary vs interior
    boundary_vids: list[int] = []
    interior_vids: list[int] = []
    for vid in sorted(vertex_labels):
        vlabel = vertex_labels[vid]
        vtype, _ = encoder.decode_vertex(vlabel)
        if vtype == _VT_BOUNDARY:
            boundary_vids.append(vid)
        else:
            interior_vids.append(vid)

    g = zx.Graph()
    dfscode_to_pyzx: dict[int, int] = {}

    # Add interior vertices (row=1).
    for idx, vid in enumerate(interior_vids):
        vlabel = vertex_labels[vid]
        vtype, phase_bin = encoder.decode_vertex(vlabel)
        phase: Fraction | int = 0
        if phase_bin is not None and encoder.include_phase:
            phase = Fraction(2 * phase_bin, encoder.phase_bins)
        pv = g.add_vertex(ty=vtype, phase=phase, qubit=idx, row=1)
        dfscode_to_pyzx[vid] = pv

    # Add boundary vertices.
    input_qubit = 0
    output_qubit = 0
    input_pvs: list[int] = []
    output_pvs: list[int] = []

    for vid in boundary_vids:
        vlabel = vertex_labels[vid]
        vtype, phase_bin = encoder.decode_vertex(vlabel)
        phase_val: Fraction | int = 0
        if phase_bin is not None and encoder.include_phase:
            phase_val = Fraction(2 * phase_bin, encoder.phase_bins)

        if encoder.is_output_boundary(vlabel):
            row = 2
            qubit = output_qubit
            output_qubit += 1
        else:
            row = 0
            qubit = input_qubit
            input_qubit += 1

        pv = g.add_vertex(ty=vtype, phase=phase_val, qubit=qubit, row=row)
        dfscode_to_pyzx[vid] = pv

        if encoder.is_output_boundary(vlabel):
            output_pvs.append(pv)
        else:
            input_pvs.append(pv)

    # Add edges (each DFSCode edge is unique, no dedup needed).
    for a, b, la, lab, lb in edges:
        etype = encoder.decode_edge(lab)
        pv_from = dfscode_to_pyzx[a]
        pv_to = dfscode_to_pyzx[b]
        g.add_edge((pv_from, pv_to), edgetype=etype)

    if input_pvs:
        g.set_inputs(tuple(input_pvs))
    if output_pvs:
        g.set_outputs(tuple(output_pvs))

    return g


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
    # Binary detection
    # ------------------------------------------------------------------

    def _find_gspan_binary(self) -> Path | None:
        """Locate the C++ gSpan binary.

        Checks, in order:
        1. ``config.gspan_binary_path`` if explicitly set.
        2. ``vendor/gspan-cpp/gSpan`` relative to the project root.

        Returns ``None`` if no usable binary is found.
        """
        if self.config.gspan_binary_path:
            p = Path(self.config.gspan_binary_path)
            if p.is_file():
                return p
            logger.warning("Configured gspan_binary_path %s not found", p)
            return None

        # Auto-detect: walk up from this file to find the project root.
        # This file lives at src/zx_webs/stage3_mining/gspan_adapter.py,
        # so project root is 4 levels up.
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / "vendor" / "gspan-cpp" / "gSpan"
        if candidate.is_file():
            return candidate
        return None

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def mine(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Run gSpan mining on a corpus of ZX-diagrams.

        Attempts to use the C++ gSpan binary if available, falling back
        to the Python ``gspan-mining`` library otherwise.

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

        binary = self._find_gspan_binary()
        if binary is not None:
            try:
                return self._mine_cpp(graphs, binary)
            except Exception:
                logger.warning(
                    "C++ gSpan binary failed, falling back to Python",
                    exc_info=True,
                )

        return self._mine_python(graphs)

    def _mine_python(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Run mining using the Python gspan-mining library."""
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

        results = sorted(
            gs.captured_results,
            key=lambda r: (-r.support, r.dfscode.get_num_vertices()),
        )
        return results

    def _mine_cpp(
        self, graphs: list[zx.Graph], binary: Path
    ) -> list[GSpanResult]:
        """Run mining using the C++ gSpan binary.

        Parameters
        ----------
        graphs:
            List of PyZX graphs.
        binary:
            Path to the compiled gSpan executable.

        Returns
        -------
        list[GSpanResult]
            Frequent sub-graphs with support computed via subgraph
            isomorphism against the input corpus.
        """
        n_graphs = len(graphs)
        min_support_frac = self.config.min_support / n_graphs

        with tempfile.TemporaryDirectory() as tmpdir:
            gspan_path = Path(tmpdir) / "corpus.gspan"
            pyzx_graphs_to_gspan_file(graphs, gspan_path, self.encoder)

            # Build reverse label maps from the input file.
            vertex_rev, edge_rev = _build_input_filter_reverse_maps(gspan_path)

            # Parse input graphs for support computation.
            input_graphs = _parse_gspan_input_file(gspan_path)

            # Run the C++ binary with ulimit -s unlimited.
            max_v = self.config.max_vertices
            min_v = self.config.min_vertices
            cmd = (
                f"ulimit -s unlimited && "
                f"{binary} {gspan_path} {min_support_frac} {max_v} {min_v}"
            )
            logger.info("Running C++ gSpan: %s", cmd)
            proc = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"C++ gSpan exited with code {proc.returncode}.\n"
                    f"stderr: {proc.stderr[:2000]}"
                )

            # Parse the DFSCode output.
            raw_patterns = _parse_cpp_dfscode_output(proc.stdout)

        # Convert each pattern to a GSpanResult.
        results: list[GSpanResult] = []
        for pattern_edges_remapped in raw_patterns:
            if not pattern_edges_remapped:
                continue

            # Reverse the label remapping.
            edges_original: list[tuple[int, int, int, int, int]] = []
            for a, b, la_r, lab_r, lb_r in pattern_edges_remapped:
                la = vertex_rev.get(la_r, la_r)
                lab = edge_rev.get(lab_r, lab_r)
                lb = vertex_rev.get(lb_r, lb_r)
                edges_original.append((a, b, la, lab, lb))

            # Collect vertex labels and edges for this pattern.
            pattern_vlabels: dict[int, int] = {}
            pattern_edge_list: list[tuple[int, int, int]] = []
            for a, b, la, lab, lb in edges_original:
                if a not in pattern_vlabels:
                    pattern_vlabels[a] = la
                if b not in pattern_vlabels:
                    pattern_vlabels[b] = lb
                pattern_edge_list.append((a, b, lab))

            n_verts = len(pattern_vlabels)

            # Apply vertex count filters.
            if n_verts < self.config.min_vertices:
                continue
            if n_verts > self.config.max_vertices:
                continue

            # Compute support via subgraph isomorphism.
            support, source_ids = _compute_support(
                pattern_vlabels, pattern_edge_list, input_graphs,
            )

            # Build a PyZX graph for this pattern.
            pyzx_g = _dfscode_edges_to_pyzx(edges_original, self.encoder)

            result = GSpanResult(
                dfscode=DFScode(),
                gspan_graph=None,
                support=support,
                source_graph_ids=source_ids,
            )
            # Stash the PyZX graph on the result for use by result_to_pyzx.
            result._pyzx_graph = pyzx_g  # type: ignore[attr-defined]
            results.append(result)

        # Sort by descending support, then ascending vertex count.
        results.sort(
            key=lambda r: (-r.support, r._pyzx_graph.num_vertices()),  # type: ignore[attr-defined]
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
        # If this result came from the C++ backend, return the pre-built graph.
        if hasattr(result, "_pyzx_graph") and result._pyzx_graph is not None:  # type: ignore[attr-defined]
            return result._pyzx_graph  # type: ignore[attr-defined]

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
