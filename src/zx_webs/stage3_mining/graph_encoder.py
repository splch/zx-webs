"""Encode PyZX graph attributes as integer labels for gSpan mining.

The gSpan algorithm operates on integer-labelled graphs.  This module maps
the rich PyZX vertex/edge attributes (spider type, phase, edge type) into a
flat integer label space and provides the inverse mapping.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pyzx as zx


# PyZX vertex type constants (from pyzx.utils.VertexType).
_VT_BOUNDARY = 0

# PyZX edge type constants (from pyzx.utils.EdgeType).
_ET_SIMPLE = 1
_ET_HADAMARD = 2


class ZXLabelEncoder:
    """Encode PyZX vertex/edge attributes as integer labels for gSpan.

    Vertex label
        ``encode_vertex(vtype, phase) -> int``

        For boundary vertices (type 0) the label is always 0 regardless of
        phase.  For non-boundary vertices the label combines the spider type
        and a discretised phase bucket::

            label = N_VERTEX_TYPES + (vtype - 1) * phase_bins + phase_bin

        When ``include_phase`` is *False* the label is simply the vertex type.

    Edge label
        ``encode_edge(etype) -> int``

        simple -> 0, hadamard -> 1.

    Phase discretisation
        The continuous phase (a ``Fraction`` representing multiples of pi) is
        normalised to [0, 2) and binned into ``phase_bins`` equal buckets.

    Parameters
    ----------
    phase_bins:
        Number of phase buckets.  Defaults to 8 (bins of pi/4).
    include_phase:
        Whether to incorporate phase into the vertex label.
    """

    N_VERTEX_TYPES = 4  # 0=boundary, 1=Z, 2=X, 3=H_BOX

    def __init__(self, phase_bins: int = 8, include_phase: bool = True) -> None:
        self.phase_bins = phase_bins
        self.include_phase = include_phase

    # ------------------------------------------------------------------
    # Vertex encoding / decoding
    # ------------------------------------------------------------------

    def encode_vertex(self, vtype: int, phase: Fraction | float | int = 0) -> int:
        """Map ``(vertex_type, phase)`` to a single integer label.

        Boundary vertices always receive label 0.  When ``include_phase`` is
        *False* the label equals the raw vertex type.
        """
        if vtype == _VT_BOUNDARY or not self.include_phase:
            return int(vtype)
        phase_bin = self._discretize_phase(phase)
        return self.N_VERTEX_TYPES + (vtype - 1) * self.phase_bins + phase_bin

    def decode_vertex(self, label: int) -> tuple[int, int | None]:
        """Decode an integer label to ``(vertex_type, phase_bin | None)``.

        Returns ``phase_bin = None`` for boundary vertices or when phase
        encoding was disabled.
        """
        if label < self.N_VERTEX_TYPES:
            return label, None
        adjusted = label - self.N_VERTEX_TYPES
        vtype = adjusted // self.phase_bins + 1  # +1: boundary is excluded
        phase_bin = adjusted % self.phase_bins
        return vtype, phase_bin

    # ------------------------------------------------------------------
    # Edge encoding / decoding
    # ------------------------------------------------------------------

    def encode_edge(self, etype: int) -> int:
        """Map PyZX edge type to label.  simple (1) -> 0, hadamard (2) -> 1."""
        return 0 if etype == _ET_SIMPLE else 1

    def decode_edge(self, label: int) -> int:
        """Decode edge label back to a PyZX edge type constant."""
        return _ET_SIMPLE if label == 0 else _ET_HADAMARD

    # ------------------------------------------------------------------
    # Phase discretisation
    # ------------------------------------------------------------------

    def _discretize_phase(self, phase: Fraction | float | int) -> int:
        """Bin a phase to ``[0, phase_bins)``.

        Phase in PyZX is stored as a ``Fraction`` representing multiples of
        pi.  ``phase = 1/2`` means pi/2, ``phase = 1`` means pi, etc.
        We normalise to [0, 2) then bin.
        """
        val = float(phase) % 2  # normalise to [0, 2)
        bin_idx = int(val * self.phase_bins / 2) % self.phase_bins
        return bin_idx


# ---------------------------------------------------------------------------
# PyZX -> gSpan text format conversion
# ---------------------------------------------------------------------------


def pyzx_graph_to_gspan_lines(
    graph: zx.Graph,
    graph_id: int,
    encoder: ZXLabelEncoder,
) -> list[str]:
    """Convert a single PyZX graph to gSpan text-format lines.

    The returned list contains ``t # <id>``, ``v ...``, and ``e ...`` lines
    ready to be joined and written to a file.

    The PyZX vertex ids are re-indexed to a dense ``0..n-1`` range because
    the gSpan file format uses positional vertex ids.

    Parameters
    ----------
    graph:
        A PyZX ``Graph`` instance.
    graph_id:
        Integer identifier for this graph in the gSpan database.
    encoder:
        Label encoder to map vertex/edge attributes to integers.

    Returns
    -------
    list[str]
        Lines in gSpan text format (without trailing newlines).
    """
    # Build a dense vertex mapping: pyzx_vid -> gspan_vid (0-based).
    pyzx_vids = sorted(graph.vertices())
    vid_map: dict[int, int] = {pv: i for i, pv in enumerate(pyzx_vids)}

    lines: list[str] = [f"t # {graph_id}"]

    # Vertices
    for pv in pyzx_vids:
        gv = vid_map[pv]
        vlabel = encoder.encode_vertex(graph.type(pv), graph.phase(pv))
        lines.append(f"v {gv} {vlabel}")

    # Edges -- emit each undirected edge once.
    seen_edges: set[tuple[int, int]] = set()
    for e in graph.edges():
        s, t = graph.edge_st(e)
        gs, gt = vid_map[s], vid_map[t]
        edge_key = (min(gs, gt), max(gs, gt))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        elabel = encoder.encode_edge(graph.edge_type(e))
        lines.append(f"e {edge_key[0]} {edge_key[1]} {elabel}")

    return lines


def pyzx_graphs_to_gspan_file(
    graphs: list[zx.Graph],
    path: Path,
    encoder: ZXLabelEncoder,
) -> dict[int, dict[int, int]]:
    """Write multiple PyZX graphs to a gSpan input file.

    Parameters
    ----------
    graphs:
        List of PyZX graphs.
    path:
        Destination file path.
    encoder:
        Label encoder for vertex/edge attributes.

    Returns
    -------
    dict[int, dict[int, int]]
        Mapping ``{graph_id: {pyzx_vertex_id: gspan_vertex_id}}``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    vid_maps: dict[int, dict[int, int]] = {}

    for gid, graph in enumerate(graphs):
        lines = pyzx_graph_to_gspan_lines(graph, gid, encoder)
        all_lines.extend(lines)

        # Reconstruct the vid map (same logic as pyzx_graph_to_gspan_lines).
        pyzx_vids = sorted(graph.vertices())
        vid_maps[gid] = {pv: i for i, pv in enumerate(pyzx_vids)}

    # Terminator line.
    all_lines.append("t # -1")

    path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    return vid_maps
