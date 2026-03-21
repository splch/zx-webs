"""ZXWeb -- a frequently occurring sub-diagram in a corpus of ZX-diagrams.

This module defines the core data structures produced by Stage 3 mining:

* :class:`BoundaryWire` -- metadata for a wire at the sub-diagram boundary
* :class:`ZXWeb` -- the sub-diagram itself plus frequency statistics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyzx as zx


# ---------------------------------------------------------------------------
# Boundary wire
# ---------------------------------------------------------------------------


@dataclass
class BoundaryWire:
    """A wire at the boundary of a ZX-Web sub-diagram.

    When a sub-diagram is "cut" from a larger ZX-diagram, the severed edge
    leaves a dangling wire.  This record captures enough information to
    reconnect the wire during composition (Stage 4).

    Attributes
    ----------
    internal_vertex:
        Vertex ID *inside* the sub-diagram that the boundary wire connects to.
    spider_type:
        PyZX vertex type of the internal vertex (1 = Z, 2 = X, etc.).
    spider_phase:
        Phase of the internal vertex (as a float, in multiples of pi).
    edge_type:
        PyZX edge type of the cut edge (1 = simple, 2 = hadamard).
    direction:
        ``"input"`` or ``"output"`` -- indicates the role of the external
        endpoint in the original diagram's boundary.  ``"unknown"`` when the
        direction cannot be determined.
    """

    internal_vertex: int
    spider_type: int
    spider_phase: float
    edge_type: int
    direction: str = "unknown"


# ---------------------------------------------------------------------------
# ZXWeb
# ---------------------------------------------------------------------------


@dataclass
class ZXWeb:
    """A frequently occurring sub-diagram in the ZX-diagram corpus.

    Attributes
    ----------
    web_id:
        Unique identifier for this web (e.g. ``"web_000"``).
    graph_json:
        PyZX graph serialised as a JSON string.
    boundary_wires:
        Wires that were "cut" when the sub-diagram was extracted.
    support:
        Number of source graphs in which this sub-diagram appears.
    source_graph_ids:
        Integer indices of the source graphs that contain this pattern.
    n_spiders:
        Total number of non-boundary vertices in the sub-diagram.
    n_inputs:
        Number of input-boundary vertices.
    n_outputs:
        Number of output-boundary vertices.
    """

    web_id: str = ""
    graph_json: str = ""
    boundary_wires: list[BoundaryWire] = field(default_factory=list)
    support: int = 0
    source_graph_ids: list[int] = field(default_factory=list)
    source_families: list[str] = field(default_factory=list)
    n_spiders: int = 0
    n_inputs: int = 0
    n_outputs: int = 0

    # ------------------------------------------------------------------
    # PyZX round-trip
    # ------------------------------------------------------------------

    def to_pyzx_graph(self) -> zx.Graph:
        """Reconstruct the PyZX ``Graph`` from the stored JSON string."""
        return zx.Graph.from_json(self.graph_json)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON storage."""
        return {
            "web_id": self.web_id,
            "graph_json": self.graph_json,
            "boundary_wires": [
                {
                    "internal_vertex": bw.internal_vertex,
                    "spider_type": bw.spider_type,
                    "spider_phase": bw.spider_phase,
                    "edge_type": bw.edge_type,
                    "direction": bw.direction,
                }
                for bw in self.boundary_wires
            ],
            "support": self.support,
            "source_graph_ids": self.source_graph_ids,
            "source_families": self.source_families,
            "n_spiders": self.n_spiders,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ZXWeb:
        """Deserialise from a plain dict."""
        bwires = [
            BoundaryWire(
                internal_vertex=bw["internal_vertex"],
                spider_type=bw["spider_type"],
                spider_phase=bw["spider_phase"],
                edge_type=bw["edge_type"],
                direction=bw.get("direction", "unknown"),
            )
            for bw in d.get("boundary_wires", [])
        ]
        return cls(
            web_id=d.get("web_id", ""),
            graph_json=d.get("graph_json", ""),
            boundary_wires=bwires,
            support=d.get("support", 0),
            source_graph_ids=d.get("source_graph_ids", []),
            source_families=d.get("source_families", []),
            n_spiders=d.get("n_spiders", 0),
            n_inputs=d.get("n_inputs", 0),
            n_outputs=d.get("n_outputs", 0),
        )
