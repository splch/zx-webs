"""CandidateAlgorithm -- a candidate quantum algorithm composed from ZX-Webs.

A candidate is produced by Stage 4 (combinatorial stitching) and carries:

* The composed ZX-diagram serialised as JSON.
* Provenance: which webs were used and how they were composed.
* Structural metadata (qubit count, spider count).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateAlgorithm:
    """A candidate algorithm composed from one or more ZX-Webs.

    Attributes
    ----------
    candidate_id:
        Unique identifier (e.g. ``"cand_0042"``).
    graph_json:
        The composed PyZX graph serialised via ``Graph.to_json()``.
    component_web_ids:
        Ordered list of :class:`ZXWeb` identifiers that were stitched together.
    composition_type:
        How the webs were combined: ``"sequential"``, ``"parallel"``, or
        ``"hybrid"`` (a mix of both).
    n_qubits:
        Number of qubits (input boundary wires) in the composed diagram.
    n_spiders:
        Total number of non-boundary vertices.
    """

    candidate_id: str = ""
    graph_json: str = ""
    component_web_ids: list[str] = field(default_factory=list)
    composition_type: str = ""
    n_qubits: int = 0
    n_spiders: int = 0
    source_families: list[str] = field(default_factory=list)
    is_cross_family: bool = False

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON storage."""
        return {
            "candidate_id": self.candidate_id,
            "graph_json": self.graph_json,
            "component_web_ids": self.component_web_ids,
            "composition_type": self.composition_type,
            "n_qubits": self.n_qubits,
            "n_spiders": self.n_spiders,
            "source_families": self.source_families,
            "is_cross_family": self.is_cross_family,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateAlgorithm:
        """Deserialise from a plain dict."""
        return cls(
            candidate_id=d.get("candidate_id", ""),
            graph_json=d.get("graph_json", ""),
            component_web_ids=d.get("component_web_ids", []),
            composition_type=d.get("composition_type", ""),
            n_qubits=d.get("n_qubits", 0),
            n_spiders=d.get("n_spiders", 0),
            source_families=d.get("source_families", []),
            is_cross_family=d.get("is_cross_family", False),
        )
