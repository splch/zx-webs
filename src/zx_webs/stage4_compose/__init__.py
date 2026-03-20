"""Stage 4 -- combinatorial stitching of ZX-Webs into candidate algorithms."""
from __future__ import annotations

from zx_webs.stage4_compose.boundary import (
    count_boundary_wires,
    junction_edge_type,
    wires_compatible,
)
from zx_webs.stage4_compose.candidate import CandidateAlgorithm
from zx_webs.stage4_compose.stitcher import Stitcher, run_stage4

__all__ = [
    "CandidateAlgorithm",
    "Stitcher",
    "count_boundary_wires",
    "junction_edge_type",
    "run_stage4",
    "wires_compatible",
]
