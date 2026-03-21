"""Boundary wire compatibility analysis for ZX-Web composition.

Before two ZX-Webs can be composed sequentially, their boundary wires must
be compatible.  This module provides helpers that:

1. Count input/output wires on a web.
2. Decide whether two boundary wires can be connected.
3. Score how well two boundary wires match for prioritisation.
4. Choose the appropriate edge type for the junction.
"""
from __future__ import annotations

from zx_webs.stage3_mining.zx_web import BoundaryWire, ZXWeb

# PyZX edge type constants (used throughout the project).
_EDGE_SIMPLE = 1
_EDGE_HADAMARD = 2


# ---------------------------------------------------------------------------
# Wire counting
# ---------------------------------------------------------------------------


def count_boundary_wires(web: ZXWeb) -> tuple[int, int]:
    """Return ``(n_input_wires, n_output_wires)`` for *web*.

    Wires with ``direction == "unknown"`` are **not** counted in either
    category.  Use :attr:`ZXWeb.n_inputs` / :attr:`ZXWeb.n_outputs` for the
    stored values; this function re-derives them from the boundary wire list.
    """
    n_in = sum(1 for bw in web.boundary_wires if bw.direction == "input")
    n_out = sum(1 for bw in web.boundary_wires if bw.direction == "output")
    return n_in, n_out


# ---------------------------------------------------------------------------
# Wire compatibility
# ---------------------------------------------------------------------------


def wires_compatible(out_wire: BoundaryWire, in_wire: BoundaryWire) -> bool:
    """Check whether two boundary wires can be connected.

    In ZX-calculus *any* two spiders may be connected by either a simple
    or a Hadamard edge, so this always returns ``True``.  The function
    exists as a hook for future heuristic filters (e.g. restricting
    connections to same-type spiders only).
    """
    return True


def wire_compatibility_score(
    out_wire: BoundaryWire, in_wire: BoundaryWire
) -> float:
    """Score how well two boundary wires match for composition.

    Higher scores indicate cleaner, more meaningful compositions.

    Scoring heuristic:

    * **Base score** = 1.0 (every connection is valid in ZX-calculus).
    * **Same spider type bonus** (+1.0) -- same-type spiders connected by
      a simple edge fuse cleanly under ZX-calculus spider fusion.
    * **Zero-phase bonus** (+0.5) -- connecting zero-phase spiders produces
      simpler compositions that are easier to reason about.

    Parameters
    ----------
    out_wire:
        The output boundary wire of the first web.
    in_wire:
        The input boundary wire of the second web.

    Returns
    -------
    float
        A non-negative compatibility score (higher is better).
    """
    score = 1.0
    if out_wire.spider_type == in_wire.spider_type:
        score += 1.0  # same type bonus (will fuse cleanly)
    if abs(out_wire.spider_phase) < 0.01 and abs(in_wire.spider_phase) < 0.01:
        score += 0.5  # zero-phase bonus (simpler composition)
    return score


# ---------------------------------------------------------------------------
# Junction edge type
# ---------------------------------------------------------------------------


def junction_edge_type(out_wire: BoundaryWire, in_wire: BoundaryWire) -> int:
    """Determine the edge type to use when connecting two boundary wires.

    Heuristic:

    * **Same** spider type on both sides -> simple edge (1).  When two
      same-type spiders are connected by a simple edge, the ZX-calculus
      spider fusion rule allows them to merge into one spider.
    * **Different** spider types -> Hadamard edge (2).  This preserves the
      type distinction and is the natural choice when mixing Z and X spiders.

    Returns
    -------
    int
        ``1`` (simple) or ``2`` (Hadamard).
    """
    if out_wire.spider_type == in_wire.spider_type:
        return _EDGE_SIMPLE
    return _EDGE_HADAMARD
