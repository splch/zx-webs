"""Wrapper around PyZX simplification strategies."""
from __future__ import annotations

import copy
from typing import Literal

import pyzx as zx

# The set of supported reduction method names.
ReductionMethod = Literal["full_reduce", "teleport_reduce", "clifford_simp", "none"]

_VALID_METHODS: frozenset[str] = frozenset(
    {"full_reduce", "teleport_reduce", "clifford_simp", "none"}
)


def simplify_graph(
    g: zx.Graph,
    method: str = "full_reduce",
    normalize: bool = True,
) -> zx.Graph:
    """Simplify a ZX-diagram using the specified method.

    The input graph is **never** modified; all work is performed on a deep
    copy.

    Parameters
    ----------
    g:
        The input ZX-diagram.
    method:
        Simplification strategy.

        ``"full_reduce"``
            Most powerful simplification -- applies all available ZX-calculus
            rewrite rules.  Loses circuit structure.
        ``"teleport_reduce"``
            Preserves circuit structure while moving phases through the
            diagram via teleportation.
        ``"clifford_simp"``
            Applies only Clifford-group simplification rules.
        ``"none"``
            Return a copy of the graph without any simplification.
    normalize:
        Whether to call ``g.normalize()`` after simplification to reorder
        boundary vertices.

    Returns
    -------
    zx.Graph
        A new, simplified graph.

    Raises
    ------
    ValueError
        If *method* is not one of the recognised strategies.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown simplification method {method!r}. "
            f"Expected one of {sorted(_VALID_METHODS)}."
        )

    g_copy: zx.Graph = copy.deepcopy(g)

    if method == "full_reduce":
        zx.simplify.full_reduce(g_copy)
    elif method == "teleport_reduce":
        zx.simplify.teleport_reduce(g_copy)
    elif method == "clifford_simp":
        zx.simplify.clifford_simp(g_copy)
    # method == "none" -- nothing to do

    if normalize:
        g_copy.normalize()

    return g_copy
