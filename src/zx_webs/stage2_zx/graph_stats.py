"""Compute statistics about a PyZX ZX-diagram."""
from __future__ import annotations

from typing import Any

import pyzx as zx


# PyZX vertex type constants (from pyzx.utils.VertexType).
_VT_BOUNDARY = 0
_VT_Z = 1
_VT_X = 2
_VT_H_BOX = 3

# PyZX edge type constants (from pyzx.utils.EdgeType).
_ET_SIMPLE = 1
_ET_HADAMARD = 2


def compute_graph_stats(g: zx.Graph) -> dict[str, Any]:
    """Compute statistics about a ZX-diagram.

    Parameters
    ----------
    g:
        A PyZX ``Graph`` instance.

    Returns
    -------
    dict
        Keys:

        - **n_vertices** -- total vertex count
        - **n_edges** -- total edge count
        - **n_z_spiders** -- count of Z-type vertices
        - **n_x_spiders** -- count of X-type vertices
        - **n_boundary** -- count of boundary vertices
        - **n_h_boxes** -- count of H-box vertices
        - **n_simple_edges** -- count of simple (regular) edges
        - **n_hadamard_edges** -- count of Hadamard edges
        - **n_inputs** -- number of input boundary vertices
        - **n_outputs** -- number of output boundary vertices
    """
    # -- Vertex counts -------------------------------------------------------
    n_z = 0
    n_x = 0
    n_boundary = 0
    n_h_box = 0

    for v in g.vertices():
        vtype = g.type(v)
        if vtype == _VT_Z:
            n_z += 1
        elif vtype == _VT_X:
            n_x += 1
        elif vtype == _VT_BOUNDARY:
            n_boundary += 1
        elif vtype == _VT_H_BOX:
            n_h_box += 1

    n_vertices = n_z + n_x + n_boundary + n_h_box

    # -- Edge counts ---------------------------------------------------------
    n_simple = 0
    n_hadamard = 0

    for e in g.edges():
        etype = g.edge_type(e)
        if etype == _ET_SIMPLE:
            n_simple += 1
        elif etype == _ET_HADAMARD:
            n_hadamard += 1

    n_edges = n_simple + n_hadamard

    # -- Boundary I/O --------------------------------------------------------
    n_inputs = len(g.inputs())
    n_outputs = len(g.outputs())

    return {
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "n_z_spiders": n_z,
        "n_x_spiders": n_x,
        "n_boundary": n_boundary,
        "n_h_boxes": n_h_box,
        "n_simple_edges": n_simple,
        "n_hadamard_edges": n_hadamard,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
    }
