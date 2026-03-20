"""Shared type aliases and enums used across the ZX-Webs pipeline."""
from __future__ import annotations

from enum import IntEnum
from fractions import Fraction
from typing import Union


# ---------------------------------------------------------------------------
# Vertex and edge enums (mirror PyZX conventions)
# ---------------------------------------------------------------------------

class VertexType(IntEnum):
    """ZX-diagram vertex types, mirroring ``pyzx.utils.VertexType``."""

    BOUNDARY = 0
    Z = 1
    X = 2
    H_BOX = 3


class EdgeType(IntEnum):
    """ZX-diagram edge types, mirroring ``pyzx.utils.EdgeType``."""

    SIMPLE = 0
    HADAMARD = 1


# ---------------------------------------------------------------------------
# Lightweight identifier aliases
# ---------------------------------------------------------------------------

AlgorithmId = str
"""Unique name for an algorithm family or circuit instance."""

WebId = str
"""Unique identifier for a mined ZX sub-diagram (web)."""

CandidateId = str
"""Unique identifier for a composed candidate circuit."""

SurvivorId = str
"""Identifier for a candidate that survived the filtering stage."""

# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

FractionLike = Union[Fraction, float, int]
"""Anything that can represent a ZX-diagram phase value."""
