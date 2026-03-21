"""Algorithm family metadata for corpus generation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlgorithmFamily:
    """Describes a family of quantum algorithms."""

    name: str
    description: str


FAMILIES: dict[str, AlgorithmFamily] = {
    "oracular": AlgorithmFamily(
        "oracular",
        "Oracle-based algorithms (Grover, Deutsch-Jozsa, etc.)",
    ),
    "arithmetic": AlgorithmFamily(
        "arithmetic",
        "Arithmetic and transform algorithms (QFT, QPE, adders)",
    ),
    "variational": AlgorithmFamily(
        "variational",
        "Variational hybrid algorithms (VQE, QAOA)",
    ),
    "simulation": AlgorithmFamily(
        "simulation",
        "Hamiltonian simulation algorithms",
    ),
    "entanglement": AlgorithmFamily(
        "entanglement",
        "Entanglement preparation circuits",
    ),
    "error_correction": AlgorithmFamily(
        "error_correction",
        "Quantum error correction encoding and syndrome circuits",
    ),
    "linear_algebra": AlgorithmFamily(
        "linear_algebra",
        "Linear algebra primitives (Hadamard test, swap test, HHL-style)",
    ),
    "communication": AlgorithmFamily(
        "communication",
        "Quantum communication protocols (teleportation, superdense coding, QKD)",
    ),
}
