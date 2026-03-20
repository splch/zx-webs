"""Stage 1 -- Algorithm Corpus.

Builds a library of quantum algorithm circuits (as Qiskit QuantumCircuits)
and provides a bridge to export them as PyZX-compatible QASM 2.0 strings.
"""
from __future__ import annotations

from zx_webs.stage1_corpus.algorithms import ALGORITHM_REGISTRY, build_corpus
from zx_webs.stage1_corpus.families import FAMILIES, AlgorithmFamily
from zx_webs.stage1_corpus.qasm_bridge import circuit_to_pyzx_qasm

__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmFamily",
    "FAMILIES",
    "build_corpus",
    "circuit_to_pyzx_qasm",
]
