"""Bridge from Qiskit QuantumCircuit to PyZX-compatible QASM 2.0.

PyZX's QASM parser supports a limited gate set.  This module transpiles
arbitrary Qiskit circuits down to that basis and exports clean QASM 2.0
strings with measurements, barriers, and classical registers removed.
"""
from __future__ import annotations

import logging

from qiskit import QuantumCircuit, transpile

logger = logging.getLogger(__name__)

# Gates that PyZX can parse from QASM 2.0.
PYZX_BASIS_GATES: list[str] = [
    "h", "cx", "cz", "x", "z", "s", "sdg", "t", "tdg",
    "rz", "rx", "ry", "ccx", "swap",
]


def _strip_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of *qc* with measurements, barriers, and classical bits removed."""
    clean = QuantumCircuit(qc.num_qubits)

    # Preserve qubit register names if possible
    for inst in qc.data:
        op = inst.operation
        if op.name in ("measure", "barrier", "reset"):
            continue
        # Map qubits by index
        qubit_indices = [qc.find_bit(q).index for q in inst.qubits]
        clean.append(op, qubit_indices)

    return clean


def _export_qasm2(qc: QuantumCircuit) -> str:
    """Export a QuantumCircuit as a QASM 2.0 string.

    Handles both Qiskit >=1.0 (``qiskit.qasm2.dumps``) and older versions
    (``QuantumCircuit.qasm()``).
    """
    # Qiskit >= 1.0 uses qiskit.qasm2.dumps
    try:
        from qiskit.qasm2 import dumps
        return dumps(qc)
    except ImportError:
        pass

    # Fallback for older Qiskit versions
    if hasattr(qc, "qasm"):
        result = qc.qasm()
        if isinstance(result, str):
            return result

    raise RuntimeError(
        "Unable to export QASM 2.0: neither qiskit.qasm2.dumps nor "
        "QuantumCircuit.qasm() is available."
    )


def circuit_to_pyzx_qasm(qc: QuantumCircuit) -> str:
    """Convert a Qiskit QuantumCircuit to QASM 2.0 compatible with PyZX.

    Steps:
        1. Remove measurements, barriers, and classical registers.
        2. Transpile to the PyZX-compatible basis gate set.
        3. Export as an OpenQASM 2.0 string.

    Parameters
    ----------
    qc:
        An arbitrary Qiskit ``QuantumCircuit``.

    Returns
    -------
    str
        A valid QASM 2.0 string that PyZX can parse.
    """
    # Step 1: strip non-unitary operations
    clean = _strip_measurements(qc)

    # Step 2: decompose / transpile to the PyZX basis
    transpiled = transpile(
        clean,
        basis_gates=PYZX_BASIS_GATES,
        optimization_level=0,
    )

    # Step 3: export as QASM 2.0
    qasm_str = _export_qasm2(transpiled)

    logger.debug(
        "Converted %d-qubit circuit to QASM (%d chars)",
        qc.num_qubits,
        len(qasm_str),
    )
    return qasm_str
