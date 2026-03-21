"""Circuit extraction filter for candidate ZX-diagrams.

:func:`try_extract_circuit` attempts to convert a ZX-diagram into a quantum
circuit.  The attempt may fail (the diagram is not graph-like or extraction
diverges) or succeed but produce an unacceptably large circuit.  Both cases
are handled gracefully and reported via :class:`ExtractionResult`.

The extraction pipeline uses a multi-step approach:

1. Ensure graph-like form via ``to_graph_like()``.
2. Apply ``full_reduce()`` for simplification.
3. Check generalized flow (gflow) as a fast pre-filter.
4. Extract the circuit with ``extract_circuit()``.

:func:`run_stage5` is the Stage 5 entry point that loads candidates produced
by Stage 4, attempts extraction on each, deduplicates the survivors, and
persists the results.
"""
from __future__ import annotations

import logging
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyzx as zx
from tqdm import tqdm

from zx_webs.config import FilterConfig
from zx_webs.persistence import load_json, load_manifest, save_json, save_manifest
from zx_webs.stage4_compose.candidate import CandidateAlgorithm
from zx_webs.stage5_filter.deduplicator import deduplicate_circuits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timeout context manager (SIGALRM-based, Linux only)
# ---------------------------------------------------------------------------


class _Timeout:
    """Context manager that raises :class:`TimeoutError` after *seconds*.

    Uses ``signal.SIGALRM``, which is only available on Unix-like systems and
    only works in the **main thread**.  If SIGALRM is unavailable (Windows or
    non-main thread), the timeout is silently skipped.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = max(1, int(seconds))
        self._can_alarm = hasattr(signal, "SIGALRM")
        self._old_handler: Any = None

    def __enter__(self) -> _Timeout:
        if self._can_alarm:
            try:
                self._old_handler = signal.signal(signal.SIGALRM, self._handler)
                signal.alarm(self.seconds)
            except ValueError:
                # Not in the main thread -- skip alarm.
                self._can_alarm = False
        return self

    def __exit__(self, *args: Any) -> None:
        if self._can_alarm:
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)

    @staticmethod
    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Circuit extraction timed out")


# ---------------------------------------------------------------------------
# Extraction result
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Result of a circuit extraction attempt.

    Attributes
    ----------
    success:
        ``True`` if a circuit was extracted successfully.
    circuit_qasm:
        QASM string of the extracted circuit (empty on failure).
    stats:
        Gate-count statistics (keys like ``"n_gates"``, ``"two_qubit_count"``,
        ``"t_count"``).  Empty on failure.
    error:
        Human-readable error description (empty on success).
    """

    success: bool = False
    circuit_qasm: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _parse_stats(circuit: Any) -> dict[str, Any]:
    """Extract structured gate-count statistics from a PyZX circuit."""
    return {
        "n_gates": len(circuit.gates),
        "two_qubit_count": circuit.twoqubitcount(),
        "qubits": circuit.qubits,
        "stats_str": circuit.stats(),
    }


def try_extract_circuit(
    graph: zx.Graph,
    timeout: float = 30.0,
    max_cnot_blowup: float = 5.0,
) -> ExtractionResult:
    """Attempt to extract a quantum circuit from a ZX-diagram.

    The function **always operates on a copy** of *graph* so the caller's
    graph is never mutated.

    Steps
    -----
    1. Copy the graph.
    2. Ensure graph-like form via ``to_graph_like()``.
    3. Run ``zx.full_reduce`` for simplification.
    4. Check generalized flow (gflow) as a fast pre-filter.
    5. Call ``zx.extract_circuit`` inside a timeout guard.
    6. Check for CNOT blowup (more than *max_cnot_blowup* x qubits).

    Parameters
    ----------
    graph:
        A PyZX ``Graph`` instance (will not be mutated).
    timeout:
        Maximum seconds to spend on extraction.
    max_cnot_blowup:
        If the two-qubit gate count exceeds ``max_cnot_blowup * n_qubits``,
        the result is marked as failed with a blowup warning.

    Returns
    -------
    ExtractionResult
    """
    g = graph.copy()

    # Ensure the graph has boundary information.
    if not g.inputs() and not g.outputs():
        return ExtractionResult(
            success=False,
            error="Graph has no input/output boundary vertices",
        )

    # Step 1: Ensure graph-like form.
    try:
        from pyzx.simplify import to_graph_like
        to_graph_like(g)
    except Exception:  # noqa: BLE001
        pass  # Best effort; full_reduce may still succeed.

    # Step 2: Apply full_reduce.
    try:
        zx.full_reduce(g)
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(
            success=False,
            error=f"full_reduce failed: {exc}",
        )

    # Step 3: Check gflow (fast pre-filter).
    try:
        from pyzx.gflow import gflow
        gf = gflow(g)
        if gf is None:
            return ExtractionResult(
                success=False,
                error="No generalized flow (gflow) found",
            )
    except Exception:  # noqa: BLE001
        pass  # gflow check failed; try extraction anyway.

    # Step 4: Extract circuit.
    try:
        with _Timeout(timeout):
            circuit = zx.extract_circuit(g.copy(), optimize_cnots=2)
    except TimeoutError:
        return ExtractionResult(success=False, error="Extraction timed out")
    except ValueError as exc:
        return ExtractionResult(
            success=False,
            error=f"extract_circuit ValueError: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(
            success=False,
            error=f"extract_circuit failed: {exc}",
        )

    # -- Check for CNOT blowup -----------------------------------------------
    n_qubits = circuit.qubits
    two_q = circuit.twoqubitcount()
    if n_qubits > 0 and two_q > max_cnot_blowup * n_qubits:
        return ExtractionResult(
            success=False,
            error=(
                f"CNOT blowup: {two_q} two-qubit gates for {n_qubits} qubits "
                f"(limit {max_cnot_blowup}x)"
            ),
        )

    stats = _parse_stats(circuit)
    qasm = circuit.to_qasm()

    return ExtractionResult(
        success=True,
        circuit_qasm=qasm,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Worker function for parallel evaluation
# ---------------------------------------------------------------------------


def _evaluate_candidate_data(
    cand_data: dict[str, Any],
    timeout: float,
    max_cnot_blowup: float,
) -> dict[str, Any] | None:
    """Evaluate a single candidate (designed for process-pool execution).

    Parameters
    ----------
    cand_data:
        Serialised candidate dict (must contain ``graph_json``).
    timeout:
        Extraction timeout in seconds.
    max_cnot_blowup:
        Maximum CNOT blowup factor.

    Returns
    -------
    dict | None
        Survivor dict on success, ``None`` on failure.
    """
    try:
        g = zx.Graph.from_json(cand_data["graph_json"])
        result = try_extract_circuit(
            g, timeout=timeout, max_cnot_blowup=max_cnot_blowup,
        )
        if result.success:
            return {
                "candidate_id": cand_data.get("candidate_id", ""),
                "circuit_qasm": result.circuit_qasm,
                "stats": result.stats,
                "composition_type": cand_data.get("composition_type", ""),
                "component_web_ids": cand_data.get("component_web_ids", []),
                "n_qubits": cand_data.get("n_qubits", 0),
            }
    except Exception:  # noqa: BLE001
        pass
    return None


def _extract_worker(args: tuple[dict[str, Any], float, float]) -> dict[str, Any] | None:
    """Worker function for multiprocessing Pool-based parallel extraction.

    Parameters
    ----------
    args:
        Tuple of ``(candidate_data, timeout, max_cnot_blowup)``.

    Returns
    -------
    dict | None
        Survivor dict on success, ``None`` on failure.
    """
    cand_data, timeout, max_cnot_blowup = args
    return _evaluate_candidate_data(cand_data, timeout, max_cnot_blowup)


# ---------------------------------------------------------------------------
# Stage 5 entry point
# ---------------------------------------------------------------------------


def run_stage5(
    candidates_dir: Path,
    output_dir: Path,
    config: FilterConfig | None = None,
) -> list[dict[str, Any]]:
    """Run Stage 5: filter candidates by circuit extractability.

    Workflow
    -------
    1. Load candidate algorithms from Stage 4 output.
    2. Attempt circuit extraction on each candidate (in parallel if n_workers > 1).
    3. Deduplicate surviving circuits.
    4. Persist the survivors and a manifest under *output_dir*.

    Parameters
    ----------
    candidates_dir:
        Directory containing Stage 4 outputs (``manifest.json`` and
        ``candidates/*.json``).
    output_dir:
        Where Stage 5 artefacts will be written.
    config:
        Filtering parameters.  Falls back to ``FilterConfig()`` defaults
        when *None*.

    Returns
    -------
    list[dict]
        Manifest entries for surviving candidates.
    """
    if config is None:
        config = FilterConfig()

    # -- 1. Load candidates from Stage 4 manifest ----------------------------
    manifest = load_manifest(candidates_dir)
    if not manifest:
        logger.warning(
            "Stage 4 manifest at %s is empty -- nothing to filter.",
            candidates_dir,
        )
        return []

    candidates: list[CandidateAlgorithm] = []
    for entry in manifest:
        cand_path = Path(entry["candidate_path"])
        if not cand_path.exists():
            logger.warning("Candidate file not found: %s -- skipping.", cand_path)
            continue
        cand_data = load_json(cand_path)
        candidates.append(CandidateAlgorithm.from_dict(cand_data))

    if not candidates:
        logger.warning("No valid candidates loaded from %s.", candidates_dir)
        return []

    logger.info("Loaded %d candidates for filtering.", len(candidates))

    # -- 2. Attempt extraction ------------------------------------------------
    # Determine number of worker processes.
    n_workers = config.n_workers
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, len(candidates), 16)
    else:
        n_workers = min(n_workers, len(candidates))

    survivors: list[dict[str, Any]] = []
    n_success = 0
    n_fail = 0

    # Prepare worker arguments.
    worker_args: list[tuple[dict[str, Any], float, float]] = [
        (
            cand.to_dict(),
            config.extract_timeout_seconds,
            config.max_cnot_blowup_factor,
        )
        for cand in candidates
    ]

    if n_workers > 1 and len(candidates) > 1:
        logger.info(
            "Running parallel extraction with %d workers on %d candidates.",
            n_workers,
            len(candidates),
        )
        with Pool(processes=n_workers) as pool:
            results = pool.map(_extract_worker, worker_args)

        for result in tqdm(results, desc="Stage 5: Collecting results", unit="cand"):
            if result is not None:
                n_success += 1
                survivors.append(result)
            else:
                n_fail += 1
    else:
        # Sequential fallback.
        for cand in tqdm(candidates, desc="Stage 5: Extracting circuits", unit="cand"):
            g = zx.Graph.from_json(cand.graph_json)
            result = try_extract_circuit(
                g,
                timeout=config.extract_timeout_seconds,
                max_cnot_blowup=config.max_cnot_blowup_factor,
            )
            if result.success:
                n_success += 1
                survivors.append(
                    {
                        "candidate_id": cand.candidate_id,
                        "circuit_qasm": result.circuit_qasm,
                        "stats": result.stats,
                        "composition_type": cand.composition_type,
                        "component_web_ids": cand.component_web_ids,
                        "n_qubits": cand.n_qubits,
                    }
                )
            else:
                n_fail += 1
                logger.debug(
                    "Candidate %s failed extraction: %s",
                    cand.candidate_id,
                    result.error,
                )

    logger.info(
        "Extraction: %d succeeded, %d failed out of %d candidates.",
        n_success,
        n_fail,
        len(candidates),
    )

    # -- 3. Deduplicate -------------------------------------------------------
    survivors = deduplicate_circuits(survivors, method=config.dedup_method)

    logger.info("%d unique circuits after deduplication.", len(survivors))

    # -- 4. Persist results ---------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    circuits_dir = output_dir / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    for idx, surv in enumerate(survivors):
        survivor_id = f"surv_{idx:04d}"
        surv["survivor_id"] = survivor_id

        circuit_path = circuits_dir / f"{survivor_id}.json"
        save_json(surv, circuit_path)

        manifest_entries.append(
            {
                "survivor_id": survivor_id,
                "circuit_path": str(circuit_path),
                "candidate_id": surv["candidate_id"],
                "n_qubits": surv.get("n_qubits", 0),
                "stats": surv.get("stats", {}),
            }
        )

    save_manifest(manifest_entries, output_dir)

    logger.info(
        "Stage 5 complete -- %d survivors written to %s",
        len(survivors),
        output_dir,
    )
    return manifest_entries
