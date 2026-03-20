"""Stage 5 -- circuit extraction filter and deduplication."""
from __future__ import annotations

from zx_webs.stage5_filter.deduplicator import (
    circuits_equivalent,
    deduplicate_circuits,
)
from zx_webs.stage5_filter.extractor import (
    ExtractionResult,
    run_stage5,
    try_extract_circuit,
)

__all__ = [
    "ExtractionResult",
    "circuits_equivalent",
    "deduplicate_circuits",
    "run_stage5",
    "try_extract_circuit",
]
