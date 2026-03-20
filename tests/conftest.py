"""Shared test fixtures for ZX-Webs."""
from __future__ import annotations

import pytest
from pathlib import Path

from zx_webs.config import CorpusConfig, PipelineConfig, ZXConfig


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary data directory inside pytest's tmp_path."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def small_config(tmp_data_dir: Path) -> PipelineConfig:
    """A minimal pipeline config suitable for fast unit tests.

    Uses only ``oracular`` and ``entanglement`` families with small qubit
    counts so that corpus generation and ZX conversion complete quickly.
    """
    return PipelineConfig(
        data_dir=tmp_data_dir,
        corpus=CorpusConfig(
            families=["oracular", "entanglement"],
            max_qubits=5,
            qubit_counts=[3, 5],
        ),
        zx=ZXConfig(
            reduction="full_reduce",
            normalize=True,
        ),
    )
