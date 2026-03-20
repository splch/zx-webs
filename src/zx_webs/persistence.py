"""Stage artifact save/load system using JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Generic JSON helpers
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "manifest.json"


def save_json(data: Any, path: Path) -> None:
    """Serialize *data* as pretty-printed JSON to *path*.

    Parent directories are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Read and return the JSON content at *path*."""
    with open(path, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Manifest helpers (one manifest per pipeline stage directory)
# ---------------------------------------------------------------------------

def save_manifest(entries: list[dict[str, Any]], stage_dir: Path) -> None:
    """Write a stage manifest (a list of entry dicts) to *stage_dir*/manifest.json."""
    save_json(entries, stage_dir / MANIFEST_FILENAME)


def load_manifest(stage_dir: Path) -> list[dict[str, Any]]:
    """Load the manifest from *stage_dir*/manifest.json.

    Returns an empty list if the manifest file does not exist.
    """
    manifest_path = stage_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []
    data = load_json(manifest_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {manifest_path}, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Graph-specific JSON helpers
# ---------------------------------------------------------------------------

def save_graph_json(graph_data: dict[str, Any], path: Path) -> None:
    """Persist a ZX-graph dictionary to *path* as JSON.

    The dictionary should follow the project's internal graph schema
    (vertices, edges, metadata, etc.).
    """
    save_json(graph_data, path)


def load_graph_json(path: Path) -> dict[str, Any]:
    """Load a ZX-graph dictionary from a JSON file at *path*."""
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data
