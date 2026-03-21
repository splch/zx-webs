"""Stage artifact save/load system using orjson (with stdlib json fallback)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import orjson

    def _dumps(data: Any) -> bytes:
        return orjson.dumps(data, option=orjson.OPT_INDENT_2, default=str)

    def _loads(raw: bytes | str) -> Any:
        return orjson.loads(raw)

    _HAS_ORJSON = True
except ModuleNotFoundError:  # pragma: no cover
    import json as _json

    def _dumps(data: Any) -> bytes:  # type: ignore[misc]
        return _json.dumps(data, indent=2, default=str).encode("utf-8")

    def _loads(raw: bytes | str) -> Any:  # type: ignore[misc]
        return _json.loads(raw)

    _HAS_ORJSON = False

# ---------------------------------------------------------------------------
# Generic JSON helpers
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "manifest.json"


def save_json(data: Any, path: Path) -> None:
    """Serialize *data* as pretty-printed JSON to *path*.

    Parent directories are created automatically.
    Uses orjson when available (3-10x faster than stdlib json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_dumps(data))


def load_json(path: Path) -> Any:
    """Read and return the JSON content at *path*."""
    return _loads(path.read_bytes())


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


# ---------------------------------------------------------------------------
# Bulk save/load for web lists (single-file format)
# ---------------------------------------------------------------------------

WEBS_BULK_FILENAME = "webs_bulk.json"


def save_webs_bulk(webs_data: list[dict[str, Any]], stage_dir: Path) -> Path:
    """Persist a list of web dicts to a single JSON file.

    Returns the path to the written file.  This avoids writing thousands of
    individual JSON files (the #1 I/O bottleneck in the pipeline).
    """
    bulk_path = stage_dir / WEBS_BULK_FILENAME
    save_json(webs_data, bulk_path)
    return bulk_path


def load_webs_bulk(stage_dir: Path) -> list[dict[str, Any]]:
    """Load all web dicts from the bulk JSON file.

    Returns an empty list if the file does not exist.
    """
    bulk_path = stage_dir / WEBS_BULK_FILENAME
    if not bulk_path.exists():
        return []
    data = load_json(bulk_path)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {bulk_path}, got {type(data).__name__}"
        )
    return data
