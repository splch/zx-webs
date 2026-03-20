"""Stage 7 -- report generation.

:func:`run_stage7` collects statistics from all previous pipeline stages and
produces a JSON summary and (optionally) an HTML report.
"""
from __future__ import annotations

import html as html_mod
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zx_webs.config import ReportConfig
from zx_webs.persistence import load_json, load_manifest, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_manifest_safe(stage_dir: Path) -> list[dict[str, Any]]:
    """Load a manifest, returning an empty list on any error."""
    try:
        return load_manifest(stage_dir)
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(data_dir: Path) -> dict[str, Any]:
    """Collect stats from all pipeline stages into a summary dict.

    Parameters
    ----------
    data_dir:
        Root data directory (e.g. ``data/``).  Each stage writes its
        artefacts in a subdirectory.

    Returns
    -------
    dict
        A structured summary with per-stage statistics.
    """
    summary: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "stages": {},
    }

    # Stage 1: corpus
    corpus_manifest = _load_manifest_safe(data_dir / "corpus")
    summary["stages"]["corpus"] = {"n_algorithms": len(corpus_manifest)}

    # Stage 2: ZX diagrams
    zx_manifest = _load_manifest_safe(data_dir / "zx_diagrams")
    summary["stages"]["zx_diagrams"] = {"n_diagrams": len(zx_manifest)}

    # Stage 3: mined webs
    webs_manifest = _load_manifest_safe(data_dir / "mined_webs")
    summary["stages"]["mining"] = {"n_webs": len(webs_manifest)}

    # Stage 4: candidates
    candidates_manifest = _load_manifest_safe(data_dir / "candidates")
    summary["stages"]["compose"] = {"n_candidates": len(candidates_manifest)}

    # Stage 5: filtered
    filtered_manifest = _load_manifest_safe(data_dir / "filtered")
    summary["stages"]["filter"] = {"n_survivors": len(filtered_manifest)}

    # Stage 6: benchmarks
    bench_results_path = data_dir / "benchmarks" / "results.json"
    if bench_results_path.exists():
        try:
            bench_results = load_json(bench_results_path)
        except Exception:  # noqa: BLE001
            bench_results = []

        if isinstance(bench_results, list):
            n_dominant = sum(1 for r in bench_results if r.get("dominates_any_baseline"))
            summary["stages"]["bench"] = {
                "n_benchmarked": len(bench_results),
                "n_dominating": n_dominant,
            }

    return summary


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZX-Webs Pipeline Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ color: #1a1a2e; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.5rem 1rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .meta {{ color: #666; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>ZX-Webs Pipeline Report</h1>
<p class="meta">Generated at: {generated_at}</p>

<h2>Pipeline Stage Summary</h2>
<table>
<tr><th>Stage</th><th>Metric</th><th>Value</th></tr>
{rows}
</table>
</body>
</html>
"""

# Mapping from stage key to human-readable label.
_STAGE_LABELS: dict[str, str] = {
    "corpus": "1. Corpus",
    "zx_diagrams": "2. ZX Diagrams",
    "mining": "3. Mining",
    "compose": "4. Composition",
    "filter": "5. Filtering",
    "bench": "6. Benchmarking",
}


def generate_report_html(summary: dict[str, Any], output_path: Path) -> None:
    """Generate a simple HTML report from the summary.

    Parameters
    ----------
    summary:
        The dict returned by :func:`generate_summary`.
    output_path:
        File path for the HTML output.
    """
    rows_html: list[str] = []
    stages = summary.get("stages", {})

    for stage_key, label in _STAGE_LABELS.items():
        stage_data = stages.get(stage_key, {})
        if not stage_data:
            continue
        for metric_name, value in stage_data.items():
            escaped_label = html_mod.escape(label)
            escaped_metric = html_mod.escape(str(metric_name))
            escaped_value = html_mod.escape(str(value))
            rows_html.append(
                f"<tr><td>{escaped_label}</td>"
                f"<td>{escaped_metric}</td>"
                f"<td>{escaped_value}</td></tr>"
            )

    generated_at = html_mod.escape(summary.get("generated_at", "unknown"))
    html_content = _HTML_TEMPLATE.format(
        generated_at=generated_at,
        rows="\n".join(rows_html),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)
    logger.info("HTML report written to %s", output_path)


# ---------------------------------------------------------------------------
# Stage 7 entry point
# ---------------------------------------------------------------------------


def run_stage7(
    data_dir: Path,
    output_dir: Path,
    config: ReportConfig | None = None,
) -> dict[str, Any]:
    """Run Stage 7: generate reports.

    Parameters
    ----------
    data_dir:
        Root data directory containing outputs from all previous stages.
    output_dir:
        Where report artefacts will be written.
    config:
        Report parameters.  Falls back to ``ReportConfig()`` defaults
        when *None*.

    Returns
    -------
    dict
        The summary dictionary that was persisted as ``summary.json``.
    """
    if config is None:
        config = ReportConfig()

    summary = generate_summary(data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "summary.json")

    if "html" in config.output_format:
        generate_report_html(summary, output_dir / "report.html")

    logger.info("Stage 7 complete: report written to %s", output_dir)
    return summary
