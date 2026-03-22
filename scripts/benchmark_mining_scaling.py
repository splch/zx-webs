#!/usr/bin/env python3
"""Benchmark submine C++ mining time vs max_vertices.

Run on SLURM:
    srun --partition=short --ntasks=1 --cpus-per-task=4 --mem=16G \
        bash -c ".venv/bin/python -u scripts/benchmark_mining_scaling.py"
"""
import json
import multiprocessing as mp
import queue as _queue_mod
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import pyzx as zx

from zx_webs.config import MiningConfig
from zx_webs.stage3_mining.gspan_adapter import GSpanAdapter
from zx_webs.stage3_mining.graph_encoder import ZXLabelEncoder, pyzx_graphs_to_gspan_file

try:
    from submine.algorithms import gspan_cpp
    print("Backend: submine C++ (available)")
except ImportError:
    print("ERROR: submine C++ not available")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------
manifest_path = Path("data/zx_diagrams/manifest.json")
manifest = json.loads(manifest_path.read_text())

all_graphs = []
for entry in manifest:
    gp = Path(entry["graph_path"])
    if gp.exists():
        g = zx.Graph.from_json(gp.read_text())
        all_graphs.append(g)

graphs_50 = [g for g in all_graphs if g.num_vertices() <= 50]
print(f"Corpus: {len(graphs_50)} graphs (<=50 vertices)")
sizes = [g.num_vertices() for g in graphs_50]
print(f"  Vertex counts: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")

encoder = ZXLabelEncoder(phase_bins=8, include_phase=True)
with tempfile.TemporaryDirectory() as tmpdir:
    gspan_path = Path(tmpdir) / "corpus.gspan"
    pyzx_graphs_to_gspan_file(graphs_50, gspan_path, encoder)
    gspan_data = gspan_path.read_text()
print(f"gSpan data: {len(gspan_data)} chars\n")


def _mine_worker(data, minsup, minv, maxv, result_queue):
    from submine.algorithms import gspan_cpp as _cpp
    t0 = time.time()
    results = _cpp.mine_from_string(
        data, minsup=minsup, maxpat_min=minv, maxpat_max=maxv,
        directed=False, where=True,
    )
    elapsed = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result_queue.put((len(results), elapsed, peak_mb))


def run_test(data, minsup, minv, maxv, timeout_s=300):
    q = mp.Queue()
    p = mp.Process(target=_mine_worker, args=(data, minsup, minv, maxv, q))
    t0 = time.time()
    p.start()
    # Read queue BEFORE join to avoid pipe-buffer deadlock.
    try:
        result = q.get(timeout=timeout_s)
    except _queue_mod.Empty:
        result = None
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
        p.join()
        return None, time.time() - t0, 0
    if result is not None:
        return result
    return None, time.time() - t0, 0


# ---------------------------------------------------------------------------
# Test: max_v scaling at minsup=20 (all values tractable)
# ---------------------------------------------------------------------------
max_v_values = [4, 6, 8, 10, 12, 14, 16]

print(f"=== minsup=20, minv=2, {len(graphs_50)} graphs ===")
print(f"{'max_v':>6}  {'patterns':>10}  {'time_s':>8}  {'peak_MB':>8}")
print("-" * 40)

for max_v in max_v_values:
    n, elapsed, peak = run_test(gspan_data, 20, 2, max_v, timeout_s=300)
    if n is not None:
        print(f"{max_v:>6}  {n:>10}  {elapsed:>8.2f}  {peak:>8.0f}")
    else:
        print(f"{max_v:>6}  {'TIMEOUT':>10}  {elapsed:>8.0f}  {'---':>8}")

# ---------------------------------------------------------------------------
# Test: minsup=3 (default config), max_v=4 and 6
# ---------------------------------------------------------------------------
print(f"\n=== minsup=3, minv=2, {len(graphs_50)} graphs ===")
print(f"{'max_v':>6}  {'patterns':>10}  {'time_s':>8}  {'peak_MB':>8}")
print("-" * 40)

for max_v in [4, 6]:
    n, elapsed, peak = run_test(gspan_data, 3, 2, max_v, timeout_s=300)
    if n is not None:
        print(f"{max_v:>6}  {n:>10}  {elapsed:>8.2f}  {peak:>8.0f}")
    else:
        print(f"{max_v:>6}  {'TIMEOUT':>10}  {elapsed:>8.0f}  {'---':>8}")

# ---------------------------------------------------------------------------
# Adapter test: inline vs subprocess, verify correctness
# ---------------------------------------------------------------------------
print(f"\n=== Adapter correctness: inline vs subprocess ===")

mc_inline = MiningConfig(
    min_support=20, min_vertices=2, max_vertices=12,
    max_input_vertices=50, mining_timeout=0,
)
mc_sub = MiningConfig(
    min_support=20, min_vertices=2, max_vertices=12,
    max_input_vertices=50, mining_timeout=300,
)

adapter_inline = GSpanAdapter(mc_inline)
t0 = time.time()
r_inline = adapter_inline.mine(graphs_50)
print(f"  Inline:     {len(r_inline):>6} results in {time.time()-t0:.2f}s")

adapter_sub = GSpanAdapter(mc_sub)
t0 = time.time()
r_sub = adapter_sub.mine(graphs_50)
print(f"  Subprocess: {len(r_sub):>6} results in {time.time()-t0:.2f}s")

if len(r_inline) == len(r_sub):
    print("  PASS: results match")
else:
    print(f"  FAIL: {len(r_inline)} != {len(r_sub)}")

# Metadata extraction
t0 = time.time()
for r in r_inline:
    adapter_inline.extract_metadata(r)
print(f"  Metadata: {len(r_inline)} extractions in {time.time()-t0:.3f}s")

print("\nDone.")
