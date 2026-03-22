"""Minimal reproduction script for gSpan mining hang on head node."""
import json
import sys
import time
import tempfile
import signal
from pathlib import Path

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Mining exceeded timeout")

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zx_webs.stage3_mining.graph_encoder import ZXLabelEncoder, pyzx_graphs_to_gspan_file
import pyzx as zx

# Load graphs
manifest_path = Path("data/zx_diagrams/manifest.json")
manifest = json.loads(manifest_path.read_text())
print(f"Total graphs in manifest: {len(manifest)}")

graphs = []
for entry in manifest:
    gp = Path(entry["graph_path"])
    if gp.exists():
        g = zx.Graph.from_json(gp.read_text())
        graphs.append(g)

print(f"Loaded {len(graphs)} graphs")

# Filter like the pipeline does (max_input_vertices=75)
max_v = 75
small_graphs = [g for g in graphs if g.num_vertices() <= max_v]
print(f"Small graphs (<=75 vertices): {len(small_graphs)}")

# Show graph size distribution
sizes = [g.num_vertices() for g in small_graphs]
print(f"  min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")

# Encode to gSpan format
encoder = ZXLabelEncoder(phase_bins=8, include_phase=True)
with tempfile.TemporaryDirectory() as tmpdir:
    gspan_path = Path(tmpdir) / "corpus.gspan"
    pyzx_graphs_to_gspan_file(small_graphs, gspan_path, encoder)
    gspan_data = gspan_path.read_text()

print(f"gSpan data size: {len(gspan_data)} chars, {gspan_data.count('t #')-1} graphs")

# Check if submine C++ is available
try:
    from submine.algorithms import gspan_cpp
    print("submine C++ backend: AVAILABLE")
except ImportError:
    print("submine C++ backend: NOT AVAILABLE")
    sys.exit(1)

# Test 1: Quick test with small subset
print("\n--- Test 1: 10 graphs, max_vertices=6 ---")
subset_10 = small_graphs[:10]
with tempfile.TemporaryDirectory() as tmpdir:
    gspan_path = Path(tmpdir) / "test.gspan"
    pyzx_graphs_to_gspan_file(subset_10, gspan_path, encoder)
    data_10 = gspan_path.read_text()

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)
t0 = time.time()
try:
    res = gspan_cpp.mine_from_string(data_10, minsup=2, maxpat_min=4, maxpat_max=6, directed=False, where=True)
    print(f"  Completed in {time.time()-t0:.1f}s, found {len(res)} patterns")
except TimeoutError:
    print(f"  TIMEOUT after {time.time()-t0:.1f}s")
signal.alarm(0)

# Test 2: Same subset but with where=False
print("\n--- Test 2: 10 graphs, max_vertices=6, where=False ---")
signal.alarm(30)
t0 = time.time()
try:
    res = gspan_cpp.mine_from_string(data_10, minsup=2, maxpat_min=4, maxpat_max=6, directed=False, where=False)
    print(f"  Completed in {time.time()-t0:.1f}s, found {len(res)} patterns")
except TimeoutError:
    print(f"  TIMEOUT after {time.time()-t0:.1f}s")
signal.alarm(0)

# Test 3: Full dataset with small max_vertices
print("\n--- Test 3: All 252 graphs, max_vertices=6, where=True ---")
signal.alarm(60)
t0 = time.time()
try:
    res = gspan_cpp.mine_from_string(gspan_data, minsup=2, maxpat_min=4, maxpat_max=6, directed=False, where=True)
    print(f"  Completed in {time.time()-t0:.1f}s, found {len(res)} patterns")
except TimeoutError:
    print(f"  TIMEOUT after {time.time()-t0:.1f}s")
signal.alarm(0)

# Test 4: Full dataset with larger max_vertices (the real config)
print("\n--- Test 4: All 252 graphs, max_vertices=12, where=True ---")
signal.alarm(300)
t0 = time.time()
try:
    res = gspan_cpp.mine_from_string(gspan_data, minsup=2, maxpat_min=4, maxpat_max=12, directed=False, where=True)
    print(f"  Completed in {time.time()-t0:.1f}s, found {len(res)} patterns")
except TimeoutError:
    print(f"  TIMEOUT after {time.time()-t0:.1f}s")
signal.alarm(0)

print("\nDone.")
