#!/usr/bin/env python -u
"""Benchmark mining time vs max_vertices."""
import sys
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from zx_webs.config import MiningConfig, CorpusConfig
from zx_webs.stage1_corpus.algorithms import build_corpus
from zx_webs.stage1_corpus.qasm_bridge import circuit_to_pyzx_qasm
from zx_webs.stage3_mining.gspan_adapter import GSpanAdapter
import pyzx as zx

config = CorpusConfig(
    families=["oracular", "simulation", "entanglement", "arithmetic"],
    qubit_counts=[3, 5],
    max_qubits=5,
)
corpus = build_corpus(config)
graphs = []
for entry in corpus:
    qasm = circuit_to_pyzx_qasm(entry["circuit"])
    c = zx.Circuit.from_qasm(qasm)
    g = c.to_graph()
    zx.full_reduce(g)
    if g.num_vertices() <= 75:
        graphs.append(g)
print(f"{len(graphs)} graphs")

for max_v in [4, 6, 8, 10, 12, 14]:
    mc = MiningConfig(
        min_support=2, min_vertices=4, max_vertices=max_v, max_input_vertices=75
    )
    adapter = GSpanAdapter(mc)
    t0 = time.time()
    results = adapter.mine(graphs)

    # Count 3+ boundary webs
    multi = 0
    for r in results:
        meta = adapter.extract_metadata(r)
        if meta.get("n_inputs", 0) >= 3 or meta.get("n_outputs", 0) >= 3:
            multi += 1

    elapsed = time.time() - t0
    print(
        f"max_v={max_v:2d}: {len(results):>8} patterns, "
        f"{multi:>5} with 3+ boundaries ({multi/max(len(results),1)*100:.1f}%), "
        f"in {elapsed:>7.1f}s"
    )
