# ZX-Webs v1: Baseline

## Overview

v1 establishes an unbiased, exploratory baseline for quantum algorithm discovery via ZX-diagram pattern mining and recombination. The pipeline is intentionally "vanilla" — no premature optimization, no clever heuristics. It answers the question: **what happens when you mine frequent sub-patterns from known quantum algorithms' ZX-diagrams, recombine them, and check what you get?**

## Pipeline

| Stage | What it does |
|-------|-------------|
| 1. Corpus | 13 algorithms across 5 families (oracular, arithmetic, variational, simulation, entanglement), instantiated at multiple qubit counts |
| 2. ZX Convert | Convert to ZX-diagrams via PyZX. `full_reduce` for extraction, `teleport_reduce` for mining |
| 3. Mining | Frequent subgraph mining via gSpan with phase-discretized vertex labels |
| 4. Compose | Sequential (PyZX `compose()`), parallel (`tensor()` + Hadamard stitching), phase perturbation |
| 5. Filter | Circuit extraction with `to_graph_like` → `full_reduce` → gflow check → `extract_circuit` |
| 6. Benchmark | Process fidelity against target unitaries from corpus. Clifford classification, entanglement capacity |
| 7. Report | JSON summary + HTML report |

## v1 Results (small_run.yaml)

```
Corpus:       26 circuits (5 families × 2 qubit counts)
ZX Diagrams:  26 (12 skipped for mining due to size)
Mined Webs:   2,767
Candidates:   500
Survivors:    31 (6.2% extraction rate)
Task Matches: 19 (matched against known algorithms by qubit count)
Real Improvements: 0 (no candidate achieves fidelity ≥0.99 to a known algorithm with fewer gates)
Best Fidelity: 0.12 (to QFT-3q; random baseline for 3 qubits ≈ 0.016)
```

## What v1 Tells Us

1. **The pipeline infrastructure works.** All 7 stages execute correctly end-to-end, 156 tests pass, benchmarking is scientifically rigorous.

2. **Naive recombination does not discover algorithms.** Random composition of mined sub-patterns produces unitaries that are essentially random — fidelity to known algorithms is near the random baseline.

3. **The evaluation is honest.** Process fidelity correctly identifies that no candidate approximates any known algorithm efficiently. The old gate-count-only comparison would have produced false positives.

4. **Some structure is preserved.** Best fidelity (0.12) is 7.5x above the random baseline (0.016), suggesting the mined patterns carry some structural information from the source algorithms.

5. **Multi-qubit compositions work.** 31 survivors include 12 two-qubit and 19 three-qubit circuits. The boundary assignment and PyZX native compose/tensor produce valid extractable diagrams.

## Known Biases & Limitations (from audit)

These are intentional in v1 as a baseline. They represent directions for v2:

**Hard constraints:**
- `_MAX_QUBITS = 10` hard-coded in stitcher
- Phase perturbation restricted to Clifford+T phases (k*pi/4)
- Z-spider-only Hadamard stitching and phase perturbation (X-spiders excluded)
- gflow check as hard rejection before extraction
- `n_inputs == n_outputs` enforced (square unitaries only)

**Default biases:**
- `prefer_cross_family: True` by default (favors cross-family over same-family)
- `optimize_cnots=2` hard-coded in extraction
- `max_cnot_blowup_factor=5.0` premature filtering
- Improvement criterion uses `total_gates` only (not Pareto dominance on T-count/CNOT/depth)
- Benchmark `qubit_counts=[3,4,5]` disconnected from corpus `qubit_counts=[3,5]`

**Information loss:**
- Failed extraction candidates completely discarded (no failure manifest)
- Boundary wire direction assignment is arbitrary (no reversed-orientation variants)
- Phase discretization at 8 bins loses non-Clifford+T phases

**Corpus gaps:**
- Single parameter instance per algorithm (fixed oracle, fixed angles)
- QAOA ring graph only
- Missing families: error correction, quantum walks, linear algebra, communication

## Architecture

```
src/zx_webs/
├── config.py              # Pydantic v2 config, YAML loader
├── types.py               # Shared enums and type aliases
├── persistence.py         # JSON save/load
├── pipeline.py            # Orchestrator + Stage 1 runner
├── stage1_corpus/         # 13 Qiskit algorithm implementations
├── stage2_zx/             # PyZX conversion + simplification
├── stage3_mining/         # gSpan adapter + ZXWeb dataclass
├── stage4_compose/        # Boundary analysis + stitcher (3 strategies)
├── stage5_filter/         # Extraction + hash-based dedup
├── stage6_bench/          # Process fidelity + classification
└── stage7_report/         # JSON + HTML reporting
```

## Dependencies

Core: qiskit, pyzx, gspan-mining, networkx, pydantic, numpy, pandas, matplotlib, tqdm

Optional: cupy-cuda12x (GPU dedup), qiskit-aer (simulation), supermarq (benchmarks), qiskit-ibm-runtime / qiskit-braket-provider / qiskit-addon-cutting (QPU)

## Running

```bash
pip install -e ".[dev]"

# Tests
python -m pytest tests/ -v

# Tiny smoke test (~30s)
python scripts/run_pipeline.py --config configs/tiny_run.yaml

# Small experiment (~5-10 min)
python scripts/run_pipeline.py --config configs/small_run.yaml

# On SLURM
sbatch --partition=notebooks --ntasks=1 --cpus-per-task=8 --mem=64G \
  --output=logs/pipeline_%j.log --error=logs/pipeline_%j.log \
  --wrap="python scripts/run_pipeline.py --config configs/small_run.yaml"
```

## v2 Directions

These are all equally valid next steps — v1's value is as a neutral baseline from which to explore any of them:

- **Scale corpus**: 100+ algorithms from Quantum Algorithm Zoo, multiple parameter instances
- **Broaden search**: Include X-spiders, continuous phases, deeper compositions
- **Guided search**: RL-based rewrite rules, evolutionary composition, simulated annealing
- **Better mining**: Larger corpora, semantic pattern filtering, teleport_reduce only
- **Hardware**: Real QPU execution via Qiskit Runtime / Amazon Braket
- **Optimization focus**: Target-specific circuit optimization (T-count reduction, etc.)
