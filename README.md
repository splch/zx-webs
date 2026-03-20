# ZX-Webs

Quantum algorithm discovery via ZX-diagram pattern mining and recombination.

ZX-Webs implements a 7-stage pipeline that mines frequent sub-structures ("ZX-Webs") from simplified ZX-diagrams of known quantum algorithms, recombines them into novel candidate algorithms, and benchmarks the results.

## Pipeline

| Stage | Description |
|-------|-------------|
| 1. Corpus | Build a corpus of quantum algorithms as Qiskit circuits (13 algorithms, 5 families) |
| 2. ZX Convert | Convert to ZX-diagrams via PyZX and simplify with `full_reduce`/`teleport_reduce` |
| 3. Mining | Mine frequent sub-diagrams using gSpan with phase-discretized vertex labels |
| 4. Compose | Combinatorial stitching of ZX-Webs via sequential and parallel composition |
| 5. Filter | Extract circuits from composed ZX-diagrams, reject failures and CNOT blowups |
| 6. Benchmark | Compare survivors against baselines using circuit metrics and SupermarQ features |
| 7. Report | Generate JSON summary and HTML report |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
make test

# Run the pipeline (tiny smoke test)
make run-small
# or with a specific config:
python scripts/run_pipeline.py --config configs/tiny_run.yaml
```

## Configuration

Pipeline parameters are defined in YAML config files under `configs/`:

- `tiny_run.yaml` - Fast smoke test (~30s, entanglement circuits only)
- `small_run.yaml` - Small experiment (oracular + entanglement, 3-5 qubits)
- `default.yaml` - Full experiment (all families, up to 10 qubits)

Key tunable parameters:

| Parameter | Description |
|-----------|-------------|
| `mining.min_support` | Minimum graphs containing a subgraph to be "frequent" |
| `mining.phase_discretization` | Phase bins (N in k*pi/N), default 8 (T-gate resolution) |
| `compose.max_candidates` | Hard cap on generated candidates |
| `filter.extract_timeout_seconds` | Per-extraction time limit |

## Algorithm Families

| Family | Algorithms |
|--------|-----------|
| Oracular | Deutsch-Jozsa, Bernstein-Vazirani, Grover, Simon |
| Arithmetic | QFT, QPE, Ripple Adder |
| Variational | QAOA (MaxCut), VQE (hardware-efficient) |
| Simulation | Trotter-Ising, Hamiltonian Sim |
| Entanglement | GHZ, W-state |

## Project Structure

```
src/zx_webs/
  config.py              # Pydantic config models + YAML loader
  types.py               # Shared type aliases and enums
  persistence.py         # JSON save/load utilities
  pipeline.py            # Pipeline orchestrator
  stage1_corpus/         # Qiskit algorithm implementations + QASM bridge
  stage2_zx/             # PyZX conversion + simplification
  stage3_mining/         # gSpan adapter + ZXWeb dataclass
  stage4_compose/        # Boundary analysis + combinatorial stitcher
  stage5_filter/         # Circuit extraction + deduplication
  stage6_bench/          # Metrics + comparator + benchmarking runner
  stage7_report/         # Summary + HTML report generation
```

## Dependencies

Core: Qiskit, PyZX, gspan-mining, NetworkX, Pydantic, NumPy, Pandas, Matplotlib

Optional:
- `pip install -e ".[bench]"` - Adds qiskit-aer and supermarq
- `pip install -e ".[qpu]"` - Adds qiskit-ibm-runtime, qiskit-braket-provider, qiskit-addon-cutting

## References

- [PyZX](https://github.com/zxcalc/pyzx) - ZX-calculus engine
- [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/) - Algorithm taxonomy
- [QASMBench](https://github.com/pnnl/QASMBench) - Quantum benchmarking suite
- [SupermarQ](https://arxiv.org/abs/2202.11045) - Scalable quantum benchmarks (HPCA 2022)
- [gSpan](https://github.com/betterenvi/gSpan) - Frequent subgraph mining
