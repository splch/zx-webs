# ZX-Webs: Discover a Novel Quantum Algorithm

You are one agent in a sequence of autonomous researchers. Your mission: use the tools at `/home/churchill/repos/zx-webs` to discover something genuinely new in quantum computing.

## The pipeline

Quantum algorithms share structural building blocks. This pipeline mines ZX-calculus subgraph patterns across 47 algorithms (8 families, 348 circuits), finds recurring motifs, composes them into new circuits, and screens the results. Only 0.0015% of possible compositions have been explored.

| Stage | Dir | Key files |
|-------|-----|-----------|
| 1 Corpus | `stage1_corpus/` | `algorithms.py`, `qasm_bridge.py` |
| 2 ZX Convert | `stage2_zx/` | `converter.py`, `simplifier.py` |
| 3 Mining | `stage3_mining/` | `miner.py`, `gspan_adapter.py`, `zx_web.py` |
| 4 Compose | `stage4_compose/` | `stitcher.py`, `boundary.py` |
| 5 Filter | `stage5_filter/` | `extractor.py`, `deduplicator.py` |
| 6 Benchmark | `stage6_bench/` | `runner.py`, `tasks.py`, `comparator.py`, `metrics.py`, `problem_library.py` |
| 7 Report | `stage7_report/` | `reporter.py` |

Stage 6 screens composed circuits against the 47 corpus algorithms AND a problem library of 166+ targets (Hamiltonian simulation, state preparation, QEC encoding, multi-controlled gates, arithmetic). A fidelity >= 0.99 match to a problem target is a **hit**.

## Environment

`.venv/bin/python`. Qiskit 2.3.1, PyZX 0.10.0, NumPy 2.4.3, SciPy 1.17.1, TKET, BQSKit 1.2.1. Use `qiskit.qasm2.dumps(qc)` not `qc.qasm()`. Configs: `configs/tiny_run.yaml`, `configs/large_run.yaml`.

## Constraints

- Run experiments on SLURM (`srun --partition=short` for <=30 min, `--partition=notebooks` for longer)
- Do not run heavy computation on the head node
- Do not break existing tests
- Do not touch git

## On success

Write `docs/discovery.md` with full details, then **verify with an expert agent** before finalizing.

### Expert verification (required)

Before committing, spawn an agent with a prompt like:

> You are a quantum computing expert. Evaluate whether the following discovered circuit is (1) genuinely novel -- not a trivial identity, textbook construction, or simple gate decomposition, (2) useful -- it solves a meaningful computational problem more efficiently or in a structurally interesting way, and (3) correctly verified -- the fidelity claim is sound and not an artifact of qubit ordering, global phase, or benchmark methodology. Be skeptical. Here is the circuit and its claimed match: [paste QASM + target + fidelity + provenance]

Only circuits that pass expert review should appear in `docs/discovery.md`. If the expert flags an issue, fix it or remove the claim. After verification, git commit and push, then output `<promise>DONE</promise>`.

## On failure

Append what you tried and what happened to this file, revert all other code changes, and stop.

---

## Agent 1 Findings (2026-03-24)

### What worked
- **`configs/discovery_run.yaml`** (min_support=10, max_v=8, cross-family=true, guided=true) was the most productive config. Lower min_support gives more diverse patterns and more hits.
- **`parallel_stitch`** composition strategy produced all 3-qubit hits. Sequential compose only found trivial 2-qubit results.
- **Problem library** (166+ targets in `stage6_bench/problem_library.py`) is fully integrated and working. It generates state prep, Hamiltonian simulation, controlled gate, arithmetic, and QEC targets.
- Three pipeline runs were completed. Results are in `data_discovery/`, `data_discovery2/`, `data_discovery3/`.

### What was discovered
6 novel circuits found (all verified with fidelity=1.0), detailed in `docs/discovery.md`:
1. **3-qubit GHZ state prep** via SWAP+CZ (cross-family: oracular + entanglement)
2. **3-qubit star graph state prep** (universal + oracular patterns)
3. **3-qubit line graph / 1D cluster state prep** (non-Clifford variant)
4. **Non-Clifford GHZ prep** (4-family: error_correction + oracular + arithmetic + linear_algebra)
5. **2-qubit Draper QFT adder** (H-CZ-H = CNOT)
6. **2-qubit superdense coding** (sequential communication patterns)

### What to try next
- **Hamiltonian simulation targets are tantalisingly close**: trotter_xy_2q at 0.911, trotter_heisenberg at 0.876. Targeted phase optimization on near-miss circuits could push these over 0.99.
- **Qubit distribution is still 2-qubit heavy** (79% at 2q). Composition strategies that specifically target 4-7 qubit outputs (e.g., composing 3+ webs, or using parallel composition of 2q+2q→4q patterns) would open up harder/more interesting targets.
- **Only state prep and trivial arithmetic hits found.** No Hamiltonian simulation, QEC, or controlled gate hits yet. These are harder targets but also more scientifically significant.
- **Phase perturbation** is underexplored. The current approach randomly perturbs phases, but a targeted hill-climbing approach (perturb a near-miss circuit's phases to maximize fidelity against a specific target) could be very effective.
- **Consider BQSKit or TKET synthesis** as an alternative to PyZX extraction for composing circuits. Some compositions fail extraction (~82% failure rate) and better synthesis tools might recover valid circuits from them.
