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

Agent 1 ran 3 pipeline runs and found 6 circuits matching problem library targets with fidelity=1.0 (GHZ prep, star graph prep, line graph prep, etc.). **All were rejected by expert review** -- every circuit used MORE gates than the textbook implementation. The `_NO_BASELINE = 999999` sentinel caused false positives.

## Agent 2 Findings (2026-03-24)

### Code fix: real baselines
Fixed `problem_library.py` to compute real baseline metrics from reference circuits instead of 999999. State prep baselines from textbook circuits (H+CNOT for GHZ, H+CZ for graph states). Controlled gates, arithmetic, QEC baselines from Qiskit's standard decompositions. Hamiltonian baselines from analytical Trotter formulas.

### Pipeline runs (with real baselines)
- **8 total pipeline runs** across both agents (3 from Agent 1, 5 from Agent 2)
- Configs tested: min_support=5/10/20, max_v=6/8/16, seeds=42/137/271828/31415/27182
- Phase perturbation rates: 0.3 - 0.7, resolutions: 8 - 16
- Cross-family + guided composition enabled
- ~140,000 unique survivors benchmarked against ~200 targets (corpus + problem library)

### Results
- **0 genuine improvements** -- no composed circuit beats any textbook implementation on any metric while maintaining fidelity >= 0.99
- **22-24 fidelity=1.0 matches per run** -- all against trivial targets (GHZ, Bell state, graph states, Deutsch) and all using MORE gates than textbook
- **Best Hamiltonian near-miss: 0.90 fidelity** (TFIM 3q at t=0.1) -- but the matching circuit was a trivial single-qubit rotation close to identity, not a real Hamiltonian simulation
- **Best Trotter near-miss: 0.88** (Heisenberg 2q) -- still far from 0.99

### Root cause analysis
1. **Compositions are structurally bloated.** Stitching two ZX patterns produces circuits with redundant SWAP+H+CZ patterns that PyZX extraction doesn't simplify away. The extraction adds overhead that textbook circuits don't have.
2. **80% of survivors are 2-qubit.** Mining with min_support ≥ 5 produces small, universal patterns. Their pairwise compositions are small circuits that can only match trivial targets.
3. **Phase perturbation is random.** Randomly perturbing phases creates random unitaries, not targeted approximations to Hamiltonian evolution or other structured targets.
4. **Near-identity fidelity inflation.** At small Hamiltonian simulation times (t=0.1), even trivial circuits achieve 0.85+ fidelity because both the target and candidate are close to identity.

### Variational optimization attempt
Tried differential_evolution on 200 3-qubit compositions, optimizing interior spider phases to maximize fidelity against Hamiltonian + controlled gate targets. Best result: **0.32 fidelity** (vs 0.90 from random search). The composed ZX graph structures are too rigid -- tuning phases alone cannot reach the target unitaries. The topology (which spiders are connected, Z vs X types, edge types) constrains the reachable unitary space too much.

### What to try next (for future agents)
1. **Topology mutation, not just phase tuning.** The limitation is structural, not parametric. Future agents should mutate the ZX graph topology (add/remove edges, change spider types, insert/remove spiders) alongside phase optimization. This requires extending the Stitcher with topology-aware operators.
2. **Multi-pattern composition at scale.** The `triple_sequential` strategy only tries 200 webs. Scaling to 3-4 pattern compositions with thousands of webs could produce more complex, higher-qubit circuits that span a richer unitary space.
3. **Better circuit synthesis.** Replace PyZX `extract_circuit` with BQSKit's `QSearchSynthesisPass` or TKET's `FullPeepholeOptimise`. The ~82% extraction failure rate is a major bottleneck. Many valid ZX diagrams fail extraction, and better synthesis tools could recover them.
4. **Evolutionary/MCTS compositional search.** Instead of composing random pairs, use a genetic algorithm or Monte Carlo tree search to evolve compositions toward specific target unitaries. Fitness = fidelity against target. Mutations = add/remove webs, change stitching strategy, perturb phases.
5. **Expand the corpus.** 47 algorithms may not be enough. Adding algorithms from the Quantum Algorithm Zoo (200+ algorithms) would provide more diverse patterns.
6. **Focus on T-count reduction.** ZX full_reduce can reduce T-count (e.g., QFT-3: 8→7 T-gates, QFT-4: 15→12). This is a known ZX-calculus strength but hasn't been systematically applied. A pipeline stage that applies ZX T-count optimization to ALL corpus circuits and reports improvements would produce verifiable results.
