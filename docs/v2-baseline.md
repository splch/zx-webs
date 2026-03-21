# ZX-Webs v2: Debiased Baseline

## Overview

v2 removes all identified biases from v1 and expands the algorithm corpus from 13 to 47 algorithms across 8 families. The pipeline remains exploratory and unguided — no heuristics favor any particular outcome. The key question shifts from "does this work at all?" (v1: barely) to "how close can unbiased recombination get to known algorithms?"

## What Changed from v1

### Corpus expansion
- 13 → 47 algorithms across 5 → 8 families
- New families: error correction, linear algebra, communication
- Multiple parameter variants per algorithm (different oracles, coupling strengths, graph topologies)
- 134 corpus circuits (was 26)

### Bias removal (23 findings from audit)
- **Boundary direction encoding**: Input and output boundaries now have distinct gSpan labels, so mined patterns preserve which wires are inputs vs outputs. This fixed the qubit collapse (v1: 100% 2-qubit survivors → v2: 2, 3, and 4-qubit survivors).
- **Balanced web sampling**: All boundary-size groups represented proportionally when sampling from large web sets. Rare groups (like 3-input/3-output) fully included.
- **Balanced composition budgets**: Sequential, parallel, parallel+stitch, and phase perturbation each get equal allocation instead of sequential monopolizing the candidate cap.
- **X-spider inclusion**: Hadamard stitching and phase perturbation now operate on both Z and X spiders (was Z-only).
- **Configurable everything**: Phase palette resolution, perturbation rate, CNOT blowup threshold, gflow precheck, extraction optimization level, unitary qubit limits — all configurable with neutral defaults.
- **No hard rejections**: gflow precheck off by default, CNOT blowup configurable.
- **Pareto dominance**: Improvement criterion uses Pareto dominance on (T-count, CNOT-count, depth) instead of total gates only.
- **Dead code removed**: `boundary_match_strategy` config removed, `BENCHMARK_TASKS` dict replaced with functional `BenchmarkTask` class.

### Functional evaluation
- Process fidelity: |Tr(U_target† @ U_candidate)|² / d² against all corpus algorithms at matching qubit count
- Circuit classification: Clifford detection, entanglement capacity estimation
- Hash-based O(n) deduplication (was O(n²) pairwise)
- Failures manifest preserved for analysis

### Infrastructure
- tqdm progress bars on all stages including gSpan mining
- Multiprocessing for candidate extraction (16 workers)
- Web sampling to prevent OOM on large mining runs
- GPU-accelerated deduplication via CuPy (optional)

## v2 Results (small_run.yaml)

```
Corpus:          134 circuits (47 algorithms × ~3 variants × 2 qubit counts)
ZX Diagrams:     134 (87 included for mining, 47 too large)
Mined Webs:      151,434 (75,717 patterns × 2 reversed boundaries)
Candidates:      2,000
Survivors:       445 (22.3% extraction rate)
  2-qubit:       369
  3-qubit:        58
  4-qubit:        18
Benchmark Tasks: 108
Best Fidelity:   0.89 to inverse QFT-2q
Mean Fidelity:   0.22
Cross-Family:    68% of candidates
Real Improvements: 0 (none at fidelity ≥ 0.99 with Pareto-dominant gates)
Search Coverage: 0.00002% of composition space (2,000 / 11.5 billion pairs)
```

## What v2 Tells Us

1. **Debiasing dramatically improved results.** Best fidelity went from 0.12 (v1) to 0.89 (v2). Mean fidelity from 0.05 to 0.22. The pipeline is finding circuits that substantially overlap with known algorithms.

2. **0.89 fidelity from 0.00002% search coverage is remarkable.** A candidate with 4 gates achieves 89% process fidelity to the 2-qubit inverse QFT. At this search coverage, either we're very lucky or there's genuine structure in the ZX-Web composition space that the pipeline is exploiting.

3. **Multi-qubit compositions work.** Boundary direction encoding and balanced sampling produce 2, 3, and 4-qubit survivors. The qubit collapse from v1 is substantially fixed.

4. **The expanded corpus enables richer cross-family recombination.** 68% of candidates combine patterns from different algorithm families (error correction + simulation, arithmetic + communication, etc.).

5. **No real discoveries yet.** The 0.99 fidelity threshold has not been crossed. The 0.89 result is close but the candidate circuit (4 gates, 1 T-gate) doesn't Pareto-dominate the baseline inverse QFT.

## Algorithm Corpus

| Family | Count | Algorithms |
|---|---|---|
| Arithmetic | 7 | QFT, inverse QFT, QPE, Draper adder, ripple adder, multiplier, comparator |
| Communication | 6 | Teleportation, superdense coding, entanglement swapping, GHZ distribution, secret sharing, BB84 |
| Entanglement | 6 | GHZ, W-state, Bell state, cluster state, graph state, Dicke state |
| Error Correction | 6 | Bit-flip, phase-flip, Shor [[9,1,3]], Steane [[7,1,3]], repetition, surface code |
| Linear Algebra | 5 | Swap test, Hadamard test, inner product, HHL rotation, quantum walk |
| Oracular | 6 | Deutsch, Deutsch-Jozsa, Bernstein-Vazirani, Grover, Simon, quantum counting |
| Simulation | 6 | Trotter-Ising, Suzuki-Trotter, Heisenberg, XY, Hubbard, Hamiltonian sim |
| Variational | 5 | QAOA MaxCut, QAOA SK model, VQE hardware-efficient, VQE UCCSD, excitation-preserving |

## Progression

| Metric | v1.0 | v2.0 |
|---|---|---|
| Algorithms | 13 (5 families) | 47 (8 families) |
| Corpus circuits | 26 | 134 |
| Mined webs | 2,767 | 151,434 |
| Survivors | 31 | 445 |
| Best fidelity | 0.12 | 0.89 |
| Mean fidelity | 0.05 | 0.22 |
| Qubit diversity | 2q only (collapsed) | 2q, 3q, 4q |
| Evaluation | Gate counts | Process fidelity + Pareto |
| Biases | 23 found | All removed |
| Tests | 156 | 170 |

## v3 Directions

- **Scale candidates**: 10K-100K candidates to cover more of the 11.5 billion pair space
- **Scale corpus**: More algorithms, more qubit counts, more parameter variants
- **Larger qubit compositions**: Current patterns max at 10 vertices; larger patterns could enable 5+-qubit compositions
- **Guided search**: Use fidelity signal to steer composition (RL, evolutionary, simulated annealing)
- **Alternative mining**: Discriminative patterns (unique to families) vs frequent patterns (common across families)
- **QPU validation**: Run top candidates on real quantum hardware
