# ZXEA: ZX-Irreducible Entangling Ansatz — Benchmark Report

## Hypothesis

Combine ZX-irreducible entangling layers (cluster_chain from TVH analysis)
with per-qubit variational parameters (from HEA) to achieve both high
expressibility AND trainability.

TVH achieved near-Haar expressibility from just 2 parameters but failed at VQE
(48% error) due to flat energy landscapes. HEA with 24 per-qubit parameters
achieves ~3.5% error. ZXEA should combine the best of both.

Three entangling topologies are tested to assess whether the chain topology
(which creates nearest-neighbour graph-state entanglement) limits scaling:
- **ZXEA**: linear CZ chain (original cluster_chain motif)
- **ZXEA-grid**: 2D grid CZ pattern (adds cross-row connectivity)
- **ZXEA-alt**: alternating even/odd CZ pairs (brick-layer pattern)

## Ansatz Summary

| Ansatz | Params (4q, 2L) | Entangling | Key Feature |
|--------|-----------------|------------|-------------|
| ZXEA | 24 | 6 2q-gates | CZ chain entangling (cluster_chain) |
| ZXEA-grid | 24 | 8 2q-gates | 2D grid CZ entangling |
| ZXEA-alt | 24 | 3 2q-gates | Alternating (brick-layer) CZ entangling |
| HEA | 24 | 6 2q-gates | Standard CX-chain entangling |
| QAOA-flex | 4 | 12 2q-gates | Per-layer gamma/beta ZZ+mixer |
| TVH | 2 | 18 2q-gates | Original 2-param global TVH |

## Gate Counts

| Ansatz | Total Gates | 2-Qubit Gates | Depth |
|--------|-------------|---------------|-------|
| ZXEA | 46 | 6 | 15 |
| ZXEA-grid | 48 | 8 | 14 |
| ZXEA-alt | 43 | 3 | 12 |
| HEA | 30 | 6 | 11 |
| QAOA-flex | 30 | 12 | 18 |
| TVH | 60 | 18 | 31 |

## VQE Results

### 4q_Heisenberg
Exact ground-state energy: **-6.4641**

| Ansatz | Best Energy | Error (%) | Mean Energy | Std |
|--------|-------------|-----------|-------------|-----|
| ZXEA | -6.3273 | 2.1% | -6.2061 | 0.3383 |
| ZXEA-grid | -5.0414 | 22.0% | -4.5961 | 0.3870 |
| ZXEA-alt | -4.7167 | 27.0% | -4.7058 | 0.0261 |
| HEA | -6.2347 | 3.5% | -6.1343 | 0.2284 |
| QAOA-flex | -3.4766 | 46.2% | -3.1482 | 0.6568 |
| TVH | -3.3556 | 48.1% | -2.1569 | 0.9586 |

### 4q_TFIM
Exact ground-state energy: **-4.7588**

| Ansatz | Best Energy | Error (%) | Mean Energy | Std |
|--------|-------------|-----------|-------------|-----|
| ZXEA | -4.7430 | 0.3% | -4.6524 | 0.0833 |
| ZXEA-grid | -4.6089 | 3.1% | -4.4888 | 0.0401 |
| ZXEA-alt | -4.5618 | 4.1% | -4.5439 | 0.0203 |
| HEA | -4.7570 | 0.0% | -4.6097 | 0.1185 |
| QAOA-flex | -4.7353 | 0.5% | -4.5917 | 0.7052 |
| TVH | -3.0000 | 37.0% | -2.1153 | 0.8407 |

### 6q_Heisenberg
Exact ground-state energy: **-9.9743**

| Ansatz | Best Energy | Error (%) | Mean Energy | Std |
|--------|-------------|-----------|-------------|-----|
| ZXEA | -8.4770 | 15.0% | -8.2494 | 0.1368 |
| ZXEA-grid | -7.5793 | 24.0% | -7.0094 | 0.4474 |
| ZXEA-alt | -8.2807 | 17.0% | -8.1781 | 0.1898 |
| HEA | -9.3600 | 6.2% | -8.9634 | 0.5488 |
| QAOA-flex | -3.6881 | 63.0% | -3.1230 | 0.7623 |
| TVH | -5.1306 | 48.6% | -2.7017 | 2.1712 |

### 8q_Heisenberg
Exact ground-state energy: **-13.4997**

| Ansatz | Best Energy | Error (%) | Mean Energy | Std |
|--------|-------------|-----------|-------------|-----|
| ZXEA | -11.6644 | 13.6% | -11.2633 | 0.3285 |
| ZXEA-grid | -9.9313 | 26.4% | -9.4322 | 0.3519 |
| ZXEA-alt | -11.1987 | 17.0% | -11.1384 | 0.0574 |
| HEA | -12.5860 | 6.8% | -11.5233 | 0.8239 |
| QAOA-flex | -4.7137 | 65.1% | -3.8787 | 1.0260 |
| TVH | -7.0862 | 47.5% | -4.0974 | 2.7963 |

## Expressibility

Haar random reference (4 qubits): **0.0588**

| Ansatz | Mean Fidelity | Ratio to Haar | Interpretation |
|--------|---------------|---------------|----------------|
| ZXEA | 0.0624 | 1.06x | Near-Haar (excellent) |
| ZXEA-grid | 0.0615 | 1.04x | Near-Haar (excellent) |
| ZXEA-alt | 0.0636 | 1.08x | Near-Haar (excellent) |
| HEA | 0.0619 | 1.05x | Near-Haar (excellent) |
| QAOA-flex | 0.2311 | 3.93x | Moderate |
| TVH | 0.1147 | 1.95x | Good |

## Gradient Variance (Trainability)

Higher variance = more trainable (further from barren plateau).

| Ansatz | Mean Var(∂E/∂θ) | Max Var | Min Var |
|--------|-----------------|---------|---------|
| ZXEA | 0.285739 | 0.429126 | 0.141891 |
| ZXEA-grid | 0.207660 | 0.337025 | 0.088642 |
| ZXEA-alt | 0.263679 | 0.474033 | 0.122166 |
| HEA | 0.243412 | 0.397277 | 0.072706 |
| QAOA-flex | 4.756816 | 13.167888 | 0.000000 |
| TVH | 9.373302 | 17.363164 | 1.383439 |

## Noise Resilience

Energy at optimal noiseless parameters under depolarising noise.

| Ansatz | p=0.001 | p=0.005 | p=0.01 |
|--------| ------ | ------ | ------ |
| ZXEA | -6.0427 | -5.0243 | -3.9851 |
| ZXEA-grid | -4.8050 | -3.9633 | -3.1120 |
| ZXEA-alt | -4.5181 | -3.8022 | -3.0616 |
| HEA | -6.0503 | -5.3642 | -4.6118 |
| QAOA-flex | -3.3738 | -2.9912 | -2.5716 |
| TVH | -3.1601 | -2.4840 | -1.8360 |

## Convergence Speed

Best energy found at selected function evaluation counts:

| Ansatz | @50 evals | @100 evals | @200 evals | @400 evals | @800 evals |
|--------| ------ | ------ | ------ | ------ | ------ |
| ZXEA | -4.9383 | -5.7294 | -6.2670 | -6.3092 | -6.3270 |
| ZXEA-grid | -3.5235 | -4.1450 | -4.5635 | -4.7833 | -5.0193 |
| ZXEA-alt | -4.4607 | -4.6614 | -4.7133 | -4.7166 | -4.7167 |
| HEA | -5.1921 | -5.8940 | -6.2199 | -6.2322 | -6.2346 |
| QAOA-flex | -3.3638 | -3.4611 | -3.4729 | -3.4764 | -3.4766 |
| TVH | -3.3556 | -3.3556 | -3.3556 | -3.3556 | -3.3556 |

## Gate Efficiency

| Ansatz | Error/Gate | Error/2q-Gate | Best Error (%) |
|--------|-----------|---------------|----------------|
| ZXEA | 0.0030 | 0.0228 | 2.1% |
| ZXEA-grid | 0.0296 | 0.1778 | 22.0% |
| ZXEA-alt | 0.0406 | 0.5825 | 27.0% |
| HEA | 0.0076 | 0.0382 | 3.5% |
| QAOA-flex | 0.0996 | 0.2490 | 46.2% |
| TVH | 0.0518 | 0.1727 | 48.1% |

## Scaling Analysis

Error (%) on Heisenberg chain across qubit counts:

| Ansatz | 4q | 6q | 8q | Trend |
|--------| ------ | ------ | ------ | ----- |
| ZXEA | 2.1% | 15.0% | 13.6% | degrading fast |
| ZXEA-grid | 22.0% | 24.0% | 26.4% | stable |
| ZXEA-alt | 27.0% | 17.0% | 17.0% | improving |
| HEA | 3.5% | 6.2% | 6.8% | degrading |
| QAOA-flex | 46.2% | 63.0% | 65.1% | stable |
| TVH | 48.1% | 48.6% | 47.5% | stable |

## Conclusions

### What Worked

- **ZXEA beats HEA at 4 qubits**: 2.1% vs 3.5% error on Heisenberg model
- All ZXEA variants dramatically outperform TVH (48.1% error), confirming that per-qubit parameters fix TVH's trainability problem

The ZX motif phylogeny pipeline produced an actionable design principle:
use cluster_chain as an entangling primitive. This yields a measurable
improvement at 4 qubits on the Heisenberg model.

### What Didn't Work

- **ZXEA loses to HEA at 6 qubits**: 15.0% vs 6.2%
- **ZXEA loses to HEA at 8 qubits**: 13.6% vs 6.8%
- **ZXEA-grid loses to HEA at 6 qubits**: 24.0% vs 6.2%
- **ZXEA-grid loses to HEA at 8 qubits**: 26.4% vs 6.8%
- **ZXEA-alt loses to HEA at 6 qubits**: 17.0% vs 6.2%
- **ZXEA-alt loses to HEA at 8 qubits**: 17.0% vs 6.8%

**Topology comparison at 8 qubits**: best = ZXEA (13.6%), worst = ZXEA-grid (26.4%)
The 12.8pp spread between topologies suggests connectivity
matters and further topology exploration is warranted.

### Honest Assessment

The cluster_chain entangling layer is a genuine insight from the ZX motif
analysis, and it works at 4 qubits. Whether it generalises to larger systems
is an open question — the 6- and 8-qubit data suggest it may not without
architectural modifications beyond simple topology changes.

The core limitation: graph-state entanglement (H-CZ-H) creates a specific
correlation structure that may not match the entanglement pattern needed for
larger Heisenberg ground states. CX-chain entangling (HEA) may be more
naturally suited to nearest-neighbour spin Hamiltonians because CNOT directly
creates the Bell-type correlations these ground states require.

---
*Generated by benchmark_zxea.py*