# ZXEA: ZX-Irreducible Entangling Ansatz — Benchmark Report

## Hypothesis

Combine ZX-irreducible entangling layers (cluster_chain, hadamard_sandwich
from TVH analysis) with per-qubit variational parameters (from HEA) to achieve
both high expressibility AND trainability.

TVH achieved near-Haar expressibility from just 2 parameters but failed at VQE
(48% error) due to flat energy landscapes. HEA with 24 per-qubit parameters
achieves 3.6% error. ZXEA should combine the best of both.

## Ansatz Summary

| Ansatz | Params (4q, 2L) | Entangling | Key Feature |
|--------|-----------------|------------|-------------|
| ZXEA | 24 | 6 2q-gates | ZX-irreducible cluster_chain entangling |
| ZXEA-H | 24 | 6 2q-gates | ZXEA + hadamard_sandwich (H-S-H) |
| HEA | 24 | 6 2q-gates | Standard CX-chain entangling |
| QAOA-flex | 4 | 12 2q-gates | Per-layer gamma/beta ZZ+mixer |
| TVH | 2 | 18 2q-gates | Original 2-param global TVH |

## Gate Counts

| Ansatz | Total Gates | 2-Qubit Gates | Depth |
|--------|-------------|---------------|-------|
| ZXEA | 46 | 6 | 15 |
| ZXEA-H | 70 | 6 | 21 |
| HEA | 30 | 6 | 11 |
| QAOA-flex | 30 | 12 | 18 |
| TVH | 60 | 18 | 31 |

## VQE Results

### 4q_Heisenberg
Exact ground-state energy: **-6.4641**

| Ansatz | Best Energy | Error (%) | Mean Energy | Std |
|--------|-------------|-----------|-------------|-----|
| ZXEA | -6.3273 | 2.1% | -6.2061 | 0.3383 |
| ZXEA-H | -6.3273 | 2.1% | -6.2339 | 0.1907 |
| HEA | -6.2347 | 3.5% | -6.0641 | 0.3787 |
| QAOA-flex | -3.4766 | 46.2% | -3.1893 | 0.6239 |
| TVH | -3.3556 | 48.1% | -2.0461 | 1.1161 |

### 4q_TFIM
Exact ground-state energy: **-4.7588**

| Ansatz | Best Energy | Error (%) | Mean Energy | Std |
|--------|-------------|-----------|-------------|-----|
| ZXEA | -4.7449 | 0.3% | -4.6535 | 0.0834 |
| ZXEA-H | -4.6812 | 1.6% | -4.5992 | 0.0449 |
| HEA | -4.7555 | 0.1% | -4.5930 | 0.1226 |
| QAOA-flex | -4.7353 | 0.5% | -4.5887 | 0.7046 |
| TVH | -3.0000 | 37.0% | -2.3006 | 0.7551 |

### 6q_Heisenberg
Exact ground-state energy: **-9.9743**

| Ansatz | Best Energy | Error (%) | Mean Energy | Std |
|--------|-------------|-----------|-------------|-----|
| ZXEA | -8.5013 | 14.8% | -8.2666 | 0.1557 |
| ZXEA-H | -8.5885 | 13.9% | -8.3024 | 0.2438 |
| HEA | -9.3133 | 6.6% | -8.6398 | 0.7779 |
| QAOA-flex | -3.6881 | 63.0% | -3.1967 | 0.7634 |
| TVH | -5.1306 | 48.6% | -2.8340 | 1.8707 |

## Expressibility

Haar random reference (4 qubits): **0.0588**

| Ansatz | Mean Fidelity | Ratio to Haar | Interpretation |
|--------|---------------|---------------|----------------|
| ZXEA | 0.0631 | 1.07x | Near-Haar (excellent) |
| ZXEA-H | 0.0599 | 1.02x | Near-Haar (excellent) |
| HEA | 0.0621 | 1.06x | Near-Haar (excellent) |
| QAOA-flex | 0.2214 | 3.76x | Moderate |
| TVH | 0.1175 | 2.00x | Good |

## Gradient Variance (Trainability)

Higher variance = more trainable (further from barren plateau).

| Ansatz | Mean Var(∂E/∂θ) | Max Var | Min Var |
|--------|-----------------|---------|---------|
| ZXEA | 0.273874 | 0.446751 | 0.149915 |
| ZXEA-H | 0.257607 | 0.366671 | 0.141136 |
| HEA | 0.251718 | 0.387929 | 0.075272 |
| QAOA-flex | 4.579130 | 12.656593 | 0.000000 |
| TVH | 8.317705 | 15.568166 | 1.067244 |

## Noise Resilience

Energy at optimal noiseless parameters under depolarising noise.

| Ansatz | p=0.001 | p=0.005 | p=0.01 |
|--------| ------ | ------ | ------ |
| ZXEA | -6.0427 | -5.0243 | -3.9851 |
| ZXEA-H | -5.8993 | -4.4548 | -3.1310 |
| HEA | -6.0503 | -5.3642 | -4.6118 |
| QAOA-flex | -3.3738 | -2.9912 | -2.5716 |
| TVH | -3.1601 | -2.4840 | -1.8360 |

## Convergence Speed

Best energy found at selected function evaluation counts:

| Ansatz | @50 evals | @100 evals | @200 evals | @400 evals | @800 evals |
|--------| ------ | ------ | ------ | ------ | ------ |
| ZXEA | -4.3395 | -5.9040 | -6.2921 | -6.3267 | -6.3273 |
| ZXEA-H | -4.5136 | -4.7607 | -5.8143 | -6.3086 | -6.3261 |
| HEA | -3.6760 | -5.1466 | -6.0938 | -6.2038 | -6.2337 |
| QAOA-flex | -3.4591 | -3.4711 | -3.4759 | -3.4766 | -3.4766 |
| TVH | -3.3556 | -3.3556 | -3.3556 | -3.3556 | -3.3556 |

## Gate Efficiency

| Ansatz | Error/Gate | Error/2q-Gate | Best Error (%) |
|--------|-----------|---------------|----------------|
| ZXEA | 0.0030 | 0.0228 | 2.1% |
| ZXEA-H | 0.0020 | 0.0228 | 2.1% |
| HEA | 0.0076 | 0.0382 | 3.5% |
| QAOA-flex | 0.0996 | 0.2490 | 46.2% |
| TVH | 0.0518 | 0.1727 | 48.1% |

## Conclusions

1. **Best VQE performer (4q Heisenberg)**: ZXEA-H with 2.1% error
2. **Most expressive**: ZXEA-H (mean fidelity 0.0599)
3. **Most trainable**: TVH (mean grad var 8.317705)

### ZXEA Assessment

- ZXEA VQE error: 2.1% vs HEA: 3.5% vs TVH: 48.1%
- ZXEA expressibility: 0.0631 vs HEA: 0.0621 (lower = more expressive)
- ZXEA trainability: 0.273874 vs HEA: 0.251718

### Hypothesis Verdict

The results above show whether ZX-irreducible entangling layers combined
with per-qubit parameters successfully bridge the expressibility-trainability
gap between TVH and HEA.

---
*Generated by benchmark_zxea.py*