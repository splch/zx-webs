# Discovery: 5-CNOT Preparation of the Dicke State D(4,2)

## Summary

We find a quantum circuit that prepares the 4-qubit Dicke state D(4,2) using
**5 CNOT gates**, improving on the best published construction of 6 CNOT
(Wang, Cong & De Micheli, arXiv:2401.01009, 2024). Numerical evidence strongly
suggests that 4 CNOT are insufficient, making 5 the likely minimum.

## Background

The Dicke state D(n,k) is the equal superposition of all n-qubit computational
basis states with Hamming weight k:

    D(4,2) = (|0011> + |0101> + |0110> + |1001> + |1010> + |1100>) / sqrt(6)

Dicke states are fundamental in quantum information: they arise in quantum
metrology, entanglement witnesses, secret sharing protocols, and variational
quantum algorithms. Efficient preparation circuits are an active research topic.

### Prior art

| Source | Year | CNOT count for D(4,2) |
|--------|------|-----------------------|
| Bartschi & Eidenbenz (deterministic) | 2019 | ~12 |
| Mukherjee et al. (manual design) | 2020 | ~12 |
| Wang, Cong & De Micheli (exact synthesis) | 2024 | 6 |
| **This work** | 2026 | **5** |

## Circuit

The circuit uses 5 CNOT gates interleaved with Ry single-qubit rotations.
The CNOT topology is (1->3), (2->3), (0->3), (0->1), (3->0) -- a "fan-in
to qubit 3, then redistribute" pattern not found in standard constructions.

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-3.7860919141784164) q[0];
ry(1.570794632179319) q[1];
ry(-1.4088045911871458) q[2];
ry(-2.902013421742535e-08) q[3];
cx q[1],q[3];
ry(-0.9641838512782351) q[0];
ry(2.378005873761298) q[1];
ry(-0.16199149300896754) q[2];
ry(0.05792708363003102) q[3];
cx q[2],q[3];
ry(-2.488226662209728) q[0];
ry(-2.96574889874431) q[1];
ry(3.3381270147265103) q[2];
ry(1.5707947433890304) q[3];
cx q[0],q[3];
ry(0.9553175865992846) q[0];
ry(-0.30964573873511536) q[1];
ry(-2.5195249654442584) q[2];
ry(-3.336490951770009) q[3];
cx q[0],q[1];
ry(3.141593082901989) q[0];
ry(-2.2879010845141936) q[1];
ry(2.476207330246064) q[2];
ry(-2.9466930378372624) q[3];
cx q[3],q[0];
ry(-0.9553169133588916) q[0];
ry(-0.23821285767527342) q[1];
ry(2.6485393663760073) q[2];
ry(2.1862760162858486) q[3];
```

### Metrics

| Metric | This circuit | Problem library baseline | Wang et al. 2024 |
|--------|-------------|-------------------------|------------------|
| CNOT count | 5 | 8 (estimate) | 6 |
| Total gates | 29 | 12 (estimate) | ~18 |
| Depth | 5 | 12 (estimate) | ~8 |
| State fidelity | 0.999999999998 | - | 1.0 |

## Lower bound evidence

Systematic numerical search over random CNOT topologies with L-BFGS-B and
differential evolution optimization:

| k (CNOT count) | Best fidelity | Time (s) | Interpretation |
|----------------|---------------|----------|----------------|
| 2 | 0.6667 (=4/6) | 57 | Insufficient |
| 3 | 0.8333 (=5/6) | 392 | Insufficient |
| 4 | 0.8727 | 689 | Insufficient |
| 5 | 1.0000 | 1102 | Exact |

The k=4 bound was confirmed with:
- 500+ random topologies (out of 20,736 possible)
- 14 hand-picked structured topologies (cascade, star, chain, fan, ladder, etc.)
- Differential evolution global optimizer
- All consistently give max fidelity 0.8727 at k=4

The clean fractional values at k=2,3 (exactly 4/6 and 5/6) suggest deep
mathematical structure, likely related to Schmidt rank constraints across
bipartitions of D(4,2).

**Caveat**: The lower bound is numerical, not a mathematical proof. An
exhaustive enumeration of all 20,736 possible 4-CNOT topologies or an
analytical proof (e.g., via entanglement monotones) would be needed for
a rigorous lower bound.

## Topology analysis

The optimal topology CX(1,3), CX(2,3), CX(0,3), CX(0,1), CX(3,0):

1. First 3 CNOTs fan into q3 from q1, q2, q0 (q3 collects information)
2. CX(0,1) entangles q0 and q1
3. CX(3,0) redistributes from q3 back to q0

This "collect-then-redistribute" pattern was not found among 14 standard
structured topologies and appears to be a novel circuit motif for Dicke
state preparation.

## Method

The circuit was found by numerical optimization over parameterized Ry+CNOT
circuits. D(4,2) has all-real amplitudes, so the Ry+CNOT gate set (which
generates all real unitaries) is sufficient.

For each candidate CNOT count k=2..8:
1. Sample 500 random CNOT topologies (from 12^k possible placements)
2. For each topology, run 6-8 L-BFGS-B optimizations from random starting points
3. Record the best state fidelity |<D(4,2)|U|0000>|^2

The search was run on a SLURM cluster using scipy.optimize.minimize.

## Expert review

An independent expert review (included in the agent log) rated this discovery
as **CONDITIONALLY VALID**:

- Circuit correctness: **VERIFIED** (fidelity to machine precision)
- Lower bound: **STRONGLY SUPPORTED** (numerical, not proven)
- Novelty: **LIKELY NOVEL** (improves on Wang et al. 2024 by 1 CNOT)
- Usefulness: **MARGINAL** (1 CNOT saving; modest NISQ fidelity improvement)
- Methodology: Numerical circuit synthesis (not ZX-calculus pipeline)

## Provenance

- **Agent**: Agent 8 of the ZX-webs discovery pipeline
- **Date**: 2026-03-25
- **Tools**: scipy.optimize.minimize (L-BFGS-B), numpy, PyZX, Qiskit
- **Compute**: SLURM notebooks partition, ~20 minutes for D(4,2) search
- **Code**: `scripts/agent8_fast_dicke.py`
