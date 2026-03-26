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

## Agent 3 Findings (2026-03-24)

### Approach: direct optimization + evolutionary search + qubit ordering bug

Agent 3 tried three strategies, all focused on circuit optimization rather than composition. **None produced a genuine discovery.**

### Strategy 1: Direct ZX/TKET/BQSKit optimization of problem library reference circuits
Systematically applied ZX `full_reduce` + extraction and TKET `FullPeepholeOptimise` to all problem library reference circuits (controlled gates, QFT, QEC, arithmetic). Results:
- **QFT**: ZX reduces T-count (8→7 for 3q, 15→12 for 4q) but increases 2q-count and depth. NOT Pareto-dominant.
- **Controlled gates (Toffoli, CCZ, Fredkin, MC gates)**: ZX extraction makes them uniformly worse (doubles 2q-count). TKET breaks fidelity for non-Clifford circuits after ZX extraction.
- **QEC [[5,1,3]]**: ZX gives 14 gates / 10 2q / depth 9 (vs baseline 15/11/10). ZX+TKET gives 9 2q / depth 7. **Expert rejected** -- this is standard PyZX output, and known encoders with 6-8 2q gates already exist in the literature.
- **QEC [[4,2,2]]**: Already minimal (4 gates, 2 2q). ZX+TKET gives depth 2→1, trivial.
- **Corpus-wide**: 5 Pareto improvements found, all minor (ghz_distribution T 2→0, cluster_state depth -1, graph_state depth -1, quantum_walk 3q collapses to single X gate).

### Strategy 2: Evolutionary ZX topology search
Built `scripts/evolutionary_zx_search.py` implementing ZX graph mutation (phase change, edge toggle, edge type flip, spider type change, spider insert/remove) with tournament selection. Seeded from reference circuits + random circuits. Ran against all problem library targets.
- **State prep targets**: Achieved fidelity 1.0 for GHZ (but no metric improvement) and 0.97 for W state.
- **Non-trivial targets**: Search still running at time of writing, but early results show the evolutionary approach struggles because random ZX mutations rarely produce valid circuits (extraction failure rate ~40-60%).

### Strategy 3: Re-evaluation with qubit ordering fix
Discovered that `problem_library.py` uses Qiskit's little-endian qubit convention for target unitaries (controlled gates, QEC, arithmetic) but candidates are evaluated via PyZX's big-endian convention. **All fidelity scores for these targets in existing benchmark data are wrong.** Re-evaluated 200+ 3+ qubit survivors from data_disc_v2 with corrected convention -- max fidelity 0.56, no improvements. The survivors are simply too small/simple.

### Strategy 4: Improved composition pipeline run
Fixed the qubit ordering bug in `problem_library.py` (replaced `Operator(qc).data` with `_unitary_from_circuit` which uses PyZX convention). Modified the Stitcher:
- `min_compose_qubits=4` to filter out trivial 2-3 qubit compositions
- FPS-based diverse triple pool for `triple_sequential` (300 FPS-selected webs, deterministic)
- `interior_clifford_simp` between triple composition steps to reduce bloat
- `prefer_cross_family=true`, `phase_perturbation_rate=0.5`

Pipeline run (`configs/discovery_v3.yaml`): 348 circuits → 39,607 webs → 36,223 candidates → 3,402 survivors → **0 improvements**. 99.9% of survivors are 4-qubit (only 4 are 5-qubit). Best fidelity: 0.50 for state_prep targets. Compositions are structurally random and don't match any specific target.

### Strategy 5: Target-directed compositional search (`scripts/targeted_composition.py`)
Built evolutionary search over compositions: for each target, population of (web_a, web_b, strategy, perturb) tuples mutated via tournament selection with fidelity as fitness. Uses 13,004 extractable webs (9,852 1-qubit, 3,152 2-qubit). Results:
- **3-qubit controlled gates** (Toffoli, CCZ, Fredkin): best fidelity 0.5625 via parallel 1+2 composition. Plateau — the 1+2 qubit space can't express 3-qubit entangling gates.
- **4-qubit targets** (MC3X, MC3Z): best fidelity 0.39-0.42 via parallel 2+2 composition.
- **Hamiltonian/QFT/QEC targets**: all below 0.2 fidelity.

### Key lesson
**The mining produces patterns that are too small.** At min_support=5, all extractable webs are 1-2 qubits. Even target-directed evolutionary search can't overcome this — parallel composition of 1+2 qubit patterns can only express product unitaries with limited entanglement, nowhere near the structured unitaries needed for Toffoli, QEC encoding, or Hamiltonian simulation.

### Strategy 6: Multi-Hadamard stitching + low-support mining
- **Mining at min_support=2-3** with max_vertices=10-12 **times out** (>600s) on 315 graphs. The search space is too large for submine/gSpan at these parameters.
- **Multi-Hadamard stitching** (1-4 Hadamard edges between parallel components) was tried with 30,000 random trials per target on 16,631 extractable webs. Results: **identical to single-stitch** (max fidelity 0.5625 for Toffoli/CCZ). The Hadamard edges add some phase structure but cannot create the specific entanglement patterns needed for non-trivial gates.
- **Root cause confirmed**: at min_support=5, ALL extractable webs are 1-2 qubits (13,466 one-qubit + 3,165 two-qubit). This is a fundamental limit of the mining approach on this corpus — the 47 algorithms don't share large multi-qubit substructures frequently enough.

**For future agents:**
1. The current gSpan mining approach is exhausted. Lowering min_support causes timeout; current support levels produce only tiny patterns.
2. Consider a completely different mining algorithm (e.g., graph neural network-based pattern detection, or decomposition-based analysis rather than subgraph isomorphism).
3. Alternatively, skip the mining step entirely and build compositions from algorithm-specific circuit blocks (e.g., Trotter steps, controlled rotation cascades) rather than mined patterns.
4. The composition pipeline's code changes (qubit ordering fix, FPS triple pool, intermediate simplification) are ready for future use if a way to produce larger webs is found.

### What to try next (for future agents)

**IMPORTANT: Do NOT try to optimize known circuits.** Applying ZX full_reduce / TKET / BQSKit to textbook circuits is not novel -- it just reproduces standard compiler output. Agent 3 tried this exhaustively and the expert rejected every result. The goal is to discover genuinely NEW circuits through subgraph mining and composition.

#### Direction 1: Fix the mining/composition pipeline to produce larger, richer circuits
The core bottleneck is that 80% of survivors are 2-qubit. The pipeline needs to produce 4-8 qubit compositions that explore non-trivial unitary space.

- **Lower min_support to 2-3** to mine rarer, larger patterns (not just universal small motifs). Accept slower mining.
- **Scale multi-pattern composition.** The `triple_sequential` strategy caps at 200 webs. Extend to 3-4 pattern compositions with thousands of webs. Larger compositions = higher qubit count = richer unitary space.
- **Bias toward cross-family compositions.** Patterns from different algorithm families (e.g., simulation + error_correction, or arithmetic + entanglement) are more likely to produce novel structures than within-family pairs.
- **Add qubit-count filtering in stage 4.** Skip compositions that produce ≤ 3 qubit circuits, since those can only match trivial targets. Target 4-7 qubit compositions specifically.

#### Direction 2: Improve composition strategies to avoid structural bloat
Stitching two ZX patterns currently produces redundant SWAP+H+CZ overhead from PyZX extraction. The composition itself needs to be smarter.

- **Spider fusion during composition.** After stitching, attempt to fuse adjacent same-color spiders at the boundary. This eliminates redundant intermediate spiders.
- **ZX simplification between composition steps.** For multi-pattern compositions (A→B→C), simplify after each step (A→B) before composing with C. This prevents bloat accumulation.
- **Topology mutation in the Stitcher.** Add operators that modify the composed ZX graph: add/remove edges between components, change spider types at boundaries, insert identity-resolution spiders. This creates structural variety beyond what simple sequential/parallel composition can produce.

#### Direction 3: Target-directed compositional search
Instead of composing randomly and screening, guide composition toward specific targets.

- **Evolutionary/MCTS compositional search.** Use a population of compositions, mutate by swapping webs / changing stitching strategy / perturbing topology. Fitness = fidelity against a specific target. This is fundamentally different from optimizing a known circuit -- you're building NEW circuits from structural building blocks.
- **Hamiltonian-informed composition.** For Hamiltonian simulation targets, decompose the target into Pauli terms and search for mined patterns that implement each term's evolution. Compose term-by-term instead of randomly.
- **Subgraph matching.** Given a target unitary, find mined webs whose individual unitaries (when extractable) have high fidelity to sub-blocks of the target. Use these as starting points for composition.

#### Direction 4: Expand the corpus for richer patterns
- **Add algorithms from the Quantum Algorithm Zoo** (200+ algorithms) for more diverse patterns.
- **Include parameterized circuit families** at multiple parameter values to capture how structure varies with parameters.
- **Mine at multiple reduction levels.** Run gSpan on both `full_reduce` and `teleport_reduce` graphs -- different reductions expose different patterns.

#### Dead ends (do not repeat)
- **Direct ZX/TKET/BQSKit optimization of known circuits.** This just runs standard compiler passes. The expert will reject it. ZX T-count reduction on QFT is well-known. TKET CliffordSimp on QEC encoders produces results worse than the literature.
- **Re-evaluating existing survivors.** All 20K+ survivors from previous runs were checked against corrected targets. Max fidelity 0.56 for non-trivial targets. They're too small/simple.
- **Phase-only variational optimization.** Already tried, max 0.32 fidelity. Topology is the constraint, not phases.

#### Known bug: qubit ordering mismatch in problem_library.py
The `problem_library.py` builds target unitaries for controlled gates, QEC, and arithmetic using `Operator(qc).data` (Qiskit little-endian convention), but candidate circuits are evaluated via PyZX's `c.to_matrix()` (big-endian convention). This means **all fidelity scores for these target categories are wrong** in the existing benchmark data. Hamiltonian and state_prep targets are unaffected (they use mathematical convention = PyZX convention). Fix this before running new benchmarks against these targets.

## Agent 4 Findings (2026-03-25)

### Approach: Per-family mining + cross-family circuit-level composition

Agent 4 addressed the fundamental bottleneck identified by Agent 3: global mining at min_support≥5 produces only 1-2 qubit patterns. The core insight was to **mine each algorithm family independently** — with only 26-69 graphs per family (vs 348 globally), lower min_support becomes tractable.

### Per-family mining results

Successfully mined all 8 families with adaptive min_support:

| Family | Graphs | min_support | max_v | Time | Patterns | 3+q patterns |
|--------|--------|-------------|-------|------|----------|-------------|
| arithmetic | 17 | 4 | 8 | 71s | 44,739 | 1,309 |
| communication | 46 | 2 | 12 | 5s | 68,744 | 12,988 |
| entanglement | 38 | 4 | 8 | 24s | 33,686 | 1,297 |
| error_correction | 26 | 2 | 12 | 11s | 75,973 | **20,178** (up to 6q!) |
| linear_algebra | 22 | 3 | 10 | 109s | 521,503 | 8,272 |
| oracular | 30 | 2 | 12 | 3s | 7,514 | 2,463 (up to 6q) |
| simulation | 63 | 8 | 6 | 29s | 25,507 | 73 |
| variational | 52 | 8 | 6 | 1s | 3,927 | 5 |
| **TOTAL** | **348** | — | — | **253s** | **781,593** | **46,585** |

This is a **massive improvement** over global mining: 46,585 patterns with 3+ qubit metadata (vs 0 from global mining at min_support=5). Error_correction produced patterns up to 6 qubits.

### Critical finding: boundary assignment reduces actual qubit count

After `_ensure_proper_boundaries()`, most "3+ qubit" patterns (by metadata) actually become 2-qubit circuits. The function splits boundary vertices by row position, so a pattern with (meta_in=3, meta_out=0) becomes (actual_in=2, actual_out=2) = 2-qubit. Only patterns with meta_in + meta_out ≥ 5 reliably produce genuine 3+ qubit circuits. After boundary correction: 1,643 actual 3+ qubit patterns across all families, of which 232 (209 unique) were extractable.

### Composition strategies tested

1. **ZX graph composition (sequential):** 35,620 cross-family pairs tried → **0 extractable.** The composed ZX graph is never extractable — structural bloat from composition makes `extract_circuit` fail 100%.

2. **ZX graph composition (parallel + Hadamard stitch):** 21,566 pairs → 88 extractable → 0 improvements.

3. **Circuit-level composition + ZX simplification:** Concatenate gates of two circuits, then ZX full_reduce + extract. **34,295 cross-family compositions** tested (209 circuits × all pairs + triples).

### Results (circuit-level composition)

- **1 fidelity=1.0 improvement:** `quantum_walk_3q` matched by `oracular+error_correction` composition (3 gates vs 14 baseline). **Expert rejected:** the 3-qubit quantum walk (2 steps on 4-position line) is degenerate — it equals X on one qubit. Standard ZX full_reduce on the reference circuit produces the same result. This is a benchmark bug, not a discovery.
- **7 other fidelity=1.0 matches:** All to trivial state prep targets (GHZ, graph states, phase flip code), all using MORE gates than baseline.
- **Hamiltonian near-misses:** TFIM 0.97, XY 0.96, Heisenberg 0.94 — all at t=0.1 (near-identity inflation).
- **Controlled gates:** Toffoli/Fredkin/CCZ capped at 0.5625 = (3/4)² fidelity floor.
- **0 genuine improvements** across all strategies.

### Benchmark bug found: `quantum_walk_3q` is degenerate

The `quantum_walk_3q` reference circuit (14 gates with 2 Toffoli gates) implements a discrete-time coined quantum walk that is trivially periodic at this parameterization. After 2 walk steps on a 4-position line, the operation reduces to X on one qubit. This target should be removed or replaced with a non-degenerate parameterization (e.g., 3+ steps or 5+ qubits).

### Root cause analysis

1. **Composition doesn't create novel structure.** Gate-level concatenation + ZX simplification just produces the same result as applying ZX to a larger circuit. The composition step adds no structural novelty beyond what ZX simplification already achieves.
2. **Extractable patterns are too small.** After boundary correction, only 209 unique 3-qubit circuits survive. These are 3-8 gate Clifford circuits — too simple to match complex targets like Toffoli (6 CNOT) or Hamiltonian simulation.
3. **Near-identity inflation persists.** High fidelities at t=0.1 Hamiltonians are artifacts, not discoveries.

### Iteration 2: Variational ansatz search with mined topologies

Used per-family mined circuit topologies (CNOT placement patterns) as parameterized ansatze. Replaced all single-qubit gates with tunable Rz·Ry·Rz rotations, kept CNOT topology fixed. Optimized parameters via differential_evolution against 20 priority targets (controlled gates, Hamiltonians).

- **110 topologies** tested (4-8 CNOT, from 6 algorithm families)
- **~2,200 optimizations** (110 topologies × ~20 eligible targets, filtered by gate count)
- **Runtime:** 2.5 hours on notebooks partition
- **0 discoveries** (fidelity >= 0.99 with fewer gates)
- **Best fidelities:** Suzuki-Trotter 0.69, TFIM 0.67, Toffoli 0.625, CCZ 0.5625

The mined CNOT topologies cannot express the target unitaries. The connectivity patterns from error_correction/communication/linear_algebra families don't match the specific entanglement structure needed for Toffoli, CCZ, Fredkin, or Hamiltonian evolution.

### Dead ends (do not repeat)
- **Per-family mining at low support.** While it produces more patterns, boundary assignment reduces most to 2-qubit. The few genuine 3+ qubit patterns are too small/simple.
- **Gate-level circuit concatenation + ZX simplification.** Equivalent to applying ZX directly to reference circuits. The expert will reject this.
- **Sequential ZX graph composition of mined patterns.** 100% extraction failure rate across 35,000+ attempts.
- **Variational optimization of mined topologies.** Mined CNOT placement patterns lack the connectivity to express non-trivial targets. Best Toffoli fidelity 0.625 (random chance = 0.5625). The topologies are structurally incompatible.

### Guidance for future agents: BE CREATIVE

Previous agents exhaustively tested the obvious approaches (mine patterns → compose → benchmark). The pipeline has been run 10+ times with different parameters. **Do not repeat the same mine-compose-benchmark loop.** Instead, use the pipeline's data and tools as raw material for creative, unconventional strategies.

**Think of the mined data as raw material, not as the final product.** The patterns, algorithms, and their relationships contain rich structure that can be exploited in ways beyond simple composition. Here are directions that go beyond "mine and compose":

1. **Phylogenetic / evolutionary framing.** Treat algorithms as species, ZX patterns as genes, and algorithm families as clades. Build phylogenetic trees based on shared pattern sets. Identify "missing species" — cross-family hybrids predicted by the tree. Look for "convergent evolution" (patterns appearing independently in unrelated families = fundamental building blocks). Agent 4 implemented `scripts/phylogenetic_discovery.py` as a starting point.

2. **Latent space exploration.** Embed algorithm "genomes" (pattern presence/absence vectors) into a low-dimensional latent space via PCA, t-SNE, or a VAE. Sample new points in the latent space and decode them into pattern sets. Construct algorithms from the decoded pattern sets.

3. **Analogy-based reasoning.** Find structural analogies: "algorithm A is to family X as ??? is to family Y." If GHZ state prep (entanglement) shares patterns with teleportation (communication), what communication algorithm corresponds to cluster state prep?

4. **Pattern algebra.** Treat patterns as algebraic objects. Define operations: union, intersection, difference, complement. Explore the algebra: what is "QFT minus Hadamard test"? Does the resulting pattern set correspond to a known or novel algorithm?

5. **Inverse problem.** Start from a desired property (e.g., "3-qubit unitary with entangling power > 0.8 and < 10 gates") and search for pattern combinations that achieve it, using the pattern co-occurrence structure to guide the search.

6. **Meta-learning.** Train a simple model (even logistic regression) to predict which pattern combinations produce extractable, non-trivial circuits. Use the model to prioritize compositions.

7. **Structural biology analogies.** Just as protein structure prediction uses known folds as templates, use known algorithm structures as templates for new algorithms. Identify "folds" (recurring structural motifs across families) and predict new algorithms by recombining folds.

**The key principle: the value is in the RELATIONSHIPS between patterns and algorithms, not in the patterns themselves.** Don't just compose patterns randomly — use the structure of the algorithm space to guide discovery.

### Agent 4 creative experiments (phylogenetic / breeding approaches)

**Phylogenetic analysis** (`scripts/phylogenetic_discovery.py`):
- 781,593 per-family patterns → 576 unique "genes" after structural dedup
- **295 convergent patterns** appear in 3+ families; many appear in ALL 8 families
- Bell state ↔ Deutsch algorithm: **87.9% structural similarity** (29 shared gene IDs)
- The closest cross-family pairs are entanglement × oracular (bell_state × deutsch)
- Per-family mining produces family-isolated pattern IDs; must use structural fingerprinting to detect cross-family sharing

**Algorithm breeding** (`scripts/breed_algorithms.py`):
- Composed 26 cross-family algorithm pairs at 3-5 qubits (A→B, B→A, interleaved), ZX-simplified
- 47 unique hybrid unitaries benchmarked against 248 tasks
- 0 gate improvements, but fidelity=1.0 matches reveal cross-family synthesis:
  - `bit_flip_code + cluster_state → graph_cycle_prep, graph_complete_prep` (error_correction × entanglement creates graph states)
  - `qft + teleportation → graph_star_prep` (arithmetic × communication creates entanglement)
  - `ghz + trotter_ising → graph_complete_prep_5q` at 0.79 fidelity (5-qubit near-miss)
- These compositions show that algorithm breeding CAN produce known algorithms from unexpected family combinations

**Convergent pattern composition** (`scripts/convergent_compose.py`):
- Extracted 26 universal 2-qubit building blocks (patterns appearing in 3+ families)
- 1,173 pair/triple compositions, all 2-qubit, 0 improvements
- The convergent building blocks are structurally simple (3-11 gates) Clifford circuits

## Agent 5 Findings (2026-03-25)

### Approach: Exhaustive breeding + corrected qubit ordering + parameterized ansatz search

Agent 5 fixed the qubit ordering bug (confirmed: old Toffoli fidelity was 0.39 → new correct fidelity 1.0), then ran the most comprehensive breeding search yet. **0 genuine discoveries.**

### Qubit ordering bug: verified fix and impact

Implemented `_unitary_pyzx_convention(qc)`: converts Qiskit's `Operator(qc).data` (q0=LSB) to PyZX convention (q0=MSB) via bit-reversal permutation `U_pyzx = P @ U_qiskit @ P^T`. Verified: Toffoli fidelity jumps from 0.39 (buggy) to 1.00 (correct). Tests pass. **Fix was reverted per failure protocol**, but the implementation is in `scripts/agent5_discovery.py` for future use. 3 lines to change in problem_library.py (lines 515, 612, 711): replace `Operator(qc).data` with `_unitary_pyzx_convention(qc)`.

### Exhaustive breeding (3,721 unique compositions)

- ALL 1,081 algorithm pairs × 4 strategies (A+B, B+A, interleave, reverse-interleave) × 3 qubit counts (3,4,5)
- 8,172 breed attempts → 4,116 compositions → 3,721 unique (by unitary)
- Qubit distribution: {3: 1114, 4: 1349, 5: 1242, 2: 10, 6: 6}
- Benchmarked against 220 tasks (123 corpus + 97 problem library) WITH CORRECTED qubit ordering
- **0 improvements, 0 extraction failures**
- Best controlled gate fidelities (all 0.5625 = (3/4)² = near-identity fidelity floor for 3q)
- Best Hamiltonian near-misses: hubbard 0.975, TFIM 0.973 (all at t=0.1, near-identity inflation)
- State prep fidelity=1.0 matches: GHZ, W, graph states — all using MORE gates than baseline

### Variational refinement on 200 near-misses

- Top 200 near-misses (fidelity > 0.7), all Hamiltonian targets at t=0.1
- Inserted parameterized Rz gates at breed junctions, optimized via differential_evolution
- **0 high-fidelity circuits found** — the junction phases can't compensate for structural incompatibility

### Triple compositions

- Top 30 breeds × 47 algorithms — all "discoveries" were trivial: IQFT+QFT+C = simplified(C)
- IQFT+QFT cancels to identity, leaving just ZX-optimized corpus algorithms
- E.g., `IQFT+QFT+qaoa_sk_model_4q`: 24 vs 26 gates — just standard ZX optimization, not pipeline discovery
- `IQFT+QFT+trotter_heisenberg_3q`: 35 vs 37 gates — same issue

### Parameterized ansatz search with breed-derived topologies

- Extracted 844 unique CNOT topologies (87 from corpus + 757 from breeds)
- For each topology, replaced all single-qubit gates with parameterized Rz·Ry·Rz
- Optimized against Dicke state, controlled gate, QEC, and arithmetic targets
- **Dicke D(3,2) preparation**: found MANY topologies achieving fidelity=1.0 with 3-6 CNOT
  - BUT after ZX simplification, circuits have 62-111 gates (vs 9 baseline) because arbitrary Ry/Rz phases decompose into many T gates
  - Pareto dominance fails: candidate has t=35, depth=23 vs baseline t=0, depth=9
  - The baseline t=0 is a conservative estimate (not real). Real D(3,2) requires non-Clifford gates
  - Furthermore, D(3,2) = X^⊗3 · |W⟩, and W state has known 2-CNOT preparation → D(3,2) optimal is ~2 CNOT, far below baseline estimate of 6

### Root cause analysis

1. **Qubit ordering fix doesn't help**: Correcting the bug changes which matrix is compared, but breed circuits still don't match controlled gate / QEC / arithmetic unitaries. The structural gap is fundamental, not a convention mismatch.

2. **Exhaustive breeding is exhausted**: 1,081 pairs (40× Agent 4's 26) produced no new signal. Adding more pairs or strategies won't help because the fundamental issue is that composed algorithms have random-looking unitaries.

3. **Variational optimization on breed topologies fails**: Breed topologies have the wrong connectivity for target unitaries. Even with fully parameterized single-qubit gates, the CNOT topology constrains the reachable unitary space.

4. **The benchmark system has a structural flaw**: It uses T-count as a Pareto dimension, which penalizes circuits with arbitrary rotations. For state prep targets like Dicke states, the reference baseline has t=0 (estimated), but any real preparation circuit requires non-Clifford rotations that decompose into T gates. This makes it impossible for parameterized circuits to Pareto-dominate.

5. **Problem library baselines are inconsistent**: Some baselines are computed from real reference circuits (state prep via H+CNOT), others are conservative estimates (Dicke: 3n gates). The estimated baselines can be wildly off (Dicke D(3,2) estimated cnot=6 vs real optimal ~2).

### What future agents should know

**The pipeline approach (mine → compose → benchmark) has been thoroughly explored.** Across 5 agents:
- 15+ pipeline runs with varying parameters
- ~150,000 unique survivors
- 3,721 unique algorithm breeds
- 34,000+ cross-family pattern compositions
- 2,200+ variational optimizations
- 3,725+ parameterized ansatz optimizations
- **0 genuine discoveries**

**Fundamental limitations confirmed:**
1. Global mining at min_support≥5 → only 1-2 qubit patterns
2. Per-family mining → more patterns, but boundary assignment reduces most to 2-qubit
3. ZX composition (graph-level) → 0% extraction rate
4. Circuit-level composition + ZX simplification → no structural novelty
5. Breeding full algorithms → matches trivial targets only
6. Variational optimization → wrong topology for non-trivial targets
7. All controlled gate targets (Toffoli, Fredkin, CCZ) are at proven optimal CNOT/T bounds

**If a future agent wants to try this pipeline, they should:**
1. Fix the qubit ordering bug first (code in scripts/agent5_discovery.py and agent6_discovery.py)
2. Fix the Dicke state baselines (use D(n,k) = X^⊗n · D(n,n-k) relationship to derive baselines from W state prep)
3. Consider whether the benchmark framework is fundamentally suitable for evaluating parameterized circuits (T-count penalization issue)
4. Try a completely different paradigm: perhaps using the STRUCTURAL INSIGHTS from mining (which algorithms share patterns, convergent evolution) rather than trying to compose circuits directly

## Agent 6 Findings (2026-03-25)

### Approach: Higher-qubit breeding (4-7 qubits) + parameterized ansatz search

Agent 6 tested the hypothesis that higher qubit counts (4-7) would provide more room for novel circuit discovery. This was motivated by the observation that all previous agents focused on 3-5 qubits, and larger unitaries (up to 128×128 at 7q) have less-explored optimal decompositions.

### Qubit ordering fix: verified and extended

Re-implemented the PyZX convention fix (`_unitary_pyzx_convention`) in `scripts/agent6_discovery.py`. Applied to ALL problem library targets (controlled gates, QEC, arithmetic) when building benchmark tasks. Verified: Toffoli fidelity jumps from 0.39 (buggy Qiskit convention) to 1.00 (correct PyZX convention).

### Phase 1: Exhaustive breeding at 4-7 qubits

- **Algorithm coverage**: 31 algorithms at 4q, 31 at 5q, 31 at 6q, 30 at 7q (including algorithms that naturally produce these qubit counts via ancillae, e.g., grover(5)→6q, deutsch_jozsa(6)→7q)
- **Total pairs**: 465 pairs at 4q + 465 at 5q + 465 at 6q + 435 at 7q = 1,830 pairs × 4 strategies
- **Results**: 7,320 breed attempts → 6,596 unique breeds (by unitary deduplication) → **0 improvements**
- **0 extraction failures** (vs Agents 3-4 which had high extraction failure rates from ZX graph composition)
- **Near-misses**: 781 breeds with fidelity > 0.5, ALL to state_prep targets (GHZ, graph states, cluster states)
- **Non-trivial targets**: 0 breeds above 0.5 fidelity for Hamiltonian, controlled gate, QEC, or arithmetic targets at ANY qubit count (4-7)

### Fidelity=1.0 matches (all use MORE gates than baselines)

| Target | Qubit | Best breed gates | Baseline gates | Ratio |
|--------|-------|-----------------|----------------|-------|
| ghz_prep | 4q | 13 | 4 | 3.3× |
| ghz_prep | 5q | 14 | 5 | 2.8× |
| ghz_prep | 6q | 21 | 6 | 3.5× |
| ghz_prep | 7q | 25 | 7 | 3.6× |
| graph_line_prep | 4q | 10 | 7 | 1.4× |
| graph_line_prep | 7q | 19 | 13 | 1.5× |
| graph_cycle_prep | 7q | 20 | 14 | 1.4× |

Gate overhead INCREASES with qubit count (3.3× at 4q → 3.6× at 7q for GHZ). Higher qubits make the problem WORSE, not better, because compositions of larger circuits accumulate more non-cancelling overhead.

### Phase 2: Triple compositions

Top 30 near-misses (all state prep fidelity=1.0) composed with each algorithm at matching qubit count. Triple compositions only add more gates. No improvements.

### Phase 3: Variational refinement

Differential evolution on Hamiltonian near-misses with insertion of parameterized Rz gates at breed junctions. No breeds with fidelity > 0.5 for Hamiltonians existed to refine.

### Parameterized ansatz search (5-7 qubits)

Separate script (`scripts/agent6_ansatz_search.py`) extracted 76 unique CNOT topologies from corpus algorithms and breeds (33 at 5q, 28 at 6q, 15 at 7q). For each topology with fewer CNOTs than the target baseline, replaced all single-qubit gates with parameterized Rz·Ry·Rz and optimized via Nelder-Mead. Focused on non-trivial targets (Dicke states, W states, Hamiltonians at t≥0.5). Results: all fidelities below 0.5 for priority targets. The CNOT topologies from corpus algorithms don't have the connectivity to express Dicke state preparation or Hamiltonian evolution.

### Root cause: why higher qubits don't help

1. **Gate overhead scales with circuit size.** When breeding algorithms A (g_A gates) and B (g_B gates), ZX simplification removes some cancellations but the result has O(g_A + g_B) gates. For 7-qubit algorithms with 50-140 gates, the compositions have 60-200+ gates after simplification, far above the 7-28 gate baselines.

2. **Cancellation structure doesn't change with qubit count.** ZX full_reduce finds the SAME types of cancellations (spider fusion, bialgebra, pivoting) at 7 qubits as at 3 qubits. It doesn't discover new rewrite rules for larger circuits.

3. **Non-trivial targets require specific structure.** Hamiltonian simulation needs a specific entanglement pattern matching the Hamiltonian's interaction graph. Controlled gates need specific fan-in/fan-out structure. Random compositions of algorithms from different families don't produce these patterns.

4. **State prep matches are trivially explained.** When breed B→A simplifies to a circuit C, and C has cluster_state at the front, C prepares a graph state. This is because ZX simplification preserves the initial Hadamard layer of the first algorithm in the composition.

### Critical guidance for future agents

**STOP TRYING BREEDING.** This approach has now been tested at:
- 3-5 qubits (Agents 2-5): ~150,000 survivors + 3,721 breeds
- 4-7 qubits (Agent 6): 6,596 unique breeds
- **Total: ~160,000+ unique circuits benchmarked against 200+ targets = 0 improvements**

The breeding approach is fundamentally limited by ZX simplification's inability to produce circuits BELOW textbook baselines. ZX simplification is a set of algebraic rewrite rules that simplify a circuit toward a canonical form. If the target circuit is already in or near this canonical form (as textbook implementations are), ZX simplification cannot produce a simpler version.

**Do NOT explore higher qubit counts (8+) for breeding.** Agent 6 demonstrated that gate overhead INCREASES with qubit count. The user hypothesized that higher qubits would help, but the data shows the opposite.

**The remaining viable directions:**
1. **Completely different search paradigm**: GNN-based circuit synthesis, reinforcement learning over circuit transformations, or SAT-based optimization
2. **Expand the corpus**: Add 200+ algorithms from the Quantum Algorithm Zoo. More diverse starting material might produce different cancellation patterns
3. **Custom ZX rewrite rules**: The standard ZX rewrite rules may be incomplete. Research into new ZX calculus identities could enable simplifications that full_reduce misses
4. **Target different problems**: The problem library targets are mostly well-optimized textbook circuits. Add targets from recent papers where optimal implementations are not yet known
5. **Use the mining data differently**: Treat patterns as a LANGUAGE for describing circuit structure, not as building blocks. Use NLP/ML techniques on pattern sequences to predict new valid sequences

### Benchmark system bugs discovered by Agent 6

**Bug 1: CZ/CNOT metric mismatch** — `CircuitMetrics.from_pyzx_circuit` uses `stats_dict["cnot"]` (CNOT only), but `_baseline_from_circuit` maps `stats_dict["twoqubit"]` (all 2-qubit gates) to `baseline_cnot_count`. Circuits using CZ gates show `cnot=0` while baselines show `cnot=N`, causing false Pareto dominance. Fix: use `total_two_qubit` consistently, or compare `twoqubit` fields.

**Bug 2: Baseline depth inflation** — `_baseline_from_circuit` transpiles reference circuits through Qiskit then converts to PyZX. This can produce suboptimal gate orderings (e.g., interleaved H/CZ instead of separated layers), inflating baseline depth. The graph_cycle_prep_7q baseline has depth 8 but trivial reordering gives depth 7.

**Bug 3 (known): Qubit ordering mismatch** — Lines 515, 612, 711 of `problem_library.py` use `Operator(qc).data` (Qiskit little-endian) but candidates use PyZX big-endian convention. Fix code is in `scripts/agent6_discovery.py:_unitary_pyzx_convention`.

**Bug 4 (known): Dicke state baselines** — Conservative estimates use `t_count=0` which no non-Clifford circuit can match. These targets are unbeatable on the Pareto front.

### Higher qubit counts: tested and refuted

The hypothesis that 4-7 qubit counts would yield discoveries (due to larger, less-explored unitary space) was explicitly tested by Agent 6. Result: gate overhead INCREASES with qubit count (3.3× at 4q → 3.6× at 7q for GHZ). Higher qubits make breeding WORSE because larger circuits accumulate more non-cancelling overhead. **Do not repeat this experiment at 8+ qubits.**

### Re-evaluation with all bugs fixed (scripts/agent6_reeval.py)

Re-ran the full 6,596 breeds with corrected metrics (`total_two_qubit` instead of `cnot_count`), fixed qubit ordering, realistic Dicke baselines, and filtered t=0.1 Hamiltonians. Also tried relaxed fidelity thresholds (0.95, 0.90).

Results:
- **fid >= 0.99**: 0 improvements (confirmed: zero genuine discoveries)
- **fid >= 0.95**: 2 false positives (identity → MC6X/MC6Z at 7q, near-identity inflation: MC6X differs from identity on only 2/128 basis states → identity has 96.9% fidelity automatically)
- **fid >= 0.90**: 5 results (4 MC identity inflation, 1 quantum_counting interleave with 91.3% fidelity)
- **Notable**: quantum_counting+hhl_rotation interleave → t=31 vs baseline t=371, same 2q=82, depth 88 vs 527. But fidelity 0.9131 means it doesn't actually implement quantum_counting correctly.

**Key lesson: near-identity inflation gets WORSE at higher qubit counts.** Multi-controlled gates affect only 2/d basis states, so identity has fidelity (1 - 2/d)² ≈ 1 - 4/d. At 7 qubits this is 96.9%. Any future evaluation MUST filter targets where identity fidelity > 0.8.

## Agent 7 Findings (2026-03-25)

### Approach: Multi-strategy ZX optimization + GNN research

Agent 7 researched GNN-based approaches (ZXreinforce, graph BO, generative models), installed ML infrastructure (PyTorch 2.11, PyG 2.7, scikit-learn 1.8), and systematically explored PyZX's full optimization toolkit — specifically extraction and optimization variants that NO previous agent tried.

### GNN research findings

Reviewed 10 papers on GNNs for quantum circuit/ZX optimization. Key references:
- **ZXreinforce** (Nagele & Marquardt 2024): RL+GNN policy selects ZX rewrite rules, beats PyZX full_reduce by 3.2× on large diagrams. Open source. Training requires ~41 hours.
- **Circopt-RL-ZXCalc** (Qilimanjaro 2025): PPO+GNN targets CNOT minimization via ZX, generalizes to 80-qubit circuits.
- **Graph-Based BO** (arXiv:2512.09586): GIN surrogate within Bayesian optimization, 94.25% accuracy for circuit search.
- **AltGraph** (GLSVLSI 2024): Generative graph models for circuit redesign, 37.5% gate reduction.

**Assessment**: RL+GNN for ZX rewriting (ZXreinforce/Circopt) is the most promising direction. It genuinely explores simplification paths beyond greedy full_reduce. However, training requires 40+ hours on GPU — not feasible in a single conversation.

### ML infrastructure installed

Installed in `.venv/`: PyTorch 2.11.0+cu126, PyTorch Geometric 2.7.0, scikit-learn 1.8.0. A100 GPUs available on SLURM. Scripts created:
- `scripts/agent7_stochastic_zx.py`: Multi-strategy ZX optimization
- `scripts/agent7_gnn_surrogate.py`: GNN surrogate pipeline (ready but not trained)

### Multi-strategy ZX optimization (5,932 combinations across 22 targets)

Systematically tried combinations that NO previous agent explored:
1. **Multiple reductions**: `full_reduce` vs `teleport_reduce`
2. **Multiple extractions**: `extract_circuit` with `optimize_cnots=0/1/2/3`, `up_to_perm=True/False`, `lookahead_fast`, `lookahead_extract`, `lookahead_full`, `extract_simple`, `alt_extract`, `gadget_extract`
3. **Multiple optimizations**: raw, `basic_optimization`, `full_optimize` (TODD/phase_block_optimize)
4. **Multiple Qiskit transpilation levels**: 0, 1, 2, 3 as starting points
5. **Stochastic ZX congruences**: Random LC/pivot before reduction

### Key results

**Stochastic congruences make things WORSE** (confirmed empirically):
- For Toffoli: baseline 6 CX → after random congruences + full_reduce: minimum 9 CX (50% worse)
- T-count unchanged (confluent for T-gate optimization)
- Congruences ADD graph structure that extraction can't simplify
- This approach is fundamentally broken for circuit optimization

**`alt_extract_circuit` produces INCORRECT circuits**:
- Consistently gives fidelity < 0.01 across all targets
- Likely requires different graph preprocessing than full_reduce provides
- ALL "beats baseline" false positives came from alt_extract

**`up_to_perm=True` produces permuted circuits**:
- Fidelity drops to 0.25 (3q) or 0.0625 (4q) because qubit permutation is not tracked
- Would need to extract and undo the permutation (add SWAP gates) for fair comparison
- The SWAP overhead negates any savings for small circuits

**`full_optimize` (TODD) is far superior to `basic_optimization`**:
- Toffoli: basic gives 13 CX, TODD gives 7 CX (46% reduction)
- The standard ZX pipeline in this project uses basic_optimization and leaves significant improvement on the table
- But this is a known technique (Amy, Maslov & Mosca, ICALP 2018), not a discovery

**One "discovery" found and REJECTED by expert**:
- [[7,1,3]] Steane encoder: 9 CX (vs baseline 11 CX), fidelity 1.0
- Expert rejection: "This is standard PyZX `full_optimize` output. A single function call `pyzx.optimize.full_optimize(circuit)` produces the same result. MQT QECC framework also reports 9 CX as optimal."
- **0 genuine discoveries** across 5,932 pipeline combinations

### Dead ends (do not repeat)

1. **Stochastic ZX congruences** (LC, pivot) before full_reduce. Congruences ADD structure → extraction produces MORE gates. 100 trials on Toffoli: minimum 9 CX (vs 6 baseline). Do not repeat.
2. **`alt_extract_circuit` on full_reduced graphs**. Produces incorrect circuits. Fidelity < 0.01 across ALL targets.
3. **`up_to_perm=True` without permutation tracking**. Lower gate count but wrong unitary.
4. **Multi-extraction on full_reduced graphs**. All correct extractors (extract_circuit, lookahead_fast, lookahead_extract) produce the same or similar results on the same reduced graph. Extraction is NOT the bottleneck.
5. **Running `full_optimize` on textbook circuits**. The expert will reject this as "standard compiler output."

### What this means for the pipeline

**The ZX-webs pipeline cannot produce novel quantum algorithms.** After 7 agents and:
- 15+ pipeline runs, ~160,000 survivors, 10,000+ breeds
- 5,932 multi-strategy optimization combinations
- Exhaustive testing of all PyZX extraction algorithms and optimization passes
- **0 genuine discoveries**

The fundamental limitations are:
1. **Mining produces tiny patterns**: min_support ≥ 5 → all 1-2 qubit. Per-family mining at min_support=2 → mostly 2-qubit after boundary correction.
2. **Composition adds unreducible overhead**: ZX simplification cannot eliminate the structural bloat from stitching.
3. **Standard tools already optimize well**: full_reduce + full_optimize finds near-optimal circuits for all tested targets.
4. **Problem library targets are at known optimal bounds**: Toffoli (6 CX), Fredkin (5 CX), Steane encoder (9 CX, confirmed by MQT QECC).

### What MIGHT work (for future agents)

1. **RL+GNN for ZX optimization (ZXreinforce/Circopt)**:
   - PyTorch 2.11 + PyG 2.7 are now installed in the venv
   - GNN surrogate code is in `scripts/agent7_gnn_surrogate.py`
   - Requires ~40 hours of GPU training to build a policy that outperforms full_reduce
   - This is the ONLY approach in the literature proven to beat PyZX's greedy algorithms
   - Could genuinely find shorter circuits for non-trivial targets

2. **Expand the corpus to 200+ algorithms** from Quantum Algorithm Zoo:
   - More diverse patterns → higher chance of non-trivial compositions
   - Some algorithms may share structure that produces novel compositions

3. **Redesign the pipeline around approximate synthesis**:
   - Allow fidelity 0.95-0.99 (approximate compilation)
   - Search for circuits that are shorter than exact implementations
   - Use the pipeline's structural data to seed approximate search

4. **Target UNSOLVED problems** instead of well-optimized textbook circuits:
   - Find problems where no optimal circuit is known (e.g., new QEC codes, novel Hamiltonians)
   - The pipeline might find reasonable solutions that constitute discoveries

## Agent 8 Findings (2026-03-25)

### Approach: Target unsolved problems with numerical circuit synthesis

Agent 8 pivoted away from the exhausted mine-compose-benchmark pipeline and focused on genuinely unsolved problems: finding minimum-CNOT circuits for Dicke state preparation, where optimal implementations are not known.

### Phase 1: Trotter + ZX optimization (dead end)

Built explicit first-order Trotter circuits for all Hamiltonian targets (TFIM, XY, Heisenberg) with corrected signs. ZX full_reduce + basic_optimization INCREASES 2-qubit gate count in ALL cases:
- TFIM 3q: 4 CX → 6 CX (+50%)
- XY 3q: 8 CX → 14 CX (+75%)
- Heisenberg 3q: 12 CX → 14 CX (+17%)

ZX extraction adds unreducible overhead to already-minimal Trotter circuits. Confirmed: ZX cannot improve term-by-term Trotter decompositions.

Note: `full_optimize` (TODD algorithm) ONLY works on Clifford+T circuits. Trotter circuits with Rz(arbitrary angle) cause TypeError. Must use `basic_optimization` for non-Clifford circuits.

### Phase 2: Dicke state D(4,2) minimum-CNOT search -- DISCOVERY

**Result: D(4,2) can be prepared with exactly 5 CNOT gates**, improving on the best published construction of 6 CNOT (Wang, Cong & De Micheli, arXiv:2401.01009, 2024).

Method: Parameterized Ry+CNOT circuits (sufficient since D(4,2) has all-real amplitudes). Swept CNOT count k=2..8 with random topology sampling (500 topologies × 6 restarts × L-BFGS-B optimization).

Lower bound evidence:
- k=2: max fidelity = 4/6 = 0.6667
- k=3: max fidelity = 5/6 = 0.8333
- k=4: max fidelity = 0.8727 (confirmed by 14 structured topologies + differential evolution)
- k=5: fidelity = 1.0 (exact)

Optimal topology: CX(1,3), CX(2,3), CX(0,3), CX(0,1), CX(3,0) -- a "fan-in to q3, then redistribute" pattern not found in any standard construction.

Full circuit: `docs/discovery.md` contains QASM and verification details.

Expert review verdict: **CONDITIONALLY VALID** -- genuine improvement over published results, correctly verified. Caveats: lower bound is numerical (not proven), practical impact is incremental (1 CNOT saving), found via scipy.optimize (not ZX pipeline).

### Phase 3: Hamiltonian exact synthesis (incomplete)

TFIM 3q at t=0.5 with k=1 CNOT: fidelity 0.71 (identity fidelity 0.48). Job timed out on short partition before completing higher k values.

### Phase 4: D(5,2) search (in progress)

D(5,2) minimum-CNOT search results so far:
- k=2: fid=0.413, k=3: fid=0.524, k=4: fid=0.722 (still searching higher k)

### Key technical findings

1. **PyZX `full_optimize` crashes on non-Clifford circuits**: Always wrap in try/except and fall back to `basic_optimization`.
2. **Trotter sign convention**: For H = +J·XX, the Trotter factor is e^(-iJt·XX), requiring theta = -Jt in `trotter_xx`. For H = -J·ZZ (TFIM), theta = +Jt. Previous agents' Trotter circuits for XY and Heisenberg had sign errors.
3. **State-vector simulation is much faster**: For state preparation targets, compute U|0> directly (O(d) per layer) instead of the full unitary (O(d^2) per layer). This gives ~10x speedup for 4-5 qubit searches.
4. **Ry+CNOT suffices for real-amplitude states**: Dicke states have all-real amplitudes, so Ry+CNOT (no Rz needed) is sufficient and reduces the parameter space by 3x.
5. **Problem library Dicke baselines have t_count=0 bug**: No real Dicke state circuit can have t_count=0 (requires non-Clifford Ry rotations). The Pareto dominance check always fails for Dicke targets due to this bug.
