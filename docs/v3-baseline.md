# ZX-Webs v3: Scaled Discovery

## Overview

v3 scales the pipeline with an expanded corpus, farthest-point sampling for structural diversity, performance optimizations, and 50K candidates. The pipeline rediscovered 6 known quantum circuits — including a **3-qubit error correction code** — purely from cross-family ZX-Web recombination with no guidance or bias.

## What Changed from v2

### Corpus expansion
- 47 algorithms across 8 families (was 13 across 5 in v1)
- New families: error correction, linear algebra, communication
- 276 corpus circuits at qubit counts 3, 4, 5, 7 (was 134 at 3, 5)
- Multiple parameter variants per algorithm

### Farthest-point sampling (FPS)
- Replaced uniform random sampling with FPS at web selection and pair generation
- 5D structural feature vector: (n_inputs, n_outputs, n_spiders, support, n_source_families)
- Numpy-vectorized for scalability (O(k*N*d) per selection)
- Maximizes structural diversity without biasing toward any outcome

### Performance
- In-memory data passing between Stage 3→4 (eliminates 224K file reads)
- orjson (Rust-backed) replaces stdlib json for 3-10x faster serialization
- Worker cap removed on Stage 5 extraction (was 16, system has 128 CPUs)
- Configurable `max_webs_loaded` caps FPS computation time
- tqdm progress bars on FPS selection loop and all stages

### Bug fixes
- Boundary direction encoding: input/output boundaries get distinct gSpan labels
- Balanced composition budgets: all strategies get equal allocation
- Configurable `max_webs_loaded` (was hard-coded)

## v3 Results (large_run.yaml)

```
Corpus:          276 circuits (47 algorithms × variants × 4 qubit counts)
ZX Diagrams:     276 (150 included for mining, 126 too large)
Mined Webs:      255,374 (127,687 patterns × 2 reversed boundaries)
FPS Selected:    1,414 structurally diverse webs
Candidates:      50,000
Extracted:       5,502 (11.0% extraction rate)
Unique Survivors: 2,055
  2-qubit:       1,264
  3-qubit:         627
  4-qubit:         155
  5-qubit:           9
Benchmark Tasks: 150
Perfect Matches: 6 (fidelity ≥ 0.99)
Mean Fidelity:   0.20
```

## Discoveries

6 candidates achieved perfect fidelity (≥ 0.99) to known algorithms:

| Survivor | Qubits | Gates | Perfect match to | Composition |
|---|---|---|---|---|
| surv_1144 | **3** | 6 | **3-qubit bit-flip code** | cross-family |
| surv_0027 | 2 | 6 | Bell state preparation | cross-family |
| surv_0194 | 2 | 7 | Cluster state | cross-family |
| surv_0317 | 2 | 3 | BB84 encoding | cross-family |
| surv_0346 | 2 | 7 | Deutsch's algorithm | cross-family |
| surv_0394 | 2 | 4 | Bell state preparation | cross-family |

The 3-qubit bit-flip code discovery is significant: the pipeline reconstructed an error correction encoding circuit from ZX sub-patterns mined from unrelated algorithm families. This was not guided or hinted at — it emerged from unbiased structural recombination.

None of the perfect matches are improvements over the baseline (they have equal or more gates), but they demonstrate that the pipeline can reconstruct known quantum operations from cross-family ZX-Web composition.

## Progression

| Metric | v1.0 | v2.0 | v3.0 |
|---|---|---|---|
| Algorithms | 13 (5 families) | 47 (8 families) | 47 (8 families) |
| Corpus circuits | 26 | 134 | 276 |
| Mined webs | 2,767 | 151,434 | 255,374 |
| Candidates | 500 | 2,000 | 50,000 |
| Survivors | 31 | 445 | 2,055 |
| 3+ qubit survivors | 19 (collapsed) | 76 | 791 |
| Best fidelity | 0.12 | 0.89 | **1.00** |
| Perfect matches | 0 | 0 | **6** |
| 3-qubit perfect | 0 | 0 | **1** |
| Evaluation | Gate counts | Process fidelity | Process fidelity |
| Sampling | Random | Random | **FPS** |
| Tests | 156 | 170 | 170 |

## Search Coverage

```
Total webs:           255,374
Possible pairs:       32.6 billion
Pairs attempted:      500,000
Search coverage:      0.0015%
Perfect matches found: 6
```

At 0.0015% coverage we found 6 perfect matches including a 3-qubit error correction code. The search space is vast and we've barely scratched the surface.

## Architecture

Same 7-stage pipeline as v1/v2, with these additions:

- **FPS sampling** (`_farthest_point_sample`, `_fps_dissimilar_pairs`): numpy-vectorized structural diversity maximization
- **In-memory stage passing**: Pipeline caches Stage 3 output for Stage 4, eliminating disk I/O bottleneck
- **Bulk JSON**: Single-file web serialization alongside individual files for fallback
- **orjson**: Rust-backed JSON for faster serialization

## Running

```bash
# Small run (~5 min, 2K candidates)
sbatch --partition=short --ntasks=1 --cpus-per-task=8 --mem=64G \
  --output=logs/pipeline_%j.log --error=logs/pipeline_%j.log \
  --wrap="python scripts/run_pipeline.py --config configs/small_run.yaml"

# Medium run (~10 min, 10K candidates)
sbatch --partition=short --ntasks=1 --cpus-per-task=16 --mem=128G \
  --output=logs/pipeline_%j.log --error=logs/pipeline_%j.log \
  --wrap="python scripts/run_pipeline.py --config configs/medium_run.yaml"

# Large run (~10 min, 50K candidates)
sbatch --partition=short --ntasks=1 --cpus-per-task=64 --mem=200G \
  --output=logs/pipeline_%j.log --error=logs/pipeline_%j.log \
  --wrap="python scripts/run_pipeline.py --config configs/large_run.yaml"
```

## Next Directions

- **Scale further**: 100K+ candidates, more qubit counts, more algorithm variants
- **Investigate the 3-qubit discovery**: What ZX-Webs composed into the bit-flip code? What families contributed?
- **Push for improvements**: Find circuits that match known algorithms with *fewer* gates
- **Larger qubit compositions**: Current peak is 5 qubits; need larger mined patterns for 7+ qubit discoveries
- **C++ gSpan**: Replace pure Python gSpan with C++ binary for 50-200x mining speedup
- **QuiZX**: Rust-based ZX-calculus backend for 100-5000x graph operation speedup
