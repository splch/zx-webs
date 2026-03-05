# Quantum Algorithm Phylogeny & Discovery

Two scripts that use the ZX motif pipeline to build a data-driven taxonomy of
quantum algorithms, then exploit structural gaps in that taxonomy to construct
novel circuits.

## Quick Start

```bash
# From the repo root, with the venv active:
source .venv/bin/activate

# Step 1: Build the phylogeny (~15 min, produces the fingerprint database)
python scripts/discover_phylogeny.py

# Step 2: Discover new algorithms (~10 sec, reads Step 1 outputs)
python scripts/discover_algorithm.py
```

Both scripts are self-contained. They import the existing pipeline
(`registry`, `converter`, `featurizer`, `matcher`, `motif_generators`,
`decomposer`) but do not modify any source files.

---

## Script 1: `discover_phylogeny.py`

Fingerprints all 32 registered quantum algorithms (88 instances across qubit
sizes) by their ZX motif profiles, clusters them into a phylogenetic tree, and
classifies motifs by universality and simplification survival.

### Pipeline

1. **Build corpus** — Convert every algorithm at every simplification level
   (RAW through TELEPORT_REDUCE) into coarsened-phase NetworkX graphs.
2. **Discover motifs** — Combine 15 hand-crafted motifs with bottom-up
   subgraph enumeration and neighbourhood extraction. Deduplicate via WL hash +
   VF2 confirmation. Produces ~367 motifs.
3. **Build fingerprint matrix** — Count occurrences of each motif in each
   algorithm instance at spider_fused level. L1-normalise into frequency
   vectors.
4. **Run six analyses:**
   - Hierarchical clustering (cosine distance) into a dendrogram
   - PCA scatter coloured by algorithm family
   - Motif universality spectrum (universal / common / specific)
   - Cross-level survival heatmap (which motifs resist simplification)
   - Cross-family surprise relatives (high similarity, different families)
   - Decomposition coverage by family (greedy set cover)

### Key Findings

| Finding | Detail |
|---------|--------|
| 8 universal motifs | `cx_pair` and `syndrome_extraction` appear in all 10 families |
| 4 irreducible motifs | `phase_gadget_2t`, `phase_gadget_3t`, `hadamard_sandwich`, `cluster_chain` survive full ZX reduction |
| BBPSSW ~ GHZ | Distillation and entanglement generation are 99% structurally similar at the ZX level |
| QAOA ~ Trotter | Variational optimisation and Hamiltonian simulation share 93% of their motif structure |
| Coverage gap | Distillation has the lowest motif coverage (61.1%), meaning its ZX structure is the least explained by the motif library |

### Outputs (`scripts/output/`)

| File | Description |
|------|-------------|
| `phylogeny_dendrogram.png` | Hierarchical clustering dendrogram, leaves coloured by family |
| `pca_scatter.png` | 2D PCA projection of motif fingerprints |
| `universality_spectrum.png` | Bar chart of motif universality across families |
| `cross_level_survival.png` | Heatmap of motif presence across simplification levels |
| `coverage_landscape.png` | Box plot of decomposition coverage by family |
| `fingerprint_counts.csv` | Raw motif match counts (88 instances x 367 motifs) |
| `fingerprint_frequencies.csv` | L1-normalised frequencies |
| `results.json` | Machine-readable analysis results |
| `report.txt` | Human-readable summary |

---

## Script 2: `discover_algorithm.py`

Uses the phylogeny findings to construct three novel quantum circuits that
fill structural gaps in algorithm space. Each circuit is validated as a
legitimate unitary operation, fingerprinted through the same motif pipeline,
and placed on the existing phylogeny.

### Candidates

**Trotter-Variational Hybrid (TVH)** — Motivated by the 93% QAOA-Trotter
similarity. Combines a Trotter ZZ-interaction backbone with variational
angles, then adds `cluster_chain` and `hadamard_sandwich` layers (the two
entangling motifs that survive full ZX reduction). Neither QAOA nor Trotter
contains these layers, so TVH occupies a distinct region in PCA space between
the variational and simulation families.

**Distillation-Entanglement Bridge (DEB)** — Motivated by the 99%
BBPSSW-GHZ similarity and distillation's 61% coverage gap. A parameterised
circuit that interpolates between GHZ-style fan-out and BBPSSW bilateral-CNOT
structure via a tunable RY rotation layer. This turned out to be the most
novel candidate (novelty score 0.654), with low similarity to any existing
algorithm.

**Irreducible Core Circuit (ICC)** — Built exclusively from the four motifs
that survive full ZX reduction: `phase_gadget_2t`, `phase_gadget_3t`,
`hadamard_sandwich`, and `cluster_chain`. No existing algorithm in the
registry is designed around ZX-irreducible primitives. The circuit compresses
dramatically under simplification (63V/75E raw to 19V/28E reduced) while
preserving its tensor, confirming these motifs form a hard computational core.

### Pipeline

1. **Load phylogeny results** from Step 1 outputs.
2. **Identify structural gaps** via PCA on the existing fingerprint matrix.
3. **Construct candidates** as standard Qiskit `QuantumCircuit` objects.
4. **Validate** unitarity (`qiskit.quantum_info.Operator`) and ZX tensor
   preservation (`zx.compare_tensors` before/after full reduction).
5. **Fingerprint** candidates against the same 367-motif library.
6. **Analyse placement** — nearest existing algorithm, family affinity,
   novelty score.
7. **Visualise** — overlay candidates on the PCA scatter and dendrogram.

### Results

| Candidate | Gates | Novelty | Nearest Existing Algorithm | Similarity |
|-----------|-------|---------|--------------------------|------------|
| TVH | 60 | 0.257 | cluster_state_q3 | 0.743 |
| DEB | 12 | 0.654 | w_state_q3 | 0.346 |
| ICC | 39 | 0.359 | trotter_heisenberg_q2 | 0.641 |

All three pass unitarity validation and ZX tensor preservation.

### Outputs (`scripts/output/discovery/`)

| File | Description |
|------|-------------|
| `pca_candidates.png` | PCA scatter with candidates overlaid as star markers |
| `dendrogram_candidates.png` | Full dendrogram with candidates highlighted in bold |
| `candidate_profiles.png` | Top-motif bar charts for each candidate |
| `candidate_fingerprints.csv` | Fingerprint vectors for the 3 candidates |
| `results.json` | Validation, placement, and novelty scores |
| `report.txt` | Human-readable discovery report |
