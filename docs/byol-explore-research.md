# BYOL-Explore for Guided Quantum Algorithm Discovery

## Background

[BYOL-Explore](https://arxiv.org/abs/2206.08332) (Guo et al., DeepMind 2022) is a
curiosity-driven exploration algorithm that learns a world representation by predicting
its own future latent states. Prediction error serves as an **intrinsic reward** — the
agent is drawn toward states it cannot yet predict, driving exploration of novel regions.
It achieved state-of-the-art on hard-exploration Atari games and DM-HARD-8, surpassing
Agent57 and Go-Explore with a simpler architecture (no episodic memory, no bandit).

### Core mechanism

1. An **online network** encodes observations and predicts future latent representations
2. A **target network** (exponential moving average of the online network) produces prediction targets from actual future observations
3. The **prediction error** between online predictions and target representations is the intrinsic reward
4. A policy is trained to maximize this intrinsic reward (optionally combined with extrinsic reward)

## Relevance to ZX-Webs

ZX-Webs currently uses a systematic pipeline: mine frequent sub-diagrams with gSpan,
then combinatorially compose them (sequential + parallel stitching), filter failures,
and benchmark survivors. This is thorough but does not learn from failures or
prioritize promising regions of the search space.

A BYOL-Explore–style approach could replace brute-force composition with **learned
curiosity-guided search**:

| Component         | Current (ZX-Webs)                  | BYOL-Explore Adaptation                  |
|-------------------|------------------------------------|-------------------------------------------|
| State space       | ZX-diagrams (graphs)               | GNN-encoded latent representations        |
| Actions           | All valid compositions             | Policy-selected compositions              |
| Exploration       | Exhaustive enumeration             | Curiosity-driven (prediction error)       |
| Learning          | None (static pipeline)             | Online representation + policy learning   |
| Reward            | Post-hoc benchmarking              | Intrinsic (novelty) + extrinsic (quality) |

## Proposed Architecture

```
ZX-Diagram ──► GNN Encoder ──► Latent z_t
                                    │
                    ┌───────────────┤
                    ▼               ▼
              Policy π(a|z_t)   Predictor f(z_t, a) ──► predicted z_{t+1}
                    │                                         │
                    ▼                                         ▼
            Composition action a                    Prediction error ║z̃ - z_{t+1}║²
                    │                                    = intrinsic reward
                    ▼
            New ZX-Diagram ──► GNN Encoder ──► z_{t+1} (target)
```

### Key design choices

1. **Graph representation**: Use a GNN (e.g., GIN or GAT) to encode ZX-diagrams.
   Vertex features: spider type (Z/X/H/boundary) + discretized phase.
   Edge features: Hadamard edge flag.

2. **Action space**: Select (pattern_i, pattern_j, composition_mode) triples.
   Could be factored into sub-decisions for tractability.

3. **Extrinsic reward** (optional): Circuit metrics from the benchmark stage —
   gate count reduction, depth, SupermarQ feature novelty, or functional
   equivalence to known useful unitaries.

4. **Curriculum**: Start with small diagrams/few patterns, grow as the model
   improves — analogous to how BYOL-Explore naturally progresses from easy
   to hard exploration.

## Feasibility Assessment

**In favor:**
- GNN encoders for quantum circuits are well-studied (e.g., in circuit optimization, QAS)
- BYOL-Explore's algorithm is simple to implement (~single prediction loss)
- The ZX-diagram action space is naturally discrete and relatively small per step
- Could dramatically reduce the number of compositions tried vs. exhaustive search

**Challenges:**
- ZX-diagrams are discrete graphs, not pixel observations — representation learning
  dynamics may differ from the original Atari/DM-HARD-8 settings
- Defining meaningful "episodes" in the composition process
- Evaluation: need a clear signal for what constitutes a "good" discovered algorithm
- Training data efficiency — each composition + extraction + benchmark is expensive

## Relation to Probability-of-Success Estimation

BYOL-Explore's prediction error is effectively an **uncertainty estimate** over the
search space. High prediction error ≈ "I don't understand this region yet" ≈ worth
exploring. This is conceptually similar to asking "what's the probability of finding
something interesting here?" — the model learns to estimate its own ignorance and
act on it.

## References

- Guo, Z.D. et al. "BYOL-Explore: Exploration by Bootstrapped Prediction." NeurIPS 2022.
  [arXiv:2206.08332](https://arxiv.org/abs/2206.08332)
- Grill, J.-B. et al. "Bootstrap Your Own Latent." NeurIPS 2020.
  (Foundation for the BYOL representation learning approach)
- [DeepMind blog post](https://deepmind.google/blog/byol-explore-exploration-with-bootstrapped-prediction/)
