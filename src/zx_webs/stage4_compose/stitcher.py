"""Stage 4 composition engine -- stitch ZX-Webs into candidate algorithms.

The :class:`Stitcher` takes a collection of :class:`ZXWeb` objects and
produces :class:`CandidateAlgorithm` instances by composing webs using
both **undirected** and **goal-directed** strategies.

Undirected strategies (preserved from the original implementation):

1. **Sequential compose** -- use PyZX's native ``compose()`` to chain two
   webs end-to-end.  Requires matching output/input wire counts.
   ~100% extraction rate.

2. **Parallel tensor + Hadamard stitching** -- place two webs in parallel
   via PyZX's ``tensor()`` and add Hadamard edges between interior
   Z-spiders from different webs.  Creates novel entanglement patterns.
   ~93% extraction rate.

3. **Phase perturbation** -- take a composed (or original) diagram and
   randomly modify phases of interior Z-spiders with Clifford+T values.
   Creates novel unitaries while preserving graph structure.
   ~98% extraction rate.

Goal-directed strategies (new):

4. **Cross-family recombination** -- prioritise compositions that combine
   sub-patterns from different algorithm families (e.g. QFT + Grover),
   as these are the most likely sources of genuinely novel algorithms.

5. **Target-guided composition** -- when target tasks are provided,
   prioritise compositions whose qubit count and family provenance
   match the target, increasing the chance of discovering more
   efficient implementations of known algorithms.

The :func:`run_stage4` entry point loads webs from Stage 3 output, runs
composition, and persists the results.
"""
from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyzx as zx
from tqdm import tqdm

from zx_webs.config import ComposeConfig
from zx_webs.persistence import load_json, load_manifest, save_json, save_manifest
from zx_webs.stage3_mining.zx_web import BoundaryWire, ZXWeb
from zx_webs.stage4_compose.boundary import (
    count_boundary_wires,
    junction_edge_type,
    wire_compatibility_score,
)
from zx_webs.stage4_compose.candidate import CandidateAlgorithm

logger = logging.getLogger(__name__)

# PyZX vertex-type constant for boundary vertices.
_VT_BOUNDARY = 0
_VT_Z = 1

# Maximum qubit count for generated candidates.
_MAX_QUBITS = 10

# Clifford+T phase values: k*pi/4 for k in 0..7 (as fractions of pi).
_CLIFFORD_T_PHASES = [Fraction(k, 4) for k in range(8)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_families(webs: list[ZXWeb], indices: list[int]) -> list[str]:
    """Collect the union of source families from the given web indices."""
    families: list[str] = []
    seen: set[str] = set()
    for idx in indices:
        for fam in webs[idx].source_families:
            if fam not in seen:
                families.append(fam)
                seen.add(fam)
    return families


def _is_cross_family(webs: list[ZXWeb], indices: list[int]) -> bool:
    """Return True if the webs at the given indices come from >1 family."""
    all_families: set[str] = set()
    for idx in indices:
        all_families.update(webs[idx].source_families)
    return len(all_families) > 1


def _pair_compatibility_score(
    web_a: ZXWeb, web_b: ZXWeb, prefer_cross_family: bool = True
) -> float:
    """Score how promising a pair of webs is for composition.

    Higher scores indicate more interesting potential compositions.
    """
    score = 0.0

    # Cross-family bonus: combining patterns from different families
    # is more likely to produce novel algorithms.
    if prefer_cross_family:
        families_a = set(web_a.source_families)
        families_b = set(web_b.source_families)
        if families_a and families_b and not families_a.intersection(families_b):
            score += 5.0  # strong cross-family bonus
        elif families_a and families_b and families_a != families_b:
            score += 2.0  # partial overlap bonus

    # Higher support webs are more likely to represent fundamental patterns.
    score += min(web_a.support, 10) * 0.1
    score += min(web_b.support, 10) * 0.1

    # Boundary wire compatibility scoring.
    out_wires = [bw for bw in web_a.boundary_wires if bw.direction == "output"]
    in_wires = [bw for bw in web_b.boundary_wires if bw.direction == "input"]
    if out_wires and in_wires:
        # Average the best wire compatibility scores.
        best_scores = []
        for ow in out_wires:
            wire_scores = [wire_compatibility_score(ow, iw) for iw in in_wires]
            best_scores.append(max(wire_scores))
        if best_scores:
            score += sum(best_scores) / len(best_scores)

    return score


# ---------------------------------------------------------------------------
# Stitcher
# ---------------------------------------------------------------------------


class Stitcher:
    """Compose ZX-Webs into candidate algorithms.

    Parameters
    ----------
    config:
        Stage 4 composition parameters.
    """

    def __init__(self, config: ComposeConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    # ------------------------------------------------------------------
    # Strategy 1: Sequential compose via PyZX native compose()
    # ------------------------------------------------------------------

    def compose_sequential(
        self, web_a: ZXWeb, web_b: ZXWeb
    ) -> zx.Graph | None:
        """Compose two webs using PyZX's native ``compose()``.

        1. Reconstruct PyZX graphs with proper boundary vertices.
        2. Ensure output count of A matches input count of B.
        3. Use ``g_a.compose(g_b)`` which correctly handles boundary merging.
        4. Return composed graph (or ``None`` if wire counts don't match).

        Parameters
        ----------
        web_a:
            The first web (its outputs connect to web_b's inputs).
        web_b:
            The second web.

        Returns
        -------
        zx.Graph | None
            The composed graph, or ``None`` if the webs are incompatible.
        """
        g_a = web_a.to_pyzx_graph()
        g_b = web_b.to_pyzx_graph()

        # Must have proper boundary information.
        if not g_a.outputs() or not g_b.inputs():
            return None
        if len(g_a.outputs()) != len(g_b.inputs()):
            return None
        if len(g_a.outputs()) == 0:
            return None

        try:
            result = g_a.copy()
            result.compose(g_b.copy())
            return result
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug(
                "compose_sequential failed for %s + %s: %s",
                web_a.web_id, web_b.web_id, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Strategy 2: Parallel tensor + Hadamard stitching
    # ------------------------------------------------------------------

    def compose_parallel_stitch(
        self, web_a: ZXWeb, web_b: ZXWeb, n_hadamard_edges: int = 1
    ) -> zx.Graph | None:
        """Place two webs in parallel and add Hadamard edges between
        interior Z-spiders from different webs.

        1. Create tensor product of both graphs (disjoint union with
           proper input/output merging).
        2. Track which vertices came from each original graph.
        3. Select random interior Z-spiders from each graph.
        4. Add Hadamard edges between them (creates novel entanglement).

        Parameters
        ----------
        web_a:
            The first web.
        web_b:
            The second web.
        n_hadamard_edges:
            Number of Hadamard edges to add between the two sub-diagrams.

        Returns
        -------
        zx.Graph | None
            The composed graph, or ``None`` if either web lacks boundary info.
        """
        g_a = web_a.to_pyzx_graph()
        g_b = web_b.to_pyzx_graph()

        if not g_a.inputs() or not g_a.outputs():
            return None
        if not g_b.inputs() or not g_b.outputs():
            return None

        # Remember vertices from g_a before tensor.
        a_verts = set(g_a.vertices())

        # Use PyZX tensor product (g_a @ g_b).
        combined = g_a.tensor(g_b)

        # Identify interior Z-spiders from each original graph.
        input_set = set(combined.inputs())
        output_set = set(combined.outputs())
        boundary_set = input_set | output_set

        # Vertices from g_a keep their IDs in the tensor product.
        # Vertices from g_b are remapped; they are the ones NOT in a_verts.
        all_verts = set(combined.vertices())

        a_interiors = [
            v for v in all_verts
            if combined.type(v) == _VT_Z
            and v not in boundary_set
            and v in a_verts
        ]
        b_interiors = [
            v for v in all_verts
            if combined.type(v) == _VT_Z
            and v not in boundary_set
            and v not in a_verts
        ]

        # Add Hadamard edges between random pairs.
        n_edges = min(n_hadamard_edges, len(a_interiors), len(b_interiors))
        if n_edges > 0:
            a_sample = self.rng.sample(a_interiors, min(n_edges, len(a_interiors)))
            b_sample = self.rng.sample(b_interiors, min(n_edges, len(b_interiors)))
            for a_v, b_v in zip(a_sample, b_sample):
                combined.add_edge((a_v, b_v), edgetype=2)  # Hadamard edge

        return combined

    # ------------------------------------------------------------------
    # Strategy 2b: Pure parallel composition (tensor, no stitching)
    # ------------------------------------------------------------------

    def compose_parallel(
        self, web_a: ZXWeb, web_b: ZXWeb
    ) -> zx.Graph | None:
        """Compose two webs in parallel (tensor product) without stitching.

        Always succeeds if both webs have boundary info.  The combined
        graph has both sets of vertices and edges with inputs/outputs
        properly merged.

        Parameters
        ----------
        web_a:
            The first web.
        web_b:
            The second web.

        Returns
        -------
        zx.Graph | None
            The tensor product graph, or ``None`` if a web lacks boundaries.
        """
        g_a = web_a.to_pyzx_graph()
        g_b = web_b.to_pyzx_graph()

        if not g_a.inputs() or not g_a.outputs():
            return None
        if not g_b.inputs() or not g_b.outputs():
            return None

        return g_a.tensor(g_b)

    # ------------------------------------------------------------------
    # Strategy 3: Phase perturbation
    # ------------------------------------------------------------------

    def perturb_phases(
        self, graph: zx.Graph, rate: float = 0.2
    ) -> zx.Graph:
        """Create a novel unitary by perturbing interior spider phases.

        1. Copy the graph.
        2. For each interior Z-spider, with probability ``rate``,
           replace its phase with a random Clifford+T phase (k*pi/4).
        3. Return the modified graph.

        Parameters
        ----------
        graph:
            A PyZX graph (will not be mutated).
        rate:
            Probability of perturbing each spider's phase.

        Returns
        -------
        zx.Graph
            The perturbed graph.
        """
        g = graph.copy()

        input_set = set(g.inputs()) if g.inputs() else set()
        output_set = set(g.outputs()) if g.outputs() else set()
        boundary_set = input_set | output_set

        for v in g.vertices():
            if g.type(v) != _VT_Z:
                continue
            if v in boundary_set:
                continue
            if self.rng.random() < rate:
                new_phase = self.rng.choice(_CLIFFORD_T_PHASES)
                g.set_phase(v, new_phase)

        return g

    # ------------------------------------------------------------------
    # Candidate generation -- core builder
    # ------------------------------------------------------------------

    def _make_candidate(
        self,
        graph: zx.Graph,
        webs: list[ZXWeb],
        web_indices: list[int],
        comp_type: str,
        cand_idx: int,
        min_qubits: int,
    ) -> CandidateAlgorithm | None:
        """Build a CandidateAlgorithm from a composed graph, or None."""
        n_qubits = len(graph.inputs()) if graph.inputs() else 0
        n_outputs = len(graph.outputs()) if graph.outputs() else 0
        if n_qubits > _MAX_QUBITS or n_qubits == 0:
            return None
        if n_qubits < min_qubits:
            return None
        # A valid quantum circuit requires equal numbers of inputs
        # and outputs.  Reject unbalanced compositions early.
        if n_qubits != n_outputs:
            return None
        n_spiders = sum(
            1 for v in graph.vertices() if graph.type(v) != _VT_BOUNDARY
        )
        web_ids = [webs[idx].web_id for idx in web_indices]
        families = _collect_families(webs, web_indices)
        cross_family = _is_cross_family(webs, web_indices)

        return CandidateAlgorithm(
            candidate_id=f"cand_{cand_idx:04d}",
            graph_json=graph.to_json(),
            component_web_ids=web_ids,
            composition_type=comp_type,
            n_qubits=n_qubits,
            n_spiders=n_spiders,
            source_families=families,
            is_cross_family=cross_family,
        )

    # ------------------------------------------------------------------
    # Candidate generation -- undirected (original strategies)
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        webs: list[ZXWeb],
        target_tasks: list[dict[str, Any]] | None = None,
    ) -> list[CandidateAlgorithm]:
        """Generate candidate algorithms from a list of ZX-Webs.

        Strategies employed:

        1. **Cross-family recombination** (if ``prefer_cross_family`` is True)
           -- prioritise pairs of webs from different algorithm families.
        2. **Sequential compose** -- try ordered pairs ``(A, B)`` where
           A's output count matches B's input count.
        3. **Parallel tensor** -- tensor product of pairs, optionally with
           Hadamard stitching for novelty.
        4. **Phase perturbation** -- take successful compositions and
           create variants with perturbed phases.
        5. **Target-guided composition** (if ``guided`` is True and
           *target_tasks* are provided) -- compose webs whose qubit
           counts match the target tasks.

        Candidates with more than :data:`_MAX_QUBITS` qubits are dropped.
        The total number of candidates is capped by
        ``config.max_candidates``.

        Parameters
        ----------
        webs:
            The ZX-Webs to compose.
        target_tasks:
            Optional list of target task descriptions for guided
            composition.  Each dict should have at least ``n_qubits``
            and optionally ``family``.

        Returns
        -------
        list[CandidateAlgorithm]
        """
        candidates: list[CandidateAlgorithm] = []
        max_cand = self.config.max_candidates
        cand_idx = 0
        min_qubits = self.config.min_compose_qubits

        def _try_add(
            graph: zx.Graph | None,
            web_indices: list[int],
            comp_type: str,
        ) -> bool:
            """Try to add a candidate; return True if added."""
            nonlocal cand_idx
            if graph is None:
                return False
            cand = self._make_candidate(
                graph, webs, web_indices, comp_type, cand_idx, min_qubits,
            )
            if cand is not None:
                candidates.append(cand)
                cand_idx += 1
                return True
            return False

        # -- Build pair scoring for prioritisation ----------------------------
        all_pair_indices = list(combinations(range(len(webs)), 2))

        if self.config.prefer_cross_family:
            # Sort pairs by compatibility score (descending) to try
            # cross-family compositions first.
            scored_pairs = [
                (
                    i, j,
                    _pair_compatibility_score(
                        webs[i], webs[j],
                        prefer_cross_family=True,
                    ),
                )
                for i, j in all_pair_indices
            ]
            scored_pairs.sort(key=lambda t: -t[2])
            ordered_pairs = [(i, j) for i, j, _ in scored_pairs]
        else:
            ordered_pairs = list(all_pair_indices)
            self.rng.shuffle(ordered_pairs)

        # Limit number of pairs to avoid OOM.
        if len(ordered_pairs) > max_cand * 10:
            ordered_pairs = ordered_pairs[: max_cand * 10]

        # -- 1. Pairwise sequential -------------------------------------------
        if "sequential" in self.config.composition_modes:
            for i, j in tqdm(ordered_pairs, desc="Stage 4: Sequential compose", unit="pair"):
                if len(candidates) >= max_cand:
                    break
                # Try both orderings.
                for a_idx, b_idx in [(i, j), (j, i)]:
                    if len(candidates) >= max_cand:
                        break
                    g = self.compose_sequential(webs[a_idx], webs[b_idx])
                    _try_add(g, [a_idx, b_idx], "sequential")

        # -- 2. Pairwise parallel (tensor + optional Hadamard stitch) ---------
        if "parallel" in self.config.composition_modes:
            for i, j in tqdm(ordered_pairs, desc="Stage 4: Parallel compose", unit="pair"):
                if len(candidates) >= max_cand:
                    break

                # Pure parallel (tensor).
                g = self.compose_parallel(webs[i], webs[j])
                _try_add(g, [i, j], "parallel")

                if len(candidates) >= max_cand:
                    break

                # Parallel with Hadamard stitching.
                g_stitch = self.compose_parallel_stitch(
                    webs[i], webs[j], n_hadamard_edges=1
                )
                _try_add(g_stitch, [i, j], "parallel_stitch")

        # -- 3. Phase perturbation on successful compositions -----------------
        phase_perturb_count = 0
        max_perturb = min(max_cand // 4, len(candidates))
        base_candidates = list(candidates)  # snapshot
        for base_cand in base_candidates:
            if len(candidates) >= max_cand:
                break
            if phase_perturb_count >= max_perturb:
                break

            base_g = zx.Graph.from_json(base_cand.graph_json)
            perturbed_g = self.perturb_phases(base_g, rate=0.3)

            n_qubits = len(perturbed_g.inputs()) if perturbed_g.inputs() else 0
            n_outputs = len(perturbed_g.outputs()) if perturbed_g.outputs() else 0
            if n_qubits > _MAX_QUBITS or n_qubits == 0 or n_qubits != n_outputs:
                continue
            if n_qubits < min_qubits:
                continue
            n_spiders = sum(
                1 for v in perturbed_g.vertices()
                if perturbed_g.type(v) != _VT_BOUNDARY
            )
            cand = CandidateAlgorithm(
                candidate_id=f"cand_{cand_idx:04d}",
                graph_json=perturbed_g.to_json(),
                component_web_ids=base_cand.component_web_ids,
                composition_type="phase_perturb",
                n_qubits=n_qubits,
                n_spiders=n_spiders,
                source_families=base_cand.source_families,
                is_cross_family=base_cand.is_cross_family,
            )
            candidates.append(cand)
            cand_idx += 1
            phase_perturb_count += 1

        # -- 4. Triple sequential (A -> B -> C) -------------------------------
        if "sequential" in self.config.composition_modes and len(webs) >= 3:
            all_triples = list(combinations(range(min(len(webs), 50)), 3))
            if len(all_triples) > max_cand * 5:
                triples = self.rng.sample(
                    all_triples, min(len(all_triples), max_cand * 5)
                )
            else:
                triples = list(all_triples)
                self.rng.shuffle(triples)
            for i, j, k in triples:
                if len(candidates) >= max_cand:
                    break
                g_ab = self.compose_sequential(webs[i], webs[j])
                if g_ab is None:
                    continue
                # Wrap intermediate result as a temporary ZXWeb.
                temp_web = ZXWeb(
                    web_id="__temp__",
                    graph_json=g_ab.to_json(),
                    n_inputs=len(g_ab.inputs()) if g_ab.inputs() else 0,
                    n_outputs=len(g_ab.outputs()) if g_ab.outputs() else 0,
                )
                g_abc = self.compose_sequential(temp_web, webs[k])
                _try_add(g_abc, [i, j, k], "triple_sequential")

        # -- 5. Target-guided composition -------------------------------------
        if self.config.guided and target_tasks:
            self._generate_guided_candidates(
                webs, target_tasks, candidates, cand_idx, min_qubits,
            )

        logger.info(
            "Generated %d candidates from %d webs "
            "(%d cross-family).",
            len(candidates),
            len(webs),
            sum(1 for c in candidates if c.is_cross_family),
        )
        return candidates

    # ------------------------------------------------------------------
    # Goal-directed composition
    # ------------------------------------------------------------------

    def _generate_guided_candidates(
        self,
        webs: list[ZXWeb],
        target_tasks: list[dict[str, Any]],
        candidates: list[CandidateAlgorithm],
        cand_idx: int,
        min_qubits: int,
    ) -> None:
        """Generate candidates guided by target task descriptions.

        For each target task, find webs whose qubit count and family
        provenance match the target, and try composing them together.
        This focuses the search on compositions that are likely to
        produce circuits similar to (but potentially more efficient
        than) known algorithms.

        Parameters
        ----------
        webs:
            The ZX-Webs to compose.
        target_tasks:
            List of target descriptions with ``n_qubits`` and optionally
            ``family``.
        candidates:
            Existing candidate list (mutated in place).
        cand_idx:
            Starting candidate index counter.
        min_qubits:
            Minimum qubit count for valid candidates.
        """
        max_cand = self.config.max_candidates

        # Index webs by family for fast lookup.
        family_index: dict[str, list[int]] = defaultdict(list)
        for idx, web in enumerate(webs):
            for fam in web.source_families:
                family_index[fam].append(idx)

        # Index webs by n_inputs for qubit-count matching.
        qubit_index: dict[int, list[int]] = defaultdict(list)
        for idx, web in enumerate(webs):
            qubit_index[web.n_inputs].append(idx)

        for task in target_tasks:
            if len(candidates) >= max_cand:
                break

            target_qubits = task.get("n_qubits", 0)
            target_family = task.get("family", "")

            if target_qubits < min_qubits:
                continue

            # Find webs that could contribute to the target qubit count.
            # For sequential: we need webs with matching I/O counts.
            matching_webs = qubit_index.get(target_qubits, [])

            # For the target's family, also consider webs from that family
            # composed with webs from other families (cross-pollination).
            family_webs = family_index.get(target_family, []) if target_family else []
            other_family_webs = [
                idx for idx, web in enumerate(webs)
                if target_family not in web.source_families
            ]

            # Strategy A: Sequential compose of matching-qubit webs.
            if "sequential" in self.config.composition_modes and len(matching_webs) >= 2:
                pairs = list(combinations(matching_webs, 2))
                self.rng.shuffle(pairs)
                for i, j in pairs[:max_cand // 4]:
                    if len(candidates) >= max_cand:
                        break
                    for a_idx, b_idx in [(i, j), (j, i)]:
                        if len(candidates) >= max_cand:
                            break
                        g = self.compose_sequential(webs[a_idx], webs[b_idx])
                        if g is None:
                            continue
                        cand = self._make_candidate(
                            g, webs, [a_idx, b_idx],
                            "guided_sequential", cand_idx, min_qubits,
                        )
                        if cand is not None:
                            candidates.append(cand)
                            cand_idx += 1

            # Strategy B: Cross-family sequential (family webs + other webs).
            if (
                "sequential" in self.config.composition_modes
                and family_webs
                and other_family_webs
            ):
                cross_pairs = [
                    (f_idx, o_idx)
                    for f_idx in family_webs
                    for o_idx in other_family_webs
                    if webs[f_idx].n_outputs == webs[o_idx].n_inputs
                ]
                self.rng.shuffle(cross_pairs)
                for f_idx, o_idx in cross_pairs[:max_cand // 4]:
                    if len(candidates) >= max_cand:
                        break
                    g = self.compose_sequential(webs[f_idx], webs[o_idx])
                    if g is None:
                        continue
                    cand = self._make_candidate(
                        g, webs, [f_idx, o_idx],
                        "guided_cross_family", cand_idx, min_qubits,
                    )
                    if cand is not None:
                        candidates.append(cand)
                        cand_idx += 1

            # Strategy C: Parallel compose to reach the target qubit count.
            if "parallel" in self.config.composition_modes:
                # Find pairs of webs whose combined qubit count matches target.
                target_parallel_pairs = [
                    (i, j)
                    for i, j in combinations(range(len(webs)), 2)
                    if webs[i].n_inputs + webs[j].n_inputs == target_qubits
                ]
                self.rng.shuffle(target_parallel_pairs)
                for i, j in target_parallel_pairs[:max_cand // 4]:
                    if len(candidates) >= max_cand:
                        break
                    g = self.compose_parallel(webs[i], webs[j])
                    if g is None:
                        continue
                    cand = self._make_candidate(
                        g, webs, [i, j],
                        "guided_parallel", cand_idx, min_qubits,
                    )
                    if cand is not None:
                        candidates.append(cand)
                        cand_idx += 1

    # ------------------------------------------------------------------
    # Parallel batch composition
    # ------------------------------------------------------------------

    def _compose_pair_batch(
        self,
        webs: list[ZXWeb],
        pairs: list[tuple[int, int]],
        mode: str,
    ) -> list[CandidateAlgorithm]:
        """Compose a batch of web pairs and return valid candidates.

        This method is designed to be called within a process pool worker.
        Each pair is tried and valid candidates (those under ``_MAX_QUBITS``)
        are returned.

        Parameters
        ----------
        webs:
            The full list of ZX-Webs (indexed by the pairs).
        pairs:
            List of ``(i, j)`` index pairs into *webs*.
        mode:
            ``"sequential"`` or ``"parallel"`` -- selects the composition
            strategy.

        Returns
        -------
        list[CandidateAlgorithm]
        """
        results: list[CandidateAlgorithm] = []
        for i, j in pairs:
            if mode == "sequential":
                # Try both orderings for sequential.
                for a_idx, b_idx in [(i, j), (j, i)]:
                    g = self.compose_sequential(webs[a_idx], webs[b_idx])
                    if g is not None:
                        n_qubits = len(g.inputs()) if g.inputs() else 0
                        n_outputs = len(g.outputs()) if g.outputs() else 0
                        if 0 < n_qubits <= _MAX_QUBITS and n_qubits == n_outputs:
                            n_spiders = sum(
                                1 for v in g.vertices()
                                if g.type(v) != _VT_BOUNDARY
                            )
                            families = _collect_families(webs, [a_idx, b_idx])
                            cross = _is_cross_family(webs, [a_idx, b_idx])
                            results.append(CandidateAlgorithm(
                                candidate_id="",  # assigned later
                                graph_json=g.to_json(),
                                component_web_ids=[
                                    webs[a_idx].web_id, webs[b_idx].web_id,
                                ],
                                composition_type="sequential",
                                n_qubits=n_qubits,
                                n_spiders=n_spiders,
                                source_families=families,
                                is_cross_family=cross,
                            ))
            elif mode == "parallel":
                # Pure parallel (tensor).
                g = self.compose_parallel(webs[i], webs[j])
                if g is not None:
                    n_qubits = len(g.inputs()) if g.inputs() else 0
                    n_outputs = len(g.outputs()) if g.outputs() else 0
                    if 0 < n_qubits <= _MAX_QUBITS and n_qubits == n_outputs:
                        n_spiders = sum(
                            1 for v in g.vertices()
                            if g.type(v) != _VT_BOUNDARY
                        )
                        families = _collect_families(webs, [i, j])
                        cross = _is_cross_family(webs, [i, j])
                        results.append(CandidateAlgorithm(
                            candidate_id="",
                            graph_json=g.to_json(),
                            component_web_ids=[
                                webs[i].web_id, webs[j].web_id,
                            ],
                            composition_type="parallel",
                            n_qubits=n_qubits,
                            n_spiders=n_spiders,
                            source_families=families,
                            is_cross_family=cross,
                        ))

                # Parallel with Hadamard stitching.
                g_stitch = self.compose_parallel_stitch(
                    webs[i], webs[j], n_hadamard_edges=1,
                )
                if g_stitch is not None:
                    n_qubits = len(g_stitch.inputs()) if g_stitch.inputs() else 0
                    n_outputs = len(g_stitch.outputs()) if g_stitch.outputs() else 0
                    if 0 < n_qubits <= _MAX_QUBITS and n_qubits == n_outputs:
                        n_spiders = sum(
                            1 for v in g_stitch.vertices()
                            if g_stitch.type(v) != _VT_BOUNDARY
                        )
                        families = _collect_families(webs, [i, j])
                        cross = _is_cross_family(webs, [i, j])
                        results.append(CandidateAlgorithm(
                            candidate_id="",
                            graph_json=g_stitch.to_json(),
                            component_web_ids=[
                                webs[i].web_id, webs[j].web_id,
                            ],
                            composition_type="parallel_stitch",
                            n_qubits=n_qubits,
                            n_spiders=n_spiders,
                            source_families=families,
                            is_cross_family=cross,
                        ))
        return results

    # ------------------------------------------------------------------
    # Internal helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_boundary_wires(g: zx.Graph) -> list[BoundaryWire]:
        """Build boundary wire records from a PyZX graph's inputs/outputs."""
        wires: list[BoundaryWire] = []
        input_set = set(g.inputs()) if g.inputs() else set()
        output_set = set(g.outputs()) if g.outputs() else set()

        for v in g.vertices():
            if g.type(v) != _VT_BOUNDARY:
                continue
            direction = "unknown"
            if v in input_set:
                direction = "input"
            elif v in output_set:
                direction = "output"

            for nb in g.neighbors(v):
                nb_type = g.type(nb)
                nb_phase = float(g.phase(nb))
                edge = g.edge(v, nb)
                etype = g.edge_type(edge)
                wires.append(
                    BoundaryWire(
                        internal_vertex=nb,
                        spider_type=nb_type,
                        spider_phase=nb_phase,
                        edge_type=etype,
                        direction=direction,
                    )
                )
        return wires


# ---------------------------------------------------------------------------
# Stage 4 entry point
# ---------------------------------------------------------------------------


def run_stage4(
    webs_dir: Path,
    output_dir: Path,
    config: ComposeConfig | None = None,
) -> list[CandidateAlgorithm]:
    """Run Stage 4: compose ZX-Webs into candidate algorithms.

    Workflow
    -------
    1. Load ZX-Webs from the Stage 3 output directory.
    2. Generate candidate algorithms via composition strategies.
    3. Persist the candidates and a manifest under *output_dir*.

    Parameters
    ----------
    webs_dir:
        Directory containing Stage 3 outputs (``manifest.json`` and
        ``webs/*.json``).
    output_dir:
        Where Stage 4 artefacts will be written.
    config:
        Composition parameters.  Falls back to ``ComposeConfig()`` defaults
        when *None*.

    Returns
    -------
    list[CandidateAlgorithm]
    """
    if config is None:
        config = ComposeConfig()

    # -- 1. Load webs from Stage 3 manifest -----------------------------------
    manifest = load_manifest(webs_dir)
    if not manifest:
        logger.warning(
            "Stage 3 manifest at %s is empty -- nothing to compose.", webs_dir
        )
        return []

    webs: list[ZXWeb] = []
    for entry in manifest:
        web_path = Path(entry["web_path"])
        if not web_path.exists():
            logger.warning("Web file not found: %s -- skipping.", web_path)
            continue
        web_data = load_json(web_path)
        webs.append(ZXWeb.from_dict(web_data))

    if not webs:
        logger.warning("No valid webs loaded from %s.", webs_dir)
        return []

    logger.info("Loaded %d webs for composition.", len(webs))

    # -- Build target tasks from config if guided mode is enabled. ------------
    target_tasks: list[dict[str, Any]] | None = None
    if config.guided and config.target_qubit_counts:
        target_tasks = [
            {"n_qubits": nq} for nq in config.target_qubit_counts
        ]

    # -- 2. Generate candidates -----------------------------------------------
    stitcher = Stitcher(config)
    candidates = stitcher.generate_candidates(webs, target_tasks=target_tasks)

    # -- 3. Persist results ---------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    for cand in candidates:
        cand_path = candidates_dir / f"{cand.candidate_id}.json"
        save_json(cand.to_dict(), cand_path)

        manifest_entries.append(
            {
                "candidate_id": cand.candidate_id,
                "candidate_path": str(cand_path),
                "composition_type": cand.composition_type,
                "component_web_ids": cand.component_web_ids,
                "n_qubits": cand.n_qubits,
                "n_spiders": cand.n_spiders,
                "source_families": cand.source_families,
                "is_cross_family": cand.is_cross_family,
            }
        )

    save_manifest(manifest_entries, output_dir)

    logger.info(
        "Stage 4 complete -- %d candidates written to %s",
        len(candidates),
        output_dir,
    )
    return candidates
