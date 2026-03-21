"""Stage 4 composition engine -- stitch ZX-Webs into candidate algorithms.

The :class:`Stitcher` takes a collection of :class:`ZXWeb` objects and
produces :class:`CandidateAlgorithm` instances by composing webs using
three strategies that preserve gflow and enable high circuit extraction
rates:

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

The :func:`run_stage4` entry point loads webs from Stage 3 output, runs
composition, and persists the results.
"""
from __future__ import annotations

import logging
import os
import random
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
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self, webs: list[ZXWeb]
    ) -> list[CandidateAlgorithm]:
        """Generate candidate algorithms from a list of ZX-Webs.

        Strategies employed:

        1. **Sequential compose** -- try ordered pairs ``(A, B)`` where
           A's output count matches B's input count.  Uses PyZX native
           ``compose()``.
        2. **Parallel tensor** -- tensor product of pairs, optionally with
           Hadamard stitching for novelty.
        3. **Phase perturbation** -- take successful compositions and
           create variants with perturbed phases.

        Candidates with more than :data:`_MAX_QUBITS` qubits are dropped.
        The total number of candidates is capped by
        ``config.max_candidates``.

        Parameters
        ----------
        webs:
            The ZX-Webs to compose.

        Returns
        -------
        list[CandidateAlgorithm]
        """
        candidates: list[CandidateAlgorithm] = []
        max_cand = self.config.max_candidates
        cand_idx = 0

        min_qubits = self.config.min_compose_qubits

        # Helper to build a CandidateAlgorithm from a composed graph.
        def _make_candidate(
            graph: zx.Graph,
            web_ids: list[str],
            comp_type: str,
        ) -> CandidateAlgorithm | None:
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
            nonlocal cand_idx
            cand = CandidateAlgorithm(
                candidate_id=f"cand_{cand_idx:04d}",
                graph_json=graph.to_json(),
                component_web_ids=web_ids,
                composition_type=comp_type,
                n_qubits=n_qubits,
                n_spiders=n_spiders,
            )
            cand_idx += 1
            return cand

        # -- 1. Pairwise sequential -------------------------------------------
        if "sequential" in self.config.composition_modes:
            all_pairs = list(combinations(range(len(webs)), 2))
            # Sample to avoid OOM on large web sets
            if len(all_pairs) > max_cand * 10:
                pairs = self.rng.sample(all_pairs, min(len(all_pairs), max_cand * 10))
            else:
                pairs = list(all_pairs)
                self.rng.shuffle(pairs)
            for i, j in tqdm(pairs, desc="Stage 4: Sequential compose", unit="pair"):
                if len(candidates) >= max_cand:
                    break
                # Try both orderings.
                for a_idx, b_idx in [(i, j), (j, i)]:
                    if len(candidates) >= max_cand:
                        break
                    g = self.compose_sequential(webs[a_idx], webs[b_idx])
                    if g is not None:
                        cand = _make_candidate(
                            g,
                            [webs[a_idx].web_id, webs[b_idx].web_id],
                            "sequential",
                        )
                        if cand is not None:
                            candidates.append(cand)

        # -- 2. Pairwise parallel (tensor + optional Hadamard stitch) ---------
        if "parallel" in self.config.composition_modes:
            all_pairs = list(combinations(range(len(webs)), 2))
            if len(all_pairs) > max_cand * 10:
                pairs = self.rng.sample(all_pairs, min(len(all_pairs), max_cand * 10))
            else:
                pairs = list(all_pairs)
                self.rng.shuffle(pairs)
            for i, j in tqdm(pairs, desc="Stage 4: Parallel compose", unit="pair"):
                if len(candidates) >= max_cand:
                    break

                # Pure parallel (tensor).
                g = self.compose_parallel(webs[i], webs[j])
                if g is not None:
                    cand = _make_candidate(
                        g,
                        [webs[i].web_id, webs[j].web_id],
                        "parallel",
                    )
                    if cand is not None:
                        candidates.append(cand)

                if len(candidates) >= max_cand:
                    break

                # Parallel with Hadamard stitching.
                g_stitch = self.compose_parallel_stitch(
                    webs[i], webs[j], n_hadamard_edges=1
                )
                if g_stitch is not None:
                    cand = _make_candidate(
                        g_stitch,
                        [webs[i].web_id, webs[j].web_id],
                        "parallel_stitch",
                    )
                    if cand is not None:
                        candidates.append(cand)

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
            cand = _make_candidate(
                perturbed_g,
                base_cand.component_web_ids,
                "phase_perturb",
            )
            if cand is not None:
                candidates.append(cand)
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
                if g_abc is not None:
                    cand = _make_candidate(
                        g_abc,
                        [webs[i].web_id, webs[j].web_id, webs[k].web_id],
                        "triple_sequential",
                    )
                    if cand is not None:
                        candidates.append(cand)

        logger.info(
            "Generated %d candidates from %d webs.", len(candidates), len(webs)
        )
        return candidates

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
                            results.append(CandidateAlgorithm(
                                candidate_id="",  # assigned later
                                graph_json=g.to_json(),
                                component_web_ids=[
                                    webs[a_idx].web_id, webs[b_idx].web_id,
                                ],
                                composition_type="sequential",
                                n_qubits=n_qubits,
                                n_spiders=n_spiders,
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
                        results.append(CandidateAlgorithm(
                            candidate_id="",
                            graph_json=g.to_json(),
                            component_web_ids=[
                                webs[i].web_id, webs[j].web_id,
                            ],
                            composition_type="parallel",
                            n_qubits=n_qubits,
                            n_spiders=n_spiders,
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
                        results.append(CandidateAlgorithm(
                            candidate_id="",
                            graph_json=g_stitch.to_json(),
                            component_web_ids=[
                                webs[i].web_id, webs[j].web_id,
                            ],
                            composition_type="parallel_stitch",
                            n_qubits=n_qubits,
                            n_spiders=n_spiders,
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

    # -- 2. Generate candidates -----------------------------------------------
    stitcher = Stitcher(config)
    candidates = stitcher.generate_candidates(webs)

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
            }
        )

    save_manifest(manifest_entries, output_dir)

    logger.info(
        "Stage 4 complete -- %d candidates written to %s",
        len(candidates),
        output_dir,
    )
    return candidates
