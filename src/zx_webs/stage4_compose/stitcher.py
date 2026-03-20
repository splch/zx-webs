"""Stage 4 composition engine -- stitch ZX-Webs into candidate algorithms.

The :class:`Stitcher` takes a collection of :class:`ZXWeb` objects and
produces :class:`CandidateAlgorithm` instances by composing webs in three
modes:

1. **Pairwise sequential** -- connect the outputs of web A to the inputs of
   web B (wire counts must match).
2. **Pairwise parallel** -- tensor product of two webs (always succeeds).
3. **Triple sequential** -- chain three webs A -> B -> C.

The :func:`run_stage4` entry point loads webs from Stage 3 output, runs
composition, and persists the results.
"""
from __future__ import annotations

import logging
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import pyzx as zx

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

# Maximum qubit count for generated candidates.
_MAX_QUBITS = 10


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
    # Leaf-vertex heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _find_output_leaves(g: zx.Graph) -> list[BoundaryWire]:
        """Heuristically identify output boundary wires on *g*.

        Leaf vertices (degree 1, non-boundary type) in the *later* rows are
        treated as outputs.  If row information is unavailable, all leaf
        vertices in the upper half (by sorted index) are used.
        """
        return Stitcher._find_leaves(g, direction="output")

    @staticmethod
    def _find_input_leaves(g: zx.Graph) -> list[BoundaryWire]:
        """Heuristically identify input boundary wires on *g*."""
        return Stitcher._find_leaves(g, direction="input")

    @staticmethod
    def _find_leaves(g: zx.Graph, direction: str) -> list[BoundaryWire]:
        """Find leaf vertices that serve as boundary wires in *direction*.

        Strategy:
        1. Collect all non-boundary leaf vertices (degree 1).
        2. Sort by row position (ascending = earlier = input-side).
        3. If ``direction == "input"`` return the first half; if ``"output"``
           return the second half.  For odd counts the middle vertex goes to
           the input side.
        """
        leaves: list[tuple[float, int]] = []  # (row, vertex_id)
        for v in g.vertices():
            if g.type(v) == _VT_BOUNDARY:
                continue
            if len(list(g.neighbors(v))) == 1:
                leaves.append((g.row(v), v))

        if not leaves:
            return []

        leaves.sort()

        if direction == "input":
            selected = leaves[: len(leaves) // 2] if len(leaves) > 1 else leaves
        else:
            selected = leaves[len(leaves) // 2:] if len(leaves) > 1 else []

        wires: list[BoundaryWire] = []
        for _row, v in selected:
            wires.append(
                BoundaryWire(
                    internal_vertex=v,
                    spider_type=g.type(v),
                    spider_phase=float(g.phase(v)),
                    edge_type=1,  # default simple
                    direction=direction,
                )
            )
        return wires

    # ------------------------------------------------------------------
    # Sequential composition
    # ------------------------------------------------------------------

    def compose_sequential(
        self, web_a: ZXWeb, web_b: ZXWeb
    ) -> zx.Graph | None:
        """Compose two webs sequentially (output of A -> input of B).

        Returns ``None`` when wire counts are incompatible or zero.

        Implementation
        --------------
        1. Build separate PyZX graphs for each web.
        2. Copy all vertices and edges into a single new graph.
        3. Connect A's output boundary wires to B's input boundary wires.
        4. Set the combined graph's inputs from A and outputs from B.
        """
        g_a = web_a.to_pyzx_graph()
        g_b = web_b.to_pyzx_graph()

        # -- Resolve boundary wires -------------------------------------------
        a_outputs = [bw for bw in web_a.boundary_wires if bw.direction == "output"]
        b_inputs = [bw for bw in web_b.boundary_wires if bw.direction == "input"]

        if not a_outputs:
            a_outputs = self._find_output_leaves(g_a)
        if not b_inputs:
            b_inputs = self._find_input_leaves(g_b)

        if len(a_outputs) != len(b_inputs):
            return None
        if len(a_outputs) == 0:
            return None

        # -- Build combined graph ---------------------------------------------
        combined = zx.Graph()

        a_map: dict[int, int] = {}
        for v in g_a.vertices():
            new_v = combined.add_vertex(
                ty=g_a.type(v),
                phase=g_a.phase(v),
                row=g_a.row(v),
                qubit=g_a.qubit(v),
            )
            a_map[v] = new_v

        for e in g_a.edges():
            s, t = g_a.edge_st(e)
            combined.add_edge(
                (a_map[s], a_map[t]), edgetype=g_a.edge_type(e)
            )

        b_map: dict[int, int] = {}
        for v in g_b.vertices():
            new_v = combined.add_vertex(
                ty=g_b.type(v),
                phase=g_b.phase(v),
                row=g_b.row(v) + max((g_a.row(u) for u in g_a.vertices()), default=0) + 1,
                qubit=g_b.qubit(v),
            )
            b_map[v] = new_v

        for e in g_b.edges():
            s, t = g_b.edge_st(e)
            combined.add_edge(
                (b_map[s], b_map[t]), edgetype=g_b.edge_type(e)
            )

        # -- Connect boundary wires -------------------------------------------
        for a_bw, b_bw in zip(a_outputs, b_inputs):
            a_vid = a_map.get(a_bw.internal_vertex, a_bw.internal_vertex)
            b_vid = b_map.get(b_bw.internal_vertex, b_bw.internal_vertex)
            etype = junction_edge_type(a_bw, b_bw)
            combined.add_edge((a_vid, b_vid), edgetype=etype)

        # -- Set combined inputs/outputs if available -------------------------
        a_input_bws = [bw for bw in web_a.boundary_wires if bw.direction == "input"]
        b_output_bws = [bw for bw in web_b.boundary_wires if bw.direction == "output"]

        # Try to propagate boundary vertex information from the original graphs.
        a_in_vids = tuple(a_map[v] for v in g_a.inputs()) if g_a.inputs() else ()
        b_out_vids = tuple(b_map[v] for v in g_b.outputs()) if g_b.outputs() else ()

        if a_in_vids:
            combined.set_inputs(a_in_vids)
        if b_out_vids:
            combined.set_outputs(b_out_vids)

        return combined

    # ------------------------------------------------------------------
    # Parallel composition
    # ------------------------------------------------------------------

    def compose_parallel(
        self, web_a: ZXWeb, web_b: ZXWeb
    ) -> zx.Graph:
        """Compose two webs in parallel (tensor product).

        Always succeeds.  The combined graph has both sets of vertices and
        edges with no connecting edges between them.  Qubit indices for B
        are shifted to avoid collision.
        """
        g_a = web_a.to_pyzx_graph()
        g_b = web_b.to_pyzx_graph()

        combined = zx.Graph()

        # -- Copy web A -------------------------------------------------------
        a_map: dict[int, int] = {}
        for v in g_a.vertices():
            new_v = combined.add_vertex(
                ty=g_a.type(v),
                phase=g_a.phase(v),
                row=g_a.row(v),
                qubit=g_a.qubit(v),
            )
            a_map[v] = new_v

        for e in g_a.edges():
            s, t = g_a.edge_st(e)
            combined.add_edge(
                (a_map[s], a_map[t]), edgetype=g_a.edge_type(e)
            )

        # -- Copy web B with qubit offset ------------------------------------
        max_qubit_a = max(
            (g_a.qubit(v) for v in g_a.vertices()), default=-1
        )
        qubit_offset = max_qubit_a + 1

        b_map: dict[int, int] = {}
        for v in g_b.vertices():
            new_v = combined.add_vertex(
                ty=g_b.type(v),
                phase=g_b.phase(v),
                row=g_b.row(v),
                qubit=g_b.qubit(v) + qubit_offset,
            )
            b_map[v] = new_v

        for e in g_b.edges():
            s, t = g_b.edge_st(e)
            combined.add_edge(
                (b_map[s], b_map[t]), edgetype=g_b.edge_type(e)
            )

        # -- Set combined inputs/outputs --------------------------------------
        a_ins = tuple(a_map[v] for v in g_a.inputs()) if g_a.inputs() else ()
        b_ins = tuple(b_map[v] for v in g_b.inputs()) if g_b.inputs() else ()
        a_outs = tuple(a_map[v] for v in g_a.outputs()) if g_a.outputs() else ()
        b_outs = tuple(b_map[v] for v in g_b.outputs()) if g_b.outputs() else ()

        combined.set_inputs(a_ins + b_ins)
        combined.set_outputs(a_outs + b_outs)

        return combined

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self, webs: list[ZXWeb]
    ) -> list[CandidateAlgorithm]:
        """Generate candidate algorithms from a list of ZX-Webs.

        Strategies employed:

        1. **Pairwise sequential** -- try all ordered pairs ``(A, B)`` where
           A's output count matches B's input count.
        2. **Pairwise parallel** -- try all unordered pairs.
        3. **Triple sequential** -- chain ``A -> B -> C`` for compatible
           triples.

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

        # Helper to build a CandidateAlgorithm from a composed graph.
        def _make_candidate(
            graph: zx.Graph,
            web_ids: list[str],
            comp_type: str,
        ) -> CandidateAlgorithm | None:
            n_qubits = len(graph.inputs()) if graph.inputs() else 0
            if n_qubits > _MAX_QUBITS:
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
                pairs = all_pairs
                self.rng.shuffle(pairs)
            for i, j in pairs:
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

        # -- 2. Pairwise parallel ---------------------------------------------
        if "parallel" in self.config.composition_modes:
            all_pairs = list(combinations(range(len(webs)), 2))
            if len(all_pairs) > max_cand * 10:
                pairs = self.rng.sample(all_pairs, min(len(all_pairs), max_cand * 10))
            else:
                pairs = all_pairs
                self.rng.shuffle(pairs)
            for i, j in pairs:
                if len(candidates) >= max_cand:
                    break
                g = self.compose_parallel(webs[i], webs[j])
                cand = _make_candidate(
                    g,
                    [webs[i].web_id, webs[j].web_id],
                    "parallel",
                )
                if cand is not None:
                    candidates.append(cand)

        # -- 3. Triple sequential ---------------------------------------------
        if "sequential" in self.config.composition_modes and len(webs) >= 3:
            all_triples = list(combinations(range(min(len(webs), 50)), 3))
            if len(all_triples) > max_cand * 5:
                triples = self.rng.sample(all_triples, min(len(all_triples), max_cand * 5))
            else:
                triples = all_triples
                self.rng.shuffle(triples)
            for i, j, k in triples:
                if len(candidates) >= max_cand:
                    break
                g_ab = self.compose_sequential(webs[i], webs[j])
                if g_ab is None:
                    continue
                # Wrap the intermediate result as a temporary ZXWeb for
                # further composition.
                temp_web = ZXWeb(
                    web_id="__temp__",
                    graph_json=g_ab.to_json(),
                    boundary_wires=self._extract_boundary_wires(g_ab),
                    n_inputs=len(g_ab.inputs()) if g_ab.inputs() else 0,
                    n_outputs=len(g_ab.outputs()) if g_ab.outputs() else 0,
                )
                g_abc = self.compose_sequential(temp_web, webs[k])
                if g_abc is not None:
                    cand = _make_candidate(
                        g_abc,
                        [webs[i].web_id, webs[j].web_id, webs[k].web_id],
                        "hybrid",
                    )
                    if cand is not None:
                        candidates.append(cand)

        logger.info(
            "Generated %d candidates from %d webs.", len(candidates), len(webs)
        )
        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
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
    2. Generate candidate algorithms via combinatorial composition.
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
