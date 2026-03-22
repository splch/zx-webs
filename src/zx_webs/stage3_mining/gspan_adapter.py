"""Adapter that wraps gSpan mining for use with PyZX graphs.

This module provides :class:`GSpanAdapter`, which:

1. Encodes a corpus of PyZX graphs into gSpan's integer-labelled text format.
2. Runs the gSpan frequent sub-graph mining algorithm.
3. Converts the results back into PyZX ``Graph`` objects wrapped as
   :class:`GSpanResult` records.

The primary backend is **submine** (a C++ gSpan implementation).  When
submine is not installed, we fall back to the pure-Python ``gspan-mining``
library with a silent subclass that captures results without printing.

**Performance notes**:

- The C++ ``mine_from_string`` call holds the Python GIL for the entire
  computation.  ``signal.alarm`` / ``SIGALRM`` cannot interrupt it, so
  Python-level timeouts are ineffective.  We use subprocess-based isolation
  to enforce hard timeouts when ``mining_timeout`` is set.
- Memory usage scales with ``(n_graphs * max_vertices * n_labels)``.
  With 300 graphs and ``max_vertices >= 6``, peak RSS can exceed 2 GB.
  On systems with per-user cgroup memory limits (e.g. SLURM head nodes),
  this causes swap-thrashing and apparent hangs.  Always run heavy mining
  on compute nodes with adequate RAM.
"""
from __future__ import annotations

import collections
import copy
import logging
import multiprocessing as mp
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyzx as zx
from tqdm import tqdm

from zx_webs.config import MiningConfig
from zx_webs.stage3_mining.graph_encoder import (
    ZXLabelEncoder,
    pyzx_graphs_to_gspan_file,
)

logger = logging.getLogger(__name__)


def _check_memory_environment() -> str | None:
    """Check for memory constraints that could cause mining to hang.

    Returns a warning string if problems are detected, or None if OK.
    Checks:
    - cgroup v2 memory limits (common on SLURM head nodes)
    - System swap exhaustion (causes allocation failures and hangs)
    """
    warnings = []

    # Check cgroup memory limit.
    try:
        cgroup_path = Path("/proc/self/cgroup")
        if cgroup_path.exists():
            cg_text = cgroup_path.read_text().strip()
            parts = cg_text.split("::")
            if len(parts) >= 2:
                cg_dir = Path("/sys/fs/cgroup") / parts[-1].lstrip("/")
                mem_max = cg_dir / "memory.max"
                if mem_max.exists():
                    val = mem_max.read_text().strip()
                    if val != "max":
                        limit_mb = int(val) / (1024 * 1024)
                        if limit_mb < 4096:
                            warnings.append(
                                f"cgroup memory limit is {limit_mb:.0f} MB"
                            )
    except Exception:
        pass

    # Check system swap availability.
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            content = meminfo.read_text()
            swap_total = swap_free = None
            for line in content.splitlines():
                if line.startswith("SwapTotal:"):
                    swap_total = int(line.split()[1])  # kB
                elif line.startswith("SwapFree:"):
                    swap_free = int(line.split()[1])  # kB
            if swap_total and swap_free is not None:
                swap_used_pct = 100 * (1 - swap_free / swap_total) if swap_total > 0 else 0
                if swap_used_pct > 90:
                    warnings.append(
                        f"system swap is {swap_used_pct:.0f}% full "
                        f"({swap_free // 1024} MB free of {swap_total // 1024} MB)"
                    )
    except Exception:
        pass

    if warnings:
        return "; ".join(warnings)
    return None


# Try to import the C++ backend from submine.
try:
    from submine.algorithms import gspan_cpp as _gspan_cpp

    _HAS_SUBMINE = True
except ImportError:
    _gspan_cpp = None  # type: ignore[assignment]
    _HAS_SUBMINE = False

# Import the pure-Python fallback (gspan-mining).
try:
    from gspan_mining.gspan import DFScode, DFSedge, PDFS, Projected, gSpan as _GSpanAlgo
    from gspan_mining.gspan import record_timestamp

    _HAS_GSPAN_MINING = True
except ImportError:
    _HAS_GSPAN_MINING = False


# ---------------------------------------------------------------------------
# Subprocess worker for timeout-safe mining
# ---------------------------------------------------------------------------


def _submine_worker(
    gspan_data: str,
    minsup: int,
    minv: int,
    maxv: int,
    result_queue: mp.Queue,
) -> None:
    """Run gSpan mining in a subprocess.

    Must be a module-level function for ``multiprocessing`` pickling.
    Re-imports submine in the child to avoid issues with fork-inherited
    C extension state.
    """
    # Re-import to get a clean C++ module state in the subprocess.
    from submine.algorithms import gspan_cpp as cpp

    raw = cpp.mine_from_string(
        gspan_data,
        minsup=minsup,
        maxpat_min=minv,
        maxpat_max=maxv,
        directed=False,
        where=True,
    )
    result_queue.put(raw)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GSpanResult:
    """A single frequent sub-graph discovered by gSpan.

    Attributes
    ----------
    dfscode:
        The canonical DFS code (a ``DFScode`` object from gspan-mining, or
        *None* when the submine C++ backend is used).
    gspan_graph:
        The sub-graph as a ``gspan_mining.graph.Graph`` (integer/string
        labels), or *None* when the submine backend is used.
    support:
        Number of input graphs in which this sub-graph appears.
    source_graph_ids:
        Indices of the input graphs that contain an embedding of this
        sub-graph.
    submine_dict:
        When the submine backend is used, the raw result dict containing
        ``nodes``, ``node_labels``, ``edges``, ``edge_labels``, ``support``,
        and ``graph_ids``.
    """

    dfscode: Any = None  # DFScode or None
    gspan_graph: Any = None  # gspan_mining.graph.Graph or None
    support: int = 0
    source_graph_ids: list[int] = field(default_factory=list)
    submine_dict: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Silent gSpan subclass -- captures results without printing
# (only defined when the pure-Python gspan-mining library is available)
# ---------------------------------------------------------------------------

if _HAS_GSPAN_MINING:

    class _SilentGSpan(_GSpanAlgo):
        """gSpan subclass that silently captures mining results.

        The upstream ``_report`` method prints to stdout and relies on the
        deprecated ``DataFrame.append``.  We override it to store each frequent
        sub-graph with its support and source graph IDs in ``captured_results``.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.captured_results: list[GSpanResult] = []

        @record_timestamp
        def run(self) -> None:
            """Override to add tqdm progress over root edge labels."""
            self._read_graphs()
            self._generate_1edge_frequent_subgraphs()
            if self._max_num_vertices < 2:
                return
            root = collections.defaultdict(Projected)
            for g in self.graphs.values():
                for vid, v in g.vertices.items():
                    edges = self._get_forward_root_edges(g, vid)
                    for e in edges:
                        root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                            PDFS(g.gid, e, None)
                        )

            for vevlb, projected in tqdm(
                root.items(), desc="Stage 3: gSpan mining", unit="edge-label"
            ):
                self._DFScode.append(DFSedge(0, 1, vevlb))
                self._subgraph_mining(projected)
                self._DFScode.pop()

        def _report(self, projected: Any) -> None:  # type: ignore[override]
            """Override to silently record the current DFScode + metadata."""
            dfscode = copy.copy(self._DFScode)
            self._frequent_subgraphs.append(dfscode)

            if dfscode.get_num_vertices() < self._min_num_vertices:
                return

            graph = dfscode.to_graph(
                gid=next(self._counter),
                is_undirected=self._is_undirected,
            )

            source_ids = sorted(set(p.gid for p in projected))

            self.captured_results.append(
                GSpanResult(
                    dfscode=dfscode,
                    gspan_graph=graph,
                    support=self._support,
                    source_graph_ids=source_ids,
                )
            )


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class GSpanAdapter:
    """Wrap the gspan-mining library for use with PyZX graphs.

    Parameters
    ----------
    config:
        Mining parameters (min support, vertex bounds, phase discretisation).
    """

    def __init__(self, config: MiningConfig) -> None:
        self.config = config
        self.encoder = ZXLabelEncoder(
            phase_bins=config.phase_discretization,
            include_phase=config.include_phase_in_label,
        )

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def mine(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Run gSpan mining on a corpus of ZX-diagrams.

        Steps
        -----
        1. Encode all PyZX graphs into the gSpan text format.
        2. Write to a temporary file / string.
        3. Run the gSpan algorithm (submine C++ backend preferred,
           pure-Python ``gspan-mining`` as fallback).
        4. Return :class:`GSpanResult` objects for every frequent sub-graph.

        Parameters
        ----------
        graphs:
            List of simplified PyZX ``Graph`` instances.

        Returns
        -------
        list[GSpanResult]
            Frequent sub-graphs sorted by descending support.
        """
        if not graphs:
            return []

        if _HAS_SUBMINE:
            return self._mine_submine(graphs)
        if _HAS_GSPAN_MINING:
            return self._mine_python(graphs)
        raise RuntimeError(
            "No gSpan backend available. Install submine (recommended) or gspan-mining."
        )

    def _mine_submine(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Mine using the submine C++ backend.

        When ``mining_timeout`` is configured, mining runs in a subprocess
        so it can be killed if it exceeds the time limit.  This is necessary
        because the C++ code holds the GIL and ``signal.alarm`` cannot
        interrupt it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            gspan_path = Path(tmpdir) / "corpus.gspan"
            pyzx_graphs_to_gspan_file(graphs, gspan_path, self.encoder)
            gspan_data = gspan_path.read_text(encoding="utf-8")

        # Warn if memory constraints could cause hangs.
        mem_warning = _check_memory_environment()
        if mem_warning:
            logger.warning(
                "Memory constraint detected: %s. "
                "gSpan mining with %d graphs and max_vertices=%d may require "
                "several GB of RAM and could hang due to swap-thrashing. "
                "Consider running on a SLURM compute node with adequate RAM, "
                "or increasing min_support to reduce memory usage.",
                mem_warning, len(graphs), self.config.max_vertices,
            )

        timeout = getattr(self.config, "mining_timeout", 0)

        if timeout and timeout > 0:
            raw_results = self._mine_submine_subprocess(gspan_data, timeout)
        else:
            raw_results = self._mine_submine_inline(gspan_data)

        results: list[GSpanResult] = []
        for rd in raw_results:
            graph_ids = sorted(set(rd["graph_ids"]))
            results.append(
                GSpanResult(
                    support=rd["support"],
                    source_graph_ids=graph_ids,
                    submine_dict=rd,
                )
            )

        # Sort by descending support, then ascending number of nodes.
        results.sort(key=lambda r: (-r.support, len(r.submine_dict["nodes"])))
        return results

    def _mine_submine_inline(self, gspan_data: str) -> list[dict]:
        """Run submine in the current process (no timeout protection)."""
        logger.debug("Running submine C++ gSpan backend (inline).")
        return _gspan_cpp.mine_from_string(
            gspan_data,
            minsup=self.config.min_support,
            maxpat_min=self.config.min_vertices,
            maxpat_max=self.config.max_vertices,
            directed=False,
            where=True,
        )

    def _mine_submine_subprocess(self, gspan_data: str, timeout: int) -> list[dict]:
        """Run submine in a subprocess with a hard timeout.

        The C++ ``mine_from_string`` holds the GIL, making it impossible
        to interrupt via Python signals.  Running in a subprocess allows
        us to ``kill()`` it if the timeout is exceeded.

        **Important**: we read from the result queue *before* calling
        ``proc.join()`` to avoid a pipe-buffer deadlock.  The child
        blocks in ``Queue.put()`` if the pipe buffer fills up, and the
        parent blocks in ``proc.join()`` waiting for the child to exit
        -- classic multiprocessing deadlock (see Python docs on
        ``multiprocessing.Queue``).
        """
        import queue as _queue_mod

        logger.info(
            "Running submine C++ gSpan backend in subprocess "
            "(timeout=%ds, minsup=%d, max_v=%d).",
            timeout, self.config.min_support, self.config.max_vertices,
        )

        result_queue: mp.Queue = mp.Queue()
        proc = mp.Process(
            target=_submine_worker,
            args=(
                gspan_data,
                self.config.min_support,
                self.config.min_vertices,
                self.config.max_vertices,
                result_queue,
            ),
        )
        proc.start()

        # Read from queue BEFORE join to avoid pipe-buffer deadlock.
        # The child's Queue.put() blocks if the pipe is full, so the
        # parent must consume the result before the child can exit.
        result = None
        try:
            result = result_queue.get(timeout=timeout)
        except _queue_mod.Empty:
            pass

        # Now safe to join (child has finished or will be killed).
        proc.join(timeout=5)

        if proc.is_alive():
            logger.error(
                "submine mining exceeded %ds timeout -- killing subprocess. "
                "Try increasing min_support or decreasing max_vertices.",
                timeout,
            )
            proc.kill()
            proc.join()
            return []

        if result is not None:
            return result

        logger.error("submine subprocess exited without producing results.")
        return []

    def _mine_python(self, graphs: list[zx.Graph]) -> list[GSpanResult]:
        """Mine using the pure-Python gspan-mining fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gspan_path = Path(tmpdir) / "corpus.gspan"
            pyzx_graphs_to_gspan_file(graphs, gspan_path, self.encoder)

            gs = _SilentGSpan(
                database_file_name=str(gspan_path),
                min_support=self.config.min_support,
                min_num_vertices=self.config.min_vertices,
                max_num_vertices=self.config.max_vertices,
                is_undirected=True,
                verbose=False,
                visualize=False,
                where=False,
            )
            gs.run()

        # Sort by descending support, then ascending number of vertices.
        results = sorted(
            gs.captured_results,
            key=lambda r: (-r.support, r.dfscode.get_num_vertices()),
        )
        return results

    # ------------------------------------------------------------------
    # Lightweight metadata extraction (no PyZX graph construction)
    # ------------------------------------------------------------------

    def extract_metadata(self, result: GSpanResult) -> dict[str, Any]:
        """Extract lightweight metadata from a mining result.

        This extracts ``n_spiders``, ``n_inputs``, ``n_outputs``, and
        ``boundary_wires`` from the raw result *without* constructing a
        full PyZX ``Graph`` object.  This is O(V+E) in the submine dict
        with zero PyZX overhead, making it ~100x faster than building the
        graph and inspecting it.

        Parameters
        ----------
        result:
            A single gSpan mining result.

        Returns
        -------
        dict
            Keys: ``n_spiders``, ``n_inputs``, ``n_outputs``,
            ``boundary_wires`` (list of dicts with ``internal_vertex``,
            ``spider_type``, ``spider_phase``, ``edge_type``, ``direction``).
        """
        if result.submine_dict is not None:
            return self._extract_metadata_submine(result.submine_dict)
        return self._extract_metadata_gspan(result.gspan_graph)

    def _extract_metadata_submine(self, rd: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata directly from a submine result dict.

        When the pattern contains explicit boundary vertices (type 0),
        n_inputs and n_outputs are counted from the encoder labels.

        When no boundary vertices exist (purely interior pattern),
        _ensure_proper_boundaries will later add them based on leaf
        vertices.  We estimate the boundary counts here by counting
        leaf nodes (degree 1) and splitting them into input/output
        halves, mirroring that logic.  This keeps the metadata
        consistent with what the full construction would produce.
        """
        _VT_BOUNDARY = 0
        nodes = rd["nodes"]
        node_labels = rd["node_labels"]

        n_spiders = 0
        n_inputs = 0
        n_outputs = 0
        boundary_vids: set[int] = set()
        vlabel_map: dict[int, int] = {}

        for vid, vlabel in zip(nodes, node_labels):
            vlabel_map[vid] = vlabel
            vtype, _ = self.encoder.decode_vertex(vlabel)
            if vtype == _VT_BOUNDARY:
                boundary_vids.add(vid)
                if self.encoder.is_output_boundary(vlabel):
                    n_outputs += 1
                else:
                    n_inputs += 1
            else:
                n_spiders += 1

        # Build deduplicated edge list and adjacency.
        edges = rd["edges"]
        edge_labels = rd["edge_labels"]

        # Build adjacency for all nodes (needed for leaf detection).
        adjacency: dict[int, list[tuple[int, int]]] = {vid: [] for vid in vlabel_map}
        seen: set[tuple[int, int]] = set()
        for (src, dst), elabel in zip(edges, edge_labels):
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            seen.add(key)
            adjacency[src].append((dst, elabel))
            adjacency[dst].append((src, elabel))

        # Build boundary wires from boundary vertices.
        boundary_wires: list[dict[str, Any]] = []
        for bv in sorted(boundary_vids):
            bv_label = vlabel_map[bv]
            is_output = self.encoder.is_output_boundary(bv_label)
            direction = "output" if is_output else "input"
            for nb_vid, elabel in adjacency[bv]:
                nb_vlabel = vlabel_map[nb_vid]
                nb_vtype, nb_phase_bin = self.encoder.decode_vertex(nb_vlabel)
                nb_phase = 0.0
                if nb_phase_bin is not None and self.encoder.include_phase:
                    nb_phase = float(Fraction(2 * nb_phase_bin, self.encoder.phase_bins))
                etype = self.encoder.decode_edge(elabel)
                boundary_wires.append({
                    "internal_vertex": nb_vid,
                    "spider_type": nb_vtype,
                    "spider_phase": nb_phase,
                    "edge_type": etype,
                    "direction": direction,
                })

        # When no boundary vertices exist, estimate n_inputs/n_outputs
        # from leaf (degree-1) interior vertices.  This mirrors the logic
        # in _ensure_proper_boundaries which adds boundary vertices at
        # leaf positions.
        if not boundary_vids:
            interior_nodes = [vid for vid in vlabel_map if vid not in boundary_vids]
            leaf_count = sum(
                1 for vid in interior_nodes
                if len(adjacency[vid]) <= 1
            )
            if leaf_count == 0:
                # No leaves -- _ensure_proper_boundaries picks two extremal
                # vertices.  Estimate 1 input + 1 output.
                n_inputs = max(n_inputs, 1)
                n_outputs = max(n_outputs, 1)
            else:
                # Split leaves: half inputs, half outputs.  At minimum
                # 1 of each.
                half = max(leaf_count // 2, 1)
                n_inputs = max(n_inputs, half)
                n_outputs = max(n_outputs, max(leaf_count - half, 1))

        return {
            "n_spiders": n_spiders,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "boundary_wires": boundary_wires,
        }

    def _extract_metadata_gspan(self, gg: Any) -> dict[str, Any]:
        """Extract metadata from a gspan-mining Graph object."""
        _VT_BOUNDARY = 0
        n_spiders = 0
        n_inputs = 0
        n_outputs = 0
        boundary_vids: set[Any] = set()
        vlabel_map: dict[Any, int] = {}

        for vid, vertex in gg.vertices.items():
            vlabel = int(vertex.vlb) if isinstance(vertex.vlb, str) else vertex.vlb
            vlabel_map[vid] = vlabel
            vtype, _ = self.encoder.decode_vertex(vlabel)
            if vtype == _VT_BOUNDARY:
                boundary_vids.add(vid)
                if self.encoder.is_output_boundary(vlabel):
                    n_outputs += 1
                else:
                    n_inputs += 1
            else:
                n_spiders += 1

        boundary_wires: list[dict[str, Any]] = []
        for vid in sorted(boundary_vids):
            vlabel = vlabel_map[vid]
            is_output = self.encoder.is_output_boundary(vlabel)
            direction = "output" if is_output else "input"
            vertex = gg.vertices[vid]
            for nb_vid, edge in vertex.edges.items():
                nb_vlabel = vlabel_map.get(nb_vid)
                if nb_vlabel is None:
                    continue
                nb_vtype, nb_phase_bin = self.encoder.decode_vertex(nb_vlabel)
                nb_phase = 0.0
                if nb_phase_bin is not None and self.encoder.include_phase:
                    nb_phase = float(Fraction(2 * nb_phase_bin, self.encoder.phase_bins))
                elabel = int(edge.elb) if isinstance(edge.elb, str) else edge.elb
                etype = self.encoder.decode_edge(elabel)
                boundary_wires.append({
                    "internal_vertex": nb_vid,
                    "spider_type": nb_vtype,
                    "spider_phase": nb_phase,
                    "edge_type": etype,
                    "direction": direction,
                })

        # When no boundary vertices exist, estimate from leaf nodes.
        if not boundary_vids:
            leaf_count = sum(
                1 for vid, vertex in gg.vertices.items()
                if len(vertex.edges) <= 1 and vid not in boundary_vids
            )
            if leaf_count == 0:
                n_inputs = max(n_inputs, 1)
                n_outputs = max(n_outputs, 1)
            else:
                half = max(leaf_count // 2, 1)
                n_inputs = max(n_inputs, half)
                n_outputs = max(n_outputs, max(leaf_count - half, 1))

        return {
            "n_spiders": n_spiders,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "boundary_wires": boundary_wires,
        }

    # ------------------------------------------------------------------
    # Result -> PyZX conversion
    # ------------------------------------------------------------------

    def result_to_pyzx(self, result: GSpanResult) -> zx.Graph:
        """Convert a :class:`GSpanResult` to a PyZX ``Graph``.

        Vertex and edge labels are decoded using the same
        :class:`ZXLabelEncoder` that was used for encoding.

        Supports results from both the submine C++ backend (via
        ``submine_dict``) and the pure-Python gspan-mining library (via
        ``gspan_graph``).

        Parameters
        ----------
        result:
            A single gSpan mining result.

        Returns
        -------
        zx.Graph
            A new PyZX graph with correct spider types, phases, and edge types.
        """
        if result.submine_dict is not None:
            return self._submine_dict_to_pyzx(result.submine_dict)
        return self._gspan_graph_to_pyzx(result.gspan_graph)

    def _submine_dict_to_pyzx(self, rd: dict[str, Any]) -> zx.Graph:
        """Build a PyZX graph from a submine result dict.

        The dict contains:
        - ``nodes``: list of node ids (0-based)
        - ``node_labels``: list of integer labels (parallel to ``nodes``)
        - ``edges``: list of (src, dst) tuples (each undirected edge appears twice)
        - ``edge_labels``: list of integer labels (parallel to ``edges``)
        """
        _VT_BOUNDARY = 0
        g = zx.Graph()

        nodes = rd["nodes"]
        node_labels = rd["node_labels"]

        # Separate boundary and interior vertices.
        boundary_vids: list[int] = []
        interior_vids: list[int] = []
        vlabel_map: dict[int, int] = {}

        for vid, vlabel in zip(nodes, node_labels):
            vlabel_map[vid] = vlabel
            vtype, _ = self.encoder.decode_vertex(vlabel)
            if vtype == _VT_BOUNDARY:
                boundary_vids.append(vid)
            else:
                interior_vids.append(vid)

        # Map submine node id -> PyZX vertex id.
        sm_to_pyzx: dict[int, int] = {}

        # Add interior vertices first (row=1, sequential qubit assignment).
        for idx, vid in enumerate(interior_vids):
            vlabel = vlabel_map[vid]
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase = Fraction(2 * phase_bin, self.encoder.phase_bins)
            pv = g.add_vertex(ty=vtype, phase=phase, qubit=idx, row=1)
            sm_to_pyzx[vid] = pv

        # Add boundary vertices using the label to determine direction.
        input_qubit = 0
        output_qubit = 0
        input_pvs: list[int] = []
        output_pvs: list[int] = []

        for vid in boundary_vids:
            vlabel = vlabel_map[vid]
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase_val: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase_val = Fraction(2 * phase_bin, self.encoder.phase_bins)

            if self.encoder.is_output_boundary(vlabel):
                row = 2
                qubit = output_qubit
                output_qubit += 1
            else:
                row = 0
                qubit = input_qubit
                input_qubit += 1

            pv = g.add_vertex(ty=vtype, phase=phase_val, qubit=qubit, row=row)
            sm_to_pyzx[vid] = pv

            if self.encoder.is_output_boundary(vlabel):
                output_pvs.append(pv)
            else:
                input_pvs.append(pv)

        # Collect edges (submine returns each undirected edge twice, so dedupe).
        edges = rd["edges"]
        edge_labels = rd["edge_labels"]
        seen: set[tuple[int, int]] = set()
        for (src, dst), elabel in zip(edges, edge_labels):
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            seen.add(key)
            etype = self.encoder.decode_edge(elabel)
            pv_from = sm_to_pyzx[src]
            pv_to = sm_to_pyzx[dst]
            g.add_edge((pv_from, pv_to), edgetype=etype)

        # Set inputs/outputs if we identified any from the labels.
        if input_pvs:
            g.set_inputs(tuple(input_pvs))
        if output_pvs:
            g.set_outputs(tuple(output_pvs))

        return g

    def _gspan_graph_to_pyzx(self, gg: Any) -> zx.Graph:
        """Build a PyZX graph from a gspan-mining ``Graph`` object."""
        _VT_BOUNDARY = 0
        g = zx.Graph()

        # Separate boundary and non-boundary vertices so we can assign
        # meaningful row/qubit positions that enable correct input/output
        # assignment later in _ensure_proper_boundaries.
        boundary_vids: list[Any] = []
        interior_vids: list[Any] = []

        for vid, vertex in gg.vertices.items():
            vlabel = int(vertex.vlb) if isinstance(vertex.vlb, str) else vertex.vlb
            vtype, _ = self.encoder.decode_vertex(vlabel)
            if vtype == _VT_BOUNDARY:
                boundary_vids.append(vid)
            else:
                interior_vids.append(vid)

        # Map gspan vertex ids -> pyzx vertex ids.
        gspan_to_pyzx: dict[Any, int] = {}

        # Add interior vertices first (row=1, sequential qubit assignment).
        for idx, vid in enumerate(interior_vids):
            vertex = gg.vertices[vid]
            vlabel = int(vertex.vlb) if isinstance(vertex.vlb, str) else vertex.vlb
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase = Fraction(2 * phase_bin, self.encoder.phase_bins)
            pv = g.add_vertex(ty=vtype, phase=phase, qubit=idx, row=1)
            gspan_to_pyzx[vid] = pv

        # Add boundary vertices using the label to determine direction.
        input_qubit = 0
        output_qubit = 0
        input_pvs: list[int] = []
        output_pvs: list[int] = []

        for vid in boundary_vids:
            vertex = gg.vertices[vid]
            vlabel = int(vertex.vlb) if isinstance(vertex.vlb, str) else vertex.vlb
            vtype, phase_bin = self.encoder.decode_vertex(vlabel)
            phase_val: Fraction | int = 0
            if phase_bin is not None and self.encoder.include_phase:
                phase_val = Fraction(2 * phase_bin, self.encoder.phase_bins)

            if self.encoder.is_output_boundary(vlabel):
                row = 2
                qubit = output_qubit
                output_qubit += 1
            else:
                row = 0
                qubit = input_qubit
                input_qubit += 1

            pv = g.add_vertex(ty=vtype, phase=phase_val, qubit=qubit, row=row)
            gspan_to_pyzx[vid] = pv

            if self.encoder.is_output_boundary(vlabel):
                output_pvs.append(pv)
            else:
                input_pvs.append(pv)

        # Collect edges (avoid duplicates in undirected graph).
        seen: set[tuple[int, int]] = set()
        for vid, vertex in gg.vertices.items():
            for to_vid, edge in vertex.edges.items():
                key = (min(vid, to_vid), max(vid, to_vid))
                if key in seen:
                    continue
                seen.add(key)

                elabel = int(edge.elb) if isinstance(edge.elb, str) else edge.elb
                etype = self.encoder.decode_edge(elabel)
                pv_from = gspan_to_pyzx[vid]
                pv_to = gspan_to_pyzx[to_vid]
                g.add_edge((pv_from, pv_to), edgetype=etype)

        # Set inputs/outputs if we identified any from the labels.
        if input_pvs:
            g.set_inputs(tuple(input_pvs))
        if output_pvs:
            g.set_outputs(tuple(output_pvs))

        return g
