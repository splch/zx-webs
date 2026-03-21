"""Tests for Stage 3 -- frequent sub-graph mining on ZX-diagrams."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pyzx as zx
import pytest

from zx_webs.config import MiningConfig
from zx_webs.persistence import save_json, save_manifest
from zx_webs.stage3_mining.graph_encoder import (
    ZXLabelEncoder,
    pyzx_graph_to_gspan_lines,
    pyzx_graphs_to_gspan_file,
)
from zx_webs.stage3_mining.gspan_adapter import GSpanAdapter
from zx_webs.stage3_mining.miner import _ensure_proper_boundaries
from zx_webs.stage3_mining.zx_web import BoundaryWire, ZXWeb


# ---------------------------------------------------------------------------
# Helpers -- build small PyZX graphs manually
# ---------------------------------------------------------------------------


def _make_z_z_x_chain() -> zx.Graph:
    """Return a small graph: Z --simple-- Z --simple-- X (all phase 0)."""
    g = zx.Graph()
    v1 = g.add_vertex(ty=1, phase=Fraction(0))  # Z
    v2 = g.add_vertex(ty=1, phase=Fraction(0))  # Z
    v3 = g.add_vertex(ty=2, phase=Fraction(0))  # X
    g.add_edge((v1, v2), edgetype=1)
    g.add_edge((v2, v3), edgetype=1)
    return g


def _make_z_z_x_plus_extra() -> zx.Graph:
    """Z-Z-X chain with an extra Z spider branching off the middle vertex."""
    g = _make_z_z_x_chain()
    verts = sorted(g.vertices())
    v_mid = verts[1]  # the middle Z spider
    v_extra = g.add_vertex(ty=1, phase=Fraction(0))  # extra Z
    g.add_edge((v_mid, v_extra), edgetype=1)
    return g


def _make_z_z_x_plus_different_extra() -> zx.Graph:
    """Z-Z-X chain with an extra X spider branching off the first vertex."""
    g = _make_z_z_x_chain()
    verts = sorted(g.vertices())
    v_first = verts[0]
    v_extra = g.add_vertex(ty=2, phase=Fraction(0))  # extra X
    g.add_edge((v_first, v_extra), edgetype=1)
    return g


# ---------------------------------------------------------------------------
# ZXLabelEncoder tests
# ---------------------------------------------------------------------------


class TestZXLabelEncoder:
    """Tests for the vertex/edge label encoder."""

    def test_vertex_roundtrip_z_spider(self) -> None:
        """Encode and decode a Z-spider vertex label; values should match."""
        enc = ZXLabelEncoder(phase_bins=8, include_phase=True)
        for vtype in (1, 2, 3):
            for phase in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)):
                label = enc.encode_vertex(vtype, phase)
                dec_type, dec_bin = enc.decode_vertex(label)
                assert dec_type == vtype, (
                    f"vtype mismatch for input ({vtype}, {phase}): "
                    f"got {dec_type}"
                )
                expected_bin = enc._discretize_phase(phase)
                assert dec_bin == expected_bin, (
                    f"phase_bin mismatch for input ({vtype}, {phase}): "
                    f"expected {expected_bin}, got {dec_bin}"
                )

    def test_vertex_roundtrip_no_phase(self) -> None:
        """When include_phase=False, label should equal raw vertex type."""
        enc = ZXLabelEncoder(phase_bins=8, include_phase=False)
        for vtype in (0, 1, 2, 3):
            label = enc.encode_vertex(vtype, Fraction(1, 2))
            assert label == vtype

    def test_boundary_vertex_always_zero(self) -> None:
        """Boundary vertex (type 0) should always produce label 0."""
        enc = ZXLabelEncoder(phase_bins=8, include_phase=True)
        for phase in (Fraction(0), Fraction(1, 4), Fraction(1)):
            label = enc.encode_vertex(0, phase)
            assert label == 0

    def test_phase_discretization_bins(self) -> None:
        """Verify that phases map to the correct bin indices."""
        enc = ZXLabelEncoder(phase_bins=8, include_phase=True)
        # phase=0 -> bin 0
        assert enc._discretize_phase(Fraction(0)) == 0
        # phase=1/4 -> val=0.25, bin = int(0.25*8/2)=int(1.0)=1
        assert enc._discretize_phase(Fraction(1, 4)) == 1
        # phase=1/2 -> val=0.5, bin = int(0.5*8/2)=int(2.0)=2
        assert enc._discretize_phase(Fraction(1, 2)) == 2
        # phase=1 -> val=1.0, bin = int(1.0*8/2)=int(4.0)=4
        assert enc._discretize_phase(Fraction(1)) == 4
        # phase=3/2 -> val=1.5, bin = int(1.5*8/2)=int(6.0)=6
        assert enc._discretize_phase(Fraction(3, 2)) == 6

    def test_phase_discretization_wraps(self) -> None:
        """Phases >= 2 should wrap around to [0, 2)."""
        enc = ZXLabelEncoder(phase_bins=8, include_phase=True)
        # phase=2 is equivalent to phase=0
        assert enc._discretize_phase(Fraction(2)) == 0
        # phase=5/2 is equivalent to phase=1/2
        assert enc._discretize_phase(Fraction(5, 2)) == 2

    def test_edge_roundtrip(self) -> None:
        """Encode and decode edge types; values should round-trip."""
        enc = ZXLabelEncoder()
        assert enc.encode_edge(1) == 0  # simple -> 0
        assert enc.encode_edge(2) == 1  # hadamard -> 1
        assert enc.decode_edge(0) == 1
        assert enc.decode_edge(1) == 2


# ---------------------------------------------------------------------------
# gSpan format conversion tests
# ---------------------------------------------------------------------------


class TestGSpanFormatConversion:
    """Tests for converting PyZX graphs to gSpan text format."""

    def test_pyzx_graph_to_gspan_lines(self) -> None:
        """Convert a known Z-Z-X graph; verify the gSpan lines are correct."""
        enc = ZXLabelEncoder(phase_bins=8, include_phase=True)
        g = _make_z_z_x_chain()
        lines = pyzx_graph_to_gspan_lines(g, graph_id=0, encoder=enc)

        # First line is the graph header.
        assert lines[0] == "t # 0"

        # Should have 3 vertex lines and 2 edge lines.
        v_lines = [ln for ln in lines if ln.startswith("v ")]
        e_lines = [ln for ln in lines if ln.startswith("e ")]
        assert len(v_lines) == 3
        assert len(e_lines) == 2

        # Verify vertex labels:
        # Z with phase 0 -> encode_vertex(1, 0) = 4 + 0*8 + 0 = 4
        # X with phase 0 -> encode_vertex(2, 0) = 4 + 1*8 + 0 = 12
        z_label = enc.encode_vertex(1, Fraction(0))
        x_label = enc.encode_vertex(2, Fraction(0))
        vlabels = [int(ln.split()[2]) for ln in v_lines]
        assert vlabels.count(z_label) == 2
        assert vlabels.count(x_label) == 1

    def test_pyzx_graphs_to_gspan_file(self, tmp_path: Path) -> None:
        """Write multiple graphs to a file; verify file structure."""
        enc = ZXLabelEncoder(phase_bins=8, include_phase=True)
        graphs = [_make_z_z_x_chain(), _make_z_z_x_plus_extra()]
        outfile = tmp_path / "test.gspan"

        vid_maps = pyzx_graphs_to_gspan_file(graphs, outfile, enc)

        assert outfile.exists()
        content = outfile.read_text()

        # Should have 2 graph headers.
        assert content.count("t # 0") == 1
        assert content.count("t # 1") == 1
        # Should end with terminator.
        assert "t # -1" in content

        # vid_maps should have entries for both graphs.
        assert 0 in vid_maps
        assert 1 in vid_maps
        assert len(vid_maps[0]) == 3  # 3 vertices in graph 0
        assert len(vid_maps[1]) == 4  # 4 vertices in graph 1


# ---------------------------------------------------------------------------
# GSpanAdapter mining tests
# ---------------------------------------------------------------------------


class TestGSpanAdapterMine:
    """Tests for the gSpan mining adapter."""

    def test_mine_finds_common_subgraph(self) -> None:
        """Three graphs sharing a Z-Z-X chain; mining should find it."""
        config = MiningConfig(
            min_support=3,
            min_vertices=2,
            max_vertices=20,
            phase_discretization=8,
            include_phase_in_label=True,
        )
        adapter = GSpanAdapter(config)

        graphs = [
            _make_z_z_x_chain(),
            _make_z_z_x_plus_extra(),
            _make_z_z_x_plus_different_extra(),
        ]

        results = adapter.mine(graphs)

        # Should find at least one frequent sub-graph.
        assert len(results) > 0

        # At least one result should have support == 3.
        max_support = max(r.support for r in results)
        assert max_support == 3

    def test_mine_empty_corpus(self) -> None:
        """Mining an empty corpus should return no results."""
        config = MiningConfig(min_support=1, min_vertices=2)
        adapter = GSpanAdapter(config)
        results = adapter.mine([])
        assert results == []

    def test_mine_below_support_threshold(self) -> None:
        """When min_support exceeds corpus size, nothing should be found."""
        config = MiningConfig(min_support=100, min_vertices=2)
        adapter = GSpanAdapter(config)
        graphs = [_make_z_z_x_chain()]
        results = adapter.mine(graphs)
        assert results == []

    def test_result_to_pyzx_roundtrip(self) -> None:
        """Convert a result back to PyZX and verify structure."""
        config = MiningConfig(
            min_support=3,
            min_vertices=2,
            max_vertices=20,
            phase_discretization=8,
            include_phase_in_label=True,
        )
        adapter = GSpanAdapter(config)

        graphs = [
            _make_z_z_x_chain(),
            _make_z_z_x_plus_extra(),
            _make_z_z_x_plus_different_extra(),
        ]
        results = adapter.mine(graphs)
        assert len(results) > 0

        # Convert the first result back to PyZX.
        pyzx_g = adapter.result_to_pyzx(results[0])
        assert pyzx_g.num_vertices() >= 2
        assert pyzx_g.num_edges() >= 1


# ---------------------------------------------------------------------------
# Boundary vertex handling tests
# ---------------------------------------------------------------------------


class TestEnsureProperBoundaries:
    """Tests for _ensure_proper_boundaries."""

    def test_adds_boundary_to_bare_graph(self) -> None:
        """A graph with no boundaries gets boundary vertices added."""
        g = _make_z_z_x_chain()
        assert not g.inputs()
        assert not g.outputs()

        g = _ensure_proper_boundaries(g)

        assert len(g.inputs()) > 0, "Should have inputs after boundary fix"
        assert len(g.outputs()) > 0, "Should have outputs after boundary fix"

    def test_preserves_existing_boundaries(self) -> None:
        """A graph that already has proper boundaries is unchanged."""
        g = zx.Graph()
        i0 = g.add_vertex(ty=0, qubit=0, row=0)
        z0 = g.add_vertex(ty=1, phase=Fraction(0), qubit=0, row=1)
        o0 = g.add_vertex(ty=0, qubit=0, row=2)
        g.add_edge((i0, z0), edgetype=1)
        g.add_edge((z0, o0), edgetype=1)
        g.set_inputs((i0,))
        g.set_outputs((o0,))

        n_verts_before = g.num_vertices()
        g = _ensure_proper_boundaries(g)

        assert g.num_vertices() == n_verts_before
        assert len(g.inputs()) == 1
        assert len(g.outputs()) == 1

    def test_mined_web_has_boundaries(self) -> None:
        """Mined sub-graphs should have proper boundary vertices."""
        config = MiningConfig(
            min_support=3,
            min_vertices=2,
            max_vertices=20,
            phase_discretization=8,
            include_phase_in_label=True,
        )
        adapter = GSpanAdapter(config)

        graphs = [
            _make_z_z_x_chain(),
            _make_z_z_x_plus_extra(),
            _make_z_z_x_plus_different_extra(),
        ]
        results = adapter.mine(graphs)
        assert len(results) > 0

        for result in results:
            pyzx_g = adapter.result_to_pyzx(result)
            pyzx_g = _ensure_proper_boundaries(pyzx_g)
            assert len(pyzx_g.inputs()) > 0, (
                "Mined web should have inputs after boundary fix"
            )
            assert len(pyzx_g.outputs()) > 0, (
                "Mined web should have outputs after boundary fix"
            )


# ---------------------------------------------------------------------------
# ZXWeb serialisation tests
# ---------------------------------------------------------------------------


class TestZXWebSerialization:
    """Tests for the ZXWeb data class."""

    def test_roundtrip(self) -> None:
        """Serialise a ZXWeb to dict and back; all fields should match."""
        g = _make_z_z_x_chain()
        web = ZXWeb(
            web_id="web_0000",
            graph_json=g.to_json(),
            boundary_wires=[
                BoundaryWire(
                    internal_vertex=0,
                    spider_type=1,
                    spider_phase=0.0,
                    edge_type=1,
                    direction="input",
                ),
                BoundaryWire(
                    internal_vertex=2,
                    spider_type=2,
                    spider_phase=0.0,
                    edge_type=1,
                    direction="output",
                ),
            ],
            support=5,
            source_graph_ids=[0, 1, 2, 3, 4],
            n_spiders=3,
            n_inputs=1,
            n_outputs=1,
        )

        d = web.to_dict()
        web2 = ZXWeb.from_dict(d)

        assert web2.web_id == web.web_id
        assert web2.support == web.support
        assert web2.source_graph_ids == web.source_graph_ids
        assert web2.n_spiders == web.n_spiders
        assert web2.n_inputs == web.n_inputs
        assert web2.n_outputs == web.n_outputs
        assert len(web2.boundary_wires) == 2
        assert web2.boundary_wires[0].direction == "input"
        assert web2.boundary_wires[1].direction == "output"

    def test_to_pyzx_graph(self) -> None:
        """to_pyzx_graph should reconstruct a valid PyZX graph."""
        g = _make_z_z_x_chain()
        web = ZXWeb(web_id="test", graph_json=g.to_json())
        g2 = web.to_pyzx_graph()
        assert g2.num_vertices() == 3
        assert g2.num_edges() == 2

    def test_json_serializable(self) -> None:
        """to_dict output should be JSON-serialisable."""
        g = _make_z_z_x_chain()
        web = ZXWeb(
            web_id="web_json",
            graph_json=g.to_json(),
            support=2,
            source_graph_ids=[0, 1],
        )
        d = web.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert "web_json" in json_str


# ---------------------------------------------------------------------------
# End-to-end Stage 3 test
# ---------------------------------------------------------------------------


class TestReversedBoundaries:
    """Tests for reversed boundary web generation."""

    def test_reversed_web_swaps_io(self) -> None:
        """A reversed web should swap inputs and outputs."""
        from zx_webs.stage3_mining.miner import _make_reversed_web

        g = zx.Graph()
        i0 = g.add_vertex(ty=0, qubit=0, row=0)
        z0 = g.add_vertex(ty=1, phase=Fraction(0), qubit=0, row=1)
        o0 = g.add_vertex(ty=0, qubit=0, row=2)
        g.add_edge((i0, z0), edgetype=1)
        g.add_edge((z0, o0), edgetype=1)
        g.set_inputs((i0,))
        g.set_outputs((o0,))

        web = ZXWeb(
            web_id="web_0000",
            graph_json=g.to_json(),
            boundary_wires=[
                BoundaryWire(
                    internal_vertex=1,
                    spider_type=1,
                    spider_phase=0.0,
                    edge_type=1,
                    direction="input",
                ),
                BoundaryWire(
                    internal_vertex=1,
                    spider_type=1,
                    spider_phase=0.0,
                    edge_type=1,
                    direction="output",
                ),
            ],
            support=3,
            n_spiders=1,
            n_inputs=1,
            n_outputs=1,
        )

        reversed_web = _make_reversed_web(web, "web_0001")
        assert reversed_web is not None
        assert reversed_web.n_inputs == web.n_outputs
        assert reversed_web.n_outputs == web.n_inputs
        assert reversed_web.web_id == "web_0001"
        assert reversed_web.support == web.support

        # Boundary wire directions should be swapped.
        for bw in reversed_web.boundary_wires:
            if bw.direction == "input":
                assert any(
                    obw.direction == "output"
                    for obw in web.boundary_wires
                    if obw.internal_vertex == bw.internal_vertex
                )

    def test_run_stage3_generates_reversed_webs(self, tmp_path: Path) -> None:
        """Stage 3 should produce both original and reversed webs."""
        from zx_webs.stage3_mining.miner import run_stage3

        zx_dir = tmp_path / "zx_diagrams"
        graphs_dir = zx_dir / "graphs"
        graphs_dir.mkdir(parents=True)

        test_graphs = [
            _make_z_z_x_chain(),
            _make_z_z_x_plus_extra(),
            _make_z_z_x_plus_different_extra(),
        ]

        manifest_entries = []
        for i, g in enumerate(test_graphs):
            algo_id = f"test_algo_{i}"
            graph_path = graphs_dir / f"{algo_id}.zxg.json"
            graph_path.write_text(g.to_json(), encoding="utf-8")
            manifest_entries.append({"algorithm_id": algo_id, "graph_path": str(graph_path)})

        save_manifest(manifest_entries, zx_dir)

        output_dir = tmp_path / "webs_output"
        config = MiningConfig(
            min_support=3,
            min_vertices=2,
            max_vertices=20,
            phase_discretization=8,
            include_phase_in_label=True,
        )
        webs = run_stage3(zx_dir, output_dir, config)

        # Should have more webs than results (due to reversed webs).
        # Each original web gets a reversed counterpart.
        original_count = sum(1 for w in webs if not w.web_id.endswith("reversed"))
        # The total should be roughly 2x the mined patterns.
        assert len(webs) >= 2, "Should generate both original and reversed webs"


class TestRunStage3EndToEnd:
    """End-to-end integration test for run_stage3."""

    def test_run_stage3_creates_outputs(self, tmp_path: Path) -> None:
        """Set up Stage 2 outputs, run Stage 3, and verify webs are created."""
        from zx_webs.stage3_mining.miner import run_stage3

        # Set up a mock Stage 2 output directory.
        zx_dir = tmp_path / "zx_diagrams"
        graphs_dir = zx_dir / "graphs"
        graphs_dir.mkdir(parents=True)

        # Create 3 ZX-diagrams that share a common Z-Z-X sub-pattern.
        test_graphs = [
            _make_z_z_x_chain(),
            _make_z_z_x_plus_extra(),
            _make_z_z_x_plus_different_extra(),
        ]

        manifest_entries = []
        for i, g in enumerate(test_graphs):
            algo_id = f"test_algo_{i}"
            graph_path = graphs_dir / f"{algo_id}.zxg.json"
            graph_path.write_text(g.to_json(), encoding="utf-8")
            manifest_entries.append(
                {
                    "algorithm_id": algo_id,
                    "graph_path": str(graph_path),
                }
            )

        save_manifest(manifest_entries, zx_dir)

        # Run Stage 3.
        output_dir = tmp_path / "webs_output"
        config = MiningConfig(
            min_support=3,
            min_vertices=2,
            max_vertices=20,
            phase_discretization=8,
            include_phase_in_label=True,
        )
        webs = run_stage3(zx_dir, output_dir, config)

        # Verify outputs.
        assert len(webs) > 0, "Expected at least one frequent sub-graph."

        # Manifest should exist.
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

        # Bulk file should exist (individual web files are no longer written).
        bulk_path = output_dir / "webs_bulk.json"
        assert bulk_path.exists()

        # At least one web should have support == 3.
        max_support = max(w.support for w in webs)
        assert max_support == 3

        # Webs should have proper boundary info.
        for web in webs:
            assert web.n_inputs > 0, (
                f"Web {web.web_id} should have n_inputs > 0"
            )
            assert web.n_outputs > 0, (
                f"Web {web.web_id} should have n_outputs > 0"
            )

    def test_run_stage3_empty_manifest(self, tmp_path: Path) -> None:
        """run_stage3 on an empty manifest should return no webs."""
        from zx_webs.stage3_mining.miner import run_stage3

        zx_dir = tmp_path / "zx_empty"
        zx_dir.mkdir()
        save_manifest([], zx_dir)

        output_dir = tmp_path / "webs_empty"
        webs = run_stage3(zx_dir, output_dir)
        assert webs == []
