"""
Subgraph isomorphism engine for ZX diagrams.

Uses NetworkX's VF2 implementation with semantic node/edge matching
to find occurrences of small ZX patterns in larger diagrams.

Key design decisions:
  1. Node matching respects vertex_type and phase_class (not exact phase).
  2. Edge matching respects edge_type (SIMPLE vs HADAMARD).
  3. Boundary nodes are excluded from interior matching — we only match
     the "guts" of a pattern, not its I/O connections.

This gives you commutation-aware pattern detection "for free" because
ZX diagrams have already absorbed commutation relations into their topology.
"""
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
from networkx.algorithms import isomorphism


@dataclass
class MotifMatch:
    """A single match of a motif pattern in a host graph."""

    motif_id: str
    host_algorithm: str
    host_level: str
    mapping: dict  # {pattern_node: host_node}
    matched_node_types: list


@dataclass
class MotifPattern:
    """A candidate motif pattern to search for."""

    motif_id: str
    graph: nx.Graph
    source: str  # Which algorithm / strategy it came from
    description: str = ""
    occurrences: list[MotifMatch] = field(default_factory=list)


# ── Semantic Matching Functions ──────────────────────────────────────


def node_match_fn(n1_attrs: dict, n2_attrs: dict) -> bool:
    """
    Node compatibility for VF2 subgraph isomorphism.

    Matches on vertex_type (Z/X/H_BOX must match exactly) and
    phase_class (coarsened phase). Does NOT match on exact phase
    or degree — degree is handled implicitly by subgraph topology.
    """
    if n1_attrs.get("vertex_type") != n2_attrs.get("vertex_type"):
        return False
    if n1_attrs.get("phase_class") != n2_attrs.get("phase_class"):
        return False
    return True


def edge_match_fn(e1_attrs: dict, e2_attrs: dict) -> bool:
    """Edge compatibility: SIMPLE and HADAMARD edges are distinct."""
    return e1_attrs.get("edge_type") == e2_attrs.get("edge_type")


# ── Core Matching ────────────────────────────────────────────────────


def find_motif_in_graph(
    pattern: nx.Graph,
    host: nx.Graph,
    max_matches: int = 100,
    exclude_boundary: bool = True,
) -> list[dict]:
    """
    Find all subgraph isomorphism matches of `pattern` in `host`.

    Args:
        pattern: The motif to search for (small graph, ~3-10 nodes).
        host: The algorithm's ZX graph to search in.
        max_matches: Cap on matches to avoid combinatorial explosion.
        exclude_boundary: If True, remove BOUNDARY nodes from the host
            before matching (we want interior structure).

    Returns:
        List of node mappings {pattern_node: host_node}.
    """
    if exclude_boundary:
        interior_nodes = [
            n for n, d in host.nodes(data=True)
            if d.get("vertex_type") != "BOUNDARY"
        ]
        host_interior = host.subgraph(interior_nodes)
    else:
        host_interior = host

    if host_interior.number_of_nodes() < pattern.number_of_nodes():
        return []

    gm = isomorphism.GraphMatcher(
        host_interior,
        pattern,
        node_match=node_match_fn,
        edge_match=edge_match_fn,
    )

    matches = []
    for mapping in gm.subgraph_isomorphisms_iter():
        # mapping is {host_node: pattern_node}; invert for our convention
        inv_mapping = {v: k for k, v in mapping.items()}
        matches.append(inv_mapping)
        if len(matches) >= max_matches:
            break

    return matches


def find_motif_across_corpus(
    pattern: MotifPattern,
    corpus: dict,  # {(algo_name, level): nx.Graph}
    target_level: str = "spider_fused",
    max_matches_per_graph: int = 50,
) -> MotifPattern:
    """
    Search for a motif across all algorithm graphs at a given level.
    Populates pattern.occurrences with MotifMatch objects.
    """
    pattern.occurrences = []

    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue

        matches = find_motif_in_graph(
            pattern.graph,
            host_graph,
            max_matches=max_matches_per_graph,
        )

        for m in matches:
            pattern.occurrences.append(
                MotifMatch(
                    motif_id=pattern.motif_id,
                    host_algorithm=algo_name,
                    host_level=level,
                    mapping=m,
                    matched_node_types=[
                        host_graph.nodes[v]["vertex_type"] for v in m.values()
                    ],
                )
            )

    return pattern
