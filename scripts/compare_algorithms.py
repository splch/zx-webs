#!/usr/bin/env python3
"""
Comparison Script — New Candidates vs Industry Standards
=========================================================

Systematically tests whether any of the 48 data-driven discovery candidates
outperform industry-standard quantum algorithms on 5 metric categories:

  1. Gate Efficiency       — gate count, depth, gates-per-qubit
  2. ZX Compression        — vertex/edge reduction across simplification levels
  3. Noise Resilience      — analytical depolarising model + exact simulation
  4. Entanglement Quality  — von Neumann entropy, concurrence
  5. Structural Novelty    — loaded from discovery results

Outputs (scripts/output/comparison/):
  - comparison_report.txt      Ranked tables per metric
  - gate_efficiency.png        Gate count / depth scatter
  - noise_resilience.png       Fidelity vs noise rate
  - entanglement_quality.png   Entropy / concurrence heatmap
  - zx_compression.png         Compression ratio bar chart
  - radar_top10.png            Radar plots of top candidates
  - results.json               Machine-readable metrics
"""

from __future__ import annotations

import copy
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from qiskit import QuantumCircuit

# ── Project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zx_motifs.algorithms.registry import ALGORITHM_FAMILY_MAP, REGISTRY
from zx_motifs.pipeline.converter import (
    SimplificationLevel,
    convert_at_all_levels,
    count_t_gates,
    qiskit_to_zx,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DISCOVERY_DIR = SCRIPT_DIR / "output" / "discovery"
OUTPUT_DIR = SCRIPT_DIR / "output" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import discover_algorithm helpers (for rebuilding candidates) ─────
sys.path.insert(0, str(SCRIPT_DIR))
from discover_algorithm import (
    CandidateSpec,
    _HANDCRAFTED_TEMPLATES,
    build_circuit_from_spec,
    build_corpus,
    build_template_registry,
    discover_motifs,
    load_phylogeny_results,
    make_distillation_entanglement_bridge,
    make_irreducible_core_circuit,
    make_trotter_variational_hybrid,
)

# ═══════════════════════════════════════════════════════════════════════
# Industry Standard Baselines
# ═══════════════════════════════════════════════════════════════════════

# Representative algorithm from each major family.
# (name_in_registry, n_qubits_to_use)
STANDARD_SPECS = [
    ("bell_state", 2),
    ("ghz_state", 4),
    ("cluster_state", 4),
    ("w_state", 4),
    ("teleportation", 3),
    ("grover", 4),
    ("qft", 4),
    ("phase_estimation", 4),
    ("qaoa_maxcut", 4),
    ("vqe_uccsd", 4),
    ("trotter_ising", 4),
    ("bit_flip_code", 5),
    ("bbpssw_distillation", 4),
    ("bb84_encode", 4),
    ("iqp_sampling", 4),
]

REGISTRY_MAP = {entry.name: entry for entry in REGISTRY}


def build_standard_circuits() -> list[tuple[str, QuantumCircuit, str]]:
    """Build industry-standard baseline circuits.

    Returns list of (label, QuantumCircuit, family).
    """
    circuits = []
    for name, nq in STANDARD_SPECS:
        entry = REGISTRY_MAP.get(name)
        if entry is None:
            continue
        try:
            qc = entry.generator(nq)
            label = f"std_{name}"
            circuits.append((label, qc, entry.family))
        except Exception as exc:
            print(f"  WARNING: failed to build {name}({nq}q): {exc}")
    return circuits


# ═══════════════════════════════════════════════════════════════════════
# Rebuild Discovery Candidates
# ═══════════════════════════════════════════════════════════════════════


def rebuild_candidates(
    discovery_results: dict, templates: dict
) -> list[tuple[str, QuantumCircuit, str]]:
    """Rebuild candidate circuits from saved discovery results.

    Returns list of (name, QuantumCircuit, strategy).
    """
    validations = discovery_results.get("validations", {})
    scored = discovery_results.get("scored", {})

    circuits = []
    for cname, score_info in scored.items():
        # Reconstruct a CandidateSpec from saved data
        strategy = score_info.get("strategy", "unknown")
        val = validations.get(cname, {})
        n_qubits = val.get("n_qubits", 4)
        top_motifs = score_info.get("top_motifs", [])
        motif_ids = [mid for mid, _ in top_motifs[:6]]

        spec = CandidateSpec(
            name=cname,
            strategy=strategy,
            motif_ids=motif_ids,
            n_qubits=n_qubits,
            source_algo_a=score_info.get("nearest_algorithm"),
            source_algo_b=None,
        )

        qc = build_circuit_from_spec(spec, templates)
        if qc is not None and qc.size() > 0:
            circuits.append((cname, qc, strategy))

    return circuits


def build_legacy_circuits() -> list[tuple[str, QuantumCircuit, str]]:
    """Build the 3 legacy candidate circuits."""
    return [
        ("legacy_tvh", make_trotter_variational_hybrid(n_qubits=4), "legacy"),
        ("legacy_deb", make_distillation_entanglement_bridge(n_qubits=4), "legacy"),
        ("legacy_icc", make_irreducible_core_circuit(n_qubits=4), "legacy"),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Metric 1: Gate Efficiency
# ═══════════════════════════════════════════════════════════════════════


def compute_gate_efficiency(name: str, qc: QuantumCircuit) -> dict:
    """Compute gate-level metrics for a circuit."""
    n_qubits = qc.num_qubits
    total_gates = qc.size()
    depth = qc.depth()

    n_1q = 0
    n_2q = 0
    for instruction in qc.data:
        nq = instruction.operation.num_qubits
        if nq == 1:
            n_1q += 1
        elif nq >= 2:
            n_2q += 1

    return {
        "name": name,
        "n_qubits": n_qubits,
        "total_gates": total_gates,
        "n_1q_gates": n_1q,
        "n_2q_gates": n_2q,
        "depth": depth,
        "gates_per_qubit": total_gates / max(1, n_qubits),
        "two_qubit_ratio": n_2q / max(1, total_gates),
    }


# ═══════════════════════════════════════════════════════════════════════
# Metric 2: ZX Compression
# ═══════════════════════════════════════════════════════════════════════


def compute_zx_compression(name: str, qc: QuantumCircuit) -> dict:
    """Compute ZX compression metrics across simplification levels."""
    try:
        snapshots = convert_at_all_levels(qc, name)
    except Exception:
        return {
            "name": name,
            "raw_vertices": 0,
            "raw_edges": 0,
            "reduced_vertices": 0,
            "reduced_edges": 0,
            "compression_ratio": 0.0,
            "t_count": 0,
            "levels": {},
        }

    levels = {}
    raw_v = raw_e = 0
    reduced_v = reduced_e = 0
    t_count = 0

    for snap in snapshots:
        lv = snap.level.value
        levels[lv] = {
            "vertices": snap.num_vertices,
            "edges": snap.num_edges,
            "t_gates": snap.num_t_gates,
        }
        if snap.level == SimplificationLevel.RAW:
            raw_v = snap.num_vertices
            raw_e = snap.num_edges
            t_count = snap.num_t_gates
        if snap.level == SimplificationLevel.FULL_REDUCE:
            reduced_v = snap.num_vertices
            reduced_e = snap.num_edges

    compression = 1.0 - (reduced_v / max(1, raw_v))

    return {
        "name": name,
        "raw_vertices": raw_v,
        "raw_edges": raw_e,
        "reduced_vertices": reduced_v,
        "reduced_edges": reduced_e,
        "compression_ratio": compression,
        "t_count": t_count,
        "levels": levels,
    }


# ═══════════════════════════════════════════════════════════════════════
# Metric 3: Noise Resilience
# ═══════════════════════════════════════════════════════════════════════

NOISE_RATES = [0.001, 0.005, 0.01, 0.05]


def compute_noise_resilience_analytical(gate_eff: dict) -> dict:
    """Analytical depolarising noise model.

    Output fidelity ≈ (1-p)^n_2q * (1-p/10)^n_1q for each noise rate p.
    """
    n_1q = gate_eff["n_1q_gates"]
    n_2q = gate_eff["n_2q_gates"]

    fidelities = {}
    for p in NOISE_RATES:
        f_2q = (1.0 - p) ** n_2q
        f_1q = (1.0 - p / 10.0) ** n_1q
        fidelities[str(p)] = f_2q * f_1q

    return {"name": gate_eff["name"], "analytical_fidelity": fidelities}


def compute_noise_resilience_exact(
    name: str, qc: QuantumCircuit, noise_rates: list[float] | None = None
) -> dict | None:
    """Exact DensityMatrix simulation with depolarising noise (≤5 qubits)."""
    if qc.num_qubits > 5:
        return None

    try:
        from qiskit.quantum_info import (
            DensityMatrix,
            Operator,
            SuperOp,
            state_fidelity,
        )
    except ImportError:
        return None

    if noise_rates is None:
        noise_rates = NOISE_RATES

    # Ideal output
    try:
        ideal = DensityMatrix.from_instruction(qc)
    except Exception:
        return None

    fidelities = {}
    for p in noise_rates:
        try:
            # Build noisy circuit by applying depolarising channel after each gate
            noisy_dm = DensityMatrix.from_int(0, 2**qc.num_qubits)
            # Simulate gate-by-gate
            for instruction in qc.data:
                op = Operator(instruction.operation)
                qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]
                noisy_dm = noisy_dm.evolve(op, qubit_indices)

                # Apply depolarising channel to involved qubits
                nq = len(qubit_indices)
                if nq == 1:
                    p_eff = p / 10.0
                else:
                    p_eff = p

                # Depolarising channel: ρ → (1-p_eff)ρ + p_eff * I/d
                d = 2**nq
                # Build partial depolarising as a SuperOp on the involved qubits
                identity_dm = np.eye(d) / d
                # Apply: trace over involved qubits, mix
                sub_dm = noisy_dm.data
                noisy_dm = DensityMatrix(
                    (1.0 - p_eff) * sub_dm + p_eff * np.trace(sub_dm) * np.eye(sub_dm.shape[0]) / sub_dm.shape[0]
                )

            fid = state_fidelity(ideal, noisy_dm)
            fidelities[str(p)] = float(fid)
        except Exception:
            fidelities[str(p)] = None

    return {"name": name, "exact_fidelity": fidelities}


# ═══════════════════════════════════════════════════════════════════════
# Metric 4: Entanglement Quality
# ═══════════════════════════════════════════════════════════════════════


def compute_entanglement_quality(name: str, qc: QuantumCircuit) -> dict | None:
    """Compute entanglement metrics for circuits ≤ 6 qubits."""
    if qc.num_qubits > 6:
        return None

    try:
        from qiskit.quantum_info import (
            Statevector,
            concurrence,
            entropy,
            partial_trace,
        )
    except ImportError:
        return None

    try:
        sv = Statevector.from_instruction(qc)
        rho = sv.to_operator()  # full density matrix

        # Bipartite von Neumann entropy (half-cut)
        n = qc.num_qubits
        half = n // 2
        # Trace out second half to get reduced density matrix of first half
        trace_out = list(range(half, n))
        rho_A = partial_trace(sv, trace_out)
        vn_entropy = float(entropy(rho_A, base=2))

        # Pairwise concurrence for first 2 qubits
        conc = 0.0
        if n >= 2:
            trace_out_for_pair = [i for i in range(n) if i >= 2]
            if trace_out_for_pair:
                rho_pair = partial_trace(sv, trace_out_for_pair)
            else:
                rho_pair = sv.to_operator()
            try:
                conc = float(concurrence(rho_pair))
            except Exception:
                conc = 0.0

        # Purity of reduced state
        purity = float(np.real(np.trace(np.array(rho_A.data) @ np.array(rho_A.data))))

        return {
            "name": name,
            "n_qubits": n,
            "von_neumann_entropy": vn_entropy,
            "concurrence": conc,
            "purity": purity,
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Metric 5: Structural Novelty (from discovery results)
# ═══════════════════════════════════════════════════════════════════════


def load_novelty_scores(discovery_results: dict) -> dict:
    """Extract novelty scores from discovery results."""
    novelty = {}
    for section in ["scored", "legacy_scored"]:
        for cname, info in discovery_results.get(section, {}).items():
            novelty[cname] = {
                "cosine_novelty": info.get("cosine_novelty", 0.0),
                "pca_isolation": info.get("pca_isolation", 0.0),
                "motif_diversity": info.get("motif_diversity", 0.0),
                "composite_novelty": info.get("composite_novelty", 0.0),
            }
    return novelty


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Ranking + "Outperform" Detection
# ═══════════════════════════════════════════════════════════════════════


def find_winners(all_metrics: dict, standard_names: set) -> list[dict]:
    """Find candidates that beat ALL standards on at least one metric.

    Returns list of {candidate, metric, value, best_standard, standard_value, margin}.
    """
    winners = []

    # Metrics where LOWER is better
    lower_better = [
        "total_gates", "depth", "gates_per_qubit", "two_qubit_ratio", "t_count",
    ]
    # Metrics where HIGHER is better
    higher_better = [
        "compression_ratio", "von_neumann_entropy", "concurrence", "composite_novelty",
    ]

    # Also noise resilience fidelities (higher is better)
    for p in NOISE_RATES:
        higher_better.append(f"fidelity_p{p}")

    candidate_names = set(all_metrics.keys()) - standard_names

    for metric in lower_better + higher_better:
        is_lower = metric in lower_better

        # Get standard values
        std_values = []
        for sn in standard_names:
            m = all_metrics.get(sn, {})
            v = m.get(metric)
            if v is not None and v != 0:
                std_values.append((sn, v))

        if not std_values:
            continue

        if is_lower:
            best_std_name, best_std_val = min(std_values, key=lambda x: x[1])
        else:
            best_std_name, best_std_val = max(std_values, key=lambda x: x[1])

        for cn in candidate_names:
            m = all_metrics.get(cn, {})
            v = m.get(metric)
            if v is None or v == 0:
                continue

            beats = (v < best_std_val) if is_lower else (v > best_std_val)
            if beats:
                if is_lower:
                    margin = (best_std_val - v) / max(1e-9, abs(best_std_val))
                else:
                    margin = (v - best_std_val) / max(1e-9, abs(best_std_val))

                # Skip trivial wins (< 0.1% margin)
                if margin < 0.001:
                    continue

                winners.append({
                    "candidate": cn,
                    "metric": metric,
                    "candidate_value": v,
                    "best_standard": best_std_name,
                    "standard_value": best_std_val,
                    "margin": margin,
                })

    return winners


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Visualization
# ═══════════════════════════════════════════════════════════════════════

CATEGORY_COLORS = {
    "standard": "#2166ac",
    "coverage_gap": "#e41a1c",
    "cross_family": "#377eb8",
    "irreducible": "#4daf4a",
    "pca_void": "#984ea3",
    "legacy": "#999999",
}


def _get_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, "#333333")


def _get_marker(category: str) -> str:
    markers = {
        "standard": "s",
        "coverage_gap": "o",
        "cross_family": "D",
        "irreducible": "^",
        "pca_void": "v",
        "legacy": "p",
    }
    return markers.get(category, "o")


def plot_gate_efficiency(all_metrics: dict, categories: dict):
    """Scatter: total gates vs depth, colored by category."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for name, m in all_metrics.items():
        cat = categories.get(name, "unknown")
        gates = m.get("total_gates", 0)
        depth = m.get("depth", 0)
        if gates == 0 and depth == 0:
            continue
        ax.scatter(
            gates, depth,
            c=_get_color(cat), marker=_get_marker(cat),
            s=80, alpha=0.7, edgecolors="k", linewidths=0.5,
        )
        if cat == "standard":
            ax.annotate(
                name.replace("std_", ""), (gates, depth),
                fontsize=7, alpha=0.8,
                xytext=(4, 4), textcoords="offset points",
            )

    # Legend
    handles = [
        Line2D([0], [0], marker=_get_marker(c), color="w",
               markerfacecolor=_get_color(c), markersize=8, label=c.replace("_", " ").title())
        for c in ["standard", "coverage_gap", "cross_family", "irreducible", "pca_void", "legacy"]
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.set_xlabel("Total Gates")
    ax.set_ylabel("Circuit Depth")
    ax.set_title("Gate Efficiency: Total Gates vs Depth")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gate_efficiency.png", dpi=150)
    plt.close(fig)


def plot_noise_resilience(all_metrics: dict, categories: dict):
    """Line plot: fidelity vs noise rate for each circuit."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for name, m in all_metrics.items():
        cat = categories.get(name, "unknown")
        fids = []
        for p in NOISE_RATES:
            f = m.get(f"fidelity_p{p}")
            if f is not None:
                fids.append((p, f))

        if not fids:
            continue

        ps, fs = zip(*fids)
        lw = 2.0 if cat == "standard" else 0.8
        alpha = 1.0 if cat == "standard" else 0.5
        ax.plot(ps, fs, color=_get_color(cat), linewidth=lw, alpha=alpha)

    handles = [
        Line2D([0], [0], color=_get_color(c), linewidth=2, label=c.replace("_", " ").title())
        for c in ["standard", "coverage_gap", "cross_family", "irreducible", "pca_void", "legacy"]
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.set_xlabel("Noise Rate (p)")
    ax.set_ylabel("Output Fidelity")
    ax.set_title("Noise Resilience: Fidelity vs Depolarising Noise Rate")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "noise_resilience.png", dpi=150)
    plt.close(fig)


def plot_entanglement_quality(all_metrics: dict, categories: dict):
    """Heatmap-style bar chart: entropy and concurrence."""
    # Collect circuits that have entanglement data
    ent_data = []
    for name, m in sorted(all_metrics.items()):
        vn = m.get("von_neumann_entropy")
        conc = m.get("concurrence")
        if vn is not None:
            ent_data.append((name, vn, conc or 0.0, categories.get(name, "unknown")))

    if not ent_data:
        return

    # Sort by entropy descending
    ent_data.sort(key=lambda x: -x[1])
    ent_data = ent_data[:30]  # Top 30

    names = [d[0] for d in ent_data]
    entropies = [d[1] for d in ent_data]
    concurrences = [d[2] for d in ent_data]
    colors = [_get_color(d[3]) for d in ent_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, entropies, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("Von Neumann Entropy (bits)")
    ax1.set_title("Bipartite Entanglement (half-cut)")
    ax1.invert_yaxis()

    ax2.barh(y_pos, concurrences, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=6)
    ax2.set_xlabel("Concurrence (qubits 0-1)")
    ax2.set_title("Pairwise Concurrence")
    ax2.invert_yaxis()

    # Legend
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_get_color(c),
               markersize=8, label=c.replace("_", " ").title())
        for c in ["standard", "coverage_gap", "cross_family", "irreducible", "pca_void", "legacy"]
    ]
    ax1.legend(handles=handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "entanglement_quality.png", dpi=150)
    plt.close(fig)


def plot_zx_compression(all_metrics: dict, categories: dict):
    """Bar chart: compression ratio sorted descending."""
    comp_data = []
    for name, m in all_metrics.items():
        cr = m.get("compression_ratio", 0)
        if cr > 0:
            comp_data.append((name, cr, categories.get(name, "unknown")))

    if not comp_data:
        return

    comp_data.sort(key=lambda x: -x[1])
    comp_data = comp_data[:40]  # Top 40

    names = [d[0] for d in comp_data]
    ratios = [d[1] for d in comp_data]
    colors = [_get_color(d[2]) for d in comp_data]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, ratios, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Compression Ratio (1 - reduced/raw vertices)")
    ax.set_title("ZX Compression: Vertex Reduction via full_reduce")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_get_color(c),
               markersize=8, label=c.replace("_", " ").title())
        for c in ["standard", "coverage_gap", "cross_family", "irreducible", "pca_void", "legacy"]
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "zx_compression.png", dpi=150)
    plt.close(fig)


def plot_radar_top10(all_metrics: dict, categories: dict, standard_names: set):
    """Radar plots for the top 10 candidates by composite score."""
    # 6 axes: gate efficiency (inverted), compression, noise (p=0.01),
    #         entropy, concurrence, novelty
    axes_labels = [
        "Gate Eff.", "ZX Compress.", "Noise Resil.",
        "Entanglement", "Concurrence", "Novelty",
    ]

    def _get_vector(m):
        """Extract normalised 6-D vector for radar."""
        total_gates = m.get("total_gates", 100)
        # Invert gate count (lower is better → higher score)
        gate_score = 1.0 / max(1, total_gates) * 10.0
        compress = m.get("compression_ratio", 0)
        noise = m.get("fidelity_p0.01", 0) or 0
        ent = m.get("von_neumann_entropy", 0) or 0
        conc = m.get("concurrence", 0) or 0
        nov = m.get("composite_novelty", 0) or 0
        return [gate_score, compress, noise, ent, conc, nov]

    # Rank candidates by a composite of these metrics
    candidate_scores = []
    for name, m in all_metrics.items():
        if name in standard_names:
            continue
        vec = _get_vector(m)
        score = sum(vec)
        candidate_scores.append((name, score, vec))

    candidate_scores.sort(key=lambda x: -x[1])
    top10 = candidate_scores[:10]

    if len(top10) < 2:
        return

    # Compute max values for normalisation
    all_vecs = [_get_vector(m) for m in all_metrics.values()]
    maxes = [max(v[i] for v in all_vecs) for i in range(6)]
    maxes = [m if m > 0 else 1.0 for m in maxes]

    # Also compute average standard vector
    std_vecs = [_get_vector(all_metrics[sn]) for sn in standard_names if sn in all_metrics]
    if std_vecs:
        avg_std = [np.mean([v[i] for v in std_vecs]) for i in range(6)]
    else:
        avg_std = [0] * 6

    n_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, axes = plt.subplots(2, 5, figsize=(20, 9), subplot_kw={"polar": True})
    axes_flat = axes.flatten()

    for idx, (name, score, vec) in enumerate(top10):
        ax = axes_flat[idx]
        cat = categories.get(name, "unknown")

        # Normalise
        normed = [v / m for v, m in zip(vec, maxes)]
        normed += normed[:1]

        std_normed = [v / m for v, m in zip(avg_std, maxes)]
        std_normed += std_normed[:1]

        ax.fill(angles, normed, color=_get_color(cat), alpha=0.25)
        ax.plot(angles, normed, color=_get_color(cat), linewidth=2)
        ax.plot(angles, std_normed, color="#2166ac", linewidth=1, linestyle="--", alpha=0.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_labels, fontsize=6)
        ax.set_title(name, fontsize=7, pad=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticklabels([])

    fig.suptitle("Top 10 Candidates — Radar Profiles (dashed = avg standard)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "radar_top10.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Phase 8: Report Generation
# ═══════════════════════════════════════════════════════════════════════


def generate_report(
    all_metrics: dict,
    categories: dict,
    standard_names: set,
    winners: list[dict],
    elapsed: float,
) -> str:
    """Generate human-readable comparison report."""
    lines = [
        "=" * 72,
        "COMPARISON REPORT: New Candidates vs Industry Standards",
        "=" * 72,
        "",
        f"Total circuits analysed: {len(all_metrics)}",
        f"  Industry standards:    {len(standard_names)}",
        f"  Discovery candidates:  {len(all_metrics) - len(standard_names)}",
        f"  Elapsed time:          {elapsed:.1f}s",
        "",
    ]

    # ── Gate Efficiency Rankings ──────────────────────────────────────
    lines.append("-" * 72)
    lines.append("GATE EFFICIENCY (ranked by total gates, ascending)")
    lines.append("-" * 72)
    ranked = sorted(
        all_metrics.items(),
        key=lambda x: x[1].get("total_gates", 9999),
    )
    lines.append(f"{'Rank':>4s}  {'Name':40s} {'Gates':>6s} {'Depth':>6s} {'2Q%':>6s} {'Cat':>12s}")
    for rank, (name, m) in enumerate(ranked[:30], 1):
        cat = categories.get(name, "?")
        lines.append(
            f"{rank:4d}  {name:40s} {m.get('total_gates', 0):6d} "
            f"{m.get('depth', 0):6d} {m.get('two_qubit_ratio', 0):6.2f} {cat:>12s}"
        )
    lines.append("")

    # ── ZX Compression Rankings ───────────────────────────────────────
    lines.append("-" * 72)
    lines.append("ZX COMPRESSION (ranked by compression ratio, descending)")
    lines.append("-" * 72)
    ranked = sorted(
        all_metrics.items(),
        key=lambda x: -x[1].get("compression_ratio", 0),
    )
    lines.append(
        f"{'Rank':>4s}  {'Name':40s} {'Raw V':>6s} {'Red V':>6s} "
        f"{'Ratio':>6s} {'T-cnt':>6s} {'Cat':>12s}"
    )
    for rank, (name, m) in enumerate(ranked[:30], 1):
        cat = categories.get(name, "?")
        lines.append(
            f"{rank:4d}  {name:40s} {m.get('raw_vertices', 0):6d} "
            f"{m.get('reduced_vertices', 0):6d} "
            f"{m.get('compression_ratio', 0):6.3f} "
            f"{m.get('t_count', 0):6d} {cat:>12s}"
        )
    lines.append("")

    # ── Noise Resilience Rankings (p=0.01) ────────────────────────────
    lines.append("-" * 72)
    lines.append("NOISE RESILIENCE at p=0.01 (ranked by fidelity, descending)")
    lines.append("-" * 72)
    ranked = sorted(
        all_metrics.items(),
        key=lambda x: -(x[1].get("fidelity_p0.01") or 0),
    )
    lines.append(f"{'Rank':>4s}  {'Name':40s} {'p=0.001':>8s} {'p=0.01':>8s} {'p=0.05':>8s} {'Cat':>12s}")
    for rank, (name, m) in enumerate(ranked[:30], 1):
        cat = categories.get(name, "?")
        f001 = m.get("fidelity_p0.001")
        f01 = m.get("fidelity_p0.01")
        f05 = m.get("fidelity_p0.05")
        lines.append(
            f"{rank:4d}  {name:40s} "
            f"{f001 or 0:8.5f} {f01 or 0:8.5f} {f05 or 0:8.5f} {cat:>12s}"
        )
    lines.append("")

    # ── Entanglement Quality Rankings ─────────────────────────────────
    lines.append("-" * 72)
    lines.append("ENTANGLEMENT QUALITY (ranked by von Neumann entropy, descending)")
    lines.append("-" * 72)
    ent_items = [
        (n, m) for n, m in all_metrics.items()
        if m.get("von_neumann_entropy") is not None
    ]
    ent_items.sort(key=lambda x: -(x[1].get("von_neumann_entropy") or 0))
    lines.append(f"{'Rank':>4s}  {'Name':40s} {'Entropy':>8s} {'Concur':>8s} {'Purity':>8s} {'Cat':>12s}")
    for rank, (name, m) in enumerate(ent_items[:30], 1):
        cat = categories.get(name, "?")
        lines.append(
            f"{rank:4d}  {name:40s} "
            f"{m.get('von_neumann_entropy', 0):8.4f} "
            f"{m.get('concurrence', 0):8.4f} "
            f"{m.get('purity', 0):8.4f} {cat:>12s}"
        )
    lines.append("")

    # ── Winners ───────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("CANDIDATES THAT OUTPERFORM ALL INDUSTRY STANDARDS")
    lines.append("=" * 72)

    if not winners:
        lines.append("  (none found)")
    else:
        # Group by candidate
        by_candidate: dict[str, list[dict]] = {}
        for w in winners:
            by_candidate.setdefault(w["candidate"], []).append(w)

        for cand in sorted(by_candidate, key=lambda c: -len(by_candidate[c])):
            wins = by_candidate[cand]
            lines.append(f"\n  {cand} — beats standards on {len(wins)} metric(s):")
            for w in sorted(wins, key=lambda x: -x["margin"]):
                lines.append(
                    f"    {w['metric']:30s}: "
                    f"{w['candidate_value']:.4f} vs {w['best_standard']}'s "
                    f"{w['standard_value']:.4f} (margin: {w['margin']:+.1%})"
                )

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"Outputs saved to: {OUTPUT_DIR}")
    lines.append("=" * 72)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    # ── Phase 1: Load discovery results + build all circuits ──────────
    print("Phase 1: Loading data and building circuits...")

    discovery_results = json.loads((DISCOVERY_DIR / "results.json").read_text())
    print(f"  Discovery results: {discovery_results['n_candidates']} candidates")

    # Load phylogeny data + rebuild templates for candidate reconstruction
    print("  Rebuilding corpus & templates...")
    t_build = time.time()
    phylo_results, existing_freq_df = load_phylogeny_results()
    survivors = phylo_results["cross_level_survival"]["survivors_at_full_reduce"]
    corpus = build_corpus()
    motifs = discover_motifs(corpus)
    templates = build_template_registry(motifs, existing_freq_df, corpus, survivors)
    print(f"  Corpus/templates rebuilt ({time.time() - t_build:.1f}s)")

    # Build standard circuits
    standards = build_standard_circuits()
    print(f"  Industry standards: {len(standards)} circuits")

    # Rebuild candidate circuits
    candidates = rebuild_candidates(discovery_results, templates)
    print(f"  Rebuilt candidates: {len(candidates)} circuits")

    # Legacy circuits
    legacy = build_legacy_circuits()
    print(f"  Legacy candidates:  {len(legacy)} circuits")

    # Combine all circuits
    all_circuits: list[tuple[str, QuantumCircuit, str]] = []
    all_circuits.extend(standards)
    all_circuits.extend(candidates)
    all_circuits.extend(legacy)

    categories = {}
    for name, _, cat in all_circuits:
        categories[name] = cat if cat != "standard" else cat

    # Standards get "standard" category
    standard_names = set()
    for name, _, _ in standards:
        categories[name] = "standard"
        standard_names.add(name)

    print(f"  Total circuits: {len(all_circuits)}")

    # ── Phase 2: Gate Efficiency ──────────────────────────────────────
    print("\nPhase 2: Computing gate efficiency...")
    all_metrics: dict[str, dict] = {}
    for name, qc, _ in all_circuits:
        ge = compute_gate_efficiency(name, qc)
        all_metrics.setdefault(name, {}).update(ge)

    # ── Phase 3: ZX Compression ───────────────────────────────────────
    print("Phase 3: Computing ZX compression...")
    t3 = time.time()
    for name, qc, _ in all_circuits:
        zx_m = compute_zx_compression(name, qc)
        all_metrics[name].update(zx_m)
    print(f"  Done ({time.time() - t3:.1f}s)")

    # ── Phase 4: Noise Resilience ─────────────────────────────────────
    print("Phase 4: Computing noise resilience...")
    t4 = time.time()
    for name, qc, _ in all_circuits:
        ge = all_metrics[name]
        nr = compute_noise_resilience_analytical(ge)
        # Flatten fidelities into all_metrics
        for p_str, fid in nr["analytical_fidelity"].items():
            all_metrics[name][f"fidelity_p{p_str}"] = fid

    # Exact simulation for small circuits
    n_exact = 0
    for name, qc, _ in all_circuits:
        if qc.num_qubits <= 5:
            exact = compute_noise_resilience_exact(name, qc)
            if exact is not None:
                for p_str, fid in exact["exact_fidelity"].items():
                    if fid is not None:
                        all_metrics[name][f"exact_fidelity_p{p_str}"] = fid
                n_exact += 1

    print(f"  Analytical: {len(all_circuits)}, Exact: {n_exact} ({time.time() - t4:.1f}s)")

    # ── Phase 5: Entanglement Quality ─────────────────────────────────
    print("Phase 5: Computing entanglement quality...")
    t5 = time.time()
    n_ent = 0
    for name, qc, _ in all_circuits:
        eq = compute_entanglement_quality(name, qc)
        if eq is not None:
            all_metrics[name].update(eq)
            n_ent += 1
    print(f"  Computed for {n_ent} circuits ({time.time() - t5:.1f}s)")

    # ── Load novelty scores ───────────────────────────────────────────
    novelty = load_novelty_scores(discovery_results)
    for cname, nov in novelty.items():
        if cname in all_metrics:
            all_metrics[cname].update(nov)

    # ── Phase 6: Find winners ─────────────────────────────────────────
    print("\nPhase 6: Ranking and finding winners...")
    winners = find_winners(all_metrics, standard_names)
    print(f"  Found {len(winners)} winning entries across {len(set(w['candidate'] for w in winners))} candidates")

    # ── Phase 7: Visualize ────────────────────────────────────────────
    print("\nPhase 7: Generating visualizations...")
    plot_gate_efficiency(all_metrics, categories)
    print("  Saved gate_efficiency.png")
    plot_noise_resilience(all_metrics, categories)
    print("  Saved noise_resilience.png")
    plot_entanglement_quality(all_metrics, categories)
    print("  Saved entanglement_quality.png")
    plot_zx_compression(all_metrics, categories)
    print("  Saved zx_compression.png")
    plot_radar_top10(all_metrics, categories, standard_names)
    print("  Saved radar_top10.png")

    # ── Phase 8: Report ───────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\nPhase 8: Generating report...")
    report = generate_report(all_metrics, categories, standard_names, winners, elapsed)
    print(report)

    (OUTPUT_DIR / "comparison_report.txt").write_text(report)

    # JSON results
    json_results = {
        "n_circuits": len(all_metrics),
        "n_standards": len(standard_names),
        "n_candidates": len(all_metrics) - len(standard_names),
        "standards": list(standard_names),
        "metrics": {},
        "winners": winners,
        "elapsed_seconds": elapsed,
    }
    for name, m in all_metrics.items():
        # Filter out non-serialisable values
        json_results["metrics"][name] = {
            k: v for k, v in m.items()
            if isinstance(v, (int, float, str, bool, list, dict, type(None)))
        }

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_results, indent=2, default=str)
    )

    print(f"\nDone in {elapsed:.1f}s. All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
