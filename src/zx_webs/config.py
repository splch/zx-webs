"""Pydantic v2 configuration models and YAML helpers for ZX-Webs."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class CorpusConfig(BaseModel):
    """Stage 1 -- corpus generation parameters."""

    families: list[str] = [
        "oracular",
        "arithmetic",
        "variational",
        "simulation",
        "entanglement",
        "error_correction",
        "linear_algebra",
        "communication",
    ]
    max_qubits: int = 10
    qubit_counts: list[int] = [3, 5, 7, 10]
    seed: int = 42


class ZXConfig(BaseModel):
    """Stage 2 -- ZX-diagram extraction / simplification."""

    reduction: str = "full_reduce"
    mining_reduction: str = "teleport_reduce"  # reduction used for Stage 3 input
    normalize: bool = True


class MiningConfig(BaseModel):
    """Stage 3 -- frequent sub-graph mining.

    **Performance guidance** -- runtime scales exponentially with
    ``max_vertices`` at low ``min_support``.  Benchmark data on a
    294-graph corpus (<=50 vertices each):

    ==========  =========  ===========  ==========
    min_support  max_v=6   max_v=12     max_v=16
    ==========  =========  ===========  ==========
    2-3          835K/246s  intractable  intractable
    10           29K/48s    intractable  intractable
    20           1.2K/2.7s  2.2K/22s     2.2K/22s
    30           296/1.0s   421/3.5s     421/3.5s
    ==========  =========  ===========  ==========

    Pattern counts saturate around max_v=12-14 for most corpora,
    so increasing beyond that adds no new patterns.
    """

    min_support: int = 3
    min_vertices: int = 2
    max_vertices: int = 20
    max_input_vertices: int = 50  # skip graphs larger than this for gSpan
    phase_discretization: int = 8
    include_phase_in_label: bool = True
    mining_reduction: str = "teleport_reduce"  # "full_reduce", "teleport_reduce", or "none"
    mining_timeout: int = 0  # seconds; 0 = no timeout (runs inline)
    discriminative_mining: bool = False  # mine family-specific rare patterns
    discriminative_min_support: int = 2  # lower support threshold for per-family mining
    discriminative_max_family_ratio: float = 0.5  # max fraction of families a pattern can appear in


class ComposeConfig(BaseModel):
    """Stage 4 -- candidate circuit composition."""

    max_webs_per_candidate: int = 3
    max_candidates: int = 10000
    max_webs_loaded: int = 50000  # cap on webs loaded for composition (FPS + pair gen)
    composition_modes: list[str] = ["sequential", "parallel"]
    min_compose_qubits: int = 2
    max_compose_qubits: int = 20  # upper bound on qubits for composed candidates
    seed: int = 42
    prefer_cross_family: bool = False  # neutral baseline: no cross-family bias
    guided: bool = False  # enable target-guided composition
    target_qubit_counts: list[int] = [3, 5, 7]  # qubit counts to target when guided
    phase_perturbation_resolution: int = 8  # k*pi/N for k in 0..2N-1
    phase_perturbation_rate: float = 0.3  # probability of perturbing each spider
    continuous_phase_perturbation: bool = False  # use uniform random phases instead of discrete palette
    # BYOL-Explore curiosity-driven composition
    compose_strategy: str = "standard"  # "standard", "byol", or "hybrid"
    byol_episodes: int = 5  # number of BYOL-Explore exploration episodes
    byol_steps_per_episode: int = 200  # composition attempts per episode
    byol_budget_fraction: float = 0.5  # fraction of max_candidates allocated to BYOL in hybrid mode
    qaoa_reward_weight: float = 0.0  # blend weight for problem-driven QAOA fitness in BYOL reward (0=pure curiosity, 1=pure QAOA)


class FilterConfig(BaseModel):
    """Stage 5 -- candidate filtering and deduplication."""

    extract_timeout_seconds: float = 30.0
    max_cnot_blowup_factor: float = 5.0
    cnot_blowup_enabled: bool = True  # when False, skip the CNOT blowup check
    dedup_method: str = "unitary"
    n_workers: int = 0  # 0 = auto (os.cpu_count()), 1 = sequential
    optimize_cnots_level: int = 2  # passed to extract_circuit optimize_cnots
    gflow_precheck: bool = False  # when False (default), skip gflow pre-filter
    max_unitary_qubits: int = 10  # max qubits for unitary matrix computation
    zx_anneal: bool = False  # post-extraction ZX rewrite search via simulated annealing
    zx_anneal_iters: int = 200  # iterations per candidate for ZX annealing
    zx_anneal_max_qubits: int = 6  # only anneal circuits with <= this many qubits


class BackendConfig(BaseModel):
    """Configuration for a single execution backend."""

    name: str = "aer_ideal"
    type: str = "simulator"  # "simulator" or "qpu"
    provider: str = "aer"  # "aer", "ibm", "ionq", "braket", etc.
    config: dict = {}
    enabled: bool = True


class BenchConfig(BaseModel):
    """Stage 6 -- benchmarking surviving candidates."""

    suites: list[str] = ["qasmbench", "supermarq"]
    tasks: list[str] = [
        "grover_oracle",
        "maxcut",
        "qpe",
        "ghz",
        "arithmetic",
    ]
    qasmbench_path: Path = Path("data/qasmbench")
    supermarq_qubits: list[int] = [3, 5, 8]
    fidelity_shots: int = 8192
    fidelity_threshold: float = 0.99
    novelty_scoring: bool = False  # compute novelty score (distance from all known unitaries)
    max_unitary_qubits: int = 10  # max qubits for unitary matrix computation
    backends: list[BackendConfig] = [
        BackendConfig(name="aer_ideal", type="simulator", provider="aer"),
    ]
    # Usefulness scoring: test circuits for computational utility, not just novelty.
    usefulness_scoring: bool = False
    usefulness_weight: float = 0.5  # weight in composite fitness (vs novelty)
    magic_max_qubits: int = 4  # max qubits for exact stabiliser state enumeration
    scrambling_samples: int = 10  # random Pauli pairs for OTOC estimation


class ReportConfig(BaseModel):
    """Stage 7 -- report generation."""

    output_format: list[str] = ["json", "html"]
    figure_dpi: int = 150


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Root configuration for the entire ZX-Webs pipeline."""

    data_dir: Path = Path("data")
    corpus: CorpusConfig = CorpusConfig()
    zx: ZXConfig = ZXConfig()
    mining: MiningConfig = MiningConfig()
    compose: ComposeConfig = ComposeConfig()
    filter: FilterConfig = FilterConfig()
    bench: BenchConfig = BenchConfig()
    report: ReportConfig = ReportConfig()
    # Iterative refinement: feed survivors back as corpus entries for re-mining.
    refinement_rounds: int = 0  # 0 = no refinement (single pass)
    refinement_top_k: int = 50  # how many top survivors to feed back per round
    # Fitness-guided evolutionary search (active when refinement_rounds > 0).
    fitness_guided: bool = True  # use fitness profiles to bias web selection
    fitness_decay: float = 0.7  # exponential decay for older fitness signals
    phase_optimize_near_misses: bool = True  # run Nelder-Mead on near-miss candidates
    phase_optimize_max_iters: int = 200  # Nelder-Mead iterations per near-miss
    near_miss_fidelity_lo: float = 0.80  # lower bound of near-miss fidelity window
    near_miss_fidelity_hi: float = 0.99  # upper bound of near-miss fidelity window


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def load_config(path: Path) -> PipelineConfig:
    """Read a YAML file and return a validated ``PipelineConfig``."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return PipelineConfig.model_validate(raw)


def save_config(config: PipelineConfig, path: Path) -> None:
    """Serialize a ``PipelineConfig`` to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    # Convert Path objects that were serialized as strings back for clean YAML
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
