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
    """Stage 3 -- frequent sub-graph mining."""

    min_support: int = 3
    min_vertices: int = 2
    max_vertices: int = 20
    max_input_vertices: int = 50  # skip graphs larger than this for gSpan
    phase_discretization: int = 8
    include_phase_in_label: bool = True
    mining_reduction: str = "teleport_reduce"  # "full_reduce", "teleport_reduce", or "none"


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
    max_unitary_qubits: int = 10  # max qubits for unitary matrix computation
    backends: list[BackendConfig] = [
        BackendConfig(name="aer_ideal", type="simulator", provider="aer"),
    ]


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
