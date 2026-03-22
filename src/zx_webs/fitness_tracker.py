"""Fitness-guided recipe attribution for evolutionary composition.

Tracks which web IDs, composition strategies, and family combinations
produced high-fitness survivors in benchmarking.  The resulting
:class:`FitnessProfile` is consumed by Stage 4 to bias web selection
and pairing toward historically productive recipes.

Fitness is a composite score combining:
- **Novelty**: distance from known corpus unitaries (higher = more novel)
- **Near-miss potential**: fidelity in [0.8, 0.99) to a known task
  (these candidates almost implement something useful)
- **Gate efficiency**: Pareto improvements in T-count / CNOT / depth
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zx_webs.persistence import load_json, save_json

logger = logging.getLogger(__name__)

PROFILE_FILENAME = "fitness_profile.json"


@dataclass
class FitnessProfile:
    """Accumulated fitness signal from benchmarking results.

    Attributes
    ----------
    web_fitness : dict[str, float]
        Mapping from web_id to accumulated fitness score.  Webs that
        contributed to high-fitness survivors get higher scores.
    strategy_fitness : dict[str, float]
        Mapping from composition_type to accumulated fitness.
    family_pair_fitness : dict[str, float]
        Mapping from sorted family-pair key (e.g. "arithmetic+oracular")
        to accumulated fitness.
    near_miss_candidates : list[dict]
        Candidates with fidelity in [near_miss_lo, near_miss_hi) to a
        known task — prime targets for phase optimisation.
    round_num : int
        Which refinement round produced this profile.
    """

    web_fitness: dict[str, float] = field(default_factory=dict)
    strategy_fitness: dict[str, float] = field(default_factory=dict)
    family_pair_fitness: dict[str, float] = field(default_factory=dict)
    near_miss_candidates: list[dict[str, Any]] = field(default_factory=list)
    round_num: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "web_fitness": self.web_fitness,
            "strategy_fitness": self.strategy_fitness,
            "family_pair_fitness": self.family_pair_fitness,
            "near_miss_candidates": self.near_miss_candidates,
            "round_num": self.round_num,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FitnessProfile:
        return cls(
            web_fitness=d.get("web_fitness", {}),
            strategy_fitness=d.get("strategy_fitness", {}),
            family_pair_fitness=d.get("family_pair_fitness", {}),
            near_miss_candidates=d.get("near_miss_candidates", []),
            round_num=d.get("round_num", 0),
        )


def _family_pair_key(families: list[str]) -> str:
    """Canonical key for a set of source families."""
    return "+".join(sorted(set(families)))


def _compute_candidate_fitness(
    bench_result: dict[str, Any],
    near_miss_lo: float = 0.80,
    near_miss_hi: float = 0.99,
) -> float:
    """Compute a composite fitness score for a single benchmarked survivor.

    Components (all in [0, 1], summed):
    - novelty_score (directly from benchmarking)
    - near_miss_bonus: 0.5 * (fidelity - near_miss_lo) / (near_miss_hi - near_miss_lo)
      for candidates in the near-miss fidelity window
    - improvement_bonus: 1.0 if the candidate is a real improvement over
      a baseline (high fidelity + fewer gates)
    """
    score = 0.0

    # Novelty component
    novelty = bench_result.get("novelty_score", 0.0)
    score += novelty

    # Near-miss bonus: candidates close to implementing a known algorithm
    best_fidelity = bench_result.get("best_fidelity", 0.0)
    if near_miss_lo <= best_fidelity < near_miss_hi:
        # Scale within the near-miss window
        near_miss_bonus = 0.5 * (best_fidelity - near_miss_lo) / (
            near_miss_hi - near_miss_lo
        )
        score += near_miss_bonus

    # Real improvement bonus
    if bench_result.get("n_improvements", 0) > 0:
        score += 1.0

    return score


def build_fitness_profile(
    bench_results: list[dict[str, Any]],
    candidate_manifest: list[dict[str, Any]],
    round_num: int = 0,
    near_miss_lo: float = 0.80,
    near_miss_hi: float = 0.99,
) -> FitnessProfile:
    """Build a fitness profile from Stage 6 benchmark results.

    Parameters
    ----------
    bench_results:
        Stage 6 results (one entry per survivor with metrics, fidelity,
        novelty, etc.)
    candidate_manifest:
        Stage 4 candidate manifest entries (links survivor_id back to
        component_web_ids, composition_type, and source_families).
    round_num:
        Current refinement round.
    near_miss_lo, near_miss_hi:
        Fidelity window for near-miss candidates.

    Returns
    -------
    FitnessProfile
    """
    profile = FitnessProfile(round_num=round_num)

    # Build lookup from candidate_id / survivor_id to candidate metadata.
    cand_lookup: dict[str, dict[str, Any]] = {}
    for entry in candidate_manifest:
        cid = entry.get("candidate_id", entry.get("survivor_id", ""))
        if cid:
            cand_lookup[cid] = entry

    for result in bench_results:
        survivor_id = result.get("survivor_id", "")
        fitness = _compute_candidate_fitness(
            result, near_miss_lo=near_miss_lo, near_miss_hi=near_miss_hi,
        )

        if fitness <= 0:
            continue

        # Look up the candidate metadata for provenance info.
        cand_meta = cand_lookup.get(survivor_id, {})
        web_ids = cand_meta.get("component_web_ids", [])
        comp_type = cand_meta.get("composition_type", "unknown")
        families = cand_meta.get("source_families", [])

        # Attribute fitness to component webs.
        per_web = fitness / max(len(web_ids), 1)
        for wid in web_ids:
            profile.web_fitness[wid] = (
                profile.web_fitness.get(wid, 0.0) + per_web
            )

        # Attribute to composition strategy.
        profile.strategy_fitness[comp_type] = (
            profile.strategy_fitness.get(comp_type, 0.0) + fitness
        )

        # Attribute to family pair.
        if len(families) >= 2:
            fkey = _family_pair_key(families)
            profile.family_pair_fitness[fkey] = (
                profile.family_pair_fitness.get(fkey, 0.0) + fitness
            )

        # Collect near-miss candidates for phase optimisation.
        best_fidelity = result.get("best_fidelity", 0.0)
        if near_miss_lo <= best_fidelity < near_miss_hi:
            # Find the task match with highest fidelity for targeting.
            matches = result.get("task_matches", [])
            best_match = max(matches, key=lambda m: m.get("fidelity", 0.0)) if matches else {}
            profile.near_miss_candidates.append({
                "survivor_id": survivor_id,
                "best_fidelity": best_fidelity,
                "target_task": best_match.get("task_name", ""),
                "n_qubits": result.get("n_qubits", 0),
                "component_web_ids": web_ids,
                "composition_type": comp_type,
            })

    # Log summary.
    n_webs_scored = len(profile.web_fitness)
    n_strategies = len(profile.strategy_fitness)
    n_near_misses = len(profile.near_miss_candidates)
    logger.info(
        "Fitness profile (round %d): %d webs scored, %d strategies, "
        "%d near-miss candidates for phase optimisation.",
        round_num, n_webs_scored, n_strategies, n_near_misses,
    )

    if profile.web_fitness:
        top_webs = sorted(
            profile.web_fitness.items(), key=lambda x: x[1], reverse=True,
        )[:5]
        logger.info("Top webs: %s", top_webs)

    if profile.strategy_fitness:
        logger.info("Strategy fitness: %s", dict(profile.strategy_fitness))

    return profile


def save_fitness_profile(profile: FitnessProfile, data_dir: Path) -> Path:
    """Persist a fitness profile to the data directory."""
    path = data_dir / PROFILE_FILENAME
    save_json(profile.to_dict(), path)
    return path


def load_fitness_profile(data_dir: Path) -> FitnessProfile | None:
    """Load a fitness profile from the data directory, if it exists."""
    path = data_dir / PROFILE_FILENAME
    if not path.exists():
        return None
    data = load_json(path)
    return FitnessProfile.from_dict(data)


def merge_profiles(
    existing: FitnessProfile | None,
    new: FitnessProfile,
    decay: float = 0.7,
) -> FitnessProfile:
    """Merge a new profile into an existing one with exponential decay.

    Older fitness signals are decayed by *decay* before adding the new
    round's signals.  This ensures recent discoveries have more influence
    while retaining memory of historically productive recipes.
    """
    if existing is None:
        return new

    merged = FitnessProfile(round_num=new.round_num)

    # Decay and merge web fitness.
    all_web_ids = set(existing.web_fitness) | set(new.web_fitness)
    for wid in all_web_ids:
        old_val = existing.web_fitness.get(wid, 0.0) * decay
        new_val = new.web_fitness.get(wid, 0.0)
        merged.web_fitness[wid] = old_val + new_val

    # Decay and merge strategy fitness.
    all_strategies = set(existing.strategy_fitness) | set(new.strategy_fitness)
    for s in all_strategies:
        old_val = existing.strategy_fitness.get(s, 0.0) * decay
        new_val = new.strategy_fitness.get(s, 0.0)
        merged.strategy_fitness[s] = old_val + new_val

    # Decay and merge family pair fitness.
    all_pairs = set(existing.family_pair_fitness) | set(new.family_pair_fitness)
    for fp in all_pairs:
        old_val = existing.family_pair_fitness.get(fp, 0.0) * decay
        new_val = new.family_pair_fitness.get(fp, 0.0)
        merged.family_pair_fitness[fp] = old_val + new_val

    # Keep only the new round's near-miss candidates.
    merged.near_miss_candidates = new.near_miss_candidates

    return merged
