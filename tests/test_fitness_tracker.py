"""Tests for the fitness tracker module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zx_webs.fitness_tracker import (
    FitnessProfile,
    _compute_candidate_fitness,
    _family_pair_key,
    build_fitness_profile,
    load_fitness_profile,
    merge_profiles,
    save_fitness_profile,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestFamilyPairKey:
    def test_sorted_and_deduped(self):
        assert _family_pair_key(["oracular", "arithmetic"]) == "arithmetic+oracular"

    def test_single_family(self):
        assert _family_pair_key(["variational"]) == "variational"

    def test_duplicates_removed(self):
        assert _family_pair_key(["a", "a", "b"]) == "a+b"


class TestComputeCandidateFitness:
    def test_zero_for_empty_result(self):
        assert _compute_candidate_fitness({}) == 0.0

    def test_novelty_contributes(self):
        score = _compute_candidate_fitness({"novelty_score": 0.8})
        assert score == pytest.approx(0.8)

    def test_near_miss_bonus(self):
        # Fidelity 0.90 is in the [0.80, 0.99) window.
        score = _compute_candidate_fitness({"best_fidelity": 0.90})
        # near_miss_bonus = 0.5 * (0.90 - 0.80) / (0.99 - 0.80)
        expected = 0.5 * 0.10 / 0.19
        assert score == pytest.approx(expected, abs=1e-6)

    def test_improvement_bonus(self):
        score = _compute_candidate_fitness({"n_improvements": 2})
        assert score == pytest.approx(1.0)

    def test_combined_score(self):
        result = {
            "novelty_score": 0.5,
            "best_fidelity": 0.85,
            "n_improvements": 1,
        }
        score = _compute_candidate_fitness(result)
        near_miss = 0.5 * (0.85 - 0.80) / (0.99 - 0.80)
        assert score == pytest.approx(0.5 + near_miss + 1.0, abs=1e-6)

    def test_no_near_miss_outside_window(self):
        # Fidelity below the near-miss window.
        score = _compute_candidate_fitness({"best_fidelity": 0.70})
        assert score == pytest.approx(0.0)

        # Fidelity at/above the threshold (no bonus -- it's an exact match).
        score = _compute_candidate_fitness({"best_fidelity": 0.99})
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests for build_fitness_profile
# ---------------------------------------------------------------------------


class TestBuildFitnessProfile:
    def test_empty_inputs(self):
        profile = build_fitness_profile([], [])
        assert profile.web_fitness == {}
        assert profile.near_miss_candidates == []

    def test_attributes_fitness_to_webs(self):
        bench_results = [
            {
                "survivor_id": "surv_0001",
                "novelty_score": 0.6,
                "best_fidelity": 0.5,
                "n_improvements": 0,
                "task_matches": [],
            },
        ]
        cand_manifest = [
            {
                "candidate_id": "surv_0001",
                "component_web_ids": ["web_001", "web_002"],
                "composition_type": "sequential",
                "source_families": ["oracular", "arithmetic"],
            },
        ]
        profile = build_fitness_profile(bench_results, cand_manifest)

        # Each web gets fitness / n_webs.
        per_web = 0.6 / 2
        assert profile.web_fitness["web_001"] == pytest.approx(per_web)
        assert profile.web_fitness["web_002"] == pytest.approx(per_web)
        assert profile.strategy_fitness["sequential"] == pytest.approx(0.6)
        assert "arithmetic+oracular" in profile.family_pair_fitness

    def test_near_miss_collection(self):
        bench_results = [
            {
                "survivor_id": "surv_0042",
                "novelty_score": 0.0,
                "best_fidelity": 0.92,
                "n_improvements": 0,
                "task_matches": [
                    {"task_name": "grover_oracle", "fidelity": 0.92},
                ],
            },
        ]
        cand_manifest = [
            {
                "candidate_id": "surv_0042",
                "component_web_ids": ["web_010"],
                "composition_type": "parallel",
                "source_families": ["oracular"],
            },
        ]
        profile = build_fitness_profile(bench_results, cand_manifest)

        assert len(profile.near_miss_candidates) == 1
        nm = profile.near_miss_candidates[0]
        assert nm["survivor_id"] == "surv_0042"
        assert nm["target_task"] == "grover_oracle"


# ---------------------------------------------------------------------------
# Tests for FitnessProfile serialisation
# ---------------------------------------------------------------------------


class TestFitnessProfileSerde:
    def test_round_trip(self, tmp_path: Path):
        profile = FitnessProfile(
            web_fitness={"web_1": 2.5, "web_2": 1.0},
            strategy_fitness={"sequential": 3.0},
            family_pair_fitness={"a+b": 1.5},
            near_miss_candidates=[
                {"survivor_id": "s1", "best_fidelity": 0.9},
            ],
            round_num=1,
        )
        save_fitness_profile(profile, tmp_path)
        loaded = load_fitness_profile(tmp_path)
        assert loaded is not None
        assert loaded.web_fitness == profile.web_fitness
        assert loaded.strategy_fitness == profile.strategy_fitness
        assert loaded.round_num == 1
        assert len(loaded.near_miss_candidates) == 1

    def test_load_nonexistent(self, tmp_path: Path):
        assert load_fitness_profile(tmp_path) is None


# ---------------------------------------------------------------------------
# Tests for merge_profiles
# ---------------------------------------------------------------------------


class TestMergeProfiles:
    def test_merge_none_returns_new(self):
        new = FitnessProfile(web_fitness={"w1": 1.0}, round_num=1)
        merged = merge_profiles(None, new)
        assert merged.web_fitness == {"w1": 1.0}

    def test_decay_applied(self):
        old = FitnessProfile(
            web_fitness={"w1": 10.0, "w2": 5.0},
            strategy_fitness={"sequential": 8.0},
        )
        new = FitnessProfile(
            web_fitness={"w1": 2.0, "w3": 3.0},
            strategy_fitness={"parallel": 4.0},
            round_num=1,
        )
        merged = merge_profiles(old, new, decay=0.5)

        # w1: old * 0.5 + new = 5.0 + 2.0 = 7.0
        assert merged.web_fitness["w1"] == pytest.approx(7.0)
        # w2: old * 0.5 + 0 = 2.5
        assert merged.web_fitness["w2"] == pytest.approx(2.5)
        # w3: 0 + new = 3.0
        assert merged.web_fitness["w3"] == pytest.approx(3.0)
        # sequential: old * 0.5 = 4.0
        assert merged.strategy_fitness["sequential"] == pytest.approx(4.0)
        # parallel: new = 4.0
        assert merged.strategy_fitness["parallel"] == pytest.approx(4.0)
        assert merged.round_num == 1
