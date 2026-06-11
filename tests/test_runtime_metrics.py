from __future__ import annotations

from dataclasses import dataclass

import pytest

from trm_pipeline.runtime_metrics import death_cause_counts, episode_metrics, homeostatic_error


@dataclass(frozen=True)
class _Cfg:
    G_target: float = 0.55
    B_target: float = 0.65
    G0: float = 0.4
    B0: float = 0.5
    steps: int = 16


def test_homeostatic_error_matches_absolute_distance_sum() -> None:
    cfg = _Cfg(G_target=0.6, B_target=0.7)
    assert homeostatic_error(0.4, 0.9, cfg) == pytest.approx(0.4)


def test_death_cause_counts_counts_only_dead_rows() -> None:
    history = [
        {"dead": False, "death_cause": "expected"},
        {"dead": True, "death_cause": "expected"},
        {"dead": True, "death_cause": "degenerate"},
        {"dead": True, "death_cause": "unknown"},
    ]
    counts = death_cause_counts(
        history,
        expected_label="expected",
        degenerate_label="degenerate",
        policy_forbidden_label="policy_forbidden",
    )

    assert counts == {
        "expected": 1,
        "degenerate": 1,
        "policy_forbidden": 0,
    }


def test_episode_metrics_empty_history_returns_safe_defaults() -> None:
    cfg = _Cfg()
    metrics = episode_metrics(
        [],
        cfg,
        actions=("approach", "withdraw"),
        policy_action_cost=lambda _: 0.0,
        expected_death_label="expected",
        degenerate_death_label="degenerate",
        policy_forbidden_death_label="policy_forbidden",
    )

    assert metrics["survival_fraction"] == 0.0
    assert metrics["mean_G"] == 0.0
    assert metrics["mean_B"] == 0.0
    assert metrics["final_homeostatic_error"] == pytest.approx(0.3)
    assert metrics["mean_homeostatic_error"] == pytest.approx(0.3)


def test_episode_metrics_keeps_resource_hazard_shelter_aliases() -> None:
    cfg = _Cfg(steps=1)
    history = [
        {
            "G": 0.55,
            "B": 0.65,
            "action": "approach",
            "policy_entropy": 0.0,
            "contact_energy": 0.7,
            "contact_thermal": 0.5,
            "contact_toxicity": 0.25,
            "contact_niche": 0.4,
            "contact_species_energy": 0.0,
            "contact_species_thermal": 0.0,
            "contact_species_toxicity": 0.0,
            "contact_species_niche": 0.0,
            "dead": False,
        }
    ]

    metrics = episode_metrics(
        history,
        cfg,
        actions=("approach", "withdraw"),
        policy_action_cost=lambda _: 0.0,
        expected_death_label="expected",
        degenerate_death_label="degenerate",
        policy_forbidden_death_label="policy_forbidden",
    )

    assert metrics["mean_contact_energy"] == pytest.approx(0.7)
    assert metrics["mean_contact_thermal"] == pytest.approx(0.5)
    assert metrics["mean_contact_toxicity"] == pytest.approx(0.25)
    assert metrics["mean_contact_niche"] == pytest.approx(0.4)
    assert metrics["mean_contact_resource"] == pytest.approx(0.7)
    assert metrics["mean_contact_hazard"] == pytest.approx(0.4)
    assert metrics["mean_contact_shelter"] == pytest.approx(0.4)
