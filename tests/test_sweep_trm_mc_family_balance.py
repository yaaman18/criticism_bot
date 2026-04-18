from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trm_pipeline import sweep_trm_mc_family_balance


def test_sweep_trm_mc_family_balance_writes_family_and_aggregate_summaries(tmp_path: Path, monkeypatch) -> None:
    captured: list[tuple[str, int]] = []

    def fake_compare_one_seed(**kwargs):
        family = kwargs["episode_family"]
        seed = kwargs["seed"]
        captured.append((family, seed))
        family_bonus = {
            "fragile_boundary": 0.03,
            "vent_edge": 0.01,
            "uncertain_corridor": -0.005,
        }[family]
        assistive_gain = max(0.0, family_bonus)
        return {
            "derived": {
                "best_mode_by_final_homeostasis": "assistive__assistive" if assistive_gain > 0.0 else "analytic__analytic",
                "best_mode_by_mean_homeostasis": "assistive__assistive" if family == "fragile_boundary" else "analytic__analytic",
                "context_gain_by_action_mode": {
                    "analytic": {
                        "final_homeostatic_error_delta": 0.0,
                        "mean_homeostatic_error_delta": 0.0,
                    },
                    "assistive": {
                        "final_homeostatic_error_delta": assistive_gain,
                        "mean_homeostatic_error_delta": assistive_gain / 2.0,
                    },
                    "module_primary": {
                        "final_homeostatic_error_delta": assistive_gain / 2.0,
                        "mean_homeostatic_error_delta": assistive_gain / 4.0,
                    },
                },
            }
        }

    monkeypatch.setattr(sweep_trm_mc_family_balance, "_compare_one_seed", fake_compare_one_seed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_trm_mc_family_balance",
            "--output-root",
            str(tmp_path / "mc_family_balance"),
            "--seed-start",
            "700",
            "--num-seeds",
            "2",
            "--families",
            "fragile_boundary",
            "vent_edge",
            "uncertain_corridor",
        ],
    )

    sweep_trm_mc_family_balance.main()

    assert captured == [
        ("fragile_boundary", 700),
        ("fragile_boundary", 701),
        ("vent_edge", 700),
        ("vent_edge", 701),
        ("uncertain_corridor", 700),
        ("uncertain_corridor", 701),
    ]

    aggregate = json.loads((tmp_path / "mc_family_balance" / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert aggregate["families"] == ["fragile_boundary", "vent_edge", "uncertain_corridor"]
    assert aggregate["counts_by_best_final_homeostasis"] == {"assistive__assistive": 4, "analytic__analytic": 2}
    assert aggregate["counts_by_best_mean_homeostasis"] == {"assistive__assistive": 2, "analytic__analytic": 4}
    assert aggregate["balanced_mean_context_gain_by_action_mode"]["assistive"]["final_homeostatic_error_delta"] == pytest.approx(
        (0.03 + 0.01 + 0.0) / 3.0
    )
    assert (tmp_path / "mc_family_balance" / "fragile_boundary" / "family_summary.json").exists()
    assert (tmp_path / "mc_family_balance" / "vent_edge" / "family_summary.json").exists()
    assert (tmp_path / "mc_family_balance" / "uncertain_corridor" / "family_summary.json").exists()
