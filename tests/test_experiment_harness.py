from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from trm_pipeline import experiment_harness
from trm_pipeline.common import save_json


def _write_seed_compare(
    compare_root: Path,
    *,
    seed: int,
    candidate_mode: str,
    baseline_mode: str,
    candidate_final: float,
    candidate_mean: float,
    baseline_final: float,
    baseline_mean: float,
    candidate_actions: list[str],
    candidate_contacts: list[tuple[float, float]],
    best_mode: str,
) -> None:
    seed_root = compare_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    results = {
        candidate_mode: {
            "dead": False,
            "survival_fraction": 1.0,
            "final_homeostatic_error": candidate_final,
            "mean_homeostatic_error": candidate_mean,
            "action_cost_total": 0.2,
            "runtime_config": {"G_target": 0.55, "B_target": 0.65},
        },
        baseline_mode: {
            "dead": False,
            "survival_fraction": 1.0,
            "final_homeostatic_error": baseline_final,
            "mean_homeostatic_error": baseline_mean,
            "action_cost_total": 0.2,
            "runtime_config": {"G_target": 0.55, "B_target": 0.65},
        },
    }
    save_json(
        seed_root / "comparison_summary.json",
        {
            "results": results,
            "derived": {
                "best_mode_by_final_homeostasis": best_mode,
                "best_mode_by_mean_homeostasis": best_mode,
            },
        },
    )
    candidate_history = []
    for index, action in enumerate(candidate_actions):
        thermal, toxicity = candidate_contacts[index]
        candidate_history.append(
            {
                "t": index,
                "G": 0.60,
                "B": 0.62,
                "action": action,
                "contact_thermal": thermal,
                "contact_toxicity": toxicity,
            }
        )
    baseline_history = [
        {
            "t": 0,
            "G": 0.62,
            "B": 0.60,
            "action": "withdraw",
            "contact_thermal": 0.45,
            "contact_toxicity": 0.40,
        }
    ]
    save_json(seed_root / candidate_mode / "episode_history.json", candidate_history)
    save_json(seed_root / baseline_mode / "episode_history.json", baseline_history)


def test_build_experiment_contract_sets_artifact_paths(tmp_path: Path) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "harness_run",
        experiment_name="vm_as_stage1",
    )

    assert contract["experiment_name"] == "vm_as_stage1"
    assert contract["candidate_mode"] == "analytic__module_primary"
    assert contract["baseline_mode"] == "analytic__analytic"
    assert Path(contract["artifacts"]["contract"]) == tmp_path / "harness_run" / "contract.json"
    assert Path(contract["artifacts"]["compare_root"]) == tmp_path / "harness_run" / "compare"
    assert contract["acceptance"]["min_best_mode_frequency"] == 0.60
    assert [track["name"] for track in contract["family_tracks"]] == list(experiment_harness.DEFAULT_FAMILY_ORDER)


def test_evaluate_contract_reports_pass_for_candidate_mode(tmp_path: Path) -> None:
    output_root = tmp_path / "harness_eval"
    contract = experiment_harness.build_experiment_contract(
        output_root=output_root,
        experiment_name="candidate_promotion",
        seed_start=10,
        num_seeds=2,
        family_tracks=[
            {
                "name": "energy_starved",
                "promotion_target": "Promote low-energy recovery when candidate beats baseline cleanly.",
            },
            {
                "name": "toxic_band",
                "promotion_target": "Promote toxic-band robustness when defensive behavior stays high.",
                "acceptance_overrides": {"min_stress_defensive_rate": 0.50},
            },
        ],
    )
    compare_root = Path(contract["artifacts"]["compare_root"])
    energy_root = compare_root / "energy_starved"
    toxic_root = compare_root / "toxic_band"
    _write_seed_compare(
        energy_root,
        seed=10,
        candidate_mode=contract["candidate_mode"],
        baseline_mode=contract["baseline_mode"],
        candidate_final=0.20,
        candidate_mean=0.22,
        baseline_final=0.34,
        baseline_mean=0.30,
        candidate_actions=["withdraw", "reconfigure", "withdraw"],
        candidate_contacts=[(0.45, 0.40), (0.50, 0.42), (0.47, 0.41)],
        best_mode=contract["candidate_mode"],
    )
    _write_seed_compare(
        energy_root,
        seed=11,
        candidate_mode=contract["candidate_mode"],
        baseline_mode=contract["baseline_mode"],
        candidate_final=0.24,
        candidate_mean=0.25,
        baseline_final=0.36,
        baseline_mean=0.31,
        candidate_actions=["reconfigure", "withdraw", "seal"],
        candidate_contacts=[(0.46, 0.40), (0.44, 0.39), (0.05, 0.05)],
        best_mode=contract["candidate_mode"],
    )
    _write_seed_compare(
        toxic_root,
        seed=10,
        candidate_mode=contract["candidate_mode"],
        baseline_mode=contract["baseline_mode"],
        candidate_final=0.28,
        candidate_mean=0.27,
        baseline_final=0.35,
        baseline_mean=0.32,
        candidate_actions=["withdraw", "reconfigure", "withdraw"],
        candidate_contacts=[(0.48, 0.42), (0.51, 0.43), (0.47, 0.41)],
        best_mode=contract["candidate_mode"],
    )
    _write_seed_compare(
        toxic_root,
        seed=11,
        candidate_mode=contract["candidate_mode"],
        baseline_mode=contract["baseline_mode"],
        candidate_final=0.29,
        candidate_mean=0.28,
        baseline_final=0.36,
        baseline_mean=0.33,
        candidate_actions=["reconfigure", "withdraw", "withdraw"],
        candidate_contacts=[(0.49, 0.43), (0.50, 0.42), (0.48, 0.41)],
        best_mode=contract["candidate_mode"],
    )
    save_json(
        energy_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "seed_start": 10,
            "num_seeds": 2,
            "policy_mode": "closed_loop",
            "module_manifest": None,
            "counts_by_best_final_homeostasis": {contract["candidate_mode"]: 2},
            "counts_by_best_mean_homeostasis": {contract["candidate_mode"]: 2},
            "per_seed": [
                {
                    "seed": 10,
                    "best_mode_by_final_homeostasis": contract["candidate_mode"],
                    "best_mode_by_mean_homeostasis": contract["candidate_mode"],
                },
                {
                    "seed": 11,
                    "best_mode_by_final_homeostasis": contract["candidate_mode"],
                    "best_mode_by_mean_homeostasis": contract["candidate_mode"],
                },
            ],
        },
    )
    save_json(
        toxic_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "seed_start": 10,
            "num_seeds": 2,
            "policy_mode": "closed_loop",
            "module_manifest": None,
            "counts_by_best_final_homeostasis": {contract["candidate_mode"]: 2},
            "counts_by_best_mean_homeostasis": {contract["candidate_mode"]: 2},
            "per_seed": [
                {
                    "seed": 10,
                    "best_mode_by_final_homeostasis": contract["candidate_mode"],
                    "best_mode_by_mean_homeostasis": contract["candidate_mode"],
                },
                {
                    "seed": 11,
                    "best_mode_by_final_homeostasis": contract["candidate_mode"],
                    "best_mode_by_mean_homeostasis": contract["candidate_mode"],
                },
            ],
        },
    )
    save_json(
        compare_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "compare_root": str(compare_root),
            "family_order": ["energy_starved", "toxic_band"],
            "families": {
                "energy_starved": {
                    "aggregate_summary": str(energy_root / "aggregate_summary.json"),
                    "required_for_promotion": True,
                    "promotion_target": "Promote low-energy recovery when candidate beats baseline cleanly.",
                },
                "toxic_band": {
                    "aggregate_summary": str(toxic_root / "aggregate_summary.json"),
                    "required_for_promotion": True,
                    "promotion_target": "Promote toxic-band robustness when defensive behavior stays high.",
                },
            },
        },
    )

    report = experiment_harness.evaluate_contract(contract)

    assert report["overall_pass"] is True
    assert report["inspected_seeds_total"] == 4
    assert report["eligible_family_tracks"] == ["energy_starved", "toxic_band"]
    assert report["family_reports"]["energy_starved"]["summary"]["final_improvement_vs_baseline"] > 0.0
    assert report["family_reports"]["toxic_band"]["criteria"]["stress_defensive_rate"]["passed"] is True
    assert "mean_p_t_min" in report["family_reports"]["energy_starved"]["criteria"]
    assert "mean_challenge_fraction_min" in report["family_reports"]["energy_starved"]["criteria"]
    assert "role_switch_events_total" in report["family_reports"]["energy_starved"]["criteria"]
    assert "mean_aux_nontrivial_action_count" in report["family_reports"]["energy_starved"]["criteria"]
    assert "Promote the candidate for the required family tracks" in report["next_steps"][0]


def test_evaluate_contract_holdout_with_seed_leakage_blocks_promotion(tmp_path: Path) -> None:
    output_root = tmp_path / "harness_eval_holdout"
    contract = experiment_harness.build_experiment_contract(
        output_root=output_root,
        experiment_name="candidate_holdout",
        seed_start=10,
        num_seeds=2,
        holdout_seed_start=10,
        holdout_num_seeds=2,
        family_tracks=[
            {
                "name": "energy_starved",
                "promotion_target": "Promote only after holdout confirms improvement.",
            },
        ],
    )
    compare_root = Path(contract["artifacts"]["compare_root"])
    energy_root = compare_root / "energy_starved"
    holdout_root = energy_root / "holdout"
    for seed in (10, 11):
        _write_seed_compare(
            energy_root,
            seed=seed,
            candidate_mode=contract["candidate_mode"],
            baseline_mode=contract["baseline_mode"],
            candidate_final=0.22,
            candidate_mean=0.24,
            baseline_final=0.34,
            baseline_mean=0.31,
            candidate_actions=["withdraw", "reconfigure", "withdraw"],
            candidate_contacts=[(0.45, 0.40), (0.50, 0.42), (0.47, 0.41)],
            best_mode=contract["candidate_mode"],
        )
        _write_seed_compare(
            holdout_root,
            seed=seed,
            candidate_mode=contract["candidate_mode"],
            baseline_mode=contract["baseline_mode"],
            candidate_final=0.23,
            candidate_mean=0.25,
            baseline_final=0.35,
            baseline_mean=0.32,
            candidate_actions=["withdraw", "reconfigure", "withdraw"],
            candidate_contacts=[(0.45, 0.40), (0.50, 0.42), (0.47, 0.41)],
            best_mode=contract["candidate_mode"],
        )
    save_json(
        energy_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "seed_start": 10,
            "num_seeds": 2,
            "per_seed": [
                {"seed": 10, "best_mode_by_final_homeostasis": contract["candidate_mode"]},
                {"seed": 11, "best_mode_by_final_homeostasis": contract["candidate_mode"]},
            ],
            "counts_by_best_final_homeostasis": {contract["candidate_mode"]: 2},
            "counts_by_best_mean_homeostasis": {contract["candidate_mode"]: 2},
        },
    )
    save_json(
        holdout_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "seed_start": 10,
            "num_seeds": 2,
            "split": "holdout",
            "per_seed": [
                {"seed": 10, "best_mode_by_final_homeostasis": contract["candidate_mode"]},
                {"seed": 11, "best_mode_by_final_homeostasis": contract["candidate_mode"]},
            ],
            "counts_by_best_final_homeostasis": {contract["candidate_mode"]: 2},
            "counts_by_best_mean_homeostasis": {contract["candidate_mode"]: 2},
        },
    )
    save_json(
        compare_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "compare_root": str(compare_root),
            "family_order": ["energy_starved"],
            "families": {
                "energy_starved": {
                    "aggregate_summary": str(energy_root / "aggregate_summary.json"),
                    "holdout_aggregate_summary": str(holdout_root / "aggregate_summary.json"),
                    "required_for_promotion": True,
                    "promotion_target": "Promote only after holdout confirms improvement.",
                },
            },
        },
    )

    report = experiment_harness.evaluate_contract(contract)
    family = report["family_reports"]["energy_starved"]
    assert report["overall_pass"] is False
    assert family["overall_pass_holdout"] is True
    assert family["seed_leakage"]["has_overlap"] is True
    assert "Seed leakage detected between dev and holdout splits" in family["next_steps"][0]


def test_evaluate_contract_handles_missing_aggregate_as_failed_report(tmp_path: Path) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "missing_aggregate",
        experiment_name="missing_aggregate_case",
        family_tracks=[
            {
                "name": "energy_starved",
                "promotion_target": "Promote when metrics pass.",
            }
        ],
    )

    report = experiment_harness.evaluate_contract(contract)
    family = report["family_reports"]["energy_starved"]

    assert report["overall_pass"] is False
    assert family["overall_pass"] is False
    assert "evaluation_error" in family
    assert "missing aggregate summary" in family["evaluation_error"]


def test_evaluate_contract_requires_holdout_when_configured(tmp_path: Path) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "require_holdout",
        experiment_name="require_holdout_case",
        seed_start=10,
        num_seeds=1,
        holdout_num_seeds=0,
        require_holdout_for_promotion=True,
        family_tracks=[
            {
                "name": "energy_starved",
                "promotion_target": "Promote only with holdout.",
            }
        ],
    )
    compare_root = Path(contract["artifacts"]["compare_root"])
    energy_root = compare_root / "energy_starved"
    _write_seed_compare(
        energy_root,
        seed=10,
        candidate_mode=contract["candidate_mode"],
        baseline_mode=contract["baseline_mode"],
        candidate_final=0.20,
        candidate_mean=0.21,
        baseline_final=0.34,
        baseline_mean=0.31,
        candidate_actions=["withdraw", "reconfigure", "withdraw"],
        candidate_contacts=[(0.45, 0.40), (0.50, 0.42), (0.47, 0.41)],
        best_mode=contract["candidate_mode"],
    )
    save_json(
        energy_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "seed_start": 10,
            "num_seeds": 1,
            "per_seed": [
                {"seed": 10, "best_mode_by_final_homeostasis": contract["candidate_mode"]},
            ],
            "counts_by_best_final_homeostasis": {contract["candidate_mode"]: 1},
            "counts_by_best_mean_homeostasis": {contract["candidate_mode"]: 1},
        },
    )
    save_json(
        compare_root / "aggregate_summary.json",
        {
            "experiment_name": contract["experiment_name"],
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "compare_root": str(compare_root),
            "family_order": ["energy_starved"],
            "families": {
                "energy_starved": {
                    "aggregate_summary": str(energy_root / "aggregate_summary.json"),
                    "required_for_promotion": True,
                    "promotion_target": "Promote only with holdout.",
                },
            },
        },
    )

    report = experiment_harness.evaluate_contract(contract)
    family = report["family_reports"]["energy_starved"]
    assert report["holdout_required_for_promotion"] is True
    assert report["overall_pass"] is False
    assert family["overall_pass"] is False
    assert "Holdout split is required for promotion" in family["next_steps"][0]


def test_run_contract_stops_when_doctor_is_blocked(tmp_path: Path, monkeypatch) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "blocked_run",
        experiment_name="blocked_candidate",
    )
    save_json(contract["artifacts"]["contract"], contract)

    monkeypatch.setattr(
        experiment_harness,
        "run_doctor",
        lambda: {
            "status": "blocked",
            "blocking_issues": ["torch/numpy bridge failed"],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        experiment_harness,
        "_run_sweep_from_contract",
        lambda _contract: (_ for _ in ()).throw(AssertionError("sweep should not run when doctor is blocked")),
    )

    summary = experiment_harness.run_contract(contract["artifacts"]["contract"])

    assert summary["status"] == "blocked"
    assert Path(contract["artifacts"]["doctor_report"]).exists()
    promotion_decision = json.loads(Path(contract["artifacts"]["promotion_decision"]).read_text(encoding="utf-8"))
    assert promotion_decision["status"] == "blocked"
    next_steps = json.loads(Path(contract["artifacts"]["next_steps"]).read_text(encoding="utf-8"))
    assert next_steps["status"] == "blocked"
    assert "Fix the runtime environment first." in next_steps["next_steps"][0]


def test_run_sweep_from_contract_writes_family_aggregate_index(tmp_path: Path, monkeypatch) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "family_run",
        experiment_name="family_stage",
        seed_start=20,
        num_seeds=2,
        family_tracks=[
            {
                "name": "energy_starved",
                "runtime_overrides": {"G0": 0.25},
                "env_overrides": {"hazard_patches": 2},
                "promotion_target": "Promote low-energy track.",
            },
            {
                "name": "fragile_boundary",
                "runtime_overrides": {"B0": 0.28},
                "env_overrides": {"shelter_patches": 0},
                "promotion_target": "Promote fragile-boundary track.",
            },
        ],
    )
    captured: list[tuple[str, dict, dict]] = []

    def fake_compare_one_seed(**kwargs):
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        captured.append(
            (
                output_root.parts[-2],
                dict(kwargs.get("runtime_overrides") or {}),
                dict(kwargs.get("env_overrides") or {}),
            )
        )
        return {
            "derived": {
                "best_mode_by_final_homeostasis": contract["candidate_mode"],
                "best_mode_by_mean_homeostasis": contract["candidate_mode"],
            }
        }

    monkeypatch.setattr(experiment_harness, "_compare_one_seed", fake_compare_one_seed)

    aggregate = experiment_harness._run_sweep_from_contract(contract)

    assert aggregate["family_order"] == ["energy_starved", "fragile_boundary"]
    assert set(aggregate["families"]) == {"energy_starved", "fragile_boundary"}
    assert len(captured) == 4
    assert ("energy_starved", {"G0": 0.25}, {"hazard_patches": 2}) in captured
    assert ("fragile_boundary", {"B0": 0.28}, {"shelter_patches": 0}) in captured


def test_build_promotion_decision_marks_failed_tracks(tmp_path: Path) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "decision_run",
        experiment_name="decision_case",
        family_tracks=[
            {
                "name": "toxic_band",
                "promotion_target": "Promote toxic-band robustness.",
            }
        ],
    )
    eval_report = {
        "doctor_status": "ok",
        "overall_pass": False,
        "required_family_tracks": ["toxic_band"],
        "eligible_family_tracks": [],
        "blocked_family_tracks": ["toxic_band"],
        "family_reports": {
            "toxic_band": {
                "overall_pass": False,
                "required_for_promotion": True,
                "promotion_target": "Promote toxic-band robustness.",
                "criteria": {
                    "best_mode_frequency": {"passed": False},
                    "stress_defensive_rate": {"passed": False},
                    "dead_fraction": {"passed": True},
                },
                "next_steps": ["Increase toxic family weight."],
                "summary": {
                    "candidate": {"mean_final_homeostatic_error": 0.31},
                    "baseline": {"mean_final_homeostatic_error": 0.28},
                },
            }
        },
        "next_steps": ["Keep `toxic_band` below promotion."],
    }

    decision = experiment_harness.build_promotion_decision(contract, eval_report=eval_report)

    assert decision["status"] == "revise"
    assert decision["blocked_tracks"] == ["toxic_band"]
    assert "Keep the candidate below promotion" in decision["recommendation"]
    assert decision["track_decisions"][0]["failed_criteria"] == ["best_mode_frequency", "stress_defensive_rate"]


def test_build_promotion_decision_includes_holdout_and_leakage_reasons(tmp_path: Path) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "decision_holdout",
        experiment_name="decision_holdout_case",
        family_tracks=[
            {
                "name": "energy_starved",
                "promotion_target": "Promote only after holdout passes.",
            }
        ],
    )
    eval_report = {
        "doctor_status": "ok",
        "overall_pass": False,
        "required_family_tracks": ["energy_starved"],
        "eligible_family_tracks": [],
        "blocked_family_tracks": ["energy_starved"],
        "family_reports": {
            "energy_starved": {
                "overall_pass": False,
                "required_for_promotion": True,
                "promotion_target": "Promote only after holdout passes.",
                "criteria": {
                    "stress_exploit_rate": {"passed": False},
                },
                "criteria_holdout": {
                    "final_improvement_ci_lower": {"passed": False},
                },
                "overall_pass_holdout": False,
                "seed_leakage": {"has_overlap": True, "overlap_seeds": [10, 11]},
                "next_steps": ["Rebuild with disjoint holdout seeds."],
                "summary": {
                    "candidate": {"mean_final_homeostatic_error": 0.34},
                    "baseline": {"mean_final_homeostatic_error": 0.28},
                },
            }
        },
        "next_steps": ["Keep `energy_starved` below promotion."],
    }

    decision = experiment_harness.build_promotion_decision(contract, eval_report=eval_report)
    track = decision["track_decisions"][0]

    assert decision["status"] == "revise"
    assert "blocked:" in decision["ci_summary"]
    assert "seed_leakage" in decision["ci_summary"]
    assert track["holdout_enabled"] is True
    assert track["holdout_pass"] is False
    assert track["seed_leakage_overlap"] is True
    assert "seed_leakage" in track["failure_reasons"]
    assert "holdout_fail" in track["failure_reasons"]


def test_apply_tuning_updates_clamps_and_targets_blocked_tracks(tmp_path: Path) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "tune_contract",
        experiment_name="tune_case",
        family_tracks=[
            {
                "name": "energy_starved",
                "runtime_overrides": {"aperture_gain": 0.20},
            },
            {
                "name": "fragile_boundary",
                "runtime_overrides": {"aperture_gain": 0.60},
            },
        ],
    )
    applied = experiment_harness._apply_tuning_updates(
        contract,
        proposals=[
            {
                "criterion": "stress_exploit_rate",
                "param": "aperture_gain",
                "delta": -0.40,
                "failed_track_count": 1,
            }
        ],
        blocked_tracks=["energy_starved"],
    )

    tracks = {track["name"]: track for track in contract["family_tracks"]}
    assert tracks["energy_starved"]["runtime_overrides"]["aperture_gain"] == 0.15
    assert tracks["fragile_boundary"]["runtime_overrides"]["aperture_gain"] == 0.60
    assert len(applied) == 1
    assert applied[0]["track"] == "energy_starved"


def test_propose_tuning_updates_handles_intake_lock_criteria() -> None:
    eval_report = {
        "required_family_tracks": ["energy_starved"],
        "blocked_family_tracks": ["energy_starved"],
        "family_reports": {
            "energy_starved": {
                "required_for_promotion": True,
                "criteria": {
                    "intake_rate": {"passed": False},
                    "action_diversity": {"passed": False},
                    "navigation_rate": {"passed": False},
                },
            }
        },
    }
    proposals = experiment_harness._propose_tuning_updates(eval_report, max_updates_per_round=3)
    params = [row["param"] for row in proposals]

    assert "aperture_gain" in params
    assert "action_gating_blend" in params
    assert "move_step" in params


def test_propose_tuning_updates_handles_trace_ablation_criteria() -> None:
    eval_report = {
        "required_family_tracks": ["energy_starved"],
        "blocked_family_tracks": ["energy_starved"],
        "family_reports": {
            "energy_starved": {
                "required_for_promotion": True,
                "criteria": {
                    "trace_ablation_spawn_delta": {"passed": False},
                },
            }
        },
    }
    proposals = experiment_harness._propose_tuning_updates(eval_report, max_updates_per_round=2)
    params = [row["param"] for row in proposals]

    assert "move_step" in params
    assert "aperture_width_deg" in params


def test_propose_tuning_updates_handles_role_distribution_criteria() -> None:
    eval_report = {
        "required_family_tracks": ["energy_starved"],
        "blocked_family_tracks": ["energy_starved"],
        "family_reports": {
            "energy_starved": {
                "required_for_promotion": True,
                "criteria": {
                    "role_switch_events_total": {"passed": False},
                },
            }
        },
    }
    proposals = experiment_harness._propose_tuning_updates(eval_report, max_updates_per_round=2)
    params = [row["param"] for row in proposals]

    assert "move_step" in params
    assert "lookahead_horizon" in params


def test_run_tuning_loop_stops_on_promotion(tmp_path: Path, monkeypatch) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "tune_run",
        experiment_name="tune_promote",
        family_tracks=[
            {
                "name": "energy_starved",
                "promotion_target": "Promote only after stress metrics improve.",
            }
        ],
    )
    save_json(contract["artifacts"]["contract"], contract)
    call_count = {"n": 0}

    def fake_run_contract(round_contract, *, force=False, skip_doctor=False):
        call_count["n"] += 1
        passed = call_count["n"] >= 2
        eval_report = {
            "overall_pass": passed,
            "required_family_tracks": ["energy_starved"],
            "eligible_family_tracks": ["energy_starved"] if passed else [],
            "blocked_family_tracks": [] if passed else ["energy_starved"],
            "family_reports": {
                "energy_starved": {
                    "required_for_promotion": True,
                    "summary": {
                        "candidate": {
                            "mean_final_homeostatic_error": 0.24 if passed else 0.33,
                        }
                    },
                    "criteria": {
                        "stress_exploit_rate": {"passed": passed},
                        "mean_final_homeostatic_error": {"passed": passed},
                    },
                }
            },
            "next_steps": [],
        }
        save_json(round_contract["artifacts"]["eval_report"], eval_report)
        save_json(
            round_contract["artifacts"]["run_summary"],
            {
                "status": "passed" if passed else "failed",
                "experiment_name": round_contract["experiment_name"],
            },
        )
        return {"status": "passed" if passed else "failed"}

    monkeypatch.setattr(experiment_harness, "run_contract", fake_run_contract)

    summary = experiment_harness.run_tuning_loop(
        contract["artifacts"]["contract"],
        max_rounds=4,
        min_primary_improvement=0.001,
        stagnation_patience=1,
        max_updates_per_round=2,
    )

    assert summary["status"] == "promote"
    assert summary["rounds_run"] == 3
    assert call_count["n"] == 3
    assert Path(summary["tune_summary_path"]).exists()
    assert Path(summary["recommended_contract_path"]).exists()
    assert summary["selected_round"] == 2
    assert (tmp_path / "tune_run" / "autotune" / "round_01" / "contract.json").exists()
    assert (tmp_path / "tune_run" / "autotune" / "round_02" / "contract.json").exists()
    assert (tmp_path / "tune_run" / "autotune" / "round_03" / "contract.json").exists()


def test_run_tuning_loop_apply_best_updates_contract_with_recommended(tmp_path: Path, monkeypatch) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "tune_apply_best",
        experiment_name="tune_apply_best",
        family_tracks=[
            {
                "name": "energy_starved",
                "runtime_overrides": {"aperture_gain": 0.50},
                "promotion_target": "Promote only after energy-starved track improves.",
            },
            {
                "name": "fragile_boundary",
                "runtime_overrides": {"aperture_gain": 0.60},
                "promotion_target": "Keep fragile-boundary stable.",
            },
        ],
    )
    contract_path = Path(contract["artifacts"]["contract"])
    save_json(contract_path, contract)
    call_count = {"n": 0}

    def fake_run_contract(round_contract, *, force=False, skip_doctor=False):
        call_count["n"] += 1
        passed = call_count["n"] >= 2
        eval_report = {
            "overall_pass": passed,
            "required_family_tracks": ["energy_starved", "fragile_boundary"],
            "eligible_family_tracks": ["energy_starved", "fragile_boundary"] if passed else [],
            "blocked_family_tracks": [] if passed else ["energy_starved"],
            "family_reports": {
                "energy_starved": {
                    "required_for_promotion": True,
                    "summary": {
                        "candidate": {
                            "mean_final_homeostatic_error": 0.22 if passed else 0.36,
                        }
                    },
                    "criteria": {
                        "stress_exploit_rate": {"passed": passed},
                        "mean_final_homeostatic_error": {"passed": passed},
                    },
                },
                "fragile_boundary": {
                    "required_for_promotion": True,
                    "summary": {
                        "candidate": {
                            "mean_final_homeostatic_error": 0.24 if passed else 0.25,
                        }
                    },
                    "criteria": {
                        "stress_exploit_rate": {"passed": True},
                        "mean_final_homeostatic_error": {"passed": True},
                    },
                },
            },
            "next_steps": [],
        }
        save_json(round_contract["artifacts"]["eval_report"], eval_report)
        save_json(
            round_contract["artifacts"]["run_summary"],
            {
                "status": "passed" if passed else "failed",
                "experiment_name": round_contract["experiment_name"],
            },
        )
        return {"status": "passed" if passed else "failed"}

    monkeypatch.setattr(experiment_harness, "run_contract", fake_run_contract)

    summary = experiment_harness.run_tuning_loop(
        contract_path,
        max_rounds=3,
        min_primary_improvement=0.001,
        stagnation_patience=1,
        max_updates_per_round=2,
        apply_best=True,
    )

    assert summary["status"] == "promote"
    assert summary["applied_contract_path"] == str(contract_path)
    assert summary["recommended_contract_path"].endswith("recommended_contract.json")
    recommended = json.loads(Path(summary["recommended_contract_path"]).read_text(encoding="utf-8"))
    applied = json.loads(contract_path.read_text(encoding="utf-8"))
    assert applied == recommended

    tracks = {track["name"]: track for track in applied["family_tracks"]}
    assert tracks["energy_starved"]["runtime_overrides"]["aperture_gain"] == pytest.approx(0.46)
    assert tracks["energy_starved"]["runtime_overrides"]["action_gating_blend"] == pytest.approx(0.40)
    assert tracks["fragile_boundary"]["runtime_overrides"]["aperture_gain"] == pytest.approx(0.60)


def test_run_tuning_loop_resume_continues_from_existing_rounds(tmp_path: Path, monkeypatch) -> None:
    contract = experiment_harness.build_experiment_contract(
        output_root=tmp_path / "tune_resume",
        experiment_name="tune_resume_case",
        family_tracks=[
            {
                "name": "energy_starved",
                "promotion_target": "Promote after confirmation.",
            }
        ],
    )
    contract_path = Path(contract["artifacts"]["contract"])
    save_json(contract_path, contract)
    call_count = {"n": 0}

    def fake_run_contract(round_contract, *, force=False, skip_doctor=False):
        call_count["n"] += 1
        passed = call_count["n"] >= 2
        eval_report = {
            "overall_pass": passed,
            "required_family_tracks": ["energy_starved"],
            "eligible_family_tracks": ["energy_starved"] if passed else [],
            "blocked_family_tracks": [] if passed else ["energy_starved"],
            "family_reports": {
                "energy_starved": {
                    "required_for_promotion": True,
                    "summary": {
                        "candidate": {
                            "mean_final_homeostatic_error": 0.21 if passed else 0.33,
                        }
                    },
                    "criteria": {
                        "stress_exploit_rate": {"passed": passed},
                    },
                }
            },
            "next_steps": [],
        }
        save_json(round_contract["artifacts"]["eval_report"], eval_report)
        save_json(
            round_contract["artifacts"]["run_summary"],
            {
                "status": "passed" if passed else "failed",
                "experiment_name": round_contract["experiment_name"],
            },
        )
        return {"status": "passed" if passed else "failed"}

    monkeypatch.setattr(experiment_harness, "run_contract", fake_run_contract)

    first = experiment_harness.run_tuning_loop(
        contract_path,
        max_rounds=1,
        promotion_streak_required=1,
    )
    assert first["rounds_run"] == 1
    assert first["status"] == "max_rounds"

    resumed = experiment_harness.run_tuning_loop(
        contract_path,
        max_rounds=3,
        promotion_streak_required=1,
        resume=True,
    )
    assert resumed["resumed"] is True
    assert resumed["start_round_index"] == 2
    assert resumed["rounds_run"] == 2
    assert resumed["status"] == "promote"
    assert call_count["n"] == 2


def test_check_promotion_decision_script_returns_expected_exit_codes(tmp_path: Path) -> None:
    decision_promote = tmp_path / "promote.json"
    decision_revise = tmp_path / "revise.json"
    save_json(
        decision_promote,
        {"status": "promote", "ci_summary": "promote: energy_starved"},
    )
    save_json(
        decision_revise,
        {"status": "revise", "ci_summary": "blocked: energy_starved(seed_leakage)"},
    )

    ok = subprocess.run(
        [sys.executable, "scripts/check_promotion_decision.py", "--decision", str(decision_promote)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    ng = subprocess.run(
        [sys.executable, "scripts/check_promotion_decision.py", "--decision", str(decision_revise)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert ok.returncode == 0
    assert ng.returncode == 1
