from __future__ import annotations

import json
from pathlib import Path

from trm_pipeline import dataset_harness
from trm_pipeline import production_runner
from trm_pipeline.prepare_trm_va_data import EPISODE_FAMILIES


def test_build_dataset_contract_sets_collection_artifacts(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "dataset_run",
        dataset_name="passive_pretrain",
        dataset_kind=dataset_harness.DATASET_KIND_PASSIVE,
        num_seeds=20,
        record_steps=64,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )

    assert contract["dataset_kind"] == dataset_harness.DATASET_KIND_PASSIVE
    assert contract["generator"]["target_modules"] == ["trm_a", "trm_b"]
    assert contract["acceptance"]["min_mean_num_frames"] == 65
    assert Path(contract["artifacts"]["collection_decision"]) == tmp_path / "dataset_run" / "collection_decision.json"
    assert Path(contract["artifacts"]["training_plan"]) == tmp_path / "dataset_run" / "training_plan.json"


def test_evaluate_passive_dataset_reports_pass_on_balanced_manifest(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "passive_eval",
        dataset_name="passive_eval",
        dataset_kind=dataset_harness.DATASET_KIND_PASSIVE,
        num_seeds=12,
        record_steps=32,
        registry_path=tmp_path / "dataset_registry.jsonl",
        acceptance={
            "min_successful_episodes": 10,
            "min_unique_seed_count": 10,
            "min_perturbed_fraction": 0.10,
            "max_perturbed_fraction": 0.50,
            "min_regime_diversity": 2,
            "max_single_regime_fraction": 0.90,
        },
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(10):
        rows.append(
            {
                "episode_id": f"ep_{index:03d}",
                "seed_id": f"seed_{index:03d}",
                "split": "train" if index < 7 else ("val" if index == 7 else "test"),
                "num_frames": 33,
                "perturb_mode": None if index < 8 else "local",
                "regime": "stable" if index < 5 else "chaotic",
            }
        )
    (dataset_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "summary.json").write_text(
        json.dumps({"num_selected_seeds": 12, "num_successful_episodes": 10}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = dataset_harness.evaluate_dataset_contract(contract)

    assert report["overall_pass"] is True
    assert report["summary"]["successful_episodes"] == 10
    assert report["criteria"]["regime_diversity"]["passed"] is True


def test_evaluate_agentic_dataset_requires_family_coverage(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_eval",
        dataset_name="agentic_eval",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        episodes=5,
        required_families=["toxic_band", "fragile_boundary", "vent_edge"],
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"episode_id": "va_1", "episode_family": "toxic_band"},
        {"episode_id": "va_2", "episode_family": "fragile_boundary"},
    ]
    (dataset_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "summary.json").write_text(
        json.dumps(
            {
                "retained_episodes": 2,
                "rejected_episodes": 1,
                "attempted_episodes": 3,
                "family_counts": {
                    "energy_starved": 0,
                    "toxic_band": 1,
                    "fragile_boundary": 1,
                    "vent_edge": 0,
                    "uncertain_corridor": 0,
                },
                "aggregate_action_counts": {
                    "approach": 1,
                    "withdraw": 8,
                    "seal": 0,
                    "reconfigure": 0,
                    "no_action": 0,
                },
                "aggregate_policy_entropy_mean": 0.10,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = dataset_harness.evaluate_dataset_contract(contract)
    decision = dataset_harness.build_collection_decision(contract, eval_report=report)

    assert report["overall_pass"] is False
    assert report["summary"]["missing_required_families"] == ["vent_edge"]
    assert decision["status"] == "revise"
    assert "required_family_coverage" in decision["failed_criteria"]


def test_run_dataset_contract_stops_when_doctor_is_blocked(tmp_path: Path, monkeypatch) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "blocked_dataset",
        dataset_name="blocked_dataset",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        episodes=len(EPISODE_FAMILIES),
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    Path(contract["artifacts"]["contract"]).parent.mkdir(parents=True, exist_ok=True)
    Path(contract["artifacts"]["contract"]).write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        dataset_harness,
        "run_doctor",
        lambda: {"status": "blocked", "blocking_issues": ["torch import failed"], "warnings": []},
    )

    summary = dataset_harness.run_dataset_contract(contract["artifacts"]["contract"])

    assert summary["status"] == "blocked"
    decision = json.loads(Path(contract["artifacts"]["collection_decision"]).read_text(encoding="utf-8"))
    assert decision["status"] == "blocked"
    training_plan = json.loads(Path(contract["artifacts"]["training_plan"]).read_text(encoding="utf-8"))
    assert training_plan["status"] == "blocked"
    registry_rows = Path(contract["artifacts"]["registry_path"]).read_text(encoding="utf-8").splitlines()
    assert len(registry_rows) == 1


def test_build_training_plan_for_agentic_collect_includes_train_steps(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_collect",
        dataset_name="agentic_collect",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        episodes=5,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    decision = {
        "status": "collect",
        "recommendation": "Proceed with downstream training for: trm_vm, trm_as.",
    }

    plan = dataset_harness.build_training_plan(contract, collection_decision=decision)

    assert plan["status"] == "ready"
    assert [step["name"] for step in plan["steps"]] == ["train_trm_vm", "train_trm_as"]


def test_build_training_plan_for_agentic_prefers_role_view_manifests_when_present(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_collect_views",
        dataset_name="agentic_collect_views",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        episodes=5,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "summary.json").write_text(
        json.dumps(
            {
                "role_view_manifests": {
                    "trm_vm": str(dataset_root / "views" / "trm_vm.jsonl"),
                    "trm_as": str(dataset_root / "views" / "trm_as.jsonl"),
                    "trm_ag": str(dataset_root / "views" / "trm_ag.jsonl"),
                    "trm_bp": str(dataset_root / "views" / "trm_bp.jsonl"),
                    "trm_mc": str(dataset_root / "views" / "trm_mc.jsonl"),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    decision = {
        "status": "collect",
        "recommendation": "Proceed with downstream training for: trm_vm, trm_as.",
    }

    plan = dataset_harness.build_training_plan(contract, collection_decision=decision)

    assert plan["role_view_manifests"]["trm_vm"].endswith("views/trm_vm.jsonl")
    assert plan["steps"][0]["manifest_path"].endswith("views/trm_vm.jsonl")
    assert "--manifest " + plan["steps"][0]["manifest_path"] in plan["steps"][0]["command"]
    assert plan["steps"][1]["manifest_path"].endswith("views/trm_as.jsonl")
    assert [step["name"] for step in plan["steps"]] == ["train_trm_vm", "train_trm_as", "train_trm_ag", "train_trm_bp", "train_trm_mc"]
    assert plan["steps"][2]["manifest_path"].endswith("views/trm_ag.jsonl")
    assert "train_trm_ag" in plan["steps"][2]["command"]
    assert plan["steps"][3]["manifest_path"].endswith("views/trm_bp.jsonl")
    assert "train_trm_bp" in plan["steps"][3]["command"]
    assert plan["steps"][4]["manifest_path"].endswith("views/trm_mc.jsonl")
    assert "train_trm_mc" in plan["steps"][4]["command"]


def test_build_training_plan_for_passive_prefers_role_view_manifests_when_present(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "passive_collect_views",
        dataset_name="passive_collect_views",
        dataset_kind=dataset_harness.DATASET_KIND_PASSIVE,
        num_seeds=5,
        record_steps=16,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "summary.json").write_text(
        json.dumps(
            {
                "role_view_manifests": {
                    "trm_wp": str(dataset_root / "views" / "trm_wp.jsonl"),
                    "trm_bd": str(dataset_root / "views" / "trm_bd.jsonl"),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    decision = {
        "status": "collect",
        "recommendation": "Proceed with downstream training for: trm_a, trm_b.",
    }

    plan = dataset_harness.build_training_plan(contract, collection_decision=decision)

    assert [step["name"] for step in plan["steps"]] == ["train_trm_a", "train_trm_b"]
    assert plan["steps"][0]["manifest_path"].endswith("views/trm_wp.jsonl")
    assert "--input-key wp_input_view" in plan["steps"][0]["command"]
    assert plan["steps"][1]["manifest_path"].endswith("views/trm_bd.jsonl")
    assert "--sensor-gate-key bd_sensor_gate" in plan["steps"][1]["command"]


def test_production_preset_adds_mode_mix_and_readiness_thresholds(tmp_path: Path) -> None:
    preset = dataset_harness._preset_overrides("agentic_production")
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_prod",
        dataset_name="agentic_prod",
        registry_path=tmp_path / "dataset_registry.jsonl",
        **preset,
    )

    assert contract["generator"]["config"]["policy_mode_mix"] == {
        "closed_loop": 0.50,
        "random": 0.30,
        "no_action": 0.20,
    }
    assert contract["acceptance"]["min_retained_episodes"] == 192
    assert contract["acceptance"]["required_policy_modes"] == ["closed_loop", "random", "no_action"]


def test_evaluate_agentic_dataset_checks_production_mode_mix_and_dead_mix(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_prod_eval",
        dataset_name="agentic_prod_eval",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        episodes=6,
        required_families=list(EPISODE_FAMILIES),
        registry_path=tmp_path / "dataset_registry.jsonl",
        acceptance={
            "min_retained_episodes": 6,
            "min_source_seed_count": 6,
            "min_effective_step_samples": 60,
            "required_families": list(EPISODE_FAMILIES),
            "min_distinct_families": 5,
            "required_policy_modes": ["closed_loop", "random", "no_action"],
            "policy_mode_share_min": {"closed_loop": 0.3, "random": 0.2, "no_action": 0.1},
            "policy_mode_share_max": {"closed_loop": 0.7, "random": 0.5, "no_action": 0.3},
            "min_non_dead_fraction": 0.5,
            "max_non_dead_fraction": 0.8,
            "max_rejected_fraction": 0.5,
            "min_aggregate_policy_entropy": 1.0,
            "min_action_entropy_ratio": 0.45,
            "require_all_actions_present": True,
            "max_aggregate_dominant_action_fraction": 0.55,
            "min_dead_dominant_action_diversity": 2,
            "min_recovery_fraction": 0.55,
            "min_stress_defensive_fraction": 0.4,
            "max_stress_exploit_fraction": 0.55,
        },
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = []
    policy_modes = ["closed_loop", "closed_loop", "closed_loop", "random", "random", "no_action"]
    families = list(EPISODE_FAMILIES) + ["energy_starved"]
    terminal_dead = [False, False, False, True, True, False]
    dominant_actions = ["seal", "reconfigure", "seal", "withdraw", "approach", "seal"]
    for index in range(6):
        rows.append(
            {
                "episode_id": f"va_{index}",
                "seed_id": f"seed_{index}",
                "episode_family": families[index],
                "policy_mode": policy_modes[index],
                "num_samples": 12,
                "terminal_dead": terminal_dead[index],
                "quality": {
                    "terminal_dead": terminal_dead[index],
                    "dominant_action": dominant_actions[index],
                },
            }
        )
    (dataset_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "summary.json").write_text(
        json.dumps(
            {
                "retained_episodes": 6,
                "rejected_episodes": 0,
                "attempted_episodes": 6,
                "family_counts": {
                    "energy_starved": 2,
                    "toxic_band": 1,
                    "fragile_boundary": 1,
                    "vent_edge": 1,
                    "uncertain_corridor": 1,
                },
                "policy_mode_counts": {"closed_loop": 3, "random": 2, "no_action": 1},
                "aggregate_action_counts": {
                    "approach": 5,
                    "intake": 6,
                    "withdraw": 10,
                    "seal": 18,
                    "reconfigure": 8,
                },
                "aggregate_policy_entropy_mean": 1.2,
                "aggregate_recovery_fraction_mean": 0.6,
                "aggregate_stress_defensive_fraction_mean": 0.7,
                "aggregate_stress_exploit_fraction_mean": 0.1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = dataset_harness.evaluate_dataset_contract(contract)

    assert report["overall_pass"] is True
    assert report["summary"]["missing_policy_modes"] == []
    assert report["criteria"]["dead_dominant_action_diversity"]["passed"] is True
    assert report["criteria"]["all_actions_present"]["passed"] is True
    assert report["ideal_advisory"]["status"] in {"aligned", "near"}


def test_agentic_ideal_advisory_is_reported_without_overriding_overall_pass(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_advisory",
        dataset_name="agentic_advisory",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        episodes=6,
        required_families=list(EPISODE_FAMILIES),
        registry_path=tmp_path / "dataset_registry.jsonl",
        acceptance={
            "min_retained_episodes": 6,
            "min_distinct_families": 5,
            "required_families": list(EPISODE_FAMILIES),
            "max_rejected_fraction": 0.5,
            "min_aggregate_policy_entropy": 0.8,
            "min_action_entropy_ratio": 0.2,
            "max_aggregate_dominant_action_fraction": 0.9,
            "min_recovery_fraction": 0.5,
            "min_stress_defensive_fraction": 0.3,
            "max_stress_exploit_fraction": 0.8,
        },
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = []
    families = list(EPISODE_FAMILIES) + ["energy_starved"]
    for index in range(6):
        rows.append(
            {
                "episode_id": f"adv_{index}",
                "seed_id": f"seed_{index}",
                "episode_family": families[index],
                "policy_mode": "closed_loop",
                "num_samples": 10,
                "terminal_dead": False,
            }
        )
    (dataset_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "summary.json").write_text(
        json.dumps(
            {
                "retained_episodes": 6,
                "rejected_episodes": 0,
                "attempted_episodes": 6,
                "family_counts": {
                    "energy_starved": 2,
                    "toxic_band": 1,
                    "fragile_boundary": 1,
                    "vent_edge": 1,
                    "uncertain_corridor": 1,
                },
                "policy_mode_counts": {"closed_loop": 6},
                "aggregate_action_counts": {
                    "approach": 0,
                    "intake": 24,
                    "withdraw": 2,
                    "seal": 26,
                    "reconfigure": 1,
                },
                "aggregate_policy_entropy_mean": 0.85,
                "aggregate_recovery_fraction_mean": 0.58,
                "aggregate_stress_defensive_fraction_mean": 0.34,
                "aggregate_stress_exploit_fraction_mean": 0.32,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = dataset_harness.evaluate_dataset_contract(contract)
    decision = dataset_harness.build_collection_decision(contract, eval_report=report)

    assert report["overall_pass"] is True
    assert report["ideal_advisory"]["status"] in {"near", "far"}
    assert "ideal_advisory" in decision
    assert decision["status"] == "collect"


def test_build_revised_contract_rebalances_mode_mix_after_survival_skew(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_skewed",
        dataset_name="agentic_skewed",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        episodes=64,
        policy_mode_mix={"closed_loop": 0.8, "random": 0.1, "no_action": 0.1},
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    decision = {
        "status": "revise",
        "failed_criteria": ["non_dead_fraction_max", "all_actions_present"],
        "next_steps": [],
    }
    eval_report = {"summary": {}, "criteria": {}, "overall_pass": False}

    revised = dataset_harness.build_revised_contract(
        contract,
        eval_report=eval_report,
        collection_decision=decision,
        output_root=tmp_path / "agentic_next",
    )

    mix = revised["generator"]["config"]["policy_mode_mix"]
    assert mix["closed_loop"] < 0.8
    assert mix["random"] >= 0.25
    assert mix["no_action"] >= 0.2
    assert Path(revised["artifacts"]["contract"]).parent == tmp_path / "agentic_next"
    search_report = json.loads(Path(revised["artifacts"]["revision_search_report"]).read_text(encoding="utf-8"))
    assert search_report["selected_candidate"]["candidate_suffix"] in {
        "anti_collapse_mix",
        "composite_search",
    }
    assert len(search_report["candidates"]) >= 2


def test_run_training_plan_records_step_results(tmp_path: Path, monkeypatch) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "handoff_dataset",
        dataset_name="handoff_dataset",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )

    class _Result:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    monkeypatch.setattr(dataset_harness.subprocess, "run", lambda *args, **kwargs: _Result())
    report = dataset_harness.run_training_plan(
        contract,
        training_plan={
            "status": "ready",
            "target_modules": ["trm_vm"],
            "steps": [{"name": "train_trm_vm", "command": "./.venv/bin/python -m trm_pipeline.train_trm_vm --help"}],
        },
    )

    assert report["status"] == "passed"
    saved = json.loads(Path(contract["artifacts"]["training_run_report"]).read_text(encoding="utf-8"))
    assert saved["steps"][0]["returncode"] == 0


def test_evaluate_trained_models_and_promotion_decision_for_agentic(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "agentic_models",
        dataset_name="agentic_models",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    plan = dataset_harness.build_training_plan(
        contract,
        collection_decision={"status": "collect", "recommendation": ""},
    )
    for step in plan["steps"]:
        output_dir = Path(step["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["steps"][0]["metrics_path"]).write_text(
        json.dumps(
            {
                "val_homeostatic_error_mae": 0.08,
                "val_viability_risk_auroc": 0.71,
                "val_margin_to_failure_corr": 0.22,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(plan["steps"][1]["metrics_path"]).write_text(
        json.dumps(
            {
                "val_pairwise_ranking_accuracy": 0.63,
                "val_policy_entropy_mean": 1.14,
                "val_action_collapse_rate": 0.41,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = dataset_harness.evaluate_trained_models(
        contract,
        training_plan=plan,
        training_run_report={"status": "passed"},
    )
    decision = dataset_harness.build_promotion_decision(
        contract,
        model_eval_report=report,
        training_run_report={"status": "passed"},
    )

    assert report["overall_pass"] is True
    assert decision["status"] == "promote"


def test_run_dataset_campaign_auto_handoff_writes_promotion_artifacts(tmp_path: Path, monkeypatch) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "campaign_dataset",
        dataset_name="campaign_dataset",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )

    eval_report = {
        "target_modules": ["trm_vm", "trm_as"],
        "criteria": {},
        "summary": {},
        "overall_pass": True,
    }
    decision = {
        "status": "collect",
        "target_modules": ["trm_vm", "trm_as"],
        "failed_criteria": [],
        "summary": {},
        "next_steps": [],
    }
    Path(contract["artifacts"]["eval_report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(contract["artifacts"]["eval_report"]).write_text(json.dumps(eval_report, ensure_ascii=False), encoding="utf-8")
    Path(contract["artifacts"]["collection_decision"]).write_text(json.dumps(decision, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        dataset_harness,
        "run_dataset_contract",
        lambda *args, **kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        dataset_harness,
        "run_training_plan",
        lambda *args, **kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        dataset_harness,
        "evaluate_trained_models",
        lambda *args, **kwargs: {"training_status": "passed", "criteria": {}, "overall_pass": True},
    )
    monkeypatch.setattr(
        dataset_harness,
        "build_promotion_decision",
        lambda *args, **kwargs: {"status": "promote"},
    )

    result = dataset_harness.run_dataset_campaign(contract, auto_handoff=True)

    assert result["promotion_status"] == "promote"
    assert result["training_status"] == "passed"


def test_run_dataset_campaign_until_acceptance_revises_then_collects(tmp_path: Path, monkeypatch) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "campaign_loop",
        dataset_name="campaign_loop",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    revised = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "campaign_loop_revised",
        dataset_name="campaign_loop_revised",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    Path(revised["artifacts"]["contract"]).parent.mkdir(parents=True, exist_ok=True)
    Path(revised["artifacts"]["contract"]).write_text(json.dumps(revised, ensure_ascii=False), encoding="utf-8")

    calls = {"count": 0}

    def _fake_campaign(contract_or_path, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "status": "revise",
                "revised_contract": revised["artifacts"]["contract"],
            }
        return {"status": "collect"}

    monkeypatch.setattr(dataset_harness, "run_dataset_campaign", _fake_campaign)
    report = dataset_harness.run_dataset_campaign_until_acceptance(contract, max_rounds=3)

    assert report["status"] == "accepted"
    assert report["rounds_executed"] == 2
    assert Path(contract["artifacts"]["campaign_until_report"]).exists()
    assert report["rounds"][0]["status"] == "revise"
    assert report["rounds"][1]["status"] == "collect"


def test_passive_ideal_advisory_is_attached_to_report(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "passive_advisory",
        dataset_name="passive_advisory",
        dataset_kind=dataset_harness.DATASET_KIND_PASSIVE,
        num_seeds=8,
        record_steps=16,
        registry_path=tmp_path / "dataset_registry.jsonl",
        acceptance={
            "min_successful_episodes": 4,
            "min_unique_seed_count": 4,
            "min_mean_num_frames": 17,
            "min_perturbed_fraction": 0.0,
            "max_perturbed_fraction": 1.0,
            "min_regime_diversity": 1,
            "max_single_regime_fraction": 1.0,
        },
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(4):
        rows.append(
            {
                "episode_id": f"ep_{index}",
                "seed_id": f"seed_{index}",
                "split": "train",
                "num_frames": 17,
                "perturb_mode": None,
                "regime": "stable",
            }
        )
    (dataset_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "summary.json").write_text(
        json.dumps({"num_selected_seeds": 4, "num_successful_episodes": 4}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = dataset_harness.evaluate_dataset_contract(contract)

    assert report["overall_pass"] is True
    assert report["ideal_advisory"]["status"] in {"near", "far"}


def test_append_model_registry_entries_writes_one_row_per_train_step(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "registry_dataset",
        dataset_name="registry_dataset",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
        model_registry_path=tmp_path / "model_registry.jsonl",
    )
    plan = dataset_harness.build_training_plan(
        contract,
        collection_decision={"status": "collect", "recommendation": ""},
    )
    for step in plan["steps"]:
        output_dir = Path(step["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(step["metrics_path"]).write_text(json.dumps({"ok": True}, ensure_ascii=False), encoding="utf-8")
    entries = dataset_harness._append_model_registry_entries(
        contract,
        training_plan=plan,
        training_run_report={"status": "passed"},
        model_eval_report={
            "modules": {
                "train_trm_vm": {"metrics": {"vm": 1}},
                "train_trm_as": {"metrics": {"as": 1}},
            }
        },
        promotion_decision={"status": "hold", "failed_criteria": ["trm_vm.val_homeostatic_error_mae"]},
        execution_target="local",
    )

    assert len(entries) == 2
    rows = [json.loads(line) for line in (tmp_path / "model_registry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["target_module"] for row in rows] == ["trm_vm", "trm_as"]
    assert rows[0]["failed_criteria"] == ["trm_vm.val_homeostatic_error_mae"]
    assert rows[1]["failed_criteria"] == []


def test_append_model_registry_entries_tracks_trm_mc_when_present(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "registry_dataset_mc",
        dataset_name="registry_dataset_mc",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
        model_registry_path=tmp_path / "model_registry.jsonl",
    )
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "summary.json").write_text(
        json.dumps(
            {
                "role_view_manifests": {
                    "trm_vm": str(dataset_root / "views" / "trm_vm.jsonl"),
                    "trm_as": str(dataset_root / "views" / "trm_as.jsonl"),
                    "trm_mc": str(dataset_root / "views" / "trm_mc.jsonl"),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan = dataset_harness.build_training_plan(
        contract,
        collection_decision={"status": "collect", "recommendation": ""},
    )
    for step in plan["steps"]:
        output_dir = Path(step["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(step["metrics_path"]).write_text(json.dumps({"ok": True}, ensure_ascii=False), encoding="utf-8")

    entries = dataset_harness._append_model_registry_entries(
        contract,
        training_plan=plan,
        training_run_report={"status": "passed"},
        model_eval_report={"modules": {"train_trm_mc": {"metrics": {"mc": 1}}}},
        promotion_decision={"status": "hold", "failed_criteria": ["trm_mc.val_context_state_loss"]},
        execution_target="local",
    )

    assert any(entry["target_module"] == "trm_mc" for entry in entries)
    rows = [json.loads(line) for line in (tmp_path / "model_registry.jsonl").read_text(encoding="utf-8").splitlines()]
    mc_rows = [row for row in rows if row["target_module"] == "trm_mc"]
    assert len(mc_rows) == 1
    assert mc_rows[0]["failed_criteria"] == ["trm_mc.val_context_state_loss"]


def test_with_gpu_training_flags_adds_cuda_options_for_trm_mc() -> None:
    command = "./.venv/bin/python -m trm_pipeline.train_trm_mc --manifest data/trm_va_cache/views/trm_mc.jsonl --output-dir artifacts/trm_mc"
    flagged = dataset_harness._with_gpu_training_flags(command)

    assert "--device cuda" in flagged
    assert "--amp" in flagged
    assert "--log-interval 50" in flagged


def test_with_gpu_training_flags_adds_cuda_options_for_trm_ag() -> None:
    command = "./.venv/bin/python -m trm_pipeline.train_trm_ag --manifest data/trm_va_cache/views/trm_ag.jsonl --output-dir artifacts/trm_ag"
    flagged = dataset_harness._with_gpu_training_flags(command)

    assert "--device cuda" in flagged
    assert "--amp" in flagged
    assert "--log-interval 50" in flagged


def test_build_external_gpu_handoff_writes_cuda_script(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "gpu_dataset",
        dataset_name="gpu_dataset",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
        model_registry_path=tmp_path / "model_registry.jsonl",
    )
    plan = dataset_harness.build_training_plan(
        contract,
        collection_decision={"status": "collect", "recommendation": ""},
    )

    handoff = dataset_harness.build_external_gpu_handoff(
        contract,
        training_plan=plan,
        provider="vastai",
        remote_root="/workspace/criticism_bot",
    )

    assert handoff["status"] == "ready"
    script = Path(contract["artifacts"]["gpu_handoff_script"]).read_text(encoding="utf-8")
    assert "--device cuda" in script
    assert "--amp" in script
    report = json.loads(Path(contract["artifacts"]["gpu_handoff_report"]).read_text(encoding="utf-8"))
    assert report["provider"] == "vastai"
    assert report["model_registry_path"] == contract["artifacts"]["model_registry_path"]
    assert "finalize-external" in report["finalize_local_command"]


def test_run_dataset_campaign_external_gpu_returns_handoff_artifacts(tmp_path: Path, monkeypatch) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "gpu_campaign",
        dataset_name="gpu_campaign",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
    )
    eval_report = {
        "target_modules": ["trm_vm", "trm_as"],
        "criteria": {},
        "summary": {},
        "overall_pass": True,
    }
    decision = {
        "status": "collect",
        "target_modules": ["trm_vm", "trm_as"],
        "failed_criteria": [],
        "summary": {},
        "next_steps": [],
    }
    Path(contract["artifacts"]["eval_report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(contract["artifacts"]["eval_report"]).write_text(json.dumps(eval_report, ensure_ascii=False), encoding="utf-8")
    Path(contract["artifacts"]["collection_decision"]).write_text(json.dumps(decision, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        dataset_harness,
        "run_dataset_contract",
        lambda *args, **kwargs: {"status": "passed"},
    )

    result = dataset_harness.run_dataset_campaign(contract, external_gpu_provider="vastai")

    assert result["gpu_handoff_status"] == "ready"
    assert result["gpu_handoff_report"] == contract["artifacts"]["gpu_handoff_report"]
    assert Path(contract["artifacts"]["gpu_handoff_script"]).exists()


def test_finalize_external_training_promotes_and_updates_model_registry(tmp_path: Path) -> None:
    contract = dataset_harness.build_dataset_contract(
        output_root=tmp_path / "external_finalize",
        dataset_name="external_finalize",
        dataset_kind=dataset_harness.DATASET_KIND_AGENTIC,
        registry_path=tmp_path / "dataset_registry.jsonl",
        model_registry_path=tmp_path / "model_registry.jsonl",
    )
    Path(contract["artifacts"]["contract"]).write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
    plan = dataset_harness.build_training_plan(
        contract,
        collection_decision={"status": "collect", "recommendation": ""},
    )
    Path(contract["artifacts"]["training_plan"]).write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    dataset_harness.build_external_gpu_handoff(contract, training_plan=plan, provider="vastai")
    for step in plan["steps"]:
        output_dir = Path(step["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["steps"][0]["metrics_path"]).write_text(
        json.dumps(
            {
                "val_homeostatic_error_mae": 0.08,
                "val_viability_risk_auroc": 0.71,
                "val_margin_to_failure_corr": 0.22,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(plan["steps"][1]["metrics_path"]).write_text(
        json.dumps(
            {
                "val_pairwise_ranking_accuracy": 0.63,
                "val_policy_entropy_mean": 1.14,
                "val_action_collapse_rate": 0.41,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = dataset_harness.finalize_external_training(contract)

    assert report["status"] == "promote"
    assert report["training_status"] == "passed"
    saved = json.loads(Path(contract["artifacts"]["external_finalize_report"]).read_text(encoding="utf-8"))
    assert saved["execution_target"] == "external_gpu:vastai"
    rows = [json.loads(line) for line in (tmp_path / "model_registry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["execution_target"] == "external_gpu:vastai" for row in rows)


def test_production_runner_builds_contract_and_run_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        production_runner,
        "_run_preflight",
        lambda run_tests: {"status": "ok", "doctor": {"status": "ok"}, "tests": None},
    )
    monkeypatch.setattr(
        production_runner,
        "run_dataset_campaign",
        lambda *args, **kwargs: {"status": "collect", "gpu_handoff_status": "ready"},
    )

    report = production_runner.run_production_campaign(
        preset="agentic_production",
        output_root=tmp_path / "production_run",
        execution_target="gpu-handoff",
        provider="vastai",
    )

    assert report["status"] == "ready"
    assert Path(tmp_path / "production_run" / "production_preflight.json").exists()
    assert Path(tmp_path / "production_run" / "production_runner_report.json").exists()


def test_production_runner_uses_campaign_until_when_max_rounds_gt_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        production_runner,
        "_run_preflight",
        lambda run_tests: {"status": "ok", "doctor": {"status": "ok"}, "tests": None},
    )
    monkeypatch.setattr(
        production_runner,
        "run_dataset_campaign_until_acceptance",
        lambda *args, **kwargs: {"status": "accepted", "promotion_status": "promote", "rounds_executed": 2},
    )

    report = production_runner.run_production_campaign(
        preset="agentic_production",
        output_root=tmp_path / "production_loop",
        execution_target="local",
        auto_handoff=True,
        max_rounds=3,
    )

    assert report["status"] == "promote"
    assert report["max_rounds"] == 3


def test_production_runner_finalize_reads_synced_outputs(tmp_path: Path) -> None:
    contract = production_runner.build_production_contract(
        preset="agentic_production",
        output_root=tmp_path / "production_finalize",
    )
    plan = dataset_harness.build_training_plan(
        contract,
        collection_decision={"status": "collect", "recommendation": ""},
    )
    Path(contract["artifacts"]["training_plan"]).write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    dataset_harness.build_external_gpu_handoff(contract, training_plan=plan, provider="vastai")
    for step in plan["steps"]:
        output_dir = Path(step["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["steps"][0]["metrics_path"]).write_text(
        json.dumps(
            {
                "val_homeostatic_error_mae": 0.08,
                "val_viability_risk_auroc": 0.71,
                "val_margin_to_failure_corr": 0.22,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(plan["steps"][1]["metrics_path"]).write_text(
        json.dumps(
            {
                "val_pairwise_ranking_accuracy": 0.63,
                "val_policy_entropy_mean": 1.14,
                "val_action_collapse_rate": 0.41,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = production_runner.finalize_production_campaign(output_root=tmp_path / "production_finalize")

    assert report["status"] == "promote"
    assert Path(tmp_path / "production_finalize" / "production_finalize_report.json").exists()
