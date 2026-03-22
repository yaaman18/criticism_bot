from __future__ import annotations

import json
import sys
from pathlib import Path

from trm_pipeline import compare_erie_runtime
from trm_pipeline.erie_runtime import EnvironmentConfig, RuntimeConfig, run_episode


def _seed_catalog(tmp_path: Path) -> Path:
    catalog_path = tmp_path / "seed_catalog.json"
    rows = [
        {
            "source_file": "test_seed.json",
            "code": "test",
            "name": "unit-seed",
            "params": {"R": 12, "T": 10, "b": "1", "kn": 1, "gn": 1},
            "cells": "3o$3o$3o!",
        }
    ]
    catalog_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return catalog_path


def test_compare_runtime_writes_three_mode_summary(tmp_path: Path, monkeypatch) -> None:
    captured: list[tuple[object, object]] = []

    def fake_run_episode(
        output_root: str | Path,
        seed_catalog: str | Path,
        runtime_config,
        env_config,
        trm_a_checkpoint=None,
        trm_b_checkpoint=None,
        module_specs=None,
        module_manifest=None,
    ) -> Path:
        captured.append((runtime_config, env_config, module_specs, module_manifest))
        out_root = Path(output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        episode_path = out_root / f"{runtime_config.policy_mode}_episode.npz"
        episode_path.write_bytes(b"fake")
        summary_path = out_root / f"{episode_path.stem}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "policy_mode": runtime_config.policy_mode,
                    "num_steps_executed": runtime_config.steps,
                    "dead": False,
                    "final_G": {"closed_loop": 0.56, "random": 0.92, "no_action": 1.0}[runtime_config.policy_mode],
                    "final_B": {"closed_loop": 0.66, "random": 0.95, "no_action": 0.18}[runtime_config.policy_mode],
                    "final_homeostatic_error": {
                        "closed_loop": 0.02,
                        "random": 0.67,
                        "no_action": 0.92,
                    }[runtime_config.policy_mode],
                    "mean_homeostatic_error": {
                        "closed_loop": 0.08,
                        "random": 0.44,
                        "no_action": 0.61,
                    }[runtime_config.policy_mode],
                    "action_cost_total": {
                        "closed_loop": 0.18,
                        "random": 0.31,
                        "no_action": 0.0,
                    }[runtime_config.policy_mode],
                    "survival_fraction": 1.0,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return episode_path

    monkeypatch.setattr(compare_erie_runtime, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_erie_runtime",
            "--output-root",
            str(tmp_path / "compare"),
            "--steps",
            "5",
            "--warmup-steps",
            "1",
            "--seed",
            "31415",
            "--lookahead-horizon",
            "3",
            "--lookahead-discount",
            "0.9",
            "--resource-patches",
            "4",
            "--hazard-patches",
            "2",
            "--shelter-patches",
            "2",
            "--viability-mode",
            "assistive",
            "--action-mode",
            "module_primary",
        ],
    )

    compare_erie_runtime.main()

    summary_path = tmp_path / "compare" / "comparison_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["results"]) == {"closed_loop", "random", "no_action"}
    assert summary["derived"]["best_mode_by_final_homeostasis"] == "closed_loop"
    assert summary["derived"]["ranking_by_final_homeostasis"] == ["closed_loop", "random", "no_action"]
    assert summary["derived"]["ranking_by_mean_homeostasis"] == ["closed_loop", "random", "no_action"]
    assert len(captured) == 3
    for runtime_config, env_config, module_specs, module_manifest in captured:
        assert runtime_config.lookahead_horizon == 3
        assert runtime_config.lookahead_discount == 0.9
        assert runtime_config.viability_mode == "assistive"
        assert runtime_config.action_mode == "module_primary"
        assert env_config.resource_patches == 4
        assert env_config.hazard_patches == 2
        assert env_config.shelter_patches == 2
        assert module_specs is None
        assert module_manifest is None


def test_compare_runtime_forwards_module_manifest(tmp_path: Path, monkeypatch) -> None:
    captured: list[tuple[object, object, object, object]] = []

    def fake_run_episode(
        output_root: str | Path,
        seed_catalog: str | Path,
        runtime_config,
        env_config,
        trm_a_checkpoint=None,
        trm_b_checkpoint=None,
        module_specs=None,
        module_manifest=None,
    ) -> Path:
        captured.append((runtime_config, env_config, module_specs, module_manifest))
        out_root = Path(output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        episode_path = out_root / f"{runtime_config.policy_mode}_episode.npz"
        episode_path.write_bytes(b"fake")
        summary_path = out_root / f"{episode_path.stem}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "policy_mode": runtime_config.policy_mode,
                    "num_steps_executed": runtime_config.steps,
                    "dead": False,
                    "final_G": 0.5,
                    "final_B": 0.6,
                    "final_homeostatic_error": 0.1,
                    "mean_homeostatic_error": 0.2,
                    "action_cost_total": 0.0,
                    "survival_fraction": 1.0,
                }
            ),
            encoding="utf-8",
        )
        return episode_path

    manifest = tmp_path / "modules.json"
    manifest.write_text('[{"name":"trm_a","checkpoint":"dummy.pt"}]', encoding="utf-8")

    monkeypatch.setattr(compare_erie_runtime, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_erie_runtime",
            "--output-root",
            str(tmp_path / "compare_manifest"),
            "--module-manifest",
            str(manifest),
        ],
    )

    compare_erie_runtime.main()

    assert len(captured) == 3
    for _runtime_config, _env_config, module_specs, module_manifest in captured:
        assert module_specs is None
        assert Path(module_manifest) == manifest


def test_closed_loop_beats_no_action_on_reference_seed(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    env_config = EnvironmentConfig(image_size=32, target_radius=8)

    closed_loop_path = run_episode(
        tmp_path / "closed_loop",
        catalog_path,
        RuntimeConfig(steps=20, warmup_steps=2, seed=20260312, policy_mode="closed_loop"),
        env_config,
    )
    no_action_path = run_episode(
        tmp_path / "no_action",
        catalog_path,
        RuntimeConfig(steps=20, warmup_steps=2, seed=20260312, policy_mode="no_action"),
        env_config,
    )

    closed_summary = json.loads(closed_loop_path.with_name(f"{closed_loop_path.stem}_summary.json").read_text())
    no_action_summary = json.loads(no_action_path.with_name(f"{no_action_path.stem}_summary.json").read_text())

    assert closed_summary["final_homeostatic_error"] < no_action_summary["final_homeostatic_error"]
    assert closed_summary["mean_homeostatic_error"] < no_action_summary["mean_homeostatic_error"]
