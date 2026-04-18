from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trm_pipeline import compare_trm_ag_modes


def test_compare_trm_ag_modes_writes_summary(tmp_path: Path, monkeypatch) -> None:
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
        episode_path = out_root / "episode.npz"
        episode_path.write_bytes(b"fake")
        score = {
            "analytic": 0.60,
            "assistive": 0.42,
            "module_primary": 0.48,
        }[runtime_config.action_gating_mode]
        (out_root / "episode_summary.json").write_text(
            json.dumps(
                {
                    "dead": False,
                    "survival_fraction": 1.0,
                    "final_homeostatic_error": score,
                    "mean_homeostatic_error": score + 0.1,
                    "action_cost_total": 0.1,
                    "stress_exploit_rate": score / 2.0,
                    "dominant_action_fraction": 0.5,
                    "action_gating_mode": runtime_config.action_gating_mode,
                }
            ),
            encoding="utf-8",
        )
        return episode_path

    monkeypatch.setattr(compare_trm_ag_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trm_ag_modes",
            "--output-root",
            str(tmp_path / "compare_ag_modes"),
            "--policy-mode",
            "closed_loop",
            "--viability-mode",
            "assistive",
            "--action-mode",
            "assistive",
            "--boundary-control-mode",
            "assistive",
            "--context-memory-mode",
            "assistive",
            "--module-manifest",
            str(tmp_path / "modules.json"),
        ],
    )

    compare_trm_ag_modes.main()

    summary = json.loads((tmp_path / "compare_ag_modes" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["policy_mode"] == "closed_loop"
    assert summary["viability_mode"] == "assistive"
    assert summary["action_mode"] == "assistive"
    assert summary["boundary_control_mode"] == "assistive"
    assert summary["context_memory_mode"] == "assistive"
    assert set(summary["results"]) == {"analytic", "assistive", "module_primary"}
    assert summary["derived"]["best_mode_by_final_homeostasis"] == "assistive"
    assert summary["derived"]["best_mode_by_mean_homeostasis"] == "assistive"
    assert summary["derived"]["ag_gain"]["final_homeostatic_error_delta"] == pytest.approx(0.18)
    assert len(captured) == 3
    for runtime_config, env_config, module_specs, module_manifest in captured:
        assert runtime_config.policy_mode == "closed_loop"
        assert runtime_config.viability_mode == "assistive"
        assert runtime_config.action_mode == "assistive"
        assert runtime_config.boundary_control_mode == "assistive"
        assert runtime_config.context_memory_mode == "assistive"
        assert env_config.resource_patches == 3
        assert module_specs is None
        assert module_manifest == str(tmp_path / "modules.json")


def test_compare_trm_ag_modes_applies_episode_family_overrides(tmp_path: Path, monkeypatch) -> None:
    captured: list[tuple[object, object]] = []
    override_calls: list[tuple[str, int, float]] = []

    def fake_sample_episode_configs_for_family(
        family,
        base_runtime_config,
        base_env_config,
        seed,
        defensive_family_bias=0.0,
    ):
        override_calls.append((family, seed, defensive_family_bias))
        runtime_config = base_runtime_config.__class__(
            **{
                **base_runtime_config.__dict__,
                "G0": 0.31,
                "B0": 0.29,
            }
        )
        env_config = base_env_config.__class__(
            **{
                **base_env_config.__dict__,
                "hazard_patches": 5,
                "shelter_patches": 0,
            }
        )
        return runtime_config, env_config

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
        captured.append((runtime_config, env_config))
        out_root = Path(output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        episode_path = out_root / "episode.npz"
        episode_path.write_bytes(b"fake")
        (out_root / "episode_summary.json").write_text(
            json.dumps(
                {
                    "dead": False,
                    "survival_fraction": 1.0,
                    "final_homeostatic_error": 0.2,
                    "mean_homeostatic_error": 0.3,
                    "action_cost_total": 0.1,
                    "stress_exploit_rate": 0.1,
                    "dominant_action_fraction": 0.4,
                }
            ),
            encoding="utf-8",
        )
        return episode_path

    monkeypatch.setattr(compare_trm_ag_modes, "sample_episode_configs_for_family", fake_sample_episode_configs_for_family)
    monkeypatch.setattr(compare_trm_ag_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trm_ag_modes",
            "--output-root",
            str(tmp_path / "compare_ag_modes_family"),
            "--seed",
            "2718",
            "--episode-family",
            "vent_edge",
            "--defensive-family-bias",
            "1.5",
        ],
    )

    compare_trm_ag_modes.main()

    assert override_calls == [("vent_edge", 2718, 1.5)]
    assert len(captured) == 3
    for runtime_config, env_config in captured:
        assert runtime_config.G0 == pytest.approx(0.31)
        assert runtime_config.B0 == pytest.approx(0.29)
        assert env_config.hazard_patches == 5
        assert env_config.shelter_patches == 0
