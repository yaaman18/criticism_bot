from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trm_pipeline import sweep_trm_mc_modes


def test_sweep_trm_mc_modes_writes_aggregate_summary(tmp_path: Path, monkeypatch) -> None:
    captured: list[tuple[object, object, object]] = []

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
        captured.append((runtime_config, env_config, module_manifest))
        out_root = Path(output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        episode_path = out_root / "episode.npz"
        episode_path.write_bytes(b"fake")
        score = {
            ("analytic", "analytic"): 0.60,
            ("analytic", "assistive"): 0.45,
            ("assistive", "analytic"): 0.35,
            ("assistive", "assistive"): 0.20,
            ("module_primary", "analytic"): 0.42,
            ("module_primary", "assistive"): 0.28,
        }[(runtime_config.action_mode, runtime_config.context_memory_mode)]
        summary = {
            "dead": False,
            "survival_fraction": 1.0,
            "final_homeostatic_error": score,
            "mean_homeostatic_error": score + 0.1,
            "action_cost_total": 0.1,
        }
        (out_root / "episode_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return episode_path

    monkeypatch.setattr(sweep_trm_mc_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_trm_mc_modes",
            "--output-root",
            str(tmp_path / "mc_sweep"),
            "--seed-start",
            "500",
            "--num-seeds",
            "3",
            "--policy-mode",
            "closed_loop",
            "--viability-mode",
            "assistive",
            "--boundary-control-mode",
            "assistive",
            "--module-manifest",
            str(tmp_path / "mods.json"),
        ],
    )

    sweep_trm_mc_modes.main()

    summary = json.loads((tmp_path / "mc_sweep" / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert summary["seed_start"] == 500
    assert summary["num_seeds"] == 3
    assert summary["counts_by_best_final_homeostasis"] == {"assistive__assistive": 3}
    assert summary["counts_by_best_mean_homeostasis"] == {"assistive__assistive": 3}
    assert summary["mean_context_gain_by_action_mode"]["assistive"]["final_homeostatic_error_delta"] == pytest.approx(0.15)
    assert len(summary["per_seed"]) == 3
    assert len(captured) == 18
    for runtime_config, env_config, module_manifest in captured:
        assert runtime_config.policy_mode == "closed_loop"
        assert runtime_config.viability_mode == "assistive"
        assert runtime_config.boundary_control_mode == "assistive"
        assert env_config.resource_patches == 3
        assert module_manifest == str(tmp_path / "mods.json")


def test_sweep_trm_mc_modes_applies_episode_family_overrides(tmp_path: Path, monkeypatch) -> None:
    override_calls: list[tuple[str, int, float]] = []
    captured: list[tuple[object, object]] = []

    def fake_sample_episode_configs_for_family(family, base_runtime_config, base_env_config, seed, defensive_family_bias=0.0):
        override_calls.append((family, seed, defensive_family_bias))
        runtime_config = base_runtime_config.__class__(
            **{
                **base_runtime_config.__dict__,
                "G0": 0.29,
                "B0": 0.25,
                "observation_noise": 0.041,
            }
        )
        env_config = base_env_config.__class__(
            **{
                **base_env_config.__dict__,
                "hazard_patches": 4,
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
        summary = {
            "dead": False,
            "survival_fraction": 1.0,
            "final_homeostatic_error": 0.2,
            "mean_homeostatic_error": 0.3,
            "action_cost_total": 0.1,
        }
        (out_root / "episode_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return episode_path

    monkeypatch.setattr(sweep_trm_mc_modes, "sample_episode_configs_for_family", fake_sample_episode_configs_for_family)
    monkeypatch.setattr(sweep_trm_mc_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_trm_mc_modes",
            "--output-root",
            str(tmp_path / "mc_sweep_family"),
            "--seed-start",
            "600",
            "--num-seeds",
            "2",
            "--episode-family",
            "fragile_boundary",
            "--defensive-family-bias",
            "1.5",
        ],
    )

    sweep_trm_mc_modes.main()

    assert override_calls == [
        ("fragile_boundary", 600, 1.5),
        ("fragile_boundary", 601, 1.5),
    ]
    assert len(captured) == 12
    for runtime_config, env_config in captured:
        assert runtime_config.G0 == 0.29
        assert runtime_config.B0 == 0.25
        assert runtime_config.observation_noise == 0.041
        assert env_config.hazard_patches == 4
        assert env_config.shelter_patches == 0

    summary = json.loads((tmp_path / "mc_sweep_family" / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert summary["episode_family"] == "fragile_boundary"
    assert summary["defensive_family_bias"] == 1.5
