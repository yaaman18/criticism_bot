from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trm_pipeline import compare_trm_mc_modes


def test_compare_trm_mc_modes_writes_grid_summary(tmp_path: Path, monkeypatch) -> None:
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
            ("analytic", "analytic"): 0.60,
            ("analytic", "assistive"): 0.50,
            ("assistive", "analytic"): 0.35,
            ("assistive", "assistive"): 0.20,
            ("module_primary", "analytic"): 0.42,
            ("module_primary", "assistive"): 0.30,
        }[(runtime_config.action_mode, runtime_config.context_memory_mode)]
        summary_path = out_root / "episode_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "dead": False,
                    "survival_fraction": 1.0,
                    "final_homeostatic_error": score,
                    "mean_homeostatic_error": score + 0.1,
                    "action_cost_total": 0.1,
                    "viability_mode": runtime_config.viability_mode,
                    "action_mode": runtime_config.action_mode,
                    "boundary_control_mode": runtime_config.boundary_control_mode,
                    "context_memory_mode": runtime_config.context_memory_mode,
                    "policy_mode": runtime_config.policy_mode,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return episode_path

    monkeypatch.setattr(compare_trm_mc_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trm_mc_modes",
            "--output-root",
            str(tmp_path / "compare_mc_modes"),
            "--policy-mode",
            "closed_loop",
            "--viability-mode",
            "assistive",
            "--boundary-control-mode",
            "assistive",
            "--module-manifest",
            str(tmp_path / "modules.json"),
        ],
    )

    compare_trm_mc_modes.main()

    summary = json.loads((tmp_path / "compare_mc_modes" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["policy_mode"] == "closed_loop"
    assert summary["viability_mode"] == "assistive"
    assert summary["boundary_control_mode"] == "assistive"
    assert len(summary["results"]) == 6
    assert summary["derived"]["best_mode_by_final_homeostasis"] == "assistive__assistive"
    assert summary["derived"]["best_mode_by_mean_homeostasis"] == "assistive__assistive"
    assert summary["derived"]["context_gain_by_action_mode"]["assistive"]["final_homeostatic_error_delta"] == pytest.approx(0.15)
    assert len(captured) == 6
    for runtime_config, env_config, module_specs, module_manifest in captured:
        assert runtime_config.policy_mode == "closed_loop"
        assert runtime_config.viability_mode == "assistive"
        assert runtime_config.boundary_control_mode == "assistive"
        assert env_config.resource_patches == 3
        assert module_specs is None
        assert module_manifest == str(tmp_path / "modules.json")


def test_compare_trm_mc_modes_forwards_knobs(tmp_path: Path, monkeypatch) -> None:
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
                }
            ),
            encoding="utf-8",
        )
        return episode_path

    monkeypatch.setattr(compare_trm_mc_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trm_mc_modes",
            "--output-root",
            str(tmp_path / "compare_mc_modes_knobs"),
            "--steps",
            "5",
            "--warmup-steps",
            "1",
            "--seed",
            "31415",
            "--lookahead-horizon",
            "4",
            "--lookahead-discount",
            "0.9",
            "--resource-patches",
            "4",
            "--hazard-patches",
            "2",
            "--shelter-patches",
            "2",
            "--policy-mode",
            "random",
            "--viability-mode",
            "analytic",
            "--boundary-control-mode",
            "module_primary",
            "--context-memory-window-size",
            "12",
            "--context-memory-residual-scale",
            "0.5",
        ],
    )

    compare_trm_mc_modes.main()

    assert len(captured) == 6
    for runtime_config, env_config in captured:
        assert runtime_config.steps == 5
        assert runtime_config.warmup_steps == 1
        assert runtime_config.seed == 31415
        assert runtime_config.lookahead_horizon == 4
        assert runtime_config.lookahead_discount == 0.9
        assert runtime_config.policy_mode == "random"
        assert runtime_config.viability_mode == "analytic"
        assert runtime_config.boundary_control_mode == "module_primary"
        assert runtime_config.context_memory_window_size == 12
        assert runtime_config.context_memory_residual_scale == 0.5
        assert env_config.resource_patches == 4
        assert env_config.hazard_patches == 2
        assert env_config.shelter_patches == 2


def test_compare_trm_mc_modes_applies_episode_family_overrides(tmp_path: Path, monkeypatch) -> None:
    captured: list[tuple[object, object]] = []
    override_calls: list[tuple[str, int, float]] = []

    def fake_sample_episode_configs_for_family(family, base_runtime_config, base_env_config, seed, defensive_family_bias=0.0):
        override_calls.append((family, seed, defensive_family_bias))
        runtime_config = base_runtime_config.__class__(
            **{
                **base_runtime_config.__dict__,
                "G0": 0.33,
                "B0": 0.27,
                "observation_noise": 0.031,
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
                }
            ),
            encoding="utf-8",
        )
        return episode_path

    monkeypatch.setattr(compare_trm_mc_modes, "sample_episode_configs_for_family", fake_sample_episode_configs_for_family)
    monkeypatch.setattr(compare_trm_mc_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trm_mc_modes",
            "--output-root",
            str(tmp_path / "compare_mc_modes_family"),
            "--seed",
            "27182",
            "--episode-family",
            "fragile_boundary",
            "--defensive-family-bias",
            "2.0",
        ],
    )

    compare_trm_mc_modes.main()

    assert override_calls == [("fragile_boundary", 27182, 2.0)]
    assert len(captured) == 6
    for runtime_config, env_config in captured:
        assert runtime_config.G0 == 0.33
        assert runtime_config.B0 == 0.27
        assert runtime_config.observation_noise == 0.031
        assert env_config.hazard_patches == 5
        assert env_config.shelter_patches == 0
    summary = json.loads((tmp_path / "compare_mc_modes_family" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["episode_family"] == "fragile_boundary"
    assert summary["defensive_family_bias"] == 2.0
