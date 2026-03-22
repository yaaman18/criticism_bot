from __future__ import annotations

import json
import sys
from pathlib import Path

from trm_pipeline import compare_trm_va_modes


def test_compare_trm_va_modes_writes_grid_summary(tmp_path: Path, monkeypatch) -> None:
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
            ("analytic", "assistive"): 0.45,
            ("analytic", "module_primary"): 0.50,
            ("assistive", "analytic"): 0.40,
            ("assistive", "assistive"): 0.20,
            ("assistive", "module_primary"): 0.28,
            ("module_primary", "analytic"): 0.48,
            ("module_primary", "assistive"): 0.31,
            ("module_primary", "module_primary"): 0.38,
        }[(runtime_config.viability_mode, runtime_config.action_mode)]
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
                    "policy_mode": runtime_config.policy_mode,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return episode_path

    monkeypatch.setattr(compare_trm_va_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trm_va_modes",
            "--output-root",
            str(tmp_path / "compare_modes"),
            "--policy-mode",
            "closed_loop",
            "--module-manifest",
            str(tmp_path / "modules.json"),
        ],
    )

    compare_trm_va_modes.main()

    summary = json.loads((tmp_path / "compare_modes" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["policy_mode"] == "closed_loop"
    assert len(summary["results"]) == 9
    assert summary["derived"]["best_mode_by_final_homeostasis"] == "assistive__assistive"
    assert summary["derived"]["best_mode_by_mean_homeostasis"] == "assistive__assistive"
    assert summary["derived"]["ranking_by_final_homeostasis"][0] == "assistive__assistive"
    assert len(captured) == 9
    for runtime_config, env_config, module_specs, module_manifest in captured:
        assert runtime_config.policy_mode == "closed_loop"
        assert env_config.resource_patches == 3
        assert module_specs is None
        assert module_manifest == str(tmp_path / "modules.json")


def test_compare_trm_va_modes_forwards_knobs(tmp_path: Path, monkeypatch) -> None:
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

    monkeypatch.setattr(compare_trm_va_modes, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trm_va_modes",
            "--output-root",
            str(tmp_path / "compare_modes_knobs"),
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
        ],
    )

    compare_trm_va_modes.main()

    assert len(captured) == 9
    for runtime_config, env_config in captured:
        assert runtime_config.steps == 5
        assert runtime_config.warmup_steps == 1
        assert runtime_config.seed == 31415
        assert runtime_config.lookahead_horizon == 4
        assert runtime_config.lookahead_discount == 0.9
        assert runtime_config.policy_mode == "random"
        assert env_config.resource_patches == 4
        assert env_config.hazard_patches == 2
        assert env_config.shelter_patches == 2
