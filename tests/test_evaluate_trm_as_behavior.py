from __future__ import annotations

import json
from pathlib import Path

from trm_pipeline.evaluate_trm_as_behavior import evaluate_compare_root


def test_evaluate_compare_root_reports_stress_action_diagnostics(tmp_path: Path) -> None:
    compare_root = tmp_path / "compare_root"
    mode_root = compare_root / "analytic__module_primary"
    mode_root.mkdir(parents=True)
    history = [
        {
            "t": 0,
            "G": 0.75,
            "B": 0.62,
            "action": "intake",
            "contact_thermal": 0.40,
            "contact_toxicity": 0.36,
        },
        {
            "t": 1,
            "G": 0.72,
            "B": 0.58,
            "action": "reconfigure",
            "contact_thermal": 0.42,
            "contact_toxicity": 0.38,
        },
        {
            "t": 2,
            "G": 0.53,
            "B": 0.66,
            "action": "withdraw",
            "contact_thermal": 0.10,
            "contact_toxicity": 0.12,
        },
    ]
    (mode_root / "episode_history.json").write_text(json.dumps(history), encoding="utf-8")
    summary = {
        "derived": {
            "best_mode_by_final_homeostasis": "analytic__module_primary",
            "best_mode_by_mean_homeostasis": "analytic__module_primary",
        },
        "results": {
            "analytic__module_primary": {
                "runtime_config": {"G_target": 0.55, "B_target": 0.65},
            }
        },
    }
    (compare_root / "comparison_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_compare_root(compare_root, stress_threshold=0.35)
    diag = result["mode_diagnostics"]["analytic__module_primary"]

    assert result["best_mode_by_final_homeostasis"] == "analytic__module_primary"
    assert diag["num_steps"] == 3
    assert diag["mean_G_overshoot"] > 0.0
    assert diag["mean_B_undershoot"] > 0.0
    assert diag["mean_stress_load"] > 0.0
    assert diag["stress_step_fraction"] == 2 / 3
    assert diag["stress_exploit_rate"] == 0.5
    assert diag["stress_defensive_rate"] == 0.5
