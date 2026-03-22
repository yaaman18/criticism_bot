from __future__ import annotations

import json
import sys
from pathlib import Path

from trm_pipeline import sweep_trm_va_modes


def test_sweep_trm_va_modes_aggregates_seed_summaries(tmp_path: Path, monkeypatch) -> None:
    captured: list[int] = []

    def fake_compare_one_seed(
        *,
        output_root: str | Path,
        seed_catalog: str | Path,
        steps: int,
        warmup_steps: int,
        seed: int,
        lookahead_horizon: int,
        lookahead_discount: float,
        resource_patches: int,
        hazard_patches: int,
        shelter_patches: int,
        trm_a_checkpoint: str | None,
        trm_b_checkpoint: str | None,
        module_manifest: str | None,
        policy_mode: str,
    ) -> dict:
        captured.append(seed)
        Path(output_root).mkdir(parents=True, exist_ok=True)
        return {
            "seed": seed,
            "derived": {
                "best_mode_by_final_homeostasis": "analytic__assistive" if seed % 2 == 0 else "analytic__analytic",
                "best_mode_by_mean_homeostasis": "analytic__assistive",
            },
        }

    monkeypatch.setattr(sweep_trm_va_modes, "_compare_one_seed", fake_compare_one_seed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_trm_va_modes",
            "--output-root",
            str(tmp_path / "sweep"),
            "--seed-start",
            "10",
            "--num-seeds",
            "4",
            "--policy-mode",
            "closed_loop",
        ],
    )

    sweep_trm_va_modes.main()

    summary = json.loads((tmp_path / "sweep" / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert captured == [10, 11, 12, 13]
    assert summary["num_seeds"] == 4
    assert summary["counts_by_best_final_homeostasis"]["analytic__assistive"] == 2
    assert summary["counts_by_best_final_homeostasis"]["analytic__analytic"] == 2
    assert summary["counts_by_best_mean_homeostasis"]["analytic__assistive"] == 4
