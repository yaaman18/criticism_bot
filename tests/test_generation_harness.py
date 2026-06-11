from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from trm_pipeline import generation_harness
from trm_pipeline.common import load_json, save_json
from trm_pipeline.erie_runtime import EnvironmentConfig
from trm_pipeline.generation_harness import (
    GenerationHarnessConfig,
    fitness_from_summary,
    mutate_genome,
    run_generation_harness,
)


def test_fitness_prefers_surviving_low_error_candidate() -> None:
    weak = {
        "survival_fraction": 0.4,
        "mean_homeostatic_error": 0.5,
        "final_homeostatic_error": 0.6,
        "action_diversity": 0.1,
        "mean_contact_energy": 0.02,
        "mean_contact_thermal": 0.7,
        "mean_contact_toxicity": 0.6,
        "death_events_total": 1.0,
    }
    strong = {
        "survival_fraction": 1.0,
        "mean_homeostatic_error": 0.1,
        "final_homeostatic_error": 0.08,
        "action_diversity": 0.6,
        "mean_contact_energy": 0.25,
        "mean_contact_thermal": 0.1,
        "mean_contact_toxicity": 0.1,
        "death_events_total": 0.0,
    }

    assert fitness_from_summary(strong) > fitness_from_summary(weak)


def test_mutate_genome_keeps_values_inside_ranges() -> None:
    rng = generation_harness.np.random.default_rng(123)
    genome = {
        "runtime": {name: 100.0 for name in generation_harness.RUNTIME_GENES},
        "lenia": {name: -100.0 for name in generation_harness.LENIA_GENES},
    }

    mutated = mutate_genome(genome, rng, mutation_scale=10.0)

    for name, value in mutated["runtime"].items():
        low, high = generation_harness.RUNTIME_GENES[name]
        assert low <= value <= high
    for name, value in mutated["lenia"].items():
        low, high = generation_harness.LENIA_GENES[name]
        assert low <= value <= high


def test_generation_harness_runs_selection_and_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_episode(
        output_root,
        seed_catalog,
        runtime_config,
        env_config,
        trm_a_checkpoint=None,
        trm_b_checkpoint=None,
        module_specs=None,
        module_manifest=None,
        adaptive_controller_config=None,
        initial_lenia_params=None,
    ):
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        episode_path = root / f"fake_{runtime_config.seed}.npz"
        episode_path.write_bytes(b"npz")
        calls.append(
            {
                "runtime_config": runtime_config,
                "adaptive_controller_config": adaptive_controller_config,
                "initial_lenia_params": dict(initial_lenia_params or {}),
            }
        )
        score_index = len(calls)
        mean_error = 0.50 if score_index % 2 else 0.10
        final_m = float((initial_lenia_params or {}).get("m", 0.33)) + (0.01 if mean_error < 0.2 else -0.01)
        final_s = float((initial_lenia_params or {}).get("s", 0.055))
        save_json(
            root / f"{episode_path.stem}_summary.json",
            {
                "episode_id": episode_path.stem,
                "seed_id": "seed",
                "mean_homeostatic_error": mean_error,
                "final_homeostatic_error": mean_error,
                "survival_fraction": 1.0,
                "action_diversity": 0.5,
                "mean_contact_energy": 0.2,
                "mean_contact_thermal": 0.1,
                "mean_contact_toxicity": 0.1,
                "mean_contact_niche": 0.1,
                "adaptive_event_count": 1,
                "final_runtime_config": asdict(runtime_config),
                "final_lenia_params": {"m": final_m, "s": final_s},
            },
        )
        return episode_path

    monkeypatch.setattr(generation_harness, "run_episode", fake_run_episode)

    summary = run_generation_harness(
        GenerationHarnessConfig(
            output_root=tmp_path / "generations",
            seed_catalog="catalog.json",
            generations=2,
            population_size=3,
            elite_count=1,
            episodes_per_candidate=1,
            steps=4,
            warmup_steps=1,
            seed=9000,
            mutation_scale=0.0,
            adaptive_controller=True,
            adaptive_interval=1,
            adaptive_window_size=2,
        ),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
    )

    output_root = tmp_path / "generations"
    assert len(calls) == 6
    assert len(summary["generations"]) == 2
    assert summary["best_candidate"]["fitness"] == summary["generations"][1]["best_fitness"]
    assert (output_root / "generation_summary.json").exists()
    assert (output_root / "generation_candidates.jsonl").exists()
    selected = load_json(output_root / "selected_genome.json")
    assert set(selected) == {"runtime", "lenia"}
    assert calls[0]["adaptive_controller_config"].enabled is True
    assert calls[0]["initial_lenia_params"]["m"] == 0.33


def test_generation_harness_evaluates_all_environment_regimes(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_episode(
        output_root,
        seed_catalog,
        runtime_config,
        env_config,
        trm_a_checkpoint=None,
        trm_b_checkpoint=None,
        module_specs=None,
        module_manifest=None,
        adaptive_controller_config=None,
        initial_lenia_params=None,
    ):
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        episode_path = root / f"fake_{runtime_config.seed}.npz"
        episode_path.write_bytes(b"npz")
        is_hard = env_config.hazard_patches >= 6
        calls.append({"env_config": env_config, "output_root": root})
        save_json(
            root / f"{episode_path.stem}_summary.json",
            {
                "episode_id": episode_path.stem,
                "seed_id": "seed",
                "mean_homeostatic_error": 0.30 if is_hard else 0.10,
                "final_homeostatic_error": 0.25 if is_hard else 0.08,
                "survival_fraction": 1.0,
                "action_diversity": 0.5,
                "mean_contact_energy": 0.2,
                "mean_contact_thermal": 0.5 if is_hard else 0.1,
                "mean_contact_toxicity": 0.5 if is_hard else 0.1,
                "mean_contact_niche": 0.1,
                "adaptive_event_count": 1,
                "final_runtime_config": asdict(runtime_config),
                "final_lenia_params": dict(initial_lenia_params or {"m": 0.33, "s": 0.055}),
            },
        )
        return episode_path

    monkeypatch.setattr(generation_harness, "run_episode", fake_run_episode)

    summary = run_generation_harness(
        GenerationHarnessConfig(
            output_root=tmp_path / "regime_generations",
            seed_catalog="catalog.json",
            generations=1,
            population_size=1,
            elite_count=1,
            episodes_per_candidate=2,
            steps=4,
            warmup_steps=1,
            seed=9100,
            mutation_scale=0.0,
            environment_regimes=("easy", "hard"),
        ),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
    )

    assert len(calls) == 4
    assert [row["name"] for row in summary["environment_regimes"]] == ["easy", "hard"]
    assert set(summary["best_candidate"]["by_regime"]) == {"easy", "hard"}
    assert summary["best_candidate"]["worst_regime_fitness"] == summary["best_candidate"]["by_regime"]["hard"]["fitness"]
    assert summary["best_candidate"]["generalization_score"] < summary["best_candidate"]["mean_fitness"]
    assert any("hard" in str(call["output_root"]) for call in calls)
