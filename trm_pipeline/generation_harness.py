from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .adaptive_controller import AdaptiveControllerConfig
from .common import ensure_dir, load_json, save_json, save_jsonl
from .erie_runtime import (
    EnvironmentConfig,
    RuntimeConfig,
    add_environment_config_args,
    environment_config_from_args,
    run_episode,
)
from .environment_curriculum import (
    add_environment_regime_args,
    environment_config_for_regime,
    normalize_regime_names,
    regime_manifest,
)


RUNTIME_GENES: dict[str, tuple[float, float]] = {
    "beta_pi": (0.5, 8.0),
    "contact_w_energy": (0.05, 2.0),
    "contact_w_thermal": (0.05, 2.5),
    "contact_w_toxicity": (0.05, 2.8),
    "contact_w_niche": (0.05, 2.0),
    "aperture_gain": (0.05, 1.2),
    "lookahead_discount": (0.40, 0.98),
}

LENIA_GENES: dict[str, tuple[float, float]] = {
    "m": (0.23, 0.41),
    "s": (0.033, 0.080),
}


@dataclass(frozen=True)
class GenerationHarnessConfig:
    output_root: str | Path
    seed_catalog: str = "data/lenia_official/animals2d_seeds.json"
    generations: int = 2
    population_size: int = 4
    elite_count: int = 2
    episodes_per_candidate: int = 1
    steps: int = 16
    warmup_steps: int = 2
    seed: int = 20260504
    mutation_scale: float = 0.10
    adaptive_controller: bool = True
    adaptive_interval: int = 4
    adaptive_window_size: int = 8
    adaptive_learning_rate: float = 0.08
    policy_mode: str = "closed_loop"
    environment_regimes: tuple[str, ...] = ("balanced",)


def _clip_gene(name: str, value: float, ranges: dict[str, tuple[float, float]]) -> float:
    low, high = ranges[name]
    return float(np.clip(float(value), low, high))


def _finite_mean(values: list[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float(default)
    return float(sum(finite) / len(finite))


def initial_genome(runtime_config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = runtime_config or RuntimeConfig()
    return {
        "runtime": {
            name: _clip_gene(name, float(getattr(cfg, name)), RUNTIME_GENES)
            for name in RUNTIME_GENES
        },
        "lenia": {
            "m": 0.33,
            "s": 0.055,
        },
    }


def apply_genome_to_runtime_config(base: RuntimeConfig, genome: dict[str, Any]) -> RuntimeConfig:
    updates = {}
    for name, raw_value in dict(genome.get("runtime", {})).items():
        if name in RUNTIME_GENES:
            updates[name] = _clip_gene(name, float(raw_value), RUNTIME_GENES)
    return replace(base, **updates)


def mutate_genome(
    genome: dict[str, Any],
    rng: np.random.Generator,
    *,
    mutation_scale: float,
) -> dict[str, Any]:
    scale = max(0.0, float(mutation_scale))
    mutated = {
        "runtime": dict(genome.get("runtime", {})),
        "lenia": dict(genome.get("lenia", {})),
    }
    for name, (low, high) in RUNTIME_GENES.items():
        base = float(mutated["runtime"].get(name, initial_genome()["runtime"][name]))
        sigma = scale * (high - low)
        mutated["runtime"][name] = _clip_gene(name, base + float(rng.normal(0.0, sigma)), RUNTIME_GENES)
    for name, (low, high) in LENIA_GENES.items():
        base = float(mutated["lenia"].get(name, initial_genome()["lenia"][name]))
        sigma = scale * (high - low)
        mutated["lenia"][name] = _clip_gene(name, base + float(rng.normal(0.0, sigma)), LENIA_GENES)
    return mutated


def fitness_from_summary(summary: dict[str, Any]) -> float:
    mean_homeostatic_error = float(summary.get("mean_homeostatic_error", 1.0))
    final_homeostatic_error = float(summary.get("final_homeostatic_error", mean_homeostatic_error))
    survival_fraction = float(summary.get("survival_fraction", 0.0))
    action_diversity = float(summary.get("action_diversity", 0.0))
    energy_contact = float(summary.get("mean_contact_energy", summary.get("mean_contact_resource", 0.0)))
    niche_contact = float(summary.get("mean_contact_niche", summary.get("mean_contact_shelter", 0.0)))
    thermal_contact = float(summary.get("mean_contact_thermal", 0.0))
    toxicity_contact = float(summary.get("mean_contact_toxicity", 0.0))
    invalid_count = float(summary.get("invalid_body_state_count", 0.0))
    death_events = float(summary.get("death_events_total", summary.get("death_events", 0.0)))
    return float(
        1.00 * survival_fraction
        + 0.25 * action_diversity
        + 0.18 * energy_contact
        + 0.10 * niche_contact
        - 1.00 * mean_homeostatic_error
        - 0.50 * final_homeostatic_error
        - 0.20 * thermal_contact
        - 0.25 * toxicity_contact
        - 0.10 * invalid_count
        - 0.25 * death_events
    )


def _aggregate_episode_summaries(summaries: list[dict[str, Any]]) -> dict[str, float]:
    keys = {
        "fitness",
        "mean_homeostatic_error",
        "final_homeostatic_error",
        "survival_fraction",
        "action_diversity",
        "mean_contact_energy",
        "mean_contact_thermal",
        "mean_contact_toxicity",
        "mean_contact_niche",
        "adaptive_event_count",
    }
    aggregate: dict[str, float] = {}
    for key in sorted(keys):
        aggregate[key] = _finite_mean([float(summary.get(key, 0.0)) for summary in summaries])
    fitness_by_regime: dict[str, float] = {}
    for summary in summaries:
        regime = str(summary.get("environment_regime", "balanced"))
        fitness_by_regime.setdefault(regime, 0.0)
    for regime in fitness_by_regime:
        fitness_by_regime[regime] = _finite_mean(
            [
                float(summary.get("fitness", 0.0))
                for summary in summaries
                if str(summary.get("environment_regime", "balanced")) == regime
            ]
        )
    if fitness_by_regime:
        worst = min(fitness_by_regime.values())
        mean = _finite_mean(list(fitness_by_regime.values()))
        aggregate["worst_regime_fitness"] = float(worst)
        aggregate["generalization_score"] = float(0.70 * worst + 0.30 * mean)
    else:
        aggregate["worst_regime_fitness"] = float(aggregate.get("fitness", 0.0))
        aggregate["generalization_score"] = float(aggregate.get("fitness", 0.0))
    return aggregate


def _aggregate_by_regime(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    regimes = sorted({str(summary.get("environment_regime", "balanced")) for summary in summaries})
    out: dict[str, dict[str, float]] = {}
    for regime in regimes:
        rows = [
            summary
            for summary in summaries
            if str(summary.get("environment_regime", "balanced")) == regime
        ]
        out[regime] = _aggregate_episode_summaries(rows)
    return out


def _inherited_genome(parent_genome: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    inherited = {
        "runtime": dict(parent_genome.get("runtime", {})),
        "lenia": dict(parent_genome.get("lenia", {})),
    }
    for name in RUNTIME_GENES:
        values = [
            float(summary.get("final_runtime_config", {}).get(name))
            for summary in summaries
            if name in dict(summary.get("final_runtime_config", {}))
        ]
        if values:
            inherited["runtime"][name] = _clip_gene(name, _finite_mean(values), RUNTIME_GENES)
    for name in LENIA_GENES:
        values = [
            float(summary.get("final_lenia_params", {}).get(name))
            for summary in summaries
            if name in dict(summary.get("final_lenia_params", {}))
        ]
        if values:
            inherited["lenia"][name] = _clip_gene(name, _finite_mean(values), LENIA_GENES)
    return inherited


def evaluate_candidate(
    *,
    genome: dict[str, Any],
    candidate_id: str,
    generation_index: int,
    candidate_index: int,
    config: GenerationHarnessConfig,
    env_config: EnvironmentConfig,
    output_root: Path,
) -> dict[str, Any]:
    episode_summaries: list[dict[str, Any]] = []
    regime_names = normalize_regime_names(list(config.environment_regimes))
    base_runtime = RuntimeConfig(
        steps=int(config.steps),
        warmup_steps=int(config.warmup_steps),
        seed=int(config.seed),
        policy_mode=str(config.policy_mode),
    )
    runtime_with_genome = apply_genome_to_runtime_config(base_runtime, genome)
    adaptive_config = AdaptiveControllerConfig(
        enabled=bool(config.adaptive_controller),
        interval=int(config.adaptive_interval),
        window_size=int(config.adaptive_window_size),
        min_steps=min(int(config.adaptive_window_size), max(1, int(config.steps))),
        learning_rate=float(config.adaptive_learning_rate),
        lenia_mu_center=float(genome["lenia"]["m"]),
        lenia_sigma_center=float(genome["lenia"]["s"]),
    )

    for regime_index, regime_name in enumerate(regime_names):
        regime_env_config = environment_config_for_regime(env_config, regime_name)
        for episode_index in range(int(config.episodes_per_candidate)):
            episode_seed = (
                int(config.seed)
                + 1_000_000 * int(generation_index)
                + 10_000 * int(candidate_index)
                + 1_000 * int(regime_index)
                + int(episode_index)
            )
            runtime_config = replace(runtime_with_genome, seed=episode_seed)
            episode_root = ensure_dir(
                output_root
                / f"generation_{generation_index:03d}"
                / candidate_id
                / str(regime_name)
                / f"episode_{episode_index:03d}"
            )
            episode_path = run_episode(
                episode_root,
                config.seed_catalog,
                runtime_config,
                regime_env_config,
                adaptive_controller_config=adaptive_config,
                initial_lenia_params=dict(genome.get("lenia", {})),
            )
            summary_path = episode_path.with_name(f"{episode_path.stem}_summary.json")
            summary = load_json(summary_path)
            summary["environment_regime"] = regime_name
            summary["environment_regime_index"] = int(regime_index)
            summary["fitness"] = fitness_from_summary(summary)
            episode_summaries.append(summary)

    aggregate = _aggregate_episode_summaries(episode_summaries)
    by_regime = _aggregate_by_regime(episode_summaries)
    inherited = _inherited_genome(genome, episode_summaries)
    return {
        "candidate_id": candidate_id,
        "generation_index": int(generation_index),
        "candidate_index": int(candidate_index),
        "genome": genome,
        "inherited_genome": inherited,
        "aggregate": aggregate,
        "fitness": float(aggregate["generalization_score"]),
        "mean_fitness": float(aggregate["fitness"]),
        "worst_regime_fitness": float(aggregate["worst_regime_fitness"]),
        "generalization_score": float(aggregate["generalization_score"]),
        "by_regime": by_regime,
        "episode_summaries": [
            {
                "episode_id": summary.get("episode_id"),
                "seed_id": summary.get("seed_id"),
                "environment_regime": str(summary.get("environment_regime", "balanced")),
                "fitness": float(summary["fitness"]),
                "mean_homeostatic_error": float(summary.get("mean_homeostatic_error", 0.0)),
                "final_homeostatic_error": float(summary.get("final_homeostatic_error", 0.0)),
                "survival_fraction": float(summary.get("survival_fraction", 0.0)),
                "adaptive_event_count": int(summary.get("adaptive_event_count", 0)),
                "summary_path": str(
                    Path(output_root)
                    / f"generation_{generation_index:03d}"
                    / candidate_id
                    / str(summary.get("environment_regime", "balanced"))
                    / f"episode_{index % max(1, int(config.episodes_per_candidate)):03d}"
                    / f"{summary.get('episode_id')}_summary.json"
                ),
            }
            for index, summary in enumerate(episode_summaries)
        ],
    }


def run_generation_harness(
    config: GenerationHarnessConfig,
    env_config: EnvironmentConfig | None = None,
) -> dict[str, Any]:
    output_root = ensure_dir(config.output_root)
    rng = np.random.default_rng(int(config.seed))
    env = env_config or EnvironmentConfig()
    regime_names = normalize_regime_names(list(config.environment_regimes))
    population_size = max(1, int(config.population_size))
    elite_count = min(max(1, int(config.elite_count)), population_size)
    population = [initial_genome() for _ in range(population_size)]
    population = [
        genome if index == 0 else mutate_genome(genome, rng, mutation_scale=float(config.mutation_scale))
        for index, genome in enumerate(population)
    ]

    all_rows: list[dict[str, Any]] = []
    generation_reports: list[dict[str, Any]] = []
    for generation_index in range(max(1, int(config.generations))):
        candidate_reports: list[dict[str, Any]] = []
        for candidate_index, genome in enumerate(population):
            candidate_id = f"candidate_{candidate_index:03d}"
            report = evaluate_candidate(
                genome=genome,
                candidate_id=candidate_id,
                generation_index=generation_index,
                candidate_index=candidate_index,
                config=config,
                env_config=env,
                output_root=output_root,
            )
            candidate_reports.append(report)
            all_rows.append(
                {
                    "generation_index": generation_index,
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "fitness": report["fitness"],
                    "mean_fitness": report["mean_fitness"],
                    "worst_regime_fitness": report["worst_regime_fitness"],
                    "generalization_score": report["generalization_score"],
                    "by_regime": report["by_regime"],
                    "aggregate": report["aggregate"],
                    "genome": report["genome"],
                    "inherited_genome": report["inherited_genome"],
                }
            )
        ranked = sorted(candidate_reports, key=lambda row: float(row["generalization_score"]), reverse=True)
        elites = ranked[:elite_count]
        generation_reports.append(
            {
                "generation_index": generation_index,
                "best_candidate_id": elites[0]["candidate_id"],
                "best_fitness": float(elites[0]["fitness"]),
                "best_generalization_score": float(elites[0]["generalization_score"]),
                "best_worst_regime_fitness": float(elites[0]["worst_regime_fitness"]),
                "mean_fitness": _finite_mean([float(row["fitness"]) for row in candidate_reports]),
                "mean_generalization_score": _finite_mean(
                    [float(row["generalization_score"]) for row in candidate_reports]
                ),
                "elite_candidate_ids": [row["candidate_id"] for row in elites],
                "candidates": candidate_reports,
            }
        )
        if generation_index < max(1, int(config.generations)) - 1:
            next_population = [dict(elite["inherited_genome"]) for elite in elites]
            while len(next_population) < population_size:
                parent = elites[(len(next_population) - elite_count) % len(elites)]["inherited_genome"]
                next_population.append(
                    mutate_genome(parent, rng, mutation_scale=float(config.mutation_scale))
                )
            population = next_population[:population_size]

    best = max(
        (candidate for generation in generation_reports for candidate in generation["candidates"]),
        key=lambda row: float(row["generalization_score"]),
    )
    config_dict = asdict(config)
    config_dict["output_root"] = str(config.output_root)
    config_dict["environment_regimes"] = regime_names
    summary = {
        "version": 1,
        "harness": "generation_harness",
        "config": config_dict,
        "environment_config": asdict(env),
        "environment_regimes": regime_manifest(env, regime_names),
        "generations": [
            {
                "generation_index": generation["generation_index"],
                "best_candidate_id": generation["best_candidate_id"],
                "best_fitness": generation["best_fitness"],
                "best_generalization_score": generation["best_generalization_score"],
                "best_worst_regime_fitness": generation["best_worst_regime_fitness"],
                "mean_fitness": generation["mean_fitness"],
                "mean_generalization_score": generation["mean_generalization_score"],
                "elite_candidate_ids": generation["elite_candidate_ids"],
            }
            for generation in generation_reports
        ],
        "best_candidate": {
            "generation_index": int(best["generation_index"]),
            "candidate_id": best["candidate_id"],
            "fitness": float(best["fitness"]),
            "mean_fitness": float(best["mean_fitness"]),
            "worst_regime_fitness": float(best["worst_regime_fitness"]),
            "generalization_score": float(best["generalization_score"]),
            "aggregate": best["aggregate"],
            "by_regime": best["by_regime"],
            "genome": best["genome"],
            "inherited_genome": best["inherited_genome"],
        },
    }
    save_json(output_root / "generation_summary.json", summary)
    save_jsonl(output_root / "generation_candidates.jsonl", all_rows)
    save_json(output_root / "selected_genome.json", best["inherited_genome"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation-level selection over ERIE-on-Lenia parameters.")
    parser.add_argument("--output-root", default="artifacts/generation_harness")
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--episodes-per-candidate", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--mutation-scale", type=float, default=0.10)
    parser.add_argument("--policy-mode", choices=("closed_loop", "random", "no_action"), default="closed_loop")
    add_environment_regime_args(parser)
    parser.add_argument("--disable-adaptive-controller", action="store_true")
    parser.add_argument("--adaptive-interval", type=int, default=4)
    parser.add_argument("--adaptive-window-size", type=int, default=8)
    parser.add_argument("--adaptive-learning-rate", type=float, default=0.08)
    add_environment_config_args(parser)
    args = parser.parse_args()
    summary = run_generation_harness(
        GenerationHarnessConfig(
            output_root=args.output_root,
            seed_catalog=args.seed_catalog,
            generations=args.generations,
            population_size=args.population_size,
            elite_count=args.elite_count,
            episodes_per_candidate=args.episodes_per_candidate,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            mutation_scale=args.mutation_scale,
            adaptive_controller=not bool(args.disable_adaptive_controller),
            adaptive_interval=args.adaptive_interval,
            adaptive_window_size=args.adaptive_window_size,
            adaptive_learning_rate=args.adaptive_learning_rate,
            policy_mode=args.policy_mode,
            environment_regimes=tuple(normalize_regime_names(args.environment_regimes)),
        ),
        env_config=environment_config_from_args(args),
    )
    print(f"wrote generation harness summary: {Path(summary['config']['output_root']) / 'generation_summary.json'}")


if __name__ == "__main__":
    main()
