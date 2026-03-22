from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .common import ensure_dir, load_json, save_json
from .erie_runtime import EnvironmentConfig, RuntimeConfig, run_episode


MODE_VALUES = ("analytic", "assistive", "module_primary")


def _summary_path_from_episode(path: str | Path) -> Path:
    episode_path = Path(path)
    return episode_path.with_name(f"{episode_path.stem}_summary.json")


def _ranking_by_metric(results: dict[str, dict], metric: str) -> list[str]:
    return sorted(
        results,
        key=lambda key: (
            bool(results[key].get("dead", False)),
            -float(results[key].get("survival_fraction", 0.0)),
            float(results[key].get(metric, float("inf"))),
            float(results[key].get("action_cost_total", float("inf"))),
        ),
    )


def _derived_comparison_local(results: dict[str, dict]) -> dict[str, object]:
    final_ranking = _ranking_by_metric(results, "final_homeostatic_error")
    mean_ranking = _ranking_by_metric(results, "mean_homeostatic_error")
    return {
        "best_mode_by_final_homeostasis": final_ranking[0] if final_ranking else None,
        "best_mode_by_mean_homeostasis": mean_ranking[0] if mean_ranking else None,
        "ranking_by_final_homeostasis": final_ranking,
        "ranking_by_mean_homeostasis": mean_ranking,
    }


def _mode_key(viability_mode: str, action_mode: str) -> str:
    return f"{viability_mode}__{action_mode}"


def _compare_one_seed(
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
    output_root = ensure_dir(output_root)
    env_config = EnvironmentConfig(
        resource_patches=resource_patches,
        hazard_patches=hazard_patches,
        shelter_patches=shelter_patches,
    )
    results: dict[str, dict] = {}
    for viability_mode in MODE_VALUES:
        for action_mode in MODE_VALUES:
            mode_key = _mode_key(viability_mode, action_mode)
            mode_root = ensure_dir(Path(output_root) / mode_key)
            runtime_config = RuntimeConfig(
                steps=steps,
                warmup_steps=warmup_steps,
                seed=seed,
                lookahead_horizon=lookahead_horizon,
                lookahead_discount=lookahead_discount,
                viability_mode=viability_mode,
                action_mode=action_mode,
                use_trm_a=bool(trm_a_checkpoint),
                use_trm_b=bool(trm_b_checkpoint),
                policy_mode=policy_mode,
            )
            episode_path = run_episode(
                mode_root,
                seed_catalog,
                runtime_config,
                env_config,
                trm_a_checkpoint=trm_a_checkpoint,
                trm_b_checkpoint=trm_b_checkpoint,
                module_manifest=module_manifest,
            )
            results[mode_key] = load_json(_summary_path_from_episode(episode_path))
    comparison = {
        "seed": seed,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "policy_mode": policy_mode,
        "results": results,
        "derived": _derived_comparison_local(results),
    }
    save_json(Path(output_root) / "comparison_summary.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep TRM-Vm/TRM-As mode comparisons across seeds.")
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--output-root", default="artifacts/trm_va_mode_sweep")
    parser.add_argument("--seed-start", type=int, default=20260318)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--lookahead-horizon", type=int, default=2)
    parser.add_argument("--lookahead-discount", type=float, default=0.85)
    parser.add_argument("--resource-patches", type=int, default=3)
    parser.add_argument("--hazard-patches", type=int, default=3)
    parser.add_argument("--shelter-patches", type=int, default=1)
    parser.add_argument("--trm-a-checkpoint", default=None)
    parser.add_argument("--trm-b-checkpoint", default=None)
    parser.add_argument("--module-manifest", default=None)
    parser.add_argument(
        "--policy-mode",
        choices=("closed_loop", "random", "no_action"),
        default="closed_loop",
    )
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    per_seed = []
    final_counter: Counter[str] = Counter()
    mean_counter: Counter[str] = Counter()
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        seed_root = ensure_dir(output_root / f"seed_{seed}")
        comparison = _compare_one_seed(
            output_root=seed_root,
            seed_catalog=args.seed_catalog,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=seed,
            lookahead_horizon=args.lookahead_horizon,
            lookahead_discount=args.lookahead_discount,
            resource_patches=args.resource_patches,
            hazard_patches=args.hazard_patches,
            shelter_patches=args.shelter_patches,
            trm_a_checkpoint=args.trm_a_checkpoint,
            trm_b_checkpoint=args.trm_b_checkpoint,
            module_manifest=args.module_manifest,
            policy_mode=args.policy_mode,
        )
        per_seed.append(
            {
                "seed": seed,
                "best_mode_by_final_homeostasis": comparison["derived"]["best_mode_by_final_homeostasis"],
                "best_mode_by_mean_homeostasis": comparison["derived"]["best_mode_by_mean_homeostasis"],
            }
        )
        final_counter.update([comparison["derived"]["best_mode_by_final_homeostasis"]])
        mean_counter.update([comparison["derived"]["best_mode_by_mean_homeostasis"]])

    aggregate = {
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "policy_mode": args.policy_mode,
        "module_manifest": args.module_manifest,
        "counts_by_best_final_homeostasis": dict(final_counter),
        "counts_by_best_mean_homeostasis": dict(mean_counter),
        "per_seed": per_seed,
    }
    save_json(output_root / "aggregate_summary.json", aggregate)
    print(f"wrote TRM-Vm/TRM-As sweep: {output_root / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
