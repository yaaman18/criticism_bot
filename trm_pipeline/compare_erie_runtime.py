from __future__ import annotations

import argparse
from pathlib import Path

from .common import ensure_dir, load_json, save_json
from .erie_runtime import EnvironmentConfig, RuntimeConfig, run_episode


def _summary_path_from_episode(path: str | Path) -> Path:
    episode_path = Path(path)
    return episode_path.with_name(f"{episode_path.stem}_summary.json")


def _ranking_by_metric(
    results: dict[str, dict],
    metric: str,
) -> list[str]:
    return sorted(
        results,
        key=lambda mode: (
            bool(results[mode].get("dead", False)),
            -float(results[mode].get("survival_fraction", 0.0)),
            float(results[mode].get(metric, float("inf"))),
            float(results[mode].get("action_cost_total", float("inf"))),
        ),
    )


def _derived_comparison(results: dict[str, dict]) -> dict[str, object]:
    final_ranking = _ranking_by_metric(results, "final_homeostatic_error")
    mean_ranking = _ranking_by_metric(results, "mean_homeostatic_error")
    return {
        "best_mode_by_final_homeostasis": final_ranking[0] if final_ranking else None,
        "best_mode_by_mean_homeostasis": mean_ranking[0] if mean_ranking else None,
        "ranking_by_final_homeostasis": final_ranking,
        "ranking_by_mean_homeostasis": mean_ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ERIE runtime under closed_loop/random/no_action modes."
    )
    parser.add_argument(
        "--seed-catalog",
        default="data/lenia_official/animals2d_seeds.json",
    )
    parser.add_argument("--output-root", default="artifacts/erie_runtime_compare")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--lookahead-horizon", type=int, default=2)
    parser.add_argument("--lookahead-discount", type=float, default=0.85)
    parser.add_argument(
        "--viability-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument(
        "--action-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument("--resource-patches", type=int, default=3)
    parser.add_argument("--hazard-patches", type=int, default=3)
    parser.add_argument("--shelter-patches", type=int, default=1)
    parser.add_argument("--trm-a-checkpoint", default=None)
    parser.add_argument("--trm-b-checkpoint", default=None)
    parser.add_argument("--module-manifest", default=None)
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    env_config = EnvironmentConfig(
        resource_patches=args.resource_patches,
        hazard_patches=args.hazard_patches,
        shelter_patches=args.shelter_patches,
    )
    results: dict[str, dict] = {}
    for mode in ("closed_loop", "random", "no_action"):
        mode_root = ensure_dir(output_root / mode)
        runtime_config = RuntimeConfig(
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            lookahead_horizon=args.lookahead_horizon,
            lookahead_discount=args.lookahead_discount,
            viability_mode=args.viability_mode,
            action_mode=args.action_mode,
            use_trm_a=bool(args.trm_a_checkpoint),
            use_trm_b=bool(args.trm_b_checkpoint),
            policy_mode=mode,
        )
        episode_path = run_episode(
            mode_root,
            args.seed_catalog,
            runtime_config,
            env_config,
            trm_a_checkpoint=args.trm_a_checkpoint,
            trm_b_checkpoint=args.trm_b_checkpoint,
            module_manifest=args.module_manifest,
        )
        summary = load_json(_summary_path_from_episode(episode_path))
        results[mode] = summary

    comparison = {
        "seed": args.seed,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "trm_a_checkpoint": args.trm_a_checkpoint,
        "trm_b_checkpoint": args.trm_b_checkpoint,
        "module_manifest": args.module_manifest,
        "lookahead_horizon": args.lookahead_horizon,
        "lookahead_discount": args.lookahead_discount,
        "viability_mode": args.viability_mode,
        "action_mode": args.action_mode,
        "results": results,
        "derived": _derived_comparison(results),
    }
    save_json(output_root / "comparison_summary.json", comparison)
    print(f"wrote comparison: {output_root / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
