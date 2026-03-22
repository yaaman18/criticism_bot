from __future__ import annotations

import argparse
from pathlib import Path

from .common import ensure_dir, load_json, save_json
from .erie_runtime import EnvironmentConfig, RuntimeConfig, run_episode

MODE_VALUES = ("analytic", "assistive", "module_primary")


def _summary_path_from_episode(path: str | Path) -> Path:
    episode_path = Path(path)
    return episode_path.with_name(f"{episode_path.stem}_summary.json")


def _mode_key(viability_mode: str, action_mode: str) -> str:
    return f"{viability_mode}__{action_mode}"


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
        description="Compare TRM-Vm/TRM-As integration modes under a fixed ERIE policy mode."
    )
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--output-root", default="artifacts/trm_va_mode_compare")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260321)
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
    env_config = EnvironmentConfig(
        resource_patches=args.resource_patches,
        hazard_patches=args.hazard_patches,
        shelter_patches=args.shelter_patches,
    )
    results: dict[str, dict] = {}
    for viability_mode in MODE_VALUES:
        for action_mode in MODE_VALUES:
            mode_key = _mode_key(viability_mode, action_mode)
            mode_root = ensure_dir(output_root / mode_key)
            runtime_config = RuntimeConfig(
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
                lookahead_horizon=args.lookahead_horizon,
                lookahead_discount=args.lookahead_discount,
                viability_mode=viability_mode,
                action_mode=action_mode,
                use_trm_a=bool(args.trm_a_checkpoint),
                use_trm_b=bool(args.trm_b_checkpoint),
                policy_mode=args.policy_mode,
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
            summary["mode_key"] = mode_key
            results[mode_key] = summary

    comparison = {
        "seed": args.seed,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "policy_mode": args.policy_mode,
        "trm_a_checkpoint": args.trm_a_checkpoint,
        "trm_b_checkpoint": args.trm_b_checkpoint,
        "module_manifest": args.module_manifest,
        "lookahead_horizon": args.lookahead_horizon,
        "lookahead_discount": args.lookahead_discount,
        "results": results,
        "derived": _derived_comparison(results),
    }
    save_json(output_root / "comparison_summary.json", comparison)
    print(f"wrote TRM-Vm/TRM-As comparison: {output_root / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
