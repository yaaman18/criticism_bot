from __future__ import annotations

import argparse
from pathlib import Path

from .common import ensure_dir, load_json, save_json
from .erie_runtime import EnvironmentConfig, RuntimeConfig, run_episode
from .prepare_trm_va_data import EPISODE_FAMILIES, sample_episode_configs_for_family

ACTION_GATING_MODE_VALUES = ("analytic", "assistive", "module_primary")


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


def _ag_gain(results: dict[str, dict], metric: str) -> float | None:
    analytic = results.get("analytic", {}).get(metric)
    assistive = results.get("assistive", {}).get(metric)
    if analytic is None or assistive is None:
        return None
    return float(analytic) - float(assistive)


def _derived_comparison(results: dict[str, dict]) -> dict[str, object]:
    final_ranking = _ranking_by_metric(results, "final_homeostatic_error")
    mean_ranking = _ranking_by_metric(results, "mean_homeostatic_error")
    return {
        "best_mode_by_final_homeostasis": final_ranking[0] if final_ranking else None,
        "best_mode_by_mean_homeostasis": mean_ranking[0] if mean_ranking else None,
        "ranking_by_final_homeostasis": final_ranking,
        "ranking_by_mean_homeostasis": mean_ranking,
        "ag_gain": {
            "final_homeostatic_error_delta": _ag_gain(results, "final_homeostatic_error"),
            "mean_homeostatic_error_delta": _ag_gain(results, "mean_homeostatic_error"),
            "stress_exploit_rate_delta": _ag_gain(results, "stress_exploit_rate"),
            "dominant_action_fraction_delta": _ag_gain(results, "dominant_action_fraction"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TRM-Ag gating modes on top of As/Mc/Bp.")
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--output-root", default="artifacts/trm_ag_mode_compare")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--lookahead-horizon", type=int, default=2)
    parser.add_argument("--lookahead-discount", type=float, default=0.85)
    parser.add_argument("--resource-patches", type=int, default=3)
    parser.add_argument("--hazard-patches", type=int, default=3)
    parser.add_argument("--shelter-patches", type=int, default=1)
    parser.add_argument("--trm-a-checkpoint", default=None)
    parser.add_argument("--trm-b-checkpoint", default=None)
    parser.add_argument("--module-manifest", default=None)
    parser.add_argument("--policy-mode", choices=("closed_loop", "random", "no_action"), default="closed_loop")
    parser.add_argument("--viability-mode", choices=("analytic", "assistive", "module_primary"), default="assistive")
    parser.add_argument("--action-mode", choices=("analytic", "assistive", "module_primary"), default="assistive")
    parser.add_argument("--boundary-control-mode", choices=("analytic", "assistive", "module_primary"), default="assistive")
    parser.add_argument("--context-memory-mode", choices=("analytic", "assistive"), default="assistive")
    parser.add_argument("--episode-family", choices=EPISODE_FAMILIES, default=None)
    parser.add_argument("--defensive-family-bias", type=float, default=0.0)
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    base_env_config = EnvironmentConfig(
        resource_patches=args.resource_patches,
        hazard_patches=args.hazard_patches,
        shelter_patches=args.shelter_patches,
    )
    base_runtime_config = RuntimeConfig(
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        lookahead_horizon=args.lookahead_horizon,
        lookahead_discount=args.lookahead_discount,
        viability_mode=args.viability_mode,
        action_mode=args.action_mode,
        boundary_control_mode=args.boundary_control_mode,
        context_memory_mode=args.context_memory_mode,
        use_trm_a=bool(args.trm_a_checkpoint),
        use_trm_b=bool(args.trm_b_checkpoint),
        policy_mode=args.policy_mode,
    )
    if args.episode_family is not None:
        base_runtime_config, base_env_config = sample_episode_configs_for_family(
            family=args.episode_family,
            base_runtime_config=base_runtime_config,
            base_env_config=base_env_config,
            seed=args.seed,
            defensive_family_bias=args.defensive_family_bias,
        )
    results: dict[str, dict] = {}
    for action_gating_mode in ACTION_GATING_MODE_VALUES:
        mode_root = ensure_dir(output_root / action_gating_mode)
        runtime_config = RuntimeConfig(
            **{
                **base_runtime_config.__dict__,
                "action_gating_mode": action_gating_mode,
            }
        )
        episode_path = run_episode(
            mode_root,
            args.seed_catalog,
            runtime_config,
            base_env_config,
            trm_a_checkpoint=args.trm_a_checkpoint,
            trm_b_checkpoint=args.trm_b_checkpoint,
            module_manifest=args.module_manifest,
        )
        summary = load_json(_summary_path_from_episode(episode_path))
        summary["action_gating_mode"] = action_gating_mode
        results[action_gating_mode] = summary

    comparison = {
        "seed": args.seed,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "policy_mode": args.policy_mode,
        "viability_mode": args.viability_mode,
        "action_mode": args.action_mode,
        "boundary_control_mode": args.boundary_control_mode,
        "context_memory_mode": args.context_memory_mode,
        "trm_a_checkpoint": args.trm_a_checkpoint,
        "trm_b_checkpoint": args.trm_b_checkpoint,
        "module_manifest": args.module_manifest,
        "lookahead_horizon": args.lookahead_horizon,
        "lookahead_discount": args.lookahead_discount,
        "episode_family": args.episode_family,
        "defensive_family_bias": args.defensive_family_bias,
        "results": results,
        "derived": _derived_comparison(results),
    }
    save_json(output_root / "comparison_summary.json", comparison)
    print(f"wrote TRM-Ag comparison: {output_root / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
