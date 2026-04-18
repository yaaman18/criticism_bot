from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .common import ensure_dir, load_json, save_json
from .compare_trm_mc_modes import _mode_key
from .erie_runtime import EnvironmentConfig, RuntimeConfig, run_episode
from .prepare_trm_va_data import EPISODE_FAMILIES, sample_episode_configs_for_family

ACTION_MODE_VALUES = ("analytic", "assistive", "module_primary")
CONTEXT_MODE_VALUES = ("analytic", "assistive")


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


def _context_gain(results: dict[str, dict], action_mode: str, metric: str) -> float | None:
    analytic_key = _mode_key(action_mode, "analytic")
    assistive_key = _mode_key(action_mode, "assistive")
    if analytic_key not in results or assistive_key not in results:
        return None
    analytic_value = results[analytic_key].get(metric)
    assistive_value = results[assistive_key].get(metric)
    if analytic_value is None or assistive_value is None:
        return None
    return float(analytic_value) - float(assistive_value)


def _derived_comparison_local(results: dict[str, dict]) -> dict[str, object]:
    final_ranking = _ranking_by_metric(results, "final_homeostatic_error")
    mean_ranking = _ranking_by_metric(results, "mean_homeostatic_error")
    return {
        "best_mode_by_final_homeostasis": final_ranking[0] if final_ranking else None,
        "best_mode_by_mean_homeostasis": mean_ranking[0] if mean_ranking else None,
        "ranking_by_final_homeostasis": final_ranking,
        "ranking_by_mean_homeostasis": mean_ranking,
        "context_gain_by_action_mode": {
            action_mode: {
                "final_homeostatic_error_delta": _context_gain(results, action_mode, "final_homeostatic_error"),
                "mean_homeostatic_error_delta": _context_gain(results, action_mode, "mean_homeostatic_error"),
            }
            for action_mode in ACTION_MODE_VALUES
        },
    }


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
    viability_mode: str,
    boundary_control_mode: str,
    context_memory_window_size: int,
    context_memory_residual_scale: float,
    episode_family: str | None,
    defensive_family_bias: float,
) -> dict:
    output_root = ensure_dir(output_root)
    base_env_config = EnvironmentConfig(
        resource_patches=resource_patches,
        hazard_patches=hazard_patches,
        shelter_patches=shelter_patches,
    )
    base_runtime_config = RuntimeConfig(
        steps=steps,
        warmup_steps=warmup_steps,
        seed=seed,
        lookahead_horizon=lookahead_horizon,
        lookahead_discount=lookahead_discount,
        viability_mode=viability_mode,
        boundary_control_mode=boundary_control_mode,
        context_memory_window_size=context_memory_window_size,
        context_memory_residual_scale=context_memory_residual_scale,
        use_trm_a=bool(trm_a_checkpoint),
        use_trm_b=bool(trm_b_checkpoint),
        policy_mode=policy_mode,
    )
    if episode_family is not None:
        base_runtime_config, base_env_config = sample_episode_configs_for_family(
            family=episode_family,
            base_runtime_config=base_runtime_config,
            base_env_config=base_env_config,
            seed=seed,
            defensive_family_bias=defensive_family_bias,
        )

    results: dict[str, dict] = {}
    for action_mode in ACTION_MODE_VALUES:
        for context_memory_mode in CONTEXT_MODE_VALUES:
            mode_key = _mode_key(action_mode, context_memory_mode)
            mode_root = ensure_dir(Path(output_root) / mode_key)
            runtime_config = RuntimeConfig(
                **{
                    **base_runtime_config.__dict__,
                    "action_mode": action_mode,
                    "context_memory_mode": context_memory_mode,
                }
            )
            episode_path = run_episode(
                mode_root,
                seed_catalog,
                runtime_config,
                base_env_config,
                trm_a_checkpoint=trm_a_checkpoint,
                trm_b_checkpoint=trm_b_checkpoint,
                module_manifest=module_manifest,
            )
            summary = load_json(_summary_path_from_episode(episode_path))
            summary["mode_key"] = mode_key
            results[mode_key] = summary

    comparison = {
        "seed": seed,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "policy_mode": policy_mode,
        "viability_mode": viability_mode,
        "boundary_control_mode": boundary_control_mode,
        "module_manifest": module_manifest,
        "episode_family": episode_family,
        "defensive_family_bias": defensive_family_bias,
        "results": results,
        "derived": _derived_comparison_local(results),
    }
    save_json(Path(output_root) / "comparison_summary.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep TRM-Mc assistive vs analytic context-memory modes across seeds."
    )
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--output-root", default="artifacts/trm_mc_mode_sweep")
    parser.add_argument("--seed-start", type=int, default=20260406)
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
    parser.add_argument(
        "--viability-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument(
        "--boundary-control-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument("--context-memory-window-size", type=int, default=8)
    parser.add_argument("--context-memory-residual-scale", type=float, default=0.35)
    parser.add_argument("--episode-family", choices=EPISODE_FAMILIES, default=None)
    parser.add_argument("--defensive-family-bias", type=float, default=0.0)
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    per_seed = []
    final_counter: Counter[str] = Counter()
    mean_counter: Counter[str] = Counter()
    context_gain_final: dict[str, list[float]] = {mode: [] for mode in ACTION_MODE_VALUES}
    context_gain_mean: dict[str, list[float]] = {mode: [] for mode in ACTION_MODE_VALUES}

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
            viability_mode=args.viability_mode,
            boundary_control_mode=args.boundary_control_mode,
            context_memory_window_size=args.context_memory_window_size,
            context_memory_residual_scale=args.context_memory_residual_scale,
            episode_family=args.episode_family,
            defensive_family_bias=args.defensive_family_bias,
        )
        per_seed_entry = {
            "seed": seed,
            "best_mode_by_final_homeostasis": comparison["derived"]["best_mode_by_final_homeostasis"],
            "best_mode_by_mean_homeostasis": comparison["derived"]["best_mode_by_mean_homeostasis"],
            "context_gain_by_action_mode": comparison["derived"]["context_gain_by_action_mode"],
        }
        per_seed.append(per_seed_entry)
        final_counter.update([comparison["derived"]["best_mode_by_final_homeostasis"]])
        mean_counter.update([comparison["derived"]["best_mode_by_mean_homeostasis"]])
        for action_mode in ACTION_MODE_VALUES:
            final_delta = comparison["derived"]["context_gain_by_action_mode"][action_mode][
                "final_homeostatic_error_delta"
            ]
            mean_delta = comparison["derived"]["context_gain_by_action_mode"][action_mode][
                "mean_homeostatic_error_delta"
            ]
            if final_delta is not None:
                context_gain_final[action_mode].append(float(final_delta))
            if mean_delta is not None:
                context_gain_mean[action_mode].append(float(mean_delta))

    aggregate = {
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "policy_mode": args.policy_mode,
        "viability_mode": args.viability_mode,
        "boundary_control_mode": args.boundary_control_mode,
        "module_manifest": args.module_manifest,
        "episode_family": args.episode_family,
        "defensive_family_bias": args.defensive_family_bias,
        "counts_by_best_final_homeostasis": dict(final_counter),
        "counts_by_best_mean_homeostasis": dict(mean_counter),
        "mean_context_gain_by_action_mode": {
            action_mode: {
                "final_homeostatic_error_delta": (
                    sum(context_gain_final[action_mode]) / len(context_gain_final[action_mode])
                    if context_gain_final[action_mode]
                    else None
                ),
                "mean_homeostatic_error_delta": (
                    sum(context_gain_mean[action_mode]) / len(context_gain_mean[action_mode])
                    if context_gain_mean[action_mode]
                    else None
                ),
            }
            for action_mode in ACTION_MODE_VALUES
        },
        "per_seed": per_seed,
    }
    save_json(output_root / "aggregate_summary.json", aggregate)
    print(f"wrote TRM-Mc sweep: {output_root / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
