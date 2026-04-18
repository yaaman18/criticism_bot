from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .common import ensure_dir, load_json, save_json
from .prepare_trm_va_data import EPISODE_FAMILIES
from .sweep_trm_mc_modes import _compare_one_seed

DEFAULT_FAMILIES = ("fragile_boundary", "vent_edge", "uncertain_corridor")
ACTION_MODE_VALUES = ("analytic", "assistive", "module_primary")


def _family_gain_means(per_seed: list[dict]) -> dict[str, dict[str, float | None]]:
    final_values: dict[str, list[float]] = {mode: [] for mode in ACTION_MODE_VALUES}
    mean_values: dict[str, list[float]] = {mode: [] for mode in ACTION_MODE_VALUES}
    for row in per_seed:
        gains = row.get("context_gain_by_action_mode", {})
        for action_mode in ACTION_MODE_VALUES:
            final_delta = gains.get(action_mode, {}).get("final_homeostatic_error_delta")
            mean_delta = gains.get(action_mode, {}).get("mean_homeostatic_error_delta")
            if final_delta is not None:
                final_values[action_mode].append(float(final_delta))
            if mean_delta is not None:
                mean_values[action_mode].append(float(mean_delta))
    return {
        action_mode: {
            "final_homeostatic_error_delta": (
                sum(final_values[action_mode]) / len(final_values[action_mode]) if final_values[action_mode] else None
            ),
            "mean_homeostatic_error_delta": (
                sum(mean_values[action_mode]) / len(mean_values[action_mode]) if mean_values[action_mode] else None
            ),
        }
        for action_mode in ACTION_MODE_VALUES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep TRM-Mc context-memory balance across multiple episode families."
    )
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--output-root", default="artifacts/trm_mc_family_balance")
    parser.add_argument("--seed-start", type=int, default=20260406)
    parser.add_argument("--num-seeds", type=int, default=3)
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
    parser.add_argument(
        "--families",
        nargs="+",
        choices=EPISODE_FAMILIES,
        default=list(DEFAULT_FAMILIES),
    )
    parser.add_argument("--defensive-family-bias", type=float, default=2.0)
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    family_summaries: dict[str, dict] = {}
    balanced_context_final: dict[str, list[float]] = {mode: [] for mode in ACTION_MODE_VALUES}
    balanced_context_mean: dict[str, list[float]] = {mode: [] for mode in ACTION_MODE_VALUES}
    final_counter: Counter[str] = Counter()
    mean_counter: Counter[str] = Counter()

    for family in args.families:
        family_root = ensure_dir(output_root / family)
        per_seed = []
        family_final_counter: Counter[str] = Counter()
        family_mean_counter: Counter[str] = Counter()
        for offset in range(args.num_seeds):
            seed = args.seed_start + offset
            seed_root = ensure_dir(family_root / f"seed_{seed}")
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
                episode_family=family,
                defensive_family_bias=args.defensive_family_bias,
            )
            per_seed_entry = {
                "seed": seed,
                "best_mode_by_final_homeostasis": comparison["derived"]["best_mode_by_final_homeostasis"],
                "best_mode_by_mean_homeostasis": comparison["derived"]["best_mode_by_mean_homeostasis"],
                "context_gain_by_action_mode": comparison["derived"]["context_gain_by_action_mode"],
            }
            per_seed.append(per_seed_entry)
            family_final_counter.update([comparison["derived"]["best_mode_by_final_homeostasis"]])
            family_mean_counter.update([comparison["derived"]["best_mode_by_mean_homeostasis"]])

        family_summary = {
            "family": family,
            "counts_by_best_final_homeostasis": dict(family_final_counter),
            "counts_by_best_mean_homeostasis": dict(family_mean_counter),
            "mean_context_gain_by_action_mode": _family_gain_means(per_seed),
            "per_seed": per_seed,
        }
        family_summaries[family] = family_summary
        save_json(family_root / "family_summary.json", family_summary)

        final_counter.update(family_final_counter)
        mean_counter.update(family_mean_counter)
        for action_mode in ACTION_MODE_VALUES:
            family_final = family_summary["mean_context_gain_by_action_mode"][action_mode]["final_homeostatic_error_delta"]
            family_mean = family_summary["mean_context_gain_by_action_mode"][action_mode]["mean_homeostatic_error_delta"]
            if family_final is not None:
                balanced_context_final[action_mode].append(float(family_final))
            if family_mean is not None:
                balanced_context_mean[action_mode].append(float(family_mean))

    aggregate = {
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "families": args.families,
        "policy_mode": args.policy_mode,
        "viability_mode": args.viability_mode,
        "boundary_control_mode": args.boundary_control_mode,
        "module_manifest": args.module_manifest,
        "defensive_family_bias": args.defensive_family_bias,
        "counts_by_best_final_homeostasis": dict(final_counter),
        "counts_by_best_mean_homeostasis": dict(mean_counter),
        "balanced_mean_context_gain_by_action_mode": {
            action_mode: {
                "final_homeostatic_error_delta": (
                    sum(balanced_context_final[action_mode]) / len(balanced_context_final[action_mode])
                    if balanced_context_final[action_mode]
                    else None
                ),
                "mean_homeostatic_error_delta": (
                    sum(balanced_context_mean[action_mode]) / len(balanced_context_mean[action_mode])
                    if balanced_context_mean[action_mode]
                    else None
                ),
            }
            for action_mode in ACTION_MODE_VALUES
        },
        "family_summaries": {
            family: {
                "counts_by_best_final_homeostasis": family_summaries[family]["counts_by_best_final_homeostasis"],
                "counts_by_best_mean_homeostasis": family_summaries[family]["counts_by_best_mean_homeostasis"],
                "mean_context_gain_by_action_mode": family_summaries[family]["mean_context_gain_by_action_mode"],
            }
            for family in args.families
        },
    }
    save_json(output_root / "aggregate_summary.json", aggregate)
    print(f"wrote TRM-Mc family balance: {output_root / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
