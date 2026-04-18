from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_json, save_json


EXPLOIT_ACTIONS = {"intake", "seal"}
DEFENSIVE_ACTIONS = {"withdraw", "reconfigure"}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate_history(
    history_path: str | Path,
    *,
    g_target: float = 0.55,
    b_target: float = 0.65,
    stress_threshold: float = 0.35,
) -> dict[str, float]:
    history = load_json(history_path)
    if not history:
        return {
            "num_steps": 0,
            "mean_G_overshoot": 0.0,
            "mean_B_undershoot": 0.0,
            "mean_stress_load": 0.0,
            "stress_step_fraction": 0.0,
            "stress_exploit_rate": 0.0,
            "stress_defensive_rate": 0.0,
        }
    g_overshoot = [max(0.0, float(row["G"]) - g_target) for row in history]
    b_undershoot = [max(0.0, b_target - float(row["B"])) for row in history]
    stress_load = [
        0.5 * (float(row.get("contact_thermal", 0.0)) + float(row.get("contact_toxicity", 0.0))) for row in history
    ]
    stress_indices = [i for i, val in enumerate(stress_load) if val >= stress_threshold]
    stress_actions = [str(history[i]["action"]) for i in stress_indices]
    exploit_count = sum(action in EXPLOIT_ACTIONS for action in stress_actions)
    defensive_count = sum(action in DEFENSIVE_ACTIONS for action in stress_actions)
    return {
        "num_steps": int(len(history)),
        "mean_G_overshoot": _mean(g_overshoot),
        "mean_B_undershoot": _mean(b_undershoot),
        "mean_stress_load": _mean(stress_load),
        "stress_step_fraction": float(len(stress_indices) / max(len(history), 1)),
        "stress_exploit_rate": float(exploit_count / max(len(stress_indices), 1)),
        "stress_defensive_rate": float(defensive_count / max(len(stress_indices), 1)),
    }


def evaluate_compare_root(compare_root: str | Path, stress_threshold: float = 0.35) -> dict[str, object]:
    compare_root = Path(compare_root)
    summary = load_json(compare_root / "comparison_summary.json")
    diagnostics = {}
    for mode in summary["results"].keys():
        history_files = sorted((compare_root / mode).glob("*_history.json"))
        if not history_files:
            continue
        mode_summary = summary["results"][mode]
        cfg = mode_summary.get("runtime_config", {})
        diagnostics[mode] = evaluate_history(
            history_files[0],
            g_target=float(cfg.get("G_target", 0.55)),
            b_target=float(cfg.get("B_target", 0.65)),
            stress_threshold=stress_threshold,
        )
    result = {
        "compare_root": str(compare_root),
        "stress_threshold": float(stress_threshold),
        "best_mode_by_final_homeostasis": summary["derived"]["best_mode_by_final_homeostasis"],
        "best_mode_by_mean_homeostasis": summary["derived"]["best_mode_by_mean_homeostasis"],
        "mode_diagnostics": diagnostics,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TRM-As behavior diagnostics from a compare_trm_va_modes output.")
    parser.add_argument("--compare-root", required=True)
    parser.add_argument("--stress-threshold", type=float, default=0.35)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = evaluate_compare_root(args.compare_root, stress_threshold=args.stress_threshold)
    if args.output:
        save_json(args.output, result)
    else:
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
