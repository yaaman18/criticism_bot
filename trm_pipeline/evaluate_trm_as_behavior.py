from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import load_json, save_json


EXPLOIT_ACTIONS = {"intake", "seal"}
DEFENSIVE_ACTIONS = {"withdraw", "reconfigure"}
NAVIGATION_ACTIONS = {"approach", "withdraw", "reconfigure"}
ACTION_VOCAB = ("approach", "withdraw", "intake", "seal", "reconfigure", "no_action")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _entropy(probs: list[float]) -> float:
    total = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        total -= float(p) * math.log(float(p))
    return total


def _action_profile(history: list[dict]) -> dict[str, float]:
    if not history:
        return {
            "action_diversity": 0.0,
            "intake_rate": 0.0,
            "navigation_rate": 0.0,
        }
    actions = [str(row.get("action", "no_action")) for row in history]
    total = max(len(actions), 1)
    counts = [actions.count(name) for name in ACTION_VOCAB]
    probs = [float(c) / float(total) for c in counts]
    denom = math.log(len(ACTION_VOCAB))
    action_diversity = (_entropy(probs) / denom) if denom > 0.0 else 0.0
    intake_rate = float(sum(a == "intake" for a in actions) / total)
    navigation_rate = float(sum(a in NAVIGATION_ACTIONS for a in actions) / total)
    return {
        "action_diversity": float(action_diversity),
        "intake_rate": float(intake_rate),
        "navigation_rate": float(navigation_rate),
    }


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
            "action_diversity": 0.0,
            "intake_rate": 0.0,
            "navigation_rate": 0.0,
            "mean_spawn_drive": 0.0,
            "mean_split_drive": 0.0,
            "spawn_candidate_rate": 0.0,
            "split_candidate_rate": 0.0,
            "trace_ablation_spawn_delta": 0.0,
            "mean_p_t": 0.5,
            "mean_challenge_fraction": 0.5,
            "role_switch_events_total": 0.0,
            "mean_aux_policy_entropy": 0.0,
            "aux_full_policy_rate": 0.0,
            "mean_aux_nontrivial_action_count": 0.0,
        }
    action_profile = _action_profile(history)
    g_overshoot = [max(0.0, float(row["G"]) - g_target) for row in history]
    b_undershoot = [max(0.0, b_target - float(row["B"])) for row in history]
    stress_load = [
        0.5 * (float(row.get("contact_thermal", 0.0)) + float(row.get("contact_toxicity", 0.0))) for row in history
    ]
    stress_indices = [i for i, val in enumerate(stress_load) if val >= stress_threshold]
    stress_actions = [str(history[i]["action"]) for i in stress_indices]
    exploit_count = sum(action in EXPLOIT_ACTIONS for action in stress_actions)
    defensive_count = sum(action in DEFENSIVE_ACTIONS for action in stress_actions)
    spawn_drive = [float(row.get("spawn_drive", 0.0)) for row in history]
    split_drive = [float(row.get("split_drive", 0.0)) for row in history]
    spawn_candidate_rate = float(
        sum(1 for row in history if bool(row.get("spawn_candidate", False))) / max(len(history), 1)
    )
    split_candidate_rate = float(
        sum(1 for row in history if bool(row.get("split_candidate", False))) / max(len(history), 1)
    )
    trace_ablation_spawn_delta = [float(row.get("trace_ablation_spawn_delta", 0.0)) for row in history]
    p_t = [float(row.get("p_t", 0.5)) for row in history]
    challenge_fraction = [
        float(row.get("challenge_body_count", 0.0)) / max(float(row.get("body_count", 1.0)), 1.0)
        for row in history
    ]
    role_switch_events_total = float(sum(float(row.get("role_switch_events_step", 0.0)) for row in history))
    aux_mean_policy_entropy = [float(row.get("aux_mean_policy_entropy", 0.0)) for row in history]
    aux_full_policy_count = float(
        sum(float(row.get("aux_policy_source_counts", {}).get("full_policy", 0.0)) for row in history)
    )
    aux_role_heuristic_count = float(
        sum(float(row.get("aux_policy_source_counts", {}).get("role_heuristic", 0.0)) for row in history)
    )
    aux_passive_count = float(
        sum(float(row.get("aux_policy_source_counts", {}).get("passive", 0.0)) for row in history)
    )
    aux_policy_total = max(aux_full_policy_count + aux_role_heuristic_count + aux_passive_count, 1.0)
    aux_nontrivial_action_count = [float(row.get("aux_nontrivial_action_count", 0.0)) for row in history]
    return {
        "num_steps": int(len(history)),
        "mean_G_overshoot": _mean(g_overshoot),
        "mean_B_undershoot": _mean(b_undershoot),
        "mean_stress_load": _mean(stress_load),
        "stress_step_fraction": float(len(stress_indices) / max(len(history), 1)),
        "stress_exploit_rate": float(exploit_count / max(len(stress_indices), 1)),
        "stress_defensive_rate": float(defensive_count / max(len(stress_indices), 1)),
        "action_diversity": float(action_profile["action_diversity"]),
        "intake_rate": float(action_profile["intake_rate"]),
        "navigation_rate": float(action_profile["navigation_rate"]),
        "mean_spawn_drive": _mean(spawn_drive),
        "mean_split_drive": _mean(split_drive),
        "spawn_candidate_rate": spawn_candidate_rate,
        "split_candidate_rate": split_candidate_rate,
        "trace_ablation_spawn_delta": _mean(trace_ablation_spawn_delta),
        "mean_p_t": _mean(p_t),
        "mean_challenge_fraction": _mean(challenge_fraction),
        "role_switch_events_total": role_switch_events_total,
        "mean_aux_policy_entropy": _mean(aux_mean_policy_entropy),
        "aux_full_policy_rate": float(aux_full_policy_count / aux_policy_total),
        "mean_aux_nontrivial_action_count": _mean(aux_nontrivial_action_count),
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
        mode_diag = evaluate_history(
            history_files[0],
            g_target=float(cfg.get("G_target", 0.55)),
            b_target=float(cfg.get("B_target", 0.65)),
            stress_threshold=stress_threshold,
        )
        # Runtime summary carries aggregate metrics we can reuse when history
        # format changes or ablation diagnostics are absent in older traces.
        mode_diag["trace_ablation_spawn_delta"] = float(
            mode_summary.get("mean_trace_ablation_spawn_delta", mode_diag.get("trace_ablation_spawn_delta", 0.0))
        )
        mode_diag["mean_spawn_drive"] = float(mode_summary.get("mean_spawn_drive", mode_diag.get("mean_spawn_drive", 0.0)))
        mode_diag["mean_split_drive"] = float(mode_summary.get("mean_split_drive", mode_diag.get("mean_split_drive", 0.0)))
        mode_diag["spawn_candidate_rate"] = float(
            mode_summary.get("spawn_candidate_rate", mode_diag.get("spawn_candidate_rate", 0.0))
        )
        mode_diag["split_candidate_rate"] = float(
            mode_summary.get("split_candidate_rate", mode_diag.get("split_candidate_rate", 0.0))
        )
        mode_diag["mean_p_t"] = float(mode_summary.get("mean_p_t", mode_diag.get("mean_p_t", 0.5)))
        mode_diag["mean_challenge_fraction"] = float(
            mode_summary.get("mean_challenge_fraction", mode_diag.get("mean_challenge_fraction", 0.5))
        )
        mode_diag["role_switch_events_total"] = float(
            mode_summary.get("role_switch_events_total", mode_diag.get("role_switch_events_total", 0.0))
        )
        mode_diag["mean_aux_policy_entropy"] = float(
            mode_summary.get("mean_aux_policy_entropy", mode_diag.get("mean_aux_policy_entropy", 0.0))
        )
        mode_diag["aux_full_policy_rate"] = float(
            mode_summary.get("aux_full_policy_rate", mode_diag.get("aux_full_policy_rate", 0.0))
        )
        mode_diag["mean_aux_nontrivial_action_count"] = float(
            mode_summary.get(
                "mean_aux_nontrivial_action_count",
                mode_diag.get("mean_aux_nontrivial_action_count", 0.0),
            )
        )
        diagnostics[mode] = mode_diag
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
