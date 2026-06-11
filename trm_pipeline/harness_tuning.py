from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .common import ensure_dir, load_json, save_json
from .harness_contracts import (
    TUNABLE_RUNTIME_PARAMS,
    artifact_paths_for_output_root,
    clone_contract,
    coerce_contract,
)


RunContractFn = Callable[..., dict[str, Any]]


def required_track_names_from_eval(eval_report: dict[str, Any]) -> list[str]:
    required = [str(name) for name in eval_report.get("required_family_tracks", [])]
    if required:
        return required
    family_reports = dict(eval_report.get("family_reports", {}))
    return [
        str(name)
        for name, report in family_reports.items()
        if bool(report.get("required_for_promotion", False))
    ]


def primary_score_from_eval_report(eval_report: dict[str, Any]) -> float:
    required_tracks = required_track_names_from_eval(eval_report)
    family_reports = dict(eval_report.get("family_reports", {}))
    values: list[float] = []
    for track_name in required_tracks:
        candidate = family_reports.get(track_name, {}).get("summary", {}).get("candidate", {})
        raw_value = candidate.get("mean_final_homeostatic_error")
        if raw_value is None:
            continue
        value = float(raw_value)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("inf")
    return float(sum(values) / len(values))


def failed_criteria_counter(eval_report: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    required_tracks = set(required_track_names_from_eval(eval_report))
    blocked_tracks = [
        str(name)
        for name in eval_report.get("blocked_family_tracks", [])
        if str(name) in required_tracks
    ]
    target_tracks = blocked_tracks or list(required_tracks)
    family_reports = dict(eval_report.get("family_reports", {}))
    for track_name in target_tracks:
        criteria = dict(family_reports.get(track_name, {}).get("criteria", {}))
        for criterion_name, payload in criteria.items():
            if not bool(payload.get("passed", False)):
                counts[str(criterion_name)] += 1
    return counts


def propose_tuning_updates(
    eval_report: dict[str, Any],
    *,
    max_updates_per_round: int = 3,
) -> list[dict[str, Any]]:
    failed = failed_criteria_counter(eval_report)
    if not failed:
        return []
    priority_rules: list[tuple[str, list[tuple[str, float]]]] = [
        ("dead_fraction", [("move_step", -0.20), ("aperture_gain", -0.03)]),
        ("stress_exploit_rate", [("aperture_gain", -0.04), ("action_gating_blend", 0.05)]),
        ("stress_defensive_rate", [("aperture_width_deg", -4.0), ("action_gating_blend", 0.05)]),
        ("intake_rate", [("aperture_gain", -0.04), ("action_gating_blend", 0.05)]),
        ("navigation_rate", [("move_step", 0.20), ("aperture_gain", -0.02)]),
        ("trace_ablation_spawn_delta", [("move_step", 0.20), ("aperture_width_deg", 4.0)]),
        ("mean_p_t_min", [("move_step", 0.20), ("aperture_gain", -0.02)]),
        ("mean_p_t_max", [("move_step", -0.20), ("aperture_gain", -0.03)]),
        ("mean_challenge_fraction_min", [("move_step", 0.20), ("lookahead_horizon", 1.0)]),
        ("mean_challenge_fraction_max", [("move_step", -0.20), ("lookahead_discount", 0.02)]),
        ("role_switch_events_total", [("move_step", 0.20), ("lookahead_horizon", 1.0)]),
        ("mean_aux_nontrivial_action_count", [("move_step", 0.20), ("aperture_width_deg", 4.0)]),
        ("action_diversity", [("move_step", 0.20), ("lookahead_horizon", 1.0)]),
        ("mean_final_homeostatic_error", [("lookahead_horizon", 1.0), ("lookahead_discount", 0.03)]),
        ("mean_mean_homeostatic_error", [("lookahead_discount", 0.02)]),
        ("best_mode_frequency", [("lookahead_horizon", 1.0)]),
    ]
    proposals: list[dict[str, Any]] = []
    used_params: set[str] = set()
    for criterion_name, param_deltas in priority_rules:
        if failed.get(criterion_name, 0) <= 0:
            continue
        for param_name, delta in param_deltas:
            if param_name in used_params:
                continue
            proposals.append(
                {
                    "criterion": criterion_name,
                    "param": param_name,
                    "delta": float(delta),
                    "failed_track_count": int(failed[criterion_name]),
                }
            )
            used_params.add(param_name)
            if len(proposals) >= max_updates_per_round:
                return proposals
    return proposals


def clamp_tunable_value(param_name: str, value: float | int) -> float | int:
    spec = TUNABLE_RUNTIME_PARAMS[param_name]
    lower = float(spec["min"])
    upper = float(spec["max"])
    clamped = min(max(float(value), lower), upper)
    if spec["kind"] == "int":
        return int(round(clamped))
    return float(clamped)


def default_tunable_value(param_name: str) -> float | int:
    spec = TUNABLE_RUNTIME_PARAMS[param_name]
    if spec["kind"] == "int":
        return int(spec["default"])
    return float(spec["default"])


def target_runtime_overrides(
    contract: dict[str, Any],
    *,
    blocked_tracks: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    tracks = contract.get("family_tracks")
    if isinstance(tracks, list) and tracks:
        blocked_set = {str(name) for name in blocked_tracks}
        if blocked_set:
            target_tracks = blocked_set
        else:
            target_tracks = {
                str(track.get("name"))
                for track in tracks
                if bool(track.get("required_for_promotion", True))
            }
        targets: list[tuple[str, dict[str, Any]]] = []
        for track in tracks:
            track_name = str(track.get("name"))
            if track_name not in target_tracks:
                continue
            overrides = track.setdefault("runtime_overrides", {})
            targets.append((track_name, overrides))
        return targets
    runtime = contract.setdefault("runtime", {})
    return [("global", runtime)]


def apply_tuning_updates(
    contract: dict[str, Any],
    proposals: list[dict[str, Any]],
    *,
    blocked_tracks: list[str],
) -> list[dict[str, Any]]:
    targets = target_runtime_overrides(contract, blocked_tracks=blocked_tracks)
    applied: list[dict[str, Any]] = []
    for track_name, runtime_overrides in targets:
        for proposal in proposals:
            param_name = str(proposal["param"])
            if param_name not in TUNABLE_RUNTIME_PARAMS:
                continue
            raw_before = runtime_overrides.get(param_name, default_tunable_value(param_name))
            before = clamp_tunable_value(param_name, raw_before)
            candidate = float(before) + float(proposal["delta"])
            after = clamp_tunable_value(param_name, candidate)
            if before == after:
                continue
            runtime_overrides[param_name] = after
            applied.append(
                {
                    "track": track_name,
                    "param": param_name,
                    "before": before,
                    "after": after,
                    "criterion": str(proposal["criterion"]),
                }
            )
    return applied


def build_tuning_round_contract(
    base_contract: dict[str, Any],
    *,
    autotune_root: Path,
    round_index: int,
) -> dict[str, Any]:
    round_root = ensure_dir(autotune_root / f"round_{round_index:02d}")
    round_contract = clone_contract(base_contract)
    round_contract["experiment_name"] = f"{base_contract['experiment_name']}__tune_r{round_index:02d}"
    round_contract["output_root"] = str(round_root)
    round_contract["artifacts"] = artifact_paths_for_output_root(round_root)
    return round_contract


def recommended_contract_from_selected_round(
    base_contract: dict[str, Any],
    selected_round_contract: dict[str, Any],
    *,
    selected_round: int | None,
) -> dict[str, Any]:
    recommended = clone_contract(base_contract)
    recommended["runtime"] = clone_contract(selected_round_contract.get("runtime", base_contract.get("runtime", {})))
    if "family_tracks" in selected_round_contract:
        recommended["family_tracks"] = clone_contract(selected_round_contract.get("family_tracks", []))
    recommended["tuning_recommendation"] = {
        "selected_round": selected_round,
        "source_experiment_name": selected_round_contract.get("experiment_name"),
    }
    return recommended


def run_tuning_loop(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    run_contract_fn: RunContractFn,
    max_rounds: int = 3,
    min_primary_improvement: float = 0.005,
    stagnation_patience: int = 1,
    max_updates_per_round: int = 3,
    promotion_streak_required: int = 2,
    resume: bool = False,
    apply_best: bool = False,
    force: bool = False,
    skip_doctor: bool = False,
) -> dict[str, Any]:
    contract_path = Path(contract_or_path) if isinstance(contract_or_path, (str, Path)) else None
    base_contract = coerce_contract(contract_or_path)
    autotune_root = ensure_dir(Path(base_contract["output_root"]) / "autotune")
    save_json(autotune_root / "base_contract_snapshot.json", base_contract)

    history: list[dict[str, Any]] = []
    best_primary_score = float("inf")
    best_round: int | None = None
    best_round_contract: dict[str, Any] | None = None
    last_round_contract: dict[str, Any] | None = None
    no_progress_rounds = 0
    promotion_streak = 0
    status = "max_rounds"
    start_round_index = 1
    working_contract = clone_contract(base_contract)

    if resume:
        prior_summary_path = autotune_root / "tune_summary.json"
        prior_recommended_path = autotune_root / "recommended_contract.json"
        if prior_summary_path.exists():
            prior = load_json(prior_summary_path)
            history = list(prior.get("rounds", []))
            best_round_raw = prior.get("best_round")
            if isinstance(best_round_raw, int):
                best_round = best_round_raw
            best_score_raw = prior.get("best_primary_score")
            if best_score_raw is not None:
                try:
                    best_primary_score = float(best_score_raw)
                except (TypeError, ValueError):
                    best_primary_score = float("inf")
            if history:
                last_logged = history[-1]
                start_round_index = int(last_logged.get("round", 0)) + 1
                promotion_streak = int(last_logged.get("promotion_streak", 0))
                no_progress_rounds = 0 if bool(last_logged.get("improved", False)) else 1
            if best_round is not None:
                best_round_path = autotune_root / f"round_{best_round:02d}" / "contract.json"
                if best_round_path.exists():
                    best_round_contract = load_json(best_round_path)
            if prior_recommended_path.exists():
                working_contract = clone_contract(load_json(prior_recommended_path))
            elif history:
                last_contract_path = Path(history[-1]["contract_path"])
                if last_contract_path.exists():
                    working_contract = clone_contract(load_json(last_contract_path))

    for round_index in range(start_round_index, int(max_rounds) + 1):
        round_contract = build_tuning_round_contract(
            working_contract,
            autotune_root=autotune_root,
            round_index=round_index,
        )
        last_round_contract = clone_contract(round_contract)
        save_json(round_contract["artifacts"]["contract"], round_contract)
        run_summary = run_contract_fn(round_contract, force=force, skip_doctor=skip_doctor)

        eval_path = Path(round_contract["artifacts"]["eval_report"])
        if eval_path.exists():
            eval_report = load_json(eval_path)
        else:
            eval_report = {
                "overall_pass": False,
                "required_family_tracks": [],
                "blocked_family_tracks": [],
                "family_reports": {},
                "next_steps": [],
                "doctor_status": "blocked" if run_summary.get("status") == "blocked" else None,
            }

        primary_score = primary_score_from_eval_report(eval_report)
        improved = (
            math.isfinite(primary_score)
            and (not math.isfinite(best_primary_score) or primary_score <= best_primary_score - float(min_primary_improvement))
        )
        if improved:
            best_primary_score = primary_score
            best_round = round_index
            best_round_contract = clone_contract(round_contract)
            no_progress_rounds = 0
        else:
            no_progress_rounds += 1

        if bool(eval_report.get("overall_pass", False)):
            promotion_streak += 1
        else:
            promotion_streak = 0

        round_log: dict[str, Any] = {
            "round": round_index,
            "contract_path": round_contract["artifacts"]["contract"],
            "run_status": run_summary.get("status"),
            "overall_pass": bool(eval_report.get("overall_pass", False)),
            "primary_score": None if not math.isfinite(primary_score) else float(primary_score),
            "improved": bool(improved),
            "required_tracks": list(eval_report.get("required_family_tracks", [])),
            "blocked_tracks": list(eval_report.get("blocked_family_tracks", [])),
            "next_steps": list(eval_report.get("next_steps", [])),
            "promotion_streak": int(promotion_streak),
            "promotion_streak_required": int(promotion_streak_required),
            "resumed_run": bool(resume),
        }

        if run_summary.get("status") == "blocked":
            status = "blocked"
            history.append(round_log)
            break
        if bool(eval_report.get("overall_pass", False)):
            if promotion_streak >= int(promotion_streak_required):
                status = "promote"
                history.append(round_log)
                break
            round_log["next_steps"] = [
                f"Promotion passed this round but waiting for confirmation streak ({promotion_streak}/{promotion_streak_required})."
            ]
            history.append(round_log)
            if round_index >= int(max_rounds):
                status = "max_rounds"
                break
            continue
        if round_index >= int(max_rounds):
            status = "max_rounds"
            history.append(round_log)
            break
        if no_progress_rounds > int(stagnation_patience):
            status = "no_progress"
            history.append(round_log)
            break

        required_set = set(required_track_names_from_eval(eval_report))
        blocked_required = [
            str(name)
            for name in eval_report.get("blocked_family_tracks", [])
            if str(name) in required_set
        ]
        proposals = propose_tuning_updates(eval_report, max_updates_per_round=max_updates_per_round)
        applied = apply_tuning_updates(
            working_contract,
            proposals,
            blocked_tracks=blocked_required,
        )
        round_log["proposed_updates"] = proposals
        round_log["applied_updates"] = applied
        history.append(round_log)
        if not applied:
            status = "stalled"
            break

    selected_round = best_round
    selected_round_contract = best_round_contract
    if selected_round_contract is None:
        selected_round_contract = last_round_contract or clone_contract(base_contract)
        if selected_round is None and history:
            selected_round = int(history[-1]["round"])

    recommended_contract = recommended_contract_from_selected_round(
        base_contract,
        selected_round_contract,
        selected_round=selected_round,
    )
    recommended_contract_path = autotune_root / "recommended_contract.json"
    save_json(recommended_contract_path, recommended_contract)

    applied_contract_path: str | None = None
    if apply_best and contract_path is not None:
        save_json(contract_path, recommended_contract)
        applied_contract_path = str(contract_path)

    summary_path = autotune_root / "tune_summary.json"
    summary = {
        "experiment_name": base_contract["experiment_name"],
        "status": status,
        "autotune_root": str(autotune_root),
        "resumed": bool(resume),
        "start_round_index": int(start_round_index),
        "rounds_run": len(history),
        "best_round": best_round,
        "best_primary_score": None if not math.isfinite(best_primary_score) else float(best_primary_score),
        "max_rounds": int(max_rounds),
        "min_primary_improvement": float(min_primary_improvement),
        "stagnation_patience": int(stagnation_patience),
        "max_updates_per_round": int(max_updates_per_round),
        "promotion_streak_required": int(promotion_streak_required),
        "promotion_streak_final": int(promotion_streak),
        "selected_round": selected_round,
        "recommended_contract_path": str(recommended_contract_path),
        "apply_best": bool(apply_best),
        "applied_contract_path": applied_contract_path,
        "rounds": history,
    }
    save_json(summary_path, summary)
    summary["tune_summary_path"] = str(summary_path)
    return summary
