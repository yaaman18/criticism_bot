from __future__ import annotations

import argparse
import importlib
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from .common import ensure_dir, load_json, save_json
from .evaluate_trm_as_behavior import evaluate_compare_root
from .harness_contracts import (
    DEFAULT_ACCEPTANCE,
    DEFAULT_FAMILY_ORDER,
    DEFAULT_FAMILY_PROFILES,
    TUNABLE_RUNTIME_PARAMS,
    artifact_paths_for_output_root as _artifact_paths_for_output_root,
    clone_contract as _clone_contract,
    coerce_contract as _coerce_contract,
    default_family_tracks as _default_family_tracks,
    normalize_track as _normalize_track,
    resolve_family_tracks as _resolve_family_tracks,
)
from .harness_tuning import (
    apply_tuning_updates as _apply_tuning_updates,
    build_tuning_round_contract as _build_tuning_round_contract,
    clamp_tunable_value as _clamp_tunable_value,
    default_tunable_value as _default_tunable_value,
    failed_criteria_counter as _failed_criteria_counter,
    primary_score_from_eval_report as _primary_score_from_eval_report,
    propose_tuning_updates as _propose_tuning_updates,
    recommended_contract_from_selected_round as _recommended_contract_from_selected_round,
    required_track_names_from_eval as _required_track_names_from_eval,
    run_tuning_loop as _run_tuning_loop,
    target_runtime_overrides as _target_runtime_overrides,
)
from .erie_runtime import add_environment_config_args, environment_config_from_args
from .sweep_trm_va_modes import _compare_one_seed


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path_is_within(path: str | Path, parent: str | Path) -> bool:
    try:
        Path(path).absolute().relative_to(Path(parent).absolute())
    except ValueError:
        return False
    return True


def _safe_version(import_name: str) -> tuple[str | None, str | None]:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # pragma: no cover - exercised indirectly in doctor tests
        return None, f"{type(exc).__name__}: {exc}"
    return str(getattr(module, "__version__", "unknown")), None


def run_doctor() -> dict[str, Any]:
    repo_root = _repo_root()
    repo_venv_root = repo_root / ".venv"
    repo_venv_python = repo_venv_root / "bin" / "python"
    repo_venv_pip = repo_venv_root / "bin" / "pip"
    pip_path = shutil.which("pip")
    pytest_path = shutil.which("pytest")
    using_repo_venv = _path_is_within(sys.executable, repo_venv_root)
    pip_matches_active_python = None
    if pip_path is not None:
        pip_matches_active_python = Path(pip_path).absolute().parent == Path(sys.executable).absolute().parent

    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "python_executable": sys.executable,
        "expected_python_executable": str(repo_venv_python),
        "expected_pip_executable": str(repo_venv_pip),
        "python_version": sys.version.split()[0],
        "recommended_bootstrap_command": "./scripts/bootstrap_env.sh",
        "recommended_test_command": "./.venv/bin/python -m pytest",
        "recommended_harness_command": "./.venv/bin/python -m trm_pipeline.experiment_harness doctor",
        "cwd": str(Path.cwd()),
        "pip_path": pip_path,
        "pytest_path": pytest_path,
        "repo_venv_present": repo_venv_python.exists(),
        "using_repo_venv": using_repo_venv,
        "pip_matches_active_python": pip_matches_active_python,
        "blocking_issues": [],
        "warnings": [],
    }

    if repo_venv_python.exists() and not using_repo_venv:
        report["warnings"].append(
            "Active interpreter is outside the repo `.venv`. Prefer `./.venv/bin/python ...` or run `./scripts/bootstrap_env.sh` first."
        )
    if pip_path is not None and pip_matches_active_python is False:
        report["warnings"].append(
            "`pip` resolves to a different bin directory than the active interpreter. Use `python -m pip` or `./scripts/bootstrap_env.sh`."
        )

    try:
        import trm_pipeline  # noqa: F401

        report["package_import_ok"] = True
    except Exception as exc:
        report["package_import_ok"] = False
        report["blocking_issues"].append(f"failed to import trm_pipeline: {type(exc).__name__}: {exc}")

    numpy_version, numpy_error = _safe_version("numpy")
    torch_version, torch_error = _safe_version("torch")
    report["numpy_version"] = numpy_version
    report["torch_version"] = torch_version
    if numpy_error is not None:
        report["blocking_issues"].append(f"failed to import numpy: {numpy_error}")
    if torch_error is not None:
        report["blocking_issues"].append(f"failed to import torch: {torch_error}")

    bridge_ok = None
    bridge_error = None
    if numpy_error is None and torch_error is None:
        try:
            import numpy as np
            import torch

            arr = np.zeros((1,), dtype=np.float32)
            tensor = torch.from_numpy(arr)
            bridge_ok = bool(tuple(tensor.shape) == (1,))
        except Exception as exc:  # pragma: no cover - depends on local torch/numpy wheel state
            bridge_ok = False
            bridge_error = f"{type(exc).__name__}: {exc}"
            report["blocking_issues"].append(f"torch/numpy bridge failed: {bridge_error}")
    report["torch_numpy_bridge_ok"] = bridge_ok
    if bridge_error is not None:
        report["torch_numpy_bridge_error"] = bridge_error

    if numpy_version is not None:
        try:
            numpy_major = int(str(numpy_version).split(".", 1)[0])
        except ValueError:
            numpy_major = None
        if numpy_major is not None and numpy_major >= 2:
            report["warnings"].append(
                "NumPy 2.x detected. This repo currently expects `numpy<2`; verify the active PyTorch wheel supports NumPy 2."
            )

    report["status"] = "blocked" if report["blocking_issues"] else "ok"
    return report


def build_experiment_contract(
    *,
    output_root: str | Path,
    experiment_name: str,
    candidate_mode: str = "analytic__module_primary",
    baseline_mode: str = "analytic__analytic",
    seed_catalog: str = "data/lenia_official/animals2d_seeds.json",
    seed_start: int = 20260318,
    num_seeds: int = 5,
    holdout_seed_start: int | None = None,
    holdout_num_seeds: int = 0,
    steps: int = 24,
    warmup_steps: int = 4,
    lookahead_horizon: int = 2,
    lookahead_discount: float = 0.85,
    resource_patches: int = 3,
    hazard_patches: int = 3,
    shelter_patches: int = 1,
    trm_a_checkpoint: str | None = None,
    trm_b_checkpoint: str | None = None,
    module_manifest: str | None = None,
    policy_mode: str = "closed_loop",
    require_holdout_for_promotion: bool = False,
    acceptance: dict[str, Any] | None = None,
    family_tracks: list[dict[str, Any]] | None = None,
    families: list[str] | None = None,
) -> dict[str, Any]:
    out_root = ensure_dir(output_root)
    merged_acceptance = dict(DEFAULT_ACCEPTANCE)
    merged_acceptance["require_holdout_for_promotion"] = bool(require_holdout_for_promotion)
    if acceptance:
        merged_acceptance.update(acceptance)
    resolved_family_tracks = (
        [_normalize_track(track) for track in family_tracks]
        if family_tracks is not None
        else _default_family_tracks(families)
    )
    artifacts = _artifact_paths_for_output_root(out_root)
    resolved_holdout_seed_start = (
        int(seed_start) + int(num_seeds) if holdout_seed_start is None else int(holdout_seed_start)
    )
    return {
        "version": 1,
        "experiment_name": experiment_name,
        "experiment_kind": "trm_va_mode_sweep",
        "output_root": str(out_root),
        "candidate_mode": candidate_mode,
        "baseline_mode": baseline_mode,
        "runtime": {
            "seed_catalog": seed_catalog,
            "seed_start": int(seed_start),
            "num_seeds": int(num_seeds),
            "holdout_seed_start": int(resolved_holdout_seed_start),
            "holdout_num_seeds": int(holdout_num_seeds),
            "steps": int(steps),
            "warmup_steps": int(warmup_steps),
            "lookahead_horizon": int(lookahead_horizon),
            "lookahead_discount": float(lookahead_discount),
            "resource_patches": int(resource_patches),
            "hazard_patches": int(hazard_patches),
            "shelter_patches": int(shelter_patches),
            "trm_a_checkpoint": trm_a_checkpoint,
            "trm_b_checkpoint": trm_b_checkpoint,
            "module_manifest": module_manifest,
            "policy_mode": policy_mode,
        },
        "acceptance": merged_acceptance,
        "family_tracks": resolved_family_tracks,
        "artifacts": artifacts,
    }
def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    weight = rank - lo
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    alpha: float = 0.05,
    seed: int = 20260419,
) -> dict[str, float]:
    finite_values = [float(v) for v in values if math.isfinite(float(v))]
    if not finite_values:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    mean_value = _mean(finite_values)
    if len(finite_values) == 1:
        return {"mean": mean_value, "lower": mean_value, "upper": mean_value}
    rng = random.Random(seed)
    boots: list[float] = []
    n = len(finite_values)
    for _ in range(max(1, int(samples))):
        sample = [finite_values[rng.randrange(n)] for _ in range(n)]
        boots.append(_mean(sample))
    lower = _percentile(boots, alpha / 2.0)
    upper = _percentile(boots, 1.0 - alpha / 2.0)
    return {"mean": mean_value, "lower": lower, "upper": upper}


def _empty_criteria(acceptance: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_final_homeostatic_error": _criterion(
            name="mean_final_homeostatic_error",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_mean_final_homeostatic_error"]),
            comparator="<=",
        ),
        "mean_mean_homeostatic_error": _criterion(
            name="mean_mean_homeostatic_error",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_mean_mean_homeostatic_error"]),
            comparator="<=",
        ),
        "dead_fraction": _criterion(
            name="dead_fraction",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_dead_fraction"]),
            comparator="<=",
        ),
        "final_improvement_vs_baseline": _criterion(
            name="final_improvement_vs_baseline",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_final_improvement_vs_baseline"]),
            comparator=">=",
        ),
        "final_improvement_ci_lower": _criterion(
            name="final_improvement_ci_lower",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_final_improvement_ci_lower"]),
            comparator=">=",
        ),
        "best_mode_frequency": _criterion(
            name="best_mode_frequency",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_best_mode_frequency"]),
            comparator=">=",
        ),
        "stress_defensive_rate": _criterion(
            name="stress_defensive_rate",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_stress_defensive_rate"]),
            comparator=">=",
        ),
        "stress_exploit_rate": _criterion(
            name="stress_exploit_rate",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_stress_exploit_rate"]),
            comparator="<=",
        ),
        "action_diversity": _criterion(
            name="action_diversity",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_action_diversity"]),
            comparator=">=",
        ),
        "intake_rate": _criterion(
            name="intake_rate",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_intake_rate"]),
            comparator="<=",
        ),
        "navigation_rate": _criterion(
            name="navigation_rate",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_navigation_rate"]),
            comparator=">=",
        ),
        "trace_ablation_spawn_delta": _criterion(
            name="trace_ablation_spawn_delta",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_trace_ablation_spawn_delta"]),
            comparator=">=",
        ),
        "mean_p_t_min": _criterion(
            name="mean_p_t_min",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_mean_p_t"]),
            comparator=">=",
        ),
        "mean_p_t_max": _criterion(
            name="mean_p_t_max",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_mean_p_t"]),
            comparator="<=",
        ),
        "mean_challenge_fraction_min": _criterion(
            name="mean_challenge_fraction_min",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_mean_challenge_fraction"]),
            comparator=">=",
        ),
        "mean_challenge_fraction_max": _criterion(
            name="mean_challenge_fraction_max",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_mean_challenge_fraction"]),
            comparator="<=",
        ),
        "role_switch_events_total": _criterion(
            name="role_switch_events_total",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_role_switch_events_total"]),
            comparator=">=",
        ),
        "mean_aux_nontrivial_action_count": _criterion(
            name="mean_aux_nontrivial_action_count",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["min_mean_aux_nontrivial_action_count"]),
            comparator=">=",
        ),
        "non_degradation_mean_homeostasis": _criterion(
            name="non_degradation_mean_homeostasis",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_mean_homeostatic_degradation"]),
            comparator="<=",
        ),
        "non_degradation_stress_exploit": _criterion(
            name="non_degradation_stress_exploit",
            passed=False,
            actual=float("nan"),
            expected=float(acceptance["max_stress_exploit_degradation"]),
            comparator="<=",
        ),
    }


def _empty_track_report(
    *,
    experiment_name: str,
    track_name: str,
    split_name: str,
    compare_root: Path,
    candidate_mode: str,
    baseline_mode: str,
    acceptance: dict[str, Any],
    promotion_target: str,
    required_for_promotion: bool,
    doctor_status: str | None,
    evaluation_error: str,
) -> dict[str, Any]:
    criteria = _empty_criteria(acceptance)
    report = {
        "experiment_name": experiment_name,
        "track_name": track_name,
        "split": split_name,
        "compare_root": str(compare_root),
        "candidate_mode": candidate_mode,
        "baseline_mode": baseline_mode,
        "required_for_promotion": required_for_promotion,
        "promotion_target": promotion_target,
        "inspected_seeds": 0,
        "missing_seeds": [],
        "doctor_status": doctor_status,
        "acceptance": acceptance,
        "summary": {
            "candidate": {
                "mean_final_homeostatic_error": None,
                "mean_mean_homeostatic_error": None,
                "dead_fraction": None,
                "mean_stress_defensive_rate": None,
                "mean_stress_exploit_rate": None,
                "mean_action_diversity": None,
                "mean_intake_rate": None,
                "mean_navigation_rate": None,
                "mean_trace_ablation_spawn_delta": None,
                "mean_p_t": None,
                "mean_challenge_fraction": None,
                "role_switch_events_total": None,
                "mean_aux_nontrivial_action_count": None,
            },
            "baseline": {
                "mean_final_homeostatic_error": None,
                "mean_mean_homeostatic_error": None,
                "mean_stress_defensive_rate": None,
                "mean_stress_exploit_rate": None,
                "mean_action_diversity": None,
                "mean_intake_rate": None,
                "mean_navigation_rate": None,
                "mean_trace_ablation_spawn_delta": None,
                "mean_p_t": None,
                "mean_challenge_fraction": None,
                "role_switch_events_total": None,
                "mean_aux_nontrivial_action_count": None,
            },
            "best_mode_frequency": 0.0,
            "final_improvement_vs_baseline": None,
        },
        "statistics": {
            "final_improvement_bootstrap_ci95": {
                "mean": None,
                "lower": None,
                "upper": None,
            }
        },
        "criteria": criteria,
        "overall_pass": False,
        "evaluation_error": evaluation_error,
    }
    report["next_steps"] = _derive_track_next_steps(report)
    report["next_steps"].insert(0, f"Evaluation error: {evaluation_error}")
    return report


def _criterion(
    *,
    name: str,
    passed: bool,
    actual: float,
    expected: float,
    comparator: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": None if not math.isfinite(actual) else float(actual),
        "expected": float(expected),
        "comparator": comparator,
    }


def _default_diag() -> dict[str, float]:
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
        "trace_ablation_spawn_delta": 0.0,
        "mean_spawn_drive": 0.0,
        "mean_split_drive": 0.0,
        "spawn_candidate_rate": 0.0,
        "split_candidate_rate": 0.0,
        "mean_p_t": 0.5,
        "mean_challenge_fraction": 0.5,
        "role_switch_events_total": 0.0,
        "mean_aux_nontrivial_action_count": 0.0,
    }


def _load_per_seed_ids(compare_root: Path) -> list[int]:
    aggregate_path = compare_root / "aggregate_summary.json"
    if not aggregate_path.exists():
        return []
    aggregate = load_json(aggregate_path)
    rows = aggregate.get("per_seed", [])
    if not isinstance(rows, list):
        return []
    seeds: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or "seed" not in row:
            continue
        try:
            seeds.append(int(row["seed"]))
        except (TypeError, ValueError):
            continue
    return seeds


def _seed_leakage_report(dev_root: Path, holdout_root: Path) -> dict[str, Any]:
    dev_seeds = _load_per_seed_ids(dev_root)
    holdout_seeds = _load_per_seed_ids(holdout_root)
    overlap = sorted(set(dev_seeds).intersection(holdout_seeds))
    return {
        "dev_seeds": dev_seeds,
        "holdout_seeds": holdout_seeds,
        "overlap_seeds": overlap,
        "has_overlap": bool(overlap),
    }


def _derive_track_next_steps(report: dict[str, Any]) -> list[str]:
    criteria = report["criteria"]
    steps: list[str] = []
    if not criteria["mean_final_homeostatic_error"]["passed"]:
        steps.append(
            "Reduce candidate final homeostatic error before promotion. Revisit lookahead depth, reward shaping, or candidate module scaling."
        )
    if not criteria["mean_mean_homeostatic_error"]["passed"]:
        steps.append(
            "The candidate is drifting over the full episode. Inspect long-horizon traces and tighten stability constraints, not just final-state behavior."
        )
    if not criteria["final_improvement_vs_baseline"]["passed"]:
        steps.append(
            "Keep the baseline as default. The candidate has not yet beaten the baseline sweep on final homeostatic error."
        )
    if "final_improvement_ci_lower" in criteria and not criteria["final_improvement_ci_lower"]["passed"]:
        steps.append(
            "The improvement confidence is still weak. Increase seed count or stabilize behavior before promotion."
        )
    if not criteria["best_mode_frequency"]["passed"]:
        steps.append(
            "The candidate is not winning often enough across seeds. Expand family-diversified training or narrow the contract to the families it is supposed to improve."
        )
    if not criteria["stress_defensive_rate"]["passed"]:
        steps.append(
            "Under stress the policy is still not defensive enough. Increase toxic/fragile family weight or tighten the TRM-As quality gate for exploit-heavy traces."
        )
    if not criteria["stress_exploit_rate"]["passed"]:
        steps.append(
            "Stress-time exploitation remains too high. Penalize exploit actions more aggressively in high-toxicity or low-boundary states."
        )
    if "action_diversity" in criteria and not criteria["action_diversity"]["passed"]:
        steps.append(
            "Action diversity is too low. Reduce action collapse and confirm the candidate can switch among approach/withdraw/reconfigure under changing conditions."
        )
    if "intake_rate" in criteria and not criteria["intake_rate"]["passed"]:
        steps.append(
            "Intake is over-dominant. Add stronger anti-intake bias outside low-energy states and enforce veto behavior under stress."
        )
    if "navigation_rate" in criteria and not criteria["navigation_rate"]["passed"]:
        steps.append(
            "Navigation/reconfiguration actions are too rare. Increase movement utility and test non-static trajectories across seeds."
        )
    if "trace_ablation_spawn_delta" in criteria and not criteria["trace_ablation_spawn_delta"]["passed"]:
        steps.append(
            "Trace influence on spawn dynamics is too weak. Increase trace persistence/signal coupling and verify the spawn candidate responds to trace ablation."
        )
    if "mean_p_t_min" in criteria and not criteria["mean_p_t_min"]["passed"]:
        steps.append(
            "Challenge ratio is too low across episodes. Rebalance activity-distribution terms so challenge bodies are not collapsed."
        )
    if "mean_p_t_max" in criteria and not criteria["mean_p_t_max"]["passed"]:
        steps.append(
            "Challenge ratio is too high across episodes. Increase hazard/boundary penalties in activity-distribution updates."
        )
    if "mean_challenge_fraction_min" in criteria and not criteria["mean_challenge_fraction_min"]["passed"]:
        steps.append(
            "Assigned challenge fraction is below contract. Recheck role scoring and p_t clipping ranges."
        )
    if "mean_challenge_fraction_max" in criteria and not criteria["mean_challenge_fraction_max"]["passed"]:
        steps.append(
            "Assigned challenge fraction is above contract. Strengthen conservative assignment bias under stress."
        )
    if "role_switch_events_total" in criteria and not criteria["role_switch_events_total"]["passed"]:
        steps.append(
            "Role switching is too static. Increase sensitivity of p_t/role scoring to environment and viability deltas."
        )
    if "mean_aux_nontrivial_action_count" in criteria and not criteria["mean_aux_nontrivial_action_count"]["passed"]:
        steps.append(
            "Auxiliary bodies are too passive. Raise aux role-policy reactivity and verify nontrivial action coverage."
        )
    if "non_degradation_mean_homeostasis" in criteria and not criteria["non_degradation_mean_homeostasis"]["passed"]:
        steps.append(
            "The candidate is degrading mean homeostatic behavior against baseline. Tighten long-horizon stability before promotion."
        )
    if "non_degradation_stress_exploit" in criteria and not criteria["non_degradation_stress_exploit"]["passed"]:
        steps.append(
            "The candidate worsens stress-time exploit behavior versus baseline. Increase defensive bias under stress families."
        )
    if not criteria["dead_fraction"]["passed"]:
        steps.append(
            "The candidate is causing too many deaths. Recheck viability-monitor blending and inspect failure traces before another promotion attempt."
        )
    if not steps:
        steps.append("Promote the candidate to the next experiment stage for this track and expand the sweep width or difficulty range.")
    return steps


def _derive_harness_next_steps(
    *,
    doctor_status: str | None,
    family_reports: dict[str, dict[str, Any]],
    required_track_names: list[str],
    overall_pass: bool,
) -> list[str]:
    if doctor_status == "blocked":
        return [
            "Fix the runtime environment first. The harness should not promote results from a blocked doctor run.",
            "Align the active Python environment with the repo requirements and rerun `./.venv/bin/python -m pytest` before launching the family sweeps again.",
        ]
    if overall_pass:
        promoted = [name for name in required_track_names if family_reports[name]["overall_pass"]]
        return [
            "Promote the candidate for the required family tracks: " + ", ".join(promoted) + ".",
            "Expand the next contract with harder seeds, wider sweeps, or optional family tracks once the promoted set remains stable.",
        ]

    steps: list[str] = []
    for name in required_track_names:
        report = family_reports[name]
        if report["overall_pass"]:
            continue
        steps.append(f"Keep `{name}` below promotion. Target: {report['promotion_target']}")
        if report["next_steps"]:
            steps.append(report["next_steps"][0])
    if not steps:
        steps.append("Review the family reports and tighten the contract before the next run.")
    return steps


def _failed_criteria(report: dict[str, Any]) -> list[str]:
    return [name for name, criterion in report.get("criteria", {}).items() if not criterion.get("passed", False)]


def _track_failure_reasons(report: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if report.get("evaluation_error"):
        reasons.append("evaluation_error")
    leakage = dict(report.get("seed_leakage", {}))
    if bool(leakage.get("has_overlap", False)):
        reasons.append("seed_leakage")
    if "overall_pass_holdout" in report and not bool(report.get("overall_pass_holdout", True)):
        reasons.append("holdout_fail")

    for name in _failed_criteria(report):
        reasons.append(f"dev:{name}")
    holdout_criteria = dict(report.get("criteria_holdout", {}))
    for name, criterion in holdout_criteria.items():
        if not bool(criterion.get("passed", False)):
            reasons.append(f"holdout:{name}")

    seen: set[str] = set()
    deduped: list[str] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)
    return deduped


def _track_reason_line(track_name: str, reasons: list[str]) -> str:
    if not reasons:
        return f"{track_name}(unknown)"
    return f"{track_name}({', '.join(reasons[:3])})"


def build_promotion_decision(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    eval_report: dict[str, Any],
    doctor_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    doctor_status = (doctor_report or {}).get("status", eval_report.get("doctor_status"))
    required_tracks = list(eval_report.get("required_family_tracks", []))
    eligible_tracks = list(eval_report.get("eligible_family_tracks", []))
    blocked_tracks = list(eval_report.get("blocked_family_tracks", []))
    family_reports = dict(eval_report.get("family_reports", {}))

    status = "blocked" if doctor_status == "blocked" else ("promote" if eval_report.get("overall_pass") else "revise")
    track_decisions = []
    blocked_reason_lines: list[str] = []
    for track_name, report in family_reports.items():
        track_status = "promote" if report.get("overall_pass") else "hold"
        failure_reasons = _track_failure_reasons(report)
        if track_status == "hold":
            blocked_reason_lines.append(_track_reason_line(track_name, failure_reasons))
        track_decisions.append(
            {
                "track_name": track_name,
                "status": track_status,
                "required_for_promotion": bool(report.get("required_for_promotion", False)),
                "promotion_target": report.get("promotion_target"),
                "failed_criteria": _failed_criteria(report),
                "failure_reasons": failure_reasons,
                "holdout_enabled": "overall_pass_holdout" in report,
                "holdout_pass": report.get("overall_pass_holdout"),
                "seed_leakage_overlap": bool(report.get("seed_leakage", {}).get("has_overlap", False)),
                "top_next_step": report.get("next_steps", [None])[0],
                "candidate_summary": report.get("summary", {}).get("candidate", {}),
                "baseline_summary": report.get("summary", {}).get("baseline", {}),
            }
        )

    if status == "blocked":
        recommendation = "Do not interpret experiment results until the runtime environment passes doctor."
    elif status == "promote":
        recommendation = (
            "Promote the candidate for the required family tracks: " + ", ".join(required_tracks) + "."
            if required_tracks
            else "Promote the candidate for the evaluated tracks."
        )
    else:
        recommendation = (
            "Keep the candidate below promotion for: " + ", ".join(blocked_tracks) + "."
            if blocked_tracks
            else "Keep the candidate below promotion until the contract is clarified."
        )
        if blocked_reason_lines:
            recommendation += " Reasons: " + "; ".join(blocked_reason_lines) + "."

    ci_summary = (
        "blocked: " + "; ".join(blocked_reason_lines)
        if blocked_reason_lines
        else ("promote: " + ", ".join(eligible_tracks) if eligible_tracks else "blocked: no eligible tracks")
    )

    return {
        "experiment_name": contract["experiment_name"],
        "candidate_mode": contract["candidate_mode"],
        "baseline_mode": contract["baseline_mode"],
        "doctor_status": doctor_status,
        "status": status,
        "promotion_ready_tracks": eligible_tracks,
        "blocked_tracks": blocked_tracks,
        "required_tracks": required_tracks,
        "recommendation": recommendation,
        "ci_summary": ci_summary,
        "next_steps": list(eval_report.get("next_steps", [])),
        "track_decisions": track_decisions,
    }


def _evaluate_compare_root(
    *,
    experiment_name: str,
    track_name: str,
    split_name: str,
    compare_root: Path,
    candidate_mode: str,
    baseline_mode: str,
    acceptance: dict[str, Any],
    promotion_target: str,
    required_for_promotion: bool,
    doctor_status: str | None,
) -> dict[str, Any]:
    aggregate_path = compare_root / "aggregate_summary.json"
    if not aggregate_path.exists():
        return _empty_track_report(
            experiment_name=experiment_name,
            track_name=track_name,
            split_name=split_name,
            compare_root=compare_root,
            candidate_mode=candidate_mode,
            baseline_mode=baseline_mode,
            acceptance=acceptance,
            promotion_target=promotion_target,
            required_for_promotion=required_for_promotion,
            doctor_status=doctor_status,
            evaluation_error=f"missing aggregate summary: {aggregate_path}",
        )
    try:
        aggregate = load_json(aggregate_path)
    except Exception as exc:
        return _empty_track_report(
            experiment_name=experiment_name,
            track_name=track_name,
            split_name=split_name,
            compare_root=compare_root,
            candidate_mode=candidate_mode,
            baseline_mode=baseline_mode,
            acceptance=acceptance,
            promotion_target=promotion_target,
            required_for_promotion=required_for_promotion,
            doctor_status=doctor_status,
            evaluation_error=f"failed to parse aggregate summary: {type(exc).__name__}: {exc}",
        )
    stress_threshold = float(acceptance.get("stress_threshold", DEFAULT_ACCEPTANCE["stress_threshold"]))

    candidate_final: list[float] = []
    candidate_mean: list[float] = []
    candidate_dead: list[float] = []
    candidate_defensive: list[float] = []
    candidate_exploit: list[float] = []
    candidate_action_diversity: list[float] = []
    candidate_intake_rate: list[float] = []
    candidate_navigation_rate: list[float] = []
    candidate_trace_spawn_delta: list[float] = []
    candidate_mean_p_t: list[float] = []
    candidate_mean_challenge_fraction: list[float] = []
    candidate_role_switch_events_total: list[float] = []
    candidate_mean_aux_nontrivial_action_count: list[float] = []
    baseline_defensive: list[float] = []
    baseline_exploit: list[float] = []
    baseline_action_diversity: list[float] = []
    baseline_intake_rate: list[float] = []
    baseline_navigation_rate: list[float] = []
    baseline_trace_spawn_delta: list[float] = []
    baseline_mean_p_t: list[float] = []
    baseline_mean_challenge_fraction: list[float] = []
    baseline_role_switch_events_total: list[float] = []
    baseline_mean_aux_nontrivial_action_count: list[float] = []
    baseline_final: list[float] = []
    baseline_mean: list[float] = []
    best_candidate_count = 0
    inspected_seeds = 0
    missing_seeds: list[int] = []
    seed_errors: list[str] = []

    for row in aggregate.get("per_seed", []):
        try:
            seed = int(row["seed"])
        except Exception:
            continue
        seed_root = compare_root / f"seed_{seed}"
        try:
            comparison = load_json(seed_root / "comparison_summary.json")
            diagnostics = evaluate_compare_root(seed_root, stress_threshold=stress_threshold)
        except Exception as exc:
            seed_errors.append(f"seed={seed}: {type(exc).__name__}: {exc}")
            missing_seeds.append(seed)
            continue
        candidate_summary = comparison["results"].get(candidate_mode)
        baseline_summary = comparison["results"].get(baseline_mode)
        if candidate_summary is None or baseline_summary is None:
            missing_seeds.append(seed)
            continue

        candidate_diag = diagnostics["mode_diagnostics"].get(candidate_mode, _default_diag())
        candidate_final.append(float(candidate_summary["final_homeostatic_error"]))
        candidate_mean.append(float(candidate_summary["mean_homeostatic_error"]))
        candidate_dead.append(float(bool(candidate_summary.get("dead", False))))
        candidate_defensive.append(float(candidate_diag["stress_defensive_rate"]))
        candidate_exploit.append(float(candidate_diag["stress_exploit_rate"]))
        candidate_action_diversity.append(
            float(candidate_diag.get("action_diversity", candidate_summary.get("action_diversity", 0.0)))
        )
        candidate_intake_rate.append(float(candidate_diag.get("intake_rate", 0.0)))
        candidate_navigation_rate.append(float(candidate_diag.get("navigation_rate", 0.0)))
        candidate_trace_spawn_delta.append(
            float(
                candidate_diag.get(
                    "trace_ablation_spawn_delta",
                    candidate_summary.get("mean_trace_ablation_spawn_delta", 0.0),
                )
            )
        )
        candidate_mean_p_t.append(float(candidate_diag.get("mean_p_t", candidate_summary.get("mean_p_t", 0.5))))
        candidate_mean_challenge_fraction.append(
            float(candidate_diag.get("mean_challenge_fraction", candidate_summary.get("mean_challenge_fraction", 0.5)))
        )
        candidate_role_switch_events_total.append(
            float(candidate_diag.get("role_switch_events_total", candidate_summary.get("role_switch_events_total", 0.0)))
        )
        candidate_mean_aux_nontrivial_action_count.append(
            float(
                candidate_diag.get(
                    "mean_aux_nontrivial_action_count",
                    candidate_summary.get("mean_aux_nontrivial_action_count", 0.0),
                )
            )
        )
        baseline_diag = diagnostics["mode_diagnostics"].get(baseline_mode, _default_diag())
        baseline_defensive.append(float(baseline_diag["stress_defensive_rate"]))
        baseline_exploit.append(float(baseline_diag["stress_exploit_rate"]))
        baseline_action_diversity.append(
            float(baseline_diag.get("action_diversity", baseline_summary.get("action_diversity", 0.0)))
        )
        baseline_intake_rate.append(float(baseline_diag.get("intake_rate", 0.0)))
        baseline_navigation_rate.append(float(baseline_diag.get("navigation_rate", 0.0)))
        baseline_trace_spawn_delta.append(
            float(
                baseline_diag.get(
                    "trace_ablation_spawn_delta",
                    baseline_summary.get("mean_trace_ablation_spawn_delta", 0.0),
                )
            )
        )
        baseline_mean_p_t.append(float(baseline_diag.get("mean_p_t", baseline_summary.get("mean_p_t", 0.5))))
        baseline_mean_challenge_fraction.append(
            float(baseline_diag.get("mean_challenge_fraction", baseline_summary.get("mean_challenge_fraction", 0.5)))
        )
        baseline_role_switch_events_total.append(
            float(baseline_diag.get("role_switch_events_total", baseline_summary.get("role_switch_events_total", 0.0)))
        )
        baseline_mean_aux_nontrivial_action_count.append(
            float(
                baseline_diag.get(
                    "mean_aux_nontrivial_action_count",
                    baseline_summary.get("mean_aux_nontrivial_action_count", 0.0),
                )
            )
        )
        baseline_final.append(float(baseline_summary["final_homeostatic_error"]))
        baseline_mean.append(float(baseline_summary["mean_homeostatic_error"]))
        best_candidate_count += int(comparison["derived"]["best_mode_by_final_homeostasis"] == candidate_mode)
        inspected_seeds += 1

    candidate_mean_final = _mean(candidate_final)
    candidate_mean_mean = _mean(candidate_mean)
    candidate_dead_fraction = _mean(candidate_dead)
    candidate_mean_defensive = _mean(candidate_defensive)
    candidate_mean_exploit = _mean(candidate_exploit)
    candidate_mean_action_diversity = _mean(candidate_action_diversity)
    candidate_mean_intake_rate = _mean(candidate_intake_rate)
    candidate_mean_navigation_rate = _mean(candidate_navigation_rate)
    candidate_mean_trace_spawn_delta = _mean(candidate_trace_spawn_delta)
    candidate_agg_mean_p_t = _mean(candidate_mean_p_t)
    candidate_agg_mean_challenge_fraction = _mean(candidate_mean_challenge_fraction)
    candidate_agg_role_switch_events_total = _mean(candidate_role_switch_events_total)
    candidate_agg_mean_aux_nontrivial_action_count = _mean(candidate_mean_aux_nontrivial_action_count)
    baseline_mean_defensive = _mean(baseline_defensive)
    baseline_mean_exploit = _mean(baseline_exploit)
    baseline_mean_action_diversity = _mean(baseline_action_diversity)
    baseline_mean_intake_rate = _mean(baseline_intake_rate)
    baseline_mean_navigation_rate = _mean(baseline_navigation_rate)
    baseline_mean_trace_spawn_delta = _mean(baseline_trace_spawn_delta)
    baseline_agg_mean_p_t = _mean(baseline_mean_p_t)
    baseline_agg_mean_challenge_fraction = _mean(baseline_mean_challenge_fraction)
    baseline_agg_role_switch_events_total = _mean(baseline_role_switch_events_total)
    baseline_agg_mean_aux_nontrivial_action_count = _mean(baseline_mean_aux_nontrivial_action_count)
    baseline_mean_final = _mean(baseline_final)
    baseline_mean_mean = _mean(baseline_mean)
    final_improvement = baseline_mean_final - candidate_mean_final
    mean_homeostatic_gap = candidate_mean_mean - baseline_mean_mean
    exploit_gap = candidate_mean_exploit - baseline_mean_exploit
    best_mode_frequency = float(best_candidate_count / max(inspected_seeds, 1))
    bootstrap_samples = int(acceptance.get("bootstrap_samples", DEFAULT_ACCEPTANCE["bootstrap_samples"]))
    final_delta_ci = _bootstrap_mean_ci(
        [b - c for b, c in zip(baseline_final, candidate_final)],
        samples=bootstrap_samples,
    )

    criteria = {
        "mean_final_homeostatic_error": _criterion(
            name="mean_final_homeostatic_error",
            passed=math.isfinite(candidate_mean_final)
            and candidate_mean_final <= float(acceptance["max_mean_final_homeostatic_error"]),
            actual=candidate_mean_final,
            expected=float(acceptance["max_mean_final_homeostatic_error"]),
            comparator="<=",
        ),
        "mean_mean_homeostatic_error": _criterion(
            name="mean_mean_homeostatic_error",
            passed=math.isfinite(candidate_mean_mean)
            and candidate_mean_mean <= float(acceptance["max_mean_mean_homeostatic_error"]),
            actual=candidate_mean_mean,
            expected=float(acceptance["max_mean_mean_homeostatic_error"]),
            comparator="<=",
        ),
        "dead_fraction": _criterion(
            name="dead_fraction",
            passed=math.isfinite(candidate_dead_fraction)
            and candidate_dead_fraction <= float(acceptance["max_dead_fraction"]),
            actual=candidate_dead_fraction,
            expected=float(acceptance["max_dead_fraction"]),
            comparator="<=",
        ),
        "final_improvement_vs_baseline": _criterion(
            name="final_improvement_vs_baseline",
            passed=math.isfinite(final_improvement)
            and final_improvement >= float(acceptance["min_final_improvement_vs_baseline"]),
            actual=final_improvement,
            expected=float(acceptance["min_final_improvement_vs_baseline"]),
            comparator=">=",
        ),
        "final_improvement_ci_lower": _criterion(
            name="final_improvement_ci_lower",
            passed=math.isfinite(final_delta_ci["lower"])
            and final_delta_ci["lower"] >= float(acceptance["min_final_improvement_ci_lower"]),
            actual=final_delta_ci["lower"],
            expected=float(acceptance["min_final_improvement_ci_lower"]),
            comparator=">=",
        ),
        "best_mode_frequency": _criterion(
            name="best_mode_frequency",
            passed=best_mode_frequency >= float(acceptance["min_best_mode_frequency"]),
            actual=best_mode_frequency,
            expected=float(acceptance["min_best_mode_frequency"]),
            comparator=">=",
        ),
        "stress_defensive_rate": _criterion(
            name="stress_defensive_rate",
            passed=math.isfinite(candidate_mean_defensive)
            and candidate_mean_defensive >= float(acceptance["min_stress_defensive_rate"]),
            actual=candidate_mean_defensive,
            expected=float(acceptance["min_stress_defensive_rate"]),
            comparator=">=",
        ),
        "stress_exploit_rate": _criterion(
            name="stress_exploit_rate",
            passed=math.isfinite(candidate_mean_exploit)
            and candidate_mean_exploit <= float(acceptance["max_stress_exploit_rate"]),
            actual=candidate_mean_exploit,
            expected=float(acceptance["max_stress_exploit_rate"]),
            comparator="<=",
        ),
        "action_diversity": _criterion(
            name="action_diversity",
            passed=math.isfinite(candidate_mean_action_diversity)
            and candidate_mean_action_diversity >= float(acceptance["min_action_diversity"]),
            actual=candidate_mean_action_diversity,
            expected=float(acceptance["min_action_diversity"]),
            comparator=">=",
        ),
        "intake_rate": _criterion(
            name="intake_rate",
            passed=math.isfinite(candidate_mean_intake_rate)
            and candidate_mean_intake_rate <= float(acceptance["max_intake_rate"]),
            actual=candidate_mean_intake_rate,
            expected=float(acceptance["max_intake_rate"]),
            comparator="<=",
        ),
        "navigation_rate": _criterion(
            name="navigation_rate",
            passed=math.isfinite(candidate_mean_navigation_rate)
            and candidate_mean_navigation_rate >= float(acceptance["min_navigation_rate"]),
            actual=candidate_mean_navigation_rate,
            expected=float(acceptance["min_navigation_rate"]),
            comparator=">=",
        ),
        "trace_ablation_spawn_delta": _criterion(
            name="trace_ablation_spawn_delta",
            passed=math.isfinite(candidate_mean_trace_spawn_delta)
            and candidate_mean_trace_spawn_delta >= float(acceptance["min_trace_ablation_spawn_delta"]),
            actual=candidate_mean_trace_spawn_delta,
            expected=float(acceptance["min_trace_ablation_spawn_delta"]),
            comparator=">=",
        ),
        "mean_p_t_min": _criterion(
            name="mean_p_t_min",
            passed=math.isfinite(candidate_agg_mean_p_t)
            and candidate_agg_mean_p_t >= float(acceptance["min_mean_p_t"]),
            actual=candidate_agg_mean_p_t,
            expected=float(acceptance["min_mean_p_t"]),
            comparator=">=",
        ),
        "mean_p_t_max": _criterion(
            name="mean_p_t_max",
            passed=math.isfinite(candidate_agg_mean_p_t)
            and candidate_agg_mean_p_t <= float(acceptance["max_mean_p_t"]),
            actual=candidate_agg_mean_p_t,
            expected=float(acceptance["max_mean_p_t"]),
            comparator="<=",
        ),
        "mean_challenge_fraction_min": _criterion(
            name="mean_challenge_fraction_min",
            passed=math.isfinite(candidate_agg_mean_challenge_fraction)
            and candidate_agg_mean_challenge_fraction >= float(acceptance["min_mean_challenge_fraction"]),
            actual=candidate_agg_mean_challenge_fraction,
            expected=float(acceptance["min_mean_challenge_fraction"]),
            comparator=">=",
        ),
        "mean_challenge_fraction_max": _criterion(
            name="mean_challenge_fraction_max",
            passed=math.isfinite(candidate_agg_mean_challenge_fraction)
            and candidate_agg_mean_challenge_fraction <= float(acceptance["max_mean_challenge_fraction"]),
            actual=candidate_agg_mean_challenge_fraction,
            expected=float(acceptance["max_mean_challenge_fraction"]),
            comparator="<=",
        ),
        "role_switch_events_total": _criterion(
            name="role_switch_events_total",
            passed=math.isfinite(candidate_agg_role_switch_events_total)
            and candidate_agg_role_switch_events_total >= float(acceptance["min_role_switch_events_total"]),
            actual=candidate_agg_role_switch_events_total,
            expected=float(acceptance["min_role_switch_events_total"]),
            comparator=">=",
        ),
        "mean_aux_nontrivial_action_count": _criterion(
            name="mean_aux_nontrivial_action_count",
            passed=math.isfinite(candidate_agg_mean_aux_nontrivial_action_count)
            and candidate_agg_mean_aux_nontrivial_action_count >= float(acceptance["min_mean_aux_nontrivial_action_count"]),
            actual=candidate_agg_mean_aux_nontrivial_action_count,
            expected=float(acceptance["min_mean_aux_nontrivial_action_count"]),
            comparator=">=",
        ),
        "non_degradation_mean_homeostasis": _criterion(
            name="non_degradation_mean_homeostasis",
            passed=math.isfinite(mean_homeostatic_gap)
            and mean_homeostatic_gap <= float(acceptance["max_mean_homeostatic_degradation"]),
            actual=mean_homeostatic_gap,
            expected=float(acceptance["max_mean_homeostatic_degradation"]),
            comparator="<=",
        ),
        "non_degradation_stress_exploit": _criterion(
            name="non_degradation_stress_exploit",
            passed=math.isfinite(exploit_gap)
            and exploit_gap <= float(acceptance["max_stress_exploit_degradation"]),
            actual=exploit_gap,
            expected=float(acceptance["max_stress_exploit_degradation"]),
            comparator="<=",
        ),
    }
    overall_pass = bool(inspected_seeds > 0 and all(item["passed"] for item in criteria.values()))
    report = {
        "experiment_name": experiment_name,
        "track_name": track_name,
        "split": split_name,
        "compare_root": str(compare_root),
        "candidate_mode": candidate_mode,
        "baseline_mode": baseline_mode,
        "required_for_promotion": required_for_promotion,
        "promotion_target": promotion_target,
        "inspected_seeds": inspected_seeds,
        "missing_seeds": missing_seeds,
        "doctor_status": doctor_status,
        "acceptance": acceptance,
        "summary": {
            "candidate": {
                "mean_final_homeostatic_error": None if not math.isfinite(candidate_mean_final) else candidate_mean_final,
                "mean_mean_homeostatic_error": None if not math.isfinite(candidate_mean_mean) else candidate_mean_mean,
                "dead_fraction": None if not math.isfinite(candidate_dead_fraction) else candidate_dead_fraction,
                "mean_stress_defensive_rate": None if not math.isfinite(candidate_mean_defensive) else candidate_mean_defensive,
                "mean_stress_exploit_rate": None if not math.isfinite(candidate_mean_exploit) else candidate_mean_exploit,
                "mean_action_diversity": None
                if not math.isfinite(candidate_mean_action_diversity)
                else candidate_mean_action_diversity,
                "mean_intake_rate": None if not math.isfinite(candidate_mean_intake_rate) else candidate_mean_intake_rate,
                "mean_navigation_rate": None
                if not math.isfinite(candidate_mean_navigation_rate)
                else candidate_mean_navigation_rate,
                "mean_trace_ablation_spawn_delta": None
                if not math.isfinite(candidate_mean_trace_spawn_delta)
                else candidate_mean_trace_spawn_delta,
                "mean_p_t": None if not math.isfinite(candidate_agg_mean_p_t) else candidate_agg_mean_p_t,
                "mean_challenge_fraction": None
                if not math.isfinite(candidate_agg_mean_challenge_fraction)
                else candidate_agg_mean_challenge_fraction,
                "role_switch_events_total": None
                if not math.isfinite(candidate_agg_role_switch_events_total)
                else candidate_agg_role_switch_events_total,
                "mean_aux_nontrivial_action_count": None
                if not math.isfinite(candidate_agg_mean_aux_nontrivial_action_count)
                else candidate_agg_mean_aux_nontrivial_action_count,
            },
            "baseline": {
                "mean_final_homeostatic_error": None if not math.isfinite(baseline_mean_final) else baseline_mean_final,
                "mean_mean_homeostatic_error": None if not math.isfinite(baseline_mean_mean) else baseline_mean_mean,
                "mean_stress_defensive_rate": None if not math.isfinite(baseline_mean_defensive) else baseline_mean_defensive,
                "mean_stress_exploit_rate": None if not math.isfinite(baseline_mean_exploit) else baseline_mean_exploit,
                "mean_action_diversity": None
                if not math.isfinite(baseline_mean_action_diversity)
                else baseline_mean_action_diversity,
                "mean_intake_rate": None if not math.isfinite(baseline_mean_intake_rate) else baseline_mean_intake_rate,
                "mean_navigation_rate": None
                if not math.isfinite(baseline_mean_navigation_rate)
                else baseline_mean_navigation_rate,
                "mean_trace_ablation_spawn_delta": None
                if not math.isfinite(baseline_mean_trace_spawn_delta)
                else baseline_mean_trace_spawn_delta,
                "mean_p_t": None if not math.isfinite(baseline_agg_mean_p_t) else baseline_agg_mean_p_t,
                "mean_challenge_fraction": None
                if not math.isfinite(baseline_agg_mean_challenge_fraction)
                else baseline_agg_mean_challenge_fraction,
                "role_switch_events_total": None
                if not math.isfinite(baseline_agg_role_switch_events_total)
                else baseline_agg_role_switch_events_total,
                "mean_aux_nontrivial_action_count": None
                if not math.isfinite(baseline_agg_mean_aux_nontrivial_action_count)
                else baseline_agg_mean_aux_nontrivial_action_count,
            },
            "best_mode_frequency": best_mode_frequency,
            "final_improvement_vs_baseline": None if not math.isfinite(final_improvement) else final_improvement,
        },
        "statistics": {
            "final_improvement_bootstrap_ci95": {
                "mean": None if not math.isfinite(final_delta_ci["mean"]) else final_delta_ci["mean"],
                "lower": None if not math.isfinite(final_delta_ci["lower"]) else final_delta_ci["lower"],
                "upper": None if not math.isfinite(final_delta_ci["upper"]) else final_delta_ci["upper"],
            }
        },
        "criteria": criteria,
        "overall_pass": overall_pass,
    }
    if seed_errors:
        report["seed_errors"] = seed_errors
        report["next_steps"] = _derive_track_next_steps(report)
        report["next_steps"].insert(0, "Some seed artifacts are missing or unreadable. Re-run sweep before promotion.")
        return report
    report["next_steps"] = _derive_track_next_steps(report)
    return report


def evaluate_contract(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    compare_root: str | Path | None = None,
    doctor_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    base_acceptance = dict(DEFAULT_ACCEPTANCE)
    base_acceptance.update(dict(contract.get("acceptance", {})))
    compare_root_path = Path(compare_root or contract["artifacts"]["compare_root"])
    tracks = _resolve_family_tracks(contract)
    candidate_mode = str(contract["candidate_mode"])
    baseline_mode = str(contract["baseline_mode"])
    doctor_status = None if doctor_report is None else doctor_report.get("status")
    runtime = dict(contract.get("runtime", {}))
    holdout_num_seeds = int(runtime.get("holdout_num_seeds", 0))
    use_holdout = holdout_num_seeds > 0
    require_holdout_for_promotion = bool(base_acceptance.get("require_holdout_for_promotion", False))

    family_reports: dict[str, dict[str, Any]] = {}
    required_track_names: list[str] = []
    eligible_family_tracks: list[str] = []
    blocked_family_tracks: list[str] = []
    inspected_seeds_total = 0
    for track in tracks:
        track_name = track["name"]
        track_root = compare_root_path if track_name == "global" else compare_root_path / track_name
        acceptance = dict(base_acceptance)
        acceptance.update(dict(track["acceptance_overrides"]))
        report = _evaluate_compare_root(
            experiment_name=contract["experiment_name"],
            track_name=track_name,
            split_name="dev",
            compare_root=track_root,
            candidate_mode=candidate_mode,
            baseline_mode=baseline_mode,
            acceptance=acceptance,
            promotion_target=track["promotion_target"],
            required_for_promotion=bool(track["required_for_promotion"]),
            doctor_status=doctor_status,
        )
        if use_holdout:
            holdout_root = track_root / "holdout"
            holdout_report = _evaluate_compare_root(
                experiment_name=contract["experiment_name"],
                track_name=track_name,
                split_name="holdout",
                compare_root=holdout_root,
                candidate_mode=candidate_mode,
                baseline_mode=baseline_mode,
                acceptance=acceptance,
                promotion_target=track["promotion_target"],
                required_for_promotion=bool(track["required_for_promotion"]),
                doctor_status=doctor_status,
            )
            leakage = _seed_leakage_report(track_root, holdout_root)
            report["dev_report"] = {
                "summary": report["summary"],
                "criteria": report["criteria"],
                "overall_pass": report["overall_pass"],
                "inspected_seeds": report["inspected_seeds"],
            }
            report["criteria_dev"] = report["criteria"]
            report["summary_holdout"] = holdout_report["summary"]
            report["criteria_holdout"] = holdout_report["criteria"]
            report["holdout_report"] = holdout_report
            report["seed_leakage"] = leakage
            report["overall_pass_dev"] = bool(report["overall_pass"])
            report["overall_pass_holdout"] = bool(holdout_report["overall_pass"])
            report["overall_pass"] = bool(report["overall_pass_dev"] and report["overall_pass_holdout"] and not leakage["has_overlap"])
            if leakage["has_overlap"]:
                report["next_steps"].insert(
                    0,
                    "Seed leakage detected between dev and holdout splits. Rebuild the contract with disjoint seed ranges before promotion.",
                )
            elif not report["overall_pass_holdout"]:
                report["next_steps"].insert(
                    0,
                    "Holdout split failed despite dev performance. Keep tuning and verify generalization before promotion.",
                )
        family_reports[track_name] = report
        inspected_seeds_total += int(report["inspected_seeds"])
        if use_holdout:
            inspected_seeds_total += int(report.get("holdout_report", {}).get("inspected_seeds", 0))
        if track["required_for_promotion"]:
            required_track_names.append(track_name)
        if report["overall_pass"]:
            eligible_family_tracks.append(track_name)
        elif track["required_for_promotion"]:
            blocked_family_tracks.append(track_name)

    if require_holdout_for_promotion and not use_holdout:
        for track_name in required_track_names:
            report = family_reports[track_name]
            report["overall_pass"] = False
            report["holdout_required"] = True
            report["overall_pass_holdout"] = False
            if "holdout_report" not in report:
                report["holdout_report"] = _empty_track_report(
                    experiment_name=contract["experiment_name"],
                    track_name=track_name,
                    split_name="holdout",
                    compare_root=Path(report["compare_root"]) / "holdout",
                    candidate_mode=candidate_mode,
                    baseline_mode=baseline_mode,
                    acceptance=report.get("acceptance", base_acceptance),
                    promotion_target=report.get("promotion_target", ""),
                    required_for_promotion=bool(report.get("required_for_promotion", False)),
                    doctor_status=doctor_status,
                    evaluation_error="holdout split is required but holdout_num_seeds=0",
                )
            report["next_steps"].insert(
                0,
                "Holdout split is required for promotion. Set runtime.holdout_num_seeds > 0 and rerun.",
            )
            if track_name in eligible_family_tracks:
                eligible_family_tracks.remove(track_name)
            if track_name not in blocked_family_tracks:
                blocked_family_tracks.append(track_name)

    overall_pass = bool(required_track_names and not blocked_family_tracks)
    return {
        "experiment_name": contract["experiment_name"],
        "compare_root": str(compare_root_path),
        "candidate_mode": candidate_mode,
        "baseline_mode": baseline_mode,
        "doctor_status": doctor_status,
        "holdout_enabled": use_holdout,
        "holdout_required_for_promotion": require_holdout_for_promotion,
        "overall_pass": overall_pass,
        "required_family_tracks": required_track_names,
        "eligible_family_tracks": eligible_family_tracks,
        "blocked_family_tracks": blocked_family_tracks,
        "inspected_seeds_total": inspected_seeds_total,
        "family_reports": family_reports,
        "next_steps": _derive_harness_next_steps(
            doctor_status=doctor_status,
            family_reports=family_reports,
            required_track_names=required_track_names,
            overall_pass=overall_pass,
        ),
    }


def _track_compare_root(compare_root: Path, track_name: str) -> Path:
    return compare_root if track_name == "global" else compare_root / track_name


def _run_seed_range(
    *,
    output_root: Path,
    contract: dict[str, Any],
    track: dict[str, Any],
    seed_start: int,
    num_seeds: int,
) -> dict[str, Any]:
    runtime = dict(contract["runtime"])
    per_seed: list[dict[str, Any]] = []
    final_counts: dict[str, int] = {}
    mean_counts: dict[str, int] = {}
    for offset in range(int(num_seeds)):
        seed = int(seed_start) + offset
        seed_root = ensure_dir(output_root / f"seed_{seed}")
        comparison = _compare_one_seed(
            output_root=seed_root,
            seed_catalog=runtime["seed_catalog"],
            steps=int(runtime["steps"]),
            warmup_steps=int(runtime["warmup_steps"]),
            seed=seed,
            lookahead_horizon=int(runtime["lookahead_horizon"]),
            lookahead_discount=float(runtime["lookahead_discount"]),
            resource_patches=int(runtime["resource_patches"]),
            hazard_patches=int(runtime["hazard_patches"]),
            shelter_patches=int(runtime["shelter_patches"]),
            trm_a_checkpoint=runtime["trm_a_checkpoint"],
            trm_b_checkpoint=runtime["trm_b_checkpoint"],
            module_manifest=runtime["module_manifest"],
            policy_mode=str(runtime["policy_mode"]),
            runtime_overrides=track["runtime_overrides"],
            env_overrides=track["env_overrides"],
        )
        best_final = comparison["derived"]["best_mode_by_final_homeostasis"]
        best_mean = comparison["derived"]["best_mode_by_mean_homeostasis"]
        per_seed.append(
            {
                "seed": seed,
                "best_mode_by_final_homeostasis": best_final,
                "best_mode_by_mean_homeostasis": best_mean,
            }
        )
        final_counts[best_final] = final_counts.get(best_final, 0) + 1
        mean_counts[best_mean] = mean_counts.get(best_mean, 0) + 1
    return {
        "seed_start": int(seed_start),
        "num_seeds": int(num_seeds),
        "counts_by_best_final_homeostasis": final_counts,
        "counts_by_best_mean_homeostasis": mean_counts,
        "per_seed": per_seed,
    }


def _run_track_sweep(contract: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(contract["runtime"])
    track_name = track["name"]
    compare_root = ensure_dir(_track_compare_root(Path(contract["artifacts"]["compare_root"]), track_name))
    dev = _run_seed_range(
        output_root=compare_root,
        contract=contract,
        track=track,
        seed_start=int(runtime["seed_start"]),
        num_seeds=int(runtime["num_seeds"]),
    )

    holdout_summary: dict[str, Any] | None = None
    holdout_num_seeds = int(runtime.get("holdout_num_seeds", 0))
    if holdout_num_seeds > 0:
        holdout_root = ensure_dir(compare_root / "holdout")
        holdout_seed_start = int(runtime.get("holdout_seed_start", int(runtime["seed_start"]) + int(runtime["num_seeds"])))
        holdout_summary = _run_seed_range(
            output_root=holdout_root,
            contract=contract,
            track=track,
            seed_start=holdout_seed_start,
            num_seeds=holdout_num_seeds,
        )

    aggregate = {
        "experiment_name": contract["experiment_name"],
        "track_name": track_name,
        "candidate_mode": contract["candidate_mode"],
        "baseline_mode": contract["baseline_mode"],
        "seed_start": int(dev["seed_start"]),
        "num_seeds": int(dev["num_seeds"]),
        "holdout_seed_start": None if holdout_summary is None else int(holdout_summary["seed_start"]),
        "holdout_num_seeds": 0 if holdout_summary is None else int(holdout_summary["num_seeds"]),
        "policy_mode": str(runtime["policy_mode"]),
        "module_manifest": runtime["module_manifest"],
        "runtime_overrides": track["runtime_overrides"],
        "env_overrides": track["env_overrides"],
        "counts_by_best_final_homeostasis": dev["counts_by_best_final_homeostasis"],
        "counts_by_best_mean_homeostasis": dev["counts_by_best_mean_homeostasis"],
        "per_seed": dev["per_seed"],
    }
    save_json(compare_root / "aggregate_summary.json", aggregate)
    if holdout_summary is not None:
        holdout_aggregate = {
            "experiment_name": contract["experiment_name"],
            "track_name": track_name,
            "split": "holdout",
            "candidate_mode": contract["candidate_mode"],
            "baseline_mode": contract["baseline_mode"],
            "seed_start": int(holdout_summary["seed_start"]),
            "num_seeds": int(holdout_summary["num_seeds"]),
            "policy_mode": str(runtime["policy_mode"]),
            "module_manifest": runtime["module_manifest"],
            "runtime_overrides": track["runtime_overrides"],
            "env_overrides": track["env_overrides"],
            "counts_by_best_final_homeostasis": holdout_summary["counts_by_best_final_homeostasis"],
            "counts_by_best_mean_homeostasis": holdout_summary["counts_by_best_mean_homeostasis"],
            "per_seed": holdout_summary["per_seed"],
        }
        save_json(compare_root / "holdout" / "aggregate_summary.json", holdout_aggregate)
    return aggregate


def _run_sweep_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    compare_root = ensure_dir(contract["artifacts"]["compare_root"])
    tracks = _resolve_family_tracks(contract)
    if len(tracks) == 1 and tracks[0]["name"] == "global":
        return _run_track_sweep(contract, tracks[0])

    family_aggregates: dict[str, Any] = {}
    for track in tracks:
        aggregate = _run_track_sweep(contract, track)
        track_root = _track_compare_root(compare_root, track["name"])
        family_aggregates[track["name"]] = {
            "required_for_promotion": bool(track["required_for_promotion"]),
            "promotion_target": track["promotion_target"],
            "aggregate_summary": str(track_root / "aggregate_summary.json"),
            "holdout_aggregate_summary": str(track_root / "holdout" / "aggregate_summary.json")
            if (track_root / "holdout" / "aggregate_summary.json").exists()
            else None,
            "counts_by_best_final_homeostasis": aggregate["counts_by_best_final_homeostasis"],
            "counts_by_best_mean_homeostasis": aggregate["counts_by_best_mean_homeostasis"],
            "runtime_overrides": track["runtime_overrides"],
            "env_overrides": track["env_overrides"],
        }

    aggregate_index = {
        "experiment_name": contract["experiment_name"],
        "candidate_mode": contract["candidate_mode"],
        "baseline_mode": contract["baseline_mode"],
        "compare_root": str(compare_root),
        "family_order": [track["name"] for track in tracks],
        "families": family_aggregates,
    }
    save_json(compare_root / "aggregate_summary.json", aggregate_index)
    return aggregate_index


def run_contract(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    force: bool = False,
    skip_doctor: bool = False,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    artifacts = contract["artifacts"]
    doctor_report = {"status": "skipped", "blocking_issues": [], "warnings": []} if skip_doctor else run_doctor()
    save_json(artifacts["doctor_report"], doctor_report)

    if doctor_report["status"] == "blocked" and not force:
        promotion_decision = build_promotion_decision(
            contract,
            eval_report={
                "doctor_status": "blocked",
                "overall_pass": False,
                "required_family_tracks": [],
                "eligible_family_tracks": [],
                "blocked_family_tracks": [],
                "family_reports": {},
                "next_steps": _derive_harness_next_steps(
                    doctor_status="blocked",
                    family_reports={},
                    required_track_names=[],
                    overall_pass=False,
                ),
            },
            doctor_report=doctor_report,
        )
        run_summary = {
            "experiment_name": contract["experiment_name"],
            "status": "blocked",
            "reason": "doctor_failed",
            "doctor_report": artifacts["doctor_report"],
            "compare_root": artifacts["compare_root"],
            "promotion_decision": artifacts["promotion_decision"],
        }
        save_json(artifacts["run_summary"], run_summary)
        save_json(artifacts["promotion_decision"], promotion_decision)
        save_json(
            artifacts["next_steps"],
            {
                "status": "blocked",
                "next_steps": promotion_decision["next_steps"],
            },
        )
        return run_summary

    aggregate = _run_sweep_from_contract(contract)
    eval_report = evaluate_contract(contract, doctor_report=doctor_report)
    promotion_decision = build_promotion_decision(contract, eval_report=eval_report, doctor_report=doctor_report)
    save_json(artifacts["eval_report"], eval_report)
    save_json(artifacts["promotion_decision"], promotion_decision)
    save_json(
        artifacts["next_steps"],
        {
            "status": promotion_decision["status"],
            "next_steps": promotion_decision["next_steps"],
        },
    )
    run_summary = {
        "experiment_name": contract["experiment_name"],
        "status": "passed" if eval_report["overall_pass"] else "failed",
        "doctor_report": artifacts["doctor_report"],
        "compare_root": artifacts["compare_root"],
        "aggregate_summary": str(Path(artifacts["compare_root"]) / "aggregate_summary.json"),
        "eval_report": artifacts["eval_report"],
        "inspected_seeds_total": eval_report["inspected_seeds_total"],
        "candidate_mode": contract["candidate_mode"],
        "baseline_mode": contract["baseline_mode"],
        "eligible_family_tracks": eval_report["eligible_family_tracks"],
        "blocked_family_tracks": eval_report["blocked_family_tracks"],
        "promotion_decision": artifacts["promotion_decision"],
    }
    if "counts_by_best_final_homeostasis" in aggregate:
        run_summary["counts_by_best_final_homeostasis"] = aggregate["counts_by_best_final_homeostasis"]
    save_json(artifacts["run_summary"], run_summary)
    return run_summary



def run_tuning_loop(
    contract_or_path: dict[str, Any] | str | Path,
    *,
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
    return _run_tuning_loop(
        contract_or_path,
        run_contract_fn=run_contract,
        max_rounds=max_rounds,
        min_primary_improvement=min_primary_improvement,
        stagnation_patience=stagnation_patience,
        max_updates_per_round=max_updates_per_round,
        promotion_streak_required=promotion_streak_required,
        resume=resume,
        apply_best=apply_best,
        force=force,
        skip_doctor=skip_doctor,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Harness workflow for ERIE/TRM experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check environment and test-running prerequisites.")
    doctor_parser.add_argument("--output", default=None)

    plan_parser = subparsers.add_parser("plan", help="Write a file-based experiment contract.")
    plan_parser.add_argument("--output-root", required=True)
    plan_parser.add_argument("--experiment-name", required=True)
    plan_parser.add_argument("--candidate-mode", default="analytic__module_primary")
    plan_parser.add_argument("--baseline-mode", default="analytic__analytic")
    plan_parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    plan_parser.add_argument("--seed-start", type=int, default=20260318)
    plan_parser.add_argument("--num-seeds", type=int, default=5)
    plan_parser.add_argument("--holdout-seed-start", type=int, default=None)
    plan_parser.add_argument("--holdout-num-seeds", type=int, default=0)
    plan_parser.add_argument("--steps", type=int, default=24)
    plan_parser.add_argument("--warmup-steps", type=int, default=4)
    plan_parser.add_argument("--lookahead-horizon", type=int, default=2)
    plan_parser.add_argument("--lookahead-discount", type=float, default=0.85)
    add_environment_config_args(plan_parser)
    plan_parser.add_argument("--trm-a-checkpoint", default=None)
    plan_parser.add_argument("--trm-b-checkpoint", default=None)
    plan_parser.add_argument("--module-manifest", default=None)
    plan_parser.add_argument("--policy-mode", default="closed_loop")
    plan_parser.add_argument("--require-holdout-for-promotion", action="store_true")
    plan_parser.add_argument("--families", nargs="*", default=None)

    run_parser = subparsers.add_parser("run", help="Run the full harness from a contract file.")
    run_parser.add_argument("--contract", required=True)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--skip-doctor", action="store_true")

    tune_parser = subparsers.add_parser("tune", help="Run bounded auto-tuning rounds from a base contract.")
    tune_parser.add_argument("--contract", required=True)
    tune_parser.add_argument("--max-rounds", type=int, default=3)
    tune_parser.add_argument("--min-primary-improvement", type=float, default=0.005)
    tune_parser.add_argument("--stagnation-patience", type=int, default=1)
    tune_parser.add_argument("--max-updates-per-round", type=int, default=3)
    tune_parser.add_argument("--promotion-streak-required", type=int, default=2)
    tune_parser.add_argument("--resume", action="store_true")
    tune_parser.add_argument("--apply-best", action="store_true")
    tune_parser.add_argument("--force", action="store_true")
    tune_parser.add_argument("--skip-doctor", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an existing compare root against a contract.")
    eval_parser.add_argument("--contract", required=True)
    eval_parser.add_argument("--compare-root", default=None)
    eval_parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "doctor":
        report = run_doctor()
        if args.output:
            save_json(args.output, report)
        else:
            import json

            print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.command == "plan":
        cli_env_config = environment_config_from_args(args)
        contract = build_experiment_contract(
            output_root=args.output_root,
            experiment_name=args.experiment_name,
            candidate_mode=args.candidate_mode,
            baseline_mode=args.baseline_mode,
            seed_catalog=args.seed_catalog,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            holdout_seed_start=args.holdout_seed_start,
            holdout_num_seeds=args.holdout_num_seeds,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            lookahead_horizon=args.lookahead_horizon,
            lookahead_discount=args.lookahead_discount,
            resource_patches=cli_env_config.resource_patches,
            hazard_patches=cli_env_config.hazard_patches,
            shelter_patches=cli_env_config.shelter_patches,
            trm_a_checkpoint=args.trm_a_checkpoint,
            trm_b_checkpoint=args.trm_b_checkpoint,
            module_manifest=args.module_manifest,
            policy_mode=args.policy_mode,
            require_holdout_for_promotion=args.require_holdout_for_promotion,
            families=args.families,
        )
        save_json(contract["artifacts"]["contract"], contract)
        print(f"wrote harness contract: {contract['artifacts']['contract']}")
        return

    if args.command == "run":
        summary = run_contract(args.contract, force=args.force, skip_doctor=args.skip_doctor)
        print(f"wrote harness summary: {summary['status']} -> {load_json(args.contract)['artifacts']['run_summary']}")
        return

    if args.command == "tune":
        summary = run_tuning_loop(
            args.contract,
            max_rounds=args.max_rounds,
            min_primary_improvement=args.min_primary_improvement,
            stagnation_patience=args.stagnation_patience,
            max_updates_per_round=args.max_updates_per_round,
            promotion_streak_required=args.promotion_streak_required,
            resume=args.resume,
            apply_best=args.apply_best,
            force=args.force,
            skip_doctor=args.skip_doctor,
        )
        print(f"wrote tuning summary: {summary['status']} -> {summary['tune_summary_path']}")
        return

    report = evaluate_contract(args.contract, compare_root=args.compare_root)
    output_path = args.output or load_json(args.contract)["artifacts"]["eval_report"]
    save_json(output_path, report)
    print(f"wrote harness evaluation: {output_path}")


if __name__ == "__main__":
    main()
