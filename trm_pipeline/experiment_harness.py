from __future__ import annotations

import argparse
import copy
import importlib
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .common import ensure_dir, load_json, save_json
from .evaluate_trm_as_behavior import evaluate_compare_root
from .sweep_trm_va_modes import _compare_one_seed


DEFAULT_ACCEPTANCE = {
    "max_mean_final_homeostatic_error": 0.35,
    "max_mean_mean_homeostatic_error": 0.30,
    "max_dead_fraction": 0.20,
    "min_final_improvement_vs_baseline": 0.00,
    "min_best_mode_frequency": 0.60,
    "min_stress_defensive_rate": 0.40,
    "max_stress_exploit_rate": 0.60,
    "stress_threshold": 0.35,
}


DEFAULT_FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "energy_starved": {
        "runtime_overrides": {
            "G0": 0.28,
            "B0": 0.76,
            "move_step": 2.8,
            "aperture_gain": 0.52,
        },
        "env_overrides": {
            "resource_patches": 1,
            "hazard_patches": 2,
            "shelter_patches": 1,
            "resource_regen": 0.0018,
            "hazard_drift_sigma": 0.0010,
            "toxicity_drift_sigma": 0.0010,
        },
        "acceptance_overrides": {
            "max_mean_final_homeostatic_error": 0.30,
            "max_mean_mean_homeostatic_error": 0.27,
            "max_stress_exploit_rate": 0.55,
        },
        "promotion_target": "Promote only if the candidate recovers low-energy states faster than baseline without collapsing into exploit-only intake behavior.",
    },
    "toxic_band": {
        "runtime_overrides": {
            "G0": 0.58,
            "B0": 0.58,
            "aperture_gain": 0.33,
            "aperture_width_deg": 70.0,
        },
        "env_overrides": {
            "resource_patches": 2,
            "hazard_patches": 5,
            "shelter_patches": 1,
            "field_sigma_min": 3.5,
            "field_sigma_max": 7.5,
            "toxicity_drift_sigma": 0.0020,
        },
        "acceptance_overrides": {
            "max_mean_final_homeostatic_error": 0.38,
            "min_stress_defensive_rate": 0.55,
            "max_stress_exploit_rate": 0.45,
        },
        "promotion_target": "Promote only if the candidate remains defensive under high-toxicity contact and beats baseline on final homeostatic error.",
    },
    "fragile_boundary": {
        "runtime_overrides": {
            "G0": 0.54,
            "B0": 0.30,
            "aperture_gain": 0.25,
            "aperture_width_deg": 60.0,
            "observation_noise": 0.015,
        },
        "env_overrides": {
            "resource_patches": 2,
            "hazard_patches": 4,
            "shelter_patches": 1,
            "shelter_stability": 0.82,
        },
        "acceptance_overrides": {
            "max_mean_final_homeostatic_error": 0.34,
            "max_dead_fraction": 0.10,
            "min_stress_defensive_rate": 0.50,
        },
        "promotion_target": "Promote only if the candidate preserves boundary integrity with lower death rate than baseline in fragile-boundary cases.",
    },
    "vent_edge": {
        "runtime_overrides": {
            "G0": 0.46,
            "B0": 0.62,
            "move_step": 2.6,
            "aperture_gain": 0.40,
        },
        "env_overrides": {
            "resource_patches": 4,
            "hazard_patches": 5,
            "shelter_patches": 0,
            "field_sigma_min": 4.0,
            "field_sigma_max": 8.5,
            "resource_regen": 0.0030,
        },
        "acceptance_overrides": {
            "max_mean_final_homeostatic_error": 0.36,
            "max_mean_mean_homeostatic_error": 0.32,
            "min_best_mode_frequency": 0.50,
        },
        "promotion_target": "Promote only if the candidate sustains homeostasis near resource-hazard edges across multiple seeds, not just at the final step.",
    },
    "uncertain_corridor": {
        "runtime_overrides": {
            "G0": 0.42,
            "B0": 0.56,
            "observation_noise": 0.035,
            "epistemic_scale": 1.35,
            "aperture_width_deg": 85.0,
        },
        "env_overrides": {
            "resource_patches": 3,
            "hazard_patches": 3,
            "shelter_patches": 2,
            "hazard_drift_sigma": 0.0020,
            "toxicity_drift_sigma": 0.0020,
            "shelter_stability": 0.82,
        },
        "acceptance_overrides": {
            "max_mean_final_homeostatic_error": 0.37,
            "max_mean_mean_homeostatic_error": 0.33,
        },
        "promotion_target": "Promote only if the candidate maintains an advantage over baseline when observation noise and epistemic ambiguity are both elevated.",
    },
}

DEFAULT_FAMILY_ORDER = tuple(DEFAULT_FAMILY_PROFILES.keys())

TUNABLE_RUNTIME_PARAMS: dict[str, dict[str, float | int | str]] = {
    "aperture_gain": {"min": 0.15, "max": 0.80, "default": 0.45, "kind": "float"},
    "aperture_width_deg": {"min": 40.0, "max": 110.0, "default": 70.0, "kind": "float"},
    "action_gating_blend": {"min": 0.10, "max": 0.90, "default": 0.35, "kind": "float"},
    "move_step": {"min": 1.0, "max": 3.0, "default": 2.0, "kind": "float"},
    "lookahead_horizon": {"min": 1, "max": 4, "default": 2, "kind": "int"},
    "lookahead_discount": {"min": 0.70, "max": 0.98, "default": 0.85, "kind": "float"},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifact_paths_for_output_root(output_root: str | Path) -> dict[str, str]:
    out_root = Path(output_root)
    return {
        "contract": str(out_root / "contract.json"),
        "doctor_report": str(out_root / "doctor_report.json"),
        "compare_root": str(out_root / "compare"),
        "eval_report": str(out_root / "eval_report.json"),
        "next_steps": str(out_root / "next_steps.json"),
        "promotion_decision": str(out_root / "promotion_decision.json"),
        "run_summary": str(out_root / "run_summary.json"),
    }


def _path_is_within(path: str | Path, parent: str | Path) -> bool:
    try:
        Path(path).absolute().relative_to(Path(parent).absolute())
    except ValueError:
        return False
    return True


def _coerce_contract(contract_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(contract_or_path, (str, Path)):
        return load_json(contract_or_path)
    return dict(contract_or_path)


def _clone_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(contract)


def _normalize_track(track: dict[str, Any]) -> dict[str, Any]:
    name = str(track["name"])
    return {
        "name": name,
        "required_for_promotion": bool(track.get("required_for_promotion", True)),
        "promotion_target": str(track.get("promotion_target", f"Promote `{name}` only after it beats the baseline gate.")),
        "runtime_overrides": dict(track.get("runtime_overrides", {})),
        "env_overrides": dict(track.get("env_overrides", {})),
        "acceptance_overrides": dict(track.get("acceptance_overrides", {})),
    }


def _default_family_tracks(families: list[str] | None = None) -> list[dict[str, Any]]:
    family_names = list(families) if families else list(DEFAULT_FAMILY_ORDER)
    tracks: list[dict[str, Any]] = []
    for name in family_names:
        if name not in DEFAULT_FAMILY_PROFILES:
            raise SystemExit(f"unknown family track: {name}")
        profile = DEFAULT_FAMILY_PROFILES[name]
        tracks.append(
            _normalize_track(
                {
                    "name": name,
                    "required_for_promotion": True,
                    "promotion_target": profile["promotion_target"],
                    "runtime_overrides": profile.get("runtime_overrides", {}),
                    "env_overrides": profile.get("env_overrides", {}),
                    "acceptance_overrides": profile.get("acceptance_overrides", {}),
                }
            )
        )
    return tracks


def _resolve_family_tracks(contract: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tracks = contract.get("family_tracks")
    if not raw_tracks:
        return [
            _normalize_track(
                {
                    "name": "global",
                    "required_for_promotion": True,
                    "promotion_target": "Promote the candidate globally only after it passes the baseline gate on the full sweep.",
                }
            )
        ]
    return [_normalize_track(track) for track in raw_tracks]


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
    acceptance: dict[str, Any] | None = None,
    family_tracks: list[dict[str, Any]] | None = None,
    families: list[str] | None = None,
) -> dict[str, Any]:
    out_root = ensure_dir(output_root)
    merged_acceptance = dict(DEFAULT_ACCEPTANCE)
    if acceptance:
        merged_acceptance.update(acceptance)
    resolved_family_tracks = (
        [_normalize_track(track) for track in family_tracks]
        if family_tracks is not None
        else _default_family_tracks(families)
    )
    artifacts = _artifact_paths_for_output_root(out_root)
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
    for track_name, report in family_reports.items():
        track_status = "promote" if report.get("overall_pass") else "hold"
        track_decisions.append(
            {
                "track_name": track_name,
                "status": track_status,
                "required_for_promotion": bool(report.get("required_for_promotion", False)),
                "promotion_target": report.get("promotion_target"),
                "failed_criteria": _failed_criteria(report),
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
        "next_steps": list(eval_report.get("next_steps", [])),
        "track_decisions": track_decisions,
    }


def _evaluate_compare_root(
    *,
    experiment_name: str,
    track_name: str,
    compare_root: Path,
    candidate_mode: str,
    baseline_mode: str,
    acceptance: dict[str, Any],
    promotion_target: str,
    required_for_promotion: bool,
    doctor_status: str | None,
) -> dict[str, Any]:
    aggregate = load_json(compare_root / "aggregate_summary.json")
    stress_threshold = float(acceptance.get("stress_threshold", DEFAULT_ACCEPTANCE["stress_threshold"]))

    candidate_final: list[float] = []
    candidate_mean: list[float] = []
    candidate_dead: list[float] = []
    candidate_defensive: list[float] = []
    candidate_exploit: list[float] = []
    baseline_final: list[float] = []
    baseline_mean: list[float] = []
    best_candidate_count = 0
    inspected_seeds = 0
    missing_seeds: list[int] = []

    for row in aggregate.get("per_seed", []):
        seed = int(row["seed"])
        seed_root = compare_root / f"seed_{seed}"
        comparison = load_json(seed_root / "comparison_summary.json")
        diagnostics = evaluate_compare_root(seed_root, stress_threshold=stress_threshold)
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
        baseline_final.append(float(baseline_summary["final_homeostatic_error"]))
        baseline_mean.append(float(baseline_summary["mean_homeostatic_error"]))
        best_candidate_count += int(comparison["derived"]["best_mode_by_final_homeostasis"] == candidate_mode)
        inspected_seeds += 1

    candidate_mean_final = _mean(candidate_final)
    candidate_mean_mean = _mean(candidate_mean)
    candidate_dead_fraction = _mean(candidate_dead)
    candidate_mean_defensive = _mean(candidate_defensive)
    candidate_mean_exploit = _mean(candidate_exploit)
    baseline_mean_final = _mean(baseline_final)
    baseline_mean_mean = _mean(baseline_mean)
    final_improvement = baseline_mean_final - candidate_mean_final
    best_mode_frequency = float(best_candidate_count / max(inspected_seeds, 1))

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
    }
    overall_pass = bool(inspected_seeds > 0 and all(item["passed"] for item in criteria.values()))
    report = {
        "experiment_name": experiment_name,
        "track_name": track_name,
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
            },
            "baseline": {
                "mean_final_homeostatic_error": None if not math.isfinite(baseline_mean_final) else baseline_mean_final,
                "mean_mean_homeostatic_error": None if not math.isfinite(baseline_mean_mean) else baseline_mean_mean,
            },
            "best_mode_frequency": best_mode_frequency,
            "final_improvement_vs_baseline": None if not math.isfinite(final_improvement) else final_improvement,
        },
        "criteria": criteria,
        "overall_pass": overall_pass,
    }
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
            compare_root=track_root,
            candidate_mode=candidate_mode,
            baseline_mode=baseline_mode,
            acceptance=acceptance,
            promotion_target=track["promotion_target"],
            required_for_promotion=bool(track["required_for_promotion"]),
            doctor_status=doctor_status,
        )
        family_reports[track_name] = report
        inspected_seeds_total += int(report["inspected_seeds"])
        if track["required_for_promotion"]:
            required_track_names.append(track_name)
        if report["overall_pass"]:
            eligible_family_tracks.append(track_name)
        elif track["required_for_promotion"]:
            blocked_family_tracks.append(track_name)

    overall_pass = bool(required_track_names and not blocked_family_tracks)
    return {
        "experiment_name": contract["experiment_name"],
        "compare_root": str(compare_root_path),
        "candidate_mode": candidate_mode,
        "baseline_mode": baseline_mode,
        "doctor_status": doctor_status,
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


def _run_track_sweep(contract: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(contract["runtime"])
    track_name = track["name"]
    compare_root = ensure_dir(_track_compare_root(Path(contract["artifacts"]["compare_root"]), track_name))
    per_seed: list[dict[str, Any]] = []
    final_counts: dict[str, int] = {}
    mean_counts: dict[str, int] = {}

    for offset in range(int(runtime["num_seeds"])):
        seed = int(runtime["seed_start"]) + offset
        seed_root = ensure_dir(compare_root / f"seed_{seed}")
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

    aggregate = {
        "experiment_name": contract["experiment_name"],
        "track_name": track_name,
        "candidate_mode": contract["candidate_mode"],
        "baseline_mode": contract["baseline_mode"],
        "seed_start": int(runtime["seed_start"]),
        "num_seeds": int(runtime["num_seeds"]),
        "policy_mode": str(runtime["policy_mode"]),
        "module_manifest": runtime["module_manifest"],
        "runtime_overrides": track["runtime_overrides"],
        "env_overrides": track["env_overrides"],
        "counts_by_best_final_homeostasis": final_counts,
        "counts_by_best_mean_homeostasis": mean_counts,
        "per_seed": per_seed,
    }
    save_json(compare_root / "aggregate_summary.json", aggregate)
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


def _required_track_names_from_eval(eval_report: dict[str, Any]) -> list[str]:
    required = [str(name) for name in eval_report.get("required_family_tracks", [])]
    if required:
        return required
    family_reports = dict(eval_report.get("family_reports", {}))
    return [
        str(name)
        for name, report in family_reports.items()
        if bool(report.get("required_for_promotion", False))
    ]


def _primary_score_from_eval_report(eval_report: dict[str, Any]) -> float:
    required_tracks = _required_track_names_from_eval(eval_report)
    family_reports = dict(eval_report.get("family_reports", {}))
    values: list[float] = []
    for track_name in required_tracks:
        candidate = (
            family_reports.get(track_name, {})
            .get("summary", {})
            .get("candidate", {})
        )
        raw_value = candidate.get("mean_final_homeostatic_error")
        if raw_value is None:
            continue
        value = float(raw_value)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("inf")
    return float(sum(values) / len(values))


def _failed_criteria_counter(eval_report: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    required_tracks = set(_required_track_names_from_eval(eval_report))
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


def _propose_tuning_updates(
    eval_report: dict[str, Any],
    *,
    max_updates_per_round: int = 3,
) -> list[dict[str, Any]]:
    failed = _failed_criteria_counter(eval_report)
    if not failed:
        return []
    priority_rules: list[tuple[str, list[tuple[str, float]]]] = [
        ("dead_fraction", [("move_step", -0.20), ("aperture_gain", -0.03)]),
        ("stress_exploit_rate", [("aperture_gain", -0.04), ("action_gating_blend", 0.05)]),
        ("stress_defensive_rate", [("aperture_width_deg", -4.0), ("action_gating_blend", 0.05)]),
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


def _clamp_tunable_value(param_name: str, value: float | int) -> float | int:
    spec = TUNABLE_RUNTIME_PARAMS[param_name]
    lower = float(spec["min"])
    upper = float(spec["max"])
    clamped = min(max(float(value), lower), upper)
    if spec["kind"] == "int":
        return int(round(clamped))
    return float(clamped)


def _default_tunable_value(param_name: str) -> float | int:
    spec = TUNABLE_RUNTIME_PARAMS[param_name]
    if spec["kind"] == "int":
        return int(spec["default"])
    return float(spec["default"])


def _target_runtime_overrides(
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


def _apply_tuning_updates(
    contract: dict[str, Any],
    proposals: list[dict[str, Any]],
    *,
    blocked_tracks: list[str],
) -> list[dict[str, Any]]:
    targets = _target_runtime_overrides(contract, blocked_tracks=blocked_tracks)
    applied: list[dict[str, Any]] = []
    for track_name, runtime_overrides in targets:
        for proposal in proposals:
            param_name = str(proposal["param"])
            if param_name not in TUNABLE_RUNTIME_PARAMS:
                continue
            raw_before = runtime_overrides.get(param_name, _default_tunable_value(param_name))
            before = _clamp_tunable_value(param_name, raw_before)
            candidate = float(before) + float(proposal["delta"])
            after = _clamp_tunable_value(param_name, candidate)
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


def _build_tuning_round_contract(
    base_contract: dict[str, Any],
    *,
    autotune_root: Path,
    round_index: int,
) -> dict[str, Any]:
    round_root = ensure_dir(autotune_root / f"round_{round_index:02d}")
    round_contract = _clone_contract(base_contract)
    round_contract["experiment_name"] = f"{base_contract['experiment_name']}__tune_r{round_index:02d}"
    round_contract["output_root"] = str(round_root)
    round_contract["artifacts"] = _artifact_paths_for_output_root(round_root)
    return round_contract


def run_tuning_loop(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    max_rounds: int = 3,
    min_primary_improvement: float = 0.005,
    stagnation_patience: int = 1,
    max_updates_per_round: int = 3,
    force: bool = False,
    skip_doctor: bool = False,
) -> dict[str, Any]:
    base_contract = _coerce_contract(contract_or_path)
    working_contract = _clone_contract(base_contract)
    autotune_root = ensure_dir(Path(base_contract["output_root"]) / "autotune")
    save_json(autotune_root / "base_contract_snapshot.json", base_contract)

    history: list[dict[str, Any]] = []
    best_primary_score = float("inf")
    best_round: int | None = None
    no_progress_rounds = 0
    status = "max_rounds"

    for round_index in range(1, int(max_rounds) + 1):
        round_contract = _build_tuning_round_contract(
            working_contract,
            autotune_root=autotune_root,
            round_index=round_index,
        )
        save_json(round_contract["artifacts"]["contract"], round_contract)
        run_summary = run_contract(round_contract, force=force, skip_doctor=skip_doctor)

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

        primary_score = _primary_score_from_eval_report(eval_report)
        improved = (
            math.isfinite(primary_score)
            and (not math.isfinite(best_primary_score) or primary_score <= best_primary_score - float(min_primary_improvement))
        )
        if improved:
            best_primary_score = primary_score
            best_round = round_index
            no_progress_rounds = 0
        else:
            no_progress_rounds += 1

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
        }

        if run_summary.get("status") == "blocked":
            status = "blocked"
            history.append(round_log)
            break
        if bool(eval_report.get("overall_pass", False)):
            status = "promote"
            history.append(round_log)
            break
        if round_index >= int(max_rounds):
            status = "max_rounds"
            history.append(round_log)
            break
        if no_progress_rounds > int(stagnation_patience):
            status = "no_progress"
            history.append(round_log)
            break

        required_set = set(_required_track_names_from_eval(eval_report))
        blocked_required = [
            str(name)
            for name in eval_report.get("blocked_family_tracks", [])
            if str(name) in required_set
        ]
        proposals = _propose_tuning_updates(eval_report, max_updates_per_round=max_updates_per_round)
        applied = _apply_tuning_updates(
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

    summary_path = autotune_root / "tune_summary.json"
    summary = {
        "experiment_name": base_contract["experiment_name"],
        "status": status,
        "autotune_root": str(autotune_root),
        "rounds_run": len(history),
        "best_round": best_round,
        "best_primary_score": None if not math.isfinite(best_primary_score) else float(best_primary_score),
        "max_rounds": int(max_rounds),
        "min_primary_improvement": float(min_primary_improvement),
        "stagnation_patience": int(stagnation_patience),
        "max_updates_per_round": int(max_updates_per_round),
        "rounds": history,
    }
    save_json(summary_path, summary)
    summary["tune_summary_path"] = str(summary_path)
    return summary


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
    plan_parser.add_argument("--steps", type=int, default=24)
    plan_parser.add_argument("--warmup-steps", type=int, default=4)
    plan_parser.add_argument("--lookahead-horizon", type=int, default=2)
    plan_parser.add_argument("--lookahead-discount", type=float, default=0.85)
    plan_parser.add_argument("--resource-patches", type=int, default=3)
    plan_parser.add_argument("--hazard-patches", type=int, default=3)
    plan_parser.add_argument("--shelter-patches", type=int, default=1)
    plan_parser.add_argument("--trm-a-checkpoint", default=None)
    plan_parser.add_argument("--trm-b-checkpoint", default=None)
    plan_parser.add_argument("--module-manifest", default=None)
    plan_parser.add_argument("--policy-mode", default="closed_loop")
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
        contract = build_experiment_contract(
            output_root=args.output_root,
            experiment_name=args.experiment_name,
            candidate_mode=args.candidate_mode,
            baseline_mode=args.baseline_mode,
            seed_catalog=args.seed_catalog,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            lookahead_horizon=args.lookahead_horizon,
            lookahead_discount=args.lookahead_discount,
            resource_patches=args.resource_patches,
            hazard_patches=args.hazard_patches,
            shelter_patches=args.shelter_patches,
            trm_a_checkpoint=args.trm_a_checkpoint,
            trm_b_checkpoint=args.trm_b_checkpoint,
            module_manifest=args.module_manifest,
            policy_mode=args.policy_mode,
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
