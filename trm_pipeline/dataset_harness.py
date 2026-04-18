from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from .common import append_jsonl, ensure_dir, load_json, load_jsonl, save_json, save_jsonl
from .erie_runtime import ACTIONS, EnvironmentConfig, RuntimeConfig
from .experiment_harness import run_doctor
from .lenia_data import RolloutConfig, generate_rollouts
from .prepare_trm_va_data import EPISODE_FAMILIES, prepare_trm_va_cache


DATASET_KIND_PASSIVE = "passive_lenia_pretrain"
DATASET_KIND_AGENTIC = "agentic_bootstrap"
DATASET_KINDS = (DATASET_KIND_PASSIVE, DATASET_KIND_AGENTIC)
DATASET_PRESETS = (
    "passive_canonical",
    "agentic_canonical",
    "passive_production",
    "agentic_production",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_registry_path() -> Path:
    return _repo_root() / "artifacts" / "dataset_registry.jsonl"


def _model_registry_path() -> Path:
    return _repo_root() / "artifacts" / "model_registry.jsonl"


def _slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "dataset"


def _preset_overrides(preset: str) -> dict[str, Any]:
    if preset == "passive_canonical":
        return {
            "dataset_kind": DATASET_KIND_PASSIVE,
            "num_seeds": 200,
            "warmup_steps": 32,
            "record_steps": 256,
            "image_size": 64,
            "target_radius": 12,
            "root_seed": 20260306,
            "acceptance": {
                "min_successful_episodes": 180,
                "min_unique_seed_count": 180,
                "min_mean_num_frames": 257,
                "min_perturbed_fraction": 0.15,
                "max_perturbed_fraction": 0.35,
                "min_regime_diversity": 2,
                "max_single_regime_fraction": 0.90,
            },
        }
    if preset == "passive_production":
        return {
            "dataset_kind": DATASET_KIND_PASSIVE,
            "num_seeds": 240,
            "warmup_steps": 32,
            "record_steps": 256,
            "image_size": 64,
            "target_radius": 12,
            "root_seed": 20260306,
            "acceptance": {
                "min_successful_episodes": 192,
                "min_unique_seed_count": 192,
                "min_mean_num_frames": 257,
                "min_train_episodes": 144,
                "min_val_episodes": 24,
                "min_test_episodes": 24,
                "require_seed_disjoint_splits": True,
                "min_effective_one_step_samples": 49_000,
                "min_perturbed_fraction": 0.10,
                "max_perturbed_fraction": 0.35,
                "min_regime_diversity": 2,
                "max_single_regime_fraction": 0.85,
                "max_stable_fraction": 0.85,
                "max_chaotic_fraction": 0.85,
            },
        }
    if preset == "agentic_canonical":
        return {
            "dataset_kind": DATASET_KIND_AGENTIC,
            "episodes": 64,
            "steps": 32,
            "warmup_steps": 4,
            "runtime_seed": 20260318,
            "image_size": 64,
            "target_radius": 12,
            "defensive_family_bias": 1.0,
            "required_families": list(EPISODE_FAMILIES),
            "acceptance": {
                "min_retained_episodes": 64,
                "min_distinct_families": len(EPISODE_FAMILIES),
                "max_rejected_fraction": 0.50,
                "min_aggregate_policy_entropy": 0.95,
                "min_action_entropy_ratio": 0.45,
                "max_aggregate_dominant_action_fraction": 0.70,
                "min_recovery_fraction": 0.55,
                "min_stress_defensive_fraction": 0.40,
                "max_stress_exploit_fraction": 0.55,
            },
        }
    if preset == "agentic_production":
        return {
            "dataset_kind": DATASET_KIND_AGENTIC,
            "episodes": 192,
            "steps": 24,
            "warmup_steps": 4,
            "runtime_seed": 20260318,
            "image_size": 64,
            "target_radius": 12,
            "defensive_family_bias": 1.0,
            "policy_mode_mix": {
                "closed_loop": 0.50,
                "random": 0.30,
                "no_action": 0.20,
            },
            "required_families": list(EPISODE_FAMILIES),
            "acceptance": {
                "min_retained_episodes": 192,
                "min_source_seed_count": 96,
                "min_effective_step_samples": 4500,
                "required_families": list(EPISODE_FAMILIES),
                "min_distinct_families": len(EPISODE_FAMILIES),
                "required_policy_modes": ["closed_loop", "random", "no_action"],
                "policy_mode_share_min": {
                    "closed_loop": 0.45,
                    "random": 0.20,
                    "no_action": 0.10,
                },
                "policy_mode_share_max": {
                    "closed_loop": 0.60,
                    "random": 0.40,
                    "no_action": 0.30,
                },
                "min_non_dead_fraction": 0.55,
                "max_non_dead_fraction": 0.80,
                "max_rejected_fraction": 0.50,
                "min_aggregate_policy_entropy": 1.0,
                "min_action_entropy_ratio": 0.45,
                "require_all_actions_present": True,
                "max_aggregate_dominant_action_fraction": 0.55,
                "min_dead_dominant_action_diversity": 2,
                "min_recovery_fraction": 0.55,
                "min_stress_defensive_fraction": 0.40,
                "max_stress_exploit_fraction": 0.55,
            },
        }
    raise SystemExit(f"unknown dataset preset: {preset}")


def _coerce_contract(contract_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(contract_or_path, (str, Path)):
        contract = load_json(contract_or_path)
    else:
        contract = dict(contract_or_path)
    artifacts = dict(contract.get("artifacts", {}))
    if "contract" in artifacts:
        out_root = Path(artifacts["contract"]).parent
    elif "dataset_root" in artifacts:
        out_root = Path(artifacts["dataset_root"]).parent
    else:
        out_root = Path(".")
    artifacts.setdefault("contract", str(out_root / "contract.json"))
    artifacts.setdefault("dataset_root", str(out_root / "dataset"))
    artifacts.setdefault("doctor_report", str(out_root / "doctor_report.json"))
    artifacts.setdefault("eval_report", str(out_root / "dataset_eval_report.json"))
    artifacts.setdefault("collection_decision", str(out_root / "collection_decision.json"))
    artifacts.setdefault("training_plan", str(out_root / "training_plan.json"))
    artifacts.setdefault("training_run_report", str(out_root / "training_run_report.json"))
    artifacts.setdefault("model_eval_report", str(out_root / "model_eval_report.json"))
    artifacts.setdefault("promotion_decision", str(out_root / "promotion_decision.json"))
    artifacts.setdefault("revision_search_report", str(out_root / "revision_search_report.json"))
    artifacts.setdefault("gpu_handoff_report", str(out_root / "gpu_handoff_report.json"))
    artifacts.setdefault("gpu_handoff_script", str(out_root / "run_external_gpu.sh"))
    artifacts.setdefault("external_finalize_report", str(out_root / "external_finalize_report.json"))
    artifacts.setdefault("revised_contract", str(out_root / "revised_contract.json"))
    artifacts.setdefault("campaign_until_report", str(out_root / "campaign_until_report.json"))
    artifacts.setdefault("next_steps", str(out_root / "next_steps.json"))
    artifacts.setdefault("run_summary", str(out_root / "run_summary.json"))
    artifacts.setdefault("registry_path", str(_dataset_registry_path()))
    artifacts.setdefault("model_registry_path", str(_model_registry_path()))
    contract["artifacts"] = artifacts
    contract.setdefault("model_acceptance", _default_model_acceptance(contract["dataset_kind"]))
    return contract

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


def _entropy_ratio(counts: list[int]) -> float:
    total = int(sum(int(value) for value in counts))
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = [float(value) / float(total) for value in counts if int(value) > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return float(entropy / math.log(len(counts)))


def _parse_mode_mix(entries: list[str] | None) -> dict[str, float] | None:
    if not entries:
        return None
    parsed: dict[str, float] = {}
    for entry in entries:
        if "=" not in str(entry):
            raise SystemExit(f"invalid policy mode mix entry: {entry}")
        name, raw_value = str(entry).split("=", 1)
        name = name.strip()
        if name not in {"closed_loop", "random", "no_action"}:
            raise SystemExit(f"unknown policy mode: {name}")
        parsed[name] = float(raw_value)
    return parsed or None


def _allocate_mode_counts(total: int, requested_mix: dict[str, float] | None) -> dict[str, int]:
    if total <= 0:
        return {}
    mix = dict(requested_mix or {"closed_loop": 1.0})
    raw_total = float(sum(float(value) for value in mix.values()))
    if raw_total <= 0.0:
        mix = {"closed_loop": 1.0}
        raw_total = 1.0
    normalized = {name: float(value) / raw_total for name, value in mix.items()}
    names = list(normalized.keys())
    counts: dict[str, int] = {}
    assigned = 0
    for name in names[:-1]:
        count = int(math.floor(total * normalized[name]))
        counts[name] = count
        assigned += count
    counts[names[-1]] = max(0, total - assigned)
    for name in names:
        if counts.get(name, 0) <= 0 and total >= len(names):
            donor = max(counts, key=counts.get)
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[name] = 1
    return {name: count for name, count in counts.items() if count > 0}


def _weighted_mean(rows: list[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in rows:
        if value is None or weight <= 0 or not math.isfinite(float(value)):
            continue
        numerator += float(value) * int(weight)
        denominator += int(weight)
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _default_passive_acceptance(num_seeds: int, record_steps: int) -> dict[str, Any]:
    return {
        "min_successful_episodes": max(1, int(math.floor(num_seeds * 0.90))),
        "min_unique_seed_count": max(1, int(math.floor(num_seeds * 0.90))),
        "min_mean_num_frames": int(record_steps + 1),
        "min_perturbed_fraction": 0.0 if num_seeds < 10 else 0.10,
        "max_perturbed_fraction": 0.50,
        "min_regime_diversity": 1 if num_seeds < 12 else 2,
        "max_single_regime_fraction": 1.0 if num_seeds < 12 else 0.95,
    }


def _default_agentic_acceptance(
    num_episodes: int,
    required_families: list[str],
    min_episode_policy_entropy: float,
    max_dominant_action_fraction: float,
) -> dict[str, Any]:
    return {
        "min_retained_episodes": int(num_episodes),
        "required_families": list(required_families),
        "min_distinct_families": min(len(required_families), int(num_episodes)),
        "max_rejected_fraction": 0.75,
        "min_aggregate_policy_entropy": float(min_episode_policy_entropy) * 0.90,
        "min_action_entropy_ratio": 0.20 if num_episodes < 10 else 0.45,
        "max_aggregate_dominant_action_fraction": min(0.98, float(max_dominant_action_fraction)),
        "min_recovery_fraction": 0.45 if num_episodes < 10 else 0.55,
        "min_stress_defensive_fraction": 0.20 if num_episodes < 10 else 0.40,
        "max_stress_exploit_fraction": 0.85 if num_episodes < 10 else 0.55,
    }


def _default_model_acceptance(dataset_kind: str) -> dict[str, float]:
    if dataset_kind == DATASET_KIND_PASSIVE:
        return {
            "trm_a.max_val_nmse": 0.90,
            "trm_a.max_rollout_nmse_8": 8.0,
            "trm_a.min_improvement_over_baseline": 0.0,
            "trm_b.min_boundary_iou": 0.10,
            "trm_b.min_nucleus_separation": 0.01,
        }
    return {
        "trm_vm.max_val_homeostatic_error_mae": 0.12,
        "trm_vm.min_val_viability_risk_auroc": 0.60,
        "trm_vm.min_val_margin_to_failure_corr": 0.10,
        "trm_as.min_val_pairwise_ranking_accuracy": 0.55,
        "trm_as.min_val_policy_entropy_mean": 1.0,
        "trm_as.max_val_action_collapse_rate": 0.65,
        "trm_ag.max_val_inhibition_mask_mae": 0.35,
        "trm_ag.min_val_control_mode_accuracy": 0.35,
        "trm_ag.max_val_gated_policy_kl": 1.0,
        "trm_bp.max_val_permeability_patch_mae": 0.20,
        "trm_bp.min_val_mode_accuracy": 0.35,
        "trm_mc.max_val_context_state_loss": 0.25,
        "trm_mc.min_val_action_bias_alignment": 0.20,
    }


def _summarize_advisory(criteria: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failed = [name for name, criterion in criteria.items() if not criterion.get("passed", False)]
    passed = len(criteria) - len(failed)
    total = len(criteria)
    score = float(passed / total) if total > 0 else 1.0
    return {
        "criteria": criteria,
        "failed_criteria": failed,
        "passed_count": passed,
        "total_count": total,
        "score": score,
        "status": "aligned" if not failed else ("near" if score >= 0.70 else "far"),
    }


def _coerce_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _build_passive_ideal_advisory(summary: dict[str, Any]) -> dict[str, Any]:
    successful_episodes = int(summary.get("successful_episodes", 0))
    stable_fraction = _coerce_float_or_nan(summary.get("stable_fraction", float("nan")))
    chaotic_fraction = _coerce_float_or_nan(summary.get("chaotic_fraction", float("nan")))
    regime_counts = dict(summary.get("regime_counts", {}))
    regime_diversity = len([name for name, count in regime_counts.items() if int(count) > 0])
    criteria = {
        "regime_diversity_target": _criterion(
            name="regime_diversity_target",
            passed=regime_diversity >= 2,
            actual=float(regime_diversity),
            expected=2.0,
            comparator=">=",
        ),
        "stable_not_dominant": _criterion(
            name="stable_not_dominant",
            passed=not math.isfinite(stable_fraction) or stable_fraction <= 0.80,
            actual=stable_fraction,
            expected=0.80,
            comparator="<=",
        ),
        "chaotic_present": _criterion(
            name="chaotic_present",
            passed=not math.isfinite(chaotic_fraction) or chaotic_fraction >= 0.10 or successful_episodes < 12,
            actual=chaotic_fraction,
            expected=0.10,
            comparator=">=",
        ),
    }
    return _summarize_advisory(criteria)


def _build_agentic_ideal_advisory(summary: dict[str, Any]) -> dict[str, Any]:
    retained_episodes = int(summary.get("retained_episodes", 0))
    distinct_families = int(summary.get("distinct_families", 0))
    aggregate_policy_entropy = _coerce_float_or_nan(summary.get("aggregate_policy_entropy_mean", float("nan")))
    action_entropy_ratio = _coerce_float_or_nan(summary.get("action_entropy_ratio", float("nan")))
    dominant_action_fraction = _coerce_float_or_nan(summary.get("dominant_action_fraction", float("nan")))
    recovery_fraction = _coerce_float_or_nan(summary.get("aggregate_recovery_fraction_mean", float("nan")))
    stress_defensive_fraction = _coerce_float_or_nan(summary.get("aggregate_stress_defensive_fraction_mean", float("nan")))
    stress_exploit_fraction = _coerce_float_or_nan(summary.get("aggregate_stress_exploit_fraction_mean", float("nan")))
    non_dead_fraction = _coerce_float_or_nan(summary.get("non_dead_fraction", float("nan")))
    all_actions_present = bool(summary.get("all_actions_present", False))
    criteria = {
        "family_diversity_target": _criterion(
            name="family_diversity_target",
            passed=distinct_families >= min(len(EPISODE_FAMILIES), 5),
            actual=float(distinct_families),
            expected=float(min(len(EPISODE_FAMILIES), 5)),
            comparator=">=",
        ),
        "aggregate_policy_entropy_target": _criterion(
            name="aggregate_policy_entropy_target",
            passed=math.isfinite(aggregate_policy_entropy) and aggregate_policy_entropy >= 1.10,
            actual=aggregate_policy_entropy,
            expected=1.10,
            comparator=">=",
        ),
        "action_entropy_ratio_target": _criterion(
            name="action_entropy_ratio_target",
            passed=math.isfinite(action_entropy_ratio) and action_entropy_ratio >= 0.55,
            actual=action_entropy_ratio,
            expected=0.55,
            comparator=">=",
        ),
        "dominant_action_fraction_target": _criterion(
            name="dominant_action_fraction_target",
            passed=math.isfinite(dominant_action_fraction) and dominant_action_fraction <= 0.45,
            actual=dominant_action_fraction,
            expected=0.45,
            comparator="<=",
        ),
        "recovery_fraction_target": _criterion(
            name="recovery_fraction_target",
            passed=math.isfinite(recovery_fraction) and recovery_fraction >= 0.60,
            actual=recovery_fraction,
            expected=0.60,
            comparator=">=",
        ),
        "stress_defensive_fraction_target": _criterion(
            name="stress_defensive_fraction_target",
            passed=math.isfinite(stress_defensive_fraction) and stress_defensive_fraction >= 0.55,
            actual=stress_defensive_fraction,
            expected=0.55,
            comparator=">=",
        ),
        "stress_exploit_fraction_target": _criterion(
            name="stress_exploit_fraction_target",
            passed=math.isfinite(stress_exploit_fraction) and stress_exploit_fraction <= 0.35,
            actual=stress_exploit_fraction,
            expected=0.35,
            comparator="<=",
        ),
        "balanced_success_failure_target": _criterion(
            name="balanced_success_failure_target",
            passed=math.isfinite(non_dead_fraction) and 0.40 <= non_dead_fraction <= 0.80,
            actual=non_dead_fraction,
            expected=0.40,
            comparator="range",
        ),
        "all_actions_present_target": _criterion(
            name="all_actions_present_target",
            passed=all_actions_present,
            actual=float(sum(1 for action in ACTIONS if int(dict(summary.get("aggregate_action_counts", {})).get(action, 0)) > 0)),
            expected=float(len(ACTIONS)),
            comparator="==",
        ),
        "retained_volume_target": _criterion(
            name="retained_volume_target",
            passed=retained_episodes >= 64,
            actual=float(retained_episodes),
            expected=64.0,
            comparator=">=",
        ),
    }
    return _summarize_advisory(criteria)


def build_dataset_contract(
    *,
    output_root: str | Path,
    dataset_name: str,
    dataset_kind: str,
    seed_catalog: str = "data/lenia_official/animals2d_seeds.json",
    num_seeds: int = 200,
    warmup_steps: int = 32,
    record_steps: int = 256,
    image_size: int = 64,
    target_radius: int = 12,
    root_seed: int = 20260306,
    episodes: int = 16,
    steps: int = 32,
    runtime_seed: int = 20260318,
    target_band_weight: float = 0.0,
    target_g_overshoot_weight: float = 0.0,
    defensive_family_bias: float = 0.0,
    policy_mode_mix: dict[str, float] | None = None,
    max_attempt_multiplier: int = 4,
    min_episode_samples: int = 8,
    min_distinct_actions: int = 2,
    max_dominant_action_fraction: float = 0.90,
    min_episode_policy_entropy: float = 0.90,
    required_families: list[str] | None = None,
    acceptance: dict[str, Any] | None = None,
    registry_path: str | Path | None = None,
    model_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    if dataset_kind not in DATASET_KINDS:
        raise SystemExit(f"unknown dataset kind: {dataset_kind}")

    out_root = ensure_dir(output_root)
    dataset_root = out_root / "dataset"
    artifacts = {
        "contract": str(out_root / "contract.json"),
        "dataset_root": str(dataset_root),
        "doctor_report": str(out_root / "doctor_report.json"),
        "eval_report": str(out_root / "dataset_eval_report.json"),
        "collection_decision": str(out_root / "collection_decision.json"),
        "training_plan": str(out_root / "training_plan.json"),
        "training_run_report": str(out_root / "training_run_report.json"),
        "model_eval_report": str(out_root / "model_eval_report.json"),
        "promotion_decision": str(out_root / "promotion_decision.json"),
        "revision_search_report": str(out_root / "revision_search_report.json"),
        "gpu_handoff_report": str(out_root / "gpu_handoff_report.json"),
        "gpu_handoff_script": str(out_root / "run_external_gpu.sh"),
        "external_finalize_report": str(out_root / "external_finalize_report.json"),
        "revised_contract": str(out_root / "revised_contract.json"),
        "campaign_until_report": str(out_root / "campaign_until_report.json"),
        "next_steps": str(out_root / "next_steps.json"),
        "run_summary": str(out_root / "run_summary.json"),
        "registry_path": str(registry_path or _dataset_registry_path()),
        "model_registry_path": str(
            model_registry_path
            or (Path(registry_path).with_name("model_registry.jsonl") if registry_path else _model_registry_path())
        ),
    }

    if dataset_kind == DATASET_KIND_PASSIVE:
        generator = {
            "kind": dataset_kind,
            "target_modules": ["trm_a", "trm_b"],
            "seed_catalog": seed_catalog,
            "config": {
                "num_seeds": int(num_seeds),
                "warmup_steps": int(warmup_steps),
                "record_steps": int(record_steps),
                "image_size": int(image_size),
                "target_radius": int(target_radius),
                "root_seed": int(root_seed),
            },
        }
        merged_acceptance = _default_passive_acceptance(int(num_seeds), int(record_steps))
    else:
        required = list(required_families or EPISODE_FAMILIES)
        generator = {
            "kind": dataset_kind,
            "target_modules": ["trm_vm", "trm_as"],
            "seed_catalog": seed_catalog,
            "config": {
                "episodes": int(episodes),
                "steps": int(steps),
                "warmup_steps": int(warmup_steps),
                "seed": int(runtime_seed),
                "image_size": int(image_size),
                "target_radius": int(target_radius),
                "target_band_weight": float(target_band_weight),
                "target_g_overshoot_weight": float(target_g_overshoot_weight),
                "defensive_family_bias": float(defensive_family_bias),
                "policy_mode_mix": dict(policy_mode_mix or {"closed_loop": 1.0}),
                "max_attempt_multiplier": int(max_attempt_multiplier),
                "min_episode_samples": int(min_episode_samples),
                "min_distinct_actions": int(min_distinct_actions),
                "max_dominant_action_fraction": float(max_dominant_action_fraction),
                "min_episode_policy_entropy": float(min_episode_policy_entropy),
            },
            "required_families": required,
        }
        merged_acceptance = _default_agentic_acceptance(
            int(episodes),
            required,
            float(min_episode_policy_entropy),
            float(max_dominant_action_fraction),
        )

    if acceptance:
        merged_acceptance.update(dict(acceptance))

    return {
        "version": 1,
        "dataset_name": dataset_name,
        "dataset_kind": dataset_kind,
        "purpose": (
            "Passive Lenia rollout collection for TRM-A/TRM-B."
            if dataset_kind == DATASET_KIND_PASSIVE
            else "Agent-like bootstrap collection for TRM-Vm/TRM-As."
        ),
        "generator": generator,
        "acceptance": merged_acceptance,
        "model_acceptance": _default_model_acceptance(dataset_kind),
        "artifacts": artifacts,
    }


def _evaluate_passive_dataset(contract: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    acceptance = dict(contract["acceptance"])
    summary = load_json(dataset_root / "summary.json")
    rows = load_jsonl(dataset_root / "manifest.jsonl")
    successful_episodes = int(len(rows))
    unique_seed_count = len({str(row["seed_id"]) for row in rows})
    effective_one_step_samples = int(sum(max(0, int(row.get("num_frames", 0)) - 1) for row in rows))
    mean_num_frames = (
        float(sum(int(row.get("num_frames", 0)) for row in rows) / len(rows))
        if rows
        else float("nan")
    )
    perturbed_count = sum(1 for row in rows if row.get("perturb_mode") is not None)
    perturbed_fraction = float(perturbed_count / max(successful_episodes, 1))
    regime_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    split_seed_sets: dict[str, set[str]] = {}
    for row in rows:
        regime = str(row.get("regime", "unknown"))
        split = str(row.get("split", "unknown"))
        seed_id = str(row.get("seed_id", "unknown"))
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
        split_seed_sets.setdefault(split, set()).add(seed_id)
    regime_diversity = len([name for name, count in regime_counts.items() if count > 0])
    dominant_regime_fraction = (
        float(max(regime_counts.values()) / max(successful_episodes, 1))
        if regime_counts
        else float("nan")
    )
    stable_fraction = float(regime_counts.get("stable", 0) / max(successful_episodes, 1))
    chaotic_fraction = float(regime_counts.get("chaotic", 0) / max(successful_episodes, 1))
    seed_to_split_count: dict[str, int] = {}
    for split_name, seed_ids in split_seed_sets.items():
        for seed_id in seed_ids:
            seed_to_split_count[seed_id] = int(seed_to_split_count.get(seed_id, 0) + 1)
    seed_disjoint_splits = not any(count > 1 for count in seed_to_split_count.values())

    criteria = {
        "successful_episodes": _criterion(
            name="successful_episodes",
            passed=successful_episodes >= int(acceptance["min_successful_episodes"]),
            actual=float(successful_episodes),
            expected=float(acceptance["min_successful_episodes"]),
            comparator=">=",
        ),
        "unique_seed_count": _criterion(
            name="unique_seed_count",
            passed=unique_seed_count >= int(acceptance["min_unique_seed_count"]),
            actual=float(unique_seed_count),
            expected=float(acceptance["min_unique_seed_count"]),
            comparator=">=",
        ),
        "mean_num_frames": _criterion(
            name="mean_num_frames",
            passed=math.isfinite(mean_num_frames) and mean_num_frames >= float(acceptance["min_mean_num_frames"]),
            actual=mean_num_frames,
            expected=float(acceptance["min_mean_num_frames"]),
            comparator=">=",
        ),
        "perturbed_fraction_min": _criterion(
            name="perturbed_fraction_min",
            passed=perturbed_fraction >= float(acceptance["min_perturbed_fraction"]),
            actual=perturbed_fraction,
            expected=float(acceptance["min_perturbed_fraction"]),
            comparator=">=",
        ),
        "perturbed_fraction_max": _criterion(
            name="perturbed_fraction_max",
            passed=perturbed_fraction <= float(acceptance["max_perturbed_fraction"]),
            actual=perturbed_fraction,
            expected=float(acceptance["max_perturbed_fraction"]),
            comparator="<=",
        ),
        "regime_diversity": _criterion(
            name="regime_diversity",
            passed=regime_diversity >= int(acceptance["min_regime_diversity"]),
            actual=float(regime_diversity),
            expected=float(acceptance["min_regime_diversity"]),
            comparator=">=",
        ),
        "dominant_regime_fraction": _criterion(
            name="dominant_regime_fraction",
            passed=math.isfinite(dominant_regime_fraction)
            and dominant_regime_fraction <= float(acceptance["max_single_regime_fraction"]),
            actual=dominant_regime_fraction,
            expected=float(acceptance["max_single_regime_fraction"]),
            comparator="<=",
        ),
    }
    if "min_train_episodes" in acceptance:
        criteria["train_episodes"] = _criterion(
            name="train_episodes",
            passed=int(split_counts.get("train", 0)) >= int(acceptance["min_train_episodes"]),
            actual=float(split_counts.get("train", 0)),
            expected=float(acceptance["min_train_episodes"]),
            comparator=">=",
        )
    if "min_val_episodes" in acceptance:
        criteria["val_episodes"] = _criterion(
            name="val_episodes",
            passed=int(split_counts.get("val", 0)) >= int(acceptance["min_val_episodes"]),
            actual=float(split_counts.get("val", 0)),
            expected=float(acceptance["min_val_episodes"]),
            comparator=">=",
        )
    if "min_test_episodes" in acceptance:
        criteria["test_episodes"] = _criterion(
            name="test_episodes",
            passed=int(split_counts.get("test", 0)) >= int(acceptance["min_test_episodes"]),
            actual=float(split_counts.get("test", 0)),
            expected=float(acceptance["min_test_episodes"]),
            comparator=">=",
        )
    if acceptance.get("require_seed_disjoint_splits") is not None:
        criteria["seed_disjoint_splits"] = _criterion(
            name="seed_disjoint_splits",
            passed=bool(seed_disjoint_splits) == bool(acceptance["require_seed_disjoint_splits"]),
            actual=float(1 if seed_disjoint_splits else 0),
            expected=float(1 if acceptance["require_seed_disjoint_splits"] else 0),
            comparator="==",
        )
    if "min_effective_one_step_samples" in acceptance:
        criteria["effective_one_step_samples"] = _criterion(
            name="effective_one_step_samples",
            passed=effective_one_step_samples >= int(acceptance["min_effective_one_step_samples"]),
            actual=float(effective_one_step_samples),
            expected=float(acceptance["min_effective_one_step_samples"]),
            comparator=">=",
        )
    if "max_stable_fraction" in acceptance:
        criteria["stable_fraction"] = _criterion(
            name="stable_fraction",
            passed=stable_fraction <= float(acceptance["max_stable_fraction"]),
            actual=stable_fraction,
            expected=float(acceptance["max_stable_fraction"]),
            comparator="<=",
        )
    if "max_chaotic_fraction" in acceptance:
        criteria["chaotic_fraction"] = _criterion(
            name="chaotic_fraction",
            passed=chaotic_fraction <= float(acceptance["max_chaotic_fraction"]),
            actual=chaotic_fraction,
            expected=float(acceptance["max_chaotic_fraction"]),
            comparator="<=",
        )
    overall_pass = bool(rows and all(item["passed"] for item in criteria.values()))
    ideal_advisory = _build_passive_ideal_advisory(
        {
            "successful_episodes": successful_episodes,
            "stable_fraction": stable_fraction,
            "chaotic_fraction": chaotic_fraction,
            "regime_counts": regime_counts,
        }
    )
    return {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "dataset_root": str(dataset_root),
        "target_modules": list(contract["generator"]["target_modules"]),
        "summary": {
            "num_selected_seeds": int(summary.get("num_selected_seeds", 0)),
            "successful_episodes": successful_episodes,
            "unique_seed_count": unique_seed_count,
            "mean_num_frames": None if not math.isfinite(mean_num_frames) else mean_num_frames,
            "effective_one_step_samples": effective_one_step_samples,
            "perturbed_fraction": perturbed_fraction,
            "regime_counts": regime_counts,
            "split_counts": split_counts,
            "seed_disjoint_splits": seed_disjoint_splits,
            "stable_fraction": stable_fraction,
            "chaotic_fraction": chaotic_fraction,
            "dominant_regime_fraction": None if not math.isfinite(dominant_regime_fraction) else dominant_regime_fraction,
        },
        "criteria": criteria,
        "ideal_advisory": ideal_advisory,
        "overall_pass": overall_pass,
    }


def _evaluate_agentic_dataset(contract: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    acceptance = dict(contract["acceptance"])
    summary = load_json(dataset_root / "summary.json")
    rows = load_jsonl(dataset_root / "manifest.jsonl")
    retained_episodes = int(summary.get("retained_episodes", len(rows)))
    rejected_episodes = int(summary.get("rejected_episodes", 0))
    attempted_episodes = int(summary.get("attempted_episodes", retained_episodes + rejected_episodes))
    source_seed_count = len({str(row.get("seed_id", "unknown")) for row in rows})
    effective_step_samples = int(sum(int(row.get("num_samples", 0)) for row in rows))
    family_counts = dict(summary.get("family_counts", {}))
    required_families = list(acceptance.get("required_families", []))
    missing_required_families = [family for family in required_families if int(family_counts.get(family, 0)) <= 0]
    distinct_families = len([family for family, count in family_counts.items() if int(count) > 0])
    rejected_fraction = float(rejected_episodes / max(attempted_episodes, 1))
    policy_mode_counts = dict(summary.get("policy_mode_counts", {}))
    required_policy_modes = list(acceptance.get("required_policy_modes", []))
    missing_policy_modes = [mode for mode in required_policy_modes if int(policy_mode_counts.get(mode, 0)) <= 0]
    policy_mode_shares = {
        mode: float(int(policy_mode_counts.get(mode, 0)) / max(retained_episodes, 1))
        for mode in policy_mode_counts
    }
    aggregate_policy_entropy = float(summary.get("aggregate_policy_entropy_mean", float("nan")))
    action_counts = dict(summary.get("aggregate_action_counts", {}))
    action_values = [int(action_counts.get(action, 0)) for action in ACTIONS]
    action_entropy_ratio = _entropy_ratio(action_values)
    total_actions = max(1, sum(action_values))
    dominant_action_fraction = float(max(action_values) / total_actions) if action_values else float("nan")
    recovery_fraction = float(summary.get("aggregate_recovery_fraction_mean", float("nan")))
    stress_defensive_fraction = float(summary.get("aggregate_stress_defensive_fraction_mean", float("nan")))
    stress_exploit_fraction = float(summary.get("aggregate_stress_exploit_fraction_mean", float("nan")))
    terminal_dead_count = int(sum(1 for row in rows if bool(row.get("terminal_dead", row.get("quality", {}).get("terminal_dead", False)))))
    non_dead_fraction = float((retained_episodes - terminal_dead_count) / max(retained_episodes, 1))
    all_actions_present = all(int(action_counts.get(action, 0)) > 0 for action in ACTIONS)
    dead_dominant_actions = {
        str(row.get("quality", {}).get("dominant_action"))
        for row in rows
        if bool(row.get("terminal_dead", row.get("quality", {}).get("terminal_dead", False)))
    }
    dead_dominant_action_diversity = len([value for value in dead_dominant_actions if value and value != "None"])

    criteria = {
        "retained_episodes": _criterion(
            name="retained_episodes",
            passed=retained_episodes >= int(acceptance["min_retained_episodes"]),
            actual=float(retained_episodes),
            expected=float(acceptance["min_retained_episodes"]),
            comparator=">=",
        ),
        "distinct_families": _criterion(
            name="distinct_families",
            passed=distinct_families >= int(acceptance["min_distinct_families"]),
            actual=float(distinct_families),
            expected=float(acceptance["min_distinct_families"]),
            comparator=">=",
        ),
        "required_family_coverage": _criterion(
            name="required_family_coverage",
            passed=not missing_required_families,
            actual=float(len(required_families) - len(missing_required_families)),
            expected=float(len(required_families)),
            comparator="==",
        ),
        "rejected_fraction": _criterion(
            name="rejected_fraction",
            passed=rejected_fraction <= float(acceptance["max_rejected_fraction"]),
            actual=rejected_fraction,
            expected=float(acceptance["max_rejected_fraction"]),
            comparator="<=",
        ),
        "aggregate_policy_entropy": _criterion(
            name="aggregate_policy_entropy",
            passed=math.isfinite(aggregate_policy_entropy)
            and aggregate_policy_entropy >= float(acceptance["min_aggregate_policy_entropy"]),
            actual=aggregate_policy_entropy,
            expected=float(acceptance["min_aggregate_policy_entropy"]),
            comparator=">=",
        ),
        "action_entropy_ratio": _criterion(
            name="action_entropy_ratio",
            passed=action_entropy_ratio >= float(acceptance["min_action_entropy_ratio"]),
            actual=action_entropy_ratio,
            expected=float(acceptance["min_action_entropy_ratio"]),
            comparator=">=",
        ),
        "dominant_action_fraction": _criterion(
            name="dominant_action_fraction",
            passed=math.isfinite(dominant_action_fraction)
            and dominant_action_fraction <= float(acceptance["max_aggregate_dominant_action_fraction"]),
            actual=dominant_action_fraction,
            expected=float(acceptance["max_aggregate_dominant_action_fraction"]),
            comparator="<=",
        ),
        "recovery_fraction": _criterion(
            name="recovery_fraction",
            passed=math.isfinite(recovery_fraction)
            and recovery_fraction >= float(acceptance["min_recovery_fraction"]),
            actual=recovery_fraction,
            expected=float(acceptance["min_recovery_fraction"]),
            comparator=">=",
        ),
        "stress_defensive_fraction": _criterion(
            name="stress_defensive_fraction",
            passed=math.isfinite(stress_defensive_fraction)
            and stress_defensive_fraction >= float(acceptance["min_stress_defensive_fraction"]),
            actual=stress_defensive_fraction,
            expected=float(acceptance["min_stress_defensive_fraction"]),
            comparator=">=",
        ),
        "stress_exploit_fraction": _criterion(
            name="stress_exploit_fraction",
            passed=math.isfinite(stress_exploit_fraction)
            and stress_exploit_fraction <= float(acceptance["max_stress_exploit_fraction"]),
            actual=stress_exploit_fraction,
            expected=float(acceptance["max_stress_exploit_fraction"]),
            comparator="<=",
        ),
    }
    if "min_source_seed_count" in acceptance:
        criteria["source_seed_count"] = _criterion(
            name="source_seed_count",
            passed=source_seed_count >= int(acceptance["min_source_seed_count"]),
            actual=float(source_seed_count),
            expected=float(acceptance["min_source_seed_count"]),
            comparator=">=",
        )
    if "min_effective_step_samples" in acceptance:
        criteria["effective_step_samples"] = _criterion(
            name="effective_step_samples",
            passed=effective_step_samples >= int(acceptance["min_effective_step_samples"]),
            actual=float(effective_step_samples),
            expected=float(acceptance["min_effective_step_samples"]),
            comparator=">=",
        )
    if required_policy_modes:
        criteria["required_policy_mode_coverage"] = _criterion(
            name="required_policy_mode_coverage",
            passed=not missing_policy_modes,
            actual=float(len(required_policy_modes) - len(missing_policy_modes)),
            expected=float(len(required_policy_modes)),
            comparator="==",
        )
    for mode, minimum in dict(acceptance.get("policy_mode_share_min", {})).items():
        criteria[f"policy_mode_share_min::{mode}"] = _criterion(
            name=f"policy_mode_share_min::{mode}",
            passed=float(policy_mode_shares.get(mode, 0.0)) >= float(minimum),
            actual=float(policy_mode_shares.get(mode, 0.0)),
            expected=float(minimum),
            comparator=">=",
        )
    for mode, maximum in dict(acceptance.get("policy_mode_share_max", {})).items():
        criteria[f"policy_mode_share_max::{mode}"] = _criterion(
            name=f"policy_mode_share_max::{mode}",
            passed=float(policy_mode_shares.get(mode, 0.0)) <= float(maximum),
            actual=float(policy_mode_shares.get(mode, 0.0)),
            expected=float(maximum),
            comparator="<=",
        )
    if "min_non_dead_fraction" in acceptance:
        criteria["non_dead_fraction_min"] = _criterion(
            name="non_dead_fraction_min",
            passed=non_dead_fraction >= float(acceptance["min_non_dead_fraction"]),
            actual=non_dead_fraction,
            expected=float(acceptance["min_non_dead_fraction"]),
            comparator=">=",
        )
    if "max_non_dead_fraction" in acceptance:
        criteria["non_dead_fraction_max"] = _criterion(
            name="non_dead_fraction_max",
            passed=non_dead_fraction <= float(acceptance["max_non_dead_fraction"]),
            actual=non_dead_fraction,
            expected=float(acceptance["max_non_dead_fraction"]),
            comparator="<=",
        )
    if acceptance.get("require_all_actions_present"):
        criteria["all_actions_present"] = _criterion(
            name="all_actions_present",
            passed=all_actions_present,
            actual=float(sum(1 for action in ACTIONS if int(action_counts.get(action, 0)) > 0)),
            expected=float(len(ACTIONS)),
            comparator="==",
        )
    if "min_dead_dominant_action_diversity" in acceptance:
        criteria["dead_dominant_action_diversity"] = _criterion(
            name="dead_dominant_action_diversity",
            passed=dead_dominant_action_diversity >= int(acceptance["min_dead_dominant_action_diversity"]),
            actual=float(dead_dominant_action_diversity),
            expected=float(acceptance["min_dead_dominant_action_diversity"]),
            comparator=">=",
        )
    overall_pass = bool(rows and all(item["passed"] for item in criteria.values()))
    ideal_advisory = _build_agentic_ideal_advisory(
        {
            "retained_episodes": retained_episodes,
            "distinct_families": distinct_families,
            "aggregate_policy_entropy_mean": None if not math.isfinite(aggregate_policy_entropy) else aggregate_policy_entropy,
            "aggregate_action_counts": action_counts,
            "all_actions_present": all_actions_present,
            "action_entropy_ratio": action_entropy_ratio,
            "dominant_action_fraction": None if not math.isfinite(dominant_action_fraction) else dominant_action_fraction,
            "non_dead_fraction": non_dead_fraction,
            "aggregate_recovery_fraction_mean": None if not math.isfinite(recovery_fraction) else recovery_fraction,
            "aggregate_stress_defensive_fraction_mean": None if not math.isfinite(stress_defensive_fraction) else stress_defensive_fraction,
            "aggregate_stress_exploit_fraction_mean": None if not math.isfinite(stress_exploit_fraction) else stress_exploit_fraction,
        }
    )
    return {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "dataset_root": str(dataset_root),
        "target_modules": list(contract["generator"]["target_modules"]),
        "summary": {
            "retained_episodes": retained_episodes,
            "rejected_episodes": rejected_episodes,
            "attempted_episodes": attempted_episodes,
            "source_seed_count": source_seed_count,
            "effective_step_samples": effective_step_samples,
            "family_counts": family_counts,
            "required_families": required_families,
            "missing_required_families": missing_required_families,
            "distinct_families": distinct_families,
            "rejected_fraction": rejected_fraction,
            "policy_mode_counts": policy_mode_counts,
            "required_policy_modes": required_policy_modes,
            "missing_policy_modes": missing_policy_modes,
            "policy_mode_shares": policy_mode_shares,
            "aggregate_policy_entropy_mean": None if not math.isfinite(aggregate_policy_entropy) else aggregate_policy_entropy,
            "aggregate_action_counts": action_counts,
            "all_actions_present": all_actions_present,
            "action_entropy_ratio": action_entropy_ratio,
            "dominant_action_fraction": None if not math.isfinite(dominant_action_fraction) else dominant_action_fraction,
            "terminal_dead_count": terminal_dead_count,
            "non_dead_fraction": non_dead_fraction,
            "dead_dominant_action_diversity": dead_dominant_action_diversity,
            "aggregate_recovery_fraction_mean": None if not math.isfinite(recovery_fraction) else recovery_fraction,
            "aggregate_stress_defensive_fraction_mean": None if not math.isfinite(stress_defensive_fraction) else stress_defensive_fraction,
            "aggregate_stress_exploit_fraction_mean": None if not math.isfinite(stress_exploit_fraction) else stress_exploit_fraction,
        },
        "criteria": criteria,
        "ideal_advisory": ideal_advisory,
        "overall_pass": overall_pass,
    }


def evaluate_dataset_contract(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    dataset_root: str | Path | None = None,
    doctor_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    dataset_root_path = Path(dataset_root or contract["artifacts"]["dataset_root"])
    if contract["dataset_kind"] == DATASET_KIND_PASSIVE:
        report = _evaluate_passive_dataset(contract, dataset_root_path)
    else:
        report = _evaluate_agentic_dataset(contract, dataset_root_path)
    report["doctor_status"] = None if doctor_report is None else doctor_report.get("status")
    return report


def _failed_criteria(eval_report: dict[str, Any]) -> list[str]:
    return [name for name, criterion in eval_report.get("criteria", {}).items() if not criterion.get("passed", False)]


def _derive_dataset_next_steps(contract: dict[str, Any], eval_report: dict[str, Any]) -> list[str]:
    doctor_status = eval_report.get("doctor_status")
    if doctor_status == "blocked":
        return [
            "Fix the runtime environment before collecting more data.",
            "Rerun `./scripts/bootstrap_env.sh` and `make doctor` before the next dataset collection attempt.",
        ]
    if eval_report.get("overall_pass"):
        steps = [
            "Approve this dataset collection for downstream training: " + ", ".join(eval_report["target_modules"]) + ".",
            "Launch the corresponding training run and keep the contract alongside the collected dataset for replayability.",
        ]
        advisory = dict(eval_report.get("ideal_advisory", {}))
        if advisory.get("status") != "aligned":
            failed_advisory = list(advisory.get("failed_criteria", []))
            if failed_advisory:
                steps.append(
                    "Production minimum is satisfied, but ideal-data alignment is still incomplete. Next tightening targets: "
                    + ", ".join(failed_advisory[:4])
                    + "."
                )
        return steps

    failed = set(_failed_criteria(eval_report))
    steps: list[str] = []
    if "successful_episodes" in failed or "unique_seed_count" in failed or "retained_episodes" in failed:
        steps.append("Increase collection volume or relax rejection conditions only after checking that sample quality remains acceptable.")
    if "regime_diversity" in failed or "dominant_regime_fraction" in failed:
        steps.append("Broaden passive Lenia sampling so the dataset is not dominated by a single regime.")
    if "required_family_coverage" in failed or "distinct_families" in failed:
        missing = eval_report.get("summary", {}).get("missing_required_families", [])
        if missing:
            steps.append("Regenerate agentic episodes until these required families are covered: " + ", ".join(missing) + ".")
        else:
            steps.append("Increase family-balanced collection so agentic coverage is not concentrated in too few families.")
    if "required_policy_mode_coverage" in failed:
        missing_modes = eval_report.get("summary", {}).get("missing_policy_modes", [])
        steps.append("Rebuild the runtime bootstrap dataset with explicit policy-mode coverage: " + ", ".join(missing_modes) + ".")
    if any(name.startswith("policy_mode_share_") for name in failed):
        steps.append("The runtime mode mix drifted away from the contract. Rebalance `closed_loop/random/no_action` proportions before treating the dataset as production-ready.")
    if "aggregate_policy_entropy" in failed or "action_entropy_ratio" in failed or "dominant_action_fraction" in failed:
        steps.append("The artificial-agent traces are too behaviorally narrow. Retune shaping or family bias to increase action diversity without collapsing viability.")
    if "all_actions_present" in failed or "dead_dominant_action_diversity" in failed:
        steps.append("Teacher collapse is still visible in action labels. Increase degraded trajectory coverage and re-collect until dead trajectories exhibit multiple dominant responses.")
    if "recovery_fraction" in failed:
        steps.append("Chosen actions are not improving homeostatic error often enough. Increase recovery-oriented shaping before recollecting the agentic dataset.")
    if "stress_defensive_fraction" in failed or "stress_exploit_fraction" in failed:
        steps.append("Under stress the synthetic agent is still too exploit-heavy. Increase toxic or fragile-family pressure and reward defensive responses more strongly.")
    if "perturbed_fraction_min" in failed or "perturbed_fraction_max" in failed:
        steps.append("Adjust weak perturbation coverage so the passive dataset includes the intended share of disturbed episodes.")
    if "mean_num_frames" in failed:
        steps.append("Recorded episodes are shorter than the contract expects. Check early collapse and frame retention before re-collecting.")
    if "effective_one_step_samples" in failed or "effective_step_samples" in failed:
        steps.append("The dataset is still too small for production use. Increase retained supervision count before spending longer training time.")
    if "seed_disjoint_splits" in failed:
        steps.append("The same seed leaked across splits. Repartition at the seed level and regenerate manifests before training.")
    if "source_seed_count" in failed:
        steps.append("Bootstrap coverage is coming from too few seeds. Increase source-seed diversity before promoting this dataset.")
    if "non_dead_fraction_min" in failed or "non_dead_fraction_max" in failed:
        steps.append("The success/failure mix is too one-sided. Adjust policy-mode mix and environment difficulty until both surviving and failing trajectories are retained.")
    if "rejected_fraction" in failed:
        steps.append("Too many agentic episodes are being rejected. Inspect rejection reasons and rebalance quality thresholds or environment difficulty.")
    if not steps:
        steps.append("Review the dataset evaluation and tighten the collection contract before the next run.")
    return steps


def build_collection_decision(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    eval_report: dict[str, Any],
    doctor_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    doctor_status = (doctor_report or {}).get("status", eval_report.get("doctor_status"))
    status = "blocked" if doctor_status == "blocked" else ("collect" if eval_report.get("overall_pass") else "revise")
    failed_criteria = _failed_criteria(eval_report)
    recommendation = (
        "Proceed with downstream training for: " + ", ".join(eval_report["target_modules"]) + "."
        if status == "collect"
        else "Do not treat this collection as the canonical dataset yet."
    )
    if status == "blocked":
        recommendation = "Do not trust collection outputs from a blocked environment."
    return {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "target_modules": list(contract["generator"]["target_modules"]),
        "doctor_status": doctor_status,
        "status": status,
        "failed_criteria": failed_criteria,
        "recommendation": recommendation,
        "summary": dict(eval_report.get("summary", {})),
        "ideal_advisory": dict(eval_report.get("ideal_advisory", {})),
        "next_steps": _derive_dataset_next_steps(contract, eval_report),
    }


def build_training_plan(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    collection_decision: dict[str, Any],
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    manifest_path = dataset_root / "manifest.jsonl"
    summary_path = dataset_root / "summary.json"
    slug = _slugify(contract["dataset_name"])
    status = "ready" if collection_decision.get("status") == "collect" else "blocked"
    role_view_manifests: dict[str, str] = {}
    if summary_path.exists():
        try:
            dataset_summary = load_json(summary_path)
        except Exception:
            dataset_summary = {}
        role_view_manifests = {
            str(key): str(value)
            for key, value in dict(dataset_summary.get("role_view_manifests", {})).items()
            if value
        }

    if contract["dataset_kind"] == DATASET_KIND_PASSIVE:
        trm_a_output = f"artifacts/trm_a_{slug}"
        trm_b_cache = f"data/trm_b_cache_{slug}"
        trm_b_output = f"artifacts/trm_b_{slug}"
        if "trm_wp" in role_view_manifests and "trm_bd" in role_view_manifests:
            trm_wp_manifest = role_view_manifests["trm_wp"]
            trm_bd_manifest = role_view_manifests["trm_bd"]
            steps = [
                {
                    "name": "train_trm_a",
                    "command": (
                        f"./.venv/bin/python -m trm_pipeline.train_trm_a "
                        f"--manifest {trm_wp_manifest} --output-dir {trm_a_output} "
                        "--objective gaussian_nll "
                        "--in-channels 18 --out-channels 11 "
                        "--input-key wp_input_view --target-key wp_target_observation "
                        "--baseline-key wp_observation"
                    ),
                    "output_dir": trm_a_output,
                    "metrics_path": f"{trm_a_output}/trm_a_metrics_latest.json",
                    "summary_path": f"{trm_a_output}/trm_a_summary.json",
                    "manifest_path": trm_wp_manifest,
                },
                {
                    "name": "train_trm_b",
                    "command": (
                        f"./.venv/bin/python -m trm_pipeline.train_trm_b "
                        f"--manifest {trm_bd_manifest} --output-dir {trm_b_output} "
                        "--boundary-in-channels-total 34 "
                        "--state-key bd_observation --delta-key bd_delta_observation "
                        "--error-key bd_world_error --sensor-gate-key bd_sensor_gate "
                        "--boundary-target-key bd_boundary_target "
                        "--permeability-target-key bd_permeability_target"
                    ),
                    "output_dir": trm_b_output,
                    "metrics_path": f"{trm_b_output}/trm_b_metrics_latest.json",
                    "manifest_path": trm_bd_manifest,
                },
            ]
        else:
            steps = [
                {
                    "name": "train_trm_a",
                    "command": f"./.venv/bin/python -m trm_pipeline.train_trm_a --manifest {manifest_path} --output-dir {trm_a_output} --objective variational",
                    "output_dir": trm_a_output,
                    "metrics_path": f"{trm_a_output}/trm_a_metrics_latest.json",
                    "summary_path": f"{trm_a_output}/trm_a_summary.json",
                },
                {
                    "name": "prepare_trm_b_data",
                    "command": f"./.venv/bin/python -m trm_pipeline.prepare_trm_b_data --manifest {manifest_path} --checkpoint {trm_a_output}/trm_a.pt --output-root {trm_b_cache}",
                    "output_dir": trm_b_cache,
                    "summary_path": f"{trm_b_cache}/summary.json",
                },
                {
                    "name": "train_trm_b",
                    "command": f"./.venv/bin/python -m trm_pipeline.train_trm_b --manifest {trm_b_cache}/manifest.jsonl --output-dir {trm_b_output}",
                    "output_dir": trm_b_output,
                    "metrics_path": f"{trm_b_output}/trm_b_metrics_latest.json",
                },
            ]
    else:
        trm_vm_output = f"artifacts/trm_vm_{slug}"
        trm_as_output = f"artifacts/trm_as_{slug}"
        trm_ag_output = f"artifacts/trm_ag_{slug}"
        trm_bp_output = f"artifacts/trm_bp_{slug}"
        trm_mc_output = f"artifacts/trm_mc_{slug}"
        trm_vm_manifest = role_view_manifests.get("trm_vm", str(manifest_path))
        trm_as_manifest = role_view_manifests.get("trm_as", str(manifest_path))
        steps = [
            {
                "name": "train_trm_vm",
                "command": f"./.venv/bin/python -m trm_pipeline.train_trm_vm --manifest {trm_vm_manifest} --output-dir {trm_vm_output}",
                "output_dir": trm_vm_output,
                "metrics_path": f"{trm_vm_output}/trm_vm_metrics_latest.json",
                "manifest_path": trm_vm_manifest,
            },
            {
                "name": "train_trm_as",
                "command": f"./.venv/bin/python -m trm_pipeline.train_trm_as --manifest {trm_as_manifest} --output-dir {trm_as_output}",
                "output_dir": trm_as_output,
                "metrics_path": f"{trm_as_output}/trm_as_metrics_latest.json",
                "manifest_path": trm_as_manifest,
            },
        ]
        trm_ag_manifest = role_view_manifests.get("trm_ag")
        if trm_ag_manifest:
            steps.append(
                {
                    "name": "train_trm_ag",
                    "command": f"./.venv/bin/python -m trm_pipeline.train_trm_ag --manifest {trm_ag_manifest} --output-dir {trm_ag_output}",
                    "output_dir": trm_ag_output,
                    "metrics_path": f"{trm_ag_output}/trm_ag_metrics_latest.json",
                    "manifest_path": trm_ag_manifest,
                }
            )
        trm_bp_manifest = role_view_manifests.get("trm_bp")
        if trm_bp_manifest:
            steps.append(
                {
                    "name": "train_trm_bp",
                    "command": f"./.venv/bin/python -m trm_pipeline.train_trm_bp --manifest {trm_bp_manifest} --output-dir {trm_bp_output}",
                    "output_dir": trm_bp_output,
                    "metrics_path": f"{trm_bp_output}/trm_bp_metrics_latest.json",
                    "manifest_path": trm_bp_manifest,
                }
            )
        trm_mc_manifest = role_view_manifests.get("trm_mc")
        if trm_mc_manifest:
            steps.append(
                {
                    "name": "train_trm_mc",
                    "command": f"./.venv/bin/python -m trm_pipeline.train_trm_mc --manifest {trm_mc_manifest} --output-dir {trm_mc_output}",
                    "output_dir": trm_mc_output,
                    "metrics_path": f"{trm_mc_output}/trm_mc_metrics_latest.json",
                    "manifest_path": trm_mc_manifest,
                }
            )

    return {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "status": status,
        "target_modules": list(contract["generator"]["target_modules"]),
        "dataset_manifest": str(manifest_path),
        "role_view_manifests": role_view_manifests,
        "blocked_reason": None if status == "ready" else collection_decision.get("recommendation"),
        "steps": steps,
    }


def _candidate_weight(name: str) -> float:
    weights = {
        "successful_episodes": 2.0,
        "unique_seed_count": 1.5,
        "effective_one_step_samples": 2.5,
        "retained_episodes": 2.0,
        "effective_step_samples": 2.5,
        "required_family_coverage": 2.0,
        "required_policy_mode_coverage": 2.0,
        "all_actions_present": 1.5,
        "dead_dominant_action_diversity": 1.5,
        "non_dead_fraction_min": 1.0,
        "non_dead_fraction_max": 1.0,
        "stress_defensive_fraction": 1.0,
        "stress_exploit_fraction": 1.0,
        "aggregate_policy_entropy": 1.0,
        "action_entropy_ratio": 1.0,
        "dominant_action_fraction": 1.0,
        "source_seed_count": 1.0,
        "regime_diversity": 1.0,
        "dominant_regime_fraction": 1.0,
        "stable_fraction": 1.0,
        "chaotic_fraction": 1.0,
    }
    return float(weights.get(name, 0.5))


def _search_candidate(
    base_contract: dict[str, Any],
    *,
    suffix: str,
    notes: list[str],
    targeted_failures: list[str],
    modify_fn,
) -> dict[str, Any]:
    candidate = copy.deepcopy(base_contract)
    candidate["dataset_name"] = f"{candidate['dataset_name']}_{suffix}"
    modify_fn(candidate)
    score = sum(_candidate_weight(name) for name in targeted_failures) - 0.1 * len(notes)
    candidate["search_metadata"] = {
        "candidate_suffix": suffix,
        "score": float(score),
        "targeted_failures": list(targeted_failures),
        "notes": list(notes),
    }
    return candidate


def _explore_revised_contracts(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    eval_report: dict[str, Any],
    collection_decision: dict[str, Any],
    output_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _coerce_contract(contract_or_path)
    base_contract = copy.deepcopy(contract)
    failed = set(collection_decision.get("failed_criteria", []))
    candidates: list[dict[str, Any]] = []

    if contract["dataset_kind"] == DATASET_KIND_PASSIVE:
        if {"successful_episodes", "unique_seed_count", "effective_one_step_samples"} & failed:
            def _volume(candidate: dict[str, Any]) -> None:
                config = candidate["generator"]["config"]
                current = int(config["num_seeds"])
                config["num_seeds"] = max(current + 24, int(math.ceil(current * 1.25)))
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="volume_boost",
                    notes=["Increase passive source seed count to raise retained episodes and supervision volume."],
                    targeted_failures=sorted({"successful_episodes", "unique_seed_count", "effective_one_step_samples"} & failed),
                    modify_fn=_volume,
                )
            )
        if {"stable_fraction", "chaotic_fraction", "dominant_regime_fraction", "regime_diversity"} & failed:
            def _regime(candidate: dict[str, Any]) -> None:
                candidate["generator"]["config"]["root_seed"] = int(candidate["generator"]["config"]["root_seed"]) + 101
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="regime_resample",
                    notes=["Shift the root seed so the next passive build samples a different regime mix."],
                    targeted_failures=sorted({"stable_fraction", "chaotic_fraction", "dominant_regime_fraction", "regime_diversity"} & failed),
                    modify_fn=_regime,
                )
            )
        if {"successful_episodes", "unique_seed_count", "effective_one_step_samples", "stable_fraction", "chaotic_fraction", "dominant_regime_fraction", "regime_diversity"} & failed:
            def _combined(candidate: dict[str, Any]) -> None:
                config = candidate["generator"]["config"]
                current = int(config["num_seeds"])
                config["num_seeds"] = max(current + 24, int(math.ceil(current * 1.25)))
                config["root_seed"] = int(config["root_seed"]) + 211
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="volume_regime_combo",
                    notes=[
                        "Increase passive source seed count to raise retained episodes and supervision volume.",
                        "Shift the root seed so the next passive build samples a different regime mix.",
                    ],
                    targeted_failures=sorted(
                        {
                            "successful_episodes",
                            "unique_seed_count",
                            "effective_one_step_samples",
                            "stable_fraction",
                            "chaotic_fraction",
                            "dominant_regime_fraction",
                            "regime_diversity",
                        }
                        & failed
                    ),
                    modify_fn=_combined,
                )
            )
    else:
        if {"retained_episodes", "effective_step_samples"} & failed:
            def _episodes(candidate: dict[str, Any]) -> None:
                config = candidate["generator"]["config"]
                current = int(config["episodes"])
                config["episodes"] = max(current + 24, int(math.ceil(current * 1.25)))
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="episode_boost",
                    notes=["Increase retained runtime episodes for the next bootstrap build."],
                    targeted_failures=sorted({"retained_episodes", "effective_step_samples"} & failed),
                    modify_fn=_episodes,
                )
            )
        if "source_seed_count" in failed:
            def _seed_shift(candidate: dict[str, Any]) -> None:
                candidate["generator"]["config"]["seed"] = int(candidate["generator"]["config"]["seed"]) + 997
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="seed_shift",
                    notes=["Advance the runtime seed to widen source-seed coverage."],
                    targeted_failures=["source_seed_count"],
                    modify_fn=_seed_shift,
                )
            )
        if "non_dead_fraction_max" in failed:
            def _more_degraded(candidate: dict[str, Any]) -> None:
                mix = dict(candidate["generator"]["config"].get("policy_mode_mix", {"closed_loop": 1.0}))
                mix["closed_loop"] = max(0.35, float(mix.get("closed_loop", 0.5)) - 0.10)
                mix["random"] = float(mix.get("random", 0.3)) + 0.05
                mix["no_action"] = float(mix.get("no_action", 0.2)) + 0.05
                candidate["generator"]["config"]["policy_mode_mix"] = mix
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="more_degraded_mix",
                    notes=["Too many surviving trajectories. Shift more budget to degraded runtime modes."],
                    targeted_failures=["non_dead_fraction_max"],
                    modify_fn=_more_degraded,
                )
            )
        if "non_dead_fraction_min" in failed:
            def _more_closed_loop(candidate: dict[str, Any]) -> None:
                mix = dict(candidate["generator"]["config"].get("policy_mode_mix", {"closed_loop": 1.0}))
                mix["closed_loop"] = float(mix.get("closed_loop", 0.5)) + 0.10
                mix["random"] = max(0.10, float(mix.get("random", 0.3)) - 0.05)
                mix["no_action"] = max(0.05, float(mix.get("no_action", 0.2)) - 0.05)
                candidate["generator"]["config"]["policy_mode_mix"] = mix
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="more_closed_loop",
                    notes=["Too many failing trajectories. Increase closed-loop coverage for the next build."],
                    targeted_failures=["non_dead_fraction_min"],
                    modify_fn=_more_closed_loop,
                )
            )
        if (
            "required_policy_mode_coverage" in failed
            or any(name.startswith("policy_mode_share_") for name in failed)
            or "all_actions_present" in failed
            or "dead_dominant_action_diversity" in failed
        ):
            def _anti_collapse(candidate: dict[str, Any]) -> None:
                mix = dict(candidate["generator"]["config"].get("policy_mode_mix", {"closed_loop": 1.0}))
                mix.setdefault("closed_loop", 0.50)
                mix["random"] = max(0.25, float(mix.get("random", 0.30)))
                mix["no_action"] = max(0.20, float(mix.get("no_action", 0.20)))
                candidate["generator"]["config"]["policy_mode_mix"] = mix
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="anti_collapse_mix",
                    notes=["Rebalance runtime modes to avoid closed-loop-only teacher collapse."],
                    targeted_failures=sorted(
                        {name for name in failed if name == "required_policy_mode_coverage" or name.startswith("policy_mode_share_") or name in {"all_actions_present", "dead_dominant_action_diversity"}}
                    ),
                    modify_fn=_anti_collapse,
                )
            )
        if {"stress_defensive_fraction", "stress_exploit_fraction"} & failed:
            def _defensive(candidate: dict[str, Any]) -> None:
                config = candidate["generator"]["config"]
                config["defensive_family_bias"] = float(config.get("defensive_family_bias", 0.0)) + 1.0
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="defensive_bias",
                    notes=["Increase defensive-family pressure to strengthen stress-response supervision."],
                    targeted_failures=sorted({"stress_defensive_fraction", "stress_exploit_fraction"} & failed),
                    modify_fn=_defensive,
                )
            )
        if {"aggregate_policy_entropy", "action_entropy_ratio", "dominant_action_fraction"} & failed:
            def _shaping_relax(candidate: dict[str, Any]) -> None:
                config = candidate["generator"]["config"]
                config["target_band_weight"] = max(0.0, float(config.get("target_band_weight", 0.0)) - 0.05)
                config["target_g_overshoot_weight"] = max(0.0, float(config.get("target_g_overshoot_weight", 0.0)) - 0.05)
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="shaping_relax",
                    notes=["Reduce shaping pressure slightly to recover action diversity."],
                    targeted_failures=sorted({"aggregate_policy_entropy", "action_entropy_ratio", "dominant_action_fraction"} & failed),
                    modify_fn=_shaping_relax,
                )
            )
        if failed:
            def _combo(candidate: dict[str, Any]) -> None:
                config = candidate["generator"]["config"]
                current = int(config["episodes"])
                if {"retained_episodes", "effective_step_samples"} & failed:
                    config["episodes"] = max(current + 24, int(math.ceil(current * 1.25)))
                if "source_seed_count" in failed:
                    config["seed"] = int(config["seed"]) + 997
                mix = dict(config.get("policy_mode_mix", {"closed_loop": 1.0}))
                if "non_dead_fraction_max" in failed:
                    mix["closed_loop"] = max(0.35, float(mix.get("closed_loop", 0.5)) - 0.10)
                    mix["random"] = float(mix.get("random", 0.3)) + 0.05
                    mix["no_action"] = float(mix.get("no_action", 0.2)) + 0.05
                if "non_dead_fraction_min" in failed:
                    mix["closed_loop"] = float(mix.get("closed_loop", 0.5)) + 0.10
                    mix["random"] = max(0.10, float(mix.get("random", 0.3)) - 0.05)
                    mix["no_action"] = max(0.05, float(mix.get("no_action", 0.2)) - 0.05)
                if (
                    "required_policy_mode_coverage" in failed
                    or any(name.startswith("policy_mode_share_") for name in failed)
                    or "all_actions_present" in failed
                    or "dead_dominant_action_diversity" in failed
                ):
                    mix.setdefault("closed_loop", 0.50)
                    mix["random"] = max(0.25, float(mix.get("random", 0.30)))
                    mix["no_action"] = max(0.20, float(mix.get("no_action", 0.20)))
                config["policy_mode_mix"] = mix
                if {"stress_defensive_fraction", "stress_exploit_fraction"} & failed:
                    config["defensive_family_bias"] = float(config.get("defensive_family_bias", 0.0)) + 1.0
                if {"aggregate_policy_entropy", "action_entropy_ratio", "dominant_action_fraction"} & failed:
                    config["target_band_weight"] = max(0.0, float(config.get("target_band_weight", 0.0)) - 0.05)
                    config["target_g_overshoot_weight"] = max(0.0, float(config.get("target_g_overshoot_weight", 0.0)) - 0.05)
            candidates.append(
                _search_candidate(
                    base_contract,
                    suffix="composite_search",
                    notes=["Apply a combined revision across volume, mode mix, seed coverage, and shaping based on current failures."],
                    targeted_failures=sorted(failed),
                    modify_fn=_combo,
                )
            )

    if not candidates:
        fallback = copy.deepcopy(base_contract)
        fallback["dataset_name"] = f"{fallback['dataset_name']}_rev1"
        fallback["search_metadata"] = {
            "candidate_suffix": "fallback",
            "score": 0.0,
            "targeted_failures": [],
            "notes": list(collection_decision.get("next_steps", [])) or ["Review the dataset evaluation before the next run."],
        }
        candidates.append(fallback)

    ranked = sorted(candidates, key=lambda candidate: float(candidate["search_metadata"]["score"]), reverse=True)
    best = ranked[0]
    target_root = Path(output_root) if output_root is not None else Path(contract["artifacts"]["revised_contract"]).parent / "next_attempt"
    best["revision_of"] = contract["artifacts"]["contract"]
    best["revision_reason"] = collection_decision["status"]
    best["revision_notes"] = list(best["search_metadata"]["notes"])
    best["artifacts"] = {
        **dict(best["artifacts"]),
        "contract": str(Path(target_root) / "contract.json"),
        "dataset_root": str(Path(target_root) / "dataset"),
        "doctor_report": str(Path(target_root) / "doctor_report.json"),
        "eval_report": str(Path(target_root) / "dataset_eval_report.json"),
        "collection_decision": str(Path(target_root) / "collection_decision.json"),
        "training_plan": str(Path(target_root) / "training_plan.json"),
        "training_run_report": str(Path(target_root) / "training_run_report.json"),
        "model_eval_report": str(Path(target_root) / "model_eval_report.json"),
        "promotion_decision": str(Path(target_root) / "promotion_decision.json"),
        "revision_search_report": str(Path(target_root) / "revision_search_report.json"),
        "gpu_handoff_report": str(Path(target_root) / "gpu_handoff_report.json"),
        "gpu_handoff_script": str(Path(target_root) / "run_external_gpu.sh"),
        "external_finalize_report": str(Path(target_root) / "external_finalize_report.json"),
        "revised_contract": str(Path(target_root) / "revised_contract.json"),
        "next_steps": str(Path(target_root) / "next_steps.json"),
        "run_summary": str(Path(target_root) / "run_summary.json"),
    }
    search_report = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "failed_criteria": sorted(failed),
        "selected_candidate": {
            "dataset_name": best["dataset_name"],
            "score": best["search_metadata"]["score"],
            "candidate_suffix": best["search_metadata"]["candidate_suffix"],
            "notes": best["search_metadata"]["notes"],
        },
        "candidates": [
            {
                "dataset_name": candidate["dataset_name"],
                "score": candidate["search_metadata"]["score"],
                "candidate_suffix": candidate["search_metadata"]["candidate_suffix"],
                "targeted_failures": candidate["search_metadata"]["targeted_failures"],
                "notes": candidate["search_metadata"]["notes"],
                "generator_config": candidate["generator"]["config"],
            }
            for candidate in ranked
        ],
    }
    return best, search_report


def build_revised_contract(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    eval_report: dict[str, Any],
    collection_decision: dict[str, Any],
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    revised, search_report = _explore_revised_contracts(
        contract_or_path,
        eval_report=eval_report,
        collection_decision=collection_decision,
        output_root=output_root,
    )
    save_json(revised["artifacts"]["revision_search_report"], search_report)
    return revised


def run_training_plan(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    training_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    plan = dict(training_plan or load_json(contract["artifacts"]["training_plan"]))
    if plan.get("status") != "ready":
        report = {
            "dataset_name": contract["dataset_name"],
            "status": "blocked",
            "reason": plan.get("blocked_reason"),
            "steps": [],
        }
        save_json(contract["artifacts"]["training_run_report"], report)
        return report

    step_reports: list[dict[str, Any]] = []
    for step in plan.get("steps", []):
        command = str(step["command"])
        proc = subprocess.run(
            shlex.split(command),
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
        )
        step_reports.append(
            {
                "name": step["name"],
                "command": command,
                "returncode": int(proc.returncode),
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            }
        )
        if proc.returncode != 0:
            break

    status = "passed" if step_reports and all(int(step["returncode"]) == 0 for step in step_reports) else "failed"
    if not step_reports:
        status = "blocked"
    report = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "status": status,
        "python_executable": sys.executable,
        "target_modules": list(plan.get("target_modules", [])),
        "steps": step_reports,
    }
    save_json(contract["artifacts"]["training_run_report"], report)
    return report


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    return load_json(candidate)


def evaluate_trained_models(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    training_plan: dict[str, Any] | None = None,
    training_run_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    plan = dict(training_plan or load_json(contract["artifacts"]["training_plan"]))
    training_report = dict(training_run_report or load_json(contract["artifacts"]["training_run_report"]))
    if training_report.get("status") != "passed":
        report = {
            "dataset_name": contract["dataset_name"],
            "dataset_kind": contract["dataset_kind"],
            "training_status": training_report.get("status"),
            "modules": {},
            "criteria": {},
            "overall_pass": False,
        }
        save_json(contract["artifacts"]["model_eval_report"], report)
        return report

    acceptance = dict(contract.get("model_acceptance", _default_model_acceptance(contract["dataset_kind"])))
    modules: dict[str, Any] = {}
    criteria: dict[str, Any] = {}

    for step in plan.get("steps", []):
        metrics = _load_json_if_exists(step.get("metrics_path"))
        summary = _load_json_if_exists(step.get("summary_path"))
        modules[step["name"]] = {
            "output_dir": step.get("output_dir"),
            "metrics": metrics,
            "summary": summary,
        }

    if contract["dataset_kind"] == DATASET_KIND_PASSIVE:
        trm_a = modules.get("train_trm_a", {}).get("metrics", {})
        trm_b_cache = modules.get("prepare_trm_b_data", {}).get("summary", {})
        trm_b = modules.get("train_trm_b", {}).get("metrics", {})
        criteria = {
            "trm_a.val_nmse": _criterion(
                name="trm_a.val_nmse",
                passed=math.isfinite(float(trm_a.get("val_nmse", float("nan"))))
                and float(trm_a.get("val_nmse", float("nan"))) <= float(acceptance["trm_a.max_val_nmse"]),
                actual=float(trm_a.get("val_nmse", float("nan"))),
                expected=float(acceptance["trm_a.max_val_nmse"]),
                comparator="<=",
            ),
            "trm_a.rollout_nmse_8": _criterion(
                name="trm_a.rollout_nmse_8",
                passed=math.isfinite(float(trm_a.get("rollout_nmse_8", float("nan"))))
                and float(trm_a.get("rollout_nmse_8", float("nan"))) <= float(acceptance["trm_a.max_rollout_nmse_8"]),
                actual=float(trm_a.get("rollout_nmse_8", float("nan"))),
                expected=float(acceptance["trm_a.max_rollout_nmse_8"]),
                comparator="<=",
            ),
            "trm_a.improvement_over_baseline": _criterion(
                name="trm_a.improvement_over_baseline",
                passed=math.isfinite(float(trm_a.get("improvement_over_baseline", float("nan"))))
                and float(trm_a.get("improvement_over_baseline", float("nan")))
                >= float(acceptance["trm_a.min_improvement_over_baseline"]),
                actual=float(trm_a.get("improvement_over_baseline", float("nan"))),
                expected=float(acceptance["trm_a.min_improvement_over_baseline"]),
                comparator=">=",
            ),
            "trm_b_cache.num_cached_episodes": _criterion(
                name="trm_b_cache.num_cached_episodes",
                passed=int(trm_b_cache.get("num_cached_episodes", 0)) > 0,
                actual=float(trm_b_cache.get("num_cached_episodes", 0)),
                expected=1.0,
                comparator=">=",
            ),
            "trm_b.boundary_iou": _criterion(
                name="trm_b.boundary_iou",
                passed=math.isfinite(float(trm_b.get("boundary_iou", float("nan"))))
                and float(trm_b.get("boundary_iou", float("nan"))) >= float(acceptance["trm_b.min_boundary_iou"]),
                actual=float(trm_b.get("boundary_iou", float("nan"))),
                expected=float(acceptance["trm_b.min_boundary_iou"]),
                comparator=">=",
            ),
            "trm_b.nucleus_separation": _criterion(
                name="trm_b.nucleus_separation",
                passed=math.isfinite(float(trm_b.get("nucleus_separation", float("nan"))))
                and float(trm_b.get("nucleus_separation", float("nan")))
                >= float(acceptance["trm_b.min_nucleus_separation"]),
                actual=float(trm_b.get("nucleus_separation", float("nan"))),
                expected=float(acceptance["trm_b.min_nucleus_separation"]),
                comparator=">=",
            ),
        }
    else:
        trm_vm = modules.get("train_trm_vm", {}).get("metrics", {})
        trm_as = modules.get("train_trm_as", {}).get("metrics", {})
        trm_ag = modules.get("train_trm_ag", {}).get("metrics", {})
        trm_bp = modules.get("train_trm_bp", {}).get("metrics", {})
        trm_mc = modules.get("train_trm_mc", {}).get("metrics", {})
        criteria = {
            "trm_vm.val_homeostatic_error_mae": _criterion(
                name="trm_vm.val_homeostatic_error_mae",
                passed=math.isfinite(float(trm_vm.get("val_homeostatic_error_mae", float("nan"))))
                and float(trm_vm.get("val_homeostatic_error_mae", float("nan")))
                <= float(acceptance["trm_vm.max_val_homeostatic_error_mae"]),
                actual=float(trm_vm.get("val_homeostatic_error_mae", float("nan"))),
                expected=float(acceptance["trm_vm.max_val_homeostatic_error_mae"]),
                comparator="<=",
            ),
            "trm_vm.val_viability_risk_auroc": _criterion(
                name="trm_vm.val_viability_risk_auroc",
                passed=math.isfinite(float(trm_vm.get("val_viability_risk_auroc", float("nan"))))
                and float(trm_vm.get("val_viability_risk_auroc", float("nan")))
                >= float(acceptance["trm_vm.min_val_viability_risk_auroc"]),
                actual=float(trm_vm.get("val_viability_risk_auroc", float("nan"))),
                expected=float(acceptance["trm_vm.min_val_viability_risk_auroc"]),
                comparator=">=",
            ),
            "trm_vm.val_margin_to_failure_corr": _criterion(
                name="trm_vm.val_margin_to_failure_corr",
                passed=math.isfinite(float(trm_vm.get("val_margin_to_failure_corr", float("nan"))))
                and float(trm_vm.get("val_margin_to_failure_corr", float("nan")))
                >= float(acceptance["trm_vm.min_val_margin_to_failure_corr"]),
                actual=float(trm_vm.get("val_margin_to_failure_corr", float("nan"))),
                expected=float(acceptance["trm_vm.min_val_margin_to_failure_corr"]),
                comparator=">=",
            ),
            "trm_as.val_pairwise_ranking_accuracy": _criterion(
                name="trm_as.val_pairwise_ranking_accuracy",
                passed=math.isfinite(float(trm_as.get("val_pairwise_ranking_accuracy", float("nan"))))
                and float(trm_as.get("val_pairwise_ranking_accuracy", float("nan")))
                >= float(acceptance["trm_as.min_val_pairwise_ranking_accuracy"]),
                actual=float(trm_as.get("val_pairwise_ranking_accuracy", float("nan"))),
                expected=float(acceptance["trm_as.min_val_pairwise_ranking_accuracy"]),
                comparator=">=",
            ),
            "trm_as.val_policy_entropy_mean": _criterion(
                name="trm_as.val_policy_entropy_mean",
                passed=math.isfinite(float(trm_as.get("val_policy_entropy_mean", float("nan"))))
                and float(trm_as.get("val_policy_entropy_mean", float("nan")))
                >= float(acceptance["trm_as.min_val_policy_entropy_mean"]),
                actual=float(trm_as.get("val_policy_entropy_mean", float("nan"))),
                expected=float(acceptance["trm_as.min_val_policy_entropy_mean"]),
                comparator=">=",
            ),
            "trm_as.val_action_collapse_rate": _criterion(
                name="trm_as.val_action_collapse_rate",
                passed=math.isfinite(float(trm_as.get("val_action_collapse_rate", float("nan"))))
                and float(trm_as.get("val_action_collapse_rate", float("nan")))
                <= float(acceptance["trm_as.max_val_action_collapse_rate"]),
                actual=float(trm_as.get("val_action_collapse_rate", float("nan"))),
                expected=float(acceptance["trm_as.max_val_action_collapse_rate"]),
                comparator="<=",
            ),
        }
        if trm_ag:
            criteria["trm_ag.val_inhibition_mask_mae"] = _criterion(
                name="trm_ag.val_inhibition_mask_mae",
                passed=math.isfinite(float(trm_ag.get("val_inhibition_mask_mae", float("nan"))))
                and float(trm_ag.get("val_inhibition_mask_mae", float("nan")))
                <= float(acceptance["trm_ag.max_val_inhibition_mask_mae"]),
                actual=float(trm_ag.get("val_inhibition_mask_mae", float("nan"))),
                expected=float(acceptance["trm_ag.max_val_inhibition_mask_mae"]),
                comparator="<=",
            )
            criteria["trm_ag.val_control_mode_accuracy"] = _criterion(
                name="trm_ag.val_control_mode_accuracy",
                passed=math.isfinite(float(trm_ag.get("val_control_mode_accuracy", float("nan"))))
                and float(trm_ag.get("val_control_mode_accuracy", float("nan")))
                >= float(acceptance["trm_ag.min_val_control_mode_accuracy"]),
                actual=float(trm_ag.get("val_control_mode_accuracy", float("nan"))),
                expected=float(acceptance["trm_ag.min_val_control_mode_accuracy"]),
                comparator=">=",
            )
            criteria["trm_ag.val_gated_policy_kl"] = _criterion(
                name="trm_ag.val_gated_policy_kl",
                passed=math.isfinite(float(trm_ag.get("val_gated_policy_kl", float("nan"))))
                and float(trm_ag.get("val_gated_policy_kl", float("nan")))
                <= float(acceptance["trm_ag.max_val_gated_policy_kl"]),
                actual=float(trm_ag.get("val_gated_policy_kl", float("nan"))),
                expected=float(acceptance["trm_ag.max_val_gated_policy_kl"]),
                comparator="<=",
            )
        if trm_bp:
            criteria["trm_bp.val_permeability_patch_mae"] = _criterion(
                name="trm_bp.val_permeability_patch_mae",
                passed=math.isfinite(float(trm_bp.get("val_permeability_patch_mae", float("nan"))))
                and float(trm_bp.get("val_permeability_patch_mae", float("nan")))
                <= float(acceptance["trm_bp.max_val_permeability_patch_mae"]),
                actual=float(trm_bp.get("val_permeability_patch_mae", float("nan"))),
                expected=float(acceptance["trm_bp.max_val_permeability_patch_mae"]),
                comparator="<=",
            )
            criteria["trm_bp.val_mode_accuracy"] = _criterion(
                name="trm_bp.val_mode_accuracy",
                passed=math.isfinite(float(trm_bp.get("val_mode_accuracy", float("nan"))))
                and float(trm_bp.get("val_mode_accuracy", float("nan")))
                >= float(acceptance["trm_bp.min_val_mode_accuracy"]),
                actual=float(trm_bp.get("val_mode_accuracy", float("nan"))),
                expected=float(acceptance["trm_bp.min_val_mode_accuracy"]),
                comparator=">=",
            )
        if trm_mc:
            criteria["trm_mc.val_context_state_loss"] = _criterion(
                name="trm_mc.val_context_state_loss",
                passed=math.isfinite(float(trm_mc.get("val_context_state_loss", float("nan"))))
                and float(trm_mc.get("val_context_state_loss", float("nan")))
                <= float(acceptance["trm_mc.max_val_context_state_loss"]),
                actual=float(trm_mc.get("val_context_state_loss", float("nan"))),
                expected=float(acceptance["trm_mc.max_val_context_state_loss"]),
                comparator="<=",
            )
            criteria["trm_mc.val_action_bias_alignment"] = _criterion(
                name="trm_mc.val_action_bias_alignment",
                passed=math.isfinite(float(trm_mc.get("val_action_bias_alignment", float("nan"))))
                and float(trm_mc.get("val_action_bias_alignment", float("nan")))
                >= float(acceptance["trm_mc.min_val_action_bias_alignment"]),
                actual=float(trm_mc.get("val_action_bias_alignment", float("nan"))),
                expected=float(acceptance["trm_mc.min_val_action_bias_alignment"]),
                comparator=">=",
            )

    report = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "training_status": training_report.get("status"),
        "modules": modules,
        "criteria": criteria,
        "overall_pass": bool(criteria and all(criterion["passed"] for criterion in criteria.values())),
    }
    save_json(contract["artifacts"]["model_eval_report"], report)
    return report


def build_promotion_decision(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    model_eval_report: dict[str, Any],
    training_run_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    training_status = (training_run_report or {}).get("status", model_eval_report.get("training_status"))
    if training_status != "passed":
        status = "blocked"
        recommendation = "Training did not complete successfully, so promotion is blocked."
    elif model_eval_report.get("overall_pass"):
        status = "promote"
        recommendation = "Promote these trained modules to the next runtime evaluation stage."
    else:
        status = "hold"
        recommendation = "Do not promote these trained modules yet."
    failed_criteria = [
        name
        for name, criterion in model_eval_report.get("criteria", {}).items()
        if not criterion.get("passed", False)
    ]
    decision = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "training_status": training_status,
        "status": status,
        "failed_criteria": failed_criteria,
        "recommendation": recommendation,
    }
    save_json(contract["artifacts"]["promotion_decision"], decision)
    return decision


def _append_model_registry_entries(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    training_plan: dict[str, Any] | None = None,
    training_run_report: dict[str, Any] | None = None,
    model_eval_report: dict[str, Any] | None = None,
    promotion_decision: dict[str, Any] | None = None,
    execution_target: str = "local",
) -> list[dict[str, Any]]:
    contract = _coerce_contract(contract_or_path)
    plan = dict(training_plan or load_json(contract["artifacts"]["training_plan"]))
    training_report = dict(training_run_report or load_json(contract["artifacts"]["training_run_report"]))
    model_report = dict(model_eval_report or _load_json_if_exists(contract["artifacts"]["model_eval_report"]))
    promotion = dict(promotion_decision or _load_json_if_exists(contract["artifacts"]["promotion_decision"]))
    entries: list[dict[str, Any]] = []
    checkpoint_names = {
        "train_trm_a": ("trm_a.pt", "trm_a_best.pt"),
        "train_trm_b": ("trm_b.pt", "trm_b_best.pt"),
        "train_trm_vm": ("trm_vm.pt", "trm_vm_best.pt"),
        "train_trm_as": ("trm_as.pt", "trm_as_best.pt"),
        "train_trm_ag": ("trm_ag.pt", "trm_ag_best.pt"),
        "train_trm_bp": ("trm_bp.pt", "trm_bp_best.pt"),
        "train_trm_mc": ("trm_mc.pt", "trm_mc_best.pt"),
    }
    for step in plan.get("steps", []):
        if not str(step.get("name", "")).startswith("train_"):
            continue
        output_dir = Path(step.get("output_dir", "."))
        latest_name, best_name = checkpoint_names.get(step["name"], ("checkpoint.pt", "checkpoint_best.pt"))
        criterion_prefix = step["name"].replace("train_", "") + "."
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": contract["dataset_name"],
            "dataset_kind": contract["dataset_kind"],
            "contract_path": contract["artifacts"]["contract"],
            "dataset_root": contract["artifacts"]["dataset_root"],
            "dataset_registry_path": contract["artifacts"]["registry_path"],
            "module_name": step["name"],
            "target_module": step["name"].replace("train_", ""),
            "output_dir": str(output_dir),
            "metrics_path": step.get("metrics_path"),
            "summary_path": step.get("summary_path"),
            "latest_checkpoint": str(output_dir / latest_name),
            "best_checkpoint": str(output_dir / best_name),
            "training_status": training_report.get("status"),
            "promotion_status": promotion.get("status"),
            "promotion_decision_path": contract["artifacts"]["promotion_decision"],
            "model_eval_report_path": contract["artifacts"]["model_eval_report"],
            "execution_target": execution_target,
            "failed_criteria": [
                name
                for name in promotion.get("failed_criteria", [])
                if str(name).startswith(criterion_prefix)
            ],
            "module_metrics": dict(model_report.get("modules", {}).get(step["name"], {}).get("metrics", {})),
        }
        append_jsonl(contract["artifacts"]["model_registry_path"], entry)
        entries.append(entry)
    return entries


def _with_gpu_training_flags(command: str) -> str:
    parts = shlex.split(command)
    if not any(
        name in command
        for name in (
            "train_trm_a",
            "train_trm_b",
            "train_trm_vm",
            "train_trm_as",
            "train_trm_ag",
            "train_trm_bp",
            "train_trm_mc",
        )
    ):
        return command
    if "--device" not in parts:
        parts.extend(["--device", "cuda"])
    if "--amp" not in parts:
        parts.append("--amp")
    if "--log-interval" not in parts:
        parts.extend(["--log-interval", "50"])
    return shlex.join(parts)


def build_external_gpu_handoff(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    training_plan: dict[str, Any] | None = None,
    provider: str = "vastai",
    remote_root: str = "/workspace/criticism_bot",
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    plan = dict(training_plan or load_json(contract["artifacts"]["training_plan"]))
    if plan.get("status") != "ready":
        report = {
            "dataset_name": contract["dataset_name"],
            "dataset_kind": contract["dataset_kind"],
            "provider": provider,
            "status": "blocked",
            "reason": plan.get("blocked_reason"),
        }
        save_json(contract["artifacts"]["gpu_handoff_report"], report)
        return report
    commands = [_with_gpu_training_flags(str(step["command"])) for step in plan.get("steps", [])]
    files_to_pull: list[str] = []
    for step in plan.get("steps", []):
        if step.get("metrics_path"):
            files_to_pull.append(str(step["metrics_path"]))
        if step.get("summary_path"):
            files_to_pull.append(str(step["summary_path"]))
    files_to_pull.extend(
        [
            contract["artifacts"]["contract"],
            contract["artifacts"]["training_plan"],
        ]
    )
    finalize_local_command = (
        f"./.venv/bin/python -m trm_pipeline.dataset_harness finalize-external "
        f"--contract {shlex.quote(str(contract['artifacts']['contract']))}"
    )
    handoff = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "provider": provider,
        "status": "ready",
        "remote_root": remote_root,
        "workspace_sync_assumption": "The repository and collected dataset artifacts are mirrored under the same relative paths on the remote GPU host.",
        "contract_path": contract["artifacts"]["contract"],
        "dataset_root": contract["artifacts"]["dataset_root"],
        "training_plan": contract["artifacts"]["training_plan"],
        "model_registry_path": contract["artifacts"]["model_registry_path"],
        "files_to_pull": files_to_pull,
        "finalize_local_command": finalize_local_command,
        "commands": commands,
        "files_to_sync": [
            contract["artifacts"]["contract"],
            contract["artifacts"]["dataset_root"],
            contract["artifacts"]["training_plan"],
        ],
    }
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'cd "{remote_root}"',
        "",
    ] + commands
    script_path = Path(contract["artifacts"]["gpu_handoff_script"])
    save_json(contract["artifacts"]["gpu_handoff_report"], handoff)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return handoff


def finalize_external_training(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    training_plan: dict[str, Any] | None = None,
    gpu_handoff_report: dict[str, Any] | None = None,
    status: str = "auto",
    note: str | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    plan = dict(training_plan or load_json(contract["artifacts"]["training_plan"]))
    handoff = dict(gpu_handoff_report or _load_json_if_exists(contract["artifacts"]["gpu_handoff_report"]))
    provider = str(handoff.get("provider", "external_gpu"))
    execution_target = f"external_gpu:{provider}"

    step_reports: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    for step in plan.get("steps", []):
        expected_outputs: list[str] = []
        if step.get("metrics_path"):
            expected_outputs.append(str(step["metrics_path"]))
        if step.get("summary_path"):
            expected_outputs.append(str(step["summary_path"]))
        absent = [path for path in expected_outputs if not Path(path).exists()]
        missing_outputs.extend(absent)
        step_reports.append(
            {
                "name": step["name"],
                "command": str(step["command"]),
                "output_dir": step.get("output_dir"),
                "expected_outputs": expected_outputs,
                "missing_outputs": absent,
                "status": "passed" if not absent else "missing_outputs",
            }
        )

    final_status = status
    if final_status == "auto":
        if plan.get("status") != "ready":
            final_status = "blocked"
        elif missing_outputs:
            final_status = "failed"
        else:
            final_status = "passed"
    if final_status not in {"passed", "failed", "blocked"}:
        raise SystemExit(f"unknown external finalize status: {final_status}")

    training_report = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "status": final_status,
        "python_executable": sys.executable,
        "target_modules": list(plan.get("target_modules", [])),
        "steps": step_reports,
        "source": "external_gpu_finalize",
        "provider": provider,
        "remote_root": handoff.get("remote_root"),
        "execution_target": execution_target,
        "note": note,
    }
    save_json(contract["artifacts"]["training_run_report"], training_report)
    model_eval_report = evaluate_trained_models(
        contract,
        training_plan=plan,
        training_run_report=training_report,
    )
    promotion_decision = build_promotion_decision(
        contract,
        model_eval_report=model_eval_report,
        training_run_report=training_report,
    )
    registry_entries = _append_model_registry_entries(
        contract,
        training_plan=plan,
        training_run_report=training_report,
        model_eval_report=model_eval_report,
        promotion_decision=promotion_decision,
        execution_target=execution_target,
    )
    report = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "status": promotion_decision["status"],
        "training_status": training_report["status"],
        "provider": provider,
        "execution_target": execution_target,
        "missing_outputs": missing_outputs,
        "training_run_report": contract["artifacts"]["training_run_report"],
        "model_eval_report": contract["artifacts"]["model_eval_report"],
        "promotion_decision": contract["artifacts"]["promotion_decision"],
        "model_registry_path": contract["artifacts"]["model_registry_path"],
        "registry_rows_written": len(registry_entries),
        "note": note,
    }
    save_json(contract["artifacts"]["external_finalize_report"], report)
    return report


def _append_registry_entry(
    contract: dict[str, Any],
    *,
    run_summary: dict[str, Any],
    collection_decision: dict[str, Any],
) -> dict[str, Any]:
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "target_modules": list(contract["generator"]["target_modules"]),
        "contract_path": contract["artifacts"]["contract"],
        "dataset_root": contract["artifacts"]["dataset_root"],
        "collection_decision": contract["artifacts"]["collection_decision"],
        "training_plan": contract["artifacts"]["training_plan"],
        "training_run_report": contract["artifacts"]["training_run_report"],
        "model_eval_report": contract["artifacts"]["model_eval_report"],
        "promotion_decision": contract["artifacts"]["promotion_decision"],
        "run_summary": contract["artifacts"]["run_summary"],
        "status": run_summary["status"],
        "decision_status": collection_decision["status"],
    }
    append_jsonl(contract["artifacts"]["registry_path"], entry)
    return entry


def _run_collection_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(contract["artifacts"]["dataset_root"])
    generator = dict(contract["generator"])
    cfg = dict(generator["config"])
    if contract["dataset_kind"] == DATASET_KIND_PASSIVE:
        manifest_path = generate_rollouts(
            dataset_root,
            generator["seed_catalog"],
            RolloutConfig(
                image_size=int(cfg["image_size"]),
                warmup_steps=int(cfg["warmup_steps"]),
                record_steps=int(cfg["record_steps"]),
                target_radius=int(cfg["target_radius"]),
                num_seeds=int(cfg["num_seeds"]),
                root_seed=int(cfg["root_seed"]),
            ),
        )
    else:
        mode_mix = _allocate_mode_counts(int(cfg["episodes"]), dict(cfg.get("policy_mode_mix", {"closed_loop": 1.0})))
        merged_rows: list[dict[str, Any]] = []
        aggregate_action_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        policy_mode_counts: dict[str, int] = {}
        rejection_reasons: dict[str, int] = {}
        aggregate_policy_entropy_rows: list[tuple[float | None, int]] = []
        aggregate_risk_rows: list[tuple[float | None, int]] = []
        aggregate_recovery_rows: list[tuple[float | None, int]] = []
        aggregate_stress_defensive_rows: list[tuple[float | None, int]] = []
        aggregate_stress_exploit_rows: list[tuple[float | None, int]] = []
        rejected_total = 0
        attempted_total = 0
        for offset, (policy_mode, num_episodes) in enumerate(mode_mix.items()):
            mode_root = ensure_dir(dataset_root / "by_mode" / policy_mode)
            mode_manifest_path = prepare_trm_va_cache(
                seed_catalog=generator["seed_catalog"],
                output_root=mode_root,
                runtime_config=RuntimeConfig(
                    steps=int(cfg["steps"]),
                    warmup_steps=int(cfg["warmup_steps"]),
                    seed=int(cfg["seed"]) + 10_000 * offset,
                    policy_mode=policy_mode,
                ),
                env_config=EnvironmentConfig(
                    image_size=int(cfg["image_size"]),
                    target_radius=int(cfg["target_radius"]),
                ),
                num_episodes=int(num_episodes),
                target_band_weight=float(cfg["target_band_weight"]),
                target_g_overshoot_weight=float(cfg["target_g_overshoot_weight"]),
                defensive_family_bias=float(cfg["defensive_family_bias"]),
                max_attempt_multiplier=int(cfg["max_attempt_multiplier"]),
                min_episode_samples=int(cfg["min_episode_samples"]),
                min_distinct_actions=int(cfg["min_distinct_actions"]),
                max_dominant_action_fraction=float(cfg["max_dominant_action_fraction"]),
                min_episode_policy_entropy=float(cfg["min_episode_policy_entropy"]),
            )
            mode_rows = load_jsonl(mode_manifest_path)
            mode_summary = load_json(mode_root / "summary.json")
            merged_rows.extend(mode_rows)
            rejected_total += int(mode_summary.get("rejected_episodes", 0))
            attempted_total += int(mode_summary.get("attempted_episodes", len(mode_rows)))
            policy_mode_counts[policy_mode] = int(len(mode_rows))
            for family, count in dict(mode_summary.get("family_counts", {})).items():
                family_counts[family] = int(family_counts.get(family, 0) + int(count))
            for action, count in dict(mode_summary.get("aggregate_action_counts", {})).items():
                aggregate_action_counts[action] = int(aggregate_action_counts.get(action, 0) + int(count))
            for reason, count in dict(mode_summary.get("rejection_reasons", {})).items():
                rejection_reasons[reason] = int(rejection_reasons.get(reason, 0) + int(count))
            weight = int(mode_summary.get("retained_episodes", len(mode_rows)))
            aggregate_policy_entropy_rows.append((mode_summary.get("aggregate_policy_entropy_mean"), weight))
            aggregate_risk_rows.append((mode_summary.get("aggregate_risk_rate_mean"), weight))
            aggregate_recovery_rows.append((mode_summary.get("aggregate_recovery_fraction_mean"), weight))
            aggregate_stress_defensive_rows.append((mode_summary.get("aggregate_stress_defensive_fraction_mean"), weight))
            aggregate_stress_exploit_rows.append((mode_summary.get("aggregate_stress_exploit_fraction_mean"), weight))
        merged_rows = sorted(merged_rows, key=lambda row: str(row.get("episode_id", "")))
        manifest_path = dataset_root / "manifest.jsonl"
        save_jsonl(manifest_path, merged_rows)
        save_json(
            dataset_root / "summary.json",
            {
                "seed_catalog": generator["seed_catalog"],
                "retained_episodes": len(merged_rows),
                "rejected_episodes": rejected_total,
                "attempted_episodes": attempted_total,
                "requested_policy_mode_mix": dict(cfg.get("policy_mode_mix", {"closed_loop": 1.0})),
                "policy_mode_counts": policy_mode_counts,
                "family_counts": family_counts,
                "aggregate_action_counts": aggregate_action_counts,
                "aggregate_policy_entropy_mean": _weighted_mean(aggregate_policy_entropy_rows),
                "aggregate_risk_rate_mean": _weighted_mean(aggregate_risk_rows),
                "aggregate_recovery_fraction_mean": _weighted_mean(aggregate_recovery_rows),
                "aggregate_stress_defensive_fraction_mean": _weighted_mean(aggregate_stress_defensive_rows),
                "aggregate_stress_exploit_fraction_mean": _weighted_mean(aggregate_stress_exploit_rows),
                "rejection_reasons": rejection_reasons,
            },
        )
    return {
        "manifest_path": str(manifest_path),
        "summary_path": str(dataset_root / "summary.json"),
        "dataset_root": str(dataset_root),
    }


def run_dataset_contract(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    force: bool = False,
    skip_doctor: bool = False,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    artifacts = dict(contract["artifacts"])
    doctor_report = {"status": "skipped", "blocking_issues": [], "warnings": []} if skip_doctor else run_doctor()
    save_json(artifacts["doctor_report"], doctor_report)

    if doctor_report["status"] == "blocked" and not force:
        eval_report = {
            "dataset_name": contract["dataset_name"],
            "dataset_kind": contract["dataset_kind"],
            "dataset_root": artifacts["dataset_root"],
            "target_modules": list(contract["generator"]["target_modules"]),
            "summary": {},
            "criteria": {},
            "overall_pass": False,
            "doctor_status": "blocked",
        }
        decision = build_collection_decision(contract, eval_report=eval_report, doctor_report=doctor_report)
        training_plan = build_training_plan(contract, collection_decision=decision)
        save_json(artifacts["eval_report"], eval_report)
        save_json(artifacts["collection_decision"], decision)
        save_json(artifacts["training_plan"], training_plan)
        save_json(artifacts["next_steps"], {"status": "blocked", "next_steps": decision["next_steps"]})
        run_summary = {
            "dataset_name": contract["dataset_name"],
            "status": "blocked",
            "reason": "doctor_failed",
            "doctor_report": artifacts["doctor_report"],
            "dataset_root": artifacts["dataset_root"],
            "collection_decision": artifacts["collection_decision"],
            "training_plan": artifacts["training_plan"],
            "registry_path": artifacts["registry_path"],
        }
        save_json(artifacts["run_summary"], run_summary)
        _append_registry_entry(contract, run_summary=run_summary, collection_decision=decision)
        return run_summary

    generation = _run_collection_from_contract(contract)
    eval_report = evaluate_dataset_contract(contract, doctor_report=doctor_report)
    decision = build_collection_decision(contract, eval_report=eval_report, doctor_report=doctor_report)
    training_plan = build_training_plan(contract, collection_decision=decision)
    save_json(artifacts["eval_report"], eval_report)
    save_json(artifacts["collection_decision"], decision)
    save_json(artifacts["training_plan"], training_plan)
    save_json(artifacts["next_steps"], {"status": decision["status"], "next_steps": decision["next_steps"]})
    run_summary = {
        "dataset_name": contract["dataset_name"],
        "dataset_kind": contract["dataset_kind"],
        "status": "passed" if eval_report["overall_pass"] else "failed",
        "doctor_report": artifacts["doctor_report"],
        "dataset_root": generation["dataset_root"],
        "manifest_path": generation["manifest_path"],
        "summary_path": generation["summary_path"],
        "eval_report": artifacts["eval_report"],
        "collection_decision": artifacts["collection_decision"],
        "training_plan": artifacts["training_plan"],
        "registry_path": artifacts["registry_path"],
        "target_modules": list(contract["generator"]["target_modules"]),
    }
    save_json(artifacts["run_summary"], run_summary)
    _append_registry_entry(contract, run_summary=run_summary, collection_decision=decision)
    return run_summary


def run_dataset_campaign(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    force: bool = False,
    skip_doctor: bool = False,
    auto_handoff: bool = False,
    external_gpu_provider: str | None = None,
    external_gpu_remote_root: str = "/workspace/criticism_bot",
    revised_output_root: str | Path | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(contract_or_path)
    run_summary = run_dataset_contract(contract, force=force, skip_doctor=skip_doctor)
    decision = load_json(contract["artifacts"]["collection_decision"])
    eval_report = load_json(contract["artifacts"]["eval_report"])
    training_plan = _load_json_if_exists(contract["artifacts"]["training_plan"])
    if decision["status"] == "collect" and not training_plan:
        training_plan = build_training_plan(contract, collection_decision=decision)
        save_json(contract["artifacts"]["training_plan"], training_plan)
    result: dict[str, Any] = {
        "dataset_name": contract["dataset_name"],
        "run_summary": contract["artifacts"]["run_summary"],
        "collection_decision": contract["artifacts"]["collection_decision"],
        "status": decision["status"],
    }
    if decision["status"] == "collect" and external_gpu_provider:
        handoff = build_external_gpu_handoff(
            contract,
            training_plan=training_plan,
            provider=external_gpu_provider,
            remote_root=external_gpu_remote_root,
        )
        result["training_plan"] = contract["artifacts"]["training_plan"]
        result["gpu_handoff_report"] = contract["artifacts"]["gpu_handoff_report"]
        result["gpu_handoff_script"] = contract["artifacts"]["gpu_handoff_script"]
        result["gpu_handoff_status"] = handoff["status"]
        return result
    if decision["status"] == "collect" and auto_handoff:
        training_report = run_training_plan(contract, training_plan=training_plan)
        model_eval_report = evaluate_trained_models(
            contract,
            training_plan=training_plan,
            training_run_report=training_report,
        )
        promotion_decision = build_promotion_decision(
            contract,
            model_eval_report=model_eval_report,
            training_run_report=training_report,
        )
        _append_model_registry_entries(
            contract,
            training_plan=training_plan,
            training_run_report=training_report,
            model_eval_report=model_eval_report,
            promotion_decision=promotion_decision,
            execution_target="local",
        )
        result["training_plan"] = contract["artifacts"]["training_plan"]
        result["training_run_report"] = contract["artifacts"]["training_run_report"]
        result["training_status"] = training_report["status"]
        result["model_eval_report"] = contract["artifacts"]["model_eval_report"]
        result["promotion_decision"] = contract["artifacts"]["promotion_decision"]
        result["promotion_status"] = promotion_decision["status"]
        return result
    if decision["status"] != "collect":
        revised = build_revised_contract(
            contract,
            eval_report=eval_report,
            collection_decision=decision,
            output_root=revised_output_root,
        )
        save_json(contract["artifacts"]["revised_contract"], revised)
        result["revised_contract"] = contract["artifacts"]["revised_contract"]
    return result


def _campaign_success(
    result: dict[str, Any],
    *,
    auto_handoff: bool,
    external_gpu_provider: str | None,
) -> bool:
    if external_gpu_provider:
        return str(result.get("gpu_handoff_status")) == "ready"
    if auto_handoff:
        return str(result.get("promotion_status")) == "promote"
    return str(result.get("status")) == "collect"


def run_dataset_campaign_until_acceptance(
    contract_or_path: dict[str, Any] | str | Path,
    *,
    max_rounds: int = 3,
    force: bool = False,
    skip_doctor: bool = False,
    auto_handoff: bool = False,
    external_gpu_provider: str | None = None,
    external_gpu_remote_root: str = "/workspace/criticism_bot",
    revised_output_root: str | Path | None = None,
) -> dict[str, Any]:
    if int(max_rounds) <= 0:
        raise SystemExit("max_rounds must be >= 1")
    initial_contract = _coerce_contract(contract_or_path)
    current_contract: dict[str, Any] = initial_contract
    rounds: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None

    for round_index in range(1, int(max_rounds) + 1):
        result = run_dataset_campaign(
            current_contract,
            force=force,
            skip_doctor=skip_doctor,
            auto_handoff=auto_handoff,
            external_gpu_provider=external_gpu_provider,
            external_gpu_remote_root=external_gpu_remote_root,
            revised_output_root=revised_output_root,
        )
        round_report = {
            "round_index": round_index,
            "contract": current_contract["artifacts"]["contract"],
            "dataset_name": current_contract["dataset_name"],
            "status": result.get("status"),
            "promotion_status": result.get("promotion_status"),
            "gpu_handoff_status": result.get("gpu_handoff_status"),
            "revised_contract": result.get("revised_contract"),
        }
        rounds.append(round_report)
        final_result = result
        if _campaign_success(
            result,
            auto_handoff=auto_handoff,
            external_gpu_provider=external_gpu_provider,
        ):
            break
        revised_contract_path = result.get("revised_contract")
        if not revised_contract_path:
            break
        current_contract = _coerce_contract(revised_contract_path)

    assert final_result is not None
    report = {
        "initial_contract": initial_contract["artifacts"]["contract"],
        "final_contract": current_contract["artifacts"]["contract"],
        "max_rounds": int(max_rounds),
        "rounds_executed": len(rounds),
        "rounds": rounds,
        "final_result": final_result,
        "status": (
            "accepted"
            if _campaign_success(
                final_result,
                auto_handoff=auto_handoff,
                external_gpu_provider=external_gpu_provider,
            )
            else "max_rounds_exhausted"
        ),
    }
    save_json(initial_contract["artifacts"]["campaign_until_report"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness workflow for dataset collection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check environment prerequisites for dataset collection.")
    doctor_parser.add_argument("--output", default=None)

    plan_parser = subparsers.add_parser("plan", help="Write a file-based dataset contract.")
    plan_parser.add_argument("--output-root", required=True)
    plan_parser.add_argument("--dataset-name", required=True)
    plan_parser.add_argument("--dataset-kind", choices=DATASET_KINDS, required=True)
    plan_parser.add_argument("--preset", choices=DATASET_PRESETS, default=None)
    plan_parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    plan_parser.add_argument("--num-seeds", type=int, default=200)
    plan_parser.add_argument("--warmup-steps", type=int, default=32)
    plan_parser.add_argument("--record-steps", type=int, default=256)
    plan_parser.add_argument("--image-size", type=int, default=64)
    plan_parser.add_argument("--target-radius", type=int, default=12)
    plan_parser.add_argument("--root-seed", type=int, default=20260306)
    plan_parser.add_argument("--episodes", type=int, default=16)
    plan_parser.add_argument("--steps", type=int, default=32)
    plan_parser.add_argument("--runtime-seed", type=int, default=20260318)
    plan_parser.add_argument("--target-band-weight", type=float, default=0.0)
    plan_parser.add_argument("--target-g-overshoot-weight", type=float, default=0.0)
    plan_parser.add_argument("--defensive-family-bias", type=float, default=0.0)
    plan_parser.add_argument("--policy-mode-mix", nargs="*", default=None)
    plan_parser.add_argument("--max-attempt-multiplier", type=int, default=4)
    plan_parser.add_argument("--min-episode-samples", type=int, default=8)
    plan_parser.add_argument("--min-distinct-actions", type=int, default=2)
    plan_parser.add_argument("--max-dominant-action-fraction", type=float, default=0.90)
    plan_parser.add_argument("--min-episode-policy-entropy", type=float, default=0.90)
    plan_parser.add_argument("--required-families", nargs="*", default=None)

    run_parser = subparsers.add_parser("run", help="Run dataset collection from a contract file.")
    run_parser.add_argument("--contract", required=True)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--skip-doctor", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an existing dataset root against a contract.")
    eval_parser.add_argument("--contract", required=True)
    eval_parser.add_argument("--dataset-root", default=None)
    eval_parser.add_argument("--output", default=None)

    revise_parser = subparsers.add_parser("revise", help="Derive the next contract from the latest dataset decision.")
    revise_parser.add_argument("--contract", required=True)
    revise_parser.add_argument("--output-root", default=None)

    handoff_parser = subparsers.add_parser("handoff", help="Execute the training plan for a collected dataset.")
    handoff_parser.add_argument("--contract", required=True)

    gpu_handoff_parser = subparsers.add_parser("gpu-handoff", help="Generate external GPU handoff artifacts for a ready training plan.")
    gpu_handoff_parser.add_argument("--contract", required=True)
    gpu_handoff_parser.add_argument("--provider", default="vastai")
    gpu_handoff_parser.add_argument("--remote-root", default="/workspace/criticism_bot")

    finalize_external_parser = subparsers.add_parser(
        "finalize-external",
        help="Finalize an externally executed GPU training run after syncing artifacts back.",
    )
    finalize_external_parser.add_argument("--contract", required=True)
    finalize_external_parser.add_argument("--status", choices=("auto", "passed", "failed", "blocked"), default="auto")
    finalize_external_parser.add_argument("--note", default=None)

    model_eval_parser = subparsers.add_parser("post-train-eval", help="Evaluate completed training outputs and emit a promotion decision.")
    model_eval_parser.add_argument("--contract", required=True)

    campaign_parser = subparsers.add_parser(
        "campaign",
        help="Run collection, then either emit a revised contract or hand off to training.",
    )
    campaign_parser.add_argument("--contract", required=True)
    campaign_parser.add_argument("--force", action="store_true")
    campaign_parser.add_argument("--skip-doctor", action="store_true")
    campaign_parser.add_argument("--auto-handoff", action="store_true")
    campaign_parser.add_argument("--external-gpu-provider", default=None)
    campaign_parser.add_argument("--external-gpu-remote-root", default="/workspace/criticism_bot")
    campaign_parser.add_argument("--revised-output-root", default=None)

    campaign_until_parser = subparsers.add_parser(
        "campaign-until-pass",
        help="Repeat campaign + revise until collection/training reaches the configured success state or max rounds are exhausted.",
    )
    campaign_until_parser.add_argument("--contract", required=True)
    campaign_until_parser.add_argument("--max-rounds", type=int, default=3)
    campaign_until_parser.add_argument("--force", action="store_true")
    campaign_until_parser.add_argument("--skip-doctor", action="store_true")
    campaign_until_parser.add_argument("--auto-handoff", action="store_true")
    campaign_until_parser.add_argument("--external-gpu-provider", default=None)
    campaign_until_parser.add_argument("--external-gpu-remote-root", default="/workspace/criticism_bot")
    campaign_until_parser.add_argument("--revised-output-root", default=None)

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
        contract_kwargs = {
            "output_root": args.output_root,
            "dataset_name": args.dataset_name,
            "dataset_kind": args.dataset_kind,
            "seed_catalog": args.seed_catalog,
            "num_seeds": args.num_seeds,
            "warmup_steps": args.warmup_steps,
            "record_steps": args.record_steps,
            "image_size": args.image_size,
            "target_radius": args.target_radius,
            "root_seed": args.root_seed,
            "episodes": args.episodes,
            "steps": args.steps,
            "runtime_seed": args.runtime_seed,
            "target_band_weight": args.target_band_weight,
            "target_g_overshoot_weight": args.target_g_overshoot_weight,
            "defensive_family_bias": args.defensive_family_bias,
            "policy_mode_mix": _parse_mode_mix(args.policy_mode_mix),
            "max_attempt_multiplier": args.max_attempt_multiplier,
            "min_episode_samples": args.min_episode_samples,
            "min_distinct_actions": args.min_distinct_actions,
            "max_dominant_action_fraction": args.max_dominant_action_fraction,
            "min_episode_policy_entropy": args.min_episode_policy_entropy,
            "required_families": args.required_families,
        }
        if args.preset:
            preset = _preset_overrides(args.preset)
            preset_acceptance = dict(preset.pop("acceptance", {}))
            contract_kwargs.update(preset)
            contract_kwargs["acceptance"] = preset_acceptance
        contract = build_dataset_contract(**contract_kwargs)
        save_json(contract["artifacts"]["contract"], contract)
        print(f"wrote dataset contract: {contract['artifacts']['contract']}")
        return

    if args.command == "run":
        summary = run_dataset_contract(args.contract, force=args.force, skip_doctor=args.skip_doctor)
        print(f"wrote dataset harness summary: {summary['status']} -> {load_json(args.contract)['artifacts']['run_summary']}")
        return

    if args.command == "evaluate":
        report = evaluate_dataset_contract(args.contract, dataset_root=args.dataset_root)
        output_path = args.output or load_json(args.contract)["artifacts"]["eval_report"]
        save_json(output_path, report)
        print(f"wrote dataset evaluation: {output_path}")
        return

    if args.command == "revise":
        contract = _coerce_contract(args.contract)
        eval_report = load_json(contract["artifacts"]["eval_report"])
        decision = load_json(contract["artifacts"]["collection_decision"])
        revised = build_revised_contract(
            contract,
            eval_report=eval_report,
            collection_decision=decision,
            output_root=args.output_root,
        )
        output_path = contract["artifacts"]["revised_contract"]
        save_json(output_path, revised)
        print(f"wrote revised dataset contract: {output_path}")
        return

    if args.command == "handoff":
        contract = _coerce_contract(args.contract)
        training_report = run_training_plan(contract)
        model_eval_report = evaluate_trained_models(contract, training_run_report=training_report)
        promotion_decision = build_promotion_decision(
            contract,
            model_eval_report=model_eval_report,
            training_run_report=training_report,
        )
        _append_model_registry_entries(
            contract,
            training_run_report=training_report,
            model_eval_report=model_eval_report,
            promotion_decision=promotion_decision,
            execution_target="local",
        )
        print(
            "wrote training handoff report: "
            f"{contract['artifacts']['training_run_report']} ({training_report['status']}); "
            f"promotion={promotion_decision['status']}"
        )
        return

    if args.command == "gpu-handoff":
        contract = _coerce_contract(args.contract)
        handoff = build_external_gpu_handoff(
            contract,
            provider=args.provider,
            remote_root=args.remote_root,
        )
        print(
            "wrote GPU handoff artifact: "
            f"{contract['artifacts']['gpu_handoff_report']} ({handoff['status']})"
        )
        return

    if args.command == "finalize-external":
        contract = _coerce_contract(args.contract)
        report = finalize_external_training(
            contract,
            status=args.status,
            note=args.note,
        )
        print(
            "wrote external finalize report: "
            f"{contract['artifacts']['external_finalize_report']} ({report['status']})"
        )
        return

    if args.command == "post-train-eval":
        contract = _coerce_contract(args.contract)
        model_eval_report = evaluate_trained_models(contract)
        promotion_decision = build_promotion_decision(contract, model_eval_report=model_eval_report)
        _append_model_registry_entries(
            contract,
            model_eval_report=model_eval_report,
            promotion_decision=promotion_decision,
            execution_target="local",
        )
        print(
            "wrote post-training evaluation: "
            f"{contract['artifacts']['model_eval_report']} ({promotion_decision['status']})"
        )
        return

    if args.command == "campaign":
        result = run_dataset_campaign(
            args.contract,
            force=args.force,
            skip_doctor=args.skip_doctor,
            auto_handoff=args.auto_handoff,
            external_gpu_provider=args.external_gpu_provider,
            external_gpu_remote_root=args.external_gpu_remote_root,
            revised_output_root=args.revised_output_root,
        )
        print(f"wrote dataset campaign result: {result['status']}")
        return

    report = run_dataset_campaign_until_acceptance(
        args.contract,
        max_rounds=args.max_rounds,
        force=args.force,
        skip_doctor=args.skip_doctor,
        auto_handoff=args.auto_handoff,
        external_gpu_provider=args.external_gpu_provider,
        external_gpu_remote_root=args.external_gpu_remote_root,
        revised_output_root=args.revised_output_root,
    )
    print(f"wrote campaign-until-pass report: {load_json(args.contract)['artifacts']['campaign_until_report']} ({report['status']})")


if __name__ == "__main__":
    main()
