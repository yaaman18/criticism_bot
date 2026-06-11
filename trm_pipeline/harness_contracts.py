from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .common import load_json


DEFAULT_ACCEPTANCE = {
    "max_mean_final_homeostatic_error": 0.35,
    "max_mean_mean_homeostatic_error": 0.30,
    "max_dead_fraction": 0.20,
    "min_final_improvement_vs_baseline": 0.00,
    "min_final_improvement_ci_lower": 0.00,
    "max_mean_homeostatic_degradation": 0.01,
    "max_stress_exploit_degradation": 0.20,
    "bootstrap_samples": 400,
    "require_holdout_for_promotion": False,
    "min_best_mode_frequency": 0.60,
    "min_stress_defensive_rate": 0.40,
    "max_stress_exploit_rate": 0.60,
    "min_action_diversity": 0.20,
    "max_intake_rate": 0.70,
    "min_navigation_rate": 0.08,
    "min_trace_ablation_spawn_delta": 0.0,
    "min_mean_p_t": 0.0,
    "max_mean_p_t": 1.0,
    "min_mean_challenge_fraction": 0.0,
    "max_mean_challenge_fraction": 1.0,
    "min_role_switch_events_total": 0.0,
    "min_mean_aux_nontrivial_action_count": 0.0,
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


def artifact_paths_for_output_root(output_root: str | Path) -> dict[str, str]:
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


def coerce_contract(contract_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(contract_or_path, (str, Path)):
        return load_json(contract_or_path)
    return dict(contract_or_path)


def clone_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(contract)


def normalize_track(track: dict[str, Any]) -> dict[str, Any]:
    name = str(track["name"])
    return {
        "name": name,
        "required_for_promotion": bool(track.get("required_for_promotion", True)),
        "promotion_target": str(track.get("promotion_target", f"Promote `{name}` only after it beats the baseline gate.")),
        "runtime_overrides": dict(track.get("runtime_overrides", {})),
        "env_overrides": dict(track.get("env_overrides", {})),
        "acceptance_overrides": dict(track.get("acceptance_overrides", {})),
    }


def default_family_tracks(families: list[str] | None = None) -> list[dict[str, Any]]:
    family_names = list(families) if families else list(DEFAULT_FAMILY_ORDER)
    tracks: list[dict[str, Any]] = []
    for name in family_names:
        if name not in DEFAULT_FAMILY_PROFILES:
            raise SystemExit(f"unknown family track: {name}")
        profile = DEFAULT_FAMILY_PROFILES[name]
        tracks.append(
            normalize_track(
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


def resolve_family_tracks(contract: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tracks = contract.get("family_tracks")
    if not raw_tracks:
        return [
            normalize_track(
                {
                    "name": "global",
                    "required_for_promotion": True,
                    "promotion_target": "Promote the candidate globally only after it passes the baseline gate on the full sweep.",
                }
            )
        ]
    return [normalize_track(track) for track in raw_tracks]
