from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from typing import Any

from .erie_runtime import EnvironmentConfig, canonical_environment_config


REGIME_NAMES = (
    "balanced",
    "easy",
    "resource_rich",
    "sparse_energy",
    "toxic_band",
    "thermal_stress",
    "unstable_niche",
    "hard",
)


@dataclass(frozen=True)
class EnvironmentRegime:
    name: str
    description: str
    difficulty: float
    overrides: dict[str, Any]


REGIMES: dict[str, EnvironmentRegime] = {
    "balanced": EnvironmentRegime(
        name="balanced",
        description="Default mixed external state fields.",
        difficulty=1.0,
        overrides={},
    ),
    "easy": EnvironmentRegime(
        name="easy",
        description="Dense energy, low thermal/toxic drift, stable niche.",
        difficulty=0.6,
        overrides={
            "resource_patches": 5,
            "hazard_patches": 1,
            "shelter_patches": 3,
            "resource_regen": 0.006,
            "hazard_drift_sigma": 0.0004,
            "toxicity_drift_sigma": 0.0004,
            "flow_strength": 0.55,
            "species_field_gain_energy": 0.22,
            "species_field_gain_thermal": 0.08,
            "species_field_gain_toxicity": 0.10,
            "species_field_gain_niche": 0.20,
        },
    ),
    "resource_rich": EnvironmentRegime(
        name="resource_rich",
        description="Abundant but still moving energy gradients.",
        difficulty=0.8,
        overrides={
            "resource_patches": 7,
            "hazard_patches": 2,
            "shelter_patches": 2,
            "resource_regen": 0.007,
            "flow_strength": 0.75,
            "species_field_gain_energy": 0.25,
        },
    ),
    "sparse_energy": EnvironmentRegime(
        name="sparse_energy",
        description="Few energy patches with normal hazards.",
        difficulty=1.25,
        overrides={
            "resource_patches": 1,
            "hazard_patches": 3,
            "shelter_patches": 1,
            "resource_regen": 0.0012,
            "field_sigma_min": 3.0,
            "field_sigma_max": 6.5,
            "flow_strength": 0.80,
            "species_field_gain_energy": 0.12,
        },
    ),
    "toxic_band": EnvironmentRegime(
        name="toxic_band",
        description="High toxic contact pressure with moderate energy.",
        difficulty=1.45,
        overrides={
            "resource_patches": 3,
            "hazard_patches": 6,
            "shelter_patches": 1,
            "toxicity_drift_sigma": 0.0022,
            "hazard_drift_sigma": 0.0014,
            "flow_strength": 0.95,
            "species_field_gain_toxicity": 0.30,
            "species_field_gain_thermal": 0.17,
            "species_field_gain_niche": 0.10,
        },
    ),
    "thermal_stress": EnvironmentRegime(
        name="thermal_stress",
        description="Thermal stress dominates and moves through the field.",
        difficulty=1.35,
        overrides={
            "resource_patches": 3,
            "hazard_patches": 5,
            "shelter_patches": 2,
            "hazard_drift_sigma": 0.0024,
            "toxicity_drift_sigma": 0.0010,
            "flow_strength": 1.05,
            "flow_drift_sigma": 0.0010,
            "species_field_gain_thermal": 0.28,
        },
    ),
    "unstable_niche": EnvironmentRegime(
        name="unstable_niche",
        description="Niche support is weaker and environmental flow is less stable.",
        difficulty=1.30,
        overrides={
            "resource_patches": 3,
            "hazard_patches": 4,
            "shelter_patches": 1,
            "shelter_stability": 0.55,
            "flow_strength": 1.15,
            "flow_drift_sigma": 0.0015,
            "species_field_gain_niche": 0.08,
            "species_field_gain_thermal": 0.18,
            "species_field_gain_toxicity": 0.22,
        },
    ),
    "hard": EnvironmentRegime(
        name="hard",
        description="Sparse energy, high stress, toxic drift, and unstable flow.",
        difficulty=1.70,
        overrides={
            "resource_patches": 1,
            "hazard_patches": 6,
            "shelter_patches": 1,
            "resource_regen": 0.0010,
            "hazard_drift_sigma": 0.0025,
            "toxicity_drift_sigma": 0.0025,
            "field_sigma_min": 3.0,
            "field_sigma_max": 6.0,
            "flow_strength": 1.20,
            "flow_drift_sigma": 0.0018,
            "species_field_gain_energy": 0.10,
            "species_field_gain_thermal": 0.26,
            "species_field_gain_toxicity": 0.34,
            "species_field_gain_niche": 0.08,
        },
    ),
}


def normalize_regime_names(names: list[str] | tuple[str, ...] | None) -> list[str]:
    if not names:
        return ["balanced"]
    normalized: list[str] = []
    for raw_name in names:
        name = str(raw_name).strip()
        if not name:
            continue
        if name not in REGIMES:
            allowed = ", ".join(REGIME_NAMES)
            raise ValueError(f"unknown environment regime: {name}; allowed: {allowed}")
        if name not in normalized:
            normalized.append(name)
    return normalized or ["balanced"]


def environment_config_for_regime(base: EnvironmentConfig, regime_name: str) -> EnvironmentConfig:
    name = normalize_regime_names([regime_name])[0]
    regime = REGIMES[name]
    overrides = {
        key: value
        for key, value in regime.overrides.items()
        if hasattr(base, key)
    }
    return replace(base, **overrides)


def regime_manifest(base: EnvironmentConfig, regime_names: list[str] | tuple[str, ...] | None) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for name in normalize_regime_names(regime_names):
        regime = REGIMES[name]
        env_config = environment_config_for_regime(base, name)
        manifest.append(
            {
                "name": name,
                "description": regime.description,
                "difficulty": float(regime.difficulty),
                "overrides": dict(regime.overrides),
                "environment_config": asdict(env_config),
                "environment_config_canonical": canonical_environment_config(env_config),
            }
        )
    return manifest


def add_environment_regime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--environment-regimes",
        nargs="+",
        default=None,
        help=f"Environment regimes to evaluate per candidate. Allowed: {', '.join(REGIME_NAMES)}.",
    )
