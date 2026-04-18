from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import shutil

import numpy as np

from .common import choose_split, ensure_dir, save_json, save_jsonl, seed_everything
from .erie_runtime import (
    ACTIONS,
    ERIERuntime,
    EnvironmentConfig,
    LeniaERIEEnvironment,
    RuntimeConfig,
    RuntimeModels,
    _policy_action_cost,
    _softmax,
)
from .lenia_data import load_seed_catalog
from .trm_input_views import (
    build_trm_ag_input_view,
    build_trm_as_input_view,
    build_trm_bd_input_view,
    build_trm_bp_input_view,
    build_trm_mc_input_view,
    build_trm_vm_input_view,
    build_trm_wp_input_view,
    extract_centered_patch,
)


EPISODE_FAMILIES: tuple[str, ...] = (
    "energy_starved",
    "toxic_band",
    "fragile_boundary",
    "vent_edge",
    "uncertain_corridor",
)


ROLE_VIEW_SPECS: dict[str, dict[str, Any]] = {
    "trm_wp": {
        "role": "world_prediction",
        "input_key": "wp_input_view",
        "target_key": "wp_target_observation",
        "baseline_key": "wp_observation",
        "input_channels": 18,
        "target_channels": 11,
    },
    "trm_bd": {
        "role": "boundary_detection",
        "state_key": "bd_observation",
        "delta_key": "bd_delta_observation",
        "error_key": "bd_world_error",
        "sensor_gate_key": "bd_sensor_gate",
        "boundary_target_key": "bd_boundary_target",
        "permeability_target_key": "bd_permeability_target",
        "boundary_in_channels_total": 34,
    },
    "trm_bp": {
        "role": "boundary_permeability_control",
        "input_view_key": "bp_input_view",
        "boundary_patch_key": "bp_boundary_patch",
        "permeability_patch_key": "bp_permeability_patch",
        "observation_patch_key": "bp_observation_patch",
        "species_patch_key": "bp_species_patch",
        "flow_patch_key": "bp_flow_patch",
        "viability_state_key": "bp_viability_state",
        "target_permeability_patch_key": "bp_target_permeability_patch",
        "target_interface_gain_key": "bp_target_interface_gain",
        "target_aperture_gain_key": "bp_target_aperture_gain",
        "target_mode_key": "bp_target_mode",
        "input_channels": 21,
        "patch_size": 16,
    },
    "trm_vm": {
        "role": "viability_monitoring",
        "input_view_key": "vm_input_view",
        "target_state_key": "vm_target_state",
        "target_homeostatic_error_key": "vm_target_homeostatic_error",
        "target_risk_key": "vm_target_risk",
        "input_dim": 11,
    },
    "trm_as": {
        "role": "action_scoring",
        "input_view_key": "as_input_view",
        "target_logits_key": "as_target_logits",
        "target_policy_key": "as_target_policy",
        "target_action_key": "as_target_action",
        "input_dim": 19,
        "num_actions": len(ACTIONS),
    },
    "trm_ag": {
        "role": "action_gating",
        "input_view_key": "ag_input_view",
        "target_gated_logits_key": "ag_target_gated_logits",
        "target_inhibition_mask_key": "ag_target_inhibition_mask",
        "target_control_mode_key": "ag_target_control_mode",
        "input_dim": 22,
        "num_actions": len(ACTIONS),
        "num_modes": 3,
    },
    "trm_mc": {
        "role": "memory_context",
        "input_view_key": "mc_input_view",
        "window_mask_key": "mc_window_mask",
        "target_context_key": "mc_target_context_state",
        "target_action_bias_key": "mc_target_action_bias",
        "target_boundary_bias_key": "mc_target_boundary_bias",
        "window_size": 8,
        "input_dim": 44,
        "target_context_dim": 44,
        "num_actions": len(ACTIONS),
        "boundary_bias_dim": 3,
    },
}


def _policy_entropy(policy: np.ndarray) -> np.ndarray:
    clipped = np.clip(policy.astype(np.float64), 1e-8, 1.0)
    return (-np.sum(clipped * np.log(clipped), axis=-1)).astype(np.float32)


def _action_onehot(action: str, *, include_no_action: bool = True) -> np.ndarray:
    labels = [*ACTIONS]
    if include_no_action:
        labels.append("no_action")
    vec = np.zeros((len(labels),), dtype=np.float32)
    if action in labels:
        vec[labels.index(action)] = 1.0
    return vec


def _mc_feature_vector(
    runtime: ERIERuntime,
    viability_monitor: dict[str, Any],
    env_contact_state: np.ndarray,
    species_contact_state: np.ndarray,
    uncertainty_state: np.ndarray,
    prev_action: str,
    prev_action_cost: float,
    contact_interface_mass: float,
    flow_state: np.ndarray,
) -> np.ndarray:
    homeo_vec = viability_monitor["homeostatic_error_vector"].astype(np.float32)
    body = runtime.body
    interface_summary = np.array(
        [
            float(body.aperture_gain),
            float(np.clip(body.aperture_width_deg / 120.0, 0.0, 1.0)),
            float(np.sin(body.aperture_angle)),
            float(np.cos(body.aperture_angle)),
        ],
        dtype=np.float32,
    )
    interface_mass_feature = np.array([float(np.tanh(contact_interface_mass / 32.0))], dtype=np.float32)
    return np.concatenate(
        [
            viability_monitor["state"].astype(np.float32),
            homeo_vec.astype(np.float32),
            env_contact_state.astype(np.float32),
            species_contact_state.astype(np.float32),
            uncertainty_state.astype(np.float32),
            flow_state.astype(np.float32),
            interface_summary,
            interface_mass_feature,
            _action_onehot(prev_action, include_no_action=True),
            np.array([float(prev_action_cost)], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def _sample_flow_state(runtime: ERIERuntime) -> np.ndarray:
    y = int(np.clip(round(float(runtime.body.centroid_y)), 0, runtime.env.flow_y.shape[0] - 1))
    x = int(np.clip(round(float(runtime.body.centroid_x)), 0, runtime.env.flow_x.shape[1] - 1))
    return np.array([runtime.env.flow_y[y, x], runtime.env.flow_x[y, x]], dtype=np.float32)


def _mc_target_context_state(
    mc_input_view: np.ndarray,
    mc_window_mask: np.ndarray,
    decay: float = 0.75,
) -> np.ndarray:
    windows = np.asarray(mc_input_view, dtype=np.float32)
    masks = np.asarray(mc_window_mask, dtype=np.float32)
    time_steps, window_size, feature_dim = windows.shape
    context = np.zeros((time_steps, feature_dim), dtype=np.float32)
    weights_base = np.array([decay ** power for power in range(window_size - 1, -1, -1)], dtype=np.float32)
    for t in range(time_steps):
        weights = weights_base * masks[t]
        denom = float(weights.sum())
        if denom <= 0.0:
            continue
        normalized = (weights / denom).reshape(window_size, 1)
        context[t] = np.sum(windows[t] * normalized, axis=0).astype(np.float32)
    return context.astype(np.float32)


def _mc_target_boundary_bias(bp_target_mode: np.ndarray) -> np.ndarray:
    target_mode = np.asarray(bp_target_mode, dtype=np.int64).reshape(-1)
    bias = np.zeros((target_mode.shape[0], 3), dtype=np.float32)
    rows = np.arange(target_mode.shape[0], dtype=np.int64)
    valid = (target_mode >= 0) & (target_mode < 3)
    bias[rows[valid], target_mode[valid]] = 1.0
    return bias.astype(np.float32)


def _family_conditioned_mc_action_bias(
    family: str,
    base_action_bias: np.ndarray,
    viability_state: np.ndarray,
    env_contact_state: np.ndarray,
    species_contact_state: np.ndarray,
    uncertainty_state: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(base_action_bias, dtype=np.float32).copy()
    viability_state = np.asarray(viability_state, dtype=np.float32)
    env_contact_state = np.asarray(env_contact_state, dtype=np.float32)
    species_contact_state = np.asarray(species_contact_state, dtype=np.float32)
    uncertainty_state = np.asarray(uncertainty_state, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != len(ACTIONS):
        raise ValueError(f"expected [T, {len(ACTIONS)}] action bias logits, got {logits.shape}")

    approach_idx, withdraw_idx, intake_idx, seal_idx, reconfigure_idx = range(len(ACTIONS))
    energy = env_contact_state[:, 0] + 0.5 * species_contact_state[:, 0]
    thermal = env_contact_state[:, 1] + 0.5 * species_contact_state[:, 1]
    toxicity = env_contact_state[:, 2] + 0.5 * species_contact_state[:, 2]
    niche = env_contact_state[:, 3] + 0.5 * species_contact_state[:, 3]
    stress = thermal + toxicity
    uncertainty = np.mean(uncertainty_state, axis=1)
    g_deficit = np.clip(0.55 - viability_state[:, 0], 0.0, 1.0)
    b_deficit = np.clip(0.65 - viability_state[:, 1], 0.0, 1.0)

    if family == "fragile_boundary":
        logits[:, seal_idx] += 0.18 * stress + 0.12 * b_deficit
        logits[:, reconfigure_idx] += 0.16 * stress + 0.10 * b_deficit
        logits[:, intake_idx] -= 0.10 * stress
    elif family == "vent_edge":
        logits[:, withdraw_idx] += 0.14 * stress
        logits[:, approach_idx] += 0.05 * np.clip(energy - stress, 0.0, 1.0)
        logits[:, seal_idx] -= 0.06 * energy
        logits[:, reconfigure_idx] -= 0.04 * stress
    elif family == "uncertain_corridor":
        logits[:, reconfigure_idx] += 0.18 * uncertainty
        logits[:, approach_idx] += 0.08 * np.clip(uncertainty - niche, 0.0, 1.0)
        logits[:, seal_idx] -= 0.05 * uncertainty
    elif family == "toxic_band":
        logits[:, withdraw_idx] += 0.16 * toxicity
        logits[:, reconfigure_idx] += 0.10 * toxicity
        logits[:, intake_idx] -= 0.10 * toxicity
    elif family == "energy_starved":
        logits[:, approach_idx] += 0.12 * g_deficit + 0.06 * np.clip(energy - stress, 0.0, 1.0)
        logits[:, intake_idx] += 0.10 * g_deficit
        logits[:, seal_idx] -= 0.04 * g_deficit

    logits = logits - np.mean(logits, axis=1, keepdims=True)
    return logits.astype(np.float32)


def _family_conditioned_bp_targets(
    family: str,
    *,
    current_aperture_gain: float,
    target_permeability_patch: np.ndarray,
    target_interface_gain: float,
    target_aperture_gain: float,
    target_mode: int,
    env_contact_state: np.ndarray,
    species_contact_state: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    perm_patch = np.asarray(target_permeability_patch, dtype=np.float32).copy()
    interface_gain = float(target_interface_gain)
    aperture_gain = float(target_aperture_gain)
    mode = int(target_mode)

    env_contact_state = np.asarray(env_contact_state, dtype=np.float32).reshape(-1)
    species_contact_state = np.asarray(species_contact_state, dtype=np.float32).reshape(-1)
    energy = float(env_contact_state[0] + 0.5 * species_contact_state[0])
    thermal = float(env_contact_state[1] + 0.5 * species_contact_state[1])
    toxicity = float(env_contact_state[2] + 0.5 * species_contact_state[2])
    niche = float(env_contact_state[3] + 0.5 * species_contact_state[3])
    stress = thermal + toxicity

    if family == "vent_edge":
        stress_excess = max(0.0, stress - (energy + 0.35 * niche))
        if stress_excess > 0.0:
            mode = 2 if stress_excess > 0.08 else max(mode, 1)
            aperture_ceiling = current_aperture_gain - 0.012 * stress_excess + 0.004 * energy
            aperture_gain = float(np.clip(min(aperture_gain, aperture_ceiling), 0.05, 1.0))
            interface_gain = float(np.clip(interface_gain - 0.04 * stress_excess + 0.01 * niche, -1.0, 1.0))
            perm_patch = np.clip(perm_patch * (1.0 - 0.08 * stress_excess + 0.02 * niche), 0.0, 1.0).astype(np.float32)
    elif family == "fragile_boundary":
        stress_excess = max(0.0, stress - 0.25 * energy)
        if stress_excess > 0.0:
            mode = 2 if stress_excess > 0.3 else max(mode, 1)
            aperture_gain = float(np.clip(min(aperture_gain, current_aperture_gain), 0.05, 1.0))
            interface_gain = float(np.clip(interface_gain - 0.03 * stress_excess, -1.0, 1.0))

    return perm_patch.astype(np.float32), interface_gain, aperture_gain, mode


def _family_conditioned_ag_targets(
    family: str,
    *,
    as_target_logits: np.ndarray,
    viability_state: np.ndarray,
    homeostatic_error_vector: np.ndarray,
    viability_risk: np.ndarray,
    uncertainty_state: np.ndarray,
    env_contact_state: np.ndarray,
    species_contact_state: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    logits = np.asarray(as_target_logits, dtype=np.float32).copy()
    viability_state = np.asarray(viability_state, dtype=np.float32).reshape(-1)
    homeostatic_error_vector = np.asarray(homeostatic_error_vector, dtype=np.float32).reshape(-1)
    viability_risk = np.asarray(viability_risk, dtype=np.float32).reshape(-1)
    uncertainty_state = np.asarray(uncertainty_state, dtype=np.float32).reshape(-1)
    env_contact_state = np.asarray(env_contact_state, dtype=np.float32).reshape(-1)
    species_contact_state = np.asarray(species_contact_state, dtype=np.float32).reshape(-1)
    inhibition = np.zeros((len(ACTIONS),), dtype=np.float32)
    approach_idx, withdraw_idx, intake_idx, seal_idx, reconfigure_idx = range(len(ACTIONS))

    energy = float(env_contact_state[0] + 0.5 * species_contact_state[0])
    thermal = float(env_contact_state[1] + 0.5 * species_contact_state[1])
    toxicity = float(env_contact_state[2] + 0.5 * species_contact_state[2])
    niche = float(env_contact_state[3] + 0.5 * species_contact_state[3])
    stress = thermal + toxicity
    uncertainty = float(np.mean(uncertainty_state))
    g = float(viability_state[0])
    b = float(viability_state[1])
    g_deficit = max(0.0, 0.55 - g)
    b_deficit = max(0.0, 0.65 - b)
    g_overshoot = max(0.0, g - 0.55)
    total_homeostatic_error = float(np.sum(np.abs(homeostatic_error_vector)))
    risk = float(viability_risk[0]) if viability_risk.size else 0.0
    stress_excess = max(0.0, stress - (energy + 0.25 * niche))

    inhibition[approach_idx] = max(0.0, stress - energy + 0.5 * b_deficit)
    inhibition[intake_idx] = max(0.0, 0.75 * stress - 0.25 * energy + 0.7 * b_deficit + 0.4 * g_overshoot)
    inhibition[seal_idx] = max(0.0, 0.6 * g_deficit - 0.4 * stress - 0.2 * b_deficit)
    inhibition[withdraw_idx] = max(0.0, energy - stress - 0.8 * g_deficit)
    inhibition[reconfigure_idx] = max(0.0, 0.25 - uncertainty - 0.4 * stress)

    if family == "fragile_boundary":
        inhibition[intake_idx] += 0.22 + 0.35 * b_deficit + 0.20 * stress
        inhibition[approach_idx] += 0.18 * stress + 0.12 * b_deficit
        inhibition[seal_idx] -= 0.08 * np.clip(b_deficit + niche, 0.0, 1.0)
        inhibition[reconfigure_idx] -= 0.10 * np.clip(stress + b_deficit, 0.0, 1.0)
    elif family == "vent_edge":
        inhibition[approach_idx] += 0.22 * stress + 0.18 * stress_excess
        inhibition[intake_idx] += 0.18 * stress + 0.12 * stress_excess + 0.08 * g_overshoot
        inhibition[withdraw_idx] -= 0.10 * np.clip(stress_excess + 0.5 * b_deficit, 0.0, 1.0)
        inhibition[reconfigure_idx] -= 0.18 * np.clip(stress_excess + niche, 0.0, 1.0)
        inhibition[seal_idx] -= 0.04 * np.clip(b_deficit + niche - 0.5 * g_deficit, 0.0, 1.0)
    elif family == "uncertain_corridor":
        inhibition[approach_idx] -= 0.14 * np.clip(uncertainty + energy - stress, 0.0, 1.0)
        inhibition[reconfigure_idx] -= 0.18 * np.clip(uncertainty + niche - stress, 0.0, 1.0)
        inhibition[intake_idx] += 0.05 * np.clip(stress - 0.5 * energy, 0.0, 1.0)
        inhibition[seal_idx] += 0.08 * uncertainty
    elif family == "energy_starved":
        inhibition[approach_idx] -= 0.10 * g_deficit
        inhibition[intake_idx] -= 0.12 * g_deficit
        inhibition[seal_idx] += 0.08 * g_deficit
    elif family == "toxic_band":
        inhibition[intake_idx] += 0.15 * toxicity
        inhibition[approach_idx] += 0.12 * toxicity
        inhibition[withdraw_idx] -= 0.08 * toxicity

    inhibition = np.clip(inhibition, 0.0, 1.0).astype(np.float32)

    if (
        family == "vent_edge"
        and (stress_excess > 0.05 or risk > 0.48 or total_homeostatic_error > 0.14)
    ):
        control_mode = 2  # defensive
    elif family == "uncertain_corridor" and uncertainty > 0.24 and stress < energy + 0.18 and b_deficit < 0.10:
        control_mode = 0  # exploratory
    elif b_deficit > 0.12 or risk > 0.55 or stress > energy + 0.10:
        control_mode = 2  # defensive
    elif uncertainty > 0.28 and stress < energy + 0.15 and b_deficit < 0.08:
        control_mode = 0  # exploratory
    else:
        control_mode = 1  # maintenance

    mode_bias = np.array(
        [
            [0.10, -0.05, -0.10, -0.05, 0.10],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [-0.15, 0.15, -0.20, 0.20, 0.10],
        ],
        dtype=np.float32,
    )[control_mode]
    if family == "vent_edge":
        mode_bias = mode_bias + np.array([-0.05, 0.10, -0.12, 0.04, 0.18], dtype=np.float32)
    elif family == "uncertain_corridor":
        mode_bias = mode_bias + np.array([0.08, -0.02, -0.04, -0.06, 0.14], dtype=np.float32)
    elif family == "fragile_boundary":
        mode_bias = mode_bias + np.array([-0.06, 0.02, -0.10, 0.16, 0.10], dtype=np.float32)
    if family == "vent_edge" and control_mode == 2 and stress_excess > 0.03:
        inhibition[approach_idx] = 1.0
        inhibition[intake_idx] = 1.0
        inhibition[withdraw_idx] = min(inhibition[withdraw_idx], 0.0)
        inhibition[reconfigure_idx] = min(inhibition[reconfigure_idx], 0.0)
    if control_mode == 2:
        inhibition = (inhibition >= 0.30).astype(np.float32)
    elif control_mode == 0:
        inhibition = (inhibition >= 0.80).astype(np.float32)
    else:
        inhibition = (inhibition >= 0.50).astype(np.float32)
    gated_logits = logits + mode_bias - 4.0 * inhibition
    gated_logits = gated_logits - float(np.mean(gated_logits))
    return inhibition.astype(np.float32), int(control_mode), gated_logits.astype(np.float32)


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _integer(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _sample_episode_configs(
    family: str,
    base_runtime_config: RuntimeConfig,
    base_env_config: EnvironmentConfig,
    rng: np.random.Generator,
    defensive_family_bias: float = 0.0,
) -> tuple[RuntimeConfig, EnvironmentConfig]:
    runtime_kwargs: dict[str, Any] = {}
    env_kwargs: dict[str, Any] = {}

    if family == "energy_starved":
        runtime_kwargs.update(
            {
                "G0": _uniform(rng, 0.18, 0.38),
                "B0": _uniform(rng, 0.65, 0.85),
                "move_step": _uniform(rng, 2.4, 3.2),
                "aperture_gain": _uniform(rng, 0.42, 0.60),
            }
        )
        env_kwargs.update(
            {
                "resource_patches": _integer(rng, 1, 2),
                "hazard_patches": _integer(rng, 2, 3),
                "shelter_patches": _integer(rng, 1, 2),
                "resource_regen": _uniform(rng, 0.001, 0.0025),
                "hazard_drift_sigma": _uniform(rng, 0.0005, 0.0015),
                "toxicity_drift_sigma": _uniform(rng, 0.0005, 0.0015),
            }
        )
    elif family == "toxic_band":
        runtime_kwargs.update(
            {
                "G0": _uniform(rng, 0.45, 0.70),
                "B0": _uniform(rng, 0.45, 0.72),
                "aperture_gain": _uniform(rng, 0.25, 0.42),
                "aperture_width_deg": _uniform(rng, 50.0, 90.0),
            }
        )
        env_kwargs.update(
            {
                "resource_patches": _integer(rng, 2, 3),
                "hazard_patches": _integer(rng, 4, 5 + int(max(defensive_family_bias, 0.0))),
                "shelter_patches": _integer(rng, 1, 2),
                "field_sigma_min": _uniform(rng, 3.0, 4.5),
                "field_sigma_max": _uniform(rng, 6.0, 8.5),
                "toxicity_drift_sigma": _uniform(rng, 0.001, 0.0025 + 0.0005 * max(defensive_family_bias, 0.0)),
            }
        )
    elif family == "fragile_boundary":
        runtime_kwargs.update(
            {
                "G0": _uniform(rng, 0.42, 0.65),
                "B0": _uniform(rng, 0.22, 0.40),
                "aperture_gain": _uniform(rng, 0.18, 0.34),
                "aperture_width_deg": _uniform(rng, 45.0, 80.0),
                "observation_noise": _uniform(rng, 0.01, 0.02),
            }
        )
        env_kwargs.update(
            {
                "resource_patches": _integer(rng, 2, 3),
                "hazard_patches": _integer(rng, 3, 4 + int(max(defensive_family_bias, 0.0))),
                "shelter_patches": _integer(rng, 0, 1),
                "shelter_stability": _uniform(rng, 0.75, 0.95),
            }
        )
    elif family == "vent_edge":
        runtime_kwargs.update(
            {
                "G0": _uniform(rng, 0.35, 0.58),
                "B0": _uniform(rng, 0.50, 0.75),
                "move_step": _uniform(rng, 2.2, 3.0),
                "aperture_gain": _uniform(rng, 0.30, 0.50),
            }
        )
        env_kwargs.update(
            {
                "resource_patches": _integer(rng, 3, 4),
                "hazard_patches": _integer(rng, 4, 5 + int(max(defensive_family_bias, 0.0))),
                "shelter_patches": _integer(rng, 0, 1),
                "field_sigma_min": _uniform(rng, 3.5, 5.0),
                "field_sigma_max": _uniform(rng, 7.0, 9.5),
                "resource_regen": _uniform(rng, 0.002, 0.004),
            }
        )
    elif family == "uncertain_corridor":
        runtime_kwargs.update(
            {
                "G0": _uniform(rng, 0.30, 0.55),
                "B0": _uniform(rng, 0.42, 0.70),
                "observation_noise": _uniform(rng, 0.02, 0.05),
                "epistemic_scale": _uniform(rng, 1.1, 1.6),
                "aperture_width_deg": _uniform(rng, 60.0, 110.0),
            }
        )
        env_kwargs.update(
            {
                "resource_patches": _integer(rng, 2, 4),
                "hazard_patches": _integer(rng, 2, 4),
                "shelter_patches": _integer(rng, 1, 2),
                "hazard_drift_sigma": _uniform(rng, 0.001, 0.0025),
                "toxicity_drift_sigma": _uniform(rng, 0.001, 0.0025),
                "shelter_stability": _uniform(rng, 0.7, 0.95),
            }
        )
    else:
        raise ValueError(f"unknown episode family: {family}")

    runtime_cfg = replace(base_runtime_config, **runtime_kwargs)
    env_cfg = replace(base_env_config, **env_kwargs)
    return runtime_cfg, env_cfg


def sample_episode_configs_for_family(
    family: str,
    base_runtime_config: RuntimeConfig,
    base_env_config: EnvironmentConfig,
    seed: int,
    defensive_family_bias: float = 0.0,
) -> tuple[RuntimeConfig, EnvironmentConfig]:
    rng = np.random.default_rng(int(seed))
    return _sample_episode_configs(
        family=family,
        base_runtime_config=base_runtime_config,
        base_env_config=base_env_config,
        rng=rng,
        defensive_family_bias=defensive_family_bias,
    )


def _family_pool(defensive_family_bias: float) -> tuple[str, ...]:
    pool = list(EPISODE_FAMILIES)
    repeats = max(0, int(round(defensive_family_bias)))
    if repeats <= 0:
        return tuple(pool)
    pool.extend(["toxic_band"] * repeats)
    pool.extend(["fragile_boundary"] * repeats)
    pool.extend(["vent_edge"] * repeats)
    return tuple(pool)


def _collect_episode_samples(runtime: ERIERuntime, *, family: str | None = None) -> dict[str, np.ndarray]:
    bp_patch_size = int(ROLE_VIEW_SPECS["trm_bp"]["patch_size"])
    mc_window_size = int(ROLE_VIEW_SPECS["trm_mc"]["window_size"])
    wp_observation: list[np.ndarray] = []
    wp_sensor_gate: list[np.ndarray] = []
    wp_species_fields: list[np.ndarray] = []
    wp_flow_channels: list[np.ndarray] = []
    wp_input_view: list[np.ndarray] = []
    wp_target_observation: list[np.ndarray] = []

    bd_observation: list[np.ndarray] = []
    bd_world_error: list[np.ndarray] = []
    bd_sensor_gate: list[np.ndarray] = []
    bd_delta_observation: list[np.ndarray] = []
    bd_input_view: list[np.ndarray] = []
    bd_boundary_target: list[np.ndarray] = []
    bd_permeability_target: list[np.ndarray] = []

    bp_boundary_patch: list[np.ndarray] = []
    bp_permeability_patch: list[np.ndarray] = []
    bp_observation_patch: list[np.ndarray] = []
    bp_species_patch: list[np.ndarray] = []
    bp_flow_patch: list[np.ndarray] = []
    bp_viability_state: list[np.ndarray] = []
    bp_input_view: list[np.ndarray] = []
    bp_target_permeability_patch: list[np.ndarray] = []
    bp_target_interface_gain: list[np.ndarray] = []
    bp_target_aperture_gain: list[np.ndarray] = []
    bp_target_mode: list[np.ndarray] = []

    vm_viability_state: list[np.ndarray] = []
    vm_contact_state: list[np.ndarray] = []
    vm_species_contact_state: list[np.ndarray] = []
    vm_action_cost: list[np.ndarray] = []
    vm_input_view: list[np.ndarray] = []
    vm_target_state: list[np.ndarray] = []
    vm_target_homeostatic_error: list[np.ndarray] = []
    vm_target_risk: list[np.ndarray] = []

    as_viability_state: list[np.ndarray] = []
    as_action_scores: list[np.ndarray] = []
    as_uncertainty_state: list[np.ndarray] = []
    as_env_contact_state: list[np.ndarray] = []
    as_species_contact_state: list[np.ndarray] = []
    as_input_view: list[np.ndarray] = []
    as_target_logits: list[np.ndarray] = []
    as_target_policy: list[np.ndarray] = []
    as_target_action: list[np.ndarray] = []
    ag_input_view: list[np.ndarray] = []
    ag_target_gated_logits: list[np.ndarray] = []
    ag_target_inhibition_mask: list[np.ndarray] = []
    ag_target_control_mode: list[np.ndarray] = []
    mc_step_features: list[np.ndarray] = []

    prev_observation = np.zeros_like(runtime.last_observation, dtype=np.float32)
    prev_action = "no_action"
    prev_action_cost = 0.0

    for t in range(runtime.cfg.steps):
        runtime.env.step_lenia()
        observation, sensor_gate, _, boundary = runtime._observe()
        permeability = runtime._body_fields()[2]
        boundary_obs = np.stack([boundary, permeability], axis=-1).astype(np.float32)
        runtime._belief_update(observation, sensor_gate, boundary_obs)
        species_fields = runtime.env.species_field_channels().astype(np.float32)
        flow_channels = np.stack([runtime.env.flow_y, runtime.env.flow_x], axis=-1).astype(np.float32)
        delta_observation = (observation - prev_observation).astype(np.float32)

        scores, _ = runtime._policy_scores()
        viability_monitor = runtime._monitor_viability(action_cost=0.0)
        policy = _softmax((-runtime.cfg.beta_pi * scores).astype(np.float32))
        uniform = np.full_like(policy, 1.0 / len(ACTIONS), dtype=np.float32)
        target_policy = (0.9 * policy + 0.1 * uniform).astype(np.float32)
        policy_probs = target_policy.astype(np.float64)
        policy_probs = policy_probs / max(float(policy_probs.sum()), 1e-12)
        action_index = int(runtime.rng.choice(len(ACTIONS), p=policy_probs))
        action = ACTIONS[action_index]
        action_cost = _policy_action_cost(action)
        contact = runtime._contact_stats(runtime.body)
        species_contact = runtime._species_contact_stats(runtime.body)
        env_contact_state = np.array(
            [contact["energy"], contact["thermal"], contact["toxicity"], contact["niche"]],
            dtype=np.float32,
        )
        species_contact_state = np.array(
            [
                species_contact["species_energy"],
                species_contact["species_thermal"],
                species_contact["species_toxicity"],
                species_contact["species_niche"],
            ],
            dtype=np.float32,
        )
        uncertainty_state = runtime._uncertainty_state()
        mc_step_features.append(
            _mc_feature_vector(
                runtime,
                viability_monitor,
                env_contact_state,
                species_contact_state,
                uncertainty_state,
                prev_action,
                prev_action_cost,
                float(contact["interface_mass"]),
                _sample_flow_state(runtime),
            )
        )

        next_body = runtime._prospective_body(action)
        next_G, next_B = runtime._predicted_viability(next_body, action)
        target_state = np.array([next_G, next_B], dtype=np.float32)
        target_error = np.abs(
            target_state - np.array([runtime.cfg.G_target, runtime.cfg.B_target], dtype=np.float32)
        ).astype(np.float32)
        target_risk = np.array(
            [[float(next_G < runtime.cfg.tau_G or next_B < runtime.cfg.tau_B)]],
            dtype=np.float32,
        )
        next_body_fields = runtime._body_fields(next_body)
        next_boundary = next_body_fields[1]
        next_permeability = next_body_fields[2]
        next_sensor_gate = np.clip(next_body_fields[2][..., None] + 0.05 * next_body_fields[0][..., None], 0.0, 1.0)
        next_env_channels = runtime.env.environment_channels().astype(np.float32)
        next_observation = runtime._observation_mapping(
            env_channels=next_env_channels,
            sensor_gate=next_sensor_gate.astype(np.float32),
            thermal_stress=runtime.env.thermal_stress,
            toxicity=runtime.env.toxicity,
            niche_stability=runtime.env.niche_stability,
        )["observation"].astype(np.float32)

        wp_observation.append(observation.astype(np.float32))
        wp_sensor_gate.append(sensor_gate.astype(np.float32))
        wp_species_fields.append(species_fields)
        wp_flow_channels.append(flow_channels)
        wp_input_view.append(
            build_trm_wp_input_view(
                observation.astype(np.float32),
                sensor_gate.astype(np.float32),
                species_fields,
                flow_channels,
            )
        )
        wp_target_observation.append(next_observation)

        bd_observation.append(observation.astype(np.float32))
        bd_world_error.append(runtime.last_world_error.astype(np.float32))
        bd_sensor_gate.append(sensor_gate.astype(np.float32))
        bd_delta_observation.append(delta_observation)
        bd_input_view.append(
            build_trm_bd_input_view(
                observation.astype(np.float32),
                runtime.last_world_error.astype(np.float32),
                sensor_gate.astype(np.float32),
                delta_observation,
            )
        )
        bd_boundary_target.append(boundary[..., None].astype(np.float32))
        bd_permeability_target.append(permeability[..., None].astype(np.float32))

        center_y = runtime.body.centroid_y
        center_x = runtime.body.centroid_x
        boundary_patch = extract_centered_patch(boundary[..., None], center_y, center_x, bp_patch_size)
        permeability_patch = extract_centered_patch(permeability[..., None], center_y, center_x, bp_patch_size)
        observation_patch = extract_centered_patch(observation, center_y, center_x, bp_patch_size)
        species_patch = extract_centered_patch(species_fields, center_y, center_x, bp_patch_size)
        flow_patch = extract_centered_patch(flow_channels, center_y, center_x, bp_patch_size)
        target_permeability_patch = extract_centered_patch(next_permeability[..., None], center_y, center_x, bp_patch_size)
        current_interface = float(np.mean(boundary_patch[..., 0] * permeability_patch[..., 0]))
        next_interface_patch = extract_centered_patch(next_boundary[..., None], center_y, center_x, bp_patch_size)[..., 0] * target_permeability_patch[..., 0]
        next_interface = float(np.mean(next_interface_patch))
        viability_state = np.array([runtime.body.G, runtime.body.B], dtype=np.float32)
        bp_boundary_patch.append(boundary_patch.astype(np.float32))
        bp_permeability_patch.append(permeability_patch.astype(np.float32))
        bp_observation_patch.append(observation_patch.astype(np.float32))
        bp_species_patch.append(species_patch.astype(np.float32))
        bp_flow_patch.append(flow_patch.astype(np.float32))
        bp_viability_state.append(viability_state.astype(np.float32))
        bp_input_view.append(
            build_trm_bp_input_view(
                boundary_patch.astype(np.float32),
                permeability_patch.astype(np.float32),
                observation_patch.astype(np.float32),
                species_patch.astype(np.float32),
                flow_patch.astype(np.float32),
                viability_state.astype(np.float32),
            )
        )
        conditioned_perm_patch, conditioned_interface_gain, conditioned_aperture_gain, conditioned_mode = _family_conditioned_bp_targets(
            family or "energy_starved",
            current_aperture_gain=float(runtime.body.aperture_gain),
            target_permeability_patch=target_permeability_patch.astype(np.float32),
            target_interface_gain=float(next_interface - current_interface),
            target_aperture_gain=float(next_body.aperture_gain),
            target_mode=_bp_target_mode(action),
            env_contact_state=env_contact_state,
            species_contact_state=species_contact_state,
        )
        bp_target_permeability_patch.append(conditioned_perm_patch.astype(np.float32))
        bp_target_interface_gain.append(np.array([conditioned_interface_gain], dtype=np.float32))
        bp_target_aperture_gain.append(np.array([conditioned_aperture_gain], dtype=np.float32))
        bp_target_mode.append(np.array(conditioned_mode, dtype=np.int64))

        vm_viability_state.append(viability_state)
        vm_contact_state.append(env_contact_state)
        vm_species_contact_state.append(species_contact_state)
        vm_action_cost.append(np.array([action_cost], dtype=np.float32))
        vm_input_view.append(
            build_trm_vm_input_view(
                np.array([runtime.body.G, runtime.body.B], dtype=np.float32),
                env_contact_state,
                species_contact_state,
                np.array([action_cost], dtype=np.float32),
            )
        )
        vm_target_state.append(target_state)
        vm_target_homeostatic_error.append(target_error)
        vm_target_risk.append(target_risk[0])

        as_viability_state.append(viability_monitor["state"].astype(np.float32))
        as_action_scores.append(scores.astype(np.float32))
        as_uncertainty_state.append(uncertainty_state.astype(np.float32))
        as_env_contact_state.append(env_contact_state)
        as_species_contact_state.append(species_contact_state)
        as_input_view.append(
            build_trm_as_input_view(
                viability_monitor["state"].astype(np.float32),
                scores.astype(np.float32),
                uncertainty_state.astype(np.float32),
                env_contact_state,
                species_contact_state,
            )
        )
        target_logits = (-runtime.cfg.beta_pi * scores).astype(np.float32)
        target_logits = target_logits - float(np.mean(target_logits))
        as_target_logits.append(target_logits.astype(np.float32))
        as_target_policy.append(target_policy.astype(np.float32))
        as_target_action.append(np.array(action_index, dtype=np.int64))

        ag_viability_risk = np.array([viability_monitor["risk"]], dtype=np.float32)
        ag_input_view.append(
            build_trm_ag_input_view(
                target_logits.astype(np.float32),
                viability_monitor["state"].astype(np.float32),
                viability_monitor["homeostatic_error_vector"].astype(np.float32),
                ag_viability_risk,
                uncertainty_state.astype(np.float32),
                env_contact_state,
                species_contact_state,
            )
        )
        ag_inhibition_mask, ag_control_mode, ag_gated_logits = _family_conditioned_ag_targets(
            family or "energy_starved",
            as_target_logits=target_logits.astype(np.float32),
            viability_state=viability_monitor["state"].astype(np.float32),
            homeostatic_error_vector=viability_monitor["homeostatic_error_vector"].astype(np.float32),
            viability_risk=ag_viability_risk,
            uncertainty_state=uncertainty_state.astype(np.float32),
            env_contact_state=env_contact_state,
            species_contact_state=species_contact_state,
        )
        ag_target_gated_logits.append(ag_gated_logits.astype(np.float32))
        ag_target_inhibition_mask.append(ag_inhibition_mask.astype(np.float32))
        ag_target_control_mode.append(np.array(ag_control_mode, dtype=np.int64))

        runtime._apply_action(action)
        runtime.env.update_fields(runtime.body, action)
        prev_observation = observation.astype(np.float32)
        prev_action = action
        prev_action_cost = action_cost
        if runtime._update_death():
            break

    mc_step_feature_array = np.stack(mc_step_features, axis=0).astype(np.float32)
    mc_input_view, mc_window_mask = build_trm_mc_input_view(mc_step_feature_array, mc_window_size)
    mc_target_context_state = _mc_target_context_state(mc_input_view, mc_window_mask)
    mc_target_action_bias = _family_conditioned_mc_action_bias(
        family or "energy_starved",
        np.stack(as_target_logits, axis=0).astype(np.float32),
        np.stack(vm_viability_state, axis=0).astype(np.float32),
        np.stack(vm_contact_state, axis=0).astype(np.float32),
        np.stack(vm_species_contact_state, axis=0).astype(np.float32),
        np.stack(as_uncertainty_state, axis=0).astype(np.float32),
    )
    mc_target_boundary_bias = _mc_target_boundary_bias(np.asarray(bp_target_mode, dtype=np.int64))

    return {
        "wp_observation": np.stack(wp_observation, axis=0).astype(np.float32),
        "wp_sensor_gate": np.stack(wp_sensor_gate, axis=0).astype(np.float32),
        "wp_species_fields": np.stack(wp_species_fields, axis=0).astype(np.float32),
        "wp_flow_channels": np.stack(wp_flow_channels, axis=0).astype(np.float32),
        "wp_input_view": np.stack(wp_input_view, axis=0).astype(np.float32),
        "wp_target_observation": np.stack(wp_target_observation, axis=0).astype(np.float32),
        "bd_observation": np.stack(bd_observation, axis=0).astype(np.float32),
        "bd_world_error": np.stack(bd_world_error, axis=0).astype(np.float32),
        "bd_sensor_gate": np.stack(bd_sensor_gate, axis=0).astype(np.float32),
        "bd_delta_observation": np.stack(bd_delta_observation, axis=0).astype(np.float32),
        "bd_input_view": np.stack(bd_input_view, axis=0).astype(np.float32),
        "bd_boundary_target": np.stack(bd_boundary_target, axis=0).astype(np.float32),
        "bd_permeability_target": np.stack(bd_permeability_target, axis=0).astype(np.float32),
        "bp_boundary_patch": np.stack(bp_boundary_patch, axis=0).astype(np.float32),
        "bp_permeability_patch": np.stack(bp_permeability_patch, axis=0).astype(np.float32),
        "bp_observation_patch": np.stack(bp_observation_patch, axis=0).astype(np.float32),
        "bp_species_patch": np.stack(bp_species_patch, axis=0).astype(np.float32),
        "bp_flow_patch": np.stack(bp_flow_patch, axis=0).astype(np.float32),
        "bp_viability_state": np.stack(bp_viability_state, axis=0).astype(np.float32),
        "bp_input_view": np.stack(bp_input_view, axis=0).astype(np.float32),
        "bp_target_permeability_patch": np.stack(bp_target_permeability_patch, axis=0).astype(np.float32),
        "bp_target_interface_gain": np.stack(bp_target_interface_gain, axis=0).astype(np.float32),
        "bp_target_aperture_gain": np.stack(bp_target_aperture_gain, axis=0).astype(np.float32),
        "bp_target_mode": np.asarray(bp_target_mode, dtype=np.int64),
        "vm_viability_state": np.stack(vm_viability_state, axis=0).astype(np.float32),
        "vm_contact_state": np.stack(vm_contact_state, axis=0).astype(np.float32),
        "vm_species_contact_state": np.stack(vm_species_contact_state, axis=0).astype(np.float32),
        "vm_action_cost": np.stack(vm_action_cost, axis=0).astype(np.float32),
        "vm_input_view": np.stack(vm_input_view, axis=0).astype(np.float32),
        "vm_target_state": np.stack(vm_target_state, axis=0).astype(np.float32),
        "vm_target_homeostatic_error": np.stack(vm_target_homeostatic_error, axis=0).astype(np.float32),
        "vm_target_risk": np.stack(vm_target_risk, axis=0).astype(np.float32),
        "as_viability_state": np.stack(as_viability_state, axis=0).astype(np.float32),
        "as_action_scores": np.stack(as_action_scores, axis=0).astype(np.float32),
        "as_uncertainty_state": np.stack(as_uncertainty_state, axis=0).astype(np.float32),
        "as_env_contact_state": np.stack(as_env_contact_state, axis=0).astype(np.float32),
        "as_species_contact_state": np.stack(as_species_contact_state, axis=0).astype(np.float32),
        "as_input_view": np.stack(as_input_view, axis=0).astype(np.float32),
        "as_target_logits": np.stack(as_target_logits, axis=0).astype(np.float32),
        "as_target_policy": np.stack(as_target_policy, axis=0).astype(np.float32),
        "as_target_action": np.asarray(as_target_action, dtype=np.int64),
        "ag_input_view": np.stack(ag_input_view, axis=0).astype(np.float32),
        "ag_target_gated_logits": np.stack(ag_target_gated_logits, axis=0).astype(np.float32),
        "ag_target_inhibition_mask": np.stack(ag_target_inhibition_mask, axis=0).astype(np.float32),
        "ag_target_control_mode": np.asarray(ag_target_control_mode, dtype=np.int64),
        "mc_input_view": mc_input_view.astype(np.float32),
        "mc_window_mask": mc_window_mask.astype(np.float32),
        "mc_target_context_state": mc_target_context_state.astype(np.float32),
        "mc_target_action_bias": mc_target_action_bias.astype(np.float32),
        "mc_target_boundary_bias": mc_target_boundary_bias.astype(np.float32),
    }


def _collect_episode_samples_with_shaping(
    runtime: ERIERuntime,
    *,
    family: str | None = None,
    target_band_weight: float = 0.0,
    target_g_overshoot_weight: float = 0.0,
) -> dict[str, np.ndarray]:
    bp_patch_size = int(ROLE_VIEW_SPECS["trm_bp"]["patch_size"])
    mc_window_size = int(ROLE_VIEW_SPECS["trm_mc"]["window_size"])
    wp_observation: list[np.ndarray] = []
    wp_sensor_gate: list[np.ndarray] = []
    wp_species_fields: list[np.ndarray] = []
    wp_flow_channels: list[np.ndarray] = []
    wp_input_view: list[np.ndarray] = []
    wp_target_observation: list[np.ndarray] = []

    bd_observation: list[np.ndarray] = []
    bd_world_error: list[np.ndarray] = []
    bd_sensor_gate: list[np.ndarray] = []
    bd_delta_observation: list[np.ndarray] = []
    bd_input_view: list[np.ndarray] = []
    bd_boundary_target: list[np.ndarray] = []
    bd_permeability_target: list[np.ndarray] = []

    bp_boundary_patch: list[np.ndarray] = []
    bp_permeability_patch: list[np.ndarray] = []
    bp_observation_patch: list[np.ndarray] = []
    bp_species_patch: list[np.ndarray] = []
    bp_flow_patch: list[np.ndarray] = []
    bp_viability_state: list[np.ndarray] = []
    bp_input_view: list[np.ndarray] = []
    bp_target_permeability_patch: list[np.ndarray] = []
    bp_target_interface_gain: list[np.ndarray] = []
    bp_target_aperture_gain: list[np.ndarray] = []
    bp_target_mode: list[np.ndarray] = []

    vm_viability_state: list[np.ndarray] = []
    vm_contact_state: list[np.ndarray] = []
    vm_species_contact_state: list[np.ndarray] = []
    vm_action_cost: list[np.ndarray] = []
    vm_input_view: list[np.ndarray] = []
    vm_target_state: list[np.ndarray] = []
    vm_target_homeostatic_error: list[np.ndarray] = []
    vm_target_risk: list[np.ndarray] = []

    as_viability_state: list[np.ndarray] = []
    as_action_scores: list[np.ndarray] = []
    as_uncertainty_state: list[np.ndarray] = []
    as_env_contact_state: list[np.ndarray] = []
    as_species_contact_state: list[np.ndarray] = []
    as_input_view: list[np.ndarray] = []
    as_target_logits: list[np.ndarray] = []
    as_target_policy: list[np.ndarray] = []
    as_target_action: list[np.ndarray] = []
    executed_action: list[np.ndarray] = []
    ag_input_view: list[np.ndarray] = []
    ag_target_gated_logits: list[np.ndarray] = []
    ag_target_inhibition_mask: list[np.ndarray] = []
    ag_target_control_mode: list[np.ndarray] = []
    mc_step_features: list[np.ndarray] = []

    prev_observation = np.zeros_like(runtime.last_observation, dtype=np.float32)
    prev_action = "no_action"
    prev_action_cost = 0.0

    for _ in range(runtime.cfg.steps):
        runtime.env.step_lenia()
        observation, sensor_gate, _, boundary = runtime._observe()
        permeability = runtime._body_fields()[2]
        boundary_obs = np.stack([boundary, permeability], axis=-1).astype(np.float32)
        runtime._belief_update(observation, sensor_gate, boundary_obs)
        species_fields = runtime.env.species_field_channels().astype(np.float32)
        flow_channels = np.stack([runtime.env.flow_y, runtime.env.flow_x], axis=-1).astype(np.float32)
        delta_observation = (observation - prev_observation).astype(np.float32)

        scores, score_diag = runtime._policy_scores()
        viability_monitor = runtime._monitor_viability(action_cost=0.0)
        uncertainty_state = runtime._uncertainty_state()
        contact = runtime._contact_stats(runtime.body)
        species_contact = runtime._species_contact_stats(runtime.body)
        env_contact_state = np.array(
            [contact["energy"], contact["thermal"], contact["toxicity"], contact["niche"]],
            dtype=np.float32,
        )
        species_contact_state = np.array(
            [
                species_contact["species_energy"],
                species_contact["species_thermal"],
                species_contact["species_toxicity"],
                species_contact["species_niche"],
            ],
            dtype=np.float32,
        )
        mc_step_features.append(
            _mc_feature_vector(
                runtime,
                viability_monitor,
                env_contact_state,
                species_contact_state,
                uncertainty_state,
                prev_action,
                prev_action_cost,
                float(contact["interface_mass"]),
                _sample_flow_state(runtime),
            )
        )

        band_penalty = np.array(
            [
                abs(float(score_diag[action]["pred_G"]) - runtime.cfg.G_target)
                + abs(float(score_diag[action]["pred_B"]) - runtime.cfg.B_target)
                for action in ACTIONS
            ],
            dtype=np.float32,
        )
        g_overshoot = np.array(
            [max(0.0, float(score_diag[action]["pred_G"]) - runtime.cfg.G_target) for action in ACTIONS],
            dtype=np.float32,
        )
        shaped_scores = (
            scores.astype(np.float32)
            + float(target_band_weight) * band_penalty
            + float(target_g_overshoot_weight) * g_overshoot
        ).astype(np.float32)
        target_policy = _softmax((-runtime.cfg.beta_pi * shaped_scores).astype(np.float32))
        uniform = np.full_like(target_policy, 1.0 / len(ACTIONS), dtype=np.float32)
        target_policy = (0.9 * target_policy + 0.1 * uniform).astype(np.float32)
        policy_probs = target_policy.astype(np.float64)
        policy_probs = policy_probs / max(float(policy_probs.sum()), 1e-12)
        target_action_index = int(runtime.rng.choice(len(ACTIONS), p=policy_probs))
        target_action = ACTIONS[target_action_index]
        if runtime.cfg.policy_mode == "random":
            executed = str(runtime.rng.choice(ACTIONS))
            action_index = ACTIONS.index(executed)
        elif runtime.cfg.policy_mode == "no_action":
            executed = "no_action"
            action_index = len(ACTIONS)
        else:
            executed = target_action
            action_index = target_action_index
        action_cost = _policy_action_cost(executed)

        next_body = runtime._prospective_body(executed)
        next_G, next_B = runtime._predicted_viability(next_body, executed)
        target_state = np.array([next_G, next_B], dtype=np.float32)
        target_error = np.abs(
            target_state - np.array([runtime.cfg.G_target, runtime.cfg.B_target], dtype=np.float32)
        ).astype(np.float32)
        target_risk = np.array(
            [[float(next_G < runtime.cfg.tau_G or next_B < runtime.cfg.tau_B)]],
            dtype=np.float32,
        )
        next_body_fields = runtime._body_fields(next_body)
        next_boundary = next_body_fields[1]
        next_permeability = next_body_fields[2]
        next_sensor_gate = np.clip(next_body_fields[2][..., None] + 0.05 * next_body_fields[0][..., None], 0.0, 1.0)
        next_env_channels = runtime.env.environment_channels().astype(np.float32)
        next_observation = runtime._observation_mapping(
            env_channels=next_env_channels,
            sensor_gate=next_sensor_gate.astype(np.float32),
            thermal_stress=runtime.env.thermal_stress,
            toxicity=runtime.env.toxicity,
            niche_stability=runtime.env.niche_stability,
        )["observation"].astype(np.float32)

        wp_observation.append(observation.astype(np.float32))
        wp_sensor_gate.append(sensor_gate.astype(np.float32))
        wp_species_fields.append(species_fields)
        wp_flow_channels.append(flow_channels)
        wp_input_view.append(
            build_trm_wp_input_view(
                observation.astype(np.float32),
                sensor_gate.astype(np.float32),
                species_fields,
                flow_channels,
            )
        )
        wp_target_observation.append(next_observation)

        bd_observation.append(observation.astype(np.float32))
        bd_world_error.append(runtime.last_world_error.astype(np.float32))
        bd_sensor_gate.append(sensor_gate.astype(np.float32))
        bd_delta_observation.append(delta_observation)
        bd_input_view.append(
            build_trm_bd_input_view(
                observation.astype(np.float32),
                runtime.last_world_error.astype(np.float32),
                sensor_gate.astype(np.float32),
                delta_observation,
            )
        )
        bd_boundary_target.append(boundary[..., None].astype(np.float32))
        bd_permeability_target.append(permeability[..., None].astype(np.float32))

        center_y = runtime.body.centroid_y
        center_x = runtime.body.centroid_x
        boundary_patch = extract_centered_patch(boundary[..., None], center_y, center_x, bp_patch_size)
        permeability_patch = extract_centered_patch(permeability[..., None], center_y, center_x, bp_patch_size)
        observation_patch = extract_centered_patch(observation, center_y, center_x, bp_patch_size)
        species_patch = extract_centered_patch(species_fields, center_y, center_x, bp_patch_size)
        flow_patch = extract_centered_patch(flow_channels, center_y, center_x, bp_patch_size)
        target_permeability_patch = extract_centered_patch(next_permeability[..., None], center_y, center_x, bp_patch_size)
        current_interface = float(np.mean(boundary_patch[..., 0] * permeability_patch[..., 0]))
        next_interface_patch = extract_centered_patch(next_boundary[..., None], center_y, center_x, bp_patch_size)[..., 0] * target_permeability_patch[..., 0]
        next_interface = float(np.mean(next_interface_patch))
        current_viability_state = np.array([runtime.body.G, runtime.body.B], dtype=np.float32)
        bp_boundary_patch.append(boundary_patch.astype(np.float32))
        bp_permeability_patch.append(permeability_patch.astype(np.float32))
        bp_observation_patch.append(observation_patch.astype(np.float32))
        bp_species_patch.append(species_patch.astype(np.float32))
        bp_flow_patch.append(flow_patch.astype(np.float32))
        bp_viability_state.append(current_viability_state.astype(np.float32))
        bp_input_view.append(
            build_trm_bp_input_view(
                boundary_patch.astype(np.float32),
                permeability_patch.astype(np.float32),
                observation_patch.astype(np.float32),
                species_patch.astype(np.float32),
                flow_patch.astype(np.float32),
                current_viability_state.astype(np.float32),
            )
        )
        conditioned_perm_patch, conditioned_interface_gain, conditioned_aperture_gain, conditioned_mode = _family_conditioned_bp_targets(
            family or "energy_starved",
            current_aperture_gain=float(runtime.body.aperture_gain),
            target_permeability_patch=target_permeability_patch.astype(np.float32),
            target_interface_gain=float(next_interface - current_interface),
            target_aperture_gain=float(next_body.aperture_gain),
            target_mode=_bp_target_mode(executed),
            env_contact_state=env_contact_state,
            species_contact_state=species_contact_state,
        )
        bp_target_permeability_patch.append(conditioned_perm_patch.astype(np.float32))
        bp_target_interface_gain.append(np.array([conditioned_interface_gain], dtype=np.float32))
        bp_target_aperture_gain.append(np.array([conditioned_aperture_gain], dtype=np.float32))
        bp_target_mode.append(np.array(conditioned_mode, dtype=np.int64))

        vm_viability_state.append(current_viability_state)
        vm_contact_state.append(env_contact_state)
        vm_species_contact_state.append(species_contact_state)
        vm_action_cost.append(np.array([action_cost], dtype=np.float32))
        vm_input_view.append(
            build_trm_vm_input_view(
                np.array([runtime.body.G, runtime.body.B], dtype=np.float32),
                env_contact_state,
                species_contact_state,
                np.array([action_cost], dtype=np.float32),
            )
        )
        vm_target_state.append(target_state)
        vm_target_homeostatic_error.append(target_error)
        vm_target_risk.append(target_risk[0])

        as_viability_state.append(viability_monitor["state"].astype(np.float32))
        as_action_scores.append(scores.astype(np.float32))
        as_uncertainty_state.append(uncertainty_state.astype(np.float32))
        as_env_contact_state.append(env_contact_state)
        as_species_contact_state.append(species_contact_state)
        as_input_view.append(
            build_trm_as_input_view(
                viability_monitor["state"].astype(np.float32),
                scores.astype(np.float32),
                uncertainty_state.astype(np.float32),
                env_contact_state,
                species_contact_state,
            )
        )
        target_logits = (-runtime.cfg.beta_pi * shaped_scores).astype(np.float32)
        target_logits = target_logits - float(np.mean(target_logits))
        as_target_logits.append(target_logits.astype(np.float32))
        as_target_policy.append(target_policy.astype(np.float32))
        as_target_action.append(np.array(target_action_index, dtype=np.int64))
        executed_action.append(np.array(action_index, dtype=np.int64))

        ag_viability_risk = np.array([viability_monitor["risk"]], dtype=np.float32)
        ag_input_view.append(
            build_trm_ag_input_view(
                target_logits.astype(np.float32),
                viability_monitor["state"].astype(np.float32),
                viability_monitor["homeostatic_error_vector"].astype(np.float32),
                ag_viability_risk,
                uncertainty_state.astype(np.float32),
                env_contact_state,
                species_contact_state,
            )
        )
        ag_inhibition_mask, ag_control_mode, ag_gated_logits = _family_conditioned_ag_targets(
            family or "energy_starved",
            as_target_logits=target_logits.astype(np.float32),
            viability_state=viability_monitor["state"].astype(np.float32),
            homeostatic_error_vector=viability_monitor["homeostatic_error_vector"].astype(np.float32),
            viability_risk=ag_viability_risk,
            uncertainty_state=uncertainty_state.astype(np.float32),
            env_contact_state=env_contact_state,
            species_contact_state=species_contact_state,
        )
        ag_target_gated_logits.append(ag_gated_logits.astype(np.float32))
        ag_target_inhibition_mask.append(ag_inhibition_mask.astype(np.float32))
        ag_target_control_mode.append(np.array(ag_control_mode, dtype=np.int64))

        runtime._apply_action(executed)
        runtime.env.update_fields(runtime.body, executed)
        prev_observation = observation.astype(np.float32)
        prev_action = executed
        prev_action_cost = action_cost
        if runtime._update_death():
            break

    mc_step_feature_array = np.stack(mc_step_features, axis=0).astype(np.float32)
    mc_input_view, mc_window_mask = build_trm_mc_input_view(mc_step_feature_array, mc_window_size)
    mc_target_context_state = _mc_target_context_state(mc_input_view, mc_window_mask)
    mc_target_action_bias = _family_conditioned_mc_action_bias(
        family or "energy_starved",
        np.stack(as_target_logits, axis=0).astype(np.float32),
        np.stack(vm_viability_state, axis=0).astype(np.float32),
        np.stack(vm_contact_state, axis=0).astype(np.float32),
        np.stack(vm_species_contact_state, axis=0).astype(np.float32),
        np.stack(as_uncertainty_state, axis=0).astype(np.float32),
    )
    mc_target_boundary_bias = _mc_target_boundary_bias(np.asarray(bp_target_mode, dtype=np.int64))

    return {
        "wp_observation": np.stack(wp_observation, axis=0).astype(np.float32),
        "wp_sensor_gate": np.stack(wp_sensor_gate, axis=0).astype(np.float32),
        "wp_species_fields": np.stack(wp_species_fields, axis=0).astype(np.float32),
        "wp_flow_channels": np.stack(wp_flow_channels, axis=0).astype(np.float32),
        "wp_input_view": np.stack(wp_input_view, axis=0).astype(np.float32),
        "wp_target_observation": np.stack(wp_target_observation, axis=0).astype(np.float32),
        "bd_observation": np.stack(bd_observation, axis=0).astype(np.float32),
        "bd_world_error": np.stack(bd_world_error, axis=0).astype(np.float32),
        "bd_sensor_gate": np.stack(bd_sensor_gate, axis=0).astype(np.float32),
        "bd_delta_observation": np.stack(bd_delta_observation, axis=0).astype(np.float32),
        "bd_input_view": np.stack(bd_input_view, axis=0).astype(np.float32),
        "bd_boundary_target": np.stack(bd_boundary_target, axis=0).astype(np.float32),
        "bd_permeability_target": np.stack(bd_permeability_target, axis=0).astype(np.float32),
        "bp_boundary_patch": np.stack(bp_boundary_patch, axis=0).astype(np.float32),
        "bp_permeability_patch": np.stack(bp_permeability_patch, axis=0).astype(np.float32),
        "bp_observation_patch": np.stack(bp_observation_patch, axis=0).astype(np.float32),
        "bp_species_patch": np.stack(bp_species_patch, axis=0).astype(np.float32),
        "bp_flow_patch": np.stack(bp_flow_patch, axis=0).astype(np.float32),
        "bp_viability_state": np.stack(bp_viability_state, axis=0).astype(np.float32),
        "bp_input_view": np.stack(bp_input_view, axis=0).astype(np.float32),
        "bp_target_permeability_patch": np.stack(bp_target_permeability_patch, axis=0).astype(np.float32),
        "bp_target_interface_gain": np.stack(bp_target_interface_gain, axis=0).astype(np.float32),
        "bp_target_aperture_gain": np.stack(bp_target_aperture_gain, axis=0).astype(np.float32),
        "bp_target_mode": np.asarray(bp_target_mode, dtype=np.int64),
        "vm_viability_state": np.stack(vm_viability_state, axis=0).astype(np.float32),
        "vm_contact_state": np.stack(vm_contact_state, axis=0).astype(np.float32),
        "vm_species_contact_state": np.stack(vm_species_contact_state, axis=0).astype(np.float32),
        "vm_action_cost": np.stack(vm_action_cost, axis=0).astype(np.float32),
        "vm_input_view": np.stack(vm_input_view, axis=0).astype(np.float32),
        "vm_target_state": np.stack(vm_target_state, axis=0).astype(np.float32),
        "vm_target_homeostatic_error": np.stack(vm_target_homeostatic_error, axis=0).astype(np.float32),
        "vm_target_risk": np.stack(vm_target_risk, axis=0).astype(np.float32),
        "as_viability_state": np.stack(as_viability_state, axis=0).astype(np.float32),
        "as_action_scores": np.stack(as_action_scores, axis=0).astype(np.float32),
        "as_uncertainty_state": np.stack(as_uncertainty_state, axis=0).astype(np.float32),
        "as_env_contact_state": np.stack(as_env_contact_state, axis=0).astype(np.float32),
        "as_species_contact_state": np.stack(as_species_contact_state, axis=0).astype(np.float32),
        "as_input_view": np.stack(as_input_view, axis=0).astype(np.float32),
        "as_target_logits": np.stack(as_target_logits, axis=0).astype(np.float32),
        "as_target_policy": np.stack(as_target_policy, axis=0).astype(np.float32),
        "as_target_action": np.asarray(as_target_action, dtype=np.int64),
        "executed_action": np.asarray(executed_action, dtype=np.int64),
        "ag_input_view": np.stack(ag_input_view, axis=0).astype(np.float32),
        "ag_target_gated_logits": np.stack(ag_target_gated_logits, axis=0).astype(np.float32),
        "ag_target_inhibition_mask": np.stack(ag_target_inhibition_mask, axis=0).astype(np.float32),
        "ag_target_control_mode": np.asarray(ag_target_control_mode, dtype=np.int64),
        "mc_input_view": mc_input_view.astype(np.float32),
        "mc_window_mask": mc_window_mask.astype(np.float32),
        "mc_target_context_state": mc_target_context_state.astype(np.float32),
        "mc_target_action_bias": mc_target_action_bias.astype(np.float32),
        "mc_target_boundary_bias": mc_target_boundary_bias.astype(np.float32),
    }


def _episode_quality_metrics(
    arrays: dict[str, np.ndarray],
    *,
    G_target: float,
    B_target: float,
    tau_G: float,
    tau_B: float,
) -> dict[str, Any]:
    target_action = arrays["as_target_action"].astype(np.int64)
    executed_action = arrays.get("executed_action", target_action).astype(np.int64)
    target_policy = arrays["as_target_policy"].astype(np.float32)
    contact_state = arrays["vm_contact_state"].astype(np.float32)
    viability_state = arrays["vm_viability_state"].astype(np.float32)
    next_viability_state = arrays["vm_target_state"].astype(np.float32)
    num_actions = len(ACTIONS)
    action_counts = np.bincount(target_action, minlength=num_actions).astype(np.int64)
    executed_action_counts = np.bincount(executed_action, minlength=num_actions + 1).astype(np.int64)
    num_samples = int(target_action.shape[0])
    dominant_fraction = float(action_counts.max() / max(num_samples, 1))
    distinct_actions = int(np.sum(action_counts > 0))
    dominant_action_index = int(action_counts.argmax()) if action_counts.size else 0
    dominant_action = ACTIONS[dominant_action_index]
    policy_entropy_mean = float(np.mean(_policy_entropy(target_policy)))
    viability_risk_rate = float(np.mean(arrays["vm_target_risk"].astype(np.float32)))
    current_error = np.abs(
        viability_state - np.array([float(G_target), float(B_target)], dtype=np.float32)
    ).sum(axis=-1)
    next_error = np.abs(
        next_viability_state - np.array([float(G_target), float(B_target)], dtype=np.float32)
    ).sum(axis=-1)
    recovery_fraction = float(np.mean(next_error <= current_error))
    thermal = contact_state[:, 1]
    toxicity = contact_state[:, 2]
    stress_mask = (thermal >= 0.35) | (toxicity >= 0.35)
    defensive_action_mask = np.isin(target_action, [ACTIONS.index("withdraw"), ACTIONS.index("seal"), ACTIONS.index("reconfigure")])
    exploit_action_mask = np.isin(target_action, [ACTIONS.index("approach"), ACTIONS.index("intake")])
    if np.any(stress_mask):
        stress_defensive_fraction = float(np.mean(defensive_action_mask[stress_mask]))
        stress_exploit_fraction = float(np.mean(exploit_action_mask[stress_mask]))
        stress_fraction = float(np.mean(stress_mask.astype(np.float32)))
    else:
        stress_defensive_fraction = 0.0
        stress_exploit_fraction = 0.0
        stress_fraction = 0.0
    terminal_state = next_viability_state[-1] if next_viability_state.size else np.array([0.0, 0.0], dtype=np.float32)
    terminal_dead = bool(float(terminal_state[0]) < float(tau_G) or float(terminal_state[1]) < float(tau_B))
    return {
        "num_samples": num_samples,
        "action_counts": {action: int(action_counts[i]) for i, action in enumerate(ACTIONS)},
        "executed_action_counts": {
            action: int(executed_action_counts[i])
            for i, action in enumerate((*ACTIONS, "no_action"))
        },
        "distinct_actions": distinct_actions,
        "dominant_action_fraction": dominant_fraction,
        "dominant_action": dominant_action,
        "policy_entropy_mean": policy_entropy_mean,
        "viability_risk_rate": viability_risk_rate,
        "recovery_fraction": recovery_fraction,
        "stress_fraction": stress_fraction,
        "stress_defensive_fraction": stress_defensive_fraction,
        "stress_exploit_fraction": stress_exploit_fraction,
        "terminal_dead": terminal_dead,
    }


def _episode_quality_failures(
    quality: dict[str, Any],
    *,
    min_episode_samples: int,
    min_distinct_actions: int,
    max_dominant_action_fraction: float,
    min_episode_policy_entropy: float,
) -> list[str]:
    failures: list[str] = []
    if int(quality["num_samples"]) < int(min_episode_samples):
        failures.append("too_few_samples")
    if int(quality["distinct_actions"]) < int(min_distinct_actions):
        failures.append("too_few_distinct_actions")
    if float(quality["dominant_action_fraction"]) > float(max_dominant_action_fraction):
        failures.append("dominant_action_too_high")
    if float(quality["policy_entropy_mean"]) < float(min_episode_policy_entropy):
        failures.append("policy_entropy_too_low")
    return failures


def _species_context_summary(env: LeniaERIEEnvironment) -> dict[str, Any]:
    sources = env.external_state.species_sources()
    fields = env.species_field_channels()
    return {
        "multispecies_enabled": True,
        "species_roles": ["species_energy", "species_toxic", "species_niche"],
        "source_mean": {
            "species_energy": float(np.mean(sources[..., 0])),
            "species_toxic": float(np.mean(sources[..., 1])),
            "species_niche": float(np.mean(sources[..., 2])),
        },
        "source_std": {
            "species_energy": float(np.std(sources[..., 0])),
            "species_toxic": float(np.std(sources[..., 1])),
            "species_niche": float(np.std(sources[..., 2])),
        },
        "field_mean": {
            "energy": float(np.mean(fields[..., 0])),
            "thermal": float(np.mean(fields[..., 1])),
            "toxicity": float(np.mean(fields[..., 2])),
            "niche": float(np.mean(fields[..., 3])),
        },
    }


def _bp_target_mode(action: str) -> int:
    if action in {"intake", "approach"}:
        return 0  # open-like / exploratory
    if action in {"seal", "withdraw"}:
        return 1  # close-like / defensive
    return 2  # reconfigure-like


def _write_role_view_manifests(
    output_root: Path,
    manifest_rows: list[dict[str, Any]],
) -> dict[str, str]:
    views_root = ensure_dir(output_root / "views")
    manifest_paths: dict[str, str] = {}
    summary: dict[str, Any] = {}
    for manifest_name, spec in ROLE_VIEW_SPECS.items():
        path = views_root / f"{manifest_name}.jsonl"
        rows: list[dict[str, Any]] = []
        for row in manifest_rows:
            rows.append(
                {
                    "episode_id": row["episode_id"],
                    "split": row["split"],
                    "path": row["path"],
                    "num_samples": int(row["num_samples"]),
                    "num_pairs": int(row["num_samples"]),
                    "seed_id": row["seed_id"],
                    "episode_family": row["episode_family"],
                    "policy_mode": row["policy_mode"],
                    "terminal_dead": row["terminal_dead"],
                    "quality": row["quality"],
                    "species_context": row["species_context"],
                    "runtime_config": row["runtime_config"],
                    "environment_config": row["environment_config"],
                    "view_name": manifest_name,
                    **spec,
                }
            )
        save_jsonl(path, rows)
        manifest_paths[manifest_name] = str(path)
        summary[manifest_name] = {
            "path": str(path),
            "num_rows": len(rows),
            **spec,
        }
    save_json(views_root / "summary.json", summary)
    return manifest_paths


def prepare_trm_va_cache(
    seed_catalog: str | Path,
    output_root: str | Path,
    runtime_config: RuntimeConfig,
    env_config: EnvironmentConfig,
    num_episodes: int = 16,
    target_band_weight: float = 0.0,
    target_g_overshoot_weight: float = 0.0,
    defensive_family_bias: float = 0.0,
    max_attempt_multiplier: int = 4,
    min_episode_samples: int = 8,
    min_distinct_actions: int = 2,
    max_dominant_action_fraction: float = 0.90,
    min_episode_policy_entropy: float = 0.90,
) -> Path:
    seed_everything(runtime_config.seed)
    output_root = ensure_dir(output_root)
    episode_dir = output_root / "episodes"
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    episode_dir = ensure_dir(episode_dir)
    seeds = load_seed_catalog(seed_catalog)
    if not seeds:
        raise SystemExit(f"no seeds found in {seed_catalog}")

    manifest_rows: list[dict[str, Any]] = []
    family_counts = {family: 0 for family in EPISODE_FAMILIES}
    family_pool = _family_pool(defensive_family_bias)
    aggregate_action_counts = {action: 0 for action in ACTIONS}
    aggregate_policy_entropy: list[float] = []
    aggregate_risk_rates: list[float] = []
    aggregate_recovery_fraction: list[float] = []
    aggregate_stress_defensive_fraction: list[float] = []
    aggregate_stress_exploit_fraction: list[float] = []
    rejection_reasons: dict[str, int] = {}
    rejected_episodes = 0
    max_attempts = max(int(num_episodes), int(num_episodes) * int(max_attempt_multiplier))
    retained_index = 0

    for attempt_index in range(max_attempts):
        if retained_index >= num_episodes:
            break
        episode_seed = int(runtime_config.seed + attempt_index)
        rng = np.random.default_rng(episode_seed)
        seed = seeds[int(rng.integers(0, len(seeds)))]
        family = family_pool[attempt_index % len(family_pool)]
        episode_runtime_config, episode_env_config = _sample_episode_configs(
            family=family,
            base_runtime_config=runtime_config,
            base_env_config=env_config,
            rng=rng,
            defensive_family_bias=defensive_family_bias,
        )
        env = LeniaERIEEnvironment(seed, episode_env_config, episode_runtime_config, rng)
        runtime = ERIERuntime(env, episode_runtime_config, rng, models=RuntimeModels(None, None))
        arrays = _collect_episode_samples_with_shaping(
            runtime,
            family=family,
            target_band_weight=target_band_weight,
            target_g_overshoot_weight=target_g_overshoot_weight,
        )
        quality = _episode_quality_metrics(
            arrays,
            G_target=episode_runtime_config.G_target,
            B_target=episode_runtime_config.B_target,
            tau_G=episode_runtime_config.tau_G,
            tau_B=episode_runtime_config.tau_B,
        )
        species_context = _species_context_summary(env)
        failures = _episode_quality_failures(
            quality,
            min_episode_samples=min_episode_samples,
            min_distinct_actions=min_distinct_actions,
            max_dominant_action_fraction=max_dominant_action_fraction,
            min_episode_policy_entropy=min_episode_policy_entropy,
        )
        if failures:
            rejected_episodes += 1
            for reason in failures:
                rejection_reasons[reason] = int(rejection_reasons.get(reason, 0)) + 1
            continue
        episode_id = f"va_{episode_seed}_{family}_{seed.seed_id}"
        path = episode_dir / f"{episode_id}.npz"
        np.savez_compressed(path, **arrays)
        family_counts[family] += 1
        retained_index += 1
        for action, count in quality["action_counts"].items():
            aggregate_action_counts[action] = int(aggregate_action_counts[action] + int(count))
        aggregate_policy_entropy.append(float(quality["policy_entropy_mean"]))
        aggregate_risk_rates.append(float(quality["viability_risk_rate"]))
        aggregate_recovery_fraction.append(float(quality["recovery_fraction"]))
        aggregate_stress_defensive_fraction.append(float(quality["stress_defensive_fraction"]))
        aggregate_stress_exploit_fraction.append(float(quality["stress_exploit_fraction"]))
        manifest_rows.append(
            {
                "episode_id": episode_id,
                "split": choose_split(retained_index - 1, num_episodes),
                "path": str(path),
                "num_samples": int(arrays["vm_viability_state"].shape[0]),
                "seed_id": seed.seed_id,
                "episode_family": family,
                "policy_mode": episode_runtime_config.policy_mode,
                "terminal_dead": bool(quality["terminal_dead"]),
                "quality": quality,
                "species_context": species_context,
                "runtime_config": asdict(episode_runtime_config),
                "environment_config": asdict(episode_env_config),
                "target_band_weight": float(target_band_weight),
                "target_g_overshoot_weight": float(target_g_overshoot_weight),
                "defensive_family_bias": float(defensive_family_bias),
            }
        )

    if retained_index < num_episodes:
        raise SystemExit(
            f"could not retain enough TRM-VA episodes: retained={retained_index}, "
            f"required={num_episodes}, max_attempts={max_attempts}"
        )

    manifest_path = output_root / "manifest.jsonl"
    save_jsonl(manifest_path, manifest_rows)
    role_view_manifests = _write_role_view_manifests(output_root, manifest_rows)
    save_json(
        output_root / "summary.json",
        {
            "seed_catalog": str(seed_catalog),
            "num_episodes": int(num_episodes),
            "runtime_config": asdict(runtime_config),
            "environment_config": asdict(env_config),
            "episode_families": list(EPISODE_FAMILIES),
            "family_counts": family_counts,
            "family_pool": list(family_pool),
            "policy_mode": runtime_config.policy_mode,
            "target_band_weight": float(target_band_weight),
            "target_g_overshoot_weight": float(target_g_overshoot_weight),
            "defensive_family_bias": float(defensive_family_bias),
            "attempted_episodes": int(retained_index + rejected_episodes),
            "retained_episodes": int(sum(family_counts.values())),
            "rejected_episodes": int(rejected_episodes),
            "rejection_reasons": rejection_reasons,
            "quality_thresholds": {
                "max_attempt_multiplier": int(max_attempt_multiplier),
                "min_episode_samples": int(min_episode_samples),
                "min_distinct_actions": int(min_distinct_actions),
                "max_dominant_action_fraction": float(max_dominant_action_fraction),
                "min_episode_policy_entropy": float(min_episode_policy_entropy),
            },
            "aggregate_action_counts": aggregate_action_counts,
            "aggregate_policy_entropy_mean": float(np.mean(aggregate_policy_entropy)) if aggregate_policy_entropy else float("nan"),
            "aggregate_viability_risk_rate_mean": float(np.mean(aggregate_risk_rates)) if aggregate_risk_rates else float("nan"),
            "aggregate_recovery_fraction_mean": float(np.mean(aggregate_recovery_fraction)) if aggregate_recovery_fraction else float("nan"),
            "aggregate_stress_defensive_fraction_mean": float(np.mean(aggregate_stress_defensive_fraction)) if aggregate_stress_defensive_fraction else float("nan"),
            "aggregate_stress_exploit_fraction_mean": float(np.mean(aggregate_stress_exploit_fraction)) if aggregate_stress_exploit_fraction else float("nan"),
            "multispecies_enabled": True,
            "species_roles": ["species_energy", "species_toxic", "species_niche"],
            "role_view_manifests": role_view_manifests,
            "role_view_summary_path": str(output_root / "views" / "summary.json"),
        },
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bootstrap training cache for TRM-Vm and TRM-As.")
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--output-root", default="data/trm_va_cache")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--target-band-weight", type=float, default=0.0)
    parser.add_argument("--target-g-overshoot-weight", type=float, default=0.0)
    parser.add_argument("--defensive-family-bias", type=float, default=0.0)
    parser.add_argument("--max-attempt-multiplier", type=int, default=4)
    parser.add_argument("--min-episode-samples", type=int, default=8)
    parser.add_argument("--min-distinct-actions", type=int, default=2)
    parser.add_argument("--max-dominant-action-fraction", type=float, default=0.90)
    parser.add_argument("--min-episode-policy-entropy", type=float, default=0.90)
    args = parser.parse_args()
    manifest = prepare_trm_va_cache(
        seed_catalog=args.seed_catalog,
        output_root=args.output_root,
        runtime_config=RuntimeConfig(steps=args.steps, warmup_steps=args.warmup_steps, seed=args.seed),
        env_config=EnvironmentConfig(),
        num_episodes=args.episodes,
        target_band_weight=args.target_band_weight,
        target_g_overshoot_weight=args.target_g_overshoot_weight,
        defensive_family_bias=args.defensive_family_bias,
        max_attempt_multiplier=args.max_attempt_multiplier,
        min_episode_samples=args.min_episode_samples,
        min_distinct_actions=args.min_distinct_actions,
        max_dominant_action_fraction=args.max_dominant_action_fraction,
        min_episode_policy_entropy=args.min_episode_policy_entropy,
    )
    print(f"wrote TRM-VA cache manifest: {manifest}")


if __name__ == "__main__":
    main()
