from __future__ import annotations

import math
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class RuntimeMetricsConfig(Protocol):
    G_target: float
    B_target: float
    G0: float
    B0: float
    steps: int


def homeostatic_error(G: float, B: float, cfg: RuntimeMetricsConfig) -> float:
    return float(abs(G - cfg.G_target) + abs(B - cfg.B_target))


def death_cause_counts(
    history: list[dict[str, Any]],
    *,
    expected_label: str,
    degenerate_label: str,
    policy_forbidden_label: str,
) -> dict[str, int]:
    counts = {
        expected_label: 0,
        degenerate_label: 0,
        policy_forbidden_label: 0,
    }
    for row in history:
        if not bool(row.get("dead", False)):
            continue
        cause = str(row.get("death_cause") or "")
        if cause in counts:
            counts[cause] += 1
    return counts


def _entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(probs.astype(np.float32), eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _row_float(row: dict[str, Any], key: str, *aliases: str) -> float:
    for candidate in (key, *aliases):
        if candidate in row:
            return float(row[candidate])
    return 0.0


def episode_metrics(
    history: list[dict[str, Any]],
    cfg: RuntimeMetricsConfig,
    *,
    actions: Sequence[str],
    policy_action_cost: Callable[[str | None], float],
    expected_death_label: str,
    degenerate_death_label: str,
    policy_forbidden_death_label: str,
) -> dict[str, float]:
    death_counts = death_cause_counts(
        history,
        expected_label=expected_death_label,
        degenerate_label=degenerate_death_label,
        policy_forbidden_label=policy_forbidden_death_label,
    )
    if not history:
        return {
            "mean_G": 0.0,
            "mean_B": 0.0,
            "survival_fraction": 0.0,
            "final_homeostatic_error": homeostatic_error(cfg.G0, cfg.B0, cfg),
            "mean_homeostatic_error": homeostatic_error(cfg.G0, cfg.B0, cfg),
            "action_cost_total": 0.0,
            "action_cost_mean": 0.0,
            "mean_policy_entropy": 0.0,
            "mean_contact_energy": 0.0,
            "mean_contact_thermal": 0.0,
            "mean_contact_toxicity": 0.0,
            "mean_contact_niche": 0.0,
            "mean_contact_species_energy": 0.0,
            "mean_contact_species_thermal": 0.0,
            "mean_contact_species_toxicity": 0.0,
            "mean_contact_species_niche": 0.0,
            "mean_contact_resource": 0.0,
            "mean_contact_hazard": 0.0,
            "mean_contact_shelter": 0.0,
            "action_diversity": 0.0,
            "invalid_body_state_count": 0.0,
            "boundary_interface_usage_rate": 0.0,
            "mean_trace_mass": 0.0,
            "final_trace_mass": 0.0,
            "mean_trace_density": 0.0,
            "mean_spawn_drive": 0.0,
            "mean_spawn_drive_no_trace": 0.0,
            "mean_trace_ablation_spawn_delta": 0.0,
            "spawn_candidate_rate": 0.0,
            "mean_split_drive": 0.0,
            "split_candidate_rate": 0.0,
            "mean_body_count": 1.0,
            "max_body_count": 1.0,
            "spawn_events_total": 0.0,
            "split_events_total": 0.0,
            "death_events_total": 0.0,
            "mean_p_t": 0.5,
            "mean_challenge_fraction": 0.5,
            "role_switch_events_total": 0.0,
            "mean_aux_updated_body_count": 0.0,
            "mean_aux_policy_entropy": 0.0,
            "aux_full_policy_rate": 0.0,
            "aux_role_heuristic_rate": 0.0,
            "aux_passive_rate": 0.0,
            "mean_aux_nontrivial_action_count": 0.0,
            "mean_aux_challenge_action_count": 0.0,
            "mean_aux_conservative_action_count": 0.0,
            "death_events": 0.0,
            "expected_death_events": 0.0,
            "degenerate_death_events": 0.0,
            "policy_forbidden_death_events": 0.0,
        }

    g_values = np.array([float(row["G"]) for row in history], dtype=np.float32)
    b_values = np.array([float(row["B"]) for row in history], dtype=np.float32)
    action_costs = np.array([policy_action_cost(str(row["action"])) for row in history], dtype=np.float32)
    errors = np.abs(g_values - float(cfg.G_target)) + np.abs(b_values - float(cfg.B_target))
    policy_entropy = np.array([float(row["policy_entropy"]) for row in history], dtype=np.float32)
    contact_energy = np.array([_row_float(row, "contact_energy", "contact_resource") for row in history], dtype=np.float32)
    contact_thermal = np.array([_row_float(row, "contact_thermal") for row in history], dtype=np.float32)
    contact_toxicity = np.array([_row_float(row, "contact_toxicity") for row in history], dtype=np.float32)
    contact_niche = np.array([_row_float(row, "contact_niche", "contact_shelter") for row in history], dtype=np.float32)
    contact_species_energy = np.array([float(row["contact_species_energy"]) for row in history], dtype=np.float32)
    contact_species_thermal = np.array([float(row["contact_species_thermal"]) for row in history], dtype=np.float32)
    contact_species_toxicity = np.array([float(row["contact_species_toxicity"]) for row in history], dtype=np.float32)
    contact_species_niche = np.array([float(row["contact_species_niche"]) for row in history], dtype=np.float32)
    contact_resource = np.array([_row_float(row, "contact_resource", "contact_energy") for row in history], dtype=np.float32)
    contact_hazard = np.array(
        [
            _row_float(row, "contact_hazard")
            if "contact_hazard" in row
            else 0.6 * _row_float(row, "contact_thermal") + 0.4 * _row_float(row, "contact_toxicity")
            for row in history
        ],
        dtype=np.float32,
    )
    contact_shelter = np.array([_row_float(row, "contact_shelter", "contact_niche") for row in history], dtype=np.float32)
    action_vocab = tuple(actions) + ("no_action",)
    action_labels = [str(row["action"]) for row in history]
    counts = np.array([action_labels.count(action) for action in action_vocab], dtype=np.float32)
    probs = counts / max(float(counts.sum()), 1.0)
    action_diversity = _entropy(probs) / math.log(len(probs))
    invalid_body_state_count = float(sum(1 for row in history if bool(row.get("invalid_body_state", False))))
    boundary_observe_count = float(sum(1 for row in history if bool(row.get("boundary_interface_observe", False))))
    boundary_action_count = float(sum(1 for row in history if bool(row.get("boundary_interface_action", False))))
    boundary_interface_usage_rate = float(
        (boundary_observe_count + boundary_action_count) / max(2.0 * float(len(history)), 1.0)
    )
    trace_mass = np.array([float(row.get("trace_mass", 0.0)) for row in history], dtype=np.float32)
    trace_density = np.array([float(row.get("trace_density", 0.0)) for row in history], dtype=np.float32)
    spawn_drive = np.array([float(row.get("spawn_drive", 0.0)) for row in history], dtype=np.float32)
    spawn_drive_no_trace = np.array([float(row.get("spawn_drive_no_trace", 0.0)) for row in history], dtype=np.float32)
    trace_ablation_spawn_delta = np.array(
        [float(row.get("trace_ablation_spawn_delta", 0.0)) for row in history], dtype=np.float32
    )
    split_drive = np.array([float(row.get("split_drive", 0.0)) for row in history], dtype=np.float32)
    body_count = np.array([float(row.get("body_count", 1.0)) for row in history], dtype=np.float32)
    spawn_events = np.array([float(row.get("spawn_events", 0.0)) for row in history], dtype=np.float32)
    split_events = np.array([float(row.get("split_events", 0.0)) for row in history], dtype=np.float32)
    death_events_step = np.array([float(row.get("death_events_step", 0.0)) for row in history], dtype=np.float32)
    p_t = np.array([float(row.get("p_t", 0.5)) for row in history], dtype=np.float32)
    challenge_fraction = np.array(
        [
            float(row.get("challenge_body_count", 0.0)) / max(float(row.get("body_count", 1.0)), 1.0)
            for row in history
        ],
        dtype=np.float32,
    )
    role_switch_events_step = np.array([float(row.get("role_switch_events_step", 0.0)) for row in history], dtype=np.float32)
    aux_updated_body_count = np.array([float(row.get("aux_updated_body_count", 0.0)) for row in history], dtype=np.float32)
    aux_mean_policy_entropy = np.array([float(row.get("aux_mean_policy_entropy", 0.0)) for row in history], dtype=np.float32)
    aux_nontrivial_action_count = np.array(
        [float(row.get("aux_nontrivial_action_count", 0.0)) for row in history], dtype=np.float32
    )
    aux_challenge_action_count = np.array(
        [float(row.get("aux_challenge_action_count", 0.0)) for row in history], dtype=np.float32
    )
    aux_conservative_action_count = np.array(
        [float(row.get("aux_conservative_action_count", 0.0)) for row in history], dtype=np.float32
    )
    aux_full_policy_count = float(
        sum(float(row.get("aux_policy_source_counts", {}).get("full_policy", 0.0)) for row in history)
    )
    aux_role_heuristic_count = float(
        sum(float(row.get("aux_policy_source_counts", {}).get("role_heuristic", 0.0)) for row in history)
    )
    aux_passive_count = float(
        sum(float(row.get("aux_policy_source_counts", {}).get("passive", 0.0)) for row in history)
    )
    aux_policy_count_total = max(aux_full_policy_count + aux_role_heuristic_count + aux_passive_count, 1.0)
    spawn_candidate_rate = float(
        sum(1 for row in history if bool(row.get("spawn_candidate", False))) / max(len(history), 1)
    )
    split_candidate_rate = float(
        sum(1 for row in history if bool(row.get("split_candidate", False))) / max(len(history), 1)
    )
    return {
        "mean_G": float(g_values.mean()),
        "mean_B": float(b_values.mean()),
        "survival_fraction": float(len(history) / max(cfg.steps, 1)),
        "final_homeostatic_error": float(errors[-1]),
        "mean_homeostatic_error": float(errors.mean()),
        "action_cost_total": float(action_costs.sum()),
        "action_cost_mean": float(action_costs.mean()),
        "mean_policy_entropy": float(policy_entropy.mean()),
        "mean_contact_energy": float(contact_energy.mean()),
        "mean_contact_thermal": float(contact_thermal.mean()),
        "mean_contact_toxicity": float(contact_toxicity.mean()),
        "mean_contact_niche": float(contact_niche.mean()),
        "mean_contact_species_energy": float(contact_species_energy.mean()),
        "mean_contact_species_thermal": float(contact_species_thermal.mean()),
        "mean_contact_species_toxicity": float(contact_species_toxicity.mean()),
        "mean_contact_species_niche": float(contact_species_niche.mean()),
        "mean_contact_resource": float(contact_resource.mean()),
        "mean_contact_hazard": float(contact_hazard.mean()),
        "mean_contact_shelter": float(contact_shelter.mean()),
        "action_diversity": float(action_diversity),
        "invalid_body_state_count": invalid_body_state_count,
        "boundary_interface_usage_rate": boundary_interface_usage_rate,
        "mean_trace_mass": float(trace_mass.mean()),
        "final_trace_mass": float(trace_mass[-1]),
        "mean_trace_density": float(trace_density.mean()),
        "mean_spawn_drive": float(spawn_drive.mean()),
        "mean_spawn_drive_no_trace": float(spawn_drive_no_trace.mean()),
        "mean_trace_ablation_spawn_delta": float(trace_ablation_spawn_delta.mean()),
        "spawn_candidate_rate": spawn_candidate_rate,
        "mean_split_drive": float(split_drive.mean()),
        "split_candidate_rate": split_candidate_rate,
        "mean_body_count": float(body_count.mean()),
        "max_body_count": float(body_count.max()),
        "spawn_events_total": float(spawn_events.sum()),
        "split_events_total": float(split_events.sum()),
        "death_events_total": float(death_events_step.sum()),
        "mean_p_t": float(p_t.mean()),
        "mean_challenge_fraction": float(challenge_fraction.mean()),
        "role_switch_events_total": float(role_switch_events_step.sum()),
        "mean_aux_updated_body_count": float(aux_updated_body_count.mean()),
        "mean_aux_policy_entropy": float(aux_mean_policy_entropy.mean()),
        "aux_full_policy_rate": float(aux_full_policy_count / aux_policy_count_total),
        "aux_role_heuristic_rate": float(aux_role_heuristic_count / aux_policy_count_total),
        "aux_passive_rate": float(aux_passive_count / aux_policy_count_total),
        "mean_aux_nontrivial_action_count": float(aux_nontrivial_action_count.mean()),
        "mean_aux_challenge_action_count": float(aux_challenge_action_count.mean()),
        "mean_aux_conservative_action_count": float(aux_conservative_action_count.mean()),
        "death_events": float(sum(death_counts.values())),
        "expected_death_events": float(death_counts[expected_death_label]),
        "degenerate_death_events": float(death_counts[degenerate_death_label]),
        "policy_forbidden_death_events": float(death_counts[policy_forbidden_death_label]),
    }
