from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _sigmoid_scalar(x: float) -> float:
    clipped = float(np.clip(x, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def alive_bodies(bodies: list[Any]) -> list[Any]:
    return [body for body in bodies if bool(getattr(body, "alive", False))]


def select_primary_body(bodies: list[Any]) -> Any | None:
    alive = alive_bodies(bodies)
    if not alive:
        return None
    return max(alive, key=lambda body: float(getattr(body, "G", 0.0) + getattr(body, "B", 0.0)))


def can_expand_population(bodies: list[Any], max_bodies: int) -> bool:
    return len(alive_bodies(bodies)) < max(1, int(max_bodies))


def is_action_locked(history: list[dict[str, Any]], current_action: str | None, window: int) -> bool:
    window = int(window)
    if window <= 1:
        return False
    if current_action is None:
        return False
    history_window = max(0, window - 1)
    previous_actions = [str(row.get("action", "")) for row in history[-history_window:]]
    actions = previous_actions + [str(current_action)]
    if len(actions) < window:
        return False
    return len(set(actions)) == 1


def classify_death_cause(
    *,
    threshold_violation: bool,
    nonfinite_state: bool,
    invalid_body_state: bool,
    action_lock: bool,
    policy_forbidden_window: bool,
    expected_label: str,
    degenerate_label: str,
    policy_forbidden_label: str,
) -> str:
    if nonfinite_state or invalid_body_state or action_lock:
        return degenerate_label
    if policy_forbidden_window:
        return policy_forbidden_label
    if threshold_violation:
        return expected_label
    return degenerate_label


def update_death_state(
    *,
    current_dead_count: int,
    G: float,
    B: float,
    tau_G: float,
    tau_B: float,
    k_irrev: int,
    history: list[dict[str, Any]],
    action: str | None,
    action_lock_window: int,
    t: int | None,
    policy_forbidden_min_survival_steps: int,
    invalid_body_state: bool,
    expected_label: str,
    degenerate_label: str,
    policy_forbidden_label: str,
) -> dict[str, Any]:
    threshold_violation = bool(G < tau_G or B < tau_B)
    nonfinite_state = bool(not np.isfinite(G) or not np.isfinite(B))
    if threshold_violation or nonfinite_state or bool(invalid_body_state):
        dead_count = int(current_dead_count) + 1
    else:
        dead_count = 0
    dead = dead_count >= int(k_irrev)
    policy_forbidden_window = bool(
        dead
        and int(policy_forbidden_min_survival_steps) > 0
        and t is not None
        and (int(t) + 1) < int(policy_forbidden_min_survival_steps)
    )
    action_lock = bool(
        dead
        and is_action_locked(
            history,
            action,
            int(action_lock_window),
        )
    )
    death_signals = {
        "threshold_violation": threshold_violation,
        "nonfinite_state": nonfinite_state,
        "action_lock": action_lock,
        "policy_forbidden_window": policy_forbidden_window,
        "invalid_body_state": bool(invalid_body_state),
    }
    death_cause = None
    if dead:
        death_cause = classify_death_cause(
            threshold_violation=threshold_violation,
            nonfinite_state=nonfinite_state,
            invalid_body_state=bool(invalid_body_state),
            action_lock=action_lock,
            policy_forbidden_window=policy_forbidden_window,
            expected_label=expected_label,
            degenerate_label=degenerate_label,
            policy_forbidden_label=policy_forbidden_label,
        )
    return {
        "dead_count": int(dead_count),
        "dead": bool(dead),
        "death_signals": death_signals,
        "death_cause": death_cause,
    }


def spawn_drive(
    *,
    trace_term: float,
    resource: float,
    hazard: float,
    G: float,
    B: float,
    tau_G: float,
    tau_B: float,
    spawn_logit_bias: float,
    spawn_trace_gain: float,
    spawn_resource_gain: float,
    spawn_hazard_penalty: float,
    spawn_viability_gain: float,
) -> float:
    viability_margin = max(0.0, float(G - tau_G)) + max(0.0, float(B - tau_B))
    logit = (
        float(spawn_logit_bias)
        + float(spawn_trace_gain) * float(trace_term)
        + float(spawn_resource_gain) * float(resource)
        - float(spawn_hazard_penalty) * float(hazard)
        + float(spawn_viability_gain) * viability_margin
    )
    return float(np.clip(_sigmoid_scalar(logit), 0.0, 1.0))


def split_drive(
    *,
    mass: float,
    energy: float,
    boundary_integrity: float,
    split_logit_bias: float,
    split_mass_gain: float,
    split_energy_gain: float,
    split_boundary_penalty: float,
) -> float:
    boundary_damage = max(0.0, 1.0 - float(boundary_integrity))
    logit = (
        float(split_logit_bias)
        + float(split_mass_gain) * float(mass)
        + float(split_energy_gain) * float(energy)
        - float(split_boundary_penalty) * boundary_damage
    )
    return float(np.clip(_sigmoid_scalar(logit), 0.0, 1.0))


def spawn_split_signals(
    *,
    trace_density: float,
    resource: float,
    hazard: float,
    G: float,
    B: float,
    tau_G: float,
    tau_B: float,
    mass: float,
    energy: float,
    boundary_integrity: float,
    spawn_logit_bias: float,
    spawn_trace_gain: float,
    spawn_resource_gain: float,
    spawn_hazard_penalty: float,
    spawn_viability_gain: float,
    split_logit_bias: float,
    split_mass_gain: float,
    split_energy_gain: float,
    split_boundary_penalty: float,
    spawn_candidate_threshold: float,
    split_candidate_threshold: float,
) -> dict[str, float | bool]:
    spawn_with_trace = spawn_drive(
        trace_term=float(trace_density),
        resource=float(resource),
        hazard=float(hazard),
        G=float(G),
        B=float(B),
        tau_G=float(tau_G),
        tau_B=float(tau_B),
        spawn_logit_bias=float(spawn_logit_bias),
        spawn_trace_gain=float(spawn_trace_gain),
        spawn_resource_gain=float(spawn_resource_gain),
        spawn_hazard_penalty=float(spawn_hazard_penalty),
        spawn_viability_gain=float(spawn_viability_gain),
    )
    spawn_without_trace = spawn_drive(
        trace_term=0.0,
        resource=float(resource),
        hazard=float(hazard),
        G=float(G),
        B=float(B),
        tau_G=float(tau_G),
        tau_B=float(tau_B),
        spawn_logit_bias=float(spawn_logit_bias),
        spawn_trace_gain=float(spawn_trace_gain),
        spawn_resource_gain=float(spawn_resource_gain),
        spawn_hazard_penalty=float(spawn_hazard_penalty),
        spawn_viability_gain=float(spawn_viability_gain),
    )
    split = split_drive(
        mass=float(mass),
        energy=float(energy),
        boundary_integrity=float(boundary_integrity),
        split_logit_bias=float(split_logit_bias),
        split_mass_gain=float(split_mass_gain),
        split_energy_gain=float(split_energy_gain),
        split_boundary_penalty=float(split_boundary_penalty),
    )
    return {
        "trace_density": float(trace_density),
        "spawn_drive": float(spawn_with_trace),
        "spawn_drive_no_trace": float(spawn_without_trace),
        "trace_ablation_spawn_delta": float(spawn_with_trace - spawn_without_trace),
        "split_drive": float(split),
        "spawn_candidate": bool(spawn_with_trace >= float(spawn_candidate_threshold)),
        "split_candidate": bool(split >= float(split_candidate_threshold)),
    }


def spawn_child_from_primary(
    parent: Any,
    *,
    can_expand: bool,
    tau_G: float,
    tau_B: float,
    spawn_energy_share: float,
    spawn_offset: float,
    spawn_radius_scale: float,
    image_size: int,
    copy_body: Callable[[Any], Any],
    next_body_id: Callable[[], int],
) -> Any | None:
    if not bool(can_expand):
        return None
    if float(getattr(parent, "G", 0.0)) <= float(tau_G) + 0.05 or float(getattr(parent, "B", 0.0)) <= float(tau_B) + 0.05:
        return None
    share = float(np.clip(float(spawn_energy_share), 0.05, 0.8))
    child = copy_body(parent)
    direction_y = float(np.sin(float(getattr(parent, "aperture_angle", 0.0))))
    direction_x = float(np.cos(float(getattr(parent, "aperture_angle", 0.0))))
    child.centroid_y = float(
        np.clip(
            float(getattr(parent, "centroid_y", 0.0)) + float(spawn_offset) * direction_y,
            4.0,
            float(image_size) - 5.0,
        )
    )
    child.centroid_x = float(
        np.clip(
            float(getattr(parent, "centroid_x", 0.0)) + float(spawn_offset) * direction_x,
            4.0,
            float(image_size) - 5.0,
        )
    )
    child.radius = float(np.clip(float(getattr(parent, "radius", 0.0)) * float(spawn_radius_scale), 4.0, 10.0))
    child.G = float(np.clip(float(getattr(parent, "G", 0.0)) * share, 0.0, 1.0))
    child.B = float(np.clip(float(getattr(parent, "B", 0.0)) * (0.92 + 0.06 * share), 0.0, 1.0))
    child.dead_count = 0
    child.alive = True
    child.body_id = int(next_body_id())
    child.parent_id = int(getattr(parent, "body_id", -1))
    child.generation = int(getattr(parent, "generation", 0) + 1)
    parent.G = float(np.clip(float(getattr(parent, "G", 0.0)) * (1.0 - share), 0.0, 1.0))
    parent.B = float(np.clip(float(getattr(parent, "B", 0.0)) * (0.96 - 0.08 * share), 0.0, 1.0))
    return child


def split_child_from_primary(
    parent: Any,
    *,
    can_expand: bool,
    tau_G: float,
    split_energy_share: float,
    split_radius_scale: float,
    spawn_offset: float,
    image_size: int,
    copy_body: Callable[[Any], Any],
    next_body_id: Callable[[], int],
) -> Any | None:
    if not bool(can_expand):
        return None
    if float(getattr(parent, "mass", 0.0)) <= 0.5 or float(getattr(parent, "energy", 0.0)) <= float(tau_G) + 0.04:
        return None
    share = float(np.clip(float(split_energy_share), 0.15, 0.85))
    child = copy_body(parent)
    direction_y = float(np.sin(float(getattr(parent, "aperture_angle", 0.0)) + np.pi / 2.0))
    direction_x = float(np.cos(float(getattr(parent, "aperture_angle", 0.0)) + np.pi / 2.0))
    offset = 0.65 * float(spawn_offset)
    parent.centroid_y = float(
        np.clip(float(getattr(parent, "centroid_y", 0.0)) - offset * direction_y, 4.0, float(image_size) - 5.0)
    )
    parent.centroid_x = float(
        np.clip(float(getattr(parent, "centroid_x", 0.0)) - offset * direction_x, 4.0, float(image_size) - 5.0)
    )
    child.centroid_y = float(
        np.clip(float(getattr(child, "centroid_y", 0.0)) + offset * direction_y, 4.0, float(image_size) - 5.0)
    )
    child.centroid_x = float(
        np.clip(float(getattr(child, "centroid_x", 0.0)) + offset * direction_x, 4.0, float(image_size) - 5.0)
    )
    parent.radius = float(np.clip(float(getattr(parent, "radius", 0.0)) * float(split_radius_scale), 3.5, 10.0))
    child.radius = float(np.clip(float(getattr(child, "radius", 0.0)) * float(split_radius_scale), 3.5, 10.0))
    parent.G = float(np.clip(float(getattr(parent, "G", 0.0)) * (1.0 - share), 0.0, 1.0))
    child.G = float(np.clip(float(getattr(child, "G", 0.0)) * share, 0.0, 1.0))
    parent.B = float(np.clip(float(getattr(parent, "B", 0.0)) * 0.97, 0.0, 1.0))
    child.B = float(np.clip(float(getattr(child, "B", 0.0)) * 0.97, 0.0, 1.0))
    child.dead_count = 0
    child.alive = True
    child.body_id = int(next_body_id())
    child.parent_id = int(getattr(parent, "body_id", -1))
    child.generation = int(getattr(parent, "generation", 0) + 1)
    return child
