from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from trm_pipeline.runtime_population import (
    alive_bodies,
    can_expand_population,
    classify_death_cause,
    is_action_locked,
    select_primary_body,
    spawn_child_from_primary,
    spawn_drive,
    spawn_split_signals,
    split_child_from_primary,
    split_drive,
    update_death_state,
)


@dataclass
class _Body:
    alive: bool
    G: float
    B: float


@dataclass
class _SpawnBody:
    centroid_y: float
    centroid_x: float
    radius: float
    aperture_angle: float
    G: float
    B: float
    dead_count: int
    alive: bool
    body_id: int
    parent_id: int
    generation: int
    mass: float = 1.0
    energy: float = 1.0


def test_alive_bodies_filters_non_alive() -> None:
    bodies = [_Body(alive=True, G=0.1, B=0.2), _Body(alive=False, G=0.9, B=0.9)]
    result = alive_bodies(bodies)
    assert len(result) == 1
    assert result[0].alive is True


def test_select_primary_body_uses_highest_g_plus_b() -> None:
    a = _Body(alive=True, G=0.2, B=0.2)
    b = _Body(alive=True, G=0.5, B=0.5)
    c = _Body(alive=False, G=1.0, B=1.0)
    assert select_primary_body([a, b, c]) is b


def test_can_expand_population_counts_only_alive() -> None:
    bodies = [_Body(alive=True, G=0.1, B=0.1), _Body(alive=False, G=0.1, B=0.1)]
    assert can_expand_population(bodies, max_bodies=2) is True
    assert can_expand_population(bodies, max_bodies=1) is False


def test_is_action_locked_requires_repeated_actions_in_window() -> None:
    history = [{"action": "intake"}, {"action": "intake"}]
    assert is_action_locked(history, "intake", window=3) is True
    assert is_action_locked(history, "withdraw", window=3) is False


def test_classify_death_cause_priority() -> None:
    cause = classify_death_cause(
        threshold_violation=True,
        nonfinite_state=True,
        invalid_body_state=False,
        action_lock=False,
        policy_forbidden_window=True,
        expected_label="expected",
        degenerate_label="degenerate",
        policy_forbidden_label="policy_forbidden",
    )
    assert cause == "degenerate"


def test_update_death_state_reports_policy_forbidden() -> None:
    result = update_death_state(
        current_dead_count=1,
        G=0.10,
        B=0.10,
        tau_G=0.20,
        tau_B=0.20,
        k_irrev=2,
        history=[],
        action="intake",
        action_lock_window=0,
        t=0,
        policy_forbidden_min_survival_steps=8,
        invalid_body_state=False,
        expected_label="expected",
        degenerate_label="degenerate",
        policy_forbidden_label="policy_forbidden",
    )

    assert result["dead"] is True
    assert result["death_signals"]["policy_forbidden_window"] is True
    assert result["death_cause"] == "policy_forbidden"


def test_spawn_drive_increases_with_trace_density() -> None:
    low = spawn_drive(
        trace_term=0.0,
        resource=0.5,
        hazard=0.2,
        G=0.7,
        B=0.7,
        tau_G=0.2,
        tau_B=0.2,
        spawn_logit_bias=-2.0,
        spawn_trace_gain=2.5,
        spawn_resource_gain=1.0,
        spawn_hazard_penalty=1.0,
        spawn_viability_gain=1.0,
    )
    high = spawn_drive(
        trace_term=1.0,
        resource=0.5,
        hazard=0.2,
        G=0.7,
        B=0.7,
        tau_G=0.2,
        tau_B=0.2,
        spawn_logit_bias=-2.0,
        spawn_trace_gain=2.5,
        spawn_resource_gain=1.0,
        spawn_hazard_penalty=1.0,
        spawn_viability_gain=1.0,
    )
    assert high > low


def test_split_drive_decreases_when_boundary_integrity_drops() -> None:
    safe = split_drive(
        mass=1.0,
        energy=0.8,
        boundary_integrity=0.9,
        split_logit_bias=-0.2,
        split_mass_gain=1.0,
        split_energy_gain=1.0,
        split_boundary_penalty=2.0,
    )
    damaged = split_drive(
        mass=1.0,
        energy=0.8,
        boundary_integrity=0.2,
        split_logit_bias=-0.2,
        split_mass_gain=1.0,
        split_energy_gain=1.0,
        split_boundary_penalty=2.0,
    )
    assert safe > damaged


def test_spawn_split_signals_contains_expected_keys() -> None:
    signals = spawn_split_signals(
        trace_density=0.4,
        resource=0.6,
        hazard=0.2,
        G=0.7,
        B=0.8,
        tau_G=0.2,
        tau_B=0.2,
        mass=1.1,
        energy=0.9,
        boundary_integrity=0.8,
        spawn_logit_bias=-2.0,
        spawn_trace_gain=2.0,
        spawn_resource_gain=1.2,
        spawn_hazard_penalty=1.0,
        spawn_viability_gain=1.2,
        split_logit_bias=-0.5,
        split_mass_gain=1.0,
        split_energy_gain=1.0,
        split_boundary_penalty=1.0,
        spawn_candidate_threshold=0.3,
        split_candidate_threshold=0.3,
    )

    assert set(signals) == {
        "trace_density",
        "spawn_drive",
        "spawn_drive_no_trace",
        "trace_ablation_spawn_delta",
        "split_drive",
        "spawn_candidate",
        "split_candidate",
    }


def test_spawn_child_from_primary_mutates_parent_and_returns_child() -> None:
    parent = _SpawnBody(
        centroid_y=10.0,
        centroid_x=10.0,
        radius=6.0,
        aperture_angle=0.0,
        G=0.8,
        B=0.9,
        dead_count=0,
        alive=True,
        body_id=1,
        parent_id=-1,
        generation=0,
    )
    next_id = iter([9]).__next__

    child = spawn_child_from_primary(
        parent,
        can_expand=True,
        tau_G=0.2,
        tau_B=0.2,
        spawn_energy_share=0.25,
        spawn_offset=4.0,
        spawn_radius_scale=0.8,
        image_size=64,
        copy_body=lambda body: replace(body),
        next_body_id=next_id,
    )

    assert child is not None
    assert child.body_id == 9
    assert child.parent_id == 1
    assert child.generation == 1
    assert child.alive is True
    assert child.dead_count == 0
    assert child.centroid_x > parent.centroid_x
    assert parent.G < 0.8
    assert parent.B < 0.9


def test_split_child_from_primary_mutates_positions_and_returns_child() -> None:
    parent = _SpawnBody(
        centroid_y=20.0,
        centroid_x=20.0,
        radius=8.0,
        aperture_angle=0.0,
        G=0.8,
        B=0.8,
        dead_count=0,
        alive=True,
        body_id=2,
        parent_id=-1,
        generation=0,
        mass=1.2,
        energy=0.9,
    )
    next_id = iter([12]).__next__

    child = split_child_from_primary(
        parent,
        can_expand=True,
        tau_G=0.2,
        split_energy_share=0.4,
        split_radius_scale=0.7,
        spawn_offset=4.0,
        image_size=64,
        copy_body=lambda body: replace(body),
        next_body_id=next_id,
    )

    assert child is not None
    assert child.body_id == 12
    assert child.parent_id == 2
    assert child.generation == 1
    assert child.alive is True
    assert child.dead_count == 0
    assert parent.radius < 8.0
    assert child.radius < 8.0
    assert child.centroid_y > parent.centroid_y
