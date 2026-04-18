from __future__ import annotations

import numpy as np

from trm_pipeline.trm_input_views import (
    build_trm_ag_input_view,
    build_trm_as_input_view,
    build_trm_bd_input_view,
    build_trm_bp_input_view,
    build_trm_mc_input_view,
    build_temporal_context_windows,
    build_trm_vm_input_view,
    build_trm_wp_input_view,
    extract_centered_patch,
)


def test_build_trm_wp_input_view_has_expected_channels() -> None:
    observation = np.zeros((8, 8, 11), dtype=np.float32)
    sensor_gate = np.ones((8, 8, 1), dtype=np.float32)
    species_fields = np.full((8, 8, 4), 0.5, dtype=np.float32)
    flow_channels = np.dstack(
        [
            np.full((8, 8), -0.25, dtype=np.float32),
            np.full((8, 8), 0.75, dtype=np.float32),
        ]
    )

    view = build_trm_wp_input_view(observation, sensor_gate, species_fields, flow_channels)

    assert view.shape == (8, 8, 18)
    np.testing.assert_allclose(view[..., 11], 1.0)
    np.testing.assert_allclose(view[..., 12:16], 0.5)
    np.testing.assert_allclose(view[..., 16], -0.25)
    np.testing.assert_allclose(view[..., 17], 0.75)


def test_build_trm_bd_input_view_has_expected_channels() -> None:
    observation = np.zeros((8, 8, 11), dtype=np.float32)
    world_error = np.ones((8, 8, 11), dtype=np.float32)
    sensor_gate = np.full((8, 8, 1), 0.25, dtype=np.float32)
    delta_observation = np.full((8, 8, 11), -0.5, dtype=np.float32)

    view = build_trm_bd_input_view(observation, world_error, sensor_gate, delta_observation)

    assert view.shape == (8, 8, 34)
    np.testing.assert_allclose(view[..., 11:22], 1.0)
    np.testing.assert_allclose(view[..., 22], 0.25)
    np.testing.assert_allclose(view[..., 23:], -0.5)


def test_build_trm_vm_input_view_has_expected_dim() -> None:
    viability = np.array([0.4, 0.7], dtype=np.float32)
    env_contact = np.array([0.5, 0.2, 0.3, 0.6], dtype=np.float32)
    species_contact = np.array([0.1, 0.4, 0.2, 0.8], dtype=np.float32)
    action_cost = np.array([0.03], dtype=np.float32)

    view = build_trm_vm_input_view(viability, env_contact, species_contact, action_cost)

    assert view.shape == (11,)
    np.testing.assert_allclose(view[:2], viability)
    np.testing.assert_allclose(view[2:6], env_contact)
    np.testing.assert_allclose(view[6:10], species_contact)
    np.testing.assert_allclose(view[10:], action_cost)


def test_extract_centered_patch_pads_at_edges() -> None:
    field = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    patch = extract_centered_patch(field, center_y=0.0, center_x=0.0, patch_size=4)

    assert patch.shape == (4, 4, 1)
    np.testing.assert_allclose(patch[0, 0, 0], 0.0)
    np.testing.assert_allclose(patch[2, 2, 0], field[0, 0])


def test_build_trm_bp_input_view_has_expected_channels() -> None:
    boundary_patch = np.ones((8, 8, 1), dtype=np.float32)
    permeability_patch = np.full((8, 8, 1), 0.25, dtype=np.float32)
    observation_patch = np.zeros((8, 8, 11), dtype=np.float32)
    species_patch = np.full((8, 8, 4), 0.5, dtype=np.float32)
    flow_patch = np.dstack(
        [
            np.full((8, 8), -0.25, dtype=np.float32),
            np.full((8, 8), 0.75, dtype=np.float32),
        ]
    )
    viability = np.array([0.4, 0.7], dtype=np.float32)

    view = build_trm_bp_input_view(
        boundary_patch,
        permeability_patch,
        observation_patch,
        species_patch,
        flow_patch,
        viability,
    )

    assert view.shape == (8, 8, 21)
    np.testing.assert_allclose(view[..., 0], 1.0)
    np.testing.assert_allclose(view[..., 1], 0.25)
    np.testing.assert_allclose(view[..., 13:17], 0.5)
    np.testing.assert_allclose(view[..., 17], -0.25)
    np.testing.assert_allclose(view[..., 18], 0.75)
    np.testing.assert_allclose(view[..., 19], 0.4)
    np.testing.assert_allclose(view[..., 20], 0.7)


def test_build_trm_as_input_view_has_expected_dim() -> None:
    viability = np.array([0.45, 0.62], dtype=np.float32)
    scores = np.array([0.1, 0.2, -0.3, 0.4, -0.1], dtype=np.float32)
    uncertainty = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    env_contact = np.array([0.5, 0.2, 0.1, 0.6], dtype=np.float32)
    species_contact = np.array([0.3, 0.7, 0.4, 0.2], dtype=np.float32)

    view = build_trm_as_input_view(viability, scores, uncertainty, env_contact, species_contact)

    assert view.shape == (19,)
    np.testing.assert_allclose(view[:2], viability)
    np.testing.assert_allclose(view[2:7], scores)
    np.testing.assert_allclose(view[7:11], uncertainty)
    np.testing.assert_allclose(view[11:15], env_contact)
    np.testing.assert_allclose(view[15:19], species_contact)


def test_build_trm_ag_input_view_has_expected_dim() -> None:
    as_logits = np.array([0.1, -0.2, 0.3, 0.0, -0.1], dtype=np.float32)
    viability = np.array([0.45, 0.62], dtype=np.float32)
    homeostatic_error = np.array([0.10, 0.03], dtype=np.float32)
    viability_risk = np.array([0.25], dtype=np.float32)
    uncertainty = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    env_contact = np.array([0.5, 0.2, 0.1, 0.6], dtype=np.float32)
    species_contact = np.array([0.3, 0.7, 0.4, 0.2], dtype=np.float32)

    view = build_trm_ag_input_view(
        as_logits,
        viability,
        homeostatic_error,
        viability_risk,
        uncertainty,
        env_contact,
        species_contact,
    )

    assert view.shape == (22,)
    np.testing.assert_allclose(view[:5], as_logits)
    np.testing.assert_allclose(view[5:7], viability)
    np.testing.assert_allclose(view[7:9], homeostatic_error)
    np.testing.assert_allclose(view[9:10], viability_risk)
    np.testing.assert_allclose(view[10:14], uncertainty)
    np.testing.assert_allclose(view[14:18], env_contact)
    np.testing.assert_allclose(view[18:22], species_contact)


def test_build_temporal_context_windows_right_aligns_recent_history() -> None:
    features = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    windows, masks = build_temporal_context_windows(features, window_size=3)

    assert windows.shape == (4, 3, 3)
    assert masks.shape == (4, 3)
    np.testing.assert_allclose(masks[0], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(windows[0, 2], features[0])
    np.testing.assert_allclose(masks[1], np.array([0.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(windows[1, 1], features[0])
    np.testing.assert_allclose(windows[1, 2], features[1])
    np.testing.assert_allclose(masks[3], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(windows[3, 0], features[1])
    np.testing.assert_allclose(windows[3, 1], features[2])
    np.testing.assert_allclose(windows[3, 2], features[3])


def test_build_trm_mc_input_view_returns_window_and_mask() -> None:
    step_features = np.arange(5 * 30, dtype=np.float32).reshape(5, 30)

    view, mask = build_trm_mc_input_view(step_features, window_size=4)

    assert view.shape == (5, 4, 44)
    assert mask.shape == (5, 4)
    np.testing.assert_allclose(mask[0], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(view[0, 3, :30], step_features[0])
    np.testing.assert_allclose(view[0, 3, 30:], np.zeros((14,), dtype=np.float32))
    np.testing.assert_allclose(mask[-1], np.ones((4,), dtype=np.float32))
    expected_delta = step_features[4, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] - step_features[
        3, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ]
    np.testing.assert_allclose(view[-1, -1, 30:], expected_delta.astype(np.float32))
