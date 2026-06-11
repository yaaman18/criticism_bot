from __future__ import annotations

import numpy as np
import pytest

from trm_pipeline.prediction_cell import (
    LocalPredictionCellLayer,
    PredictionCell,
    PredictionCellGrid,
    PredictionCellUpdateConfig,
    ReceptiveFieldTopology,
)


def _config() -> PredictionCellUpdateConfig:
    return PredictionCellUpdateConfig(
        learning_rate=0.1,
        p_min=0.05,
        p_max=20.0,
        logvar_drift=0.02,
        evidence_logvar_gain=0.18,
    )


def test_prediction_cell_matches_precision_weighted_update() -> None:
    cell = PredictionCell(
        belief=np.array([0.0, 0.5], dtype=np.float32),
        logvar=np.array([0.0, 0.0], dtype=np.float32),
    )

    result = cell.update(np.array([1.0, 1.0], dtype=np.float32), _config())

    np.testing.assert_allclose(result.belief, np.array([0.1, 0.55], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(cell.belief, result.belief)
    np.testing.assert_allclose(cell.logvar, np.full(2, -0.16, dtype=np.float32), atol=1e-6)
    assert result.reconstruction > 0.0
    assert result.complexity > 0.0
    assert result.total == pytest.approx(result.reconstruction + result.complexity)


def test_prediction_cell_grid_can_update_only_gated_local_cells() -> None:
    grid = PredictionCellGrid.zeros((2, 2, 1), logvar_init=0.0)
    observation = np.ones((2, 2, 1), dtype=np.float32)
    gate = np.array([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=np.float32)

    result = grid.update(observation, _config(), gate=gate)

    expected = np.array([[[0.1], [0.0]], [[0.0], [0.1]]], dtype=np.float32)
    np.testing.assert_allclose(result.belief, expected, atol=1e-6)
    np.testing.assert_allclose(result.error, gate, atol=1e-6)
    np.testing.assert_allclose(
        result.logvar,
        np.array([[[-0.16], [0.02]], [[0.02], [-0.16]]], dtype=np.float32),
        atol=1e-6,
    )


def test_prediction_cell_grid_lower_logvar_produces_larger_update() -> None:
    low_uncertainty = PredictionCellGrid.zeros((1, 1, 1), logvar_init=-1.0)
    high_uncertainty = PredictionCellGrid.zeros((1, 1, 1), logvar_init=1.0)
    observation = np.ones((1, 1, 1), dtype=np.float32)

    low = low_uncertainty.update(observation, _config())
    high = high_uncertainty.update(observation, _config())

    assert float(low.belief.mean()) > float(high.belief.mean())


def test_prediction_cell_grid_rejects_shape_mismatch() -> None:
    grid = PredictionCellGrid.zeros((2, 2, 1), logvar_init=0.0)

    with pytest.raises(ValueError, match="observation shape"):
        grid.update(np.ones((2, 2, 2), dtype=np.float32), _config())


def test_receptive_field_topology_pools_and_expands_local_patches() -> None:
    topology = ReceptiveFieldTopology(input_height=4, input_width=4, rows=2, cols=2)
    image = np.arange(16, dtype=np.float32).reshape(4, 4, 1)

    pooled = topology.pool_image(image)

    np.testing.assert_allclose(
        pooled[..., 0],
        np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32),
        atol=1e-6,
    )

    expanded = topology.expand_cells(pooled)
    np.testing.assert_allclose(expanded[:2, :2, 0], 2.5, atol=1e-6)
    np.testing.assert_allclose(expanded[:2, 2:, 0], 4.5, atol=1e-6)
    np.testing.assert_allclose(expanded[2:, :2, 0], 10.5, atol=1e-6)
    np.testing.assert_allclose(expanded[2:, 2:, 0], 12.5, atol=1e-6)


def test_receptive_field_topology_reports_local_neighbors() -> None:
    topology = ReceptiveFieldTopology(input_height=6, input_width=6, rows=3, cols=3)

    assert set(topology.neighbor_indices(1, 1)) == {(0, 1), (2, 1), (1, 0), (1, 2)}
    assert set(topology.neighbor_indices(0, 0)) == {(1, 0), (0, 1)}
    assert (1, 1) in set(topology.neighbor_indices(0, 0, include_diagonal=True))


def test_local_prediction_cell_layer_updates_receptive_fields_independently() -> None:
    layer = LocalPredictionCellLayer.from_image_shape(
        (4, 4, 1),
        cell_rows=2,
        cell_cols=2,
        logvar_init=0.0,
    )
    image = np.ones((4, 4, 1), dtype=np.float32)
    gate = np.zeros((4, 4, 1), dtype=np.float32)
    gate[:2, :2, 0] = 1.0

    result = layer.update_from_image(image, _config(), gate_image=gate)

    expected = np.array([[[0.1], [0.0]], [[0.0], [0.0]]], dtype=np.float32)
    np.testing.assert_allclose(result.belief, expected, atol=1e-6)
    np.testing.assert_allclose(layer.projected_belief()[:2, :2, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(layer.projected_belief()[2:, 2:, 0], 0.0, atol=1e-6)


def test_local_prediction_cell_layer_lateral_coupling_uses_neighbor_context() -> None:
    layer = LocalPredictionCellLayer.from_image_shape(
        (4, 4, 1),
        cell_rows=2,
        cell_cols=2,
        logvar_init=0.0,
    )
    layer.cells.belief[..., 0] = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    image = np.zeros((4, 4, 1), dtype=np.float32)

    result = layer.update_from_image(image, _config(), lateral_coupling=1.0)

    assert float(result.belief[0, 1, 0]) > 0.0
    assert float(result.belief[1, 0, 0]) > 0.0
    assert float(result.belief[0, 0, 0]) < 1.0
