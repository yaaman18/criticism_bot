from __future__ import annotations

import numpy as np

from trm_pipeline.common import (
    choose_split,
    classify_regime_from_scalar_states,
    coverage_at_one_sigma,
    normalize_minmax,
    robust_percentile_range,
    standardized_residual_variance,
)


def test_choose_split_uses_expected_thresholds() -> None:
    assert choose_split(0, 10) == "train"
    assert choose_split(7, 10) == "val"
    assert choose_split(9, 10) == "test"


def test_normalize_minmax_and_percentile_range_stay_in_unit_interval() -> None:
    arr = np.array([[0.0, 1.0], [2.0, 10.0]], dtype=np.float32)
    norm = normalize_minmax(arr)
    robust = robust_percentile_range(arr)
    assert np.all((norm >= 0.0) & (norm <= 1.0))
    assert np.all((robust >= -1e-6) & (robust <= 1.0 + 1e-6))


def test_regime_classifier_distinguishes_stable_and_chaotic_sequences() -> None:
    stable = np.zeros((4, 8, 8), dtype=np.float32)
    chaotic = np.random.default_rng(0).random((4, 8, 8), dtype=np.float32)
    stable_regime, _ = classify_regime_from_scalar_states(stable)
    chaotic_regime, _ = classify_regime_from_scalar_states(chaotic)
    assert stable_regime == "stable"
    assert chaotic_regime == "chaotic"


def test_coverage_and_standardized_residual_variance_are_finite() -> None:
    mean = np.zeros((4, 4, 2), dtype=np.float32)
    logvar = np.zeros((4, 4, 2), dtype=np.float32)
    target = np.full((4, 4, 2), 0.5, dtype=np.float32)
    cov = coverage_at_one_sigma(mean, logvar, target)
    var = standardized_residual_variance(mean, logvar, target)
    assert np.isfinite(cov)
    assert np.isfinite(var)
