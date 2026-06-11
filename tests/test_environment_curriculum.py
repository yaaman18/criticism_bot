from __future__ import annotations

import pytest

from trm_pipeline.environment_curriculum import (
    environment_config_for_regime,
    normalize_regime_names,
    regime_manifest,
)
from trm_pipeline.erie_runtime import EnvironmentConfig


def test_environment_config_for_regime_overrides_external_state_pressure() -> None:
    base = EnvironmentConfig(image_size=32, target_radius=8)

    toxic = environment_config_for_regime(base, "toxic_band")
    easy = environment_config_for_regime(base, "easy")

    assert toxic.image_size == 32
    assert toxic.target_radius == 8
    assert toxic.hazard_patches > base.hazard_patches
    assert toxic.toxicity_drift_sigma > base.toxicity_drift_sigma
    assert easy.resource_patches > base.resource_patches
    assert easy.hazard_patches < base.hazard_patches


def test_regime_manifest_includes_canonical_environment_config() -> None:
    manifest = regime_manifest(EnvironmentConfig(), ["balanced", "sparse_energy"])

    assert [row["name"] for row in manifest] == ["balanced", "sparse_energy"]
    assert manifest[1]["environment_config_canonical"]["energy_gradient_patches"] == 1
    assert "compat_aliases" in manifest[1]["environment_config_canonical"]


def test_normalize_regime_names_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError, match="unknown environment regime"):
        normalize_regime_names(["balanced", "missing"])
