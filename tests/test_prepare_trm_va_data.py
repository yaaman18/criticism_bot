from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trm_pipeline.erie_runtime import EnvironmentConfig, RuntimeConfig
from trm_pipeline.prepare_trm_va_data import prepare_trm_va_cache


def _seed_catalog(tmp_path: Path) -> Path:
    catalog_path = tmp_path / "seed_catalog.json"
    rows = [
        {
            "source_file": "test_seed.json",
            "code": "test",
            "name": "unit-seed",
            "params": {"R": 12, "T": 10, "b": "1", "kn": 1, "gn": 1},
            "cells": "3o$3o$3o!",
        }
    ]
    catalog_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return catalog_path


def test_prepare_trm_va_cache_writes_vm_and_as_artifacts(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "va_cache"

    manifest_path = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=6, warmup_steps=1, seed=1234),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=2,
    )

    assert manifest_path.exists()
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    cache_path = Path(rows[0]["path"])
    assert cache_path.exists()

    with np.load(cache_path) as data:
        assert data["vm_viability_state"].shape[1] == 2
        assert data["vm_contact_state"].shape[1] == 3
        assert data["vm_action_cost"].shape[1] == 1
        assert data["vm_target_state"].shape[1] == 2
        assert data["as_action_scores"].shape[1] == 5
        assert data["as_uncertainty_state"].shape[1] == 3
        assert data["as_target_policy"].shape[1] == 5
        assert data["as_target_action"].ndim == 1
        assert np.all(data["as_target_policy"] >= 0.0)


def test_prepare_trm_va_cache_reuse_output_root_clears_stale_artifacts(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "va_cache_reuse"

    first_manifest = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=6, warmup_steps=1, seed=1234),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=3,
    )
    assert first_manifest.exists()
    stale_file = output_root / "episodes" / "stale.npz"
    stale_file.write_bytes(b"stale")

    second_manifest = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=4, warmup_steps=1, seed=1235),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=2,
    )

    rows = [json.loads(line) for line in second_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert not stale_file.exists()
    episode_files = sorted((output_root / "episodes").glob("*.npz"))
    assert len(episode_files) == 2
