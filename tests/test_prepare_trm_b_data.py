from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trm_pipeline.models import TRMModelConfig, build_trm_a, require_torch
from trm_pipeline.prepare_trm_b_data import (
    build_boundary_targets,
    gradient_magnitude,
    prepare_trm_b_cache,
)


def _synthetic_states(num_frames: int = 5, image_size: int = 32) -> np.ndarray:
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    states = []
    for t in range(num_frames):
        cy = 16.0
        cx = 12.0 + t
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        membrane = np.exp(-dist2 / (2.0 * 4.0**2)).astype(np.float32)
        membrane /= membrane.max()
        cytoplasm = np.clip(membrane * 0.8, 0.0, 1.0)
        nucleus = np.exp(-dist2 / (2.0 * 2.0**2)).astype(np.float32)
        nucleus /= max(float(nucleus.max()), 1e-6)
        dna = np.full_like(membrane, 0.5, dtype=np.float32)
        rna = np.clip(membrane - 0.5, 0.0, 1.0).astype(np.float32)
        states.append(np.stack([membrane, cytoplasm, nucleus, dna, rna], axis=-1))
    return np.stack(states, axis=0).astype(np.float32)


def _trm_a_checkpoint(tmp_path: Path) -> Path:
    torch, _, _ = require_torch()
    config = TRMModelConfig(
        image_size=32,
        patch_size=8,
        dim=32,
        recursions=2,
        num_heads=4,
        mlp_ratio=2,
        in_channels=5,
        z_dim=8,
    )
    model = build_trm_a(config)
    path = tmp_path / "trm_a_test.pt"
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state": model.state_dict(),
        },
        path,
    )
    return path


def test_gradient_magnitude_detects_edge_strength() -> None:
    field = np.zeros((8, 8), dtype=np.float32)
    field[:, 4:] = 1.0
    grad = gradient_magnitude(field)
    assert grad.shape == field.shape
    assert float(grad[:, 3:5].mean()) > float(grad[:, :2].mean())


def test_build_boundary_targets_flags_low_gradient_frames() -> None:
    states = np.zeros((3, 16, 16, 5), dtype=np.float32)
    states[1, 4:12, 4:12, 0] = 1.0
    states[1, 6:10, 6:10, 2] = 1.0
    states[2] = states[1]

    boundary, permeability, low_grad = build_boundary_targets(states)

    assert boundary.shape == (3, 16, 16, 1)
    assert permeability.shape == (3, 16, 16, 1)
    assert low_grad.shape == (3,)
    assert low_grad[0] == 1.0
    assert np.all((boundary >= 0.0) & (boundary <= 1.0))
    assert np.all((permeability >= 0.0) & (permeability <= 1.0))


def test_prepare_trm_b_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    states = _synthetic_states()
    episode_path = tmp_path / "episode.npz"
    np.savez_compressed(episode_path, multi_states=states)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_row = {
        "episode_id": "ep_0001",
        "split": "train",
        "path": str(episode_path),
        "num_pairs": int(states.shape[0] - 1),
        "seed_id": "seed_0001",
        "regime": "stable",
    }
    manifest_path.write_text(json.dumps(manifest_row, ensure_ascii=False) + "\n", encoding="utf-8")

    checkpoint_path = _trm_a_checkpoint(tmp_path)
    output_root = tmp_path / "cache"

    out_manifest = prepare_trm_b_cache(manifest_path, checkpoint_path, output_root)

    assert out_manifest.exists()
    rows = [json.loads(line) for line in out_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "ep_0001"
    cache_path = Path(rows[0]["path"])
    assert cache_path.exists()

    with np.load(cache_path) as data:
        assert data["state_t"].shape[0] == states.shape[0] - 1
        assert data["state_t1"].shape[0] == states.shape[0] - 1
        assert data["error_map"].shape == data["state_t"].shape
        assert data["precision_map"].shape == data["state_t"].shape
        assert data["boundary_target"].shape[-1] == 1
        assert data["permeability_target"].shape[-1] == 1
        assert np.all(data["precision_map"] > 0.0)
