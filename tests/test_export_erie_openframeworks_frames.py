from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from trm_pipeline.export_erie_openframeworks_frames import export_openframeworks_frames


def _write_runtime_npz(path: Path) -> None:
    frames = 3
    h = 16
    w = 16
    yy, xx = np.indices((h, w), dtype=np.float32)
    occupancy = np.stack([np.clip(1.0 - ((xx - 8.0) ** 2 + (yy - 8.0) ** 2) / 60.0, 0.0, 1.0) for _ in range(frames)])
    boundary = np.stack([np.abs(np.gradient(occupancy[i])[0]) for i in range(frames)]).astype(np.float32)
    permeability = np.stack([np.full((h, w), 0.4 + 0.1 * i, dtype=np.float32) for i in range(frames)])
    env_channels = np.zeros((frames, h, w, 9), dtype=np.float32)
    env_channels[..., 0] = boundary
    env_channels[..., 1] = occupancy * 0.8
    env_channels[..., 2] = occupancy * 0.6
    env_channels[..., 3] = 0.5
    env_channels[..., 4] = occupancy * 0.3
    env_channels[..., 5] = xx / max(w - 1, 1)
    env_channels[..., 6] = yy / max(h - 1, 1)
    env_channels[..., 7] = (xx[::-1, :] + yy[:, ::-1]) / max(h + w - 2, 1)
    env_channels[..., 8] = (xx + yy) / max(h + w - 2, 1)
    world_logvar = np.zeros((frames, h, w, 9), dtype=np.float32)
    boundary_logvar = np.zeros((frames, h, w, 2), dtype=np.float32)
    species_sources = np.zeros((frames, h, w, 3), dtype=np.float32)
    species_sources[..., 0] = occupancy * 0.7
    species_sources[..., 1] = boundary * 0.9
    species_sources[..., 2] = occupancy * 0.4 + 0.1
    species_fields = np.zeros((frames, h, w, 4), dtype=np.float32)
    species_fields[..., 0] = env_channels[..., 5] * 0.5
    species_fields[..., 1] = env_channels[..., 6] * 0.4
    species_fields[..., 2] = env_channels[..., 7] * 0.6
    species_fields[..., 3] = env_channels[..., 8] * 0.7
    np.savez_compressed(
        path,
        occupancy=occupancy.astype(np.float32),
        boundary=boundary.astype(np.float32),
        permeability=permeability.astype(np.float32),
        env_channels=env_channels.astype(np.float32),
        species_sources=species_sources.astype(np.float32),
        species_fields=species_fields.astype(np.float32),
        world_belief=np.zeros((frames, h, w, 9), dtype=np.float32),
        world_logvar=world_logvar,
        boundary_belief=np.zeros((frames, h, w, 2), dtype=np.float32),
        boundary_logvar=boundary_logvar,
    )


def test_export_openframeworks_frames_writes_manifest_and_pngs(tmp_path: Path) -> None:
    npz_path = tmp_path / "runtime.npz"
    _write_runtime_npz(npz_path)
    output_root = tmp_path / "viewer"

    manifest_path = export_openframeworks_frames(npz_path, output_root)

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 3
    assert manifest["width"] == 16
    assert manifest["height"] == 16
    assert len(manifest["frames"]) == 3

    first = manifest["frames"][0]
    for key in ("life", "field", "body", "aura", "species"):
        image_path = output_root / first[key]
        assert image_path.exists()
        image = Image.open(image_path)
        assert image.size == (16, 16)
        assert image.mode == "RGB"
