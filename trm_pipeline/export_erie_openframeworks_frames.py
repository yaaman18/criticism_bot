from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import ensure_dir


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _life_rgb(env_channels_t: np.ndarray) -> np.ndarray:
    membrane = env_channels_t[..., 0]
    cytoplasm = env_channels_t[..., 1]
    nucleus = env_channels_t[..., 2]
    dna = env_channels_t[..., 3]
    rna = env_channels_t[..., 4]
    rgb = np.stack(
        [
            _clip01(0.70 * nucleus + 0.25 * membrane + 0.15 * rna),
            _clip01(0.70 * cytoplasm + 0.20 * dna + 0.15 * membrane),
            _clip01(0.60 * membrane + 0.25 * rna + 0.20 * nucleus),
        ],
        axis=-1,
    )
    return rgb.astype(np.float32)


def _field_rgb(env_channels_t: np.ndarray) -> np.ndarray:
    energy = env_channels_t[..., 5]
    thermal = env_channels_t[..., 6]
    toxicity = env_channels_t[..., 7]
    niche = env_channels_t[..., 8]
    rgb = np.stack(
        [
            _clip01(0.70 * thermal + 0.40 * toxicity + 0.15 * energy),
            _clip01(0.80 * energy + 0.25 * niche),
            _clip01(0.90 * niche + 0.20 * thermal),
        ],
        axis=-1,
    )
    return rgb.astype(np.float32)


def _species_rgb(species_sources_t: np.ndarray, species_fields_t: np.ndarray | None = None) -> np.ndarray:
    energy_src = species_sources_t[..., 0]
    toxic_src = species_sources_t[..., 1]
    niche_src = species_sources_t[..., 2]
    if species_fields_t is None:
        thermal = toxic_src
        toxicity = toxic_src
        niche_field = niche_src
    else:
        thermal = species_fields_t[..., 1]
        toxicity = species_fields_t[..., 2]
        niche_field = species_fields_t[..., 3]
    rgb = np.stack(
        [
            _clip01(0.85 * toxic_src + 0.20 * thermal),
            _clip01(0.80 * energy_src + 0.15 * niche_field),
            _clip01(0.85 * niche_src + 0.15 * toxicity),
        ],
        axis=-1,
    )
    return rgb.astype(np.float32)


def _body_rgb(occupancy_t: np.ndarray, boundary_t: np.ndarray, permeability_t: np.ndarray) -> np.ndarray:
    boundary_norm = _normalize_map(boundary_t)
    rgb = np.stack(
        [
            _clip01(0.50 * occupancy_t + 0.60 * boundary_norm),
            _clip01(0.35 * occupancy_t + 0.80 * permeability_t),
            _clip01(0.85 * boundary_norm + 0.15 * occupancy_t),
        ],
        axis=-1,
    )
    return rgb.astype(np.float32)


def _aura_rgb(world_logvar_t: np.ndarray, boundary_logvar_t: np.ndarray) -> np.ndarray:
    world_unc = np.exp(np.mean(world_logvar_t[..., :5], axis=-1, dtype=np.float32))
    boundary_unc = np.exp(np.mean(boundary_logvar_t, axis=-1, dtype=np.float32))
    world_unc = _normalize_map(world_unc)
    boundary_unc = _normalize_map(boundary_unc)
    aura = np.stack(
        [
            _clip01(0.80 * world_unc + 0.10 * boundary_unc),
            _clip01(0.15 * world_unc + 0.40 * boundary_unc),
            _clip01(0.95 * boundary_unc + 0.10 * world_unc),
        ],
        axis=-1,
    )
    return aura.astype(np.float32)


def _save_png(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(_to_u8(rgb), mode="RGB").save(path)


def export_openframeworks_frames(npz_path: str | Path, output_root: str | Path) -> Path:
    npz_path = Path(npz_path)
    output_root = Path(output_root)
    frames_dir = ensure_dir(output_root / "frames")
    data = np.load(npz_path)

    occupancy = data["occupancy"]
    boundary = data["boundary"]
    permeability = data["permeability"]
    env_channels = data["env_channels"]
    world_logvar = data["world_logvar"]
    boundary_logvar = data["boundary_logvar"]
    species_sources = data["species_sources"] if "species_sources" in data.files else None
    species_fields = data["species_fields"] if "species_fields" in data.files else None

    frame_count, height, width = occupancy.shape
    manifest_frames: list[dict[str, Any]] = []

    for idx in range(frame_count):
        stem = f"{idx:04d}"
        life_rgb = _life_rgb(env_channels[idx])
        field_rgb = _field_rgb(env_channels[idx])
        body_rgb = _body_rgb(occupancy[idx], boundary[idx], permeability[idx])
        aura_rgb = _aura_rgb(world_logvar[idx], boundary_logvar[idx])
        species_rgb = None
        if species_sources is not None:
            species_rgb = _species_rgb(
                species_sources[idx],
                species_fields[idx] if species_fields is not None else None,
            )

        life_path = frames_dir / f"life_{stem}.png"
        field_path = frames_dir / f"field_{stem}.png"
        body_path = frames_dir / f"body_{stem}.png"
        aura_path = frames_dir / f"aura_{stem}.png"
        species_path = frames_dir / f"species_{stem}.png"
        _save_png(life_path, life_rgb)
        _save_png(field_path, field_rgb)
        _save_png(body_path, body_rgb)
        _save_png(aura_path, aura_rgb)
        if species_rgb is not None:
            _save_png(species_path, species_rgb)
        manifest_frames.append(
            {
                "index": idx,
                "life": str(life_path.relative_to(output_root)),
                "field": str(field_path.relative_to(output_root)),
                "body": str(body_path.relative_to(output_root)),
                "aura": str(aura_path.relative_to(output_root)),
                "species": str(species_path.relative_to(output_root)) if species_rgb is not None else None,
            }
        )

    manifest = {
        "source_npz": str(npz_path.resolve()),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "frames": manifest_frames,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ERIE runtime frames for openFrameworks GLSL visualization."
    )
    parser.add_argument("--npz", required=True, help="Path to ERIE runtime .npz")
    parser.add_argument("--output-root", required=True, help="Directory for exported textures and manifest")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    manifest = export_openframeworks_frames(args.npz, args.output_root)
    print(f"wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
