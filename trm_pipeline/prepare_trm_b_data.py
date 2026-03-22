from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .common import ensure_dir, load_jsonl, robust_percentile_range, save_json, save_jsonl
from .models import TRMModelConfig, build_trm_a, require_torch


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(arr.astype(np.float32))
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def build_boundary_targets(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary_targets = []
    permeability_targets = []
    low_grad_mask = []
    prev_boundary = np.zeros(states.shape[1:3], dtype=np.float32)
    for state in states:
        membrane = state[..., 0]
        nucleus = state[..., 2]
        grad = gradient_magnitude(membrane)
        low_grad = float(np.mean(np.abs(grad))) < 1e-4 or float(np.std(np.abs(grad))) < 1e-4
        if low_grad:
            boundary = prev_boundary.copy()
        else:
            threshold = float(np.percentile(grad, 85.0))
            boundary = (grad >= threshold).astype(np.float32)
        nucleus_threshold = float(np.percentile(nucleus, 80.0))
        inside_seed = (nucleus >= nucleus_threshold).astype(np.float32)
        boundary = 0.3 * boundary + 0.7 * prev_boundary
        permeability = robust_percentile_range(boundary * (membrane + inside_seed))
        prev_boundary = boundary.astype(np.float32)
        boundary_targets.append(boundary[..., None].astype(np.float32))
        permeability_targets.append(permeability[..., None].astype(np.float32))
        low_grad_mask.append(np.float32(low_grad))
    return (
        np.stack(boundary_targets, axis=0),
        np.stack(permeability_targets, axis=0),
        np.asarray(low_grad_mask, dtype=np.float32),
    )


def prepare_trm_b_cache(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
) -> Path:
    torch, _, _ = require_torch()
    manifest = load_jsonl(manifest_path)
    output_root = ensure_dir(output_root)
    cache_dir = ensure_dir(output_root / "episodes")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = TRMModelConfig(**checkpoint.get("model_config", {}))
    model = build_trm_a(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cache_manifest: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in manifest:
            with np.load(row["path"]) as data:
                states = data["multi_states"].astype(np.float32)
            inputs = torch.from_numpy(states[:-1])
            outputs = model(inputs)
            pred = outputs["pred_state_t1"].cpu().numpy()
            pred_logvar = outputs["pred_logvar_t1"].cpu().numpy()
            pred_var = np.exp(np.clip(pred_logvar, -12.0, 12.0)).astype(np.float32)
            error_map = np.abs(pred - states[1:]).astype(np.float32)
            surprise_map = (
                0.5
                * (
                    np.exp(-np.clip(pred_logvar, -12.0, 12.0)) * (states[1:] - pred) ** 2
                    + pred_logvar
                )
            ).astype(np.float32)
            precision_map = np.exp(-np.clip(pred_logvar, -12.0, 12.0)).astype(np.float32)
            delta_state = (states[1:] - states[:-1]).astype(np.float32)
            boundary_target, permeability_target, low_grad_mask = build_boundary_targets(
                states[:-1]
            )
            episode_id = row["episode_id"]
            path = cache_dir / f"{episode_id}.npz"
            np.savez_compressed(
                path,
                state_t=states[:-1].astype(np.float32),
                state_t1=states[1:].astype(np.float32),
                delta_state=delta_state,
                pred_state=pred.astype(np.float32),
                pred_logvar=pred_logvar.astype(np.float32),
                pred_var=pred_var,
                error_map=error_map,
                surprise_map=surprise_map,
                precision_map=precision_map,
                boundary_target=boundary_target,
                permeability_target=permeability_target,
                low_grad_mask=low_grad_mask,
            )
            cache_manifest.append(
                {
                    "episode_id": episode_id,
                    "split": row["split"],
                    "path": str(path),
                    "num_pairs": int(states.shape[0] - 1),
                    "seed_id": row["seed_id"],
                    "low_grad_ratio": float(low_grad_mask.mean()),
                    "regime": row.get("regime", "unknown"),
                    "mean_pred_var": float(pred_var.mean()),
                }
            )
    manifest_out = output_root / "manifest.jsonl"
    save_jsonl(manifest_out, cache_manifest)
    save_json(
        output_root / "summary.json",
        {
            "source_manifest": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "num_cached_episodes": len(cache_manifest),
        },
    )
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen TRM-A outputs for TRM-B.")
    parser.add_argument("--manifest", default="data/trm_rollouts/manifest.jsonl")
    parser.add_argument("--checkpoint", default="artifacts/trm_a/trm_a.pt")
    parser.add_argument("--output-root", default="data/trm_b_cache")
    args = parser.parse_args()
    manifest = prepare_trm_b_cache(args.manifest, args.checkpoint, args.output_root)
    print(f"wrote TRM-B cache manifest: {manifest}")


if __name__ == "__main__":
    main()
