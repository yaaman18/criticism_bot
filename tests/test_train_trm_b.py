from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trm_pipeline.train_trm_b import (
    TrainBConfig,
    boundary_iou,
    build_index,
    compute_trm_b_loss,
    evaluate_trm_b,
    train,
)
from trm_pipeline.models import TRMModelConfig, build_trm_b, require_torch


def _cache_episode(tmp_path: Path, low_grad_mask: np.ndarray | None = None) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_size = 32
    num_pairs = 4
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    dist2 = (yy - 16.0) ** 2 + (xx - 16.0) ** 2
    membrane = np.exp(-dist2 / (2.0 * 4.0**2)).astype(np.float32)
    membrane /= membrane.max()
    nucleus = np.exp(-dist2 / (2.0 * 2.0**2)).astype(np.float32)
    nucleus /= max(float(nucleus.max()), 1e-6)
    boundary_target = (membrane > 0.4).astype(np.float32)[..., None]
    permeability_target = np.clip(boundary_target * 0.8, 0.0, 1.0).astype(np.float32)
    low_grad = low_grad_mask if low_grad_mask is not None else np.zeros((num_pairs,), dtype=np.float32)
    episode_path = tmp_path / "cache_episode.npz"
    np.savez_compressed(
        episode_path,
        state_t=np.repeat(np.stack([membrane, membrane * 0.8, nucleus, membrane * 0 + 0.5, membrane * 0.2], axis=-1)[None, ...], num_pairs, axis=0).astype(np.float32),
        delta_state=np.zeros((num_pairs, image_size, image_size, 5), dtype=np.float32),
        error_map=np.zeros((num_pairs, image_size, image_size, 5), dtype=np.float32),
        boundary_target=np.repeat(boundary_target[None, ...], num_pairs, axis=0).astype(np.float32),
        permeability_target=np.repeat(permeability_target[None, ...], num_pairs, axis=0).astype(np.float32),
        state_t1=np.repeat(np.stack([membrane, membrane * 0.8, nucleus, membrane * 0 + 0.5, membrane * 0.2], axis=-1)[None, ...], num_pairs, axis=0).astype(np.float32),
        low_grad_mask=low_grad.astype(np.float32),
    )
    meta = {
        "episode_id": "cache_ep_0001",
        "split": "val",
        "path": str(episode_path),
        "num_pairs": num_pairs,
        "seed_id": "seed_0001",
    }
    return episode_path, meta


def test_boundary_iou_matches_expected_overlap() -> None:
    pred = np.array([[[1.0], [1.0]], [[0.0], [0.0]]], dtype=np.float32)
    target = np.array([[[1.0], [0.0]], [[0.0], [0.0]]], dtype=np.float32)
    assert np.isclose(boundary_iou(pred, target), 0.5)


def test_build_index_skips_low_grad_frames(tmp_path: Path) -> None:
    _, meta = _cache_episode(tmp_path, low_grad_mask=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    rows = build_index([meta], "val", skip_low_grad_frames=True)
    assert [t for _, t in rows] == [0, 2]


def test_compute_trm_b_loss_returns_finite_components() -> None:
    torch_mod, _, _ = require_torch()
    batch = {
        "boundary_target": torch_mod.full((2, 32, 32, 1), 0.5),
        "permeability_target": torch_mod.full((2, 32, 32, 1), 0.25),
        "state_t": torch_mod.zeros((2, 32, 32, 5)),
    }
    outputs = {
        "boundary_map": torch_mod.full((2, 32, 32, 1), 0.6),
        "permeability_map": torch_mod.full((2, 32, 32, 1), 0.3),
        "halt_logits": torch_mod.zeros((2, 3)),
    }
    total, parts = compute_trm_b_loss(outputs, batch, TrainBConfig())
    assert float(total.detach().cpu().item()) >= 0.0
    assert all(np.isfinite(value) for value in parts.values())


def test_evaluate_trm_b_runs_on_small_cache_manifest(tmp_path: Path) -> None:
    _, meta = _cache_episode(tmp_path)
    model = build_trm_b(
        TRMModelConfig(
            image_size=32,
            patch_size=8,
            dim=32,
            recursions=2,
            num_heads=4,
            mlp_ratio=2,
        )
    )
    metrics = evaluate_trm_b(model, [meta], TrainBConfig(batch_size=2))
    assert set(metrics) == {
        "boundary_iou",
        "boundary_occupancy",
        "mean_recursion_depth",
        "nucleus_separation",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_train_trm_b_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    _, train_meta = _cache_episode(tmp_path / "train_cache")
    train_meta["split"] = "train"
    _, val_meta = _cache_episode(tmp_path / "val_cache")
    val_meta["split"] = "val"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(train_meta, ensure_ascii=False) + "\n" + json.dumps(val_meta, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "trm_b_out"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=TRMModelConfig(
            image_size=32,
            patch_size=8,
            dim=32,
            recursions=2,
            num_heads=4,
            mlp_ratio=2,
        ),
        train_config=TrainBConfig(batch_size=2, epochs=1, learning_rate=1e-4),
        root_seed=123,
        use_amp=True,
        log_interval=1,
    )

    assert (output_dir / "trm_b.pt").exists()
    assert (output_dir / "trm_b_best.pt").exists()
    assert (output_dir / "trm_b_history.json").exists()
    assert (output_dir / "trm_b_metrics_latest.json").exists()
    assert (output_dir / "trm_b_epoch_log.jsonl").exists()
    checkpoint = torch.load(output_dir / "trm_b.pt", map_location="cpu")
    assert checkpoint["amp_requested"] is True
    assert checkpoint["amp_enabled"] is False


def test_train_trm_b_can_resume_from_checkpoint(tmp_path: Path) -> None:
    _, train_meta = _cache_episode(tmp_path / "train_cache_resume")
    train_meta["split"] = "train"
    _, val_meta = _cache_episode(tmp_path / "val_cache_resume")
    val_meta["split"] = "val"
    manifest_path = tmp_path / "manifest_resume.jsonl"
    manifest_path.write_text(
        json.dumps(train_meta, ensure_ascii=False) + "\n" + json.dumps(val_meta, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "trm_b_resume"

    config = TRMModelConfig(image_size=32, patch_size=8, dim=32, recursions=2, num_heads=4, mlp_ratio=2)
    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=config,
        train_config=TrainBConfig(batch_size=2, epochs=1, learning_rate=1e-4),
        root_seed=123,
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )
    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=config,
        train_config=TrainBConfig(batch_size=2, epochs=2, learning_rate=1e-4),
        root_seed=123,
        resume_path=output_dir / "trm_b.pt",
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )

    history = json.loads((output_dir / "trm_b_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
