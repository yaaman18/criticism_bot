from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trm_pipeline.models import TRMModelConfig
from trm_pipeline.train_trm_bp import TrainBpConfig, evaluate_trm_bp, train


def _bp_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    rng = np.random.default_rng(7)
    for split, episode_id in (("train", "bp_train"), ("val", "bp_val")):
        n = 8
        bp_input_view = rng.uniform(0.0, 1.0, size=(n, 16, 16, 21)).astype(np.float32)
        base_perm = np.clip(bp_input_view[..., 1:2] * 0.75 + bp_input_view[..., 0:1] * 0.10, 0.0, 1.0)
        boundary_mass = bp_input_view[..., 0:1].mean(axis=(1, 2, 3), keepdims=False).astype(np.float32)[:, None]
        viability = bp_input_view[..., -2:].mean(axis=(1, 2)).astype(np.float32)
        target_interface_gain = (boundary_mass - 0.35).astype(np.float32)
        target_aperture_gain = np.clip(0.5 * viability[:, :1] + 0.15, 0.0, 1.0).astype(np.float32)
        target_mode = np.where(
            target_interface_gain[:, 0] > 0.05,
            1,
            np.where(target_aperture_gain[:, 0] > 0.45, 0, 2),
        ).astype(np.int64)
        path = tmp_path / f"{episode_id}.npz"
        np.savez_compressed(
            path,
            bp_input_view=bp_input_view,
            bp_target_permeability_patch=base_perm.astype(np.float32),
            bp_target_interface_gain=target_interface_gain.astype(np.float32),
            bp_target_aperture_gain=target_aperture_gain.astype(np.float32),
            bp_target_mode=target_mode.astype(np.int64),
        )
        rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "path": str(path),
                "num_samples": n,
                "input_view_key": "bp_input_view",
                "target_permeability_patch_key": "bp_target_permeability_patch",
                "target_interface_gain_key": "bp_target_interface_gain",
                "target_aperture_gain_key": "bp_target_aperture_gain",
                "target_mode_key": "bp_target_mode",
            }
        )
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return manifest_path


def _small_model_config() -> TRMModelConfig:
    return TRMModelConfig(image_size=16, patch_size=8, dim=32, recursions=2, num_heads=4, mlp_ratio=2, in_channels=21, z_dim=8)


def test_evaluate_trm_bp_runs_on_small_manifest(tmp_path: Path) -> None:
    manifest_path = _bp_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from trm_pipeline.models import build_trm_bp

    model = build_trm_bp(_small_model_config())
    metrics = evaluate_trm_bp(model, rows, TrainBpConfig(batch_size=4))

    assert set(metrics) == {
        "val_permeability_patch_mae",
        "val_interface_gain_mae",
        "val_aperture_gain_mae",
        "val_mode_accuracy",
        "val_permeability_patch_nmse",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_train_trm_bp_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = _bp_manifest(tmp_path)
    output_dir = tmp_path / "trm_bp_out"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainBpConfig(batch_size=4, epochs=1, learning_rate=1e-4),
        root_seed=123,
        use_amp=True,
        log_interval=1,
    )

    assert (output_dir / "trm_bp.pt").exists()
    assert (output_dir / "trm_bp_best.pt").exists()
    assert (output_dir / "trm_bp_history.json").exists()
    assert (output_dir / "trm_bp_metrics_latest.json").exists()
    assert (output_dir / "trm_bp_epoch_log.jsonl").exists()
    checkpoint = torch.load(output_dir / "trm_bp.pt", map_location="cpu")
    assert checkpoint["amp_requested"] is True
    assert checkpoint["amp_enabled"] is False


def test_train_trm_bp_can_resume_from_checkpoint(tmp_path: Path) -> None:
    manifest_path = _bp_manifest(tmp_path)
    output_dir = tmp_path / "trm_bp_resume"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainBpConfig(batch_size=4, epochs=1, learning_rate=1e-4),
        root_seed=123,
        resume_path=None,
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )
    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainBpConfig(batch_size=4, epochs=2, learning_rate=1e-4),
        root_seed=123,
        resume_path=output_dir / "trm_bp.pt",
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )

    history = json.loads((output_dir / "trm_bp_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
    epoch_log_lines = [
        line
        for line in (output_dir / "trm_bp_epoch_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(epoch_log_lines) == 2
