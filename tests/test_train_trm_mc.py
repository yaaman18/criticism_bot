from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trm_pipeline.models import TRMModelConfig
from trm_pipeline.train_trm_mc import TrainMcConfig, evaluate_trm_mc, train


def _mc_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    rng = np.random.default_rng(11)
    for split, episode_id in (("train", "mc_train"), ("val", "mc_val")):
        n = 8
        steps = 8
        feat = 44
        mc_input_view = rng.uniform(-1.0, 1.0, size=(n, steps, feat)).astype(np.float32)
        mc_window_mask = np.zeros((n, steps), dtype=np.float32)
        for i in range(n):
            valid = min(steps, 1 + (i % steps))
            mc_window_mask[i, -valid:] = 1.0
            if valid < steps:
                mc_input_view[i, :-valid] = 0.0
        weights = mc_window_mask / np.clip(mc_window_mask.sum(axis=1, keepdims=True), 1.0, None)
        mc_target_context_state = (mc_input_view * weights[..., None]).sum(axis=1).astype(np.float32)
        mc_target_action_bias = mc_target_context_state[:, :5].astype(np.float32)
        mc_target_boundary_bias = mc_target_context_state[:, 5:8].astype(np.float32)
        path = tmp_path / f"{episode_id}.npz"
        np.savez_compressed(
            path,
            mc_input_view=mc_input_view,
            mc_window_mask=mc_window_mask,
            mc_target_context_state=mc_target_context_state,
            mc_target_action_bias=mc_target_action_bias,
            mc_target_boundary_bias=mc_target_boundary_bias,
        )
        rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "path": str(path),
                "num_samples": n,
                "input_view_key": "mc_input_view",
                "window_mask_key": "mc_window_mask",
                "target_context_key": "mc_target_context_state",
                "target_action_bias_key": "mc_target_action_bias",
                "target_boundary_bias_key": "mc_target_boundary_bias",
            }
        )
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return manifest_path


def _small_model_config() -> TRMModelConfig:
    return TRMModelConfig(image_size=8, patch_size=2, dim=32, recursions=2, num_heads=4, mlp_ratio=2, in_channels=44, z_dim=8)


def test_evaluate_trm_mc_runs_on_small_manifest(tmp_path: Path) -> None:
    manifest_path = _mc_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from trm_pipeline.models import build_trm_mc

    model = build_trm_mc(_small_model_config())
    metrics = evaluate_trm_mc(model, rows, TrainMcConfig(batch_size=4))

    assert set(metrics) == {
        "val_context_state_loss",
        "val_action_bias_loss",
        "val_boundary_bias_loss",
        "val_action_bias_alignment",
        "val_nonzero_context_fraction",
        "val_context_variance",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_train_trm_mc_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = _mc_manifest(tmp_path)
    output_dir = tmp_path / "trm_mc_out"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainMcConfig(batch_size=4, epochs=1, learning_rate=1e-4),
        root_seed=123,
        use_amp=True,
        log_interval=1,
    )

    assert (output_dir / "trm_mc.pt").exists()
    assert (output_dir / "trm_mc_best.pt").exists()
    assert (output_dir / "trm_mc_history.json").exists()
    assert (output_dir / "trm_mc_metrics_latest.json").exists()
    assert (output_dir / "trm_mc_epoch_log.jsonl").exists()
    checkpoint = torch.load(output_dir / "trm_mc.pt", map_location="cpu")
    assert checkpoint["amp_requested"] is True
    assert checkpoint["amp_enabled"] is False


def test_train_trm_mc_can_resume_from_checkpoint(tmp_path: Path) -> None:
    manifest_path = _mc_manifest(tmp_path)
    output_dir = tmp_path / "trm_mc_resume"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainMcConfig(batch_size=4, epochs=1, learning_rate=1e-4),
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
        train_config=TrainMcConfig(batch_size=4, epochs=2, learning_rate=1e-4),
        root_seed=123,
        resume_path=output_dir / "trm_mc.pt",
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )

    history = json.loads((output_dir / "trm_mc_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
    epoch_log_lines = [
        line
        for line in (output_dir / "trm_mc_epoch_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(epoch_log_lines) == 2
