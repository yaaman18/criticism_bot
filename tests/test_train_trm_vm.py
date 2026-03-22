from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trm_pipeline.models import TRMModelConfig
from trm_pipeline.train_trm_vm import TrainVmConfig, evaluate_trm_vm, train


def _vm_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    rng = np.random.default_rng(0)
    for split, episode_id in (("train", "vm_train"), ("val", "vm_val")):
        n = 8
        viability = rng.uniform(0.2, 0.9, size=(n, 2)).astype(np.float32)
        contacts = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
        cost = rng.uniform(0.0, 0.03, size=(n, 1)).astype(np.float32)
        target_state = np.clip(
            viability
            + np.stack(
                [
                    0.08 * contacts[:, 0] - 0.03 * cost[:, 0],
                    -0.05 * contacts[:, 1] + 0.04 * contacts[:, 2],
                ],
                axis=-1,
            ),
            0.0,
            1.0,
        ).astype(np.float32)
        target_error = np.abs(target_state - np.array([0.55, 0.65], dtype=np.float32)).astype(np.float32)
        target_risk = (
            ((target_state[:, 0] < 0.25) | (target_state[:, 1] < 0.3)).astype(np.float32)[:, None]
        )
        path = tmp_path / f"{episode_id}.npz"
        np.savez_compressed(
            path,
            vm_viability_state=viability,
            vm_contact_state=contacts,
            vm_action_cost=cost,
            vm_target_state=target_state,
            vm_target_homeostatic_error=target_error,
            vm_target_risk=target_risk,
        )
        rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "path": str(path),
                "num_samples": n,
            }
        )
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return manifest_path


def _small_model_config() -> TRMModelConfig:
    return TRMModelConfig(image_size=32, patch_size=8, dim=32, recursions=2, num_heads=4, mlp_ratio=2, z_dim=8)


def test_evaluate_trm_vm_runs_on_small_manifest(tmp_path: Path) -> None:
    manifest_path = _vm_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from trm_pipeline.models import build_trm_vm

    model = build_trm_vm(_small_model_config())
    metrics = evaluate_trm_vm(model, rows, TrainVmConfig(batch_size=4))

    assert set(metrics) == {
        "val_viability_mae_G",
        "val_viability_mae_B",
        "val_homeostatic_error_mae",
        "val_viability_risk_auroc",
        "val_margin_to_failure_corr",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_train_trm_vm_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = _vm_manifest(tmp_path)
    output_dir = tmp_path / "trm_vm_out"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainVmConfig(batch_size=4, epochs=1, learning_rate=1e-4),
        root_seed=123,
        use_amp=True,
        log_interval=1,
    )

    assert (output_dir / "trm_vm.pt").exists()
    assert (output_dir / "trm_vm_best.pt").exists()
    assert (output_dir / "trm_vm_history.json").exists()
    assert (output_dir / "trm_vm_metrics_latest.json").exists()
    assert (output_dir / "trm_vm_epoch_log.jsonl").exists()
    checkpoint = torch.load(output_dir / "trm_vm.pt", map_location="cpu")
    assert checkpoint["amp_requested"] is True
    assert checkpoint["amp_enabled"] is False


def test_train_trm_vm_can_resume_from_checkpoint(tmp_path: Path) -> None:
    manifest_path = _vm_manifest(tmp_path)
    output_dir = tmp_path / "trm_vm_resume"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainVmConfig(batch_size=4, epochs=1, learning_rate=1e-4),
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
        train_config=TrainVmConfig(batch_size=4, epochs=2, learning_rate=1e-4),
        root_seed=123,
        resume_path=output_dir / "trm_vm.pt",
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )

    history = json.loads((output_dir / "trm_vm_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
    epoch_log_lines = [line for line in (output_dir / "trm_vm_epoch_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(epoch_log_lines) == 2
