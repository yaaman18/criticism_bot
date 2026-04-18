from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trm_pipeline.models import TRMModelConfig
from trm_pipeline.train_trm_ag import TrainAgConfig, evaluate_trm_ag, filter_manifest_by_episode_family, train


def _ag_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    rng = np.random.default_rng(11)
    for split, episode_id in (("train", "ag_train"), ("val", "ag_val")):
        n = 8
        ag_input_view = rng.uniform(-1.0, 1.0, size=(n, 22)).astype(np.float32)
        base_logits = ag_input_view[:, :5]
        viability = ag_input_view[:, 5:7]
        homeo = ag_input_view[:, 7:9]
        risk = ag_input_view[:, 9:10]
        uncertainty = ag_input_view[:, 10:14]
        stress = ag_input_view[:, 15:17] + ag_input_view[:, 18:20]
        inhibition = np.clip(
            np.stack(
                [
                    np.maximum(stress[:, 0] - ag_input_view[:, 14], 0.0),
                    np.maximum(ag_input_view[:, 14] - stress[:, 0], 0.0),
                    np.maximum(stress[:, 1] + homeo[:, 1], 0.0),
                    np.maximum(homeo[:, 0] - stress[:, 0], 0.0),
                    np.maximum(np.mean(uncertainty, axis=1) - stress[:, 1], 0.0),
                ],
                axis=-1,
            ),
            0.0,
            1.0,
        ).astype(np.float32)
        control_mode = np.where(
            risk[:, 0] > 0.25,
            2,
            np.where(np.mean(uncertainty, axis=1) > 0.15, 0, 1),
        ).astype(np.int64)
        mode_bias = np.array(
            [
                [0.10, -0.05, -0.10, -0.05, 0.10],
                [0.00, 0.00, 0.00, 0.00, 0.00],
                [-0.15, 0.15, -0.20, 0.20, 0.10],
            ],
            dtype=np.float32,
        )[control_mode]
        gated_logits = (base_logits + mode_bias - 1.25 * inhibition).astype(np.float32)
        gated_logits = gated_logits - gated_logits.mean(axis=1, keepdims=True)
        path = tmp_path / f"{episode_id}.npz"
        np.savez_compressed(
            path,
            ag_input_view=ag_input_view,
            ag_target_gated_logits=gated_logits,
            ag_target_inhibition_mask=inhibition,
            ag_target_control_mode=control_mode,
        )
        rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "episode_family": "vent_edge" if split == "train" else "uncertain_corridor",
                "path": str(path),
                "num_samples": n,
                "input_view_key": "ag_input_view",
                "target_gated_logits_key": "ag_target_gated_logits",
                "target_inhibition_mask_key": "ag_target_inhibition_mask",
                "target_control_mode_key": "ag_target_control_mode",
            }
        )
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return manifest_path


def _small_model_config() -> TRMModelConfig:
    return TRMModelConfig(image_size=8, patch_size=8, dim=32, recursions=2, num_heads=4, mlp_ratio=2, in_channels=22, z_dim=8)


def test_evaluate_trm_ag_runs_on_small_manifest(tmp_path: Path) -> None:
    manifest_path = _ag_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from trm_pipeline.models import build_trm_ag

    model = build_trm_ag(_small_model_config())
    metrics = evaluate_trm_ag(model, rows, TrainAgConfig(batch_size=4))

    assert set(metrics) == {
        "val_inhibition_mask_mae",
        "val_inhibition_block_recall",
        "val_control_mode_accuracy",
        "val_gated_policy_kl",
        "val_gated_top1_agreement",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_filter_manifest_by_episode_family_selects_matching_rows(tmp_path: Path) -> None:
    manifest_path = _ag_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    filtered = filter_manifest_by_episode_family(rows, "vent_edge")

    assert len(filtered) == 1
    assert filtered[0]["episode_family"] == "vent_edge"


def test_train_trm_ag_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = _ag_manifest(tmp_path)
    output_dir = tmp_path / "trm_ag_out"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainAgConfig(batch_size=4, epochs=1, learning_rate=1e-4),
        root_seed=123,
        use_amp=True,
        log_interval=1,
    )

    assert (output_dir / "trm_ag.pt").exists()
    assert (output_dir / "trm_ag_best.pt").exists()
    assert (output_dir / "trm_ag_history.json").exists()
    assert (output_dir / "trm_ag_metrics_latest.json").exists()
    assert (output_dir / "trm_ag_epoch_log.jsonl").exists()
    checkpoint = torch.load(output_dir / "trm_ag.pt", map_location="cpu")
    assert checkpoint["amp_requested"] is True
    assert checkpoint["amp_enabled"] is False


def test_train_trm_ag_can_resume_from_checkpoint(tmp_path: Path) -> None:
    manifest_path = _ag_manifest(tmp_path)
    output_dir = tmp_path / "trm_ag_resume"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainAgConfig(batch_size=4, epochs=1, learning_rate=1e-4),
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
        train_config=TrainAgConfig(batch_size=4, epochs=2, learning_rate=1e-4),
        root_seed=123,
        resume_path=output_dir / "trm_ag.pt",
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )

    history = json.loads((output_dir / "trm_ag_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
    epoch_log_lines = [
        line
        for line in (output_dir / "trm_ag_epoch_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(epoch_log_lines) == 2


def test_train_trm_ag_with_episode_family_filter_writes_family_to_checkpoint(tmp_path: Path) -> None:
    manifest_path = _ag_manifest(tmp_path)
    output_dir = tmp_path / "trm_ag_family"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainAgConfig(batch_size=4, epochs=1, learning_rate=1e-4),
        root_seed=123,
        episode_family="vent_edge",
    )

    checkpoint = torch.load(output_dir / "trm_ag.pt", map_location="cpu")
    assert checkpoint["episode_family"] == "vent_edge"
