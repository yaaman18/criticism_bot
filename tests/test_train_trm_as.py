from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trm_pipeline.models import TRMModelConfig
from trm_pipeline.train_trm_as import TrainAsConfig, compute_trm_as_loss, evaluate_trm_as, train


def _as_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    rng = np.random.default_rng(1)
    for split, episode_id in (("train", "as_train"), ("val", "as_val")):
        n = 10
        viability = rng.uniform(0.2, 0.9, size=(n, 2)).astype(np.float32)
        action_scores = rng.normal(0.0, 0.4, size=(n, 5)).astype(np.float32)
        uncertainty = rng.uniform(0.0, 1.0, size=(n, 4)).astype(np.float32)
        target_logits = (-3.0 * action_scores + 0.5 * uncertainty[:, :1] - 0.15 * uncertainty[:, 2:3]).astype(np.float32)
        target_logits -= target_logits.mean(axis=1, keepdims=True)
        exp = np.exp(target_logits - target_logits.max(axis=1, keepdims=True))
        target_policy = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
        target_action = target_policy.argmax(axis=1).astype(np.int64)
        path = tmp_path / f"{episode_id}.npz"
        np.savez_compressed(
            path,
            as_viability_state=viability,
            as_action_scores=action_scores,
            as_uncertainty_state=uncertainty,
            as_target_logits=target_logits,
            as_target_policy=target_policy,
            as_target_action=target_action,
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


def test_evaluate_trm_as_runs_on_small_manifest(tmp_path: Path) -> None:
    manifest_path = _as_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from trm_pipeline.models import build_trm_as

    model = build_trm_as(_small_model_config())
    metrics = evaluate_trm_as(model, rows, TrainAsConfig(batch_size=4))

    assert set(metrics) == {
        "val_top1_action_agreement",
        "val_pairwise_ranking_accuracy",
        "val_expected_homeostatic_delta",
        "val_policy_entropy_mean",
        "val_action_collapse_rate",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_compute_trm_as_loss_penalizes_low_entropy_when_enabled() -> None:
    batch = {
        "as_target_action": torch.tensor([0, 1], dtype=torch.int64),
        "as_target_logits": torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0], [1.0, 2.0, 0.0, -1.0, -2.0]], dtype=torch.float32),
        "as_target_policy": torch.tensor(
            [
                [0.45, 0.35, 0.10, 0.05, 0.05],
                [0.35, 0.45, 0.10, 0.05, 0.05],
            ],
            dtype=torch.float32,
        ),
    }
    peaked = {"policy_logits": torch.tensor([[8.0, -4.0, -4.0, -4.0, -4.0], [-4.0, 8.0, -4.0, -4.0, -4.0]], dtype=torch.float32)}
    softer = {"policy_logits": torch.tensor([[2.0, 1.5, 0.5, -0.5, -1.0], [1.5, 2.0, 0.5, -0.5, -1.0]], dtype=torch.float32)}

    config = TrainAsConfig(lambda_policy_kl=0.5, lambda_entropy=0.5, min_policy_entropy=1.2)
    loss_peaked, parts_peaked = compute_trm_as_loss(peaked, batch, config)
    loss_softer, parts_softer = compute_trm_as_loss(softer, batch, config)

    assert float(loss_peaked.detach().cpu().item()) > float(loss_softer.detach().cpu().item())
    assert parts_peaked["loss_entropy"] > 0.0
    assert parts_softer["loss_entropy"] <= parts_peaked["loss_entropy"]


def test_compute_trm_as_loss_penalizes_bad_pairwise_ranking() -> None:
    batch = {
        "as_target_action": torch.tensor([0, 1], dtype=torch.int64),
        "as_target_logits": torch.tensor(
            [[3.0, 2.0, 1.0, 0.0, -1.0], [1.0, 3.0, 2.0, 0.0, -1.0]],
            dtype=torch.float32,
        ),
        "as_target_policy": torch.tensor(
            [
                [0.60, 0.25, 0.10, 0.03, 0.02],
                [0.20, 0.55, 0.18, 0.05, 0.02],
            ],
            dtype=torch.float32,
        ),
    }
    aligned = {
        "policy_logits": torch.tensor(
            [[2.5, 1.5, 0.5, -0.5, -1.5], [0.5, 2.5, 1.5, -0.5, -1.5]],
            dtype=torch.float32,
        )
    }
    reversed_rank = {
        "policy_logits": torch.tensor(
            [[-1.5, -0.5, 0.5, 1.5, 2.5], [-1.5, -0.5, 0.5, 1.5, 2.5]],
            dtype=torch.float32,
        )
    }

    config = TrainAsConfig(lambda_hard_action=0.0, lambda_logits=0.0, lambda_policy_kl=0.0, lambda_pairwise=1.0)
    loss_aligned, parts_aligned = compute_trm_as_loss(aligned, batch, config)
    loss_reversed, parts_reversed = compute_trm_as_loss(reversed_rank, batch, config)

    assert float(loss_reversed.detach().cpu().item()) > float(loss_aligned.detach().cpu().item())
    assert parts_reversed["loss_pairwise"] > parts_aligned["loss_pairwise"]


def test_train_trm_as_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = _as_manifest(tmp_path)
    output_dir = tmp_path / "trm_as_out"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainAsConfig(batch_size=4, epochs=1, learning_rate=1e-4),
        root_seed=123,
        use_amp=True,
        log_interval=1,
    )

    assert (output_dir / "trm_as.pt").exists()
    assert (output_dir / "trm_as_best.pt").exists()
    assert (output_dir / "trm_as_history.json").exists()
    assert (output_dir / "trm_as_metrics_latest.json").exists()
    assert (output_dir / "trm_as_epoch_log.jsonl").exists()
    checkpoint = torch.load(output_dir / "trm_as.pt", map_location="cpu")
    assert checkpoint["amp_requested"] is True
    assert checkpoint["amp_enabled"] is False


def test_train_trm_as_can_resume_from_checkpoint(tmp_path: Path) -> None:
    manifest_path = _as_manifest(tmp_path)
    output_dir = tmp_path / "trm_as_resume"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainAsConfig(batch_size=4, epochs=1, learning_rate=1e-4),
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
        train_config=TrainAsConfig(batch_size=4, epochs=2, learning_rate=1e-4),
        root_seed=123,
        resume_path=output_dir / "trm_as.pt",
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )

    history = json.loads((output_dir / "trm_as_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
    epoch_log_lines = [line for line in (output_dir / "trm_as_epoch_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(epoch_log_lines) == 2
