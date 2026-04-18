from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trm_pipeline.models import TRMModelConfig, build_trm_a, require_torch
from trm_pipeline.train_trm_a import (
    TrainConfig,
    build_pair_index,
    compute_trm_a_loss,
    current_beta_kl,
    evaluate_trm_a,
    objective_requires_posterior,
    train,
)


def _synthetic_episode(image_size: int = 32, num_frames: int = 10) -> np.ndarray:
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    frames = []
    for t in range(num_frames):
        cy = 16.0
        cx = 12.0 + 0.5 * t
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        membrane = np.exp(-dist2 / (2.0 * 4.0**2)).astype(np.float32)
        membrane /= max(float(membrane.max()), 1e-6)
        cytoplasm = np.clip(membrane * 0.8, 0.0, 1.0)
        nucleus = np.exp(-dist2 / (2.0 * 2.0**2)).astype(np.float32)
        nucleus /= max(float(nucleus.max()), 1e-6)
        dna = np.full_like(membrane, 0.5, dtype=np.float32)
        rna = np.clip(membrane - 0.3, 0.0, 1.0).astype(np.float32)
        frames.append(np.stack([membrane, cytoplasm, nucleus, dna, rna], axis=-1))
    return np.stack(frames, axis=0).astype(np.float32)


def _manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    for split, episode_id in (("train", "ep_train"), ("val", "ep_val")):
        states = _synthetic_episode()
        path = tmp_path / f"{episode_id}.npz"
        np.savez_compressed(path, multi_states=states)
        rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "path": str(path),
                "num_pairs": int(states.shape[0] - 1),
                "seed_id": f"seed_{episode_id}",
                "regime": "stable",
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def _multispecies_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest_multispecies.jsonl"
    rows = []
    rng = np.random.default_rng(42)
    for split, episode_id in (("train", "ep_train_multi"), ("val", "ep_val_multi")):
        wp_input_view = rng.random((10, 32, 32, 18), dtype=np.float32)
        wp_target_observation = rng.random((10, 32, 32, 11), dtype=np.float32)
        wp_observation = np.clip(wp_target_observation + 0.05, 0.0, 1.0).astype(np.float32)
        path = tmp_path / f"{episode_id}.npz"
        np.savez_compressed(
            path,
            wp_input_view=wp_input_view,
            wp_target_observation=wp_target_observation,
            wp_observation=wp_observation,
        )
        rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "path": str(path),
                "num_samples": int(wp_input_view.shape[0]),
                "seed_id": f"seed_{episode_id}",
                "regime": "stable",
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def _small_model_config() -> TRMModelConfig:
    return TRMModelConfig(
        image_size=32,
        patch_size=8,
        dim=32,
        recursions=2,
        num_heads=4,
        mlp_ratio=2,
        in_channels=5,
        z_dim=8,
        max_params=7_000_000,
    )


def test_objective_requires_posterior_only_for_variational() -> None:
    assert objective_requires_posterior("variational") is True
    assert objective_requires_posterior("deterministic") is False
    assert objective_requires_posterior("gaussian_nll") is False


def test_current_beta_kl_warmup_behaves_as_expected() -> None:
    config = TrainConfig(epochs=4, objective="variational", beta_kl=1e-3, kl_warmup_fraction=0.5)
    assert np.isclose(current_beta_kl(config, 1), 5e-4)
    assert np.isclose(current_beta_kl(config, 2), 1e-3)
    assert np.isclose(current_beta_kl(config, 3), 1e-3)


def test_build_pair_index_counts_pairs_by_split() -> None:
    manifest = [
        {"split": "train", "num_pairs": 3},
        {"split": "val", "num_pairs": 2},
        {"split": "train", "num_pairs": 1},
    ]
    assert len(build_pair_index(manifest, "train")) == 4
    assert len(build_pair_index(manifest, "val")) == 2


def test_compute_trm_a_loss_returns_finite_values_for_variational_objective() -> None:
    model = build_trm_a(_small_model_config())
    x = np.random.default_rng(0).random((2, 32, 32, 5), dtype=np.float32)
    y = np.random.default_rng(1).random((2, 32, 32, 5), dtype=np.float32)
    torch_mod, _, _ = require_torch()
    x_t = torch_mod.from_numpy(x)
    y_t = torch_mod.from_numpy(y)
    outputs = model(x_t, targets=y_t, use_posterior=True, sample_latent=True)

    total, parts = compute_trm_a_loss(outputs, y_t, TrainConfig(objective="variational"), beta_kl_now=1e-3)

    assert float(total.detach().cpu().item()) >= 0.0
    assert all(np.isfinite(value) for value in parts.values())


def test_evaluate_trm_a_runs_on_small_manifest(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = build_trm_a(_small_model_config())
    metrics = evaluate_trm_a(model, rows, TrainConfig(batch_size=2, max_val_rollout_episodes=1, objective="gaussian_nll"))

    assert set(metrics) == {
        "val_nmse",
        "baseline_nmse",
        "improvement_over_baseline",
        "rollout_nmse_8",
        "mean_recursion_depth",
        "val_nll",
        "mean_kl",
        "mean_pred_var",
        "coverage_1sigma_all",
        "coverage_1sigma_stable",
        "coverage_1sigma_chaotic",
        "standardized_residual_var",
    }
    assert np.isfinite(metrics["val_nmse"])
    assert np.isfinite(metrics["rollout_nmse_8"])


def test_train_trm_a_multispecies_wp_view_smoke(tmp_path: Path) -> None:
    manifest_path = _multispecies_manifest(tmp_path)
    output_dir = tmp_path / "trm_a_wp_out"

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
            in_channels=18,
            out_channels=11,
            z_dim=8,
            max_params=7_000_000,
        ),
        train_config=TrainConfig(
            batch_size=2,
            epochs=1,
            learning_rate=1e-4,
            objective="gaussian_nll",
            max_val_rollout_episodes=1,
        ),
        root_seed=123,
        input_key="wp_input_view",
        target_key="wp_target_observation",
        baseline_key="wp_observation",
        use_amp=True,
        log_interval=1,
    )

    metrics = json.loads((output_dir / "trm_a_metrics_latest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "trm_a_summary.json").read_text(encoding="utf-8"))
    assert np.isfinite(metrics["val_nmse"])
    assert np.isfinite(metrics["baseline_nmse"])
    assert np.isnan(metrics["rollout_nmse_8"])
    assert summary["input_key"] == "wp_input_view"
    assert summary["target_key"] == "wp_target_observation"
    assert summary["baseline_key"] == "wp_observation"


def test_train_trm_a_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    output_dir = tmp_path / "trm_a_out"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainConfig(
            batch_size=2,
            epochs=1,
            learning_rate=1e-4,
            objective="gaussian_nll",
            max_val_rollout_episodes=1,
        ),
        root_seed=123,
        use_amp=True,
        log_interval=1,
    )

    assert (output_dir / "trm_a.pt").exists()
    assert (output_dir / "trm_a_best.pt").exists()
    assert (output_dir / "trm_a_history.json").exists()
    assert (output_dir / "trm_a_metrics_latest.json").exists()
    assert (output_dir / "trm_a_summary.json").exists()
    assert (output_dir / "trm_a_epoch_log.jsonl").exists()
    checkpoint = torch.load(output_dir / "trm_a.pt", map_location="cpu")
    assert checkpoint["amp_requested"] is True
    assert checkpoint["amp_enabled"] is False


def test_train_trm_a_can_resume_from_checkpoint(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    output_dir = tmp_path / "trm_a_resume"

    train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_config=_small_model_config(),
        train_config=TrainConfig(batch_size=2, epochs=1, learning_rate=1e-4, objective="gaussian_nll", max_val_rollout_episodes=1),
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
        train_config=TrainConfig(batch_size=2, epochs=2, learning_rate=1e-4, objective="gaussian_nll", max_val_rollout_episodes=1),
        root_seed=123,
        resume_path=output_dir / "trm_a.pt",
        device="cpu",
        grad_clip=0.5,
        use_amp=True,
        log_interval=1,
    )

    history = json.loads((output_dir / "trm_a_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
    epoch_log_lines = [line for line in (output_dir / "trm_a_epoch_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(epoch_log_lines) == 2
