from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    amp_dtype_from_name,
    append_jsonl,
    ensure_dir,
    load_jsonl,
    move_to_device,
    resolve_amp_enabled,
    resolve_torch_device,
    save_json,
    seed_everything,
)
from .models import TRMModelConfig, build_trm_mc, require_torch


@dataclass(frozen=True)
class TrainMcConfig:
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 1e-4
    lambda_context: float = 1.0
    lambda_action_bias: float = 0.75
    lambda_boundary_bias: float = 0.25


def load_mc_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out: dict[str, np.ndarray] = {}
        for key in data.files:
            out[key] = data[key].astype(np.float32)
        return out


def build_index(manifest: list[dict[str, Any]], split: str) -> list[tuple[dict[str, Any], int]]:
    rows: list[tuple[dict[str, Any], int]] = []
    for meta in manifest:
        if meta["split"] != split:
            continue
        for i in range(int(meta["num_samples"])):
            rows.append((meta, i))
    return rows


def _load_batch(rows: list[tuple[dict[str, Any], int]]) -> dict[str, np.ndarray]:
    batch = {
        "mc_input_view": [],
        "mc_window_mask": [],
        "mc_target_context_state": [],
        "mc_target_action_bias": [],
        "mc_target_boundary_bias": [],
    }
    for meta, i in rows:
        episode = load_mc_episode(meta["path"])
        batch["mc_input_view"].append(episode[meta.get("input_view_key", "mc_input_view")][i].astype(np.float32))
        batch["mc_window_mask"].append(episode[meta.get("window_mask_key", "mc_window_mask")][i].astype(np.float32))
        batch["mc_target_context_state"].append(
            episode[meta.get("target_context_key", "mc_target_context_state")][i].astype(np.float32)
        )
        batch["mc_target_action_bias"].append(
            episode[meta.get("target_action_bias_key", "mc_target_action_bias")][i].astype(np.float32)
        )
        batch["mc_target_boundary_bias"].append(
            episode[meta.get("target_boundary_bias_key", "mc_target_boundary_bias")][i].astype(np.float32)
        )
    return {key: np.stack(value, axis=0).astype(np.float32) for key, value in batch.items()}


def compute_trm_mc_loss(outputs, batch, config: TrainMcConfig):
    torch, _, F = require_torch()
    loss_context = F.mse_loss(outputs["retrieved_context"], batch["mc_target_context_state"])
    loss_action = F.mse_loss(outputs["sequence_bias"], batch["mc_target_action_bias"])
    loss_boundary = F.mse_loss(outputs["boundary_control_bias"], batch["mc_target_boundary_bias"])
    total = (
        config.lambda_context * loss_context
        + config.lambda_action_bias * loss_action
        + config.lambda_boundary_bias * loss_boundary
    )
    return total, {
        "loss_context": float(loss_context.detach().cpu().item()),
        "loss_action_bias": float(loss_action.detach().cpu().item()),
        "loss_boundary_bias": float(loss_boundary.detach().cpu().item()),
    }


def evaluate_trm_mc(model, manifest: list[dict[str, Any]], config: TrainMcConfig) -> dict[str, float]:
    torch, _, _ = require_torch()
    device = next(model.parameters()).device
    index = build_index(manifest, "val")
    if not index:
        return {
            "val_context_state_loss": float("nan"),
            "val_action_bias_loss": float("nan"),
            "val_boundary_bias_loss": float("nan"),
            "val_action_bias_alignment": float("nan"),
            "val_nonzero_context_fraction": float("nan"),
            "val_context_variance": float("nan"),
        }
    pred_context = []
    pred_action = []
    pred_boundary = []
    target_context = []
    target_action = []
    target_boundary = []
    with torch.no_grad():
        for start in range(0, len(index), config.batch_size):
            rows = index[start : start + config.batch_size]
            batch_np = _load_batch(rows)
            outputs = model(
                torch.from_numpy(batch_np["mc_input_view"]).to(device),
                torch.from_numpy(batch_np["mc_window_mask"]).to(device),
            )
            pred_context.append(outputs["retrieved_context"].cpu().numpy())
            pred_action.append(outputs["sequence_bias"].cpu().numpy())
            pred_boundary.append(outputs["boundary_control_bias"].cpu().numpy())
            target_context.append(batch_np["mc_target_context_state"])
            target_action.append(batch_np["mc_target_action_bias"])
            target_boundary.append(batch_np["mc_target_boundary_bias"])
    pred_context_arr = np.concatenate(pred_context, axis=0)
    pred_action_arr = np.concatenate(pred_action, axis=0)
    pred_boundary_arr = np.concatenate(pred_boundary, axis=0)
    target_context_arr = np.concatenate(target_context, axis=0)
    target_action_arr = np.concatenate(target_action, axis=0)
    target_boundary_arr = np.concatenate(target_boundary, axis=0)
    action_alignment = float(
        np.mean(np.argmax(pred_action_arr, axis=1) == np.argmax(target_action_arr, axis=1))
    )
    context_norm = np.linalg.norm(pred_context_arr, axis=1)
    return {
        "val_context_state_loss": float(np.mean((pred_context_arr - target_context_arr) ** 2)),
        "val_action_bias_loss": float(np.mean((pred_action_arr - target_action_arr) ** 2)),
        "val_boundary_bias_loss": float(np.mean((pred_boundary_arr - target_boundary_arr) ** 2)),
        "val_action_bias_alignment": action_alignment,
        "val_nonzero_context_fraction": float(np.mean(context_norm > 1e-4)),
        "val_context_variance": float(np.var(pred_context_arr)),
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: TRMModelConfig,
    train_config: TrainMcConfig,
    root_seed: int,
    resume_path: str | Path | None = None,
    device: str | None = None,
    grad_clip: float | None = None,
    use_amp: bool = False,
    amp_dtype: str = "float16",
    log_interval: int = 0,
) -> None:
    torch, _, _ = require_torch()
    seed_everything(root_seed)
    device = resolve_torch_device(device)
    amp_enabled = resolve_amp_enabled(device, use_amp)
    amp_dtype_obj = amp_dtype_from_name(torch, amp_dtype)
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    manifest = load_jsonl(manifest_path)
    train_index = build_index(manifest, "train")
    if not train_index:
        raise SystemExit("no TRM-Mc training samples found")
    output_dir = ensure_dir(output_dir)
    model = build_trm_mc(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_metric = float("-inf")
    best_path = output_dir / "trm_mc_best.pt"
    checkpoint_path = output_dir / "trm_mc.pt"
    epoch_log_path = output_dir / "trm_mc_epoch_log.jsonl"
    if resume_path is None and epoch_log_path.exists():
        epoch_log_path.unlink()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt.get("epoch", len(history))) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
    for epoch in range(start_epoch, train_config.epochs + 1):
        model.train()
        rng = np.random.default_rng(root_seed + epoch)
        order = rng.permutation(len(train_index)).tolist()
        epoch_rows = []
        for start in range(0, len(order), train_config.batch_size):
            rows = [train_index[i] for i in order[start : start + train_config.batch_size]]
            batch_np = _load_batch(rows)
            batch = move_to_device({key: torch.from_numpy(value) for key, value in batch_np.items()}, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype_obj, enabled=amp_enabled):
                outputs = model(batch["mc_input_view"], batch["mc_window_mask"])
                total_loss, loss_parts = compute_trm_mc_loss(outputs, batch, train_config)
            scaler.scale(total_loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_parts["loss_total"] = float(total_loss.detach().cpu().item())
            epoch_rows.append(loss_parts)
            if log_interval and log_interval > 0:
                batch_index = start // train_config.batch_size + 1
                if batch_index % log_interval == 0:
                    print({"epoch": epoch, "batch": batch_index, "loss_total": loss_parts["loss_total"], "amp_enabled": amp_enabled})
        eval_metrics = evaluate_trm_mc(model, manifest, train_config)
        mean_train = {key: float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        row = {"epoch": epoch, **mean_train, **eval_metrics}
        history.append(row)
        append_jsonl(epoch_log_path, row)
        metric = float(eval_metrics["val_action_bias_alignment"])
        improved = bool(np.isfinite(metric) and metric >= best_metric)
        if improved:
            best_metric = metric
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "module_name": "trm_mc",
            "module_role": "memory_context",
            "history": history,
            "device": device,
            "best_metric": best_metric,
            "amp_requested": bool(use_amp),
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": amp_dtype,
            "log_interval": int(log_interval),
        }
        torch.save(latest, checkpoint_path)
        if improved:
            torch.save(latest, best_path)
    save_json(output_dir / "trm_mc_history.json", history)
    save_json(output_dir / "trm_mc_metrics_latest.json", history[-1] if history else {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM-Mc on contextual memory views.")
    parser.add_argument("--manifest", default="data/trm_va_cache/views/trm_mc.jsonl")
    parser.add_argument("--output-dir", default="artifacts/trm_mc")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--log-interval", type=int, default=0)
    args = parser.parse_args()
    train(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_config=TRMModelConfig(in_channels=44),
        train_config=TrainMcConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        ),
        root_seed=args.seed,
        resume_path=args.resume,
        device=args.device,
        grad_clip=args.grad_clip,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
