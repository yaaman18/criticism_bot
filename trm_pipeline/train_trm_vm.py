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
from .models import TRMModelConfig, build_trm_vm, require_torch


@dataclass(frozen=True)
class TrainVmConfig:
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 1e-4
    lambda_risk: float = 0.5


def load_vm_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) if key != "as_target_action" else data[key] for key in data.files}


def build_index(manifest: list[dict[str, Any]], split: str) -> list[tuple[dict[str, Any], int]]:
    rows: list[tuple[dict[str, Any], int]] = []
    for meta in manifest:
        if meta["split"] != split:
            continue
        for i in range(int(meta["num_samples"])):
            rows.append((meta, i))
    return rows


def _load_batch(rows: list[tuple[dict[str, Any], int]]) -> dict[str, np.ndarray]:
    keys = [
        "vm_viability_state",
        "vm_contact_state",
        "vm_action_cost",
        "vm_target_state",
        "vm_target_homeostatic_error",
        "vm_target_risk",
    ]
    batch = {key: [] for key in keys}
    for meta, i in rows:
        episode = load_vm_episode(meta["path"])
        for key in keys:
            batch[key].append(episode[key][i])
    return {key: np.stack(value, axis=0).astype(np.float32) for key, value in batch.items()}


def compute_trm_vm_loss(outputs, batch, config: TrainVmConfig):
    torch, _, F = require_torch()
    loss_state = F.mse_loss(outputs["viability_state"], batch["vm_target_state"])
    loss_homeostasis = F.mse_loss(outputs["homeostatic_error"], batch["vm_target_homeostatic_error"])
    loss_risk = F.binary_cross_entropy(outputs["viability_risk"], batch["vm_target_risk"])
    total = loss_state + loss_homeostasis + config.lambda_risk * loss_risk
    return total, {
        "loss_state": float(loss_state.detach().cpu().item()),
        "loss_homeostasis": float(loss_homeostasis.detach().cpu().item()),
        "loss_risk": float(loss_risk.detach().cpu().item()),
    }


def _binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(np.float32).reshape(-1)
    score = score.astype(np.float32).reshape(-1)
    pos = score[target >= 0.5]
    neg = score[target < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(np.isclose(p, neg)))
    return wins / float(len(pos) * len(neg))


def evaluate_trm_vm(model, manifest: list[dict[str, Any]], config: TrainVmConfig) -> dict[str, float]:
    torch, _, _ = require_torch()
    device = next(model.parameters()).device
    index = build_index(manifest, "val")
    if not index:
        return {
            "val_viability_mae_G": float("nan"),
            "val_viability_mae_B": float("nan"),
            "val_homeostatic_error_mae": float("nan"),
            "val_viability_risk_auroc": float("nan"),
            "val_margin_to_failure_corr": float("nan"),
        }
    preds_state = []
    preds_error = []
    preds_risk = []
    targets_state = []
    targets_error = []
    targets_risk = []
    with torch.no_grad():
        for start in range(0, len(index), config.batch_size):
            rows = index[start : start + config.batch_size]
            batch_np = _load_batch(rows)
            outputs = model(
                torch.from_numpy(batch_np["vm_viability_state"]).to(device),
                torch.from_numpy(batch_np["vm_contact_state"]).to(device),
                torch.from_numpy(batch_np["vm_action_cost"]).to(device),
            )
            preds_state.append(outputs["viability_state"].cpu().numpy())
            preds_error.append(outputs["homeostatic_error"].cpu().numpy())
            preds_risk.append(outputs["viability_risk"].cpu().numpy())
            targets_state.append(batch_np["vm_target_state"])
            targets_error.append(batch_np["vm_target_homeostatic_error"])
            targets_risk.append(batch_np["vm_target_risk"])
    pred_state = np.concatenate(preds_state, axis=0)
    pred_error = np.concatenate(preds_error, axis=0)
    pred_risk = np.concatenate(preds_risk, axis=0)
    target_state = np.concatenate(targets_state, axis=0)
    target_error = np.concatenate(targets_error, axis=0)
    target_risk = np.concatenate(targets_risk, axis=0)
    margin_pred = np.minimum(pred_state[:, 0] - 0.15, pred_state[:, 1] - 0.20)
    margin_true = np.minimum(target_state[:, 0] - 0.15, target_state[:, 1] - 0.20)
    corr = np.corrcoef(margin_pred, margin_true)[0, 1] if len(margin_pred) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "val_viability_mae_G": float(np.mean(np.abs(pred_state[:, 0] - target_state[:, 0]))),
        "val_viability_mae_B": float(np.mean(np.abs(pred_state[:, 1] - target_state[:, 1]))),
        "val_homeostatic_error_mae": float(np.mean(np.abs(pred_error - target_error))),
        "val_viability_risk_auroc": float(_binary_auroc(target_risk, pred_risk)),
        "val_margin_to_failure_corr": float(corr),
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: TRMModelConfig,
    train_config: TrainVmConfig,
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
        raise SystemExit("no TRM-Vm training samples found")
    output_dir = ensure_dir(output_dir)
    model = build_trm_vm(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_metric = float("inf")
    best_path = output_dir / "trm_vm_best.pt"
    checkpoint_path = output_dir / "trm_vm.pt"
    epoch_log_path = output_dir / "trm_vm_epoch_log.jsonl"
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
                outputs = model(batch["vm_viability_state"], batch["vm_contact_state"], batch["vm_action_cost"])
                total_loss, loss_parts = compute_trm_vm_loss(outputs, batch, train_config)
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
                    print(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "loss_total": loss_parts["loss_total"],
                            "amp_enabled": amp_enabled,
                        }
                    )
        eval_metrics = evaluate_trm_vm(model, manifest, train_config)
        mean_train = {key: float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        row = {"epoch": epoch, **mean_train, **eval_metrics}
        history.append(row)
        append_jsonl(epoch_log_path, row)
        metric = float(eval_metrics["val_homeostatic_error_mae"])
        improved = bool(np.isfinite(metric) and metric <= best_metric)
        if np.isfinite(metric) and metric <= best_metric:
            best_metric = metric
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "module_name": "trm_vm",
            "module_role": "viability_monitor",
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
    save_json(output_dir / "trm_vm_history.json", history)
    save_json(output_dir / "trm_vm_metrics_latest.json", history[-1] if history else {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM-Vm on bootstrap viability cache.")
    parser.add_argument("--manifest", default="data/trm_va_cache/manifest.jsonl")
    parser.add_argument("--output-dir", default="artifacts/trm_vm")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260318)
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
        model_config=TRMModelConfig(),
        train_config=TrainVmConfig(
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
