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
from .models import TRMModelConfig, build_trm_bp, require_torch


@dataclass(frozen=True)
class TrainBpConfig:
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 1e-4
    lambda_interface: float = 0.5
    lambda_aperture: float = 0.5
    lambda_mode: float = 0.5


def load_bp_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out: dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "bp_target_mode":
                out[key] = data[key].astype(np.int64)
            else:
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


def _meta_key(meta: dict[str, Any], name: str, default: str) -> str:
    value = meta.get(name)
    if value is None:
        return default
    return str(value)


def _load_batch(rows: list[tuple[dict[str, Any], int]]) -> dict[str, np.ndarray]:
    keys = [
        "bp_input_view",
        "bp_target_permeability_patch",
        "bp_target_interface_gain",
        "bp_target_aperture_gain",
        "bp_target_mode",
    ]
    batch = {key: [] for key in keys}
    for meta, i in rows:
        episode = load_bp_episode(meta["path"])
        input_view_key = _meta_key(meta, "input_view_key", "bp_input_view")
        target_perm_key = _meta_key(meta, "target_permeability_patch_key", "bp_target_permeability_patch")
        target_interface_key = _meta_key(meta, "target_interface_gain_key", "bp_target_interface_gain")
        target_aperture_key = _meta_key(meta, "target_aperture_gain_key", "bp_target_aperture_gain")
        target_mode_key = _meta_key(meta, "target_mode_key", "bp_target_mode")
        batch["bp_input_view"].append(episode[input_view_key][i])
        batch["bp_target_permeability_patch"].append(episode[target_perm_key][i])
        batch["bp_target_interface_gain"].append(episode[target_interface_key][i])
        batch["bp_target_aperture_gain"].append(episode[target_aperture_key][i])
        batch["bp_target_mode"].append(int(np.asarray(episode[target_mode_key][i]).reshape(-1)[0]))
    stacked: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        if key == "bp_target_mode":
            stacked[key] = np.asarray(value, dtype=np.int64)
        else:
            stacked[key] = np.stack(value, axis=0).astype(np.float32)
    return stacked


def compute_trm_bp_loss(outputs, batch, config: TrainBpConfig):
    torch, _, F = require_torch()
    loss_permeability = F.mse_loss(
        outputs["pred_permeability_patch"],
        batch["bp_target_permeability_patch"],
    )
    loss_interface = F.mse_loss(
        outputs["pred_interface_gain"],
        batch["bp_target_interface_gain"],
    )
    loss_aperture = F.mse_loss(
        outputs["pred_aperture_gain"],
        batch["bp_target_aperture_gain"],
    )
    loss_mode = F.cross_entropy(outputs["mode_logits"], batch["bp_target_mode"])
    total = (
        loss_permeability
        + config.lambda_interface * loss_interface
        + config.lambda_aperture * loss_aperture
        + config.lambda_mode * loss_mode
    )
    return total, {
        "loss_permeability": float(loss_permeability.detach().cpu().item()),
        "loss_interface": float(loss_interface.detach().cpu().item()),
        "loss_aperture": float(loss_aperture.detach().cpu().item()),
        "loss_mode": float(loss_mode.detach().cpu().item()),
    }


def evaluate_trm_bp(model, manifest: list[dict[str, Any]], config: TrainBpConfig) -> dict[str, float]:
    torch, _, _ = require_torch()
    device = next(model.parameters()).device
    index = build_index(manifest, "val")
    if not index:
        return {
            "val_permeability_patch_mae": float("nan"),
            "val_interface_gain_mae": float("nan"),
            "val_aperture_gain_mae": float("nan"),
            "val_mode_accuracy": float("nan"),
            "val_permeability_patch_nmse": float("nan"),
        }
    pred_perm_rows = []
    pred_interface_rows = []
    pred_aperture_rows = []
    pred_mode_rows = []
    target_perm_rows = []
    target_interface_rows = []
    target_aperture_rows = []
    target_mode_rows = []
    with torch.no_grad():
        for start in range(0, len(index), config.batch_size):
            rows = index[start : start + config.batch_size]
            batch_np = _load_batch(rows)
            outputs = model(torch.from_numpy(batch_np["bp_input_view"]).to(device))
            pred_perm_rows.append(outputs["pred_permeability_patch"].cpu().numpy())
            pred_interface_rows.append(outputs["pred_interface_gain"].cpu().numpy())
            pred_aperture_rows.append(outputs["pred_aperture_gain"].cpu().numpy())
            pred_mode_rows.append(outputs["mode_logits"].argmax(dim=-1).cpu().numpy())
            target_perm_rows.append(batch_np["bp_target_permeability_patch"])
            target_interface_rows.append(batch_np["bp_target_interface_gain"])
            target_aperture_rows.append(batch_np["bp_target_aperture_gain"])
            target_mode_rows.append(batch_np["bp_target_mode"])
    pred_perm = np.concatenate(pred_perm_rows, axis=0)
    pred_interface = np.concatenate(pred_interface_rows, axis=0)
    pred_aperture = np.concatenate(pred_aperture_rows, axis=0)
    pred_mode = np.concatenate(pred_mode_rows, axis=0)
    target_perm = np.concatenate(target_perm_rows, axis=0)
    target_interface = np.concatenate(target_interface_rows, axis=0)
    target_aperture = np.concatenate(target_aperture_rows, axis=0)
    target_mode = np.concatenate(target_mode_rows, axis=0)
    patch_nmse = float(
        np.mean((pred_perm - target_perm) ** 2) / (np.mean(target_perm**2) + 1e-8)
    )
    return {
        "val_permeability_patch_mae": float(np.mean(np.abs(pred_perm - target_perm))),
        "val_interface_gain_mae": float(np.mean(np.abs(pred_interface - target_interface))),
        "val_aperture_gain_mae": float(np.mean(np.abs(pred_aperture - target_aperture))),
        "val_mode_accuracy": float(np.mean(pred_mode == target_mode)),
        "val_permeability_patch_nmse": patch_nmse,
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: TRMModelConfig,
    train_config: TrainBpConfig,
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
        raise SystemExit("no TRM-Bp training samples found")
    output_dir = ensure_dir(output_dir)
    model = build_trm_bp(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_metric = float("inf")
    best_path = output_dir / "trm_bp_best.pt"
    checkpoint_path = output_dir / "trm_bp.pt"
    epoch_log_path = output_dir / "trm_bp_epoch_log.jsonl"
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
            batch = move_to_device(
                {
                    "bp_input_view": torch.from_numpy(batch_np["bp_input_view"]),
                    "bp_target_permeability_patch": torch.from_numpy(batch_np["bp_target_permeability_patch"]),
                    "bp_target_interface_gain": torch.from_numpy(batch_np["bp_target_interface_gain"]),
                    "bp_target_aperture_gain": torch.from_numpy(batch_np["bp_target_aperture_gain"]),
                    "bp_target_mode": torch.from_numpy(batch_np["bp_target_mode"]),
                },
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype_obj, enabled=amp_enabled):
                outputs = model(batch["bp_input_view"])
                total_loss, loss_parts = compute_trm_bp_loss(outputs, batch, train_config)
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
        eval_metrics = evaluate_trm_bp(model, manifest, train_config)
        mean_train = {key: float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        row = {"epoch": epoch, **mean_train, **eval_metrics}
        history.append(row)
        append_jsonl(epoch_log_path, row)
        metric = float(eval_metrics["val_permeability_patch_mae"])
        improved = bool(np.isfinite(metric) and metric <= best_metric)
        if improved:
            best_metric = metric
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "module_name": "trm_bp",
            "module_role": "boundary_permeability_control",
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
    save_json(output_dir / "trm_bp_history.json", history)
    save_json(output_dir / "trm_bp_metrics_latest.json", history[-1] if history else {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM-Bp on bootstrap boundary-control cache.")
    parser.add_argument("--manifest", default="data/trm_va_cache/views/trm_bp.jsonl")
    parser.add_argument("--output-dir", default="artifacts/trm_bp")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-interface", type=float, default=0.5)
    parser.add_argument("--lambda-aperture", type=float, default=0.5)
    parser.add_argument("--lambda-mode", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260404)
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
        model_config=TRMModelConfig(image_size=16, in_channels=21),
        train_config=TrainBpConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lambda_interface=args.lambda_interface,
            lambda_aperture=args.lambda_aperture,
            lambda_mode=args.lambda_mode,
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
