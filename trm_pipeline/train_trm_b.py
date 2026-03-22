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
    first_halt_step,
    load_jsonl,
    move_to_device,
    resolve_amp_enabled,
    resolve_torch_device,
    save_json,
    seed_everything,
)
from .models import TRMModelConfig, build_trm_b, require_torch


@dataclass(frozen=True)
class TrainBConfig:
    batch_size: int = 8
    epochs: int = 4
    learning_rate: float = 1e-4
    beta_temporal: float = 0.1
    gamma_separation: float = 0.1
    halt_threshold: float = 0.7
    skip_low_grad_frames: bool = True


def load_cache_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def build_index(
    manifest: list[dict[str, Any]],
    split: str,
    skip_low_grad_frames: bool,
) -> list[tuple[dict[str, Any], int]]:
    rows: list[tuple[dict[str, Any], int]] = []
    for meta in manifest:
        if meta["split"] != split:
            continue
        episode = load_cache_episode(meta["path"])
        low_grad = episode["low_grad_mask"]
        for t in range(int(meta["num_pairs"])):
            if skip_low_grad_frames and bool(low_grad[t] > 0.5):
                continue
            rows.append((meta, t))
    return rows


def _load_batch(rows: list[tuple[dict[str, Any], int]]) -> dict[str, np.ndarray]:
    data = {
        "state_t": [],
        "delta_state": [],
        "error_map": [],
        "boundary_target": [],
        "permeability_target": [],
        "state_t1": [],
    }
    for meta, t in rows:
        episode = load_cache_episode(meta["path"])
        for key in data:
            data[key].append(episode[key][t])
    return {key: np.stack(value, axis=0) for key, value in data.items()}


def compute_trm_b_loss(outputs, batch, config: TrainBConfig):
    torch, _, F = require_torch()
    boundary_pred = outputs["boundary_map"]
    permeability_pred = outputs["permeability_map"]
    boundary_target = batch["boundary_target"]
    permeability_target = batch["permeability_target"]
    state_t = batch["state_t"]
    loss_boundary = F.binary_cross_entropy(boundary_pred, boundary_target)
    loss_permeability = F.mse_loss(permeability_pred, permeability_target)
    loss_temporal = torch.mean(torch.abs(boundary_pred * state_t[..., :1] - boundary_target))

    inside_mask = (boundary_target < 0.5).float()
    outside_mask = 1.0 - inside_mask
    nucleus = state_t[..., 2:3]
    inside_mean = (nucleus * inside_mask).sum() / inside_mask.sum().clamp_min(1.0)
    outside_mean = (nucleus * outside_mask).sum() / outside_mask.sum().clamp_min(1.0)
    loss_separation = torch.relu(0.05 - torch.abs(inside_mean - outside_mean))

    halt_logits = outputs["halt_logits"]
    halt_target = torch.zeros_like(halt_logits)
    halt_target[:, -1] = 1.0
    loss_halt = F.binary_cross_entropy_with_logits(halt_logits, halt_target)
    total = (
        loss_boundary
        + loss_permeability
        + config.beta_temporal * loss_temporal
        + config.gamma_separation * loss_separation
        + loss_halt
    )
    return total, {
        "loss_boundary": float(loss_boundary.detach().cpu().item()),
        "loss_permeability": float(loss_permeability.detach().cpu().item()),
        "loss_temporal": float(loss_temporal.detach().cpu().item()),
        "loss_separation": float(loss_separation.detach().cpu().item()),
        "loss_halt": float(loss_halt.detach().cpu().item()),
    }


def boundary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_mask = pred >= 0.5
    target_mask = target >= 0.5
    inter = float(np.logical_and(pred_mask, target_mask).sum())
    union = float(np.logical_or(pred_mask, target_mask).sum())
    if union <= 0:
        return 1.0
    return inter / union


def evaluate_trm_b(model, manifest: list[dict[str, Any]], config: TrainBConfig) -> dict[str, float]:
    torch, _, _ = require_torch()
    model.eval()
    device = next(model.parameters()).device
    index = build_index(manifest, "val", config.skip_low_grad_frames)
    if not index:
        return {
            "boundary_iou": float("nan"),
            "boundary_occupancy": float("nan"),
            "mean_recursion_depth": float("nan"),
            "nucleus_separation": float("nan"),
        }
    subset = index[: min(256, len(index))]
    batch_size = min(config.batch_size, max(1, len(subset)))
    ious = []
    occupancies = []
    depths = []
    separations = []
    with torch.no_grad():
        for start in range(0, len(subset), batch_size):
            batch_np = _load_batch(subset[start : start + batch_size])
            state_t = torch.from_numpy(batch_np["state_t"]).to(device)
            delta = torch.from_numpy(batch_np["delta_state"]).to(device)
            error = torch.from_numpy(batch_np["error_map"]).to(device)
            outputs = model(state_t, delta, error)
            boundary = outputs["boundary_map"].cpu().numpy()
            target = batch_np["boundary_target"]
            ious.extend(boundary_iou(p, t) for p, t in zip(boundary, target))
            occupancies.extend(float(p.mean()) for p in boundary)
            halt_probs = outputs["halt_prob"].cpu().numpy()
            depths.extend(first_halt_step(prob_row, config.halt_threshold) for prob_row in halt_probs)
            nucleus = batch_np["state_t"][..., 2]
            inside = target[..., 0] < 0.5
            outside = target[..., 0] >= 0.5
            for n_map, inside_mask, outside_mask in zip(nucleus, inside, outside):
                if inside_mask.sum() == 0 or outside_mask.sum() == 0:
                    continue
                separations.append(float(abs(n_map[inside_mask].mean() - n_map[outside_mask].mean())))
    return {
        "boundary_iou": float(np.mean(ious)),
        "boundary_occupancy": float(np.mean(occupancies)),
        "mean_recursion_depth": float(np.mean(depths)),
        "nucleus_separation": float(np.mean(separations)) if separations else float("nan"),
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: TRMModelConfig,
    train_config: TrainBConfig,
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
    train_index = build_index(manifest, "train", train_config.skip_low_grad_frames)
    if not train_index:
        raise SystemExit("no TRM-B training samples found")
    output_dir = ensure_dir(output_dir)
    model = build_trm_b(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_metric = float("-inf")
    checkpoint_path = output_dir / "trm_b.pt"
    best_path = output_dir / "trm_b_best.pt"
    epoch_log_path = output_dir / "trm_b_epoch_log.jsonl"
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
            batch_ids = order[start : start + train_config.batch_size]
            rows = [train_index[i] for i in batch_ids]
            batch_np = _load_batch(rows)
            batch = move_to_device({key: torch.from_numpy(value) for key, value in batch_np.items()}, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype_obj, enabled=amp_enabled):
                outputs = model(batch["state_t"], batch["delta_state"], batch["error_map"])
                total_loss, loss_parts = compute_trm_b_loss(outputs, batch, train_config)
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
        eval_metrics = evaluate_trm_b(model, manifest, train_config)
        mean_train = {
            key: float(np.mean([row[key] for row in epoch_rows]))
            for key in epoch_rows[0]
        }
        row = {"epoch": epoch, **mean_train, **eval_metrics}
        history.append(row)
        append_jsonl(epoch_log_path, row)
        print(row)
        metric = float(eval_metrics["boundary_iou"])
        improved = bool(np.isfinite(metric) and metric >= best_metric)
        if improved:
            best_metric = metric
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
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
    save_json(output_dir / "trm_b_history.json", history)
    save_json(output_dir / "trm_b_metrics_latest.json", history[-1] if history else {})
    print(f"saved checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM-B on frozen TRM-A cache.")
    parser.add_argument("--manifest", default="data/trm_b_cache/manifest.jsonl")
    parser.add_argument("--output-dir", default="artifacts/trm_b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--include-low-grad", action="store_true")
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
        train_config=TrainBConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            skip_low_grad_frames=not args.include_low_grad,
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
