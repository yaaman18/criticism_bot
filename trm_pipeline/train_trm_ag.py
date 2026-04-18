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
from .models import TRMModelConfig, build_trm_ag, require_torch


@dataclass(frozen=True)
class TrainAgConfig:
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 1e-4
    lambda_gated_logits: float = 0.05
    lambda_gated_policy_kl: float = 0.10
    lambda_inhibition_mask: float = 1.50
    lambda_control_mode: float = 0.75
    inhibition_positive_weight: float = 6.0


def load_ag_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out: dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "ag_target_control_mode":
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


def filter_manifest_by_episode_family(
    manifest: list[dict[str, Any]],
    episode_family: str | None,
) -> list[dict[str, Any]]:
    if not episode_family:
        return manifest
    return [meta for meta in manifest if str(meta.get("episode_family", "")) == episode_family]


def _load_batch(rows: list[tuple[dict[str, Any], int]]) -> dict[str, np.ndarray]:
    batch = {
        "ag_input_view": [],
        "ag_target_gated_logits": [],
        "ag_target_inhibition_mask": [],
        "ag_target_control_mode": [],
    }
    for meta, i in rows:
        episode = load_ag_episode(meta["path"])
        batch["ag_input_view"].append(episode[meta.get("input_view_key", "ag_input_view")][i].astype(np.float32))
        batch["ag_target_gated_logits"].append(
            episode[meta.get("target_gated_logits_key", "ag_target_gated_logits")][i].astype(np.float32)
        )
        batch["ag_target_inhibition_mask"].append(
            episode[meta.get("target_inhibition_mask_key", "ag_target_inhibition_mask")][i].astype(np.float32)
        )
        batch["ag_target_control_mode"].append(
            episode[meta.get("target_control_mode_key", "ag_target_control_mode")][i].astype(np.int64)
        )
    return {
        "ag_input_view": np.stack(batch["ag_input_view"], axis=0).astype(np.float32),
        "ag_target_gated_logits": np.stack(batch["ag_target_gated_logits"], axis=0).astype(np.float32),
        "ag_target_inhibition_mask": np.stack(batch["ag_target_inhibition_mask"], axis=0).astype(np.float32),
        "ag_target_control_mode": np.asarray(batch["ag_target_control_mode"], dtype=np.int64),
    }


def compute_trm_ag_loss(outputs, batch, config: TrainAgConfig):
    torch, _, F = require_torch()
    loss_gated_logits = F.mse_loss(outputs["gated_policy_logits"], batch["ag_target_gated_logits"])
    target_policy = torch.softmax(batch["ag_target_gated_logits"], dim=-1)
    pred_log_policy = torch.log_softmax(outputs["gated_policy_logits"], dim=-1)
    loss_gated_policy_kl = F.kl_div(pred_log_policy, target_policy, reduction="batchmean")
    target_inhibition = torch.clamp(batch["ag_target_inhibition_mask"], 0.0, 1.0)
    pos_weight = torch.full(
        (target_inhibition.shape[-1],),
        float(config.inhibition_positive_weight),
        dtype=target_inhibition.dtype,
        device=target_inhibition.device,
    )
    loss_inhibition = F.binary_cross_entropy_with_logits(
        outputs["inhibition_logits"],
        target_inhibition,
        pos_weight=pos_weight,
    )
    loss_control_mode = F.cross_entropy(outputs["control_mode_logits"], batch["ag_target_control_mode"])
    total = (
        config.lambda_gated_logits * loss_gated_logits
        + config.lambda_gated_policy_kl * loss_gated_policy_kl
        + config.lambda_inhibition_mask * loss_inhibition
        + config.lambda_control_mode * loss_control_mode
    )
    return total, {
        "loss_gated_logits": float(loss_gated_logits.detach().cpu().item()),
        "loss_gated_policy_kl": float(loss_gated_policy_kl.detach().cpu().item()),
        "loss_inhibition_mask": float(loss_inhibition.detach().cpu().item()),
        "loss_control_mode": float(loss_control_mode.detach().cpu().item()),
    }


def evaluate_trm_ag(model, manifest: list[dict[str, Any]], config: TrainAgConfig) -> dict[str, float]:
    torch, _, _ = require_torch()
    device = next(model.parameters()).device
    index = build_index(manifest, "val")
    if not index:
        return {
            "val_inhibition_mask_mae": float("nan"),
            "val_inhibition_block_recall": float("nan"),
            "val_control_mode_accuracy": float("nan"),
            "val_gated_policy_kl": float("nan"),
            "val_gated_top1_agreement": float("nan"),
        }
    pred_gated = []
    pred_mask = []
    pred_mode = []
    target_gated = []
    target_mask = []
    target_mode = []
    with torch.no_grad():
        for start in range(0, len(index), config.batch_size):
            rows = index[start : start + config.batch_size]
            batch_np = _load_batch(rows)
            outputs = model(torch.from_numpy(batch_np["ag_input_view"]).to(device))
            pred_gated.append(outputs["gated_policy_logits"].cpu().numpy())
            pred_mask.append(outputs["inhibition_mask"].cpu().numpy())
            pred_mode.append(outputs["control_mode_logits"].cpu().numpy())
            target_gated.append(batch_np["ag_target_gated_logits"])
            target_mask.append(batch_np["ag_target_inhibition_mask"])
            target_mode.append(batch_np["ag_target_control_mode"])
    pred_gated_arr = np.concatenate(pred_gated, axis=0)
    pred_mask_arr = np.concatenate(pred_mask, axis=0)
    pred_mode_arr = np.concatenate(pred_mode, axis=0)
    target_gated_arr = np.concatenate(target_gated, axis=0)
    target_mask_arr = np.concatenate(target_mask, axis=0)
    target_mode_arr = np.concatenate(target_mode, axis=0)
    pred_policy = np.exp(pred_gated_arr - np.max(pred_gated_arr, axis=1, keepdims=True))
    pred_policy /= np.clip(pred_policy.sum(axis=1, keepdims=True), 1e-8, None)
    target_policy = np.exp(target_gated_arr - np.max(target_gated_arr, axis=1, keepdims=True))
    target_policy /= np.clip(target_policy.sum(axis=1, keepdims=True), 1e-8, None)
    gated_policy_kl = float(
        np.mean(np.sum(target_policy * (np.log(np.clip(target_policy, 1e-8, 1.0)) - np.log(np.clip(pred_policy, 1e-8, 1.0))), axis=1))
    )
    pred_block = pred_mask_arr >= 0.60
    target_block = target_mask_arr >= 0.60
    positive = float(np.sum(target_block))
    block_recall = float(np.sum(pred_block & target_block) / positive) if positive > 0 else float("nan")
    return {
        "val_inhibition_mask_mae": float(np.mean(np.abs(pred_mask_arr - target_mask_arr))),
        "val_inhibition_block_recall": block_recall,
        "val_control_mode_accuracy": float(np.mean(np.argmax(pred_mode_arr, axis=1) == target_mode_arr)),
        "val_gated_policy_kl": gated_policy_kl,
        "val_gated_top1_agreement": float(
            np.mean(np.argmax(pred_gated_arr, axis=1) == np.argmax(target_gated_arr, axis=1))
        ),
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: TRMModelConfig,
    train_config: TrainAgConfig,
    root_seed: int,
    resume_path: str | Path | None = None,
    device: str | None = None,
    grad_clip: float | None = None,
    use_amp: bool = False,
    amp_dtype: str = "float16",
    log_interval: int = 0,
    episode_family: str | None = None,
) -> None:
    torch, _, _ = require_torch()
    seed_everything(root_seed)
    device = resolve_torch_device(device)
    amp_enabled = resolve_amp_enabled(device, use_amp)
    amp_dtype_obj = amp_dtype_from_name(torch, amp_dtype)
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    manifest = filter_manifest_by_episode_family(load_jsonl(manifest_path), episode_family)
    train_index = build_index(manifest, "train")
    if not train_index:
        raise SystemExit("no TRM-Ag training samples found")
    output_dir = ensure_dir(output_dir)
    model = build_trm_ag(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_metric = float("-inf")
    best_path = output_dir / "trm_ag_best.pt"
    checkpoint_path = output_dir / "trm_ag.pt"
    epoch_log_path = output_dir / "trm_ag_epoch_log.jsonl"
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
                outputs = model(batch["ag_input_view"])
                total_loss, loss_parts = compute_trm_ag_loss(outputs, batch, train_config)
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
        eval_metrics = evaluate_trm_ag(model, manifest, train_config)
        mean_train = {key: float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        row = {"epoch": epoch, **mean_train, **eval_metrics}
        history.append(row)
        append_jsonl(epoch_log_path, row)
        metric = 0.0
        block_recall = float(eval_metrics.get("val_inhibition_block_recall", float("nan")))
        if np.isfinite(block_recall):
            metric += block_recall
        control_acc = float(eval_metrics.get("val_control_mode_accuracy", float("nan")))
        if np.isfinite(control_acc):
            metric += 0.15 * control_acc
        improved = bool(np.isfinite(metric) and metric >= best_metric)
        if improved:
            best_metric = metric
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "module_name": "trm_ag",
            "module_role": "action_gating",
            "episode_family": episode_family,
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
    save_json(output_dir / "trm_ag_history.json", history)
    save_json(output_dir / "trm_ag_metrics_latest.json", history[-1] if history else {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM-Ag on action-gating views.")
    parser.add_argument("--manifest", default="data/trm_va_cache/views/trm_ag.jsonl")
    parser.add_argument("--output-dir", default="artifacts/trm_ag")
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
    parser.add_argument("--episode-family", default=None)
    args = parser.parse_args()
    train(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_config=TRMModelConfig(in_channels=22),
        train_config=TrainAgConfig(
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
        episode_family=args.episode_family,
    )


if __name__ == "__main__":
    main()
