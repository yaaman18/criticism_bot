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
from .models import TRMModelConfig, build_trm_as, require_torch


@dataclass(frozen=True)
class TrainAsConfig:
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 1e-4
    lambda_logits: float = 0.1
    lambda_policy_kl: float = 0.5
    lambda_entropy: float = 0.05
    min_policy_entropy: float = 1.0


def load_as_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out: dict[str, np.ndarray] = {}
        for key in data.files:
            out[key] = data[key].astype(np.float32) if key != "as_target_action" else data[key].astype(np.int64)
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
    keys = [
        "as_viability_state",
        "as_action_scores",
        "as_uncertainty_state",
        "as_target_logits",
        "as_target_policy",
        "as_target_action",
    ]
    batch = {key: [] for key in keys}
    for meta, i in rows:
        episode = load_as_episode(meta["path"])
        for key in keys:
            batch[key].append(episode[key][i])
    stacked: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        if key == "as_target_action":
            stacked[key] = np.asarray(value, dtype=np.int64)
        else:
            stacked[key] = np.stack(value, axis=0).astype(np.float32)
    return stacked


def compute_trm_as_loss(outputs, batch, config: TrainAsConfig):
    torch, _, F = require_torch()
    loss_policy = F.cross_entropy(outputs["policy_logits"], batch["as_target_action"])
    loss_logits = F.mse_loss(outputs["policy_logits"], batch["as_target_logits"])
    target_policy = torch.clamp(batch["as_target_policy"], min=1e-8, max=1.0)
    pred_log_policy = torch.log_softmax(outputs["policy_logits"], dim=-1)
    loss_policy_kl = F.kl_div(pred_log_policy, target_policy, reduction="batchmean")
    pred_policy = torch.softmax(outputs["policy_logits"], dim=-1)
    entropy = -torch.sum(pred_policy * torch.log(torch.clamp(pred_policy, min=1e-8, max=1.0)), dim=-1)
    loss_entropy = torch.relu(torch.tensor(config.min_policy_entropy, device=entropy.device) - entropy.mean())
    total = (
        loss_policy
        + config.lambda_logits * loss_logits
        + config.lambda_policy_kl * loss_policy_kl
        + config.lambda_entropy * loss_entropy
    )
    return total, {
        "loss_policy": float(loss_policy.detach().cpu().item()),
        "loss_logits": float(loss_logits.detach().cpu().item()),
        "loss_policy_kl": float(loss_policy_kl.detach().cpu().item()),
        "loss_entropy": float(loss_entropy.detach().cpu().item()),
    }


def _pairwise_ranking_accuracy(pred_logits: np.ndarray, target_logits: np.ndarray) -> float:
    total = 0
    correct = 0
    for pred_row, target_row in zip(pred_logits, target_logits):
        for i in range(len(pred_row)):
            for j in range(i + 1, len(pred_row)):
                target_order = np.sign(target_row[i] - target_row[j])
                if target_order == 0:
                    continue
                pred_order = np.sign(pred_row[i] - pred_row[j])
                total += 1
                if pred_order == target_order:
                    correct += 1
    if total == 0:
        return 1.0
    return float(correct / total)


def evaluate_trm_as(model, manifest: list[dict[str, Any]], config: TrainAsConfig) -> dict[str, float]:
    torch, _, _ = require_torch()
    device = next(model.parameters()).device
    index = build_index(manifest, "val")
    if not index:
        return {
            "val_top1_action_agreement": float("nan"),
            "val_pairwise_ranking_accuracy": float("nan"),
            "val_expected_homeostatic_delta": float("nan"),
            "val_policy_entropy_mean": float("nan"),
            "val_action_collapse_rate": float("nan"),
        }
    pred_logits_rows = []
    pred_policy_rows = []
    target_logits_rows = []
    target_action_rows = []
    action_score_rows = []
    with torch.no_grad():
        for start in range(0, len(index), config.batch_size):
            rows = index[start : start + config.batch_size]
            batch_np = _load_batch(rows)
            outputs = model(
                torch.from_numpy(batch_np["as_viability_state"]).to(device),
                torch.from_numpy(batch_np["as_action_scores"]).to(device),
                torch.from_numpy(batch_np["as_uncertainty_state"]).to(device),
            )
            pred_logits_rows.append(outputs["policy_logits"].cpu().numpy())
            pred_policy_rows.append(outputs["policy_prob"].cpu().numpy())
            target_logits_rows.append(batch_np["as_target_logits"])
            target_action_rows.append(batch_np["as_target_action"])
            action_score_rows.append(batch_np["as_action_scores"])
    pred_logits = np.concatenate(pred_logits_rows, axis=0)
    pred_policy = np.concatenate(pred_policy_rows, axis=0)
    target_logits = np.concatenate(target_logits_rows, axis=0)
    target_action = np.concatenate(target_action_rows, axis=0)
    action_scores = np.concatenate(action_score_rows, axis=0)
    pred_action = pred_logits.argmax(axis=1)
    top1 = float(np.mean(pred_action == target_action))
    ranking = _pairwise_ranking_accuracy(pred_logits, target_logits)
    selected_score = action_scores[np.arange(len(pred_action)), pred_action]
    entropy = -np.sum(np.clip(pred_policy, 1e-8, 1.0) * np.log(np.clip(pred_policy, 1e-8, 1.0)), axis=1)
    dominant_count = max(int(np.sum(pred_action == action)) for action in range(pred_policy.shape[1]))
    collapse_rate = float(dominant_count / max(len(pred_action), 1))
    return {
        "val_top1_action_agreement": top1,
        "val_pairwise_ranking_accuracy": float(ranking),
        "val_expected_homeostatic_delta": float(np.mean(selected_score)),
        "val_policy_entropy_mean": float(np.mean(entropy)),
        "val_action_collapse_rate": collapse_rate,
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: TRMModelConfig,
    train_config: TrainAsConfig,
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
        raise SystemExit("no TRM-As training samples found")
    output_dir = ensure_dir(output_dir)
    model = build_trm_as(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_metric = float("-inf")
    best_path = output_dir / "trm_as_best.pt"
    checkpoint_path = output_dir / "trm_as.pt"
    epoch_log_path = output_dir / "trm_as_epoch_log.jsonl"
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
            batch = move_to_device({
                "as_viability_state": torch.from_numpy(batch_np["as_viability_state"]),
                "as_action_scores": torch.from_numpy(batch_np["as_action_scores"]),
                "as_uncertainty_state": torch.from_numpy(batch_np["as_uncertainty_state"]),
                "as_target_logits": torch.from_numpy(batch_np["as_target_logits"]),
                "as_target_policy": torch.from_numpy(batch_np["as_target_policy"]),
                "as_target_action": torch.from_numpy(batch_np["as_target_action"]),
            }, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype_obj, enabled=amp_enabled):
                outputs = model(batch["as_viability_state"], batch["as_action_scores"], batch["as_uncertainty_state"])
                total_loss, loss_parts = compute_trm_as_loss(outputs, batch, train_config)
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
        eval_metrics = evaluate_trm_as(model, manifest, train_config)
        mean_train = {key: float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        row = {"epoch": epoch, **mean_train, **eval_metrics}
        history.append(row)
        append_jsonl(epoch_log_path, row)
        metric = float(eval_metrics["val_pairwise_ranking_accuracy"])
        improved = bool(np.isfinite(metric) and metric >= best_metric)
        if improved:
            best_metric = metric
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "module_name": "trm_as",
            "module_role": "action_scoring",
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
    save_json(output_dir / "trm_as_history.json", history)
    save_json(output_dir / "trm_as_metrics_latest.json", history[-1] if history else {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM-As on bootstrap action-score cache.")
    parser.add_argument("--manifest", default="data/trm_va_cache/manifest.jsonl")
    parser.add_argument("--output-dir", default="artifacts/trm_as")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-policy-kl", type=float, default=0.5)
    parser.add_argument("--lambda-entropy", type=float, default=0.05)
    parser.add_argument("--min-policy-entropy", type=float, default=1.0)
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
        train_config=TrainAsConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lambda_policy_kl=args.lambda_policy_kl,
            lambda_entropy=args.lambda_entropy,
            min_policy_entropy=args.min_policy_entropy,
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
