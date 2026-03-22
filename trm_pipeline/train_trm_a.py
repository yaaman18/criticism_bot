from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    amp_dtype_from_name,
    append_jsonl,
    classify_regime_from_multistates,
    count_parameters,
    coverage_at_one_sigma,
    ensure_dir,
    first_halt_step,
    load_jsonl,
    nmse,
    resolve_amp_enabled,
    resolve_torch_device,
    save_json,
    seed_everything,
    standardized_residual_variance,
)
from .models import TRMModelConfig, build_trm_a, require_torch


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    epochs: int = 4
    learning_rate: float = 1e-4
    lambda_complex: float = 0.01
    halt_threshold: float = 0.7
    max_val_rollout_episodes: int = 12
    objective: str = "variational"
    beta_kl: float = 1e-3
    kl_warmup_fraction: float = 0.25
    free_bits: float = 0.25
    max_params: int = 7_000_000


def load_episode(path: str | Path) -> np.ndarray:
    with np.load(path) as data:
        return data["multi_states"].astype(np.float32)


def load_episode_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def build_pair_index(manifest: list[dict[str, Any]], split: str) -> list[tuple[dict[str, Any], int]]:
    index: list[tuple[dict[str, Any], int]] = []
    for row in manifest:
        if row["split"] != split:
            continue
        for t in range(max(0, int(row["num_pairs"]))):
            index.append((row, t))
    return index


def _load_pairs(rows: list[tuple[dict[str, Any], int]], start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row, t in rows[start:stop]:
        states = load_episode(row["path"])
        xs.append(states[t])
        ys.append(states[t + 1])
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def smoothness_loss_torch(x):
    torch, _, _ = require_torch()
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, 1:, :, :] - x[:, :-1, :, :]
    return dx.abs().mean() + dy.abs().mean()


def gaussian_nll_torch(mean, logvar, target):
    torch, _, _ = require_torch()
    return 0.5 * (torch.exp(-logvar) * (target - mean) ** 2 + logvar).mean()


def kl_divergence_diag_torch(prior_mu, prior_logvar, post_mu, post_logvar, free_bits: float):
    torch, _, _ = require_torch()
    if post_mu is None or post_logvar is None:
        return torch.zeros((), dtype=prior_mu.dtype, device=prior_mu.device)
    kl_per_dim = 0.5 * (
        prior_logvar
        - post_logvar
        + (torch.exp(post_logvar) + (post_mu - prior_mu) ** 2) / torch.exp(prior_logvar)
        - 1.0
    )
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=-1).mean()


def objective_requires_posterior(objective: str) -> bool:
    return objective == "variational"


def current_beta_kl(config: TrainConfig, epoch: int) -> float:
    if config.objective != "variational":
        return 0.0
    warmup_epochs = max(1, int(np.ceil(config.epochs * config.kl_warmup_fraction)))
    if epoch <= warmup_epochs:
        return float(config.beta_kl) * (float(epoch) / float(warmup_epochs))
    return float(config.beta_kl)


def compute_trm_a_loss(outputs, targets, config: TrainConfig, beta_kl_now: float):
    torch, _, F = require_torch()
    pred_steps = outputs["pred_steps"]
    pred_logvar_steps = outputs["pred_logvar_steps"]
    supervised_means = pred_steps[-3:] if len(pred_steps) >= 3 else pred_steps
    supervised_logvars = pred_logvar_steps[-3:] if len(pred_logvar_steps) >= 3 else pred_logvar_steps
    if config.objective == "deterministic":
        acc_losses = [F.mse_loss(pred, targets) for pred in supervised_means]
        loss_acc = torch.stack(acc_losses).mean()
    else:
        acc_losses = [
            gaussian_nll_torch(pred, logvar, targets)
            for pred, logvar in zip(supervised_means, supervised_logvars)
        ]
        loss_acc = torch.stack(acc_losses).mean()
    loss_complex = smoothness_loss_torch(pred_steps[-1]) * config.lambda_complex
    halt_logits = outputs["halt_logits"]
    halt_target = torch.zeros_like(halt_logits)
    halt_target[:, -1] = 1.0
    loss_halt = F.binary_cross_entropy_with_logits(halt_logits, halt_target)
    loss_kl = kl_divergence_diag_torch(
        outputs["prior_mu"],
        outputs["prior_logvar"],
        outputs["post_mu"],
        outputs["post_logvar"],
        free_bits=config.free_bits,
    )
    total = loss_acc + loss_complex + loss_halt + beta_kl_now * loss_kl
    return total, {
        "loss_acc": float(loss_acc.detach().cpu().item()),
        "loss_complex": float(loss_complex.detach().cpu().item()),
        "loss_halt": float(loss_halt.detach().cpu().item()),
        "loss_kl": float(loss_kl.detach().cpu().item()),
        "beta_kl": float(beta_kl_now),
    }


def _episode_regime(row: dict[str, Any], bundle: dict[str, np.ndarray]) -> str:
    regime = row.get("regime")
    if regime in {"stable", "chaotic"}:
        return str(regime)
    if "scalar_states" in bundle:
        from .common import classify_regime_from_scalar_states

        inferred, _ = classify_regime_from_scalar_states(bundle["scalar_states"])
        return inferred
    inferred, _ = classify_regime_from_multistates(bundle["multi_states"])
    return inferred


def evaluate_trm_a(model, manifest: list[dict[str, Any]], config: TrainConfig) -> dict[str, float]:
    torch, _, _ = require_torch()
    model.eval()
    device = next(model.parameters()).device
    val_pairs = build_pair_index(manifest, "val")
    if not val_pairs:
        return {
            "val_nmse": float("nan"),
            "baseline_nmse": float("nan"),
            "improvement_over_baseline": float("nan"),
            "rollout_nmse_8": float("nan"),
            "mean_recursion_depth": float("nan"),
            "val_nll": float("nan"),
            "mean_kl": float("nan"),
            "mean_pred_var": float("nan"),
            "coverage_1sigma_all": float("nan"),
            "coverage_1sigma_stable": float("nan"),
            "coverage_1sigma_chaotic": float("nan"),
            "standardized_residual_var": float("nan"),
        }
    pair_subset = val_pairs[: min(len(val_pairs), 256)]
    batch_size = min(config.batch_size, max(1, len(pair_subset)))
    preds = []
    trues = []
    baselines = []
    logvars = []
    recursion_steps = []
    kls = []
    coverage_all = []
    stable_rows = []
    chaotic_rows = []
    bundle_cache: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(pair_subset), batch_size):
            batch_rows = pair_subset[start : start + batch_size]
            x_np, y_np = _load_pairs(batch_rows, 0, len(batch_rows))
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            outputs = model(x, targets=y, use_posterior=False, sample_latent=False)
            pred = outputs["pred_state_t1"].cpu().numpy()
            pred_logvar = outputs["pred_logvar_t1"].cpu().numpy()
            preds.append(pred)
            trues.append(y_np)
            baselines.append(x_np)
            logvars.append(pred_logvar)
            halt_probs = outputs["halt_prob"].cpu().numpy()
            recursion_steps.extend(
                first_halt_step(prob_row, config.halt_threshold) for prob_row in halt_probs
            )
            if outputs["post_mu"] is not None:
                kl_value = kl_divergence_diag_torch(
                    outputs["prior_mu"],
                    outputs["prior_logvar"],
                    outputs["post_mu"],
                    outputs["post_logvar"],
                    free_bits=0.0,
                )
                kls.append(float(kl_value.cpu().item()))
            for local_idx, (row, _) in enumerate(batch_rows):
                bundle = bundle_cache.get(row["path"])
                if bundle is None:
                    bundle = load_episode_bundle(row["path"])
                    bundle_cache[row["path"]] = bundle
                regime = _episode_regime(row, bundle)
                cov = coverage_at_one_sigma(pred[local_idx], pred_logvar[local_idx], y_np[local_idx])
                coverage_all.append(cov)
                if regime == "stable":
                    stable_rows.append(cov)
                else:
                    chaotic_rows.append(cov)
    pred_all = np.concatenate(preds, axis=0)
    true_all = np.concatenate(trues, axis=0)
    baseline_all = np.concatenate(baselines, axis=0)
    logvar_all = np.concatenate(logvars, axis=0)
    val_nmse = nmse(pred_all, true_all)
    baseline_nmse = nmse(baseline_all, true_all)
    improvement = 1.0 - (val_nmse / max(baseline_nmse, 1e-8))
    val_nll = float(
        0.5
        * np.mean(
            np.exp(-np.clip(logvar_all, -12.0, 12.0)) * (true_all - pred_all) ** 2
            + logvar_all
        )
    )

    rollout_rows = [row for row in manifest if row["split"] == "val"][: config.max_val_rollout_episodes]
    rollout_errors = []
    with torch.no_grad():
        for row in rollout_rows:
            states = load_episode(row["path"])
            if states.shape[0] < 10:
                continue
            current = torch.from_numpy(states[0:1]).to(device)
            preds_roll = []
            for _ in range(8):
                outputs = model(current, use_posterior=False, sample_latent=False)
                current = outputs["pred_state_t1"]
                preds_roll.append(current.cpu().numpy())
            pred_roll = np.concatenate(preds_roll, axis=0)
            true_roll = states[1:9]
            rollout_errors.append(nmse(pred_roll, true_roll))
    rollout_nmse = float(np.mean(rollout_errors)) if rollout_errors else float("nan")
    return {
        "val_nmse": float(val_nmse),
        "baseline_nmse": float(baseline_nmse),
        "improvement_over_baseline": float(improvement),
        "rollout_nmse_8": float(rollout_nmse),
        "mean_recursion_depth": float(np.mean(recursion_steps)),
        "val_nll": float(val_nll),
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
        "mean_pred_var": float(np.mean(np.exp(np.clip(logvar_all, -12.0, 12.0)))),
        "coverage_1sigma_all": float(np.mean(coverage_all)) if coverage_all else float("nan"),
        "coverage_1sigma_stable": float(np.mean(stable_rows)) if stable_rows else float("nan"),
        "coverage_1sigma_chaotic": float(np.mean(chaotic_rows)) if chaotic_rows else float("nan"),
        "standardized_residual_var": standardized_residual_variance(pred_all, logvar_all, true_all),
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: TRMModelConfig,
    train_config: TrainConfig,
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
    train_pairs = build_pair_index(manifest, "train")
    if not train_pairs:
        raise SystemExit("no training pairs found")
    output_dir = ensure_dir(output_dir)
    model = build_trm_a(model_config).to(device)
    parameter_count = count_parameters(model)
    if parameter_count > train_config.max_params:
        raise SystemExit(
            f"TRM-A parameter count {parameter_count} exceeds limit {train_config.max_params}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_metric = float("inf")
    checkpoint_path = output_dir / "trm_a.pt"
    best_path = output_dir / "trm_a_best.pt"
    epoch_log_path = output_dir / "trm_a_epoch_log.jsonl"
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
        order = rng.permutation(len(train_pairs)).tolist()
        epoch_losses = []
        beta_kl_now = current_beta_kl(train_config, epoch)
        for start in range(0, len(order), train_config.batch_size):
            batch_ids = order[start : start + train_config.batch_size]
            batch_rows = [train_pairs[i] for i in batch_ids]
            x_np, y_np = _load_pairs(batch_rows, 0, len(batch_rows))
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype_obj, enabled=amp_enabled):
                outputs = model(
                    x,
                    targets=y if objective_requires_posterior(train_config.objective) else None,
                    use_posterior=objective_requires_posterior(train_config.objective),
                    sample_latent=objective_requires_posterior(train_config.objective),
                )
                total_loss, loss_parts = compute_trm_a_loss(
                    outputs, y, config=train_config, beta_kl_now=beta_kl_now
                )
            scaler.scale(total_loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_parts["loss_total"] = float(total_loss.detach().cpu().item())
            epoch_losses.append(loss_parts)
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
        eval_metrics = evaluate_trm_a(model, manifest, train_config)
        mean_train = {
            key: float(np.mean([row[key] for row in epoch_losses]))
            for key in epoch_losses[0]
        }
        row = {
            "epoch": epoch,
            "parameter_count": float(parameter_count),
            **mean_train,
            **eval_metrics,
        }
        history.append(row)
        append_jsonl(epoch_log_path, row)
        print(row)
        metric = float(eval_metrics["val_nmse"])
        improved = bool(np.isfinite(metric) and metric <= best_metric)
        if improved:
            best_metric = metric
        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "history": history,
            "parameter_count": parameter_count,
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
    save_json(output_dir / "trm_a_history.json", history)
    save_json(
        output_dir / "trm_a_metrics_latest.json",
        history[-1] if history else {},
    )
    save_json(
        output_dir / "trm_a_summary.json",
        {
            "objective": train_config.objective,
            "parameter_count": parameter_count,
            "max_params": train_config.max_params,
            "z_dim": model_config.z_dim,
            "amp_requested": bool(use_amp),
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": amp_dtype,
        },
    )
    print(f"saved checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM-A on Lenia rollouts.")
    parser.add_argument("--manifest", default="data/trm_rollouts/manifest.jsonl")
    parser.add_argument("--output-dir", default="artifacts/trm_a")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-complex", type=float, default=0.01)
    parser.add_argument("--objective", choices=["deterministic", "gaussian_nll", "variational"], default="variational")
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-fraction", type=float, default=0.25)
    parser.add_argument("--free-bits", type=float, default=0.25)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--max-params", type=int, default=7_000_000)
    parser.add_argument("--seed", type=int, default=20260306)
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
        model_config=TRMModelConfig(z_dim=args.z_dim, max_params=args.max_params),
        train_config=TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lambda_complex=args.lambda_complex,
            objective=args.objective,
            beta_kl=args.beta_kl,
            kl_warmup_fraction=args.kl_warmup_fraction,
            free_bits=args.free_bits,
            max_params=args.max_params,
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
