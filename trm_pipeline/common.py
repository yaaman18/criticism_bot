from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_minmax(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def robust_percentile_range(
    arr: np.ndarray, low: float = 1.0, high: float = 99.0, eps: float = 1e-8
) -> np.ndarray:
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi - lo < eps:
        return normalize_minmax(arr, eps=eps)
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def center_of_mass(arr: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    total = float(arr.sum())
    if total < eps:
        h, w = arr.shape
        return h / 2.0, w / 2.0
    yy, xx = np.indices(arr.shape, dtype=np.float32)
    cy = float((yy * arr).sum() / total)
    cx = float((xx * arr).sum() / total)
    return cy, cx


def nmse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    mse = float(np.mean((pred - target) ** 2))
    denom = float(np.mean(target**2)) + eps
    return mse / denom


def gaussian_noise(
    rng: np.random.Generator, shape: tuple[int, ...], sigma: float
) -> np.ndarray:
    return rng.normal(loc=0.0, scale=sigma, size=shape).astype(np.float32)


def choose_split(index: int, total: int) -> str:
    ratio = (index + 0.5) / max(1, total)
    if ratio <= 0.70:
        return "train"
    if ratio <= 0.85:
        return "val"
    return "test"


def activity_summary(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mass": float(arr.sum()),
    }


def reject_scalar_episode(summary: dict[str, float]) -> bool:
    if summary["mass"] < 1e-2:
        return True
    if summary["mean"] < 5e-4:
        return True
    if summary["mean"] > 0.95:
        return True
    if summary["std"] < 1e-4:
        return True
    return False


def first_halt_step(probs: np.ndarray, threshold: float) -> int:
    for i, value in enumerate(probs, start=1):
        if value >= threshold:
            return i
    return len(probs)


def smoothness_penalty(arr: np.ndarray) -> float:
    dy = np.diff(arr, axis=1)
    dx = np.diff(arr, axis=2)
    return float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))


def math_isfinite_dict(values: dict[str, float]) -> bool:
    return all(math.isfinite(v) for v in values.values())


def classify_regime_from_scalar_states(
    states: np.ndarray,
    delta_threshold: float = 0.02,
    variability_threshold: float = 0.01,
) -> tuple[str, dict[str, float]]:
    arr = states.astype(np.float32, copy=False)
    if arr.shape[0] < 2:
        return "stable", {
            "delta_mean": 0.0,
            "delta_std": 0.0,
            "frame_mean_std": 0.0,
        }
    delta = np.abs(arr[1:] - arr[:-1]).mean(axis=(1, 2))
    frame_means = arr.reshape(arr.shape[0], -1).mean(axis=1)
    stats = {
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "frame_mean_std": float(frame_means.std()),
    }
    is_stable = (
        stats["delta_mean"] <= delta_threshold
        and stats["delta_std"] <= variability_threshold
    )
    return ("stable" if is_stable else "chaotic"), stats


def classify_regime_from_multistates(states: np.ndarray) -> tuple[str, dict[str, float]]:
    arr = states.astype(np.float32, copy=False)
    if arr.shape[0] < 2:
        return "stable", {
            "delta_mean": 0.0,
            "delta_std": 0.0,
            "frame_mean_std": 0.0,
        }
    membrane = arr[..., 0]
    delta = np.abs(membrane[1:] - membrane[:-1]).mean(axis=(1, 2))
    frame_means = membrane.reshape(membrane.shape[0], -1).mean(axis=1)
    stats = {
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "frame_mean_std": float(frame_means.std()),
    }
    is_stable = (
        stats["delta_mean"] <= 0.02
        and stats["delta_std"] <= 0.01
    )
    return ("stable" if is_stable else "chaotic"), stats


def coverage_at_one_sigma(
    pred_mean: np.ndarray,
    pred_logvar: np.ndarray,
    target: np.ndarray,
) -> float:
    sigma = np.sqrt(np.exp(np.clip(pred_logvar, -12.0, 12.0))).astype(np.float32)
    inside = np.abs(target - pred_mean) <= sigma
    return float(np.mean(inside))


def standardized_residual_variance(
    pred_mean: np.ndarray,
    pred_logvar: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    sigma = np.sqrt(np.exp(np.clip(pred_logvar, -12.0, 12.0))).astype(np.float32)
    residual = (target - pred_mean) / np.maximum(sigma, eps)
    return float(np.var(residual))


def count_parameters(model: Any) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def resolve_torch_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_amp_enabled(device: str, use_amp: bool) -> bool:
    return bool(use_amp and str(device).startswith("cuda"))


def amp_dtype_from_name(torch: Any, amp_dtype: str):
    name = str(amp_dtype).lower()
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def move_to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value
