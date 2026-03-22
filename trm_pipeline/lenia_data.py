from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    activity_summary,
    classify_regime_from_scalar_states,
    center_of_mass,
    choose_split,
    ensure_dir,
    gaussian_noise,
    load_json,
    normalize_minmax,
    reject_scalar_episode,
    robust_percentile_range,
    save_json,
    save_jsonl,
    seed_everything,
)

VALUE_CHARS = {".": 0, "b": 0, "o": 255}


@dataclass(frozen=True)
class LeniaSeed:
    seed_id: str
    source_file: str
    code: str
    name: str
    params: dict[str, Any]
    cells_rle: str


@dataclass(frozen=True)
class RolloutConfig:
    image_size: int = 64
    warmup_steps: int = 32
    record_steps: int = 256
    target_radius: int = 12
    num_seeds: int = 200
    root_seed: int = 20260306
    weak_perturb_ratio: float = 0.25
    local_noise_sigma: float = 0.02
    global_noise_sigma: float = 0.01
    local_patch_size: int = 5
    mu_min: float = 0.23
    mu_max: float = 0.41
    sigma_min: float = 0.033
    sigma_max: float = 0.080
    center_mu_min: float = 0.27
    center_mu_max: float = 0.38
    center_sigma_min: float = 0.039
    center_sigma_max: float = 0.067
    center_sampling_ratio: float = 0.7
    max_attempts_per_seed: int = 12


def parse_band_list(value: str) -> list[float]:
    out: list[float] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "/" in chunk:
            num, den = chunk.split("/", 1)
            out.append(float(num) / float(den))
        else:
            out.append(float(chunk))
    return out or [1.0]


def _token_to_value(token: str) -> float:
    if token in VALUE_CHARS:
        return float(VALUE_CHARS[token]) / 255.0
    if len(token) == 1:
        return float(ord(token) - ord("A") + 1) / 255.0
    return float((ord(token[0]) - ord("p")) * 24 + (ord(token[1]) - ord("A") + 25)) / 255.0


def rle2arr_2d(st: str) -> np.ndarray:
    rows: list[list[float]] = [[]]
    i = 0
    text = st.strip()
    while i < len(text):
        count_str = ""
        while i < len(text) and text[i].isdigit():
            count_str += text[i]
            i += 1
        count = int(count_str) if count_str else 1
        if i >= len(text):
            break
        ch = text[i]
        if ch == "!":
            break
        if ch == "$":
            rows.extend([] for _ in range(count))
            i += 1
            continue
        if ch in VALUE_CHARS or ("A" <= ch <= "X"):
            token = ch
            i += 1
        else:
            token = text[i : i + 2]
            i += 2
        rows[-1].extend([_token_to_value(token)] * count)
    width = max((len(row) for row in rows), default=0)
    out = np.zeros((len(rows), width), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        if row:
            out[row_idx, : len(row)] = np.asarray(row, dtype=np.float32)
    return out


def resize_nearest(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    if arr.shape == (new_h, new_w):
        return arr.astype(np.float32, copy=True)
    yy = np.linspace(0, arr.shape[0] - 1, new_h).astype(np.int32)
    xx = np.linspace(0, arr.shape[1] - 1, new_w).astype(np.int32)
    return arr[np.ix_(yy, xx)].astype(np.float32, copy=False)


def center_seed_on_canvas(arr: np.ndarray, canvas_size: int) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    target = canvas_size // 2
    scale = min(target / max(arr.shape[0], 1), target / max(arr.shape[1], 1), 1.0)
    resized_h = max(1, int(round(arr.shape[0] * scale)))
    resized_w = max(1, int(round(arr.shape[1] * scale)))
    resized = resize_nearest(arr, resized_h, resized_w)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    top = (canvas_size - resized_h) // 2
    left = (canvas_size - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def load_seed_catalog(path: str | Path) -> list[LeniaSeed]:
    rows = load_json(path)
    seeds: list[LeniaSeed] = []
    for idx, row in enumerate(rows):
        params = dict(row.get("params", {}))
        if not params or "cells" not in row:
            continue
        seeds.append(
            LeniaSeed(
                seed_id=f"seed_{idx:06d}",
                source_file=row.get("source_file", ""),
                code=row.get("code", ""),
                name=row.get("name", ""),
                params=params,
                cells_rle=row["cells"],
            )
        )
    return seeds


def sample_params(
    rng: np.random.Generator, config: RolloutConfig, seed_params: dict[str, Any]
) -> dict[str, Any]:
    center_sample = rng.random() < config.center_sampling_ratio
    if center_sample:
        m = float(rng.uniform(config.center_mu_min, config.center_mu_max))
        s = float(rng.uniform(config.center_sigma_min, config.center_sigma_max))
    else:
        m = float(rng.uniform(config.mu_min, config.mu_max))
        s = float(rng.uniform(config.sigma_min, config.sigma_max))
    params = {
        "R": config.target_radius,
        "T": int(seed_params.get("T", 10)),
        "b": parse_band_list(seed_params.get("b", "1")),
        "m": m,
        "s": s,
        "kn": int(seed_params.get("kn", 1)),
        "gn": int(seed_params.get("gn", 1)),
        "source_R": int(seed_params.get("R", config.target_radius)),
    }
    return params


def build_kernel(size: int, radius: int, bands: list[float]) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float32)
    cy = size // 2
    cx = size // 2
    dy = yy - cy
    dx = xx - cx
    dist = np.sqrt(dx * dx + dy * dy) / max(float(radius), 1.0)
    kernel = np.zeros((size, size), dtype=np.float32)
    band_count = max(1, len(bands))
    inside = dist < 1.0
    band_pos = dist[inside] * band_count
    band_idx = np.clip(np.floor(band_pos).astype(np.int32), 0, band_count - 1)
    band_frac = band_pos - np.floor(band_pos)
    shell = np.exp(-((band_frac - 0.5) ** 2) / (2.0 * 0.15**2))
    kernel[inside] = shell * np.asarray(bands, dtype=np.float32)[band_idx]
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    return np.fft.ifftshift(kernel).astype(np.float32)


def lenia_step(state: np.ndarray, kernel_fft: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    potential = np.fft.ifft2(np.fft.fft2(state) * kernel_fft).real.astype(np.float32)
    m = float(params["m"])
    s = max(float(params["s"]), 1e-4)
    growth = np.exp(-((potential - m) ** 2) / (2.0 * s * s)).astype(np.float32) * 2.0 - 1.0
    dt = 1.0 / max(float(params["T"]), 1.0)
    updated = np.clip(state + dt * growth, 0.0, 1.0)
    return updated.astype(np.float32)


def maybe_apply_weak_perturbation(
    state: np.ndarray,
    rng: np.random.Generator,
    config: RolloutConfig,
    perturb_mode: str | None,
    perturb_step: int | None,
    current_step: int,
) -> np.ndarray:
    if perturb_mode is None or perturb_step is None or current_step != perturb_step:
        return state
    out = state.copy()
    if perturb_mode == "global":
        out += gaussian_noise(rng, out.shape, config.global_noise_sigma)
    else:
        size = config.local_patch_size
        y0 = int(rng.integers(0, max(1, out.shape[0] - size + 1)))
        x0 = int(rng.integers(0, max(1, out.shape[1] - size + 1)))
        out[y0 : y0 + size, x0 : x0 + size] += gaussian_noise(
            rng, (size, size), config.local_noise_sigma
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_dna_channel(shape: tuple[int, int], params: dict[str, Any]) -> np.ndarray:
    m_norm = (float(params["m"]) - 0.23) / max(0.41 - 0.23, 1e-8)
    s_norm = (float(params["s"]) - 0.033) / max(0.080 - 0.033, 1e-8)
    value = float(np.clip(0.5 * m_norm + 0.5 * s_norm, 0.0, 1.0))
    return np.full(shape, value, dtype=np.float32)


def derive_multichannel_state(
    prev_state: np.ndarray, current_state: np.ndarray, params: dict[str, Any]
) -> np.ndarray:
    grad_y, grad_x = np.gradient(current_state)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)
    membrane = robust_percentile_range(grad * (current_state > 0.02))
    cy, cx = center_of_mass(current_state)
    yy, xx = np.indices(current_state.shape, dtype=np.float32)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist /= max(float(dist.max()), 1e-8)
    nucleus_weight = np.exp(-(dist**2) / (2.0 * 0.18**2)).astype(np.float32)
    nucleus = robust_percentile_range(current_state * nucleus_weight)
    cytoplasm = robust_percentile_range(current_state * (1.0 - 0.7 * nucleus_weight))
    delta = current_state - prev_state
    rna = robust_percentile_range(np.clip(delta, 0.0, None))
    dna = make_dna_channel(current_state.shape, params)
    state = np.stack([membrane, cytoplasm, nucleus, dna, rna], axis=-1)
    return state.astype(np.float32)


def sample_episode(
    seed: LeniaSeed,
    config: RolloutConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]] | None:
    scalar_seed = center_seed_on_canvas(rle2arr_2d(seed.cells_rle), config.image_size)
    for _ in range(config.max_attempts_per_seed):
        params = sample_params(rng, config, seed.params)
        kernel = build_kernel(config.image_size, config.target_radius, params["b"])
        kernel_fft = np.fft.fft2(kernel)
        perturb_low = config.warmup_steps + 4
        perturb_high = config.warmup_steps + config.record_steps - 4
        can_schedule_perturb = perturb_low < perturb_high
        use_perturb = bool(rng.random() < config.weak_perturb_ratio and can_schedule_perturb)
        perturb_mode = str(rng.choice(["local", "global"])) if use_perturb else None
        perturb_step = (
            int(rng.integers(perturb_low, perturb_high))
            if use_perturb
            else None
        )
        current = scalar_seed.copy()
        prev = current.copy()
        scalar_frames: list[np.ndarray] = []
        multi_frames: list[np.ndarray] = []
        total_steps = config.warmup_steps + config.record_steps + 1
        for step in range(total_steps):
            current = maybe_apply_weak_perturbation(
                current, rng, config, perturb_mode, perturb_step, step
            )
            next_state = lenia_step(current, kernel_fft, params)
            if step >= config.warmup_steps:
                scalar_frames.append(current.astype(np.float32))
                multi_frames.append(derive_multichannel_state(prev, current, params))
            prev = current
            current = next_state
        scalar_array = np.stack(scalar_frames, axis=0)
        multi_array = np.stack(multi_frames, axis=0)
        summary = activity_summary(scalar_array[-1])
        if reject_scalar_episode(summary):
            continue
        regime, regime_stats = classify_regime_from_scalar_states(scalar_array)
        meta = {
            "seed_id": seed.seed_id,
            "source_file": seed.source_file,
            "source_code": seed.code,
            "source_name": seed.name,
            "lenia_params": {
                "R": config.target_radius,
                "T": params["T"],
                "b": [float(v) for v in params["b"]],
                "m": float(params["m"]),
                "s": float(params["s"]),
                "kn": params["kn"],
                "gn": params["gn"],
                "source_R": params["source_R"],
            },
            "perturb_mode": perturb_mode,
            "perturb_step": perturb_step,
            "summary": summary,
            "regime": regime,
            "regime_stats": regime_stats,
        }
        return scalar_array, multi_array, params, meta
    return None


def generate_rollouts(
    output_root: str | Path,
    seed_catalog_path: str | Path,
    config: RolloutConfig,
) -> Path:
    output_root = ensure_dir(output_root)
    episodes_dir = ensure_dir(output_root / "episodes")
    seeds = load_seed_catalog(seed_catalog_path)
    if not seeds:
        raise SystemExit(f"no seeds found in {seed_catalog_path}")
    rng = np.random.default_rng(config.root_seed)
    selected_indices = rng.choice(len(seeds), size=config.num_seeds, replace=False)
    manifest: list[dict[str, Any]] = []
    for order_idx, seed_idx in enumerate(selected_indices.tolist()):
        seed = seeds[seed_idx]
        episode = sample_episode(seed, config, rng)
        if episode is None:
            continue
        scalar_frames, multi_frames, params, meta = episode
        split = choose_split(order_idx, len(selected_indices))
        episode_id = f"{seed.seed_id}_ep00"
        path = episodes_dir / f"{episode_id}.npz"
        np.savez_compressed(
            path,
            scalar_states=scalar_frames.astype(np.float32),
            multi_states=multi_frames.astype(np.float32),
        )
        manifest.append(
            {
                "episode_id": episode_id,
                "seed_id": seed.seed_id,
                "split": split,
                "path": str(path),
                "num_frames": int(multi_frames.shape[0]),
                "num_pairs": int(max(0, multi_frames.shape[0] - 1)),
                "lenia_params": meta["lenia_params"],
                "perturb_mode": meta["perturb_mode"],
                "perturb_step": meta["perturb_step"],
                "source_file": meta["source_file"],
                "source_code": meta["source_code"],
                "source_name": meta["source_name"],
                "summary": meta["summary"],
                "regime": meta["regime"],
                "regime_stats": meta["regime_stats"],
            }
        )
    manifest_path = output_root / "manifest.jsonl"
    save_jsonl(manifest_path, manifest)
    save_json(
        output_root / "summary.json",
        {
            "num_selected_seeds": int(config.num_seeds),
            "num_successful_episodes": int(len(manifest)),
            "image_size": config.image_size,
            "warmup_steps": config.warmup_steps,
            "record_steps": config.record_steps,
            "target_radius": config.target_radius,
            "root_seed": config.root_seed,
        },
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lenia rollouts for TRM training.")
    parser.add_argument(
        "--seed-catalog",
        default="data/lenia_official/animals2d_seeds.json",
        help="Path to exported Lenia seed catalog.",
    )
    parser.add_argument(
        "--output-root",
        default="data/trm_rollouts",
        help="Directory to write rollout episodes and manifest.",
    )
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--record-steps", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--target-radius", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260306)
    args = parser.parse_args()
    seed_everything(args.seed)
    config = RolloutConfig(
        image_size=args.image_size,
        warmup_steps=args.warmup_steps,
        record_steps=args.record_steps,
        target_radius=args.target_radius,
        num_seeds=args.num_seeds,
        root_seed=args.seed,
    )
    manifest_path = generate_rollouts(args.output_root, args.seed_catalog, config)
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
