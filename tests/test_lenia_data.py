from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trm_pipeline.lenia_data import (
    RolloutConfig,
    build_kernel,
    center_seed_on_canvas,
    derive_multichannel_state,
    generate_rollouts,
    LeniaSeed,
    load_seed_catalog,
    maybe_apply_weak_perturbation,
    parse_band_list,
    rle2arr_2d,
    sample_episode,
    sample_params,
)


def _seed_catalog_rows(count: int = 3) -> list[dict]:
    return [
        {
            "source_file": f"seed_{idx}.json",
            "code": f"seed_{idx}",
            "name": f"seed-{idx}",
            "params": {"R": 12, "T": 10, "b": "1", "kn": 1, "gn": 1},
            "cells": "12o$12o$12o$12o$12o$12o!",
        }
        for idx in range(count)
    ]


def test_parse_band_list_supports_fraction_and_csv() -> None:
    assert parse_band_list("1/2, 1, 3/2") == [0.5, 1.0, 1.5]
    assert parse_band_list("") == [1.0]


def test_rle2arr_2d_decodes_simple_pattern() -> None:
    arr = rle2arr_2d("2o$bo!")
    assert arr.shape == (2, 2)
    assert arr[0, 0] == 1.0
    assert arr[0, 1] == 1.0
    assert arr[1, 1] == 1.0


def test_center_seed_on_canvas_places_mass_near_center() -> None:
    seed = np.ones((4, 6), dtype=np.float32)
    canvas = center_seed_on_canvas(seed, canvas_size=16)
    yy, xx = np.indices(canvas.shape, dtype=np.float32)
    cy = float((yy * canvas).sum() / canvas.sum())
    cx = float((xx * canvas).sum() / canvas.sum())
    assert canvas.shape == (16, 16)
    assert 6.0 <= cy <= 10.0
    assert 6.0 <= cx <= 10.0


def test_load_seed_catalog_filters_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    rows = [
        {"params": {"R": 12, "T": 10, "b": "1"}, "cells": "o!", "name": "ok"},
        {"params": {}, "cells": "o!", "name": "bad"},
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    seeds = load_seed_catalog(path)
    assert len(seeds) == 1
    assert seeds[0].name == "ok"


def test_sample_params_stays_within_config_ranges() -> None:
    rng = np.random.default_rng(0)
    config = RolloutConfig()
    params = sample_params(rng, config, {"T": 8, "b": "1/2,1"})
    assert config.mu_min <= params["m"] <= config.mu_max
    assert config.sigma_min <= params["s"] <= config.sigma_max
    assert params["R"] == config.target_radius
    assert params["T"] == 8


def test_build_kernel_is_normalized() -> None:
    kernel = build_kernel(32, 8, [1.0, 0.5])
    assert kernel.shape == (32, 32)
    assert np.isclose(float(kernel.sum()), 1.0, atol=1e-5)


def test_maybe_apply_weak_perturbation_only_acts_at_target_step() -> None:
    rng = np.random.default_rng(0)
    config = RolloutConfig(local_patch_size=3, local_noise_sigma=0.1, global_noise_sigma=0.1)
    state = np.zeros((8, 8), dtype=np.float32)
    unchanged = maybe_apply_weak_perturbation(state, rng, config, "local", 4, 3)
    changed = maybe_apply_weak_perturbation(state, rng, config, "local", 4, 4)
    assert np.allclose(unchanged, state)
    assert not np.allclose(changed, state)


def test_derive_multichannel_state_has_expected_shape_and_range() -> None:
    prev = np.zeros((16, 16), dtype=np.float32)
    current = np.zeros((16, 16), dtype=np.float32)
    current[5:11, 5:11] = 1.0
    state = derive_multichannel_state(prev, current, {"m": 0.3, "s": 0.05})
    assert state.shape == (16, 16, 5)
    assert np.all((state >= 0.0) & (state <= 1.0))


def test_sample_episode_returns_scalar_multi_and_meta() -> None:
    rng = np.random.default_rng(0)
    seed = LeniaSeed(
        seed_id="seed_000000",
        source_file="seed.json",
        code="seed",
        name="seed",
        params={"R": 12, "T": 10, "b": "1", "kn": 1, "gn": 1},
        cells_rle="12o$12o$12o$12o$12o$12o!",
    )
    config = RolloutConfig(image_size=32, warmup_steps=16, record_steps=32, max_attempts_per_seed=8)
    episode = sample_episode(seed, config, rng)
    assert episode is not None
    scalar_frames, multi_frames, params, meta = episode
    assert scalar_frames.shape == (config.record_steps + 1, 32, 32)
    assert multi_frames.shape == (config.record_steps + 1, 32, 32, 5)
    assert params["R"] == config.target_radius
    assert meta["seed_id"] == seed.seed_id
    assert meta["regime"] in {"stable", "chaotic"}


def test_generate_rollouts_writes_manifest_and_summary(tmp_path: Path) -> None:
    seed_catalog_path = tmp_path / "seeds.json"
    seed_catalog_path.write_text(json.dumps(_seed_catalog_rows(3), ensure_ascii=False), encoding="utf-8")
    config = RolloutConfig(
        image_size=32,
        warmup_steps=16,
        record_steps=32,
        num_seeds=3,
        root_seed=123,
        max_attempts_per_seed=8,
    )

    manifest_path = generate_rollouts(tmp_path / "rollouts", seed_catalog_path, config)

    assert manifest_path.exists()
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 1
    summary_path = manifest_path.parent / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["num_selected_seeds"] == 3
    assert summary["num_successful_episodes"] == len(rows)
    assert "regime_counts" in summary
    assert "perturb_counts" in summary
    assert "mu_range" in summary
    first_path = Path(rows[0]["path"])
    assert first_path.exists()
