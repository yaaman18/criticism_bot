from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any
import shutil

import numpy as np

from .common import choose_split, ensure_dir, save_json, save_jsonl, seed_everything
from .erie_runtime import (
    ACTIONS,
    ERIERuntime,
    EnvironmentConfig,
    LeniaERIEEnvironment,
    RuntimeConfig,
    RuntimeModels,
    _policy_action_cost,
    _softmax,
)
from .lenia_data import load_seed_catalog


def _collect_episode_samples(runtime: ERIERuntime) -> dict[str, np.ndarray]:
    vm_viability_state: list[np.ndarray] = []
    vm_contact_state: list[np.ndarray] = []
    vm_action_cost: list[np.ndarray] = []
    vm_target_state: list[np.ndarray] = []
    vm_target_homeostatic_error: list[np.ndarray] = []
    vm_target_risk: list[np.ndarray] = []

    as_viability_state: list[np.ndarray] = []
    as_action_scores: list[np.ndarray] = []
    as_uncertainty_state: list[np.ndarray] = []
    as_target_logits: list[np.ndarray] = []
    as_target_policy: list[np.ndarray] = []
    as_target_action: list[np.ndarray] = []

    for t in range(runtime.cfg.steps):
        runtime.env.step_lenia()
        observation, sensor_gate, _, boundary = runtime._observe()
        boundary_obs = np.stack([boundary, runtime._body_fields()[2]], axis=-1).astype(np.float32)
        runtime._belief_update(observation, sensor_gate, boundary_obs)

        scores, _ = runtime._policy_scores()
        viability_monitor = runtime._monitor_viability(action_cost=0.0)
        policy = _softmax((-runtime.cfg.beta_pi * scores).astype(np.float32))
        policy_probs = policy.astype(np.float64)
        policy_probs = policy_probs / max(float(policy_probs.sum()), 1e-12)
        action_index = int(runtime.rng.choice(len(ACTIONS), p=policy_probs))
        action = ACTIONS[action_index]
        action_cost = _policy_action_cost(action)
        contact = runtime._contact_stats(runtime.body)
        uncertainty_state = runtime._uncertainty_state()

        next_body = runtime._prospective_body(action)
        next_G, next_B = runtime._predicted_viability(next_body, action)
        target_state = np.array([next_G, next_B], dtype=np.float32)
        target_error = np.abs(
            target_state - np.array([runtime.cfg.G_target, runtime.cfg.B_target], dtype=np.float32)
        ).astype(np.float32)
        target_risk = np.array(
            [[float(next_G < runtime.cfg.tau_G or next_B < runtime.cfg.tau_B)]],
            dtype=np.float32,
        )

        vm_viability_state.append(np.array([runtime.body.G, runtime.body.B], dtype=np.float32))
        vm_contact_state.append(
            np.array([contact["resource"], contact["hazard"], contact["shelter"]], dtype=np.float32)
        )
        vm_action_cost.append(np.array([action_cost], dtype=np.float32))
        vm_target_state.append(target_state)
        vm_target_homeostatic_error.append(target_error)
        vm_target_risk.append(target_risk[0])

        as_viability_state.append(viability_monitor["state"].astype(np.float32))
        as_action_scores.append(scores.astype(np.float32))
        as_uncertainty_state.append(uncertainty_state.astype(np.float32))
        target_logits = (-runtime.cfg.beta_pi * scores).astype(np.float32)
        target_logits = target_logits - float(np.mean(target_logits))
        as_target_logits.append(target_logits.astype(np.float32))
        as_target_policy.append(policy.astype(np.float32))
        as_target_action.append(np.array(action_index, dtype=np.int64))

        runtime._apply_action(action)
        runtime.env.update_fields(runtime.body, action)
        if runtime._update_death():
            break

    return {
        "vm_viability_state": np.stack(vm_viability_state, axis=0).astype(np.float32),
        "vm_contact_state": np.stack(vm_contact_state, axis=0).astype(np.float32),
        "vm_action_cost": np.stack(vm_action_cost, axis=0).astype(np.float32),
        "vm_target_state": np.stack(vm_target_state, axis=0).astype(np.float32),
        "vm_target_homeostatic_error": np.stack(vm_target_homeostatic_error, axis=0).astype(np.float32),
        "vm_target_risk": np.stack(vm_target_risk, axis=0).astype(np.float32),
        "as_viability_state": np.stack(as_viability_state, axis=0).astype(np.float32),
        "as_action_scores": np.stack(as_action_scores, axis=0).astype(np.float32),
        "as_uncertainty_state": np.stack(as_uncertainty_state, axis=0).astype(np.float32),
        "as_target_logits": np.stack(as_target_logits, axis=0).astype(np.float32),
        "as_target_policy": np.stack(as_target_policy, axis=0).astype(np.float32),
        "as_target_action": np.asarray(as_target_action, dtype=np.int64),
    }


def prepare_trm_va_cache(
    seed_catalog: str | Path,
    output_root: str | Path,
    runtime_config: RuntimeConfig,
    env_config: EnvironmentConfig,
    num_episodes: int = 16,
) -> Path:
    seed_everything(runtime_config.seed)
    output_root = ensure_dir(output_root)
    episode_dir = output_root / "episodes"
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    episode_dir = ensure_dir(episode_dir)
    seeds = load_seed_catalog(seed_catalog)
    if not seeds:
        raise SystemExit(f"no seeds found in {seed_catalog}")

    manifest_rows: list[dict[str, Any]] = []
    for episode_index in range(num_episodes):
        episode_seed = int(runtime_config.seed + episode_index)
        rng = np.random.default_rng(episode_seed)
        seed = seeds[int(rng.integers(0, len(seeds)))]
        env = LeniaERIEEnvironment(seed, env_config, runtime_config, rng)
        runtime = ERIERuntime(env, runtime_config, rng, models=RuntimeModels(None, None))
        arrays = _collect_episode_samples(runtime)
        episode_id = f"va_{episode_seed}_{seed.seed_id}"
        path = episode_dir / f"{episode_id}.npz"
        np.savez_compressed(path, **arrays)
        manifest_rows.append(
            {
                "episode_id": episode_id,
                "split": choose_split(episode_index, num_episodes),
                "path": str(path),
                "num_samples": int(arrays["vm_viability_state"].shape[0]),
                "seed_id": seed.seed_id,
            }
        )

    manifest_path = output_root / "manifest.jsonl"
    save_jsonl(manifest_path, manifest_rows)
    save_json(
        output_root / "summary.json",
        {
            "seed_catalog": str(seed_catalog),
            "num_episodes": int(num_episodes),
            "runtime_config": asdict(runtime_config),
            "environment_config": asdict(env_config),
        },
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bootstrap training cache for TRM-Vm and TRM-As.")
    parser.add_argument("--seed-catalog", default="data/lenia_official/animals2d_seeds.json")
    parser.add_argument("--output-root", default="data/trm_va_cache")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260318)
    args = parser.parse_args()
    manifest = prepare_trm_va_cache(
        seed_catalog=args.seed_catalog,
        output_root=args.output_root,
        runtime_config=RuntimeConfig(steps=args.steps, warmup_steps=args.warmup_steps, seed=args.seed),
        env_config=EnvironmentConfig(),
        num_episodes=args.episodes,
    )
    print(f"wrote TRM-VA cache manifest: {manifest}")


if __name__ == "__main__":
    main()
