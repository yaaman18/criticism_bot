from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import trm_pipeline.erie_runtime as erie_runtime_module
from trm_pipeline.erie_runtime import (
    ACTIONS,
    BodyState,
    ERIERuntime,
    EnvironmentConfig,
    LeniaERIEEnvironment,
    RuntimeModels,
    RuntimeConfig,
    _body_fields,
    _risk_proxy,
    _softmax,
    run_episode,
)
from trm_pipeline.lenia_data import LeniaSeed
from trm_pipeline.models import TRMModelConfig, build_trm_a, build_trm_b, require_torch


def _seed_catalog(tmp_path: Path) -> Path:
    catalog_path = tmp_path / "seed_catalog.json"
    rows = [
        {
            "source_file": "test_seed.json",
            "code": "test",
            "name": "unit-seed",
            "params": {"R": 12, "T": 10, "b": "1", "kn": 1, "gn": 1},
            "cells": "3o$3o$3o!",
        }
    ]
    catalog_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return catalog_path


def _runtime(seed: int = 1234) -> ERIERuntime:
    env_cfg = EnvironmentConfig(image_size=32, target_radius=8)
    run_cfg = RuntimeConfig(
        steps=8,
        warmup_steps=1,
        seed=seed,
        occupancy_radius=5.0,
        move_step=1.5,
        policy_mode="closed_loop",
    )
    rng = np.random.default_rng(seed)
    seed_row = LeniaSeed(
        seed_id="seed_000000",
        source_file="test_seed.json",
        code="test",
        name="unit-seed",
        params={"R": 12, "T": 10, "b": "1", "kn": 1, "gn": 1},
        cells_rle="3o$3o$3o!",
    )
    env = LeniaERIEEnvironment(seed_row, env_cfg, run_cfg, rng)
    return ERIERuntime(env, run_cfg, rng)


def _configure_controlled_resource_corridor(runtime: ERIERuntime) -> ERIERuntime:
    runtime.cfg = RuntimeConfig(
        **{
            **runtime.cfg.__dict__,
            "move_step": 3.0,
            "epistemic_scale": 0.0,
            "observation_noise": 0.0,
            "risk_wG": 3.0,
            "risk_wB": 1.0,
            "ambiguity_w_boundary": 0.0,
        }
    )
    runtime.body.G = 0.18
    runtime.body.B = 0.85
    runtime.body.centroid_x = 8.0
    runtime.body.centroid_y = 16.0
    yy, xx = np.indices((32, 32), dtype=np.float32)
    dist2 = (yy - 16.0) ** 2 + (xx - 18.0) ** 2
    runtime.env.resource = np.exp(-dist2 / (2.0 * 5.0**2)).astype(np.float32)
    runtime.env.resource /= runtime.env.resource.max()
    runtime.env.hazard.fill(0.0)
    runtime.env.shelter.fill(0.0)
    runtime.env.step_lenia = lambda: None
    runtime.env.environment_channels = lambda: np.concatenate(
        [
            np.zeros((32, 32, 5), dtype=np.float32),
            np.stack([runtime.env.resource, runtime.env.hazard, runtime.env.shelter], axis=-1),
        ],
        axis=-1,
    )
    return runtime


def _configure_controlled_hazard_escape(runtime: ERIERuntime) -> ERIERuntime:
    runtime.cfg = RuntimeConfig(
        **{
            **runtime.cfg.__dict__,
            "move_step": 3.0,
            "epistemic_scale": 0.0,
            "observation_noise": 0.0,
            "risk_wG": 1.0,
            "risk_wB": 3.5,
            "ambiguity_w_boundary": 0.0,
        }
    )
    runtime.body.G = 0.70
    runtime.body.B = 0.28
    runtime.body.centroid_x = 18.0
    runtime.body.centroid_y = 16.0
    yy, xx = np.indices((32, 32), dtype=np.float32)
    hazard_dist2 = (yy - 16.0) ** 2 + (xx - 20.0) ** 2
    shelter_dist2 = (yy - 16.0) ** 2 + (xx - 8.0) ** 2
    runtime.env.resource.fill(0.0)
    runtime.env.hazard = np.exp(-hazard_dist2 / (2.0 * 4.5**2)).astype(np.float32)
    runtime.env.hazard /= runtime.env.hazard.max()
    runtime.env.shelter = np.exp(-shelter_dist2 / (2.0 * 5.0**2)).astype(np.float32)
    runtime.env.shelter /= runtime.env.shelter.max()
    runtime.env.step_lenia = lambda: None
    runtime.env.environment_channels = lambda: np.concatenate(
        [
            np.zeros((32, 32, 5), dtype=np.float32),
            np.stack([runtime.env.resource, runtime.env.hazard, runtime.env.shelter], axis=-1),
        ],
        axis=-1,
    )
    return runtime


def _checkpoint_paths(tmp_path: Path) -> tuple[Path, Path]:
    torch, _, _ = require_torch()
    config = TRMModelConfig(
        image_size=32,
        patch_size=8,
        dim=32,
        recursions=2,
        num_heads=4,
        mlp_ratio=2,
        in_channels=5,
        z_dim=8,
    )
    trm_a = build_trm_a(config)
    trm_b = build_trm_b(config)
    trm_a_path = tmp_path / "trm_a_test.pt"
    trm_b_path = tmp_path / "trm_b_test.pt"
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state": trm_a.state_dict(),
        },
        trm_a_path,
    )
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state": trm_b.state_dict(),
        },
        trm_b_path,
    )
    return trm_a_path, trm_b_path


def test_softmax_is_normalized() -> None:
    probs = _softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert probs.shape == (3,)
    assert np.isclose(float(probs.sum()), 1.0)
    assert np.all(probs > 0.0)


def test_body_fields_have_expected_ranges() -> None:
    body = BodyState(
        centroid_y=16.0,
        centroid_x=16.0,
        radius=5.0,
        aperture_angle=0.0,
        aperture_gain=0.5,
        aperture_width_deg=70.0,
        G=0.7,
        B=0.8,
    )
    occupancy, boundary, permeability = _body_fields(body, image_size=32, softness=1.2)
    assert occupancy.shape == (32, 32)
    assert boundary.shape == (32, 32)
    assert permeability.shape == (32, 32)
    assert np.all((occupancy >= 0.0) & (occupancy <= 1.0))
    assert np.all((boundary >= 0.0) & (boundary <= 1.0))
    assert np.all((permeability >= 0.0) & (permeability <= 1.0))
    assert float(boundary.max()) > 0.9
    assert float(occupancy[16, 16]) > 0.9


def test_predicted_viability_is_clipped_and_action_sensitive() -> None:
    runtime = _runtime()
    runtime.body.G = 0.2
    runtime.body.B = 0.25
    g_intake, b_intake = runtime._predicted_viability(runtime._prospective_body("intake"), "intake")
    g_seal, b_seal = runtime._predicted_viability(runtime._prospective_body("seal"), "seal")
    assert 0.0 <= g_intake <= 1.0
    assert 0.0 <= b_intake <= 1.0
    assert 0.0 <= g_seal <= 1.0
    assert 0.0 <= b_seal <= 1.0
    assert b_seal >= b_intake


def test_update_death_requires_k_irrev_consecutive_steps() -> None:
    runtime = _runtime()
    runtime.body.G = runtime.cfg.tau_G - 0.01
    runtime.body.B = runtime.cfg.tau_B - 0.01
    for _ in range(runtime.cfg.k_irrev - 1):
        assert runtime._update_death() is False
    assert runtime._update_death() is True


def test_step_respects_no_action_policy_mode() -> None:
    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "policy_mode": "no_action"})
    dead = runtime.step(0)
    assert dead is False
    assert runtime.history[-1]["action"] == "no_action"


def test_step_respects_random_policy_mode() -> None:
    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "policy_mode": "random"})
    runtime.step(0)
    assert runtime.history[-1]["action"] in ACTIONS


def test_observe_blends_environment_through_boundary_interface(monkeypatch) -> None:
    runtime = _runtime()
    runtime.world_belief.fill(0.0)
    runtime.env.environment_channels = lambda: np.ones((32, 32, 8), dtype=np.float32)
    monkeypatch.setattr(
        erie_runtime_module,
        "gaussian_noise",
        lambda rng, shape, sigma: np.zeros(shape, dtype=np.float32),
    )

    observation, sensor_gate, occupancy, boundary = runtime._observe()

    assert observation.shape == (32, 32, 8)
    assert sensor_gate.shape == (32, 32, 1)
    assert float(sensor_gate.max()) > float(sensor_gate.min())
    assert np.allclose(observation, np.broadcast_to(sensor_gate, observation.shape), atol=1e-6)
    assert float(boundary.max()) > float(occupancy.min())


def test_belief_update_matches_precision_weighted_error_rule() -> None:
    runtime = _runtime()
    runtime.world_belief.fill(0.0)
    runtime.world_logvar.fill(0.0)
    runtime.boundary_belief.fill(0.0)
    runtime.boundary_logvar.fill(0.0)
    observation = np.ones_like(runtime.world_belief, dtype=np.float32)
    sensor_gate = np.ones((*runtime.world_belief.shape[:2], 1), dtype=np.float32)
    boundary_obs = np.ones_like(runtime.boundary_belief, dtype=np.float32)

    runtime._belief_update(observation, sensor_gate, boundary_obs)

    assert np.allclose(runtime.world_belief, runtime.cfg.lambda_w, atol=1e-6)
    assert np.allclose(runtime.boundary_belief, runtime.cfg.lambda_b, atol=1e-6)
    assert np.allclose(runtime.world_logvar, runtime.cfg.world_logvar_drift - 0.18, atol=1e-6)
    assert np.allclose(runtime.boundary_logvar, runtime.cfg.boundary_logvar_drift - 0.20, atol=1e-6)


def test_lower_logvar_produces_larger_belief_update() -> None:
    low_unc = _runtime(seed=2001)
    high_unc = _runtime(seed=2001)
    observation = np.ones_like(low_unc.world_belief, dtype=np.float32)
    sensor_gate = np.ones((*low_unc.world_belief.shape[:2], 1), dtype=np.float32)
    boundary_obs = np.zeros_like(low_unc.boundary_belief, dtype=np.float32)

    low_unc.world_belief.fill(0.0)
    low_unc.world_logvar.fill(-1.0)
    high_unc.world_belief.fill(0.0)
    high_unc.world_logvar.fill(1.0)

    low_unc._belief_update(observation, sensor_gate, boundary_obs)
    high_unc._belief_update(observation, sensor_gate, boundary_obs)

    assert float(low_unc.world_belief.mean()) > float(high_unc.world_belief.mean())


def test_ambiguity_proxy_increases_with_logvar() -> None:
    runtime = _runtime()
    runtime.world_logvar.fill(-2.0)
    runtime.boundary_logvar.fill(-2.0)
    low = runtime._ambiguity_proxy(runtime.body)

    runtime.world_logvar.fill(2.0)
    runtime.boundary_logvar.fill(2.0)
    high = runtime._ambiguity_proxy(runtime.body)

    assert high > low


def test_policy_scores_prefer_resource_seeking_under_low_hazard() -> None:
    runtime = _runtime()
    runtime.cfg = RuntimeConfig(
        **{
            **runtime.cfg.__dict__,
            "epistemic_scale": 0.0,
            "risk_wG": 3.0,
            "risk_wB": 1.0,
            "ambiguity_w_boundary": 0.0,
        }
    )
    runtime.body.G = 0.20
    runtime.body.B = 0.80
    x_grad = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))
    runtime.env.resource = x_grad
    runtime.env.hazard = 0.1 * x_grad
    runtime.env.shelter.fill(0.0)
    runtime.world_logvar.fill(0.0)
    runtime.boundary_logvar.fill(0.0)

    scores, diagnostics = runtime._policy_scores()
    score_map = {action: float(scores[idx]) for idx, action in enumerate(ACTIONS)}

    assert diagnostics["approach"]["pred_G"] > diagnostics["withdraw"]["pred_G"]
    assert diagnostics["approach"]["risk"] < diagnostics["withdraw"]["risk"]
    assert score_map["approach"] < score_map["withdraw"]


def test_policy_scores_prefer_withdraw_under_high_hazard_and_low_boundary() -> None:
    runtime = _configure_controlled_hazard_escape(_runtime(seed=3003))

    scores, diagnostics = runtime._policy_scores()
    score_map = {action: float(scores[idx]) for idx, action in enumerate(ACTIONS)}

    best_action = min(score_map, key=score_map.get)
    assert diagnostics["reconfigure"]["risk"] < diagnostics["approach"]["risk"]
    assert diagnostics["seal"]["risk"] < diagnostics["approach"]["risk"]
    assert best_action in {"withdraw", "seal", "reconfigure"}
    assert score_map[best_action] < score_map["approach"]


def test_risk_proxy_prefers_homeostatic_band_over_safe_overshoot() -> None:
    cfg = RuntimeConfig()

    near_target = _risk_proxy(0.56, 0.66, 0.0, cfg)
    overshoot = _risk_proxy(0.92, 0.95, 0.0, cfg)

    assert near_target < overshoot


def test_risk_proxy_penalizes_low_viability_margin_before_terminal_failure() -> None:
    cfg = RuntimeConfig()

    healthy_margin = _risk_proxy(0.40, 0.70, 0.0, cfg)
    near_dead_margin = _risk_proxy(cfg.tau_G + 0.01, 0.70, 0.0, cfg)

    assert near_dead_margin > healthy_margin


def test_assemble_world_prior_uses_trm_a_next_state_cache() -> None:
    runtime = _runtime()
    runtime.models = SimpleNamespace(trm_a=object(), trm_b=None, torch=None)
    runtime.world_belief.fill(0.0)
    runtime.world_logvar.fill(0.0)
    runtime.next_world_prior_lenia = np.full((32, 32, 5), 0.7, dtype=np.float32)
    runtime.next_world_logvar_lenia = np.full((32, 32, 5), -1.5, dtype=np.float32)

    world_prior, world_logvar = runtime._assemble_world_prior()

    assert np.allclose(world_prior[..., :5], 0.7)
    assert np.allclose(world_logvar[..., :5], -1.5)
    assert np.allclose(world_prior[..., 5:], 0.0)
    assert np.allclose(world_logvar[..., 5:], 0.0)


def test_runtime_models_load_real_checkpoints(tmp_path: Path) -> None:
    trm_a_path, trm_b_path = _checkpoint_paths(tmp_path)
    models = RuntimeModels(trm_a_path, trm_b_path)

    assert models.enabled is True
    assert models.trm_a is not None
    assert models.trm_b is not None
    assert models.trm_a_config is not None
    assert models.trm_b_config is not None
    assert models.trm_a_config.image_size == 32
    assert models.trm_b_config.image_size == 32
    assert models.primary_module("world_model") is not None
    assert models.primary_module("boundary_model") is not None


def test_refresh_world_prior_from_real_trm_a_checkpoint(tmp_path: Path) -> None:
    trm_a_path, _ = _checkpoint_paths(tmp_path)
    runtime = _runtime()
    runtime.models = RuntimeModels(trm_a_path, None)

    runtime._refresh_world_prior_from_trm_a()

    assert runtime.next_world_prior_lenia is not None
    assert runtime.next_world_logvar_lenia is not None
    assert runtime.next_world_prior_lenia.shape == (32, 32, 5)
    assert runtime.next_world_logvar_lenia.shape == (32, 32, 5)
    assert np.isfinite(runtime.next_world_prior_lenia).all()
    assert np.isfinite(runtime.next_world_logvar_lenia).all()


def test_boundary_prior_from_real_trm_b_checkpoint(tmp_path: Path) -> None:
    _, trm_b_path = _checkpoint_paths(tmp_path)
    runtime = _runtime()
    runtime.models = RuntimeModels(None, trm_b_path)
    lenia_obs = np.zeros((32, 32, 5), dtype=np.float32)
    world_prior_lenia = np.zeros((32, 32, 5), dtype=np.float32)

    prior = runtime._boundary_prior_from_model(lenia_obs, world_prior_lenia)

    assert prior is not None
    assert prior.shape == (32, 32, 2)
    assert np.all((prior >= 0.0) & (prior <= 1.0))


def test_belief_update_uses_model_boundary_prior_when_available(monkeypatch) -> None:
    runtime = _runtime()
    runtime.models = SimpleNamespace(trm_a=None, trm_b=object(), torch=None)
    runtime.world_belief.fill(0.0)
    runtime.world_logvar.fill(0.0)
    runtime.boundary_belief.fill(0.0)
    runtime.boundary_logvar.fill(0.0)
    observation = np.zeros_like(runtime.world_belief, dtype=np.float32)
    sensor_gate = np.ones((*runtime.world_belief.shape[:2], 1), dtype=np.float32)
    boundary_obs = np.ones_like(runtime.boundary_belief, dtype=np.float32)
    fake_prior = np.full_like(runtime.boundary_belief, 0.5, dtype=np.float32)

    monkeypatch.setattr(runtime, "_boundary_prior_from_model", lambda *_args, **_kwargs: fake_prior.copy())

    runtime._belief_update(observation, sensor_gate, boundary_obs)

    expected = fake_prior + runtime.cfg.lambda_b * (boundary_obs - fake_prior)
    assert np.allclose(runtime.boundary_belief, expected, atol=1e-6)


def test_monitor_viability_uses_primary_trm_vm_when_available() -> None:
    torch, _, _ = require_torch()

    class FakeViabilityMonitor:
        def __call__(self, viability_state, contact_state, action_cost):
            batch = viability_state.shape[0]
            return {
                "viability_state": torch.tensor([[0.2, 0.9]], dtype=torch.float32).repeat(batch, 1),
                "viability_latent": torch.ones((batch, 32), dtype=torch.float32),
                "viability_risk": torch.full((batch, 1), 0.8, dtype=torch.float32),
                "homeostatic_error": torch.tensor([[0.3, 0.4]], dtype=torch.float32).repeat(batch, 1),
                "module_precision": torch.ones((batch,), dtype=torch.float32),
            }

    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "viability_monitor_blend": 0.5})
    runtime.models = SimpleNamespace(
        trm_a=None,
        trm_b=None,
        trm_vm=FakeViabilityMonitor(),
        trm_as=None,
        torch=torch,
    )

    monitored = runtime._monitor_viability(action_cost=0.02)

    assert monitored["source"] == "trm_vm"
    assert np.allclose(monitored["state"], np.array([0.45, 0.85], dtype=np.float32))
    assert monitored["risk"] == pytest.approx(0.8)
    assert monitored["precision"] == pytest.approx(1.0)
    assert np.allclose(monitored["homeostatic_error_vector"], np.array([0.225, 0.275], dtype=np.float32))


def test_monitor_viability_analytic_mode_ignores_trm_vm() -> None:
    torch, _, _ = require_torch()

    class FakeViabilityMonitor:
        def __call__(self, viability_state, contact_state, action_cost):
            batch = viability_state.shape[0]
            return {
                "viability_state": torch.tensor([[0.0, 0.0]], dtype=torch.float32).repeat(batch, 1),
                "viability_latent": torch.ones((batch, 32), dtype=torch.float32),
                "viability_risk": torch.full((batch, 1), 1.0, dtype=torch.float32),
                "homeostatic_error": torch.ones((batch, 2), dtype=torch.float32),
                "module_precision": torch.ones((batch,), dtype=torch.float32),
            }

    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "viability_mode": "analytic"})
    runtime.models = SimpleNamespace(trm_a=None, trm_b=None, trm_vm=FakeViabilityMonitor(), trm_as=None, torch=torch)

    monitored = runtime._monitor_viability(action_cost=0.02)

    assert monitored["source"] == "analytic"
    assert np.allclose(monitored["state"], np.array([runtime.body.G, runtime.body.B], dtype=np.float32))


def test_monitor_viability_module_primary_uses_trm_vm_state_directly() -> None:
    torch, _, _ = require_torch()

    class FakeViabilityMonitor:
        def __call__(self, viability_state, contact_state, action_cost):
            batch = viability_state.shape[0]
            return {
                "viability_state": torch.tensor([[0.2, 0.9]], dtype=torch.float32).repeat(batch, 1),
                "viability_latent": torch.ones((batch, 32), dtype=torch.float32),
                "viability_risk": torch.full((batch, 1), 0.8, dtype=torch.float32),
                "homeostatic_error": torch.tensor([[0.3, 0.4]], dtype=torch.float32).repeat(batch, 1),
                "module_precision": torch.ones((batch,), dtype=torch.float32),
            }

    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "viability_mode": "module_primary"})
    runtime.models = SimpleNamespace(trm_a=None, trm_b=None, trm_vm=FakeViabilityMonitor(), trm_as=None, torch=torch)

    monitored = runtime._monitor_viability(action_cost=0.02)

    assert monitored["source"] == "trm_vm_primary"
    assert np.allclose(monitored["state"], np.array([0.2, 0.9], dtype=np.float32))
    assert np.allclose(monitored["homeostatic_error_vector"], np.array([0.3, 0.4], dtype=np.float32))


def test_select_policy_uses_primary_trm_as_residual_logits_when_available() -> None:
    torch, _, _ = require_torch()

    class FakeActionScorer:
        def __call__(self, viability_state, action_scores, uncertainty_state):
            batch = action_scores.shape[0]
            logits = torch.tensor([[-2.0, -2.0, -2.0, 4.0, -2.0]], dtype=torch.float32).repeat(batch, 1)
            return {
                "action_state": torch.ones((batch, 32), dtype=torch.float32),
                "policy_logits": logits,
                "policy_prob": torch.softmax(logits, dim=-1),
                "action_uncertainty": torch.full((batch, 5), 0.1, dtype=torch.float32),
                "module_precision": torch.ones((batch,), dtype=torch.float32),
            }

    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "action_model_residual_scale": 8.0})
    runtime.models = SimpleNamespace(
        trm_a=None,
        trm_b=None,
        trm_vm=None,
        trm_as=FakeActionScorer(),
        torch=torch,
    )
    scores = np.array([0.2, 0.25, 0.3, 0.9, 0.35], dtype=np.float32)
    score_diag = {
        action: {
            "risk": float(scores[i]),
            "ambiguity": 0.0,
            "epistemic": 0.0,
            "pred_G": 0.5,
            "pred_B": 0.5,
            "death_risk": 0.0,
            "contact_risk": 0.0,
        }
        for i, action in enumerate(ACTIONS)
    }
    viability = {
        "state": np.array([0.55, 0.65], dtype=np.float32),
        "risk": 0.1,
        "precision": 1.0,
        "homeostatic_error": 0.0,
        "homeostatic_error_vector": np.array([0.0, 0.0], dtype=np.float32),
        "source": "analytic",
    }

    policy, action, diagnostics = runtime._select_policy(scores, score_diag, viability)

    assert action == "seal"
    assert diagnostics["source"] == "trm_as"
    assert np.isclose(float(policy.sum()), 1.0)
    assert float(policy[ACTIONS.index("seal")]) > 0.8


def test_select_policy_analytic_mode_ignores_trm_as() -> None:
    torch, _, _ = require_torch()

    class FakeActionScorer:
        def __call__(self, viability_state, action_scores, uncertainty_state):
            batch = action_scores.shape[0]
            logits = torch.tensor([[9.0, -9.0, -9.0, -9.0, -9.0]], dtype=torch.float32).repeat(batch, 1)
            return {
                "action_state": torch.ones((batch, 32), dtype=torch.float32),
                "policy_logits": logits,
                "policy_prob": torch.softmax(logits, dim=-1),
                "action_uncertainty": torch.full((batch, 5), 0.1, dtype=torch.float32),
                "module_precision": torch.ones((batch,), dtype=torch.float32),
            }

    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "action_mode": "analytic"})
    runtime.models = SimpleNamespace(trm_a=None, trm_b=None, trm_vm=None, trm_as=FakeActionScorer(), torch=torch)
    scores = np.array([0.2, 0.1, 0.3, 0.9, 0.35], dtype=np.float32)
    score_diag = {action: {"risk": float(scores[i]), "ambiguity": 0.0, "epistemic": 0.0, "pred_G": 0.5, "pred_B": 0.5, "death_risk": 0.0, "contact_risk": 0.0} for i, action in enumerate(ACTIONS)}
    viability = {
        "state": np.array([0.55, 0.65], dtype=np.float32),
        "risk": 0.1,
        "precision": 1.0,
        "homeostatic_error": 0.0,
        "homeostatic_error_vector": np.array([0.0, 0.0], dtype=np.float32),
        "source": "analytic",
    }

    policy, action, diagnostics = runtime._select_policy(scores, score_diag, viability)

    assert diagnostics["source"] == "analytic"
    assert action == "withdraw"
    assert float(policy[ACTIONS.index("withdraw")]) == pytest.approx(float(policy.max()))


def test_select_policy_module_primary_uses_trm_as_logits_directly() -> None:
    torch, _, _ = require_torch()

    class FakeActionScorer:
        def __call__(self, viability_state, action_scores, uncertainty_state):
            batch = action_scores.shape[0]
            logits = torch.tensor([[-2.0, -2.0, 5.0, -2.0, -2.0]], dtype=torch.float32).repeat(batch, 1)
            return {
                "action_state": torch.ones((batch, 32), dtype=torch.float32),
                "policy_logits": logits,
                "policy_prob": torch.softmax(logits, dim=-1),
                "action_uncertainty": torch.full((batch, 5), 0.1, dtype=torch.float32),
                "module_precision": torch.ones((batch,), dtype=torch.float32),
            }

    runtime = _runtime()
    runtime.cfg = RuntimeConfig(**{**runtime.cfg.__dict__, "action_mode": "module_primary"})
    runtime.models = SimpleNamespace(trm_a=None, trm_b=None, trm_vm=None, trm_as=FakeActionScorer(), torch=torch)
    scores = np.array([0.2, 0.1, 0.3, 0.9, 0.35], dtype=np.float32)
    score_diag = {action: {"risk": float(scores[i]), "ambiguity": 0.0, "epistemic": 0.0, "pred_G": 0.5, "pred_B": 0.5, "death_risk": 0.0, "contact_risk": 0.0} for i, action in enumerate(ACTIONS)}
    viability = {
        "state": np.array([0.55, 0.65], dtype=np.float32),
        "risk": 0.1,
        "precision": 1.0,
        "homeostatic_error": 0.0,
        "homeostatic_error_vector": np.array([0.0, 0.0], dtype=np.float32),
        "source": "analytic",
    }

    policy, action, diagnostics = runtime._select_policy(scores, score_diag, viability)

    assert diagnostics["source"] == "trm_as_primary"
    assert action == "intake"
    assert float(policy[ACTIONS.index("intake")]) > 0.98


def test_step_runs_with_real_trm_checkpoints(tmp_path: Path) -> None:
    trm_a_path, trm_b_path = _checkpoint_paths(tmp_path)
    runtime = _runtime()
    runtime.models = RuntimeModels(trm_a_path, trm_b_path)

    dead = runtime.step(0)

    assert dead is False
    assert runtime.history[-1]["action"] in ACTIONS
    assert runtime.next_world_prior_lenia is not None
    assert runtime.next_world_logvar_lenia is not None


def test_run_episode_summary_records_primary_modules(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    trm_a_path, trm_b_path = _checkpoint_paths(tmp_path)
    manifest = tmp_path / "modules.json"
    manifest.write_text(
        (
            '[{"id":"world_primary","name":"trm_a","checkpoint":"%s","primary":true},'
            '{"id":"boundary_primary","name":"trm_b","checkpoint":"%s","primary":true}]'
        )
        % (trm_a_path, trm_b_path),
        encoding="utf-8",
    )

    episode_path = run_episode(
        tmp_path / "runtime_manifest",
        catalog_path,
        RuntimeConfig(steps=4, warmup_steps=1, seed=5151),
        EnvironmentConfig(image_size=32, target_radius=8),
        module_manifest=manifest,
    )
    summary = json.loads(episode_path.with_name(f"{episode_path.stem}_summary.json").read_text())

    assert summary["module_manifest"] == str(manifest)
    assert len(summary["modules"]) == 2
    assert any(module["primary"] is True and module["id"] == "world_primary" for module in summary["modules"])
    assert any(module["primary"] is True and module["id"] == "boundary_primary" for module in summary["modules"])
    assert summary["primary_modules"]["world_model"] == "world_primary"
    assert summary["primary_modules"]["boundary_model"] == "boundary_primary"


def test_run_episode_summary_records_secondary_modules(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    trm_a_path_1, trm_b_path = _checkpoint_paths(tmp_path)
    trm_a_path_2, _ = _checkpoint_paths(tmp_path)
    manifest = tmp_path / "modules_with_secondary.json"
    manifest.write_text(
        (
            '[{"id":"world_primary","name":"trm_a","checkpoint":"%s","primary":true},'
            '{"id":"world_secondary","name":"trm_a","checkpoint":"%s"},'
            '{"id":"boundary_primary","name":"trm_b","checkpoint":"%s","primary":true}]'
        )
        % (trm_a_path_1, trm_a_path_2, trm_b_path),
        encoding="utf-8",
    )

    episode_path = run_episode(
        tmp_path / "runtime_secondary",
        catalog_path,
        RuntimeConfig(steps=4, warmup_steps=1, seed=6161),
        EnvironmentConfig(image_size=32, target_radius=8),
        module_manifest=manifest,
    )
    summary = json.loads(episode_path.with_name(f"{episode_path.stem}_summary.json").read_text())

    assert summary["primary_modules"]["world_model"] == "world_primary"
    assert summary["secondary_modules"]["world_model"] == ["world_secondary"]


def test_closed_loop_step_emits_valid_policy_distribution_and_state_ranges() -> None:
    runtime = _runtime()
    dead = runtime.step(0)
    row = runtime.history[-1]
    probs = row["policy_belief"]
    assert dead is False
    assert row["action"] in ACTIONS
    assert set(probs) == set(ACTIONS)
    assert np.isclose(sum(probs.values()), 1.0)
    assert all(0.0 <= value <= 1.0 for value in probs.values())
    assert 0.0 <= row["G"] <= 1.0
    assert 0.0 <= row["B"] <= 1.0
    assert row["sensor_gate_mean"] >= 0.0
    assert row["world_error_mean"] >= 0.0
    assert row["boundary_error_mean"] >= 0.0
    assert row["policy_entropy"] >= 0.0
    assert row["contact_resource"] >= 0.0
    assert row["contact_hazard"] >= 0.0
    assert row["contact_shelter"] >= 0.0
    assert row["homeostatic_error"] >= 0.0


def test_runtime_dies_under_sustained_extreme_hazard() -> None:
    runtime = _runtime()
    runtime.cfg = RuntimeConfig(
        **{
            **runtime.cfg.__dict__,
            "policy_mode": "no_action",
            "k_irrev": 3,
            "mu_G": 0.30,
            "mu_B": 0.30,
            "tau_G": 0.60,
            "tau_B": 0.60,
        }
    )
    runtime.body.G = 0.4
    runtime.body.B = 0.4
    runtime.env.resource.fill(0.0)
    runtime.env.hazard.fill(1.0)
    runtime.env.shelter.fill(0.0)

    dead = False
    for t in range(8):
        dead = runtime.step(t)
        if dead:
            break
    assert dead is True
    assert runtime.history[-1]["dead"] is True
    assert runtime.body.dead_count >= runtime.cfg.k_irrev


def test_closed_loop_outperforms_no_action_in_controlled_resource_corridor() -> None:
    closed_loop = _configure_controlled_resource_corridor(_runtime(seed=3001))
    no_action = _configure_controlled_resource_corridor(_runtime(seed=3001))
    no_action.cfg = RuntimeConfig(**{**no_action.cfg.__dict__, "policy_mode": "no_action"})

    for t in range(6):
        closed_loop.step(t)
        no_action.step(t)

    closed_actions = [row["action"] for row in closed_loop.history]
    no_action_actions = [row["action"] for row in no_action.history]

    assert closed_loop.body.G > no_action.body.G
    assert closed_actions[0] == "approach"
    assert "intake" in closed_actions[1:]
    assert all(action == "no_action" for action in no_action_actions)


def test_closed_loop_escapes_hazard_better_than_no_action() -> None:
    closed_loop = _configure_controlled_hazard_escape(_runtime(seed=3004))
    no_action = _configure_controlled_hazard_escape(_runtime(seed=3004))
    no_action.cfg = RuntimeConfig(**{**no_action.cfg.__dict__, "policy_mode": "no_action"})

    for t in range(6):
        closed_loop.step(t)
        no_action.step(t)

    closed_actions = [row["action"] for row in closed_loop.history]

    assert closed_loop.body.B > no_action.body.B
    assert closed_actions[0] in {"withdraw", "seal", "reconfigure"}


def test_policy_scores_expose_lookahead_diagnostics_when_horizon_gt_one() -> None:
    runtime = _configure_controlled_resource_corridor(_runtime(seed=3002))
    runtime.cfg = RuntimeConfig(
        **{
            **runtime.cfg.__dict__,
            "lookahead_horizon": 3,
            "lookahead_discount": 0.9,
        }
    )

    scores, diagnostics = runtime._policy_scores()

    assert scores.shape == (len(ACTIONS),)
    assert diagnostics["approach"]["lookahead_horizon"] == 3
    assert "continuation_score" in diagnostics["approach"]
    assert diagnostics["approach"]["lookahead_score"] <= diagnostics["withdraw"]["lookahead_score"]


def test_run_episode_writes_npz_and_json_logs(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "out"
    run_cfg = RuntimeConfig(steps=6, warmup_steps=1, seed=5678)
    env_cfg = EnvironmentConfig(image_size=32, target_radius=8)

    episode_path = run_episode(output_root, catalog_path, run_cfg, env_cfg)

    summary_path = episode_path.with_name(f"{episode_path.stem}_summary.json")
    history_path = episode_path.with_name(f"{episode_path.stem}_history.json")
    assert episode_path.exists()
    assert summary_path.exists()
    assert history_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["num_steps_executed"] >= 1
    assert "closed_loop" not in summary["action_counts"]
    assert set(summary["action_counts"]).issuperset(set(ACTIONS))


def test_run_episode_no_action_mode_records_only_no_action(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "out_no_action"
    run_cfg = RuntimeConfig(steps=6, warmup_steps=1, seed=6789, policy_mode="no_action")
    env_cfg = EnvironmentConfig(image_size=32, target_radius=8)

    episode_path = run_episode(output_root, catalog_path, run_cfg, env_cfg)

    summary_path = episode_path.with_name(f"{episode_path.stem}_summary.json")
    history_path = episode_path.with_name(f"{episode_path.stem}_history.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))

    assert summary["action_counts"]["no_action"] == summary["num_steps_executed"]
    assert all(row["action"] == "no_action" for row in history)


def test_run_episode_recorded_arrays_match_summary(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "out_shapes"
    run_cfg = RuntimeConfig(steps=7, warmup_steps=2, seed=7890)
    env_cfg = EnvironmentConfig(image_size=32, target_radius=8)

    episode_path = run_episode(output_root, catalog_path, run_cfg, env_cfg)
    summary_path = episode_path.with_name(f"{episode_path.stem}_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    arrays = np.load(episode_path)
    expected_frames = summary["num_recorded_frames"]
    assert arrays["occupancy"].shape[0] == expected_frames
    assert arrays["boundary"].shape[0] == expected_frames
    assert arrays["permeability"].shape[0] == expected_frames
    assert arrays["env_channels"].shape[0] == expected_frames
    assert arrays["world_belief"].shape[0] == expected_frames
    assert arrays["boundary_belief"].shape[0] == expected_frames


def test_run_episode_summary_contains_homeostatic_metrics(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "out_metrics"
    run_cfg = RuntimeConfig(steps=8, warmup_steps=1, seed=8100, policy_mode="no_action")
    env_cfg = EnvironmentConfig(image_size=32, target_radius=8)

    episode_path = run_episode(output_root, catalog_path, run_cfg, env_cfg)
    summary_path = episode_path.with_name(f"{episode_path.stem}_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "mean_G" in summary
    assert "mean_B" in summary
    assert "survival_fraction" in summary
    assert "final_homeostatic_error" in summary
    assert "mean_homeostatic_error" in summary
    assert "action_cost_total" in summary
    assert "mean_policy_entropy" in summary
    assert "mean_contact_resource" in summary
    assert "mean_contact_hazard" in summary
    assert "mean_contact_shelter" in summary
    assert "action_diversity" in summary
    assert 0.0 <= summary["survival_fraction"] <= 1.0
    assert 0.0 <= summary["mean_G"] <= 1.0
    assert 0.0 <= summary["mean_B"] <= 1.0
    assert summary["action_cost_total"] == 0.0
    assert 0.0 <= summary["mean_policy_entropy"]
    assert 0.0 <= summary["action_diversity"] <= 1.0


def test_main_propagates_runtime_and_environment_cli_knobs(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_episode(
        output_root: str | Path,
        seed_catalog: str | Path,
        runtime_config: RuntimeConfig,
        env_config: EnvironmentConfig,
        trm_a_checkpoint=None,
        trm_b_checkpoint=None,
        module_specs=None,
        module_manifest=None,
    ) -> Path:
        captured["output_root"] = Path(output_root)
        captured["seed_catalog"] = Path(seed_catalog)
        captured["runtime_config"] = runtime_config
        captured["env_config"] = env_config
        captured["trm_a_checkpoint"] = trm_a_checkpoint
        captured["trm_b_checkpoint"] = trm_b_checkpoint
        captured["module_specs"] = module_specs
        captured["module_manifest"] = module_manifest
        return Path(output_root) / "fake_episode.npz"

    monkeypatch.setattr(erie_runtime_module, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "erie_runtime",
            "--output-root",
            str(tmp_path / "runtime"),
            "--seed-catalog",
            str(tmp_path / "catalog.json"),
            "--steps",
            "9",
            "--warmup-steps",
            "2",
            "--seed",
            "4242",
            "--lookahead-horizon",
            "4",
            "--lookahead-discount",
            "0.7",
            "--resource-patches",
            "5",
            "--hazard-patches",
            "1",
            "--shelter-patches",
            "2",
            "--policy-mode",
            "random",
            "--viability-mode",
            "module_primary",
            "--action-mode",
            "assistive",
        ],
    )

    erie_runtime_module.main()

    runtime_config = captured["runtime_config"]
    env_config = captured["env_config"]
    assert isinstance(runtime_config, RuntimeConfig)
    assert isinstance(env_config, EnvironmentConfig)
    assert runtime_config.steps == 9
    assert runtime_config.warmup_steps == 2
    assert runtime_config.seed == 4242
    assert runtime_config.viability_mode == "module_primary"
    assert runtime_config.action_mode == "assistive"
    assert runtime_config.lookahead_horizon == 4
    assert runtime_config.lookahead_discount == 0.7
    assert runtime_config.policy_mode == "random"
    assert env_config.resource_patches == 5
    assert env_config.hazard_patches == 1
    assert env_config.shelter_patches == 2
    assert captured["module_specs"] is None
    assert captured["module_manifest"] is None
