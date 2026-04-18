from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trm_pipeline.erie_runtime import EnvironmentConfig, RuntimeConfig
from trm_pipeline.prepare_trm_va_data import (
    EPISODE_FAMILIES,
    _family_conditioned_ag_targets,
    _family_conditioned_bp_targets,
    _family_conditioned_mc_action_bias,
    prepare_trm_va_cache,
)


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


def test_prepare_trm_va_cache_writes_vm_and_as_artifacts(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "va_cache"

    manifest_path = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=6, warmup_steps=1, seed=1234),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=2,
        min_episode_samples=1,
        min_distinct_actions=1,
        max_dominant_action_fraction=1.0,
        min_episode_policy_entropy=0.0,
    )

    assert manifest_path.exists()
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert "episode_family" in rows[0]
    assert rows[0]["episode_family"] in EPISODE_FAMILIES
    assert "quality" in rows[0]
    assert rows[0]["quality"]["num_samples"] >= 1
    assert rows[0]["quality"]["distinct_actions"] >= 1
    assert rows[0]["species_context"]["multispecies_enabled"] is True
    assert rows[0]["species_context"]["species_roles"] == ["species_energy", "species_toxic", "species_niche"]
    assert "runtime_config" in rows[0]
    assert "environment_config" in rows[0]
    cache_path = Path(rows[0]["path"])
    assert cache_path.exists()

    with np.load(cache_path) as data:
        assert data["wp_observation"].shape[1:] == (32, 32, 11)
        assert data["wp_sensor_gate"].shape[1:] == (32, 32, 1)
        assert data["wp_species_fields"].shape[1:] == (32, 32, 4)
        assert data["wp_flow_channels"].shape[1:] == (32, 32, 2)
        assert data["wp_input_view"].shape[1:] == (32, 32, 18)
        assert data["wp_target_observation"].shape[1:] == (32, 32, 11)
        assert data["bd_observation"].shape[1:] == (32, 32, 11)
        assert data["bd_world_error"].shape[1:] == (32, 32, 11)
        assert data["bd_sensor_gate"].shape[1:] == (32, 32, 1)
        assert data["bd_delta_observation"].shape[1:] == (32, 32, 11)
        assert data["bd_input_view"].shape[1:] == (32, 32, 34)
        assert data["bd_boundary_target"].shape[1:] == (32, 32, 1)
        assert data["bd_permeability_target"].shape[1:] == (32, 32, 1)
        assert data["bp_boundary_patch"].shape[1:] == (16, 16, 1)
        assert data["bp_permeability_patch"].shape[1:] == (16, 16, 1)
        assert data["bp_observation_patch"].shape[1:] == (16, 16, 11)
        assert data["bp_species_patch"].shape[1:] == (16, 16, 4)
        assert data["bp_flow_patch"].shape[1:] == (16, 16, 2)
        assert data["bp_viability_state"].shape[1] == 2
        assert data["bp_input_view"].shape[1:] == (16, 16, 21)
        assert data["bp_target_permeability_patch"].shape[1:] == (16, 16, 1)
        assert data["bp_target_interface_gain"].shape[1] == 1
        assert data["bp_target_aperture_gain"].shape[1] == 1
        assert data["bp_target_mode"].ndim == 1
        assert data["vm_viability_state"].shape[1] == 2
        assert data["vm_contact_state"].shape[1] == 4
        assert data["vm_species_contact_state"].shape[1] == 4
        assert data["vm_action_cost"].shape[1] == 1
        assert data["vm_input_view"].shape[1] == 11
        assert data["vm_target_state"].shape[1] == 2
        assert data["as_action_scores"].shape[1] == 5
        assert data["as_uncertainty_state"].shape[1] == 4
        assert data["as_env_contact_state"].shape[1] == 4
        assert data["as_species_contact_state"].shape[1] == 4
        assert data["as_input_view"].shape[1] == 19
        assert data["as_target_policy"].shape[1] == 5
        assert data["as_target_action"].ndim == 1
        assert np.all(data["as_target_policy"] >= 0.0)
        assert data["ag_input_view"].shape[1] == 22
        assert data["ag_target_gated_logits"].shape[1] == 5
        assert data["ag_target_inhibition_mask"].shape[1] == 5
        assert data["ag_target_control_mode"].ndim == 1
        assert data["mc_input_view"].shape[1:] == (8, 44)
        assert data["mc_window_mask"].shape[1] == 8
        assert data["mc_target_context_state"].shape[1] == 44
        assert data["mc_target_action_bias"].shape[1] == 5
        assert data["mc_target_boundary_bias"].shape[1] == 3

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["episode_families"] == list(EPISODE_FAMILIES)
    assert sum(summary["family_counts"].values()) == 2
    assert summary["attempted_episodes"] >= summary["retained_episodes"] == 2
    assert "aggregate_action_counts" in summary
    assert "quality_thresholds" in summary
    assert "aggregate_recovery_fraction_mean" in summary
    assert "aggregate_stress_defensive_fraction_mean" in summary
    assert "aggregate_stress_exploit_fraction_mean" in summary
    assert summary["multispecies_enabled"] is True
    assert summary["species_roles"] == ["species_energy", "species_toxic", "species_niche"]
    assert "role_view_manifests" in summary
    assert set(summary["role_view_manifests"]) == {"trm_wp", "trm_bd", "trm_bp", "trm_vm", "trm_as", "trm_ag", "trm_mc"}

    views_summary = json.loads((output_root / "views" / "summary.json").read_text(encoding="utf-8"))
    assert views_summary["trm_wp"]["input_key"] == "wp_input_view"
    assert views_summary["trm_wp"]["target_key"] == "wp_target_observation"
    assert views_summary["trm_wp"]["input_channels"] == 18
    assert views_summary["trm_bd"]["state_key"] == "bd_observation"
    assert views_summary["trm_bd"]["boundary_in_channels_total"] == 34
    assert views_summary["trm_bp"]["input_view_key"] == "bp_input_view"
    assert views_summary["trm_bp"]["patch_size"] == 16
    assert views_summary["trm_vm"]["input_view_key"] == "vm_input_view"
    assert views_summary["trm_as"]["input_view_key"] == "as_input_view"
    assert views_summary["trm_ag"]["input_view_key"] == "ag_input_view"
    assert views_summary["trm_ag"]["input_dim"] == 22
    assert views_summary["trm_mc"]["input_view_key"] == "mc_input_view"
    assert views_summary["trm_mc"]["window_size"] == 8
    assert views_summary["trm_mc"]["input_dim"] == 44

    wp_rows = [
        json.loads(line)
        for line in Path(summary["role_view_manifests"]["trm_wp"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(wp_rows) == 2
    assert wp_rows[0]["view_name"] == "trm_wp"
    assert wp_rows[0]["baseline_key"] == "wp_observation"
    assert wp_rows[0]["num_pairs"] == wp_rows[0]["num_samples"]

    bd_rows = [
        json.loads(line)
        for line in Path(summary["role_view_manifests"]["trm_bd"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert bd_rows[0]["sensor_gate_key"] == "bd_sensor_gate"
    assert bd_rows[0]["boundary_target_key"] == "bd_boundary_target"

    bp_rows = [
        json.loads(line)
        for line in Path(summary["role_view_manifests"]["trm_bp"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert bp_rows[0]["input_view_key"] == "bp_input_view"
    assert bp_rows[0]["target_mode_key"] == "bp_target_mode"

    ag_rows = [
        json.loads(line)
        for line in Path(summary["role_view_manifests"]["trm_ag"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert ag_rows[0]["input_view_key"] == "ag_input_view"
    assert ag_rows[0]["target_inhibition_mask_key"] == "ag_target_inhibition_mask"
    assert ag_rows[0]["target_control_mode_key"] == "ag_target_control_mode"

    mc_rows = [
        json.loads(line)
        for line in Path(summary["role_view_manifests"]["trm_mc"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert mc_rows[0]["input_view_key"] == "mc_input_view"
    assert mc_rows[0]["window_mask_key"] == "mc_window_mask"
    assert mc_rows[0]["target_action_bias_key"] == "mc_target_action_bias"
    assert mc_rows[0]["window_size"] == 8


def test_prepare_trm_va_cache_reuse_output_root_clears_stale_artifacts(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "va_cache_reuse"

    first_manifest = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=6, warmup_steps=1, seed=1234),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=3,
        min_episode_samples=1,
        min_distinct_actions=1,
        max_dominant_action_fraction=1.0,
        min_episode_policy_entropy=0.0,
    )
    assert first_manifest.exists()
    stale_file = output_root / "episodes" / "stale.npz"
    stale_file.write_bytes(b"stale")

    second_manifest = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=4, warmup_steps=1, seed=1235),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=2,
        min_episode_samples=1,
        min_distinct_actions=1,
        max_dominant_action_fraction=1.0,
        min_episode_policy_entropy=0.0,
    )

    rows = [json.loads(line) for line in second_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert not stale_file.exists()
    episode_files = sorted((output_root / "episodes").glob("*.npz"))
    assert len(episode_files) == 2


def test_family_conditioned_bp_targets_adjusts_vent_edge_under_high_stress() -> None:
    perm_patch = np.ones((16, 16, 1), dtype=np.float32) * 0.8
    conditioned_perm, conditioned_interface, conditioned_aperture, conditioned_mode = _family_conditioned_bp_targets(
        "vent_edge",
        current_aperture_gain=0.42,
        target_permeability_patch=perm_patch,
        target_interface_gain=0.08,
        target_aperture_gain=0.48,
        target_mode=0,
        env_contact_state=np.array([0.25, 0.65, 0.55, 0.10], dtype=np.float32),
        species_contact_state=np.array([0.05, 0.30, 0.20, 0.0], dtype=np.float32),
    )

    assert conditioned_mode == 2


def test_family_conditioned_ag_targets_separates_vent_edge_and_uncertain_corridor() -> None:
    base_logits = np.array([0.2, 0.0, -0.1, 0.1, -0.2], dtype=np.float32)

    vent_inhibition, vent_mode, vent_logits = _family_conditioned_ag_targets(
        "vent_edge",
        as_target_logits=base_logits,
        viability_state=np.array([0.60, 0.58], dtype=np.float32),
        homeostatic_error_vector=np.array([0.05, 0.07], dtype=np.float32),
        viability_risk=np.array([0.52], dtype=np.float32),
        uncertainty_state=np.array([0.08, 0.09, 0.10, 0.12], dtype=np.float32),
        env_contact_state=np.array([0.22, 0.55, 0.36, 0.12], dtype=np.float32),
        species_contact_state=np.array([0.04, 0.10, 0.08, 0.02], dtype=np.float32),
    )
    uncertain_inhibition, uncertain_mode, uncertain_logits = _family_conditioned_ag_targets(
        "uncertain_corridor",
        as_target_logits=base_logits,
        viability_state=np.array([0.52, 0.70], dtype=np.float32),
        homeostatic_error_vector=np.array([0.03, 0.01], dtype=np.float32),
        viability_risk=np.array([0.18], dtype=np.float32),
        uncertainty_state=np.array([0.42, 0.39, 0.36, 0.34], dtype=np.float32),
        env_contact_state=np.array([0.28, 0.10, 0.08, 0.20], dtype=np.float32),
        species_contact_state=np.array([0.06, 0.01, 0.01, 0.04], dtype=np.float32),
    )

    assert vent_mode == 2
    assert uncertain_mode == 0
    assert vent_inhibition[0] > uncertain_inhibition[0]
    assert vent_inhibition[2] > uncertain_inhibition[2]
    assert vent_inhibition[0] == pytest.approx(1.0)
    assert vent_inhibition[2] == pytest.approx(1.0)
    assert vent_logits.shape == (5,)
    assert uncertain_logits.shape == (5,)


def test_family_conditioned_bp_targets_leaves_non_vent_edge_unchanged_when_safe() -> None:
    perm_patch = np.ones((16, 16, 1), dtype=np.float32) * 0.6
    conditioned_perm, conditioned_interface, conditioned_aperture, conditioned_mode = _family_conditioned_bp_targets(
        "uncertain_corridor",
        current_aperture_gain=0.35,
        target_permeability_patch=perm_patch,
        target_interface_gain=0.02,
        target_aperture_gain=0.38,
        target_mode=1,
        env_contact_state=np.array([0.30, 0.10, 0.05, 0.25], dtype=np.float32),
        species_contact_state=np.zeros((4,), dtype=np.float32),
    )

    assert conditioned_mode == 1
    assert conditioned_aperture == pytest.approx(0.38)
    assert conditioned_interface == pytest.approx(0.02)
    assert np.allclose(conditioned_perm, perm_patch)


def test_prepare_trm_va_cache_cycles_episode_families(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "va_cache_families"

    manifest_path = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=4, warmup_steps=1, seed=1234),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=len(EPISODE_FAMILIES),
        min_episode_samples=1,
        min_distinct_actions=1,
        max_dominant_action_fraction=1.0,
        min_episode_policy_entropy=0.0,
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    families = [row["episode_family"] for row in rows]
    assert families == list(EPISODE_FAMILIES)


def test_prepare_trm_va_cache_records_shaping_and_bias_knobs(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "va_cache_shaped"

    manifest_path = prepare_trm_va_cache(
        seed_catalog=catalog_path,
        output_root=output_root,
        runtime_config=RuntimeConfig(steps=4, warmup_steps=1, seed=1234),
        env_config=EnvironmentConfig(image_size=32, target_radius=8),
        num_episodes=3,
        target_band_weight=0.7,
        target_g_overshoot_weight=0.9,
        defensive_family_bias=2.0,
        min_episode_samples=1,
        min_distinct_actions=1,
        max_dominant_action_fraction=1.0,
        min_episode_policy_entropy=0.0,
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(row["target_band_weight"] == 0.7 for row in rows)
    assert all(row["target_g_overshoot_weight"] == 0.9 for row in rows)
    assert all(row["defensive_family_bias"] == 2.0 for row in rows)

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["target_band_weight"] == 0.7
    assert summary["target_g_overshoot_weight"] == 0.9
    assert summary["defensive_family_bias"] == 2.0
    assert len(summary["family_pool"]) > len(EPISODE_FAMILIES)


def test_family_conditioned_mc_action_bias_changes_targets_by_family() -> None:
    base = np.zeros((3, 5), dtype=np.float32)
    viability = np.array([[0.5, 0.3], [0.45, 0.55], [0.6, 0.65]], dtype=np.float32)
    env_contact = np.array([[0.3, 0.5, 0.4, 0.1], [0.2, 0.3, 0.2, 0.2], [0.1, 0.1, 0.1, 0.4]], dtype=np.float32)
    species_contact = np.array([[0.1, 0.2, 0.2, 0.0], [0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.2]], dtype=np.float32)
    uncertainty = np.array([[0.4, 0.4, 0.4, 0.4], [0.3, 0.3, 0.3, 0.3], [0.8, 0.8, 0.8, 0.8]], dtype=np.float32)

    fragile = _family_conditioned_mc_action_bias(
        "fragile_boundary", base, viability, env_contact, species_contact, uncertainty
    )
    corridor = _family_conditioned_mc_action_bias(
        "uncertain_corridor", base, viability, env_contact, species_contact, uncertainty
    )

    # fragile_boundary should bias away from intake and toward seal/reconfigure
    assert fragile[0, 3] > fragile[0, 2]
    assert fragile[0, 4] > fragile[0, 2]
    # uncertain_corridor should push reconfigure above seal under high uncertainty
    assert corridor[2, 4] > corridor[2, 3]


def test_prepare_trm_va_cache_rejects_when_quality_thresholds_are_impossible(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path)
    output_root = tmp_path / "va_cache_strict"

    with pytest.raises(SystemExit, match="could not retain enough TRM-VA episodes"):
        prepare_trm_va_cache(
            seed_catalog=catalog_path,
            output_root=output_root,
            runtime_config=RuntimeConfig(steps=4, warmup_steps=1, seed=1234),
            env_config=EnvironmentConfig(image_size=32, target_radius=8),
            num_episodes=2,
            max_attempt_multiplier=1,
            min_distinct_actions=6,
        )
