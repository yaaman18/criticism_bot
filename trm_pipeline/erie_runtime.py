from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ensure_dir,
    gaussian_noise,
    load_json,
    save_json,
    seed_everything,
)
from .adaptive_controller import AdaptiveController, AdaptiveControllerConfig
from .lenia_data import (
    LeniaSeed,
    build_kernel,
    center_seed_on_canvas,
    derive_multichannel_state,
    lenia_step,
    load_seed_catalog,
    parse_band_list,
    rle2arr_2d,
    sample_params,
)
from .models import TRMModelConfig, adapt_trm_outputs, build_trm, get_trm_registry, require_torch
from .prediction_cell import LocalPredictionCellLayer, PredictionCellGrid, PredictionCellUpdateConfig
from .runtime_metrics import death_cause_counts, episode_metrics, homeostatic_error
from .runtime_policy import apply_ag_assistive_veto
from .runtime_population import (
    alive_bodies,
    can_expand_population,
    classify_death_cause,
    is_action_locked,
    select_primary_body,
    spawn_child_from_primary,
    spawn_drive as population_spawn_drive,
    spawn_split_signals as population_spawn_split_signals,
    split_child_from_primary,
    split_drive as population_split_drive,
    update_death_state,
)
from .trm_input_views import build_trm_bp_input_view, build_trm_mc_input_view, extract_centered_patch
from .trm_input_views import build_trm_ag_input_view


ACTIONS = ("approach", "withdraw", "intake", "seal", "reconfigure")
DEATH_CAUSE_EXPECTED = "expected_extinction"
DEATH_CAUSE_DEGENERATE = "degenerate_extinction"
DEATH_CAUSE_POLICY_FORBIDDEN = "policy_forbidden_extinction"


@dataclass(frozen=True)
class EnvironmentConfig:
    image_size: int = 64
    target_radius: int = 12
    # Compatibility names retained for older contracts and CLI flags.
    resource_patches: int = 3
    hazard_patches: int = 3
    shelter_patches: int = 1
    field_sigma_min: float = 4.0
    field_sigma_max: float = 9.0
    resource_regen: float = 0.003
    hazard_drift_sigma: float = 0.001
    toxicity_drift_sigma: float = 0.001
    shelter_stability: float = 1.0
    flow_strength: float = 0.85
    flow_drift_sigma: float = 0.0005
    species_field_gain_energy: float = 0.18
    species_field_gain_thermal: float = 0.14
    species_field_gain_toxicity: float = 0.20
    species_field_gain_niche: float = 0.16

    @property
    def energy_gradient_patches(self) -> int:
        return int(self.resource_patches)

    @property
    def thermal_stress_patches(self) -> int:
        return int(self.hazard_patches)

    @property
    def toxicity_patches(self) -> int:
        return max(1, int(round(self.hazard_patches * 0.75)))

    @property
    def niche_stability_patches(self) -> int:
        return int(self.shelter_patches)


def add_environment_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--energy-gradient-patches",
        type=int,
        default=None,
        help="Canonical alias for --resource-patches.",
    )
    parser.add_argument(
        "--thermal-stress-patches",
        type=int,
        default=None,
        help="Canonical alias for --hazard-patches.",
    )
    parser.add_argument(
        "--niche-stability-patches",
        type=int,
        default=None,
        help="Canonical alias for --shelter-patches.",
    )
    parser.add_argument("--resource-patches", type=int, default=None, help="Compatibility alias for energy-gradient patches.")
    parser.add_argument("--hazard-patches", type=int, default=None, help="Compatibility alias for thermal-stress patches.")
    parser.add_argument("--shelter-patches", type=int, default=None, help="Compatibility alias for niche-stability patches.")


def environment_config_from_args(args: argparse.Namespace) -> EnvironmentConfig:
    resource_patches = (
        args.energy_gradient_patches
        if args.energy_gradient_patches is not None
        else args.resource_patches
        if args.resource_patches is not None
        else 3
    )
    hazard_patches = (
        args.thermal_stress_patches
        if args.thermal_stress_patches is not None
        else args.hazard_patches
        if args.hazard_patches is not None
        else 3
    )
    shelter_patches = (
        args.niche_stability_patches
        if args.niche_stability_patches is not None
        else args.shelter_patches
        if args.shelter_patches is not None
        else 1
    )
    return EnvironmentConfig(
        resource_patches=int(resource_patches),
        hazard_patches=int(hazard_patches),
        shelter_patches=int(shelter_patches),
    )


def canonical_environment_config(env_config: EnvironmentConfig) -> dict[str, Any]:
    return {
        "energy_gradient_patches": int(env_config.energy_gradient_patches),
        "thermal_stress_patches": int(env_config.thermal_stress_patches),
        "toxicity_patches": int(env_config.toxicity_patches),
        "niche_stability_patches": int(env_config.niche_stability_patches),
        "flow_strength": float(env_config.flow_strength),
        "flow_drift_sigma": float(env_config.flow_drift_sigma),
        "compat_aliases": {
            "resource_patches": int(env_config.resource_patches),
            "hazard_patches": int(env_config.hazard_patches),
            "shelter_patches": int(env_config.shelter_patches),
        },
    }


@dataclass(frozen=True)
class RuntimeConfig:
    steps: int = 128
    warmup_steps: int = 8
    seed: int = 20260316
    occupancy_radius: float = 7.5
    occupancy_softness: float = 1.2
    base_permeability: float = 0.18
    aperture_gain: float = 0.45
    aperture_width_deg: float = 70.0
    move_step: float = 2.0
    p_min: float = 0.05
    p_max: float = 20.0
    lambda_w: float = 0.10
    lambda_b: float = 0.08
    beta_pi: float = 4.0
    mu_G: float = 0.015
    mu_B: float = 0.010
    alpha_R: float = 0.08
    alpha_H: float = 0.06
    alpha_X: float = 0.045
    alpha_S: float = 0.03
    tau_G: float = 0.15
    tau_B: float = 0.20
    k_irrev: int = 8
    policy_forbidden_min_survival_steps: int = 0
    degenerate_action_lock_window: int = 0
    spawn_logit_bias: float = -2.2
    spawn_trace_gain: float = 2.0
    spawn_resource_gain: float = 1.2
    spawn_hazard_penalty: float = 1.8
    spawn_viability_gain: float = 1.0
    spawn_candidate_threshold: float = 0.55
    split_logit_bias: float = -2.6
    split_mass_gain: float = 0.8
    split_energy_gain: float = 1.2
    split_boundary_penalty: float = 1.6
    split_candidate_threshold: float = 0.60
    max_bodies: int = 1
    spawn_radius_scale: float = 0.78
    split_radius_scale: float = 0.82
    spawn_energy_share: float = 0.34
    split_energy_share: float = 0.45
    spawn_offset: float = 4.5
    passive_body_flow_gain: float = 1.0
    challenge_ratio_min: float = 0.05
    challenge_ratio_max: float = 0.95
    activity_a0: float = -0.35
    activity_a1: float = 2.25
    activity_a2: float = 2.10
    activity_a3: float = 1.45
    role_score_w_energy: float = 1.0
    role_score_w_confidence: float = 0.75
    role_score_w_hazard: float = 1.2
    role_score_w_boundary: float = 0.85
    role_action_bias_gain: float = 0.0
    aux_policy_mode: str = "full_policy"
    G0: float = 0.70
    B0: float = 0.80
    G_target: float = 0.55
    B_target: float = 0.65
    risk_wG: float = 2.0
    risk_wB: float = 2.3
    risk_wD: float = 3.5
    risk_wHomeostasis: float = 0.75
    risk_wReserve: float = 1.35
    reserve_G: float = 0.10
    reserve_B: float = 0.10
    contact_w_thermal: float = 0.75
    contact_w_toxicity: float = 0.95
    contact_w_energy: float = 0.35
    contact_w_niche: float = 0.45
    contact_delta_w_thermal: float = 0.55
    contact_delta_w_toxicity: float = 0.85
    contact_delta_w_energy: float = 0.35
    contact_delta_w_niche: float = 0.50
    ambiguity_w_boundary: float = 0.5
    epistemic_scale: float = 1.0
    viability_mode: str = "assistive"
    action_mode: str = "assistive"
    action_gating_mode: str = "assistive"
    boundary_control_mode: str = "assistive"
    context_memory_mode: str = "assistive"
    viability_monitor_blend: float = 0.35
    action_model_residual_scale: float = 1.0
    action_gating_blend: float = 0.35
    boundary_control_blend: float = 0.35
    context_memory_residual_scale: float = 0.35
    context_memory_window_size: int = 8
    lookahead_horizon: int = 2
    lookahead_discount: float = 0.85
    observation_noise: float = 0.01
    world_logvar_init: float = -0.5
    boundary_logvar_init: float = -0.3
    world_logvar_drift: float = 0.02
    boundary_logvar_drift: float = 0.015
    visual_cell_rows: int = 8
    visual_cell_cols: int = 8
    visual_cell_lateral_coupling: float = 0.0
    visual_attention_error_gain: float = 1.0
    visual_attention_uncertainty_gain: float = 0.25
    visual_attention_temperature: float = 8.0
    visual_attention_epistemic_gain: float = 0.0
    use_trm_a: bool = False
    use_trm_b: bool = False
    policy_mode: str = "closed_loop"


@dataclass
class ExternalState:
    scalar_state: np.ndarray
    prev_scalar_state: np.ndarray
    species_energy_state: np.ndarray
    species_toxic_state: np.ndarray
    species_niche_state: np.ndarray
    energy_gradient: np.ndarray
    thermal_stress: np.ndarray
    toxicity: np.ndarray
    niche_stability: np.ndarray
    flow_y: np.ndarray
    flow_x: np.ndarray

    def lenia_multistate(self, params: dict[str, Any]) -> np.ndarray:
        return derive_multichannel_state(self.prev_scalar_state, self.scalar_state, params)

    def species_multistates(self, params_by_species: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
        return {
            "species_energy": derive_multichannel_state(
                self.species_energy_state, self.species_energy_state, params_by_species["species_energy"]
            ),
            "species_toxic": derive_multichannel_state(
                self.species_toxic_state, self.species_toxic_state, params_by_species["species_toxic"]
            ),
            "species_niche": derive_multichannel_state(
                self.species_niche_state, self.species_niche_state, params_by_species["species_niche"]
            ),
        }

    def species_sources(self) -> np.ndarray:
        return np.stack(
            [
                self.species_energy_state,
                self.species_toxic_state,
                self.species_niche_state,
            ],
            axis=-1,
        ).astype(np.float32)

    def as_channels(self, params: dict[str, Any]) -> np.ndarray:
        multi = self.lenia_multistate(params)
        env = np.stack(
            [
                self.energy_gradient,
                self.thermal_stress,
                self.toxicity,
                self.niche_stability,
                self.flow_y,
                self.flow_x,
            ],
            axis=-1,
        ).astype(np.float32)
        return np.concatenate([multi, env], axis=-1).astype(np.float32)

    def as_external_channels(
        self,
        base_params: dict[str, Any],
        params_by_species: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        base_multi = self.lenia_multistate(base_params)
        species = self.species_multistates(params_by_species)
        env = np.stack(
            [
                self.energy_gradient,
                self.thermal_stress,
                self.toxicity,
                self.niche_stability,
                self.flow_y,
                self.flow_x,
            ],
            axis=-1,
        ).astype(np.float32)
        return np.concatenate(
            [
                base_multi,
                species["species_energy"],
                species["species_toxic"],
                species["species_niche"],
                env,
            ],
            axis=-1,
        ).astype(np.float32)


@dataclass
class BodyState:
    centroid_y: float
    centroid_x: float
    radius: float
    aperture_angle: float
    aperture_gain: float
    aperture_width_deg: float
    G: float
    B: float
    energy: float = 0.0
    mass: float = 0.0
    boundary_integrity: float = 1.0
    alive: bool = True
    dead_count: int = 0
    body_id: int = 0
    parent_id: int = -1
    generation: int = 0
    role: str = "conservative"
    prediction_confidence: float = 0.5
    local_hazard: float = 0.0


class RuntimeModels:
    def __init__(
        self,
        trm_a_checkpoint: str | Path | None,
        trm_b_checkpoint: str | Path | None,
        module_specs: list[dict[str, Any]] | None = None,
        module_manifest: str | Path | None = None,
    ) -> None:
        self.torch = None
        self.trm_a = None
        self.trm_b = None
        self.trm_vm = None
        self.trm_as = None
        self.trm_ag = None
        self.trm_bp = None
        self.trm_mc = None
        self.trm_a_config: TRMModelConfig | None = None
        self.trm_b_config: TRMModelConfig | None = None
        self.trm_vm_config: TRMModelConfig | None = None
        self.trm_as_config: TRMModelConfig | None = None
        self.trm_ag_config: TRMModelConfig | None = None
        self.trm_bp_config: TRMModelConfig | None = None
        self.trm_mc_config: TRMModelConfig | None = None
        self.modules: list[dict[str, Any]] = []
        self._primary_by_role: dict[str, dict[str, Any]] = {}

        resolved_specs = self._resolve_module_specs(
            trm_a_checkpoint,
            trm_b_checkpoint,
            module_specs,
            module_manifest,
        )
        if not resolved_specs:
            return
        torch, _, _ = require_torch()
        self.torch = torch
        for spec in resolved_specs:
            self._load_module(spec)

    @property
    def enabled(self) -> bool:
        return bool(self.modules)

    def primary_module(self, role: str) -> dict[str, Any] | None:
        return self._primary_by_role.get(role)

    def modules_by_role(self, role: str) -> list[dict[str, Any]]:
        return [module for module in self.modules if module.get("role") == role]

    def secondary_modules(self, role: str) -> list[dict[str, Any]]:
        primary = self.primary_module(role)
        return [
            module
            for module in self.modules_by_role(role)
            if primary is None or module["id"] != primary["id"]
        ]

    @staticmethod
    def _resolve_module_specs(
        trm_a_checkpoint: str | Path | None,
        trm_b_checkpoint: str | Path | None,
        module_specs: list[dict[str, Any]] | None,
        module_manifest: str | Path | None,
    ) -> list[dict[str, Any]]:
        if module_manifest is not None:
            loaded = load_json(module_manifest)
            if not isinstance(loaded, list):
                raise SystemExit("module manifest must be a JSON list")
            return [dict(item) for item in loaded]
        if module_specs is not None:
            return [dict(item) for item in module_specs]

        specs: list[dict[str, Any]] = []
        if trm_a_checkpoint is not None:
            specs.append({"name": "trm_a", "checkpoint": str(trm_a_checkpoint)})
        if trm_b_checkpoint is not None:
            specs.append({"name": "trm_b", "checkpoint": str(trm_b_checkpoint)})
        return specs

    def _load_module(self, spec: dict[str, Any]) -> None:
        assert self.torch is not None
        checkpoint_path = spec.get("checkpoint")
        module_name = spec.get("name")
        if not checkpoint_path or not module_name:
            raise SystemExit("each module spec must include `name` and `checkpoint`")

        ckpt = self.torch.load(checkpoint_path, map_location="cpu")
        model_config = TRMModelConfig(**ckpt.get("model_config", {}))
        model = build_trm(module_name, model_config)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        registry = get_trm_registry()
        registry_role = registry[module_name]["role"] if module_name in registry else None

        module_entry = {
            "id": spec.get("id", module_name),
            "name": module_name,
            "role": ckpt.get("module_role") or spec.get("role") or registry_role,
            "checkpoint": str(checkpoint_path),
            "primary": bool(spec.get("primary", False)),
            "config": model_config,
            "model": model,
            "output_adapter": lambda outputs, _name=module_name: adapt_trm_outputs(_name, outputs),
        }
        self.modules.append(module_entry)
        self._refresh_primary_roles()

    def _refresh_primary_roles(self) -> None:
        primary_counts: dict[str, int] = {}
        for module in self.modules:
            role = module.get("role")
            if not role:
                continue
            if module.get("primary", False):
                primary_counts[role] = primary_counts.get(role, 0) + 1
        duplicated_roles = [role for role, count in primary_counts.items() if count > 1]
        if duplicated_roles:
            raise SystemExit(
                "multiple primary modules declared for role(s): " + ", ".join(sorted(duplicated_roles))
            )

        primary_by_role: dict[str, dict[str, Any]] = {}
        for module in self.modules:
            role = module.get("role")
            if not role:
                continue
            current = primary_by_role.get(role)
            if current is None:
                primary_by_role[role] = module
                continue
            if module.get("primary", False) and not current.get("primary", False):
                primary_by_role[role] = module
        self._primary_by_role = primary_by_role

        world_module = self.primary_module("world_model")
        boundary_module = self.primary_module("boundary_model")
        self.trm_a = world_module["model"] if world_module is not None else None
        self.trm_a_config = world_module["config"] if world_module is not None else None
        self.trm_b = boundary_module["model"] if boundary_module is not None else None
        self.trm_b_config = boundary_module["config"] if boundary_module is not None else None
        viability_module = self.primary_module("viability_monitor")
        self.trm_vm = viability_module["model"] if viability_module is not None else None
        self.trm_vm_config = viability_module["config"] if viability_module is not None else None
        action_module = self.primary_module("action_scoring")
        self.trm_as = action_module["model"] if action_module is not None else None
        self.trm_as_config = action_module["config"] if action_module is not None else None
        action_gating_module = self.primary_module("action_gating")
        self.trm_ag = action_gating_module["model"] if action_gating_module is not None else None
        self.trm_ag_config = action_gating_module["config"] if action_gating_module is not None else None
        boundary_control_module = self.primary_module("boundary_permeability_control")
        self.trm_bp = boundary_control_module["model"] if boundary_control_module is not None else None
        self.trm_bp_config = boundary_control_module["config"] if boundary_control_module is not None else None
        memory_context_module = self.primary_module("memory_context")
        self.trm_mc = memory_context_module["model"] if memory_context_module is not None else None
        self.trm_mc_config = memory_context_module["config"] if memory_context_module is not None else None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_scalar(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - float(np.max(x))
    exp = np.exp(z)
    total = float(exp.sum())
    if total <= 0.0:
        return np.full_like(x, 1.0 / len(x))
    return exp / total


def _action_onehot_runtime(action: str | None, include_no_action: bool = True) -> np.ndarray:
    labels = list(ACTIONS) + (["no_action"] if include_no_action else [])
    vec = np.zeros((len(labels),), dtype=np.float32)
    if action in labels:
        vec[labels.index(str(action))] = 1.0
    return vec


def _entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    clipped = np.clip(probs.astype(np.float32), eps, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def _gaussian_blob_field(
    rng: np.random.Generator,
    image_size: int,
    count: int,
    sigma_min: float,
    sigma_max: float,
) -> np.ndarray:
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    field = np.zeros((image_size, image_size), dtype=np.float32)
    for _ in range(count):
        cy = float(rng.uniform(0.2 * image_size, 0.8 * image_size))
        cx = float(rng.uniform(0.2 * image_size, 0.8 * image_size))
        sigma = float(rng.uniform(sigma_min, sigma_max))
        amp = float(rng.uniform(0.5, 1.0))
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        field += amp * np.exp(-dist2 / max(2.0 * sigma * sigma, 1e-6))
    if field.max() > 0:
        field /= float(field.max())
    return field.astype(np.float32)


def _build_flow_field(
    energy_gradient: np.ndarray,
    thermal_stress: np.ndarray,
    niche_stability: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    gy_e, gx_e = np.gradient(energy_gradient.astype(np.float32))
    gy_t, gx_t = np.gradient(thermal_stress.astype(np.float32))
    gy_n, gx_n = np.gradient(niche_stability.astype(np.float32))
    flow_y = strength * (-0.85 * gy_e - 0.35 * gy_t + 0.20 * gy_n)
    flow_x = strength * (0.85 * gx_e + 0.35 * gx_t - 0.20 * gx_n)
    flow_y = np.clip(flow_y, -1.0, 1.0).astype(np.float32)
    flow_x = np.clip(flow_x, -1.0, 1.0).astype(np.float32)
    return flow_y, flow_x


def _advect_field(field: np.ndarray, flow_y: np.ndarray, flow_x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    height, width = field.shape
    yy, xx = np.indices(field.shape, dtype=np.float32)
    src_y = np.clip(np.rint(yy - scale * flow_y), 0, height - 1).astype(np.int32)
    src_x = np.clip(np.rint(xx - scale * flow_x), 0, width - 1).astype(np.int32)
    return field[src_y, src_x].astype(np.float32)


def _blur_field(field: np.ndarray, rounds: int = 2) -> np.ndarray:
    out = field.astype(np.float32)
    for _ in range(max(1, rounds)):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 5.0
    return out.astype(np.float32)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _body_fields(body: BodyState, image_size: int, softness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    dy = yy - float(body.centroid_y)
    dx = xx - float(body.centroid_x)
    dist = np.sqrt(dx * dx + dy * dy)
    occupancy = _sigmoid((float(body.radius) - dist) / max(softness, 1e-6)).astype(np.float32)
    boundary = np.exp(-((dist - float(body.radius)) ** 2) / max(2.0 * softness * softness, 1e-6)).astype(np.float32)
    boundary = boundary / max(float(boundary.max()), 1e-6)

    angle = np.arctan2(dy, dx)
    angle0 = float(body.aperture_angle)
    width = math.radians(float(body.aperture_width_deg))
    angle_delta = np.angle(np.exp(1j * (angle - angle0))).astype(np.float32)
    aperture = np.exp(-(angle_delta**2) / max(2.0 * (width / 2.0) ** 2, 1e-6)).astype(np.float32)
    permeability = boundary * np.clip(float(body.aperture_gain) * aperture + 0.05, 0.0, 1.0)
    return occupancy, boundary, permeability.astype(np.float32)


def _gradients(field: np.ndarray, y: float, x: float) -> tuple[float, float]:
    gy, gx = np.gradient(field.astype(np.float32))
    yi = int(np.clip(round(y), 0, field.shape[0] - 1))
    xi = int(np.clip(round(x), 0, field.shape[1] - 1))
    return float(gy[yi, xi]), float(gx[yi, xi])


def _normalize_vec(y: float, x: float, eps: float = 1e-8) -> tuple[float, float]:
    norm = math.sqrt(y * y + x * x)
    if norm < eps:
        return 0.0, 0.0
    return y / norm, x / norm


def _mean_masked(field: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(mask.sum())
    if denom < eps:
        return 0.0
    return float((field * mask).sum() / denom)


def _policy_action_cost(action: str | None) -> float:
    if action is None or action == "no_action":
        return 0.0
    return {
        "approach": 0.012,
        "withdraw": 0.010,
        "intake": 0.014,
        "seal": 0.011,
        "reconfigure": 0.030,
    }[action]


def _risk_proxy(G_next: float, B_next: float, death_risk: float, cfg: RuntimeConfig) -> float:
    deficit = (
        cfg.risk_wG * max(0.0, cfg.G_target - G_next)
        + cfg.risk_wB * max(0.0, cfg.B_target - B_next)
        + cfg.risk_wD * float(death_risk)
    )
    homeostatic = cfg.risk_wHomeostasis * (
        abs(G_next - cfg.G_target) + abs(B_next - cfg.B_target)
    )
    reserve = cfg.risk_wReserve * (
        max(0.0, cfg.reserve_G - (G_next - cfg.tau_G))
        + max(0.0, cfg.reserve_B - (B_next - cfg.tau_B))
    )
    return float(deficit + homeostatic + reserve)


def _contact_risk_proxy(
    current_contact: dict[str, float],
    next_contact: dict[str, float],
    cfg: RuntimeConfig,
) -> float:
    absolute = (
        cfg.contact_w_thermal * next_contact["thermal"]
        + cfg.contact_w_toxicity * next_contact["toxicity"]
        - cfg.contact_w_energy * next_contact["energy"]
        - cfg.contact_w_niche * next_contact["niche"]
    )
    delta = (
        cfg.contact_delta_w_thermal * max(0.0, next_contact["thermal"] - current_contact["thermal"])
        + cfg.contact_delta_w_toxicity * max(0.0, next_contact["toxicity"] - current_contact["toxicity"])
        - cfg.contact_delta_w_energy * max(0.0, next_contact["energy"] - current_contact["energy"])
        - cfg.contact_delta_w_niche * max(0.0, next_contact["niche"] - current_contact["niche"])
    )
    return float(absolute + delta)


def _copy_body(body: BodyState) -> BodyState:
    return BodyState(**asdict(body))


class LeniaERIEEnvironment:
    def __init__(
        self,
        seed: LeniaSeed,
        env_config: EnvironmentConfig,
        runtime_config: RuntimeConfig,
        rng: np.random.Generator,
    ) -> None:
        self.seed = seed
        self.env_config = env_config
        self.runtime_config = runtime_config
        self.rng = rng

        scalar_seed = center_seed_on_canvas(rle2arr_2d(seed.cells_rle), env_config.image_size)
        self.params = sample_params(rng, self._rollout_like_config(env_config, runtime_config), seed.params)
        kernel = build_kernel(env_config.image_size, env_config.target_radius, self.params["b"])
        self.kernel_fft = np.fft.fft2(kernel)
        self.species_params = self._build_species_params()
        self.species_kernel_fft = {
            name: np.fft.fft2(build_kernel(env_config.image_size, env_config.target_radius, params["b"]))
            for name, params in self.species_params.items()
        }
        energy_gradient = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.energy_gradient_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
        )
        thermal_stress = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.thermal_stress_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
        )
        toxicity = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.toxicity_patches,
            env_config.field_sigma_min * 0.8,
            env_config.field_sigma_max * 0.9,
        )
        niche_stability = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.niche_stability_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
        )
        toxicity = _clip01(0.65 * toxicity + 0.35 * thermal_stress)
        niche_stability = _clip01(niche_stability * (1.0 - 0.35 * thermal_stress))
        species_energy = self._initialize_species_state(base=scalar_seed, shift_y=-6, shift_x=4, scale=0.85)
        species_toxic = self._initialize_species_state(base=scalar_seed, shift_y=5, shift_x=-5, scale=0.75)
        species_niche = self._initialize_species_state(base=scalar_seed, shift_y=3, shift_x=6, scale=0.70)
        species_fields = self._species_field_contributions(species_energy, species_toxic, species_niche)
        energy_gradient = _clip01(
            0.82 * energy_gradient + env_config.species_field_gain_energy * species_fields["energy"]
        )
        thermal_stress = _clip01(
            0.78 * thermal_stress + env_config.species_field_gain_thermal * species_fields["thermal"]
        )
        toxicity = _clip01(
            0.74 * toxicity + env_config.species_field_gain_toxicity * species_fields["toxicity"]
        )
        niche_stability = _clip01(
            0.80 * niche_stability
            + env_config.species_field_gain_niche * species_fields["niche"]
            - 0.08 * species_fields["thermal"]
        )
        flow_y, flow_x = _build_flow_field(
            energy_gradient,
            thermal_stress,
            niche_stability,
            env_config.flow_strength,
        )
        base_scalar = scalar_seed.astype(np.float32)
        self.external_state = ExternalState(
            scalar_state=base_scalar,
            prev_scalar_state=base_scalar.copy(),
            species_energy_state=species_energy,
            species_toxic_state=species_toxic,
            species_niche_state=species_niche,
            energy_gradient=energy_gradient,
            thermal_stress=thermal_stress,
            toxicity=toxicity,
            niche_stability=niche_stability,
            flow_y=flow_y,
            flow_x=flow_x,
        )

    @staticmethod
    def _rollout_like_config(env_config: EnvironmentConfig, runtime_config: RuntimeConfig):
        class _Config:
            image_size = env_config.image_size
            target_radius = env_config.target_radius
            mu_min = 0.23
            mu_max = 0.41
            sigma_min = 0.033
            sigma_max = 0.080
            center_mu_min = 0.27
            center_mu_max = 0.38
            center_sigma_min = 0.039
            center_sigma_max = 0.067
            center_sampling_ratio = 0.7

        return _Config()

    def _build_species_params(self) -> dict[str, dict[str, Any]]:
        def _shift(base: dict[str, Any], *, dm: float, ds: float) -> dict[str, Any]:
            return {
                **base,
                "m": float(np.clip(float(base["m"]) + dm, 0.15, 0.45)),
                "s": float(np.clip(float(base["s"]) + ds, 0.025, 0.09)),
            }

        return {
            "species_energy": _shift(self.params, dm=-0.015, ds=0.004),
            "species_toxic": _shift(self.params, dm=0.020, ds=0.010),
            "species_niche": _shift(self.params, dm=-0.005, ds=-0.002),
        }

    def set_lenia_params(self, *, m: float | None = None, s: float | None = None) -> None:
        if m is not None:
            self.params["m"] = float(np.clip(float(m), 0.15, 0.45))
        if s is not None:
            self.params["s"] = float(np.clip(float(s), 0.020, 0.100))
        self.kernel_fft = np.fft.fft2(
            build_kernel(self.env_config.image_size, self.env_config.target_radius, self.params["b"])
        )
        self.species_params = self._build_species_params()
        self.species_kernel_fft = {
            name: np.fft.fft2(build_kernel(self.env_config.image_size, self.env_config.target_radius, params["b"]))
            for name, params in self.species_params.items()
        }

    def _initialize_species_state(self, base: np.ndarray, shift_y: int, shift_x: int, scale: float) -> np.ndarray:
        shifted = np.roll(np.roll(base.astype(np.float32), shift_y, axis=0), shift_x, axis=1)
        return _clip01(scale * shifted)

    def _species_field_contributions(
        self,
        species_energy: np.ndarray,
        species_toxic: np.ndarray,
        species_niche: np.ndarray,
    ) -> dict[str, np.ndarray]:
        e = _blur_field(species_energy, rounds=2)
        t = _blur_field(species_toxic, rounds=2)
        n = _blur_field(species_niche, rounds=2)
        return {
            "energy": _clip01(e),
            "thermal": _clip01(0.65 * t),
            "toxicity": _clip01(0.85 * t),
            "niche": _clip01(n),
        }

    def species_field_channels(self) -> np.ndarray:
        species_fields = self._species_field_contributions(
            self.external_state.species_energy_state,
            self.external_state.species_toxic_state,
            self.external_state.species_niche_state,
        )
        return np.stack(
            [
                species_fields["energy"],
                species_fields["thermal"],
                species_fields["toxicity"],
                species_fields["niche"],
            ],
            axis=-1,
        ).astype(np.float32)

    def lenia_multistate(self) -> np.ndarray:
        return self.external_state.lenia_multistate(self.params)

    @property
    def scalar_state(self) -> np.ndarray:
        return self.external_state.scalar_state

    @scalar_state.setter
    def scalar_state(self, value: np.ndarray) -> None:
        self.external_state.scalar_state = value.astype(np.float32)

    @property
    def prev_scalar_state(self) -> np.ndarray:
        return self.external_state.prev_scalar_state

    @prev_scalar_state.setter
    def prev_scalar_state(self, value: np.ndarray) -> None:
        self.external_state.prev_scalar_state = value.astype(np.float32)

    @property
    def species_energy_state(self) -> np.ndarray:
        return self.external_state.species_energy_state

    @species_energy_state.setter
    def species_energy_state(self, value: np.ndarray) -> None:
        self.external_state.species_energy_state = _clip01(value)

    @property
    def species_toxic_state(self) -> np.ndarray:
        return self.external_state.species_toxic_state

    @species_toxic_state.setter
    def species_toxic_state(self, value: np.ndarray) -> None:
        self.external_state.species_toxic_state = _clip01(value)

    @property
    def species_niche_state(self) -> np.ndarray:
        return self.external_state.species_niche_state

    @species_niche_state.setter
    def species_niche_state(self, value: np.ndarray) -> None:
        self.external_state.species_niche_state = _clip01(value)

    @property
    def energy_gradient(self) -> np.ndarray:
        return self.external_state.energy_gradient

    @energy_gradient.setter
    def energy_gradient(self, value: np.ndarray) -> None:
        self.external_state.energy_gradient = _clip01(value)

    @property
    def thermal_stress(self) -> np.ndarray:
        return self.external_state.thermal_stress

    @thermal_stress.setter
    def thermal_stress(self, value: np.ndarray) -> None:
        self.external_state.thermal_stress = _clip01(value)

    @property
    def toxicity(self) -> np.ndarray:
        return self.external_state.toxicity

    @toxicity.setter
    def toxicity(self, value: np.ndarray) -> None:
        self.external_state.toxicity = _clip01(value)

    @property
    def niche_stability(self) -> np.ndarray:
        return self.external_state.niche_stability

    @niche_stability.setter
    def niche_stability(self, value: np.ndarray) -> None:
        self.external_state.niche_stability = _clip01(value)

    @property
    def resource(self) -> np.ndarray:
        return self.energy_gradient

    @resource.setter
    def resource(self, value: np.ndarray) -> None:
        self.energy_gradient = _clip01(value)

    @property
    def hazard(self) -> np.ndarray:
        return _clip01(0.6 * self.thermal_stress + 0.4 * self.toxicity)

    @hazard.setter
    def hazard(self, value: np.ndarray) -> None:
        clipped = _clip01(value)
        self.thermal_stress = clipped.copy()
        self.toxicity = clipped.copy()

    @property
    def shelter(self) -> np.ndarray:
        return self.niche_stability

    @shelter.setter
    def shelter(self, value: np.ndarray) -> None:
        self.niche_stability = _clip01(value)

    @property
    def flow_y(self) -> np.ndarray:
        return self.external_state.flow_y

    @flow_y.setter
    def flow_y(self, value: np.ndarray) -> None:
        self.external_state.flow_y = np.clip(value, -1.0, 1.0).astype(np.float32)

    @property
    def flow_x(self) -> np.ndarray:
        return self.external_state.flow_x

    @flow_x.setter
    def flow_x(self, value: np.ndarray) -> None:
        self.external_state.flow_x = np.clip(value, -1.0, 1.0).astype(np.float32)

    def step_lenia(self) -> None:
        next_state = lenia_step(self.scalar_state, self.kernel_fft, self.params)
        self.prev_scalar_state = self.scalar_state
        self.scalar_state = next_state
        self.external_state.species_energy_state = lenia_step(
            self.external_state.species_energy_state,
            self.species_kernel_fft["species_energy"],
            self.species_params["species_energy"],
        ).astype(np.float32)
        self.external_state.species_toxic_state = lenia_step(
            self.external_state.species_toxic_state,
            self.species_kernel_fft["species_toxic"],
            self.species_params["species_toxic"],
        ).astype(np.float32)
        self.external_state.species_niche_state = lenia_step(
            self.external_state.species_niche_state,
            self.species_kernel_fft["species_niche"],
            self.species_params["species_niche"],
        ).astype(np.float32)

    def environment_channels(self) -> np.ndarray:
        return self.external_state.as_channels(self.params)

    def external_channels(self) -> np.ndarray:
        return self.external_state.as_external_channels(self.params, self.species_params)

    def update_fields(self, body: BodyState, action: str) -> None:
        _, boundary, permeability = _body_fields(
            body, self.env_config.image_size, self.runtime_config.occupancy_softness
        )
        contact_mask = boundary * np.clip(permeability, 0.0, 1.0)
        if action == "intake":
            consume = np.minimum(self.energy_gradient, 0.06 * contact_mask)
            self.energy_gradient = _clip01(self.energy_gradient - consume)
        self.energy_gradient = _clip01(
            self.energy_gradient + self.env_config.resource_regen * (1.0 - self.energy_gradient)
        )
        self.energy_gradient = _clip01(
            0.82 * self.energy_gradient
            + 0.18 * _advect_field(self.energy_gradient, self.flow_y, self.flow_x, scale=1.0)
        )
        self.thermal_stress = _clip01(
            self.thermal_stress
            + gaussian_noise(
                self.rng,
                self.thermal_stress.shape,
                self.env_config.hazard_drift_sigma,
            )
        )
        self.thermal_stress = _clip01(
            0.90 * self.thermal_stress
            + 0.10 * _advect_field(self.thermal_stress, self.flow_y, self.flow_x, scale=0.6)
        )
        self.toxicity = _clip01(
            self.toxicity
            + gaussian_noise(
                self.rng,
                self.toxicity.shape,
                self.env_config.toxicity_drift_sigma,
            )
        )
        self.toxicity = _clip01(
            0.78 * self.toxicity
            + 0.22 * _advect_field(self.toxicity, self.flow_y, self.flow_x, scale=1.1)
        )
        self.niche_stability = _clip01(
            0.92 * self.niche_stability
            + 0.08 * _advect_field(self.niche_stability, self.flow_y, self.flow_x, scale=0.5)
        )
        species_fields = self._species_field_contributions(
            self.external_state.species_energy_state,
            self.external_state.species_toxic_state,
            self.external_state.species_niche_state,
        )
        self.energy_gradient = _clip01(
            0.88 * self.energy_gradient + self.env_config.species_field_gain_energy * species_fields["energy"]
        )
        self.thermal_stress = _clip01(
            0.90 * self.thermal_stress + self.env_config.species_field_gain_thermal * species_fields["thermal"]
        )
        self.toxicity = _clip01(
            0.86 * self.toxicity + self.env_config.species_field_gain_toxicity * species_fields["toxicity"]
        )
        self.niche_stability = _clip01(
            0.90 * self.niche_stability
            + self.env_config.species_field_gain_niche * species_fields["niche"]
            - 0.05 * species_fields["thermal"]
        )
        flow_noise_y = gaussian_noise(self.rng, self.flow_y.shape, self.env_config.flow_drift_sigma)
        flow_noise_x = gaussian_noise(self.rng, self.flow_x.shape, self.env_config.flow_drift_sigma)
        flow_y, flow_x = _build_flow_field(
            self.energy_gradient,
            self.thermal_stress,
            self.niche_stability,
            self.env_config.flow_strength,
        )
        self.flow_y = np.clip(0.92 * self.flow_y + 0.08 * flow_y + flow_noise_y, -1.0, 1.0)
        self.flow_x = np.clip(0.92 * self.flow_x + 0.08 * flow_x + flow_noise_x, -1.0, 1.0)

    def advance_external_state(self, body: BodyState, action: str) -> ExternalState:
        self.step_lenia()
        self.update_fields(body, action)
        return self.external_state


class ERIERuntime:
    def __init__(
        self,
        environment: LeniaERIEEnvironment,
        runtime_config: RuntimeConfig,
        rng: np.random.Generator,
        models: RuntimeModels | None = None,
    ) -> None:
        self.env = environment
        self.cfg = runtime_config
        self.rng = rng
        self.models = models or RuntimeModels(None, None)
        center = float(self.env.env_config.image_size) / 2.0
        self.body = BodyState(
            centroid_y=center,
            centroid_x=center,
            radius=runtime_config.occupancy_radius,
            aperture_angle=0.0,
            aperture_gain=runtime_config.aperture_gain,
            aperture_width_deg=runtime_config.aperture_width_deg,
            G=runtime_config.G0,
            B=runtime_config.B0,
            energy=runtime_config.G0,
            body_id=0,
        )
        self.bodies: list[BodyState] = [self.body]
        self._next_body_id: int = 1
        self.population_event_counts: dict[str, int] = {
            "spawn": 0,
            "split": 0,
            "death": 0,
            "role_switch": 0,
        }
        self.challenge_ratio: float = 0.5
        self._role_by_body_id: dict[int, str] = {}
        channels = self.env.environment_channels().shape[-1]
        shape_world = (self.env.env_config.image_size, self.env.env_config.image_size, channels)
        self.world_cells = PredictionCellGrid.zeros(
            shape_world,
            logvar_init=runtime_config.world_logvar_init,
        )
        self.world_belief = self.world_cells.belief
        self.world_logvar = self.world_cells.logvar
        self.boundary_cells = PredictionCellGrid.zeros(
            (self.env.env_config.image_size, self.env.env_config.image_size, 2),
            logvar_init=runtime_config.boundary_logvar_init,
        )
        self.boundary_belief = self.boundary_cells.belief
        self.boundary_logvar = self.boundary_cells.logvar
        self.visual_cells = LocalPredictionCellLayer.from_image_shape(
            shape_world,
            cell_rows=max(1, min(int(runtime_config.visual_cell_rows), int(self.env.env_config.image_size))),
            cell_cols=max(1, min(int(runtime_config.visual_cell_cols), int(self.env.env_config.image_size))),
            logvar_init=runtime_config.world_logvar_init,
        )
        self.policy_belief = np.full(len(ACTIONS), 1.0 / len(ACTIONS), dtype=np.float32)
        self.history: list[dict[str, Any]] = []
        self.prev_lenia_obs = self.env.lenia_multistate().astype(np.float32)
        self.trace_field = np.zeros((self.env.env_config.image_size, self.env.env_config.image_size), dtype=np.float32)
        self.next_world_prior_lenia: np.ndarray | None = None
        self.next_world_logvar_lenia: np.ndarray | None = None
        self.last_observation = np.zeros(shape_world, dtype=np.float32)
        self.last_sensor_gate = np.zeros(
            (self.env.env_config.image_size, self.env.env_config.image_size, 1), dtype=np.float32
        )
        self.last_world_error = np.zeros(shape_world, dtype=np.float32)
        self.last_boundary_error = np.zeros_like(self.boundary_belief, dtype=np.float32)
        self.last_vfe: dict[str, float] = {
            "world_reconstruction": 0.0,
            "world_complexity": 0.0,
            "world": 0.0,
            "boundary_reconstruction": 0.0,
            "boundary_complexity": 0.0,
            "boundary": 0.0,
            "total": 0.0,
        }
        self.last_visual_cell_update: dict[str, float] = {
            "error_mean": 0.0,
            "belief_mean": 0.0,
            "precision_mean": 1.0,
            "vfe": 0.0,
        }
        self.visual_attention_map = np.zeros(self.visual_cells.belief.shape[:2], dtype=np.float32)
        self.visual_attention_projected = np.zeros(
            (self.env.env_config.image_size, self.env.env_config.image_size),
            dtype=np.float32,
        )
        self.last_visual_attention: dict[str, float | int] = {
            "target_row": 0,
            "target_col": 0,
            "target_y": center,
            "target_x": center,
            "max": 0.0,
            "entropy": 0.0,
            "salience_mean": 0.0,
        }
        self.last_bp_control: dict[str, Any] = {
            "source": "analytic",
            "model_precision": 1.0,
            "pred_interface_gain": 0.0,
            "pred_aperture_gain": float(self.body.aperture_gain),
            "pred_mode": -1,
        }
        self.mc_feature_history: list[np.ndarray] = []
        self.last_mc_context: dict[str, Any] = {
            "source": "analytic",
            "model_precision": 1.0,
            "window_length": 0,
            "sequence_bias": np.zeros(len(ACTIONS), dtype=np.float32),
            "boundary_control_bias": np.zeros(3, dtype=np.float32),
            "context_state": np.zeros(32, dtype=np.float32),
            "retrieved_context": np.zeros(28, dtype=np.float32),
        }
        self.last_death_cause: str | None = None
        self.last_death_signals: dict[str, bool] = {
            "threshold_violation": False,
            "nonfinite_state": False,
            "action_lock": False,
            "policy_forbidden_window": False,
            "invalid_body_state": False,
        }
        self.invalid_body_state_count: int = 0
        self.boundary_interface_counts: dict[str, int] = {
            "observe_calls": 0,
            "action_calls": 0,
            "direct_observe_calls": 0,
            "direct_action_calls": 0,
        }
        self._refresh_body_phenotype(self.body)
        self._update_activity_distribution_and_roles()

    def _body_fields(self, body: BodyState | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _body_fields(
            body or self.body,
            self.env.env_config.image_size,
            self.cfg.occupancy_softness,
        )

    def _update_visual_attention(self, error: np.ndarray, precision: np.ndarray) -> dict[str, float | int]:
        error_arr = np.asarray(error, dtype=np.float32)
        precision_arr = np.asarray(precision, dtype=np.float32)
        if error_arr.shape != self.visual_cells.belief.shape:
            raise ValueError("visual attention error shape must match visual cell shape")
        if precision_arr.shape != self.visual_cells.belief.shape:
            raise ValueError("visual attention precision shape must match visual cell shape")
        error_salience = np.mean(np.abs(error_arr) * precision_arr, axis=-1)
        uncertainty_salience = np.mean(np.exp(np.clip(self.visual_cells.logvar, -6.0, 2.0)), axis=-1)
        salience = (
            float(self.cfg.visual_attention_error_gain) * error_salience
            + float(self.cfg.visual_attention_uncertainty_gain) * uncertainty_salience
        ).astype(np.float32)
        if not np.isfinite(salience).all():
            salience = np.nan_to_num(salience, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        logits = (float(self.cfg.visual_attention_temperature) * (salience - float(np.max(salience)))).reshape(-1)
        attention = _softmax(logits).reshape(salience.shape).astype(np.float32)
        flat_index = int(np.argmax(attention))
        target_row, target_col = np.unravel_index(flat_index, attention.shape)
        y_slice, x_slice = self.visual_cells.topology.receptive_slice(int(target_row), int(target_col))
        target_y = 0.5 * (float(y_slice.start) + float(y_slice.stop - 1))
        target_x = 0.5 * (float(x_slice.start) + float(x_slice.stop - 1))
        entropy = _entropy(attention.reshape(-1)) / math.log(max(int(attention.size), 2))
        self.visual_attention_map = attention.astype(np.float32)
        self.visual_attention_projected = self.visual_cells.topology.expand_cells(attention[..., None])[..., 0].astype(
            np.float32
        )
        self.last_visual_attention = {
            "target_row": int(target_row),
            "target_col": int(target_col),
            "target_y": float(target_y),
            "target_x": float(target_x),
            "max": float(np.max(attention)),
            "entropy": float(entropy),
            "salience_mean": float(np.mean(salience)),
        }
        return dict(self.last_visual_attention)

    def _alive_bodies(self) -> list[BodyState]:
        return alive_bodies(self.bodies)

    def _select_primary_body(self) -> None:
        selected = select_primary_body(self.bodies)
        if selected is not None:
            self.body = selected

    def _population_body_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_size = int(self.env.env_config.image_size)
        occupancy = np.zeros((image_size, image_size), dtype=np.float32)
        boundary = np.zeros((image_size, image_size), dtype=np.float32)
        permeability = np.zeros((image_size, image_size), dtype=np.float32)
        for body in self._alive_bodies():
            body_occ, body_boundary, body_perm = self._body_fields(body)
            occupancy = np.maximum(occupancy, body_occ)
            boundary = np.maximum(boundary, body_boundary)
            permeability = np.maximum(permeability, body_perm)
        if float(occupancy.max()) <= 0.0:
            return self._body_fields(self.body)
        return occupancy.astype(np.float32), boundary.astype(np.float32), permeability.astype(np.float32)

    def _refresh_body_phenotype(self, body: BodyState) -> None:
        occupancy, boundary, _ = self._body_fields(body)
        reference_area = math.pi * max(float(body.radius), 1e-6) ** 2
        normalized_mass = float(np.clip(float(occupancy.sum()) / max(reference_area, 1e-6), 0.0, 4.0))
        boundary_proxy = float(np.clip(0.7 * float(body.B) + 0.3 * float(boundary.mean()), 0.0, 1.0))
        body.energy = float(body.G)
        body.mass = normalized_mass
        body.boundary_integrity = boundary_proxy
        body.alive = bool(body.dead_count < self.cfg.k_irrev)

    def _body_invariant_signals(self, body: BodyState) -> dict[str, bool]:
        image_size = int(self.env.env_config.image_size)
        finite_core = np.isfinite(
            np.array(
                [
                    body.centroid_y,
                    body.centroid_x,
                    body.radius,
                    body.aperture_angle,
                    body.aperture_gain,
                    body.aperture_width_deg,
                    body.G,
                    body.B,
                    body.energy,
                    body.mass,
                    body.boundary_integrity,
                ],
                dtype=np.float64,
            )
        )
        nonfinite_state = bool(not bool(np.all(finite_core)))
        centroid_out_of_bounds = bool(
            body.centroid_y < 0.0
            or body.centroid_y > (image_size - 1)
            or body.centroid_x < 0.0
            or body.centroid_x > (image_size - 1)
        )
        nonpositive_radius = bool(body.radius <= 0.0)
        invalid_vital_range = bool(body.G < 0.0 or body.G > 1.0 or body.B < 0.0 or body.B > 1.0)
        invalid_energy = bool(body.energy < 0.0 or body.energy > 1.0)
        invalid_mass = bool(body.mass <= 0.0)
        invalid_boundary_integrity = bool(body.boundary_integrity < 0.0 or body.boundary_integrity > 1.0)
        invalid_aperture = bool(body.aperture_gain < 0.0 or body.aperture_gain > 1.0)
        invalid_body_state = bool(
            nonfinite_state
            or centroid_out_of_bounds
            or nonpositive_radius
            or invalid_vital_range
            or invalid_energy
            or invalid_mass
            or invalid_boundary_integrity
            or invalid_aperture
        )
        return {
            "nonfinite_state": nonfinite_state,
            "centroid_out_of_bounds": centroid_out_of_bounds,
            "nonpositive_radius": nonpositive_radius,
            "invalid_vital_range": invalid_vital_range,
            "invalid_energy": invalid_energy,
            "invalid_mass": invalid_mass,
            "invalid_boundary_integrity": invalid_boundary_integrity,
            "invalid_aperture": invalid_aperture,
            "invalid_body_state": invalid_body_state,
        }

    def _observe_via_boundary_interface(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.boundary_interface_counts["observe_calls"] += 1
        return self._observe()

    def _enact_action_via_boundary_interface(
        self,
        action: str | None,
        context_bias: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.boundary_interface_counts["action_calls"] += 1
        bp_meta = self._apply_action(action, context_bias=context_bias)
        self.env.update_fields(self.body, action)
        return bp_meta

    def _update_trace_field(self, dead: bool, dead_bodies: list[BodyState] | None = None) -> None:
        _, boundary, permeability = self._population_body_fields()
        interface = np.clip(boundary * permeability, 0.0, 1.0)
        extra_deaths = len(dead_bodies or [])
        deposit_strength = 0.015 + (0.035 if dead else 0.0) + (0.012 * extra_deaths)
        deposited = self.trace_field + deposit_strength * interface
        decayed = 0.992 * deposited
        diffused = 0.94 * decayed + 0.06 * _blur_field(decayed, rounds=1)
        self.trace_field = _clip01(diffused)

    def _trace_density(self, body: BodyState) -> float:
        _, boundary, permeability = self._body_fields(body)
        interface = np.clip(boundary * permeability, 0.0, 1.0)
        return float(_mean_masked(self.trace_field, interface))

    def _spawn_drive(self, body: BodyState, contact: dict[str, float], *, trace_density: float | None = None) -> float:
        trace_term = float(self._trace_density(body) if trace_density is None else trace_density)
        return population_spawn_drive(
            trace_term=trace_term,
            resource=float(contact.get("resource", 0.0)),
            hazard=float(contact.get("hazard", 0.0)),
            G=float(body.G),
            B=float(body.B),
            tau_G=float(self.cfg.tau_G),
            tau_B=float(self.cfg.tau_B),
            spawn_logit_bias=float(self.cfg.spawn_logit_bias),
            spawn_trace_gain=float(self.cfg.spawn_trace_gain),
            spawn_resource_gain=float(self.cfg.spawn_resource_gain),
            spawn_hazard_penalty=float(self.cfg.spawn_hazard_penalty),
            spawn_viability_gain=float(self.cfg.spawn_viability_gain),
        )

    def _split_drive(self, body: BodyState) -> float:
        return population_split_drive(
            mass=float(body.mass),
            energy=float(body.energy),
            boundary_integrity=float(body.boundary_integrity),
            split_logit_bias=float(self.cfg.split_logit_bias),
            split_mass_gain=float(self.cfg.split_mass_gain),
            split_energy_gain=float(self.cfg.split_energy_gain),
            split_boundary_penalty=float(self.cfg.split_boundary_penalty),
        )

    def _spawn_split_signals(self, body: BodyState) -> dict[str, float | bool]:
        contact = self._contact_stats(body)
        trace_density = self._trace_density(body)
        return population_spawn_split_signals(
            trace_density=float(trace_density),
            resource=float(contact.get("resource", 0.0)),
            hazard=float(contact.get("hazard", 0.0)),
            G=float(body.G),
            B=float(body.B),
            tau_G=float(self.cfg.tau_G),
            tau_B=float(self.cfg.tau_B),
            mass=float(body.mass),
            energy=float(body.energy),
            boundary_integrity=float(body.boundary_integrity),
            spawn_logit_bias=float(self.cfg.spawn_logit_bias),
            spawn_trace_gain=float(self.cfg.spawn_trace_gain),
            spawn_resource_gain=float(self.cfg.spawn_resource_gain),
            spawn_hazard_penalty=float(self.cfg.spawn_hazard_penalty),
            spawn_viability_gain=float(self.cfg.spawn_viability_gain),
            split_logit_bias=float(self.cfg.split_logit_bias),
            split_mass_gain=float(self.cfg.split_mass_gain),
            split_energy_gain=float(self.cfg.split_energy_gain),
            split_boundary_penalty=float(self.cfg.split_boundary_penalty),
            spawn_candidate_threshold=float(self.cfg.spawn_candidate_threshold),
            split_candidate_threshold=float(self.cfg.split_candidate_threshold),
        )

    def _activity_ratio_logit(
        self,
        buffer_t: float,
        hazard_t: float,
        boundary_damage_t: float,
    ) -> float:
        return float(
            self.cfg.activity_a0
            + self.cfg.activity_a1 * buffer_t
            - self.cfg.activity_a2 * hazard_t
            - self.cfg.activity_a3 * boundary_damage_t
        )

    def _role_action_bias(self, role: str) -> np.ndarray:
        gain = float(max(0.0, self.cfg.role_action_bias_gain))
        if gain <= 0.0:
            return np.zeros((len(ACTIONS),), dtype=np.float32)
        if role == "challenge":
            base = np.array([0.32, -0.08, 0.26, -0.10, 0.18], dtype=np.float32)
        else:
            base = np.array([-0.16, 0.22, -0.12, 0.24, -0.04], dtype=np.float32)
        return (gain * base).astype(np.float32)

    def _update_activity_distribution_and_roles(self) -> dict[str, float | int]:
        alive_bodies = self._alive_bodies()
        if not alive_bodies:
            self.challenge_ratio = 0.0
            self._role_by_body_id = {}
            return {
                "p_t": 0.0,
                "buffer_t": 0.0,
                "hazard_t": 0.0,
                "boundary_damage_t": 0.0,
                "challenge_body_count": 0,
                "conservative_body_count": 0,
                "role_switch_events_step": 0,
            }

        buffer_values: list[float] = []
        hazard_values: list[float] = []
        boundary_damage_values: list[float] = []
        role_scores: list[tuple[float, BodyState]] = []
        for body in alive_bodies:
            self._refresh_body_phenotype(body)
            contact = self._contact_stats(body)
            local_hazard = float(contact.get("hazard", 0.0))
            body.local_hazard = local_hazard
            boundary_damage = float(max(0.0, 1.0 - body.boundary_integrity))
            conf = float(np.clip(1.0 / (1.0 + self._ambiguity_proxy(body)), 0.0, 1.0))
            body.prediction_confidence = conf
            reserve = max(0.0, float(body.G - self.cfg.tau_G)) + max(0.0, float(body.B - self.cfg.tau_B))
            score = (
                self.cfg.role_score_w_energy * float(body.energy)
                + self.cfg.role_score_w_confidence * conf
                - self.cfg.role_score_w_hazard * local_hazard
                - self.cfg.role_score_w_boundary * boundary_damage
                + 0.5 * reserve
            )
            role_scores.append((float(score), body))
            buffer_values.append(float(reserve))
            hazard_values.append(local_hazard)
            boundary_damage_values.append(boundary_damage)

        buffer_t = float(np.mean(np.array(buffer_values, dtype=np.float32)))
        hazard_t = float(np.mean(np.array(hazard_values, dtype=np.float32)))
        boundary_damage_t = float(np.mean(np.array(boundary_damage_values, dtype=np.float32)))
        ratio_logit = self._activity_ratio_logit(buffer_t, hazard_t, boundary_damage_t)
        p_t_raw = _sigmoid_scalar(ratio_logit)
        p_t = float(np.clip(p_t_raw, self.cfg.challenge_ratio_min, self.cfg.challenge_ratio_max))
        self.challenge_ratio = p_t

        role_scores.sort(key=lambda item: item[0], reverse=True)
        alive_count = len(alive_bodies)
        challenge_count = int(round(p_t * alive_count))
        challenge_count = int(np.clip(challenge_count, 0, alive_count))
        if alive_count >= 2:
            challenge_count = int(np.clip(challenge_count, 1, alive_count - 1))
        challenge_ids = {id(body) for _, body in role_scores[:challenge_count]}

        prev_roles = dict(self._role_by_body_id)
        role_switch_events = 0
        for body in alive_bodies:
            body.role = "challenge" if id(body) in challenge_ids else "conservative"
            previous_role = prev_roles.get(int(body.body_id))
            if previous_role is not None and previous_role != body.role:
                role_switch_events += 1
        self._role_by_body_id = {int(body.body_id): str(body.role) for body in alive_bodies}
        if role_switch_events > 0:
            self.population_event_counts["role_switch"] += int(role_switch_events)

        return {
            "p_t": float(p_t),
            "buffer_t": float(buffer_t),
            "hazard_t": float(hazard_t),
            "boundary_damage_t": float(boundary_damage_t),
            "challenge_body_count": int(challenge_count),
            "conservative_body_count": int(alive_count - challenge_count),
            "role_switch_events_step": int(role_switch_events),
        }

    def _next_body_identifier(self) -> int:
        body_id = int(self._next_body_id)
        self._next_body_id += 1
        return body_id

    def _can_expand_population(self) -> bool:
        return can_expand_population(self.bodies, int(self.cfg.max_bodies))

    def _spawn_from_primary(self) -> BodyState | None:
        parent = self.body
        child = spawn_child_from_primary(
            parent,
            can_expand=self._can_expand_population(),
            tau_G=float(self.cfg.tau_G),
            tau_B=float(self.cfg.tau_B),
            spawn_energy_share=float(self.cfg.spawn_energy_share),
            spawn_offset=float(self.cfg.spawn_offset),
            spawn_radius_scale=float(self.cfg.spawn_radius_scale),
            image_size=int(self.env.env_config.image_size),
            copy_body=_copy_body,
            next_body_id=self._next_body_identifier,
        )
        if child is None:
            return None
        self._refresh_body_phenotype(parent)
        self._refresh_body_phenotype(child)
        self.bodies.append(child)
        return child

    def _split_primary(self) -> BodyState | None:
        parent = self.body
        child = split_child_from_primary(
            parent,
            can_expand=self._can_expand_population(),
            tau_G=float(self.cfg.tau_G),
            split_energy_share=float(self.cfg.split_energy_share),
            split_radius_scale=float(self.cfg.split_radius_scale),
            spawn_offset=float(self.cfg.spawn_offset),
            image_size=int(self.env.env_config.image_size),
            copy_body=_copy_body,
            next_body_id=self._next_body_identifier,
        )
        if child is None:
            return None
        self._refresh_body_phenotype(parent)
        self._refresh_body_phenotype(child)
        self.bodies.append(child)
        return child

    def _apply_flow_drift_to_body(self, body: BodyState, scale: float = 1.0) -> None:
        yi = int(np.clip(round(float(body.centroid_y)), 0, self.env.flow_y.shape[0] - 1))
        xi = int(np.clip(round(float(body.centroid_x)), 0, self.env.flow_x.shape[1] - 1))
        flow_gain = float(self.cfg.passive_body_flow_gain) * float(scale)
        body.centroid_y = float(
            np.clip(
                body.centroid_y + flow_gain * float(self.env.flow_y[yi, xi]),
                4.0,
                self.env.env_config.image_size - 5.0,
            )
        )
        body.centroid_x = float(
            np.clip(
                body.centroid_x + flow_gain * float(self.env.flow_x[yi, xi]),
                4.0,
                self.env.env_config.image_size - 5.0,
            )
        )

    def _full_policy_action_for_body(self, body: BodyState) -> tuple[str | None, dict[str, Any]]:
        original_body = self.body
        try:
            self.body = body
            scores, score_diag = self._policy_scores()
            viability_monitor = self._monitor_viability(action_cost=0.0)
            policy, selected_action, policy_meta = self._select_policy(scores, score_diag, viability_monitor)
            if self.cfg.policy_mode == "random":
                action = str(self.rng.choice(ACTIONS))
            elif self.cfg.policy_mode == "no_action":
                action = "no_action"
            else:
                action = selected_action
            return action, {
                "source": "full_policy",
                "policy_entropy": float(_entropy(policy)),
                "selected_action": str(selected_action),
                "policy_source": str(policy_meta.get("source", "analytic")),
            }
        finally:
            self.body = original_body

    def _auxiliary_policy_decision(self, body: BodyState) -> tuple[str | None, dict[str, Any]]:
        mode = str(self.cfg.aux_policy_mode)
        if mode == "passive":
            return None, {"source": "passive", "policy_entropy": 0.0}
        if mode == "full_policy":
            return self._full_policy_action_for_body(body)
        meta = {"source": "role_heuristic", "policy_entropy": 0.0}
        contact = self._contact_stats(body)
        hazard = float(contact.get("hazard", 0.0))
        energy = float(contact.get("energy", 0.0))
        role = str(body.role)
        if role == "challenge":
            if hazard >= 0.62:
                return "withdraw", meta
            if energy <= 0.35:
                return "approach", meta
            if body.G < self.cfg.G_target and hazard < 0.45:
                return "intake", meta
            return "reconfigure", meta
        if hazard >= 0.45 or body.B < self.cfg.B_target:
            return "seal", meta
        if hazard >= 0.35:
            return "withdraw", meta
        if body.G < self.cfg.tau_G + 0.08 and energy > 0.30:
            return "intake", meta
        return "no_action", meta

    def _auxiliary_policy_action(self, body: BodyState) -> str | None:
        action, _ = self._auxiliary_policy_decision(body)
        return action

    def _execute_action_for_body(self, body: BodyState, action: str | None) -> BodyState:
        original_body = self.body
        try:
            self.body = body
            action_name = "no_action" if action is None else str(action)
            self._enact_action_via_boundary_interface(action_name, context_bias=None)
            updated_body = self.body
            if action_name in ACTIONS:
                self._apply_flow_drift_to_body(updated_body, scale=0.35)
            else:
                self._apply_flow_drift_to_body(updated_body, scale=1.0)
            self._refresh_body_phenotype(updated_body)
            return updated_body
        finally:
            self.body = original_body

    def _update_auxiliary_bodies(self) -> tuple[list[BodyState], dict[str, Any]]:
        dead_aux: list[BodyState] = []
        action_counts: dict[str, int] = {name: 0 for name in (*ACTIONS, "no_action")}
        policy_source_counts: dict[str, int] = {
            "full_policy": 0,
            "role_heuristic": 0,
            "passive": 0,
        }
        nontrivial_action_count = 0
        challenge_action_count = 0
        conservative_action_count = 0
        updated_body_count = 0
        policy_entropy_sum = 0.0
        primary_body = self.body
        for body in list(self.bodies):
            if body is primary_body or not bool(body.alive):
                continue
            updated_body_count += 1
            action, policy_meta = self._auxiliary_policy_decision(body)
            policy_source = str(policy_meta.get("source", "role_heuristic"))
            if policy_source not in policy_source_counts:
                policy_source = "role_heuristic"
            policy_source_counts[policy_source] += 1
            policy_entropy_sum += float(policy_meta.get("policy_entropy", 0.0))
            action_name = "no_action" if action is None else str(action)
            if action_name not in action_counts:
                action_name = "no_action"
            action_counts[action_name] += 1
            if action_name != "no_action":
                nontrivial_action_count += 1
                if str(body.role) == "challenge":
                    challenge_action_count += 1
                elif str(body.role) == "conservative":
                    conservative_action_count += 1
            updated_body = body
            if policy_source == "passive":
                self._apply_flow_drift_to_body(updated_body, scale=1.0)
                updated_body.G, updated_body.B = self._predicted_viability(updated_body, None)
                self._refresh_body_phenotype(updated_body)
            else:
                updated_body = self._execute_action_for_body(body, action if action_name in ACTIONS else "no_action")
            invariant = self._body_invariant_signals(updated_body)
            invalid_body_state = bool(invariant["invalid_body_state"])
            if invalid_body_state:
                self.invalid_body_state_count += 1
            if updated_body.G < self.cfg.tau_G or updated_body.B < self.cfg.tau_B or invalid_body_state:
                updated_body.dead_count += 1
            else:
                updated_body.dead_count = 0
            if updated_body.dead_count >= self.cfg.k_irrev:
                updated_body.alive = False
                dead_aux.append(updated_body)
        mean_policy_entropy = (
            float(policy_entropy_sum / max(updated_body_count, 1))
            if updated_body_count > 0
            else 0.0
        )
        return dead_aux, {
            "updated_body_count": int(updated_body_count),
            "action_counts": action_counts,
            "policy_source_counts": policy_source_counts,
            "mean_policy_entropy": float(mean_policy_entropy),
            "nontrivial_action_count": int(nontrivial_action_count),
            "challenge_action_count": int(challenge_action_count),
            "conservative_action_count": int(conservative_action_count),
        }

    def _assemble_world_prior(self) -> tuple[np.ndarray, np.ndarray]:
        world_prior = self.world_belief.copy()
        world_logvar = self.world_logvar.copy()
        if self.models.trm_a is not None and self.next_world_prior_lenia is not None:
            world_prior[..., :5] = self.next_world_prior_lenia
        if self.models.trm_a is not None and self.next_world_logvar_lenia is not None:
            world_logvar[..., :5] = self.next_world_logvar_lenia
        return world_prior, world_logvar

    def _boundary_prior_from_model(
        self,
        lenia_obs: np.ndarray,
        world_prior_lenia: np.ndarray,
    ) -> np.ndarray | None:
        if self.models.trm_b is None:
            return None
        torch = self.models.torch
        assert torch is not None
        delta_state = (lenia_obs - self.prev_lenia_obs).astype(np.float32)
        error_map = np.abs(lenia_obs - world_prior_lenia).astype(np.float32)
        with torch.no_grad():
            state_t = torch.from_numpy(lenia_obs[None, ...])
            delta_t = torch.from_numpy(delta_state[None, ...])
            error_t = torch.from_numpy(error_map[None, ...])
            outputs = self.models.trm_b(state_t, delta_t, error_t)
        boundary_map = outputs["boundary_map"][0].cpu().numpy().astype(np.float32)
        permeability_map = outputs["permeability_map"][0].cpu().numpy().astype(np.float32)
        return np.concatenate([boundary_map, permeability_map], axis=-1).astype(np.float32)

    def _refresh_world_prior_from_trm_a(self) -> None:
        if self.models.trm_a is None:
            self.next_world_prior_lenia = None
            self.next_world_logvar_lenia = None
            return
        torch = self.models.torch
        assert torch is not None
        lenia_state = self.world_belief[..., :5].astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(lenia_state[None, ...])
            outputs = self.models.trm_a(x, use_posterior=False, sample_latent=False)
        self.next_world_prior_lenia = outputs["pred_state_t1"][0].cpu().numpy().astype(np.float32)
        self.next_world_logvar_lenia = outputs["pred_logvar_t1"][0].cpu().numpy().astype(np.float32)

    def _observation_mapping(
        self,
        env_channels: np.ndarray,
        sensor_gate: np.ndarray,
        thermal_stress: np.ndarray,
        toxicity: np.ndarray,
        niche_stability: np.ndarray,
    ) -> dict[str, np.ndarray]:
        niche_bonus = niche_stability[..., None]
        noise_scale = np.clip(
            self.cfg.observation_noise
            * (1.0 + thermal_stress[..., None] + 0.75 * toxicity[..., None] - 0.6 * niche_bonus),
            0.002,
            0.05,
        ).astype(np.float32)
        noisy = _clip01(env_channels + gaussian_noise(self.rng, env_channels.shape, 1.0) * noise_scale)
        observation = sensor_gate * noisy + (1.0 - sensor_gate) * self.world_belief
        return {
            "noisy": noisy.astype(np.float32),
            "noise_scale": noise_scale.astype(np.float32),
            "observation": observation.astype(np.float32),
        }

    def _observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        occupancy, boundary, permeability = self._population_body_fields()
        env_channels = self.env.environment_channels()
        sensor_gate = np.clip(permeability[..., None] + 0.05 * occupancy[..., None], 0.0, 1.0)
        obs = self._observation_mapping(
            env_channels=env_channels,
            sensor_gate=sensor_gate,
            thermal_stress=self.env.thermal_stress,
            toxicity=self.env.toxicity,
            niche_stability=self.env.niche_stability,
        )
        observation = obs["observation"]
        self.last_observation = observation.astype(np.float32)
        self.last_sensor_gate = sensor_gate.astype(np.float32)
        return observation.astype(np.float32), sensor_gate.astype(np.float32), occupancy, boundary

    def _belief_update(
        self,
        observation: np.ndarray,
        sensor_gate: np.ndarray,
        boundary_obs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lenia_obs = observation[..., :5].astype(np.float32)
        world_prior, world_logvar = self._assemble_world_prior()
        boundary_prior = self.boundary_belief.copy()
        model_boundary = self._boundary_prior_from_model(lenia_obs, world_prior[..., :5].astype(np.float32))
        if model_boundary is not None:
            boundary_prior = model_boundary

        world_result = self.world_cells.update(
            observation,
            PredictionCellUpdateConfig(
                learning_rate=float(self.cfg.lambda_w),
                p_min=float(self.cfg.p_min),
                p_max=float(self.cfg.p_max),
                logvar_drift=float(self.cfg.world_logvar_drift),
                evidence_logvar_gain=0.18,
                reconstruction_logvar_source="precision",
            ),
            gate=sensor_gate,
            prior=world_prior,
            precision_logvar=world_logvar,
            logvar_evidence=sensor_gate,
        )
        boundary_result = self.boundary_cells.update(
            boundary_obs,
            PredictionCellUpdateConfig(
                learning_rate=float(self.cfg.lambda_b),
                p_min=float(self.cfg.p_min),
                p_max=float(self.cfg.p_max),
                logvar_drift=float(self.cfg.boundary_logvar_drift),
                evidence_logvar_gain=0.20,
                reconstruction_logvar_source="updated",
            ),
            prior=boundary_prior,
            precision_logvar=self.boundary_logvar,
            logvar_evidence=boundary_obs[..., :1],
        )
        world_error = world_result.error
        boundary_error = boundary_result.error
        world_reconstruction = float(world_result.reconstruction)
        world_complexity = float(world_result.complexity)
        boundary_reconstruction = float(boundary_result.reconstruction)
        boundary_complexity = float(boundary_result.complexity)
        world_total = float(world_result.total)
        boundary_total = float(boundary_result.total)
        visual_result = self.visual_cells.update_from_image(
            observation,
            PredictionCellUpdateConfig(
                learning_rate=float(self.cfg.lambda_w),
                p_min=float(self.cfg.p_min),
                p_max=float(self.cfg.p_max),
                logvar_drift=float(self.cfg.world_logvar_drift),
                evidence_logvar_gain=0.18,
                reconstruction_logvar_source="precision",
            ),
            gate_image=sensor_gate,
            lateral_coupling=float(self.cfg.visual_cell_lateral_coupling),
        )
        self.last_visual_cell_update = {
            "error_mean": float(np.mean(np.abs(visual_result.error))),
            "belief_mean": float(np.mean(visual_result.belief)),
            "precision_mean": float(np.mean(visual_result.precision)),
            "vfe": float(visual_result.total),
        }
        self._update_visual_attention(visual_result.error, visual_result.precision)
        self.last_vfe = {
            "world_reconstruction": world_reconstruction,
            "world_complexity": world_complexity,
            "world": world_total,
            "boundary_reconstruction": boundary_reconstruction,
            "boundary_complexity": boundary_complexity,
            "boundary": boundary_total,
            "total": float(world_total + boundary_total),
        }
        self.last_world_error = world_error.astype(np.float32)
        self.last_boundary_error = boundary_error.astype(np.float32)
        self.prev_lenia_obs = lenia_obs
        self._refresh_world_prior_from_trm_a()
        return world_error, boundary_error

    def _prospective_body_for_fields(
        self,
        body: BodyState,
        action: str | None,
        energy_gradient: np.ndarray,
        thermal_stress: np.ndarray,
        toxicity: np.ndarray,
        niche_stability: np.ndarray,
    ) -> BodyState:
        body = _copy_body(body)
        if action is None or action == "no_action":
            return body
        gy_e, gx_e = _gradients(energy_gradient, body.centroid_y, body.centroid_x)
        gy_t, gx_t = _gradients(thermal_stress, body.centroid_y, body.centroid_x)
        gy_x, gx_x = _gradients(toxicity, body.centroid_y, body.centroid_x)
        gy_n, gx_n = _gradients(niche_stability, body.centroid_y, body.centroid_x)
        if action == "approach":
            dy, dx = _normalize_vec(gy_e + 0.25 * gy_n - 0.75 * gy_t - 0.55 * gy_x, gx_e + 0.25 * gx_n - 0.75 * gx_t - 0.55 * gx_x)
            body.centroid_y += self.cfg.move_step * dy
            body.centroid_x += self.cfg.move_step * dx
        elif action == "withdraw":
            dy, dx = _normalize_vec(0.9 * gy_t + 0.8 * gy_x - 0.35 * gy_n, 0.9 * gx_t + 0.8 * gx_x - 0.35 * gx_n)
            body.centroid_y -= self.cfg.move_step * dy
            body.centroid_x -= self.cfg.move_step * dx
        elif action == "intake":
            body.aperture_gain = min(1.0, body.aperture_gain + 0.12)
        elif action == "seal":
            body.aperture_gain = max(self.cfg.base_permeability, body.aperture_gain - 0.15)
            body.B = min(1.0, body.B + 0.05)
        elif action == "reconfigure":
            target_angle = math.atan2(gy_e - gy_t - 0.6 * gy_x + 0.35 * gy_n, gx_e - gx_t - 0.6 * gx_x + 0.35 * gx_n)
            body.aperture_angle = float(target_angle)
            body.aperture_width_deg = float(np.clip(body.aperture_width_deg * 0.9 + 10.0, 40.0, 120.0))
            body.radius = float(np.clip(body.radius + self.rng.normal(0.0, 0.3), 6.0, 10.0))
        body.centroid_y = float(np.clip(body.centroid_y, 4.0, self.env.env_config.image_size - 5.0))
        body.centroid_x = float(np.clip(body.centroid_x, 4.0, self.env.env_config.image_size - 5.0))
        return body

    def _prospective_body(self, action: str | None) -> BodyState:
        return self._prospective_body_for_fields(
            self.body,
            action,
            self.env.energy_gradient,
            self.env.thermal_stress,
            self.env.toxicity,
            self.env.niche_stability,
        )

    def _bp_local_view(self, body: BodyState) -> np.ndarray:
        _, boundary, permeability = self._body_fields(body)
        center_y = float(body.centroid_y)
        center_x = float(body.centroid_x)
        patch_size = 16
        boundary_patch = extract_centered_patch(boundary, center_y, center_x, patch_size)
        permeability_patch = extract_centered_patch(permeability, center_y, center_x, patch_size)
        observation_patch = extract_centered_patch(self.last_observation, center_y, center_x, patch_size)
        species_patch = extract_centered_patch(self.env.species_field_channels(), center_y, center_x, patch_size)
        flow_patch = extract_centered_patch(
            np.stack([self.env.flow_y, self.env.flow_x], axis=-1).astype(np.float32),
            center_y,
            center_x,
            patch_size,
        )
        viability_state = np.array([body.G, body.B], dtype=np.float32)
        return build_trm_bp_input_view(
            boundary_patch=boundary_patch,
            permeability_patch=permeability_patch,
            observation_patch=observation_patch,
            species_patch=species_patch,
            flow_patch=flow_patch,
            viability_state=viability_state,
        )

    def _apply_bp_control(
        self,
        action: str | None,
        body: BodyState,
        context_bias: dict[str, Any] | None = None,
    ) -> tuple[BodyState, dict[str, Any]]:
        result = _copy_body(body)
        diagnostics = {
            "source": "analytic",
            "model_precision": 1.0,
            "pred_interface_gain": 0.0,
            "pred_aperture_gain": float(body.aperture_gain),
            "pred_mode": -1,
            "context_source": "analytic",
            "context_model_precision": 1.0,
            "context_boundary_bias_norm": 0.0,
        }
        model = getattr(self.models, "trm_bp", None)
        if model is None or self.cfg.boundary_control_mode == "analytic":
            return result, diagnostics
        torch = self.models.torch
        assert torch is not None
        bp_input_view = self._bp_local_view(body)
        with torch.no_grad():
            outputs = model(torch.from_numpy(bp_input_view[None, ...].astype(np.float32)))
        pred_interface_gain = float(outputs["pred_interface_gain"][0, 0].cpu().item())
        pred_aperture_gain = float(outputs["pred_aperture_gain"][0, 0].cpu().item())
        mode_logits = outputs["mode_logits"][0].cpu().numpy().astype(np.float32)
        pred_mode = int(np.argmax(mode_logits))
        model_precision = float(outputs["module_precision"][0].cpu().item())
        diagnostics.update(
            {
                "model_precision": model_precision,
                "pred_interface_gain": pred_interface_gain,
                "pred_aperture_gain": pred_aperture_gain,
                "pred_mode": pred_mode,
            }
        )
        resolved_context = context_bias if context_bias is not None else self.last_mc_context
        mc_source = str(resolved_context.get("source", "analytic"))
        mc_precision = float(resolved_context.get("model_precision", 1.0))
        mc_boundary_bias = np.asarray(
            resolved_context.get("boundary_control_bias", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        )
        diagnostics.update(
            {
                "context_source": mc_source,
                "context_model_precision": mc_precision,
                "context_boundary_bias_norm": float(np.linalg.norm(mc_boundary_bias)),
            }
        )
        mc_scale = 0.0
        effective_mode = pred_mode
        effective_interface_gain = pred_interface_gain
        effective_aperture_gain = pred_aperture_gain
        effective_reconfigure_bias = 0.0
        if self.cfg.context_memory_mode != "analytic" and mc_source != "analytic":
            mc_scale = float(np.clip(0.15 * self.cfg.context_memory_residual_scale * mc_precision, 0.0, 0.5))
            open_bias = float(mc_boundary_bias[0] - 0.5 * mc_boundary_bias[1])
            seal_bias = float(mc_boundary_bias[1] - 0.25 * mc_boundary_bias[0])
            reconfigure_bias = float(mc_boundary_bias[2])
            effective_interface_gain = float(
                pred_interface_gain + mc_scale * (0.55 * open_bias - 0.75 * seal_bias)
            )
            effective_aperture_gain = float(
                pred_aperture_gain + mc_scale * (0.35 * open_bias - 0.45 * seal_bias)
            )
            effective_reconfigure_bias = mc_scale * reconfigure_bias
            if effective_reconfigure_bias > 0.08 and effective_mode != 2:
                effective_mode = 2
        diagnostics.update(
            {
                "context_boundary_scale": mc_scale,
                "effective_interface_gain": effective_interface_gain,
                "effective_aperture_gain": effective_aperture_gain,
                "effective_mode": effective_mode,
            }
        )
        if self.cfg.boundary_control_mode == "module_primary":
            result.aperture_gain = float(
                np.clip(
                    effective_aperture_gain + 0.15 * effective_interface_gain,
                    self.cfg.base_permeability,
                    1.0,
                )
            )
            if effective_mode == 2:
                result.aperture_width_deg = float(
                    np.clip(
                        result.aperture_width_deg + 12.0 + 6.0 * np.tanh(effective_reconfigure_bias),
                        40.0,
                        120.0,
                    )
                )
            diagnostics["source"] = "trm_bp_primary"
            return result, diagnostics
        blend = float(np.clip(self.cfg.boundary_control_blend * model_precision, 0.0, 1.0))
        adjusted_gain = np.clip(
            (1.0 - blend) * result.aperture_gain
            + blend * effective_aperture_gain
            + 0.10 * blend * effective_interface_gain,
            self.cfg.base_permeability,
            1.0,
        )
        result.aperture_gain = float(adjusted_gain)
        if effective_mode == 2 or action == "reconfigure":
            result.aperture_width_deg = float(
                np.clip(
                    result.aperture_width_deg
                    + 8.0 * blend * np.tanh(0.35 * effective_interface_gain + effective_reconfigure_bias),
                    40.0,
                    120.0,
                )
            )
        diagnostics["source"] = "trm_bp"
        return result, diagnostics

    def _contact_stats(
        self,
        body: BodyState,
        energy_gradient: np.ndarray | None = None,
        thermal_stress: np.ndarray | None = None,
        toxicity: np.ndarray | None = None,
        niche_stability: np.ndarray | None = None,
    ) -> dict[str, float]:
        energy_gradient = self.env.energy_gradient if energy_gradient is None else energy_gradient
        thermal_stress = self.env.thermal_stress if thermal_stress is None else thermal_stress
        toxicity = self.env.toxicity if toxicity is None else toxicity
        niche_stability = self.env.niche_stability if niche_stability is None else niche_stability
        _, boundary, permeability = self._body_fields(body)
        interface = np.clip(boundary * permeability, 0.0, 1.0)
        energy = _mean_masked(energy_gradient, interface)
        thermal = _mean_masked(thermal_stress, interface)
        toxic = _mean_masked(toxicity, interface)
        niche = _mean_masked(niche_stability, interface)
        return {
            "energy": energy,
            "thermal": thermal,
            "toxicity": toxic,
            "niche": niche,
            "resource": energy,
            "hazard": float(0.6 * thermal + 0.4 * toxic),
            "shelter": niche,
            "interface_mass": float(interface.sum()),
        }

    def _species_contact_stats(self, body: BodyState) -> dict[str, float]:
        species_fields = self.env.species_field_channels()
        _, boundary, permeability = self._body_fields(body)
        interface = np.clip(boundary * permeability, 0.0, 1.0)
        return {
            "species_energy": _mean_masked(species_fields[..., 0], interface),
            "species_thermal": _mean_masked(species_fields[..., 1], interface),
            "species_toxicity": _mean_masked(species_fields[..., 2], interface),
            "species_niche": _mean_masked(species_fields[..., 3], interface),
        }

    def _uncertainty_state(self) -> np.ndarray:
        _, boundary, permeability = self._body_fields()
        interface = np.clip(boundary * np.maximum(permeability, 0.05), 0.0, 1.0)
        world_unc = float(_mean_masked(np.mean(np.exp(np.clip(self.world_logvar, -6.0, 2.0)), axis=-1), interface))
        boundary_unc = float(
            _mean_masked(np.mean(np.exp(np.clip(self.boundary_logvar, -6.0, 2.0)), axis=-1), interface)
        )
        contact = self._contact_stats(self.body)
        species_contact = self._species_contact_stats(self.body)
        return np.array(
            [world_unc, boundary_unc, float(contact["thermal"]), float(contact["toxicity"])],
            dtype=np.float32,
        )

    def _mc_previous_action_summary(self) -> tuple[str, float]:
        if not self.history:
            return "no_action", 0.0
        prev_action = str(self.history[-1].get("action", "no_action"))
        return prev_action, _policy_action_cost(prev_action)

    def _sample_flow_state(self) -> np.ndarray:
        y = int(np.clip(round(float(self.body.centroid_y)), 0, self.env.flow_y.shape[0] - 1))
        x = int(np.clip(round(float(self.body.centroid_x)), 0, self.env.flow_x.shape[1] - 1))
        return np.array([self.env.flow_y[y, x], self.env.flow_x[y, x]], dtype=np.float32)

    def _build_mc_feature_vector(
        self,
        viability_monitor: dict[str, Any],
        uncertainty_state: np.ndarray,
        contact: dict[str, float],
        species_contact: dict[str, float],
    ) -> np.ndarray:
        prev_action, prev_action_cost = self._mc_previous_action_summary()
        body = self.body
        interface_summary = np.array(
            [
                float(body.aperture_gain),
                float(np.clip(body.aperture_width_deg / 120.0, 0.0, 1.0)),
                float(np.sin(body.aperture_angle)),
                float(np.cos(body.aperture_angle)),
            ],
            dtype=np.float32,
        )
        interface_mass_feature = np.array(
            [float(np.tanh(float(contact["interface_mass"]) / 32.0))],
            dtype=np.float32,
        )
        env_contact_state = np.array(
            [contact["energy"], contact["thermal"], contact["toxicity"], contact["niche"]],
            dtype=np.float32,
        )
        species_contact_state = np.array(
            [
                species_contact["species_energy"],
                species_contact["species_thermal"],
                species_contact["species_toxicity"],
                species_contact["species_niche"],
            ],
            dtype=np.float32,
        )
        flow_state = self._sample_flow_state()
        return np.concatenate(
            [
                viability_monitor["state"].astype(np.float32),
                viability_monitor["homeostatic_error_vector"].astype(np.float32),
                env_contact_state.astype(np.float32),
                species_contact_state.astype(np.float32),
                uncertainty_state.astype(np.float32),
                flow_state.astype(np.float32),
                interface_summary.astype(np.float32),
                interface_mass_feature.astype(np.float32),
                _action_onehot_runtime(prev_action, include_no_action=True),
                np.array([float(prev_action_cost)], dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

    def _context_memory_bias(
        self,
        viability_monitor: dict[str, Any],
        uncertainty_state: np.ndarray,
        contact: dict[str, float],
        species_contact: dict[str, float],
    ) -> dict[str, Any]:
        model = getattr(self.models, "trm_mc", None)
        feature = self._build_mc_feature_vector(viability_monitor, uncertainty_state, contact, species_contact)
        if model is None or self.cfg.context_memory_mode == "analytic":
            return {
                "source": "analytic",
                "model_precision": 1.0,
                "window_length": min(len(self.mc_feature_history) + 1, int(self.cfg.context_memory_window_size)),
                "sequence_bias": np.zeros(len(ACTIONS), dtype=np.float32),
                "boundary_control_bias": np.zeros(3, dtype=np.float32),
                "context_state": np.zeros(32, dtype=np.float32),
                "retrieved_context": feature.astype(np.float32),
                "current_feature": feature.astype(np.float32),
            }

        history_features = self.mc_feature_history + [feature.astype(np.float32)]
        step_features = np.stack(history_features, axis=0).astype(np.float32)
        mc_input_view, mc_window_mask = build_trm_mc_input_view(
            step_features,
            window_size=int(self.cfg.context_memory_window_size),
        )
        input_window = mc_input_view[-1:].astype(np.float32)
        window_mask = mc_window_mask[-1:].astype(np.float32)
        torch = self.models.torch
        assert torch is not None
        with torch.no_grad():
            outputs = model(
                torch.from_numpy(input_window),
                torch.from_numpy(window_mask),
            )
        return {
            "source": "trm_mc",
            "model_precision": float(outputs["module_precision"][0].cpu().item()),
            "window_length": int(outputs["window_lengths"][0].cpu().item()),
            "sequence_bias": outputs["sequence_bias"][0].cpu().numpy().astype(np.float32),
            "boundary_control_bias": outputs["boundary_control_bias"][0].cpu().numpy().astype(np.float32),
            "context_state": outputs["context_state"][0].cpu().numpy().astype(np.float32),
            "retrieved_context": outputs["retrieved_context"][0].cpu().numpy().astype(np.float32),
            "current_feature": feature.astype(np.float32),
        }

    def _monitor_viability(self, action_cost: float) -> dict[str, Any]:
        analytic_state = np.array([self.body.G, self.body.B], dtype=np.float32)
        analytic_error_vector = np.abs(
            analytic_state - np.array([self.cfg.G_target, self.cfg.B_target], dtype=np.float32)
        )
        result = {
            "state": analytic_state,
            "risk": float(
                _risk_proxy(self.body.G, self.body.B, float(self.body.G < self.cfg.tau_G or self.body.B < self.cfg.tau_B), self.cfg)
            ),
            "precision": 1.0,
            "homeostatic_error": float(analytic_error_vector.sum()),
            "homeostatic_error_vector": analytic_error_vector,
            "source": "analytic",
        }
        if self.models.trm_vm is None or self.cfg.viability_mode == "analytic":
            return result
        torch = self.models.torch
        assert torch is not None
        contact = self._contact_stats(self.body)
        species_contact = self._species_contact_stats(self.body)
        contact_state = np.array(
            [
                [
                    contact["energy"],
                    contact["thermal"],
                    contact["toxicity"],
                    contact["niche"],
                    species_contact["species_energy"],
                    species_contact["species_thermal"],
                    species_contact["species_toxicity"],
                    species_contact["species_niche"],
                ]
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            outputs = self.models.trm_vm(
                torch.from_numpy(analytic_state[None, ...]),
                torch.from_numpy(contact_state),
                torch.from_numpy(np.array([[action_cost]], dtype=np.float32)),
            )
        predicted_state = outputs["viability_state"][0].cpu().numpy().astype(np.float32)
        predicted_error = outputs["homeostatic_error"][0].cpu().numpy().astype(np.float32)
        predicted_risk = float(outputs["viability_risk"][0, 0].cpu().item())
        predicted_precision = float(outputs["module_precision"][0].cpu().item())
        if self.cfg.viability_mode == "module_primary":
            return {
                "state": predicted_state,
                "risk": predicted_risk,
                "precision": predicted_precision,
                "homeostatic_error": float(predicted_error.sum()),
                "homeostatic_error_vector": predicted_error,
                "source": "trm_vm_primary",
            }
        blend = float(np.clip(self.cfg.viability_monitor_blend * predicted_precision, 0.0, 1.0))
        blended_state = ((1.0 - blend) * analytic_state + blend * predicted_state).astype(np.float32)
        blended_error = ((1.0 - blend) * analytic_error_vector + blend * predicted_error).astype(np.float32)
        return {
            "state": blended_state,
            "risk": predicted_risk,
            "precision": predicted_precision,
            "homeostatic_error": float(blended_error.sum()),
            "homeostatic_error_vector": blended_error,
            "source": "trm_vm",
        }

    def _epistemic_proxy(
        self,
        body: BodyState,
        energy_gradient: np.ndarray | None = None,
        thermal_stress: np.ndarray | None = None,
        toxicity: np.ndarray | None = None,
        niche_stability: np.ndarray | None = None,
    ) -> float:
        energy_gradient = self.env.energy_gradient if energy_gradient is None else energy_gradient
        thermal_stress = self.env.thermal_stress if thermal_stress is None else thermal_stress
        toxicity = self.env.toxicity if toxicity is None else toxicity
        niche_stability = self.env.niche_stability if niche_stability is None else niche_stability
        _, boundary, permeability = self._body_fields(body)
        interface = np.clip(boundary * np.maximum(permeability, 0.05), 0.0, 1.0)
        world_unc = np.mean(np.exp(self.world_logvar), axis=-1)
        gy_r, gx_r = np.gradient(energy_gradient)
        gy_t, gx_t = np.gradient(thermal_stress)
        gy_x, gx_x = np.gradient(toxicity)
        gy_n, gx_n = np.gradient(niche_stability)
        cue_grad = np.sqrt(
            gy_r * gy_r
            + gx_r * gx_r
            + 0.8 * (gy_t * gy_t + gx_t * gx_t)
            + 0.7 * (gy_x * gy_x + gx_x * gx_x)
            + 0.35 * (gy_n * gy_n + gx_n * gx_n)
        ).astype(np.float32)
        visual_attention = float(_mean_masked(self.visual_attention_projected, interface))
        return float(
            self.cfg.epistemic_scale * _mean_masked(world_unc * cue_grad, interface)
            + float(self.cfg.visual_attention_epistemic_gain) * visual_attention
        )

    def _ambiguity_proxy(self, body: BodyState) -> float:
        _, boundary, permeability = self._body_fields(body)
        interface = np.clip(boundary * np.maximum(permeability, 0.05), 0.0, 1.0)
        world_unc = np.mean(np.exp(np.clip(self.world_logvar, -6.0, 2.0)), axis=-1)
        boundary_unc = np.mean(np.exp(np.clip(self.boundary_logvar, -6.0, 2.0)), axis=-1)
        return float(
            _mean_masked(world_unc, interface) + self.cfg.ambiguity_w_boundary * _mean_masked(boundary_unc, interface)
        )

    def _predicted_viability_for_fields(
        self,
        body: BodyState,
        action: str | None,
        energy_gradient: np.ndarray,
        thermal_stress: np.ndarray,
        toxicity: np.ndarray,
        niche_stability: np.ndarray,
    ) -> tuple[float, float]:
        contact = self._contact_stats(
            body,
            energy_gradient=energy_gradient,
            thermal_stress=thermal_stress,
            toxicity=toxicity,
            niche_stability=niche_stability,
        )
        intake_bonus = 1.25 if action == "intake" else 1.0
        leakage_penalty = 1.25 if action == "intake" else 1.0
        seal_gain = 0.05 if action == "seal" else 0.0
        reconfigure_gain = 0.03 if action == "reconfigure" else 0.0
        G_next = np.clip(
            body.G
            - self.cfg.mu_G
            + self.cfg.alpha_R * contact["energy"] * intake_bonus
            - _policy_action_cost(action),
            0.0,
            1.0,
        )
        B_next = np.clip(
            body.B
            - self.cfg.mu_B
            - self.cfg.alpha_H * contact["thermal"] * leakage_penalty
            - self.cfg.alpha_X * contact["toxicity"] * leakage_penalty
            + self.cfg.alpha_S * contact["niche"]
            + seal_gain
            + reconfigure_gain,
            0.0,
            1.0,
        )
        return float(G_next), float(B_next)

    def _predicted_viability(self, body: BodyState, action: str | None) -> tuple[float, float]:
        return self._predicted_viability_for_fields(
            body,
            action,
            self.env.energy_gradient,
            self.env.thermal_stress,
            self.env.toxicity,
            self.env.niche_stability,
        )

    def _updated_fields_for_policy(
        self,
        body: BodyState,
        action: str | None,
        energy_gradient: np.ndarray,
        thermal_stress: np.ndarray,
        toxicity: np.ndarray,
        niche_stability: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        next_energy_gradient = energy_gradient.copy()
        next_thermal_stress = thermal_stress.copy()
        next_toxicity = toxicity.copy()
        next_niche_stability = niche_stability.copy()
        _, boundary, permeability = self._body_fields(body)
        contact_mask = boundary * np.clip(permeability, 0.0, 1.0)
        if action == "intake":
            consume = np.minimum(next_energy_gradient, 0.06 * contact_mask)
            next_energy_gradient = _clip01(next_energy_gradient - consume)
        next_energy_gradient = _clip01(
            next_energy_gradient + self.env.env_config.resource_regen * (1.0 - next_energy_gradient)
        )
        return next_energy_gradient, next_thermal_stress, next_toxicity, next_niche_stability

    def _single_step_policy_terms(
        self,
        body: BodyState,
        action: str | None,
        energy_gradient: np.ndarray,
        thermal_stress: np.ndarray,
        toxicity: np.ndarray,
        niche_stability: np.ndarray,
    ) -> tuple[float, dict[str, float], BodyState, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        current_contact = self._contact_stats(
            body,
            energy_gradient=energy_gradient,
            thermal_stress=thermal_stress,
            toxicity=toxicity,
            niche_stability=niche_stability,
        )
        next_body = self._prospective_body_for_fields(
            body,
            action,
            energy_gradient,
            thermal_stress,
            toxicity,
            niche_stability,
        )
        G_next, B_next = self._predicted_viability_for_fields(
            next_body,
            action,
            energy_gradient,
            thermal_stress,
            toxicity,
            niche_stability,
        )
        next_body.G = G_next
        next_body.B = B_next
        death_risk = float(G_next < self.cfg.tau_G or B_next < self.cfg.tau_B)
        next_contact = self._contact_stats(
            next_body,
            energy_gradient=energy_gradient,
            thermal_stress=thermal_stress,
            toxicity=toxicity,
            niche_stability=niche_stability,
        )
        contact_risk = _contact_risk_proxy(current_contact, next_contact, self.cfg)
        risk = _risk_proxy(G_next, B_next, death_risk, self.cfg) + contact_risk
        ambiguity = self._ambiguity_proxy(next_body)
        epistemic = self._epistemic_proxy(
            next_body,
            energy_gradient=energy_gradient,
            thermal_stress=thermal_stress,
            toxicity=toxicity,
            niche_stability=niche_stability,
        )
        score = risk + ambiguity - epistemic
        next_fields = self._updated_fields_for_policy(
            next_body,
            action,
            energy_gradient,
            thermal_stress,
            toxicity,
            niche_stability,
        )
        diagnostics = {
            "risk": float(risk),
            "contact_risk": float(contact_risk),
            "ambiguity": float(ambiguity),
            "epistemic": float(epistemic),
            "pred_G": float(G_next),
            "pred_B": float(B_next),
            "death_risk": float(death_risk),
        }
        return float(score), diagnostics, next_body, next_fields

    def _rollout_policy_score(
        self,
        body: BodyState,
        action: str,
        energy_gradient: np.ndarray,
        thermal_stress: np.ndarray,
        toxicity: np.ndarray,
        niche_stability: np.ndarray,
        horizon: int,
    ) -> tuple[float, float, dict[str, float]]:
        immediate_score, diagnostics, next_body, next_fields = self._single_step_policy_terms(
            body,
            action,
            energy_gradient,
            thermal_stress,
            toxicity,
            niche_stability,
        )
        continuation_score = 0.0
        if horizon > 1:
            future_scores = [
                self._rollout_policy_score(
                    next_body,
                    future_action,
                    next_fields[0],
                    next_fields[1],
                    next_fields[2],
                    next_fields[3],
                    horizon - 1,
                )[0]
                for future_action in ACTIONS
            ]
            continuation_score = float(self.cfg.lookahead_discount * min(future_scores))
        total_score = float(immediate_score + continuation_score)
        diagnostics["continuation_score"] = float(continuation_score)
        diagnostics["lookahead_horizon"] = int(horizon)
        diagnostics["lookahead_score"] = float(total_score)
        return total_score, continuation_score, diagnostics

    def _policy_scores(self) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        scores = np.zeros(len(ACTIONS), dtype=np.float32)
        diagnostics: dict[str, dict[str, float]] = {}
        horizon = max(1, int(self.cfg.lookahead_horizon))
        for i, action in enumerate(ACTIONS):
            score, _, diag = self._rollout_policy_score(
                self.body,
                action,
                self.env.energy_gradient,
                self.env.thermal_stress,
                self.env.toxicity,
                self.env.niche_stability,
                horizon,
            )
            scores[i] = float(score)
            diagnostics[action] = diag
        return scores, diagnostics

    def _select_policy(
        self,
        scores: np.ndarray,
        score_diag: dict[str, dict[str, float]],
        viability_monitor: dict[str, Any],
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        role_bias = self._role_action_bias(str(self.body.role))
        base_logits = (-self.cfg.beta_pi * scores.astype(np.float32)).astype(np.float32) + role_bias
        diagnostics = {
            "base_logits": base_logits.astype(np.float32),
            "role_bias_logits": role_bias.astype(np.float32),
            "role": str(self.body.role),
        }
        trm_as_model = getattr(self.models, "trm_as", None)
        trm_ag_model = getattr(self.models, "trm_ag", None)
        trm_mc_model = getattr(self.models, "trm_mc", None)
        if trm_as_model is None or self.cfg.action_mode == "analytic":
            policy = _softmax(base_logits)
            action = ACTIONS[int(np.argmax(policy))]
            diagnostics["source"] = "analytic"
            diagnostics["final_logits"] = base_logits.astype(np.float32)
            return policy.astype(np.float32), action, diagnostics

        torch = self.models.torch
        assert torch is not None
        uncertainty_state = self._uncertainty_state()
        contact = self._contact_stats(self.body)
        species_contact = self._species_contact_stats(self.body)
        env_contact_state = np.array(
            [[contact["energy"], contact["thermal"], contact["toxicity"], contact["niche"]]],
            dtype=np.float32,
        )
        species_contact_state = np.array(
            [
                [
                    species_contact["species_energy"],
                    species_contact["species_thermal"],
                    species_contact["species_toxicity"],
                    species_contact["species_niche"],
                ]
            ],
            dtype=np.float32,
        )
        mc_context = self._context_memory_bias(
            viability_monitor,
            uncertainty_state,
            contact,
            species_contact,
        )
        with torch.no_grad():
            outputs = trm_as_model(
                torch.from_numpy(viability_monitor["state"][None, ...].astype(np.float32)),
                torch.from_numpy(scores[None, ...].astype(np.float32)),
                torch.from_numpy(uncertainty_state[None, ...].astype(np.float32)),
                torch.from_numpy(env_contact_state),
                torch.from_numpy(species_contact_state),
            )
        residual_logits = outputs["policy_logits"][0].cpu().numpy().astype(np.float32)
        residual_logits = residual_logits - float(np.mean(residual_logits))
        model_precision = float(outputs["module_precision"][0].cpu().item())
        context_bias = np.zeros(len(ACTIONS), dtype=np.float32)
        context_scale = 0.0
        if trm_mc_model is not None and self.cfg.context_memory_mode != "analytic":
            context_scale = float(
                self.cfg.context_memory_residual_scale * float(mc_context["model_precision"])
            )
            context_bias = (context_scale * mc_context["sequence_bias"].astype(np.float32)).astype(np.float32)
        if self.cfg.action_mode == "module_primary":
            pre_ag_logits = (residual_logits + context_bias + role_bias).astype(np.float32)
            final_logits = pre_ag_logits.astype(np.float32)
            if trm_ag_model is not None and self.cfg.action_gating_mode != "analytic":
                ag_input_view = build_trm_ag_input_view(
                    pre_ag_logits.astype(np.float32),
                    viability_monitor["state"].astype(np.float32),
                    viability_monitor["homeostatic_error_vector"].astype(np.float32),
                    np.array([viability_monitor["risk"]], dtype=np.float32),
                    uncertainty_state.astype(np.float32),
                    env_contact_state[0].astype(np.float32),
                    species_contact_state[0].astype(np.float32),
                )
                with torch.no_grad():
                    ag_outputs = trm_ag_model(torch.from_numpy(ag_input_view[None, ...].astype(np.float32)))
                ag_gated_logits = ag_outputs["gated_policy_logits"][0].cpu().numpy().astype(np.float32)
                ag_inhibition_mask = ag_outputs["inhibition_mask"][0].cpu().numpy().astype(np.float32)
                ag_control_mode_logits = ag_outputs["control_mode_logits"][0].cpu().numpy().astype(np.float32)
                ag_control_mode = int(np.argmax(ag_control_mode_logits))
                ag_precision = float(ag_outputs["module_precision"][0].cpu().item())
                if self.cfg.action_gating_mode == "module_primary":
                    final_logits = ag_gated_logits.astype(np.float32)
                    diagnostics["source"] = "trm_ag_primary"
                else:
                    # Assistive TRM-Ag acts as a downstream veto gate. It should
                    # suppress unsafe actions rather than replace the scorer.
                    final_logits = apply_ag_assistive_veto(
                        pre_ag_logits.astype(np.float32),
                        ag_inhibition_mask.astype(np.float32),
                        int(ag_control_mode),
                    )
                    diagnostics["source"] = "trm_ag"
                diagnostics.update(
                    {
                        "ag_source": diagnostics["source"],
                        "ag_model_precision": ag_precision,
                        "ag_pre_logits": pre_ag_logits.astype(np.float32),
                        "ag_gated_logits": ag_gated_logits.astype(np.float32),
                        "ag_inhibition_mask": ag_inhibition_mask.astype(np.float32),
                        "ag_control_mode_logits": ag_control_mode_logits.astype(np.float32),
                        "ag_control_mode": ag_control_mode,
                    }
                )
            policy = _softmax(final_logits)
            action = ACTIONS[int(np.argmax(policy))]
            diagnostics.update(
                {
                    "source": diagnostics.get("source") or "trm_as_primary",
                    "model_precision": model_precision,
                    "residual_logits": residual_logits.astype(np.float32),
                    "context_source": mc_context["source"],
                    "context_model_precision": float(mc_context["model_precision"]),
                    "context_window_length": int(mc_context["window_length"]),
                    "context_bias_logits": context_bias.astype(np.float32),
                    "context_boundary_bias": mc_context["boundary_control_bias"].astype(np.float32),
                    "context_state": mc_context["context_state"].astype(np.float32),
                    "context_retrieved": mc_context["retrieved_context"].astype(np.float32),
                    "final_logits": final_logits.astype(np.float32),
                    "uncertainty_state": uncertainty_state.astype(np.float32),
                }
            )
            return policy.astype(np.float32), action, diagnostics
        pre_ag_logits = base_logits + (
            self.cfg.action_model_residual_scale * model_precision * residual_logits
        ).astype(np.float32) + context_bias.astype(np.float32)
        final_logits = pre_ag_logits.astype(np.float32)
        if trm_ag_model is not None and self.cfg.action_gating_mode != "analytic":
            ag_input_view = build_trm_ag_input_view(
                pre_ag_logits.astype(np.float32),
                viability_monitor["state"].astype(np.float32),
                viability_monitor["homeostatic_error_vector"].astype(np.float32),
                np.array([viability_monitor["risk"]], dtype=np.float32),
                uncertainty_state.astype(np.float32),
                env_contact_state[0].astype(np.float32),
                species_contact_state[0].astype(np.float32),
            )
            with torch.no_grad():
                ag_outputs = trm_ag_model(torch.from_numpy(ag_input_view[None, ...].astype(np.float32)))
            ag_gated_logits = ag_outputs["gated_policy_logits"][0].cpu().numpy().astype(np.float32)
            ag_inhibition_mask = ag_outputs["inhibition_mask"][0].cpu().numpy().astype(np.float32)
            ag_control_mode_logits = ag_outputs["control_mode_logits"][0].cpu().numpy().astype(np.float32)
            ag_control_mode = int(np.argmax(ag_control_mode_logits))
            ag_precision = float(ag_outputs["module_precision"][0].cpu().item())
            if self.cfg.action_gating_mode == "module_primary":
                final_logits = ag_gated_logits.astype(np.float32)
                diagnostics["source"] = "trm_ag_primary"
            else:
                # Assistive TRM-Ag acts as a downstream veto gate. It should
                # suppress unsafe actions rather than replace the scorer.
                final_logits = apply_ag_assistive_veto(
                    pre_ag_logits.astype(np.float32),
                    ag_inhibition_mask.astype(np.float32),
                    int(ag_control_mode),
                )
                diagnostics["source"] = "trm_ag"
            diagnostics.update(
                {
                    "ag_source": diagnostics["source"],
                    "ag_model_precision": ag_precision,
                    "ag_pre_logits": pre_ag_logits.astype(np.float32),
                    "ag_gated_logits": ag_gated_logits.astype(np.float32),
                    "ag_inhibition_mask": ag_inhibition_mask.astype(np.float32),
                    "ag_control_mode_logits": ag_control_mode_logits.astype(np.float32),
                    "ag_control_mode": ag_control_mode,
                }
            )
        policy = _softmax(final_logits)
        action = ACTIONS[int(np.argmax(policy))]
        diagnostics.update(
            {
                "source": diagnostics.get("source") or "trm_as",
                "model_precision": model_precision,
                "residual_logits": residual_logits.astype(np.float32),
                "context_source": mc_context["source"],
                "context_model_precision": float(mc_context["model_precision"]),
                "context_window_length": int(mc_context["window_length"]),
                "context_bias_logits": context_bias.astype(np.float32),
                "context_boundary_bias": mc_context["boundary_control_bias"].astype(np.float32),
                "context_state": mc_context["context_state"].astype(np.float32),
                "context_retrieved": mc_context["retrieved_context"].astype(np.float32),
                "final_logits": final_logits.astype(np.float32),
                "uncertainty_state": uncertainty_state.astype(np.float32),
            }
        )
        return policy.astype(np.float32), action, diagnostics

    def _apply_action(self, action: str | None, context_bias: dict[str, Any] | None = None) -> dict[str, Any]:
        primary_index = -1
        for idx, body in enumerate(self.bodies):
            if body is self.body:
                primary_index = idx
                break
        next_body = self._prospective_body(action)
        next_body.body_id = int(self.body.body_id)
        next_body.parent_id = int(self.body.parent_id)
        next_body.generation = int(self.body.generation)
        next_body.dead_count = int(self.body.dead_count)
        next_body.alive = bool(self.body.alive)
        next_body, bp_meta = self._apply_bp_control(action, next_body, context_bias=context_bias)
        self.body = next_body
        if primary_index >= 0:
            self.bodies[primary_index] = next_body
        elif self.bodies:
            self.bodies[0] = next_body
        else:
            self.bodies = [next_body]
        G_next, B_next = self._predicted_viability(self.body, action)
        self.body.G = G_next
        self.body.B = B_next
        self.last_bp_control = dict(bp_meta)
        return bp_meta

    def _is_action_locked(self, current_action: str | None) -> bool:
        return is_action_locked(
            self.history,
            current_action,
            int(self.cfg.degenerate_action_lock_window),
        )

    def _classify_death_cause(
        self,
        *,
        threshold_violation: bool,
        nonfinite_state: bool,
        invalid_body_state: bool,
        action_lock: bool,
        policy_forbidden_window: bool,
    ) -> str:
        return classify_death_cause(
            threshold_violation=threshold_violation,
            nonfinite_state=nonfinite_state,
            invalid_body_state=invalid_body_state,
            action_lock=action_lock,
            policy_forbidden_window=policy_forbidden_window,
            expected_label=DEATH_CAUSE_EXPECTED,
            degenerate_label=DEATH_CAUSE_DEGENERATE,
            policy_forbidden_label=DEATH_CAUSE_POLICY_FORBIDDEN,
        )

    def _update_death(
        self,
        t: int | None = None,
        action: str | None = None,
        invalid_body_state: bool = False,
    ) -> bool:
        result = update_death_state(
            current_dead_count=int(self.body.dead_count),
            G=float(self.body.G),
            B=float(self.body.B),
            tau_G=float(self.cfg.tau_G),
            tau_B=float(self.cfg.tau_B),
            k_irrev=int(self.cfg.k_irrev),
            history=self.history,
            action=action,
            action_lock_window=int(self.cfg.degenerate_action_lock_window),
            t=t,
            policy_forbidden_min_survival_steps=int(self.cfg.policy_forbidden_min_survival_steps),
            invalid_body_state=bool(invalid_body_state),
            expected_label=DEATH_CAUSE_EXPECTED,
            degenerate_label=DEATH_CAUSE_DEGENERATE,
            policy_forbidden_label=DEATH_CAUSE_POLICY_FORBIDDEN,
        )
        self.body.dead_count = int(result["dead_count"])
        self.last_death_signals = dict(result["death_signals"])
        self.last_death_cause = str(result["death_cause"]) if result["death_cause"] is not None else None
        return bool(result["dead"])

    def step(self, t: int) -> bool:
        self.env.step_lenia()
        observation, sensor_gate, occupancy, boundary = self._observe_via_boundary_interface()
        _, _, permeability = self._population_body_fields()
        boundary_obs = np.stack([boundary, permeability], axis=-1).astype(np.float32)
        contact = self._contact_stats(self.body)
        species_contact = self._species_contact_stats(self.body)
        world_error, boundary_error = self._belief_update(observation, sensor_gate, boundary_obs)
        role_stats = self._update_activity_distribution_and_roles()

        scores, score_diag = self._policy_scores()
        viability_monitor = self._monitor_viability(action_cost=0.0)
        policy, selected_action, policy_meta = self._select_policy(scores, score_diag, viability_monitor)
        self.policy_belief = policy.astype(np.float32)
        if self.cfg.policy_mode == "random":
            action = str(self.rng.choice(ACTIONS))
        elif self.cfg.policy_mode == "no_action":
            action = "no_action"
        else:
            action = selected_action

        current_mc_context = {
            "source": str(policy_meta.get("context_source", "analytic")),
            "model_precision": float(policy_meta.get("context_model_precision", 1.0)),
            "window_length": int(policy_meta.get("context_window_length", 0)),
            "sequence_bias": np.asarray(
                policy_meta.get("context_bias_logits", np.zeros(len(ACTIONS), dtype=np.float32)),
                dtype=np.float32,
            ),
            "boundary_control_bias": np.asarray(
                policy_meta.get(
                    "context_boundary_bias",
                    self.last_mc_context.get("boundary_control_bias", np.zeros(3, dtype=np.float32)),
                ),
                dtype=np.float32,
            ),
            "context_state": np.asarray(
                policy_meta.get("context_state", np.zeros(32, dtype=np.float32)),
                dtype=np.float32,
            ),
            "retrieved_context": np.asarray(
                policy_meta.get("context_retrieved", self.last_mc_context.get("retrieved_context", np.zeros(28, dtype=np.float32))),
                dtype=np.float32,
            ),
        }
        bp_meta = self._enact_action_via_boundary_interface(action, context_bias=current_mc_context)
        self._refresh_body_phenotype(self.body)
        invariant_signals = self._body_invariant_signals(self.body)
        invalid_body_state = bool(invariant_signals["invalid_body_state"])
        spawn_split = self._spawn_split_signals(self.body)
        if invalid_body_state:
            self.invalid_body_state_count += 1
        spawn_events = 0
        split_events = 0
        if not invalid_body_state and self._can_expand_population():
            split_candidate = bool(spawn_split["split_candidate"])
            spawn_candidate = bool(spawn_split["spawn_candidate"])
            if split_candidate:
                child = self._split_primary()
                if child is not None:
                    split_events += 1
            elif spawn_candidate:
                child = self._spawn_from_primary()
                if child is not None:
                    spawn_events += 1
        if spawn_events > 0:
            self.population_event_counts["spawn"] += int(spawn_events)
        if split_events > 0:
            self.population_event_counts["split"] += int(split_events)

        dead_primary = self._update_death(t=t, action=action, invalid_body_state=invalid_body_state)
        self.body.alive = not dead_primary
        dead_aux, aux_stats = self._update_auxiliary_bodies()

        dead_bodies: list[BodyState] = list(dead_aux)
        if dead_primary:
            dead_bodies.append(self.body)
        if dead_bodies:
            dead_ids = {body.body_id for body in dead_bodies}
            self.bodies = [body for body in self.bodies if body.body_id not in dead_ids]
            self.population_event_counts["death"] += len(dead_bodies)
        if self.bodies:
            self._select_primary_body()
        self._update_trace_field(dead_primary, dead_bodies=dead_bodies)
        episode_dead = len(self.bodies) == 0
        selected_diag = score_diag.get(selected_action, score_diag[ACTIONS[0]])
        self.last_mc_context = current_mc_context
        current_feature = self._build_mc_feature_vector(
            viability_monitor,
            policy_meta.get("uncertainty_state", self._uncertainty_state()),
            contact,
            species_contact,
        )
        self.mc_feature_history.append(current_feature.astype(np.float32))
        max_history = max(1, int(self.cfg.context_memory_window_size))
        if len(self.mc_feature_history) > max_history:
            self.mc_feature_history = self.mc_feature_history[-max_history:]

        self.history.append(
            {
                "t": int(t),
                "action": action,
                "G": float(self.body.G),
                "B": float(self.body.B),
                "dead_count": int(self.body.dead_count),
                "policy_belief": {name: float(policy[i]) for i, name in enumerate(ACTIONS)},
                "policy_mode": self.cfg.policy_mode,
                "policy_score": score_diag,
                "policy_source": policy_meta["source"],
                "policy_model_precision": float(policy_meta.get("model_precision", 1.0)),
                "p_t": float(role_stats["p_t"]),
                "activity_buffer_t": float(role_stats["buffer_t"]),
                "activity_hazard_t": float(role_stats["hazard_t"]),
                "activity_boundary_damage_t": float(role_stats["boundary_damage_t"]),
                "challenge_body_count": int(role_stats["challenge_body_count"]),
                "conservative_body_count": int(role_stats["conservative_body_count"]),
                "role_switch_events_step": int(role_stats["role_switch_events_step"]),
                "body_role": str(self.body.role),
                "ag_source": str(policy_meta.get("ag_source", "analytic")),
                "ag_model_precision": float(policy_meta.get("ag_model_precision", 1.0)),
                "ag_control_mode": int(policy_meta.get("ag_control_mode", -1)),
                "ag_max_inhibition": float(
                    np.max(policy_meta.get("ag_inhibition_mask", np.zeros(len(ACTIONS), dtype=np.float32)))
                ),
                "ag_blocked_action_count": int(
                    np.sum(policy_meta.get("ag_inhibition_mask", np.zeros(len(ACTIONS), dtype=np.float32)) >= 0.60)
                ),
                "mc_context_source": str(policy_meta.get("context_source", "analytic")),
                "mc_model_precision": float(policy_meta.get("context_model_precision", 1.0)),
                "mc_window_length": int(policy_meta.get("context_window_length", 0)),
                "mc_bias_norm": float(np.linalg.norm(policy_meta.get("context_bias_logits", np.zeros(len(ACTIONS))))),
                "bp_control_source": bp_meta["source"],
                "bp_model_precision": float(bp_meta.get("model_precision", 1.0)),
                "bp_pred_interface_gain": float(bp_meta.get("pred_interface_gain", 0.0)),
                "bp_pred_aperture_gain": float(bp_meta.get("pred_aperture_gain", self.body.aperture_gain)),
                "bp_pred_mode": int(bp_meta.get("pred_mode", -1)),
                "bp_context_source": str(bp_meta.get("context_source", "analytic")),
                "bp_context_model_precision": float(bp_meta.get("context_model_precision", 1.0)),
                "bp_context_bias_norm": float(bp_meta.get("context_boundary_bias_norm", 0.0)),
                "policy_entropy": _entropy(policy),
                "monitor_viability_source": viability_monitor["source"],
                "monitor_viability_risk": float(viability_monitor["risk"]),
                "monitor_viability_precision": float(viability_monitor["precision"]),
                "monitor_homeostatic_error": float(viability_monitor["homeostatic_error"]),
                "monitor_G": float(viability_monitor["state"][0]),
                "monitor_B": float(viability_monitor["state"][1]),
                "vfe_world": float(self.last_vfe["world"]),
                "vfe_boundary": float(self.last_vfe["boundary"]),
                "vfe_total": float(self.last_vfe["total"]),
                "visual_cell_error_mean": float(self.last_visual_cell_update["error_mean"]),
                "visual_cell_belief_mean": float(self.last_visual_cell_update["belief_mean"]),
                "visual_cell_precision_mean": float(self.last_visual_cell_update["precision_mean"]),
                "visual_cell_vfe": float(self.last_visual_cell_update["vfe"]),
                "visual_attention_target_row": int(self.last_visual_attention["target_row"]),
                "visual_attention_target_col": int(self.last_visual_attention["target_col"]),
                "visual_attention_target_y": float(self.last_visual_attention["target_y"]),
                "visual_attention_target_x": float(self.last_visual_attention["target_x"]),
                "visual_attention_max": float(self.last_visual_attention["max"]),
                "visual_attention_entropy": float(self.last_visual_attention["entropy"]),
                "visual_attention_salience_mean": float(self.last_visual_attention["salience_mean"]),
                "efe_selected_action": selected_action,
                "efe_selected": float(selected_diag["lookahead_score"]),
                "efe_selected_risk": float(selected_diag["risk"]),
                "efe_selected_ambiguity": float(selected_diag["ambiguity"]),
                "efe_selected_epistemic": float(selected_diag["epistemic"]),
                "sensor_gate_mean": float(sensor_gate.mean()),
                "world_error_mean": float(np.mean(np.abs(world_error))),
                "boundary_error_mean": float(np.mean(np.abs(boundary_error))),
                "contact_energy": float(contact["energy"]),
                "contact_thermal": float(contact["thermal"]),
                "contact_toxicity": float(contact["toxicity"]),
                "contact_niche": float(contact["niche"]),
                "contact_species_energy": float(species_contact["species_energy"]),
                "contact_species_thermal": float(species_contact["species_thermal"]),
                "contact_species_toxicity": float(species_contact["species_toxicity"]),
                "contact_species_niche": float(species_contact["species_niche"]),
                "contact_resource": float(contact["resource"]),
                "contact_hazard": float(contact["hazard"]),
                "contact_shelter": float(contact["shelter"]),
                "homeostatic_error": homeostatic_error(self.body.G, self.body.B, self.cfg),
                "centroid_y": float(self.body.centroid_y),
                "centroid_x": float(self.body.centroid_x),
                "aperture_angle": float(self.body.aperture_angle),
                "aperture_gain": float(self.body.aperture_gain),
                "body_energy": float(self.body.energy),
                "body_mass": float(self.body.mass),
                "boundary_integrity": float(self.body.boundary_integrity),
                "alive": bool(self.body.alive),
                "invalid_body_state": bool(invalid_body_state),
                "body_invariant_signals": dict(invariant_signals),
                "boundary_interface_observe": True,
                "boundary_interface_action": True,
                "trace_density": float(spawn_split["trace_density"]),
                "spawn_drive": float(spawn_split["spawn_drive"]),
                "spawn_drive_no_trace": float(spawn_split["spawn_drive_no_trace"]),
                "trace_ablation_spawn_delta": float(spawn_split["trace_ablation_spawn_delta"]),
                "split_drive": float(spawn_split["split_drive"]),
                "spawn_candidate": bool(spawn_split["spawn_candidate"]),
                "split_candidate": bool(spawn_split["split_candidate"]),
                "trace_mass": float(self.trace_field.sum()),
                "trace_peak": float(self.trace_field.max()),
                "death_cause": self.last_death_cause,
                "death_signals": dict(self.last_death_signals),
                "dead": bool(episode_dead),
                "spawn_events": int(spawn_events),
                "split_events": int(split_events),
                "death_events_step": int(len(dead_bodies)),
                "body_count": int(len(self.bodies)),
                "alive_body_count": int(len(self._alive_bodies())),
                "aux_updated_body_count": int(aux_stats["updated_body_count"]),
                "aux_action_counts": dict(aux_stats["action_counts"]),
                "aux_policy_source_counts": dict(aux_stats["policy_source_counts"]),
                "aux_mean_policy_entropy": float(aux_stats["mean_policy_entropy"]),
                "aux_nontrivial_action_count": int(aux_stats["nontrivial_action_count"]),
                "aux_challenge_action_count": int(aux_stats["challenge_action_count"]),
                "aux_conservative_action_count": int(aux_stats["conservative_action_count"]),
            }
        )
        return episode_dead

    def snapshot(self) -> dict[str, np.ndarray]:
        occupancy, boundary, permeability = self._population_body_fields()
        env_channels = self.env.environment_channels()
        external_channels = self.env.external_channels()
        species_sources = self.env.external_state.species_sources()
        species_fields = self.env.species_field_channels()
        return {
            "occupancy": occupancy.astype(np.float32),
            "boundary": boundary.astype(np.float32),
            "permeability": permeability.astype(np.float32),
            "env_channels": env_channels.astype(np.float32),
            "external_state": external_channels.astype(np.float32),
            "species_sources": species_sources.astype(np.float32),
            "species_fields": species_fields.astype(np.float32),
            "observation": self.last_observation.astype(np.float32),
            "sensor_gate": self.last_sensor_gate.astype(np.float32),
            "world_error": self.last_world_error.astype(np.float32),
            "boundary_error": self.last_boundary_error.astype(np.float32),
            "world_belief": self.world_belief.astype(np.float32),
            "world_logvar": self.world_logvar.astype(np.float32),
            "boundary_belief": self.boundary_belief.astype(np.float32),
            "boundary_logvar": self.boundary_logvar.astype(np.float32),
            "visual_cell_belief": self.visual_cells.belief.astype(np.float32),
            "visual_cell_logvar": self.visual_cells.logvar.astype(np.float32),
            "visual_cell_projected_belief": self.visual_cells.projected_belief().astype(np.float32),
            "visual_attention_map": self.visual_attention_map.astype(np.float32),
            "visual_attention_projected": self.visual_attention_projected.astype(np.float32),
            "mc_context_state": self.last_mc_context["context_state"].astype(np.float32),
            "mc_sequence_bias": self.last_mc_context["sequence_bias"].astype(np.float32),
            "trace_field": self.trace_field.astype(np.float32),
        }


def run_episode(
    output_root: str | Path,
    seed_catalog: str | Path,
    runtime_config: RuntimeConfig,
    env_config: EnvironmentConfig,
    trm_a_checkpoint: str | Path | None = None,
    trm_b_checkpoint: str | Path | None = None,
    module_specs: list[dict[str, Any]] | None = None,
    module_manifest: str | Path | None = None,
    adaptive_controller_config: AdaptiveControllerConfig | None = None,
    initial_lenia_params: dict[str, float] | None = None,
) -> Path:
    seed_everything(runtime_config.seed)
    rng = np.random.default_rng(runtime_config.seed)
    seeds = load_seed_catalog(seed_catalog)
    if not seeds:
        raise SystemExit(f"no seeds found in {seed_catalog}")
    seed = seeds[int(rng.integers(0, len(seeds)))]
    env = LeniaERIEEnvironment(seed, env_config, runtime_config, rng)
    if initial_lenia_params:
        env.set_lenia_params(
            m=initial_lenia_params.get("m"),
            s=initial_lenia_params.get("s"),
        )
    models = RuntimeModels(
        trm_a_checkpoint,
        trm_b_checkpoint,
        module_specs=module_specs,
        module_manifest=module_manifest,
    )
    runtime = ERIERuntime(env, runtime_config, rng, models=models)
    adaptive_controller = AdaptiveController(adaptive_controller_config)

    frames: list[dict[str, np.ndarray]] = []
    parameter_history: list[dict[str, Any]] = []
    for t in range(runtime_config.steps):
        dead = runtime.step(t)
        event = adaptive_controller.maybe_update(runtime, t)
        if event is not None:
            parameter_history.append(event)
        if t >= runtime_config.warmup_steps:
            frames.append(runtime.snapshot())
        if dead:
            break

    output_root = ensure_dir(output_root)
    episode_id = f"erie_{runtime_config.seed}_{seed.seed_id}"
    out_path = Path(output_root) / f"{episode_id}.npz"
    np.savez_compressed(
        out_path,
        occupancy=np.stack([f["occupancy"] for f in frames], axis=0),
        boundary=np.stack([f["boundary"] for f in frames], axis=0),
        permeability=np.stack([f["permeability"] for f in frames], axis=0),
        env_channels=np.stack([f["env_channels"] for f in frames], axis=0),
        external_state=np.stack([f["external_state"] for f in frames], axis=0),
        species_sources=np.stack([f["species_sources"] for f in frames], axis=0),
        species_fields=np.stack([f["species_fields"] for f in frames], axis=0),
        observation=np.stack([f["observation"] for f in frames], axis=0),
        sensor_gate=np.stack([f["sensor_gate"] for f in frames], axis=0),
        world_error=np.stack([f["world_error"] for f in frames], axis=0),
        boundary_error=np.stack([f["boundary_error"] for f in frames], axis=0),
        world_belief=np.stack([f["world_belief"] for f in frames], axis=0),
        world_logvar=np.stack([f["world_logvar"] for f in frames], axis=0),
        boundary_belief=np.stack([f["boundary_belief"] for f in frames], axis=0),
        boundary_logvar=np.stack([f["boundary_logvar"] for f in frames], axis=0),
        visual_cell_belief=np.stack([f["visual_cell_belief"] for f in frames], axis=0),
        visual_cell_logvar=np.stack([f["visual_cell_logvar"] for f in frames], axis=0),
        visual_cell_projected_belief=np.stack([f["visual_cell_projected_belief"] for f in frames], axis=0),
        visual_attention_map=np.stack([f["visual_attention_map"] for f in frames], axis=0),
        visual_attention_projected=np.stack([f["visual_attention_projected"] for f in frames], axis=0),
        mc_context_state=np.stack([f["mc_context_state"] for f in frames], axis=0),
        mc_sequence_bias=np.stack([f["mc_sequence_bias"] for f in frames], axis=0),
        trace_field=np.stack([f["trace_field"] for f in frames], axis=0),
    )
    summary = {
        "episode_id": episode_id,
        "seed_id": seed.seed_id,
        "source_name": seed.name,
        "multispecies_enabled": True,
        "species_roles": ["species_energy", "species_toxic", "species_niche"],
        "num_steps_requested": runtime_config.steps,
        "num_steps_executed": len(runtime.history),
        "num_recorded_frames": len(frames),
        "final_G": float(runtime.body.G),
        "final_B": float(runtime.body.B),
        "dead": bool(runtime.history[-1]["dead"]) if runtime.history else False,
        "final_death_cause": (
            str(runtime.history[-1].get("death_cause"))
            if runtime.history and bool(runtime.history[-1]["dead"])
            else None
        ),
        "final_alive": bool(runtime.body.alive),
        "final_body_count": int(len(runtime.bodies)),
        "final_challenge_ratio": float(runtime.challenge_ratio),
        "visual_cell_shape": list(runtime.visual_cells.belief.shape),
        "final_visual_attention": dict(runtime.last_visual_attention),
        "invalid_body_state_count": int(runtime.invalid_body_state_count),
        "boundary_interface_counts": dict(runtime.boundary_interface_counts),
        "population_event_counts": dict(runtime.population_event_counts),
        "death_cause_counts": death_cause_counts(
            runtime.history,
            expected_label=DEATH_CAUSE_EXPECTED,
            degenerate_label=DEATH_CAUSE_DEGENERATE,
            policy_forbidden_label=DEATH_CAUSE_POLICY_FORBIDDEN,
        ),
        "action_counts": {
            action: int(sum(1 for row in runtime.history if row["action"] == action))
            for action in (*ACTIONS, "no_action")
        },
        "runtime_config": asdict(runtime_config),
        "final_runtime_config": asdict(runtime.cfg),
        "environment_config": asdict(env_config),
        "environment_config_canonical": canonical_environment_config(env_config),
        "adaptive_controller": asdict(adaptive_controller.config),
        "adaptive_event_count": len(parameter_history),
        "final_lenia_params": {
            "m": float(runtime.env.params["m"]),
            "s": float(runtime.env.params["s"]),
            "R": int(runtime.env.params["R"]),
            "T": int(runtime.env.params["T"]),
            "b": [float(v) for v in runtime.env.params["b"]],
            "kn": int(runtime.env.params["kn"]),
            "gn": int(runtime.env.params["gn"]),
        },
        "trm_a_checkpoint": str(trm_a_checkpoint) if trm_a_checkpoint else None,
        "trm_b_checkpoint": str(trm_b_checkpoint) if trm_b_checkpoint else None,
        "module_manifest": str(module_manifest) if module_manifest else None,
        "modules": [
            {
                "id": module["id"],
                "name": module["name"],
                "role": module["role"],
                "checkpoint": module["checkpoint"],
                "primary": module["primary"],
            }
            for module in models.modules
        ],
        "primary_modules": {
            role: module["id"] for role, module in models._primary_by_role.items()
        },
        "secondary_modules": {
            role: [module["id"] for module in models.secondary_modules(role)]
            for role in sorted({module["role"] for module in models.modules if module.get("role")})
        },
    }
    summary.update(
        episode_metrics(
            runtime.history,
            runtime_config,
            actions=ACTIONS,
            policy_action_cost=_policy_action_cost,
            expected_death_label=DEATH_CAUSE_EXPECTED,
            degenerate_death_label=DEATH_CAUSE_DEGENERATE,
            policy_forbidden_death_label=DEATH_CAUSE_POLICY_FORBIDDEN,
        )
    )
    save_json(Path(output_root) / f"{episode_id}_summary.json", summary)
    save_json(Path(output_root) / f"{episode_id}_history.json", runtime.history)
    if adaptive_controller.config.enabled:
        save_json(Path(output_root) / f"{episode_id}_parameter_history.json", parameter_history)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal ERIE self-maintenance runtime on Lenia.")
    parser.add_argument(
        "--seed-catalog",
        default="data/lenia_official/animals2d_seeds.json",
        help="Path to exported Lenia seed catalog.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/erie_runtime",
        help="Directory to write episode arrays and JSON logs.",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260316)
    parser.add_argument("--lookahead-horizon", type=int, default=2)
    parser.add_argument("--lookahead-discount", type=float, default=0.85)
    parser.add_argument(
        "--viability-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument(
        "--action-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument(
        "--boundary-control-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument(
        "--action-gating-mode",
        choices=("analytic", "assistive", "module_primary"),
        default="assistive",
    )
    parser.add_argument(
        "--context-memory-mode",
        choices=("analytic", "assistive"),
        default="assistive",
    )
    parser.add_argument("--context-memory-window-size", type=int, default=8)
    parser.add_argument("--context-memory-residual-scale", type=float, default=0.35)
    parser.add_argument("--visual-cell-rows", type=int, default=8)
    parser.add_argument("--visual-cell-cols", type=int, default=8)
    parser.add_argument("--visual-cell-lateral-coupling", type=float, default=0.0)
    parser.add_argument("--visual-attention-error-gain", type=float, default=1.0)
    parser.add_argument("--visual-attention-uncertainty-gain", type=float, default=0.25)
    parser.add_argument("--visual-attention-temperature", type=float, default=8.0)
    parser.add_argument("--visual-attention-epistemic-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-controller", action="store_true")
    parser.add_argument("--adaptive-interval", type=int, default=4)
    parser.add_argument("--adaptive-window-size", type=int, default=8)
    parser.add_argument("--adaptive-min-steps", type=int, default=4)
    parser.add_argument("--adaptive-learning-rate", type=float, default=0.08)
    parser.add_argument("--adaptive-target-homeostatic-error", type=float, default=0.18)
    parser.add_argument("--adaptive-min-policy-entropy", type=float, default=0.75)
    parser.add_argument("--adaptive-target-energy-contact", type=float, default=0.18)
    parser.add_argument("--adaptive-max-stress-contact", type=float, default=0.35)
    parser.add_argument("--disable-adaptive-runtime", action="store_true")
    parser.add_argument("--disable-adaptive-lenia", action="store_true")
    add_environment_config_args(parser)
    parser.add_argument("--trm-a-checkpoint", default=None)
    parser.add_argument("--trm-b-checkpoint", default=None)
    parser.add_argument("--module-manifest", default=None)
    parser.add_argument(
        "--policy-mode",
        choices=("closed_loop", "random", "no_action"),
        default="closed_loop",
    )
    args = parser.parse_args()

    runtime_config = RuntimeConfig(
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        lookahead_horizon=args.lookahead_horizon,
        lookahead_discount=args.lookahead_discount,
        viability_mode=args.viability_mode,
        action_mode=args.action_mode,
        action_gating_mode=args.action_gating_mode,
        boundary_control_mode=args.boundary_control_mode,
        context_memory_mode=args.context_memory_mode,
        context_memory_window_size=args.context_memory_window_size,
        context_memory_residual_scale=args.context_memory_residual_scale,
        visual_cell_rows=args.visual_cell_rows,
        visual_cell_cols=args.visual_cell_cols,
        visual_cell_lateral_coupling=args.visual_cell_lateral_coupling,
        visual_attention_error_gain=args.visual_attention_error_gain,
        visual_attention_uncertainty_gain=args.visual_attention_uncertainty_gain,
        visual_attention_temperature=args.visual_attention_temperature,
        visual_attention_epistemic_gain=args.visual_attention_epistemic_gain,
        use_trm_a=bool(args.trm_a_checkpoint),
        use_trm_b=bool(args.trm_b_checkpoint),
        policy_mode=args.policy_mode,
    )
    adaptive_config = AdaptiveControllerConfig(
        enabled=bool(args.adaptive_controller),
        interval=args.adaptive_interval,
        window_size=args.adaptive_window_size,
        min_steps=args.adaptive_min_steps,
        learning_rate=args.adaptive_learning_rate,
        target_homeostatic_error=args.adaptive_target_homeostatic_error,
        min_policy_entropy=args.adaptive_min_policy_entropy,
        target_energy_contact=args.adaptive_target_energy_contact,
        max_stress_contact=args.adaptive_max_stress_contact,
        adapt_runtime=not bool(args.disable_adaptive_runtime),
        adapt_lenia=not bool(args.disable_adaptive_lenia),
    )
    env_config = environment_config_from_args(args)
    episode_path = run_episode(
        args.output_root,
        args.seed_catalog,
        runtime_config,
        env_config,
        trm_a_checkpoint=args.trm_a_checkpoint,
        trm_b_checkpoint=args.trm_b_checkpoint,
        module_manifest=args.module_manifest,
        adaptive_controller_config=adaptive_config,
    )
    print(f"wrote ERIE runtime episode: {episode_path}")


if __name__ == "__main__":
    main()
