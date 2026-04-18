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
from .trm_input_views import build_trm_bp_input_view, build_trm_mc_input_view, extract_centered_patch
from .trm_input_views import build_trm_ag_input_view


ACTIONS = ("approach", "withdraw", "intake", "seal", "reconfigure")


@dataclass(frozen=True)
class EnvironmentConfig:
    image_size: int = 64
    target_radius: int = 12
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
    dead_count: int = 0


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
            env_config.resource_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
        )
        thermal_stress = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.hazard_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
        )
        toxicity = _gaussian_blob_field(
            rng,
            env_config.image_size,
            max(1, int(round(env_config.hazard_patches * 0.75))),
            env_config.field_sigma_min * 0.8,
            env_config.field_sigma_max * 0.9,
        )
        niche_stability = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.shelter_patches,
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
        )
        channels = self.env.environment_channels().shape[-1]
        shape_world = (self.env.env_config.image_size, self.env.env_config.image_size, channels)
        self.world_belief = np.zeros(shape_world, dtype=np.float32)
        self.world_logvar = np.full(shape_world, runtime_config.world_logvar_init, dtype=np.float32)
        self.boundary_belief = np.zeros(
            (self.env.env_config.image_size, self.env.env_config.image_size, 2), dtype=np.float32
        )
        self.boundary_logvar = np.full(
            (self.env.env_config.image_size, self.env.env_config.image_size, 2),
            runtime_config.boundary_logvar_init,
            dtype=np.float32,
        )
        self.policy_belief = np.full(len(ACTIONS), 1.0 / len(ACTIONS), dtype=np.float32)
        self.history: list[dict[str, Any]] = []
        self.prev_lenia_obs = self.env.lenia_multistate().astype(np.float32)
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

    def _body_fields(self, body: BodyState | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _body_fields(
            body or self.body,
            self.env.env_config.image_size,
            self.cfg.occupancy_softness,
        )

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
        occupancy, boundary, permeability = self._body_fields()
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

        world_error = sensor_gate * (observation - world_prior)
        boundary_error = boundary_obs - boundary_prior

        precision_w = np.clip(np.exp(-world_logvar), self.cfg.p_min, self.cfg.p_max)
        precision_b = np.clip(np.exp(-self.boundary_logvar), self.cfg.p_min, self.cfg.p_max)

        self.world_belief = _clip01(world_prior + self.cfg.lambda_w * precision_w * world_error)
        self.boundary_belief = _clip01(boundary_prior + self.cfg.lambda_b * precision_b * boundary_error)

        # More reliable observations reduce uncertainty; unsensed regions drift upward.
        sensed = sensor_gate
        self.world_logvar = np.clip(
            self.world_logvar + self.cfg.world_logvar_drift - 0.18 * sensed,
            -4.0,
            2.5,
        ).astype(np.float32)
        self.boundary_logvar = np.clip(
            self.boundary_logvar + self.cfg.boundary_logvar_drift - 0.20 * boundary_obs[..., :1],
            -4.0,
            2.5,
        ).astype(np.float32)
        world_reconstruction = float(
            np.mean(0.5 * ((world_error**2) * precision_w + np.clip(world_logvar, -6.0, 4.0)))
        )
        world_complexity = float(
            np.mean(
                0.5
                * (((self.world_belief - world_prior) ** 2) * precision_w)
            )
        )
        boundary_reconstruction = float(
            np.mean(
                0.5 * ((boundary_error**2) * precision_b + np.clip(self.boundary_logvar, -6.0, 4.0))
            )
        )
        boundary_complexity = float(
            np.mean(
                0.5
                * (((self.boundary_belief - boundary_prior) ** 2) * precision_b)
            )
        )
        world_total = float(world_reconstruction + world_complexity)
        boundary_total = float(boundary_reconstruction + boundary_complexity)
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
        return float(self.cfg.epistemic_scale * _mean_masked(world_unc * cue_grad, interface))

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
        base_logits = (-self.cfg.beta_pi * scores.astype(np.float32)).astype(np.float32)
        diagnostics = {
            "base_logits": base_logits.astype(np.float32),
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
            pre_ag_logits = (residual_logits + context_bias).astype(np.float32)
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
                    final_logits = pre_ag_logits.astype(np.float32)
                    if ag_control_mode == 2:
                        soft_threshold = 0.45
                        hard_threshold = 0.60
                        prune_scale = 8.0
                    elif ag_control_mode == 0:
                        soft_threshold = 0.75
                        hard_threshold = 0.90
                        prune_scale = 3.0
                    else:
                        soft_threshold = 0.60
                        hard_threshold = 0.80
                        prune_scale = 5.0
                    inhibition_excess = np.clip(ag_inhibition_mask - soft_threshold, 0.0, 1.0).astype(np.float32)
                    if float(np.max(inhibition_excess)) > 0.0:
                        final_logits = (final_logits - prune_scale * inhibition_excess).astype(np.float32)
                    hard_gate_mask = ag_inhibition_mask >= hard_threshold
                    if bool(np.any(hard_gate_mask)):
                        gate_floor = float(np.min(final_logits) - 7.0)
                        final_logits = np.where(hard_gate_mask, gate_floor, final_logits).astype(np.float32)
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
                final_logits = pre_ag_logits.astype(np.float32)
                if ag_control_mode == 2:
                    soft_threshold = 0.45
                    hard_threshold = 0.60
                    prune_scale = 8.0
                elif ag_control_mode == 0:
                    soft_threshold = 0.75
                    hard_threshold = 0.90
                    prune_scale = 3.0
                else:
                    soft_threshold = 0.60
                    hard_threshold = 0.80
                    prune_scale = 5.0
                inhibition_excess = np.clip(ag_inhibition_mask - soft_threshold, 0.0, 1.0).astype(np.float32)
                if float(np.max(inhibition_excess)) > 0.0:
                    final_logits = (final_logits - prune_scale * inhibition_excess).astype(np.float32)
                hard_gate_mask = ag_inhibition_mask >= hard_threshold
                if bool(np.any(hard_gate_mask)):
                    gate_floor = float(np.min(final_logits) - 7.0)
                    final_logits = np.where(hard_gate_mask, gate_floor, final_logits).astype(np.float32)
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
        next_body = self._prospective_body(action)
        next_body, bp_meta = self._apply_bp_control(action, next_body, context_bias=context_bias)
        self.body = next_body
        G_next, B_next = self._predicted_viability(self.body, action)
        self.body.G = G_next
        self.body.B = B_next
        self.last_bp_control = dict(bp_meta)
        return bp_meta

    def _update_death(self) -> bool:
        if self.body.G < self.cfg.tau_G or self.body.B < self.cfg.tau_B:
            self.body.dead_count += 1
        else:
            self.body.dead_count = 0
        return self.body.dead_count >= self.cfg.k_irrev

    def step(self, t: int) -> bool:
        self.env.step_lenia()
        observation, sensor_gate, occupancy, boundary = self._observe()
        _, _, permeability = self._body_fields()
        boundary_obs = np.stack([boundary, permeability], axis=-1).astype(np.float32)
        contact = self._contact_stats(self.body)
        species_contact = self._species_contact_stats(self.body)
        world_error, boundary_error = self._belief_update(observation, sensor_gate, boundary_obs)

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
        bp_meta = self._apply_action(action, context_bias=current_mc_context)
        self.env.update_fields(self.body, action)
        dead = self._update_death()
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
                "homeostatic_error": _homeostatic_error(self.body.G, self.body.B, self.cfg),
                "centroid_y": float(self.body.centroid_y),
                "centroid_x": float(self.body.centroid_x),
                "aperture_angle": float(self.body.aperture_angle),
                "aperture_gain": float(self.body.aperture_gain),
                "dead": bool(dead),
            }
        )
        return dead

    def snapshot(self) -> dict[str, np.ndarray]:
        occupancy, boundary, permeability = self._body_fields()
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
            "mc_context_state": self.last_mc_context["context_state"].astype(np.float32),
            "mc_sequence_bias": self.last_mc_context["sequence_bias"].astype(np.float32),
        }


def _homeostatic_error(G: float, B: float, cfg: RuntimeConfig) -> float:
    return float(abs(G - cfg.G_target) + abs(B - cfg.B_target))


def _episode_metrics(history: list[dict[str, Any]], cfg: RuntimeConfig) -> dict[str, float]:
    if not history:
        return {
            "mean_G": 0.0,
            "mean_B": 0.0,
            "survival_fraction": 0.0,
            "final_homeostatic_error": _homeostatic_error(cfg.G0, cfg.B0, cfg),
            "mean_homeostatic_error": _homeostatic_error(cfg.G0, cfg.B0, cfg),
            "action_cost_total": 0.0,
            "action_cost_mean": 0.0,
            "mean_policy_entropy": 0.0,
            "mean_contact_energy": 0.0,
            "mean_contact_thermal": 0.0,
            "mean_contact_toxicity": 0.0,
            "mean_contact_niche": 0.0,
            "mean_contact_species_energy": 0.0,
            "mean_contact_species_thermal": 0.0,
            "mean_contact_species_toxicity": 0.0,
            "mean_contact_species_niche": 0.0,
            "mean_contact_resource": 0.0,
            "mean_contact_hazard": 0.0,
            "mean_contact_shelter": 0.0,
            "action_diversity": 0.0,
        }

    g_values = np.array([float(row["G"]) for row in history], dtype=np.float32)
    b_values = np.array([float(row["B"]) for row in history], dtype=np.float32)
    action_costs = np.array([_policy_action_cost(str(row["action"])) for row in history], dtype=np.float32)
    errors = np.abs(g_values - float(cfg.G_target)) + np.abs(b_values - float(cfg.B_target))
    policy_entropy = np.array([float(row["policy_entropy"]) for row in history], dtype=np.float32)
    contact_energy = np.array([float(row["contact_energy"]) for row in history], dtype=np.float32)
    contact_thermal = np.array([float(row["contact_thermal"]) for row in history], dtype=np.float32)
    contact_toxicity = np.array([float(row["contact_toxicity"]) for row in history], dtype=np.float32)
    contact_niche = np.array([float(row["contact_niche"]) for row in history], dtype=np.float32)
    contact_species_energy = np.array([float(row["contact_species_energy"]) for row in history], dtype=np.float32)
    contact_species_thermal = np.array([float(row["contact_species_thermal"]) for row in history], dtype=np.float32)
    contact_species_toxicity = np.array([float(row["contact_species_toxicity"]) for row in history], dtype=np.float32)
    contact_species_niche = np.array([float(row["contact_species_niche"]) for row in history], dtype=np.float32)
    contact_resource = np.array([float(row["contact_resource"]) for row in history], dtype=np.float32)
    contact_hazard = np.array([float(row["contact_hazard"]) for row in history], dtype=np.float32)
    contact_shelter = np.array([float(row["contact_shelter"]) for row in history], dtype=np.float32)
    action_labels = [str(row["action"]) for row in history]
    counts = np.array([action_labels.count(action) for action in (*ACTIONS, "no_action")], dtype=np.float32)
    probs = counts / max(float(counts.sum()), 1.0)
    action_diversity = _entropy(probs) / math.log(len(probs))
    return {
        "mean_G": float(g_values.mean()),
        "mean_B": float(b_values.mean()),
        "survival_fraction": float(len(history) / max(cfg.steps, 1)),
        "final_homeostatic_error": float(errors[-1]),
        "mean_homeostatic_error": float(errors.mean()),
        "action_cost_total": float(action_costs.sum()),
        "action_cost_mean": float(action_costs.mean()),
        "mean_policy_entropy": float(policy_entropy.mean()),
        "mean_contact_energy": float(contact_energy.mean()),
        "mean_contact_thermal": float(contact_thermal.mean()),
        "mean_contact_toxicity": float(contact_toxicity.mean()),
        "mean_contact_niche": float(contact_niche.mean()),
        "mean_contact_species_energy": float(contact_species_energy.mean()),
        "mean_contact_species_thermal": float(contact_species_thermal.mean()),
        "mean_contact_species_toxicity": float(contact_species_toxicity.mean()),
        "mean_contact_species_niche": float(contact_species_niche.mean()),
        "mean_contact_resource": float(contact_resource.mean()),
        "mean_contact_hazard": float(contact_hazard.mean()),
        "mean_contact_shelter": float(contact_shelter.mean()),
        "action_diversity": float(action_diversity),
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
) -> Path:
    seed_everything(runtime_config.seed)
    rng = np.random.default_rng(runtime_config.seed)
    seeds = load_seed_catalog(seed_catalog)
    if not seeds:
        raise SystemExit(f"no seeds found in {seed_catalog}")
    seed = seeds[int(rng.integers(0, len(seeds)))]
    env = LeniaERIEEnvironment(seed, env_config, runtime_config, rng)
    models = RuntimeModels(
        trm_a_checkpoint,
        trm_b_checkpoint,
        module_specs=module_specs,
        module_manifest=module_manifest,
    )
    runtime = ERIERuntime(env, runtime_config, rng, models=models)

    frames: list[dict[str, np.ndarray]] = []
    for t in range(runtime_config.steps):
        dead = runtime.step(t)
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
        mc_context_state=np.stack([f["mc_context_state"] for f in frames], axis=0),
        mc_sequence_bias=np.stack([f["mc_sequence_bias"] for f in frames], axis=0),
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
        "action_counts": {
            action: int(sum(1 for row in runtime.history if row["action"] == action))
            for action in (*ACTIONS, "no_action")
        },
        "runtime_config": asdict(runtime_config),
        "environment_config": asdict(env_config),
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
    summary.update(_episode_metrics(runtime.history, runtime_config))
    save_json(Path(output_root) / f"{episode_id}_summary.json", summary)
    save_json(Path(output_root) / f"{episode_id}_history.json", runtime.history)
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
    parser.add_argument("--resource-patches", type=int, default=3)
    parser.add_argument("--hazard-patches", type=int, default=3)
    parser.add_argument("--shelter-patches", type=int, default=1)
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
        use_trm_a=bool(args.trm_a_checkpoint),
        use_trm_b=bool(args.trm_b_checkpoint),
        policy_mode=args.policy_mode,
    )
    env_config = EnvironmentConfig(
        resource_patches=args.resource_patches,
        hazard_patches=args.hazard_patches,
        shelter_patches=args.shelter_patches,
    )
    episode_path = run_episode(
        args.output_root,
        args.seed_catalog,
        runtime_config,
        env_config,
        trm_a_checkpoint=args.trm_a_checkpoint,
        trm_b_checkpoint=args.trm_b_checkpoint,
        module_manifest=args.module_manifest,
    )
    print(f"wrote ERIE runtime episode: {episode_path}")


if __name__ == "__main__":
    main()
