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
    shelter_stability: float = 1.0


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
    contact_w_hazard: float = 0.9
    contact_w_resource: float = 0.25
    contact_w_shelter: float = 0.35
    contact_delta_w_hazard: float = 0.8
    contact_delta_w_resource: float = 0.3
    contact_delta_w_shelter: float = 0.4
    ambiguity_w_boundary: float = 0.5
    epistemic_scale: float = 1.0
    viability_mode: str = "assistive"
    action_mode: str = "assistive"
    viability_monitor_blend: float = 0.35
    action_model_residual_scale: float = 1.0
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
        self.trm_a_config: TRMModelConfig | None = None
        self.trm_b_config: TRMModelConfig | None = None
        self.trm_vm_config: TRMModelConfig | None = None
        self.trm_as_config: TRMModelConfig | None = None
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - float(np.max(x))
    exp = np.exp(z)
    total = float(exp.sum())
    if total <= 0.0:
        return np.full_like(x, 1.0 / len(x))
    return exp / total


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
        cfg.contact_w_hazard * next_contact["hazard"]
        - cfg.contact_w_resource * next_contact["resource"]
        - cfg.contact_w_shelter * next_contact["shelter"]
    )
    delta = (
        cfg.contact_delta_w_hazard * max(0.0, next_contact["hazard"] - current_contact["hazard"])
        - cfg.contact_delta_w_resource * max(0.0, next_contact["resource"] - current_contact["resource"])
        - cfg.contact_delta_w_shelter * max(0.0, next_contact["shelter"] - current_contact["shelter"])
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
        self.scalar_state = scalar_seed.astype(np.float32)
        self.prev_scalar_state = self.scalar_state.copy()

        self.resource = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.resource_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
        )
        self.hazard = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.hazard_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
        )
        self.shelter = _gaussian_blob_field(
            rng,
            env_config.image_size,
            env_config.shelter_patches,
            env_config.field_sigma_min,
            env_config.field_sigma_max,
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

    def lenia_multistate(self) -> np.ndarray:
        return derive_multichannel_state(self.prev_scalar_state, self.scalar_state, self.params)

    def step_lenia(self) -> None:
        next_state = lenia_step(self.scalar_state, self.kernel_fft, self.params)
        self.prev_scalar_state = self.scalar_state
        self.scalar_state = next_state

    def environment_channels(self) -> np.ndarray:
        multi = self.lenia_multistate()
        env = np.stack([self.resource, self.hazard, self.shelter], axis=-1).astype(np.float32)
        return np.concatenate([multi, env], axis=-1).astype(np.float32)

    def update_fields(self, body: BodyState, action: str) -> None:
        _, boundary, permeability = _body_fields(
            body, self.env_config.image_size, self.runtime_config.occupancy_softness
        )
        contact_mask = boundary * np.clip(permeability, 0.0, 1.0)
        if action == "intake":
            consume = np.minimum(self.resource, 0.06 * contact_mask)
            self.resource = _clip01(self.resource - consume)
        self.resource = _clip01(self.resource + self.env_config.resource_regen * (1.0 - self.resource))
        self.hazard = _clip01(
            self.hazard
            + gaussian_noise(
                self.rng,
                self.hazard.shape,
                self.env_config.hazard_drift_sigma,
            )
        )


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

    def _observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        occupancy, boundary, permeability = self._body_fields()
        env_channels = self.env.environment_channels()
        sensor_gate = np.clip(permeability[..., None] + 0.05 * occupancy[..., None], 0.0, 1.0)
        shelter_bonus = self.env.shelter[..., None]
        noise_scale = np.clip(
            self.cfg.observation_noise * (1.0 + self.env.hazard[..., None] - 0.5 * shelter_bonus),
            0.002,
            0.05,
        )
        noisy = _clip01(env_channels + gaussian_noise(self.rng, env_channels.shape, 1.0) * noise_scale)
        observation = sensor_gate * noisy + (1.0 - sensor_gate) * self.world_belief
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
        self.prev_lenia_obs = lenia_obs
        self._refresh_world_prior_from_trm_a()
        return world_error, boundary_error

    def _prospective_body_for_fields(
        self,
        body: BodyState,
        action: str | None,
        resource: np.ndarray,
        hazard: np.ndarray,
        shelter: np.ndarray,
    ) -> BodyState:
        body = _copy_body(body)
        if action is None or action == "no_action":
            return body
        gy_r, gx_r = _gradients(resource, body.centroid_y, body.centroid_x)
        gy_h, gx_h = _gradients(hazard, body.centroid_y, body.centroid_x)
        gy_s, gx_s = _gradients(shelter, body.centroid_y, body.centroid_x)
        if action == "approach":
            dy, dx = _normalize_vec(gy_r - 0.6 * gy_h, gx_r - 0.6 * gx_h)
            body.centroid_y += self.cfg.move_step * dy
            body.centroid_x += self.cfg.move_step * dx
        elif action == "withdraw":
            dy, dx = _normalize_vec(0.8 * gy_h - 0.3 * gy_s, 0.8 * gx_h - 0.3 * gx_s)
            body.centroid_y -= self.cfg.move_step * dy
            body.centroid_x -= self.cfg.move_step * dx
        elif action == "intake":
            body.aperture_gain = min(1.0, body.aperture_gain + 0.12)
        elif action == "seal":
            body.aperture_gain = max(self.cfg.base_permeability, body.aperture_gain - 0.15)
            body.B = min(1.0, body.B + 0.05)
        elif action == "reconfigure":
            target_angle = math.atan2(gy_r - gy_h + 0.2 * gy_s, gx_r - gx_h + 0.2 * gx_s)
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
            self.env.resource,
            self.env.hazard,
            self.env.shelter,
        )

    def _contact_stats(
        self,
        body: BodyState,
        resource: np.ndarray | None = None,
        hazard: np.ndarray | None = None,
        shelter: np.ndarray | None = None,
    ) -> dict[str, float]:
        resource = self.env.resource if resource is None else resource
        hazard = self.env.hazard if hazard is None else hazard
        shelter = self.env.shelter if shelter is None else shelter
        _, boundary, permeability = self._body_fields(body)
        interface = np.clip(boundary * permeability, 0.0, 1.0)
        return {
            "resource": _mean_masked(resource, interface),
            "hazard": _mean_masked(hazard, interface),
            "shelter": _mean_masked(shelter, interface),
            "interface_mass": float(interface.sum()),
        }

    def _uncertainty_state(self) -> np.ndarray:
        _, boundary, permeability = self._body_fields()
        interface = np.clip(boundary * np.maximum(permeability, 0.05), 0.0, 1.0)
        world_unc = float(_mean_masked(np.mean(np.exp(np.clip(self.world_logvar, -6.0, 2.0)), axis=-1), interface))
        boundary_unc = float(
            _mean_masked(np.mean(np.exp(np.clip(self.boundary_logvar, -6.0, 2.0)), axis=-1), interface)
        )
        contact = self._contact_stats(self.body)
        return np.array([world_unc, boundary_unc, float(contact["hazard"])], dtype=np.float32)

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
        with torch.no_grad():
            outputs = self.models.trm_vm(
                torch.from_numpy(analytic_state[None, ...]),
                torch.from_numpy(
                    np.array([[contact["resource"], contact["hazard"], contact["shelter"]]], dtype=np.float32)
                ),
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
        resource: np.ndarray | None = None,
        hazard: np.ndarray | None = None,
        shelter: np.ndarray | None = None,
    ) -> float:
        resource = self.env.resource if resource is None else resource
        hazard = self.env.hazard if hazard is None else hazard
        shelter = self.env.shelter if shelter is None else shelter
        _, boundary, permeability = self._body_fields(body)
        interface = np.clip(boundary * np.maximum(permeability, 0.05), 0.0, 1.0)
        world_unc = np.mean(np.exp(self.world_logvar), axis=-1)
        gy_r, gx_r = np.gradient(resource)
        gy_h, gx_h = np.gradient(hazard)
        gy_s, gx_s = np.gradient(shelter)
        cue_grad = np.sqrt(
            gy_r * gy_r + gx_r * gx_r + gy_h * gy_h + gx_h * gx_h + 0.5 * (gy_s * gy_s + gx_s * gx_s)
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
        resource: np.ndarray,
        hazard: np.ndarray,
        shelter: np.ndarray,
    ) -> tuple[float, float]:
        contact = self._contact_stats(body, resource=resource, hazard=hazard, shelter=shelter)
        intake_bonus = 1.25 if action == "intake" else 1.0
        leakage_penalty = 1.25 if action == "intake" else 1.0
        seal_gain = 0.05 if action == "seal" else 0.0
        reconfigure_gain = 0.03 if action == "reconfigure" else 0.0
        G_next = np.clip(
            body.G
            - self.cfg.mu_G
            + self.cfg.alpha_R * contact["resource"] * intake_bonus
            - _policy_action_cost(action),
            0.0,
            1.0,
        )
        B_next = np.clip(
            body.B
            - self.cfg.mu_B
            - self.cfg.alpha_H * contact["hazard"] * leakage_penalty
            + self.cfg.alpha_S * contact["shelter"]
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
            self.env.resource,
            self.env.hazard,
            self.env.shelter,
        )

    def _updated_fields_for_policy(
        self,
        body: BodyState,
        action: str | None,
        resource: np.ndarray,
        hazard: np.ndarray,
        shelter: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_resource = resource.copy()
        next_hazard = hazard.copy()
        next_shelter = shelter.copy()
        _, boundary, permeability = self._body_fields(body)
        contact_mask = boundary * np.clip(permeability, 0.0, 1.0)
        if action == "intake":
            consume = np.minimum(next_resource, 0.06 * contact_mask)
            next_resource = _clip01(next_resource - consume)
        next_resource = _clip01(next_resource + self.env.env_config.resource_regen * (1.0 - next_resource))
        return next_resource, next_hazard, next_shelter

    def _single_step_policy_terms(
        self,
        body: BodyState,
        action: str | None,
        resource: np.ndarray,
        hazard: np.ndarray,
        shelter: np.ndarray,
    ) -> tuple[float, dict[str, float], BodyState, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        current_contact = self._contact_stats(body, resource=resource, hazard=hazard, shelter=shelter)
        next_body = self._prospective_body_for_fields(body, action, resource, hazard, shelter)
        G_next, B_next = self._predicted_viability_for_fields(next_body, action, resource, hazard, shelter)
        next_body.G = G_next
        next_body.B = B_next
        death_risk = float(G_next < self.cfg.tau_G or B_next < self.cfg.tau_B)
        next_contact = self._contact_stats(next_body, resource=resource, hazard=hazard, shelter=shelter)
        contact_risk = _contact_risk_proxy(current_contact, next_contact, self.cfg)
        risk = _risk_proxy(G_next, B_next, death_risk, self.cfg) + contact_risk
        ambiguity = self._ambiguity_proxy(next_body)
        epistemic = self._epistemic_proxy(next_body, resource=resource, hazard=hazard, shelter=shelter)
        score = risk + ambiguity - epistemic
        next_fields = self._updated_fields_for_policy(next_body, action, resource, hazard, shelter)
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
        resource: np.ndarray,
        hazard: np.ndarray,
        shelter: np.ndarray,
        horizon: int,
    ) -> tuple[float, float, dict[str, float]]:
        immediate_score, diagnostics, next_body, next_fields = self._single_step_policy_terms(
            body,
            action,
            resource,
            hazard,
            shelter,
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
                self.env.resource,
                self.env.hazard,
                self.env.shelter,
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
            "source": "analytic",
            "base_logits": base_logits.astype(np.float32),
        }
        if self.models.trm_as is None or self.cfg.action_mode == "analytic":
            policy = _softmax(base_logits)
            action = ACTIONS[int(np.argmax(policy))]
            diagnostics["final_logits"] = base_logits.astype(np.float32)
            return policy.astype(np.float32), action, diagnostics

        torch = self.models.torch
        assert torch is not None
        uncertainty_state = self._uncertainty_state()
        with torch.no_grad():
            outputs = self.models.trm_as(
                torch.from_numpy(viability_monitor["state"][None, ...].astype(np.float32)),
                torch.from_numpy(scores[None, ...].astype(np.float32)),
                torch.from_numpy(uncertainty_state[None, ...].astype(np.float32)),
            )
        residual_logits = outputs["policy_logits"][0].cpu().numpy().astype(np.float32)
        residual_logits = residual_logits - float(np.mean(residual_logits))
        model_precision = float(outputs["module_precision"][0].cpu().item())
        if self.cfg.action_mode == "module_primary":
            final_logits = residual_logits.astype(np.float32)
            policy = _softmax(final_logits)
            action = ACTIONS[int(np.argmax(policy))]
            diagnostics.update(
                {
                    "source": "trm_as_primary",
                    "model_precision": model_precision,
                    "residual_logits": residual_logits.astype(np.float32),
                    "final_logits": final_logits.astype(np.float32),
                    "uncertainty_state": uncertainty_state.astype(np.float32),
                }
            )
            return policy.astype(np.float32), action, diagnostics
        final_logits = base_logits + (
            self.cfg.action_model_residual_scale * model_precision * residual_logits
        ).astype(np.float32)
        policy = _softmax(final_logits)
        action = ACTIONS[int(np.argmax(policy))]
        diagnostics.update(
            {
                "source": "trm_as",
                "model_precision": model_precision,
                "residual_logits": residual_logits.astype(np.float32),
                "final_logits": final_logits.astype(np.float32),
                "uncertainty_state": uncertainty_state.astype(np.float32),
            }
        )
        return policy.astype(np.float32), action, diagnostics

    def _apply_action(self, action: str | None) -> None:
        self.body = self._prospective_body(action)
        G_next, B_next = self._predicted_viability(self.body, action)
        self.body.G = G_next
        self.body.B = B_next

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

        self._apply_action(action)
        self.env.update_fields(self.body, action)
        dead = self._update_death()

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
                "policy_entropy": _entropy(policy),
                "monitor_viability_source": viability_monitor["source"],
                "monitor_viability_risk": float(viability_monitor["risk"]),
                "monitor_viability_precision": float(viability_monitor["precision"]),
                "monitor_homeostatic_error": float(viability_monitor["homeostatic_error"]),
                "monitor_G": float(viability_monitor["state"][0]),
                "monitor_B": float(viability_monitor["state"][1]),
                "sensor_gate_mean": float(sensor_gate.mean()),
                "world_error_mean": float(np.mean(np.abs(world_error))),
                "boundary_error_mean": float(np.mean(np.abs(boundary_error))),
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
        return {
            "occupancy": occupancy.astype(np.float32),
            "boundary": boundary.astype(np.float32),
            "permeability": permeability.astype(np.float32),
            "env_channels": env_channels.astype(np.float32),
            "world_belief": self.world_belief.astype(np.float32),
            "world_logvar": self.world_logvar.astype(np.float32),
            "boundary_belief": self.boundary_belief.astype(np.float32),
            "boundary_logvar": self.boundary_logvar.astype(np.float32),
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
        world_belief=np.stack([f["world_belief"] for f in frames], axis=0),
        world_logvar=np.stack([f["world_logvar"] for f in frames], axis=0),
        boundary_belief=np.stack([f["boundary_belief"] for f in frames], axis=0),
        boundary_logvar=np.stack([f["boundary_logvar"] for f in frames], axis=0),
    )
    summary = {
        "episode_id": episode_id,
        "seed_id": seed.seed_id,
        "source_name": seed.name,
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
