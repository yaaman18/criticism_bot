from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for TRM training. Install `torch` first."
        ) from exc
    return torch, nn, F


@dataclass(frozen=True)
class TRMModelConfig:
    image_size: int = 64
    patch_size: int = 8
    dim: int = 256
    recursions: int = 6
    num_heads: int = 4
    mlp_ratio: int = 4
    halt_threshold: float = 0.7
    in_channels: int = 5
    out_channels: int | None = None
    boundary_in_channels_total: int = 15
    z_dim: int = 32
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    max_params: int = 7_000_000


def get_trm_registry() -> dict[str, dict[str, Any]]:
    return {
        "trm_a": {
            "role": "world_model",
            "builder": build_trm_a,
            "adapter": _adapt_trm_a_outputs,
        },
        "trm_b": {
            "role": "boundary_model",
            "builder": build_trm_b,
            "adapter": _adapt_trm_b_outputs,
        },
        "trm_vm": {
            "role": "viability_monitor",
            "builder": build_trm_vm,
            "adapter": _adapt_trm_vm_outputs,
        },
        "trm_as": {
            "role": "action_scoring",
            "builder": build_trm_as,
            "adapter": _adapt_trm_as_outputs,
        },
        "trm_ag": {
            "role": "action_gating",
            "builder": build_trm_ag,
            "adapter": _adapt_trm_ag_outputs,
        },
        "trm_bp": {
            "role": "boundary_permeability_control",
            "builder": build_trm_bp,
            "adapter": _adapt_trm_bp_outputs,
        },
        "trm_mc": {
            "role": "memory_context",
            "builder": build_trm_mc,
            "adapter": _adapt_trm_mc_outputs,
        },
    }


def build_trm(name: str, config: TRMModelConfig):
    registry = get_trm_registry()
    if name not in registry:
        raise SystemExit(f"unknown TRM module name: {name}")
    return registry[name]["builder"](config)


def adapt_trm_outputs(name: str, outputs: dict[str, Any]) -> dict[str, Any]:
    registry = get_trm_registry()
    if name not in registry:
        raise SystemExit(f"unknown TRM module name: {name}")
    adapted = registry[name]["adapter"](outputs)
    adapted["module_name"] = name
    adapted["module_role"] = registry[name]["role"]
    return adapted


def _adapt_trm_a_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    module_precision = outputs.get("module_precision")
    if module_precision is None:
        torch, _, _ = require_torch()
        pred_logvar = outputs["pred_logvar_t1"]
        module_precision = torch.exp(-pred_logvar).mean(dim=(1, 2, 3))
    pred_state = outputs["pred_state_t1"]
    pred_error = outputs.get("pred_error_t1")
    return {
        "module_state": outputs["pred_patches_t1"],
        "module_precision": module_precision,
        "module_error": pred_error,
        "module_aux": {
            "pred_state_t1": pred_state,
            "pred_logvar_t1": outputs.get("pred_logvar_t1"),
            "halt_prob": outputs.get("halt_prob"),
            "latent": outputs.get("latent"),
        },
    }


def _adapt_trm_b_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    torch, _, _ = require_torch()
    boundary_map = outputs["boundary_map"]
    permeability_map = outputs["permeability_map"]
    halt_prob = outputs.get("halt_prob")
    if halt_prob is None:
        module_precision = boundary_map.mean(dim=(1, 2, 3))
    else:
        module_precision = halt_prob.mean(dim=1)
    module_error = 1.0 - torch.mean(boundary_map, dim=(1, 2, 3))
    return {
        "module_state": outputs["boundary_state"],
        "module_precision": module_precision,
        "module_error": module_error,
        "module_aux": {
            "boundary_map": boundary_map,
            "permeability_map": permeability_map,
            "halt_prob": halt_prob,
        },
    }


def _adapt_trm_vm_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_state": outputs["viability_latent"],
        "module_precision": outputs["module_precision"],
        "module_error": outputs["viability_risk"].squeeze(-1),
        "module_aux": {
            "viability_state": outputs["viability_state"],
            "viability_risk": outputs["viability_risk"],
            "homeostatic_error": outputs["homeostatic_error"],
        },
    }


def _adapt_trm_as_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_state": outputs["action_state"],
        "module_precision": outputs["module_precision"],
        "module_error": outputs["action_uncertainty"].mean(dim=-1),
        "module_aux": {
            "policy_logits": outputs["policy_logits"],
            "policy_prob": outputs["policy_prob"],
            "action_uncertainty": outputs["action_uncertainty"],
        },
    }


def _adapt_trm_ag_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_state": outputs["gating_state"],
        "module_precision": outputs["module_precision"],
        "module_error": outputs["inhibition_mask"].mean(dim=-1),
        "module_aux": {
            "gating_logits": outputs["gating_logits"],
            "gated_policy_logits": outputs["gated_policy_logits"],
            "inhibition_mask": outputs["inhibition_mask"],
            "control_mode_logits": outputs["control_mode_logits"],
            "control_mode_prob": outputs["control_mode_prob"],
        },
    }


def _adapt_trm_bp_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_state": outputs["bp_state"],
        "module_precision": outputs["module_precision"],
        "module_error": outputs["mode_uncertainty"],
        "module_aux": {
            "pred_permeability_patch": outputs["pred_permeability_patch"],
            "pred_interface_gain": outputs["pred_interface_gain"],
            "pred_aperture_gain": outputs["pred_aperture_gain"],
            "mode_logits": outputs["mode_logits"],
            "mode_prob": outputs["mode_prob"],
        },
    }


def _adapt_trm_mc_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_state": outputs["context_state"],
        "module_precision": outputs["module_precision"],
        "module_error": outputs["context_uncertainty"],
        "module_aux": {
            "retrieved_context": outputs["retrieved_context"],
            "sequence_bias": outputs["sequence_bias"],
            "boundary_control_bias": outputs["boundary_control_bias"],
            "window_lengths": outputs["window_lengths"],
        },
    }


def patchify(images, patch_size: int):
    torch, _, _ = require_torch()
    batch, height, width, channels = images.shape
    assert height % patch_size == 0 and width % patch_size == 0
    p = patch_size
    h = height // p
    w = width // p
    x = images.reshape(batch, h, p, w, p, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(batch, h * w, p * p * channels)


def unpatchify(patches, image_size: int, patch_size: int, channels: int):
    torch, _, _ = require_torch()
    batch, tokens, dims = patches.shape
    h = image_size // patch_size
    w = image_size // patch_size
    assert tokens == h * w
    x = patches.reshape(batch, h, w, patch_size, patch_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(batch, image_size, image_size, channels)


def _build_recursive_token_block(config: TRMModelConfig):
    torch, nn, F = require_torch()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_q = nn.LayerNorm(config.dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=config.dim,
                num_heads=config.num_heads,
                batch_first=True,
            )
            hidden = config.dim * config.mlp_ratio
            self.fuse = nn.Sequential(
                nn.LayerNorm(config.dim * 3),
                nn.Linear(config.dim * 3, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.dim),
            )
            self.out = nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.dim),
            )

        def forward(self, x, y, z):
            attn_in = self.norm_q(y)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
            y = y + attn_out
            fused = self.fuse(torch.cat([x, y, z], dim=-1))
            z = z + fused
            y = y + self.out(z)
            return y, z

    return _Block()


def build_trm_a(config: TRMModelConfig):
    torch, nn, F = require_torch()

    class TRMAPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            out_channels = config.out_channels if config.out_channels is not None else config.in_channels
            token_dim_in = config.patch_size * config.patch_size * config.in_channels
            token_dim_out = config.patch_size * config.patch_size * out_channels
            self.input_proj = nn.Linear(token_dim_in, config.dim)
            self.position = nn.Parameter(
                torch.zeros(1, (config.image_size // config.patch_size) ** 2, config.dim)
            )
            self.core = _build_recursive_token_block(config)
            self.posterior_input_proj = nn.Linear(token_dim_in + token_dim_out, config.dim)
            self.posterior_mlp = nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, config.dim),
                nn.GELU(),
                nn.Linear(config.dim, config.dim),
            )
            self.prior_head = nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, config.z_dim * 2),
            )
            self.posterior_head = nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, config.z_dim * 2),
            )
            self.latent_proj = nn.Linear(config.z_dim, config.dim)
            self.mean_head = nn.Linear(config.dim, token_dim_out)
            self.logvar_head = nn.Linear(config.dim, token_dim_out)
            self.halt_head = nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, 1),
            )
            self.out_channels = out_channels

        def _split_stats(self, tensor):
            mu, logvar = tensor.chunk(2, dim=-1)
            return mu, torch.clamp(logvar, min=config.logvar_min, max=config.logvar_max)

        def _posterior_stats(self, states, targets):
            if targets is None:
                return None, None
            stacked = torch.cat([states, targets], dim=-1)
            tokens = patchify(stacked, config.patch_size)
            hidden = self.posterior_input_proj(tokens)
            hidden = hidden + self.position
            hidden = hidden + self.posterior_mlp(hidden)
            pooled = hidden.mean(dim=1)
            return self._split_stats(self.posterior_head(pooled))

        def _sample_latent(self, mu, logvar, sample_latent: bool):
            if not sample_latent:
                return mu
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)

        def forward(
            self,
            states,
            targets=None,
            use_posterior: bool = False,
            sample_latent: bool = False,
        ):
            x = patchify(states, config.patch_size)
            x = self.input_proj(x) + self.position
            y = x.clone()
            z = torch.zeros_like(x)
            token_steps = []
            halt_logits = []
            for _ in range(config.recursions):
                y, z = self.core(x, y, z)
                token_steps.append(y)
                halt_logits.append(self.halt_head(y.mean(dim=1)).squeeze(-1))
            prior_mu, prior_logvar = self._split_stats(self.prior_head(token_steps[-1].mean(dim=1)))
            post_mu, post_logvar = self._posterior_stats(states, targets)
            latent_mu = post_mu if use_posterior and post_mu is not None else prior_mu
            latent_logvar = post_logvar if use_posterior and post_logvar is not None else prior_logvar
            latent = self._sample_latent(latent_mu, latent_logvar, sample_latent=sample_latent)
            latent_tokens = self.latent_proj(latent).unsqueeze(1)
            pred_steps = []
            pred_logvar_steps = []
            for step_tokens in token_steps:
                decode_tokens = step_tokens + latent_tokens
                mean_patches = self.mean_head(decode_tokens)
                logvar_patches = torch.clamp(
                    self.logvar_head(decode_tokens),
                    min=config.logvar_min,
                    max=config.logvar_max,
                )
                pred_steps.append(
                    unpatchify(
                        mean_patches,
                        image_size=config.image_size,
                        patch_size=config.patch_size,
                        channels=self.out_channels,
                    )
                )
                pred_logvar_steps.append(
                    unpatchify(
                        logvar_patches,
                        image_size=config.image_size,
                        patch_size=config.patch_size,
                        channels=self.out_channels,
                    )
                )
            halt_logits_tensor = torch.stack(halt_logits, dim=1)
            halt_prob = torch.sigmoid(halt_logits_tensor)
            pred_state = pred_steps[-1]
            pred_logvar = pred_logvar_steps[-1]
            pred_patches = patchify(pred_state, config.patch_size)
            precision_map = torch.exp(-pred_logvar)
            module_precision = precision_map.mean(dim=(1, 2, 3))
            return {
                "pred_state_t1": pred_state,
                "pred_mean_t1": pred_state,
                "pred_logvar_t1": pred_logvar,
                "pred_var_t1": torch.exp(pred_logvar),
                "precision_map_t1": precision_map,
                "module_precision": module_precision,
                "pred_patches_t1": pred_patches,
                "halt_logits": halt_logits_tensor,
                "halt_prob": halt_prob,
                "pred_steps": pred_steps,
                "pred_logvar_steps": pred_logvar_steps,
                "prior_mu": prior_mu,
                "prior_logvar": prior_logvar,
                "post_mu": post_mu,
                "post_logvar": post_logvar,
                "latent": latent,
            }

    return TRMAPredictor()


def build_trm_b(config: TRMModelConfig):
    torch, nn, F = require_torch()

    class TRMBoundaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            in_channels = int(config.boundary_in_channels_total)
            patch_dim = config.patch_size * config.patch_size * in_channels
            self.input_proj = nn.Linear(patch_dim, config.dim)
            self.position = nn.Parameter(
                torch.zeros(1, (config.image_size // config.patch_size) ** 2, config.dim)
            )
            self.core = _build_recursive_token_block(config)
            out_patch_dim = config.patch_size * config.patch_size * 2
            self.out_proj = nn.Linear(config.dim, out_patch_dim)
            self.halt_head = nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, 1),
            )

        def forward(self, states, delta_states, error_maps):
            model_input = torch.cat([states, delta_states, error_maps], dim=-1)
            x = patchify(model_input, config.patch_size)
            x = self.input_proj(x) + self.position
            y = x.clone()
            z = torch.zeros_like(x)
            halt_logits = []
            output_steps = []
            for _ in range(config.recursions):
                y, z = self.core(x, y, z)
                output_steps.append(self.out_proj(y))
                halt_logits.append(self.halt_head(y.mean(dim=1)).squeeze(-1))
            patches = output_steps[-1]
            decoded = unpatchify(
                patches,
                image_size=config.image_size,
                patch_size=config.patch_size,
                channels=2,
            )
            boundary_map = torch.sigmoid(decoded[..., :1])
            permeability_map = torch.sigmoid(decoded[..., 1:2])
            return {
                "boundary_map": boundary_map,
                "permeability_map": permeability_map,
                "boundary_state": y,
                "halt_logits": torch.stack(halt_logits, dim=1),
                "halt_prob": torch.sigmoid(torch.stack(halt_logits, dim=1)),
            }

    return TRMBoundaryModel()


def build_trm_vm(config: TRMModelConfig):
    torch, nn, _ = require_torch()

    class TRMViabilityMonitor(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = 11  # viability(2) + env_contacts(4) + species_contacts(4) + action_cost(1)
            hidden = max(config.dim // 2, 16)
            self.encoder = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.dim),
                nn.GELU(),
            )
            self.viability_head = nn.Linear(config.dim, 2)
            self.risk_head = nn.Linear(config.dim, 1)
            self.precision_head = nn.Linear(config.dim, 1)

        def forward(self, viability_state, contact_state, action_cost):
            if contact_state.shape[-1] == 4:
                zeros = torch.zeros(
                    (*contact_state.shape[:-1], 4),
                    dtype=contact_state.dtype,
                    device=contact_state.device,
                )
                contact_state = torch.cat([contact_state, zeros], dim=-1)
            x = torch.cat([viability_state, contact_state, action_cost], dim=-1)
            latent = self.encoder(x)
            viability_delta = torch.tanh(self.viability_head(latent)) * 0.1
            next_viability = torch.clamp(viability_state + viability_delta, 0.0, 1.0)
            homeostatic_error = torch.abs(next_viability - torch.tensor([0.55, 0.65], device=x.device))
            viability_risk = torch.sigmoid(self.risk_head(latent))
            module_precision = torch.sigmoid(self.precision_head(latent)).squeeze(-1)
            return {
                "viability_state": next_viability,
                "viability_latent": latent,
                "viability_risk": viability_risk,
                "homeostatic_error": homeostatic_error,
                "module_precision": module_precision,
            }

    return TRMViabilityMonitor()


def build_trm_as(config: TRMModelConfig):
    torch, nn, _ = require_torch()

    class TRMActionScoring(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = 19  # viability(2) + scores(5) + uncertainty(4) + env_contacts(4) + species_contacts(4)
            hidden = max(config.dim // 2, 16)
            self.encoder = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.dim),
                nn.GELU(),
            )
            self.policy_head = nn.Linear(config.dim, 5)
            self.uncertainty_head = nn.Linear(config.dim, 5)
            self.precision_head = nn.Linear(config.dim, 1)
            # Preserve analytic action ranking by default and let the residual head
            # learn only the correction needed under uncertainty and viability pressure.
            self.base_logit_scale = nn.Parameter(torch.full((5,), 4.0, dtype=torch.float32))
            self.residual_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        def forward(
            self,
            viability_state,
            action_scores,
            uncertainty_state,
            env_contact_state=None,
            species_contact_state=None,
        ):
            if env_contact_state is None:
                env_contact_state = torch.zeros(
                    (*viability_state.shape[:-1], 4),
                    dtype=viability_state.dtype,
                    device=viability_state.device,
                )
            if species_contact_state is None:
                species_contact_state = torch.zeros(
                    (*viability_state.shape[:-1], 4),
                    dtype=viability_state.dtype,
                    device=viability_state.device,
                )
            x = torch.cat(
                [
                    viability_state,
                    action_scores,
                    uncertainty_state,
                    env_contact_state,
                    species_contact_state,
                ],
                dim=-1,
            )
            latent = self.encoder(x)
            base_scale = torch.nn.functional.softplus(self.base_logit_scale)
            base_logits = -(action_scores * base_scale)
            base_logits = base_logits - base_logits.mean(dim=-1, keepdim=True)
            residual_logits = self.policy_head(latent)
            residual_scale = torch.nn.functional.softplus(self.residual_scale)
            policy_logits = base_logits + residual_scale * residual_logits
            policy_prob = torch.softmax(policy_logits, dim=-1)
            action_uncertainty = torch.sigmoid(self.uncertainty_head(latent))
            module_precision = torch.sigmoid(self.precision_head(latent)).squeeze(-1)
            return {
                "action_state": latent,
                "base_logits": base_logits,
                "residual_logits": residual_logits,
                "policy_logits": policy_logits,
                "policy_prob": policy_prob,
                "action_uncertainty": action_uncertainty,
                "module_precision": module_precision,
            }

    return TRMActionScoring()


def build_trm_ag(config: TRMModelConfig):
    torch, nn, _ = require_torch()

    class TRMActionGating(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = int(config.in_channels)
            hidden = max(config.dim // 2, 16)
            self.encoder = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.dim),
                nn.GELU(),
            )
            self.inhibition_head = nn.Linear(config.dim, 5)
            self.control_mode_head = nn.Linear(config.dim, 3)
            self.precision_head = nn.Linear(config.dim, 1)

        def forward(self, input_view):
            latent = self.encoder(input_view)
            as_policy_logits = input_view[..., :5]
            inhibition_logits = self.inhibition_head(latent)
            inhibition_mask = torch.sigmoid(inhibition_logits)
            control_mode_logits = self.control_mode_head(latent)
            control_mode_prob = torch.softmax(control_mode_logits, dim=-1)
            control_weights = control_mode_prob @ torch.tensor(
                [
                    [0.10, -0.05, -0.10, -0.05, 0.10],   # exploratory
                    [0.00, 0.00, 0.00, 0.00, 0.00],      # maintenance
                    [-0.15, 0.15, -0.20, 0.20, 0.10],    # defensive
                ],
                dtype=input_view.dtype,
                device=input_view.device,
            )
            gated_policy_logits = as_policy_logits + control_weights - 2.25 * inhibition_mask
            module_precision = torch.sigmoid(self.precision_head(latent)).squeeze(-1)
            return {
                "gating_state": latent,
                "gating_logits": torch.zeros_like(as_policy_logits),
                "gated_policy_logits": gated_policy_logits,
                "inhibition_logits": inhibition_logits,
                "inhibition_mask": inhibition_mask,
                "control_mode_logits": control_mode_logits,
                "control_mode_prob": control_mode_prob,
                "module_precision": module_precision,
            }

    return TRMActionGating()


def build_trm_bp(config: TRMModelConfig):
    torch, nn, _ = require_torch()

    class TRMBoundaryPermeabilityControl(nn.Module):
        def __init__(self):
            super().__init__()
            in_channels = int(config.in_channels)
            hidden1 = max(config.dim // 8, 16)
            hidden2 = max(config.dim // 4, 32)
            hidden3 = max(config.dim // 2, 64)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.patch_head = nn.Sequential(
                nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden2, 1, kernel_size=1),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.latent = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(hidden2),
                nn.Linear(hidden2, hidden3),
                nn.GELU(),
                nn.Linear(hidden3, config.dim),
                nn.GELU(),
            )
            self.interface_head = nn.Linear(config.dim, 1)
            self.aperture_head = nn.Linear(config.dim, 1)
            self.mode_head = nn.Linear(config.dim, 3)
            self.precision_head = nn.Linear(config.dim, 1)

        def forward(self, input_view):
            x = input_view.permute(0, 3, 1, 2)
            feat = self.stem(x)
            pred_permeability_patch = torch.sigmoid(self.patch_head(feat)).permute(0, 2, 3, 1)
            pooled = self.pool(feat)
            state = self.latent(pooled)
            pred_interface_gain = torch.tanh(self.interface_head(state))
            pred_aperture_gain = torch.sigmoid(self.aperture_head(state))
            mode_logits = self.mode_head(state)
            mode_prob = torch.softmax(mode_logits, dim=-1)
            mode_uncertainty = 1.0 - torch.max(mode_prob, dim=-1).values
            module_precision = torch.sigmoid(self.precision_head(state)).squeeze(-1)
            return {
                "bp_state": state,
                "pred_permeability_patch": pred_permeability_patch,
                "pred_interface_gain": pred_interface_gain,
                "pred_aperture_gain": pred_aperture_gain,
                "mode_logits": mode_logits,
                "mode_prob": mode_prob,
                "mode_uncertainty": mode_uncertainty,
                "module_precision": module_precision,
            }

    return TRMBoundaryPermeabilityControl()


def build_trm_mc(config: TRMModelConfig):
    torch, nn, _ = require_torch()

    class TRMMemoryContext(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = int(config.in_channels)
            hidden = max(config.dim // 2, 32)
            self.input_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
            )
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.context_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, config.dim),
                nn.GELU(),
            )
            self.retrieval_head = nn.Linear(config.dim, input_dim)
            self.action_bias_head = nn.Linear(config.dim, 5)
            self.boundary_bias_head = nn.Linear(config.dim, 3)
            self.precision_head = nn.Linear(config.dim, 1)

        def forward(self, input_view, window_mask=None):
            if window_mask is None:
                window_mask = torch.ones(
                    input_view.shape[:2],
                    dtype=input_view.dtype,
                    device=input_view.device,
                )
            x = self.input_proj(input_view)
            seq, _ = self.gru(x)
            lengths = torch.clamp(window_mask.sum(dim=1).long(), min=1)
            gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
            last_state = torch.gather(seq, 1, gather_index).squeeze(1)
            context_state = self.context_head(last_state)
            retrieved_context = self.retrieval_head(context_state)
            sequence_bias = self.action_bias_head(context_state)
            boundary_control_bias = self.boundary_bias_head(context_state)
            module_precision = torch.sigmoid(self.precision_head(context_state)).squeeze(-1)
            context_uncertainty = 1.0 - module_precision
            return {
                "context_state": context_state,
                "retrieved_context": retrieved_context,
                "sequence_bias": sequence_bias,
                "boundary_control_bias": boundary_control_bias,
                "module_precision": module_precision,
                "context_uncertainty": context_uncertainty,
                "window_lengths": lengths.float(),
            }

    return TRMMemoryContext()
