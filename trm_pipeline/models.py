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
            token_dim = config.patch_size * config.patch_size * config.in_channels
            self.input_proj = nn.Linear(token_dim, config.dim)
            self.position = nn.Parameter(
                torch.zeros(1, (config.image_size // config.patch_size) ** 2, config.dim)
            )
            self.core = _build_recursive_token_block(config)
            self.posterior_input_proj = nn.Linear(token_dim * 2, config.dim)
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
            self.mean_head = nn.Linear(config.dim, token_dim)
            self.logvar_head = nn.Linear(config.dim, token_dim)
            self.halt_head = nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, 1),
            )

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
                        channels=config.in_channels,
                    )
                )
                pred_logvar_steps.append(
                    unpatchify(
                        logvar_patches,
                        image_size=config.image_size,
                        patch_size=config.patch_size,
                        channels=config.in_channels,
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
            in_channels = 15
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
            input_dim = 6  # viability(2) + contacts(3) + action_cost(1)
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
            input_dim = 10  # viability(2) + scores(5) + uncertainty(3)
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

        def forward(self, viability_state, action_scores, uncertainty_state):
            x = torch.cat([viability_state, action_scores, uncertainty_state], dim=-1)
            latent = self.encoder(x)
            policy_logits = self.policy_head(latent)
            policy_prob = torch.softmax(policy_logits, dim=-1)
            action_uncertainty = torch.sigmoid(self.uncertainty_head(latent))
            module_precision = torch.sigmoid(self.precision_head(latent)).squeeze(-1)
            return {
                "action_state": latent,
                "policy_logits": policy_logits,
                "policy_prob": policy_prob,
                "action_uncertainty": action_uncertainty,
                "module_precision": module_precision,
            }

    return TRMActionScoring()
