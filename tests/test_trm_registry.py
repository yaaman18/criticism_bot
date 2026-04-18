from __future__ import annotations

from pathlib import Path

import pytest

from trm_pipeline.models import (
    TRMModelConfig,
    adapt_trm_outputs,
    build_trm,
    get_trm_registry,
    require_torch,
)
from trm_pipeline.erie_runtime import RuntimeModels


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
    trm_a = build_trm("trm_a", config)
    trm_b = build_trm("trm_b", config)
    trm_a_path = tmp_path / "trm_a_registry.pt"
    trm_b_path = tmp_path / "trm_b_registry.pt"
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state": trm_a.state_dict(),
            "module_name": "trm_a",
            "module_role": "world_model",
        },
        trm_a_path,
    )
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state": trm_b.state_dict(),
            "module_name": "trm_b",
            "module_role": "boundary_model",
        },
        trm_b_path,
    )
    return trm_a_path, trm_b_path


def _checkpoint_path_for_module(tmp_path: Path, module_name: str) -> Path:
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
    model = build_trm(module_name, config)
    checkpoint_path = tmp_path / f"{module_name}.pt"
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state": model.state_dict(),
            "module_name": module_name,
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_builtin_trm_registry_contains_a_and_b() -> None:
    registry = get_trm_registry()
    assert "trm_a" in registry
    assert "trm_b" in registry
    assert "trm_vm" in registry
    assert "trm_as" in registry
    assert "trm_ag" in registry
    assert "trm_bp" in registry
    assert "trm_mc" in registry
    assert registry["trm_a"]["role"] == "world_model"
    assert registry["trm_b"]["role"] == "boundary_model"
    assert registry["trm_vm"]["role"] == "viability_monitor"
    assert registry["trm_as"]["role"] == "action_scoring"
    assert registry["trm_ag"]["role"] == "action_gating"
    assert registry["trm_bp"]["role"] == "boundary_permeability_control"
    assert registry["trm_mc"]["role"] == "memory_context"


def test_build_trm_uses_registry_name() -> None:
    config = TRMModelConfig(image_size=32, dim=32, recursions=2, mlp_ratio=2, z_dim=8)
    model_a = build_trm("trm_a", config)
    model_b = build_trm("trm_b", config)
    model_vm = build_trm("trm_vm", config)
    model_as = build_trm("trm_as", config)
    model_ag = build_trm("trm_ag", TRMModelConfig(image_size=8, dim=32, recursions=2, mlp_ratio=2, z_dim=8, in_channels=22))
    model_bp = build_trm("trm_bp", TRMModelConfig(image_size=16, dim=32, recursions=2, mlp_ratio=2, z_dim=8, in_channels=21))
    model_mc = build_trm("trm_mc", TRMModelConfig(image_size=8, dim=32, recursions=2, mlp_ratio=2, z_dim=8, in_channels=44))
    assert model_a is not None
    assert model_b is not None
    assert model_vm is not None
    assert model_as is not None
    assert model_ag is not None
    assert model_bp is not None
    assert model_mc is not None


def test_adapt_trm_a_outputs_to_common_contract() -> None:
    torch, _, _ = require_torch()
    config = TRMModelConfig(image_size=32, dim=32, recursions=2, mlp_ratio=2, z_dim=8)
    model = build_trm("trm_a", config)
    x = torch.zeros((1, 32, 32, 5), dtype=torch.float32)
    outputs = model(x, use_posterior=False, sample_latent=False)
    adapted = adapt_trm_outputs("trm_a", outputs)

    assert adapted["module_role"] == "world_model"
    assert adapted["module_state"].shape[0] == 1
    assert "pred_state_t1" in adapted["module_aux"]
    assert adapted["module_precision"].shape == (1,)


def test_adapt_trm_b_outputs_to_common_contract() -> None:
    torch, _, _ = require_torch()
    config = TRMModelConfig(image_size=32, dim=32, recursions=2, mlp_ratio=2, z_dim=8)
    model = build_trm("trm_b", config)
    x = torch.zeros((1, 32, 32, 5), dtype=torch.float32)
    delta = torch.zeros((1, 32, 32, 5), dtype=torch.float32)
    err = torch.zeros((1, 32, 32, 5), dtype=torch.float32)
    outputs = model(x, delta, err)
    adapted = adapt_trm_outputs("trm_b", outputs)

    assert adapted["module_role"] == "boundary_model"
    assert adapted["module_state"].shape[0] == 1
    assert "boundary_map" in adapted["module_aux"]
    assert adapted["module_precision"].shape == (1,)


def test_adapt_trm_vm_outputs_to_common_contract() -> None:
    torch, _, _ = require_torch()
    config = TRMModelConfig(image_size=32, dim=32, recursions=2, mlp_ratio=2, z_dim=8)
    model = build_trm("trm_vm", config)
    viability = torch.tensor([[0.7, 0.8]], dtype=torch.float32)
    contact = torch.tensor([[0.2, 0.1, 0.0, 0.3]], dtype=torch.float32)
    cost = torch.tensor([[0.05]], dtype=torch.float32)
    outputs = model(viability, contact, cost)
    adapted = adapt_trm_outputs("trm_vm", outputs)

    assert adapted["module_role"] == "viability_monitor"
    assert adapted["module_state"].shape == (1, 32)
    assert adapted["module_precision"].shape == (1,)
    assert "viability_state" in adapted["module_aux"]


def test_adapt_trm_as_outputs_to_common_contract() -> None:
    torch, _, _ = require_torch()
    config = TRMModelConfig(image_size=32, dim=32, recursions=2, mlp_ratio=2, z_dim=8)
    model = build_trm("trm_as", config)
    viability = torch.tensor([[0.7, 0.8]], dtype=torch.float32)
    scores = torch.zeros((1, 5), dtype=torch.float32)
    uncertainty = torch.tensor([[0.2, 0.1, 0.0, 0.3]], dtype=torch.float32)
    outputs = model(viability, scores, uncertainty)
    adapted = adapt_trm_outputs("trm_as", outputs)

    assert adapted["module_role"] == "action_scoring"
    assert adapted["module_state"].shape == (1, 32)
    assert adapted["module_precision"].shape == (1,)
    assert "policy_logits" in adapted["module_aux"]


def test_adapt_trm_bp_outputs_to_common_contract() -> None:
    torch, _, _ = require_torch()
    config = TRMModelConfig(image_size=16, dim=32, recursions=2, mlp_ratio=2, z_dim=8, in_channels=21)
    model = build_trm("trm_bp", config)
    x = torch.zeros((1, 16, 16, 21), dtype=torch.float32)
    outputs = model(x)
    adapted = adapt_trm_outputs("trm_bp", outputs)

    assert adapted["module_role"] == "boundary_permeability_control"
    assert adapted["module_state"].shape == (1, 32)
    assert adapted["module_precision"].shape == (1,)
    assert "pred_permeability_patch" in adapted["module_aux"]


def test_adapt_trm_ag_outputs_to_common_contract() -> None:
    torch, _, _ = require_torch()
    config = TRMModelConfig(image_size=8, dim=32, recursions=2, mlp_ratio=2, z_dim=8, in_channels=22)
    model = build_trm("trm_ag", config)
    x = torch.zeros((1, 22), dtype=torch.float32)
    outputs = model(x)
    adapted = adapt_trm_outputs("trm_ag", outputs)

    assert adapted["module_role"] == "action_gating"
    assert adapted["module_state"].shape == (1, 32)
    assert adapted["module_precision"].shape == (1,)
    assert "gated_policy_logits" in adapted["module_aux"]


def test_adapt_trm_mc_outputs_to_common_contract() -> None:
    torch, _, _ = require_torch()
    config = TRMModelConfig(image_size=8, dim=32, recursions=2, mlp_ratio=2, z_dim=8, in_channels=44)
    model = build_trm("trm_mc", config)
    x = torch.zeros((1, 8, 44), dtype=torch.float32)
    mask = torch.ones((1, 8), dtype=torch.float32)
    outputs = model(x, mask)
    adapted = adapt_trm_outputs("trm_mc", outputs)

    assert adapted["module_role"] == "memory_context"
    assert adapted["module_state"].shape == (1, 32)
    assert adapted["module_precision"].shape == (1,)
    assert "sequence_bias" in adapted["module_aux"]


def test_runtime_models_can_load_module_specs(tmp_path: Path) -> None:
    trm_a_path, trm_b_path = _checkpoint_paths(tmp_path)
    models = RuntimeModels(
        None,
        None,
        module_specs=[
            {"name": "trm_a", "checkpoint": str(trm_a_path)},
            {"name": "trm_b", "checkpoint": str(trm_b_path)},
        ],
    )

    assert models.enabled is True
    assert len(models.modules) == 2
    assert {module["name"] for module in models.modules} == {"trm_a", "trm_b"}
    assert models.trm_a is not None
    assert models.trm_b is not None


def test_runtime_models_choose_primary_module_per_role(tmp_path: Path) -> None:
    trm_a_path_1, trm_b_path = _checkpoint_paths(tmp_path)
    trm_a_path_2, _ = _checkpoint_paths(tmp_path)
    models = RuntimeModels(
        None,
        None,
        module_specs=[
            {"id": "world_a_1", "name": "trm_a", "checkpoint": str(trm_a_path_1)},
            {"id": "world_a_2", "name": "trm_a", "checkpoint": str(trm_a_path_2), "primary": True},
            {"id": "boundary_b_1", "name": "trm_b", "checkpoint": str(trm_b_path), "primary": True},
        ],
    )

    world_modules = [module for module in models.modules if module["role"] == "world_model"]
    boundary_modules = [module for module in models.modules if module["role"] == "boundary_model"]

    assert len(world_modules) == 2
    assert len(boundary_modules) == 1
    assert models.primary_module("world_model")["id"] == "world_a_2"
    assert models.primary_module("boundary_model")["id"] == "boundary_b_1"
    assert models.trm_a is models.primary_module("world_model")["model"]
    assert models.trm_b is models.primary_module("boundary_model")["model"]


def test_runtime_models_default_to_first_module_when_primary_is_absent(tmp_path: Path) -> None:
    trm_a_path_1, _ = _checkpoint_paths(tmp_path)
    trm_a_path_2, _ = _checkpoint_paths(tmp_path)
    models = RuntimeModels(
        None,
        None,
        module_specs=[
            {"id": "world_a_1", "name": "trm_a", "checkpoint": str(trm_a_path_1)},
            {"id": "world_a_2", "name": "trm_a", "checkpoint": str(trm_a_path_2)},
        ],
    )

    assert models.primary_module("world_model")["id"] == "world_a_1"
    assert models.trm_a is models.primary_module("world_model")["model"]


def test_runtime_models_reject_multiple_primary_modules_in_same_role(tmp_path: Path) -> None:
    trm_a_path_1, _ = _checkpoint_paths(tmp_path)
    trm_a_path_2, _ = _checkpoint_paths(tmp_path)

    with pytest.raises(SystemExit):
        RuntimeModels(
            None,
            None,
            module_specs=[
                {"id": "world_a_1", "name": "trm_a", "checkpoint": str(trm_a_path_1), "primary": True},
                {"id": "world_a_2", "name": "trm_a", "checkpoint": str(trm_a_path_2), "primary": True},
            ],
        )


def test_runtime_models_can_load_manifest_file(tmp_path: Path) -> None:
    trm_a_path, trm_b_path = _checkpoint_paths(tmp_path)
    manifest = tmp_path / "modules.json"
    manifest.write_text(
        (
            '[{"id":"world_a","name":"trm_a","checkpoint":"%s","primary":true},'
            '{"id":"boundary_b","name":"trm_b","checkpoint":"%s","primary":true}]'
        )
        % (trm_a_path, trm_b_path),
        encoding="utf-8",
    )

    models = RuntimeModels(None, None, module_manifest=manifest)

    assert models.enabled is True
    assert len(models.modules) == 2
    assert models.primary_module("world_model")["id"] == "world_a"


def test_runtime_models_can_list_secondary_modules(tmp_path: Path) -> None:
    trm_a_path_1, _ = _checkpoint_paths(tmp_path)
    trm_a_path_2, _ = _checkpoint_paths(tmp_path)
    models = RuntimeModels(
        None,
        None,
        module_specs=[
            {"id": "world_primary", "name": "trm_a", "checkpoint": str(trm_a_path_1), "primary": True},
            {"id": "world_secondary", "name": "trm_a", "checkpoint": str(trm_a_path_2)},
        ],
    )

    secondaries = models.secondary_modules("world_model")
    assert len(secondaries) == 1
    assert secondaries[0]["id"] == "world_secondary"


def test_runtime_models_reject_unknown_module_name(tmp_path: Path) -> None:
    trm_a_path, _ = _checkpoint_paths(tmp_path)
    with pytest.raises(SystemExit):
        RuntimeModels(
            None,
            None,
            module_specs=[{"name": "trm_unknown", "checkpoint": str(trm_a_path)}],
        )


def test_runtime_models_can_load_viability_and_action_modules(tmp_path: Path) -> None:
    vm_path = _checkpoint_path_for_module(tmp_path, "trm_vm")
    as_path = _checkpoint_path_for_module(tmp_path, "trm_as")
    models = RuntimeModels(
        None,
        None,
        module_specs=[
            {"id": "vm_primary", "name": "trm_vm", "checkpoint": str(vm_path), "primary": True},
            {"id": "as_primary", "name": "trm_as", "checkpoint": str(as_path), "primary": True},
        ],
    )

    assert models.primary_module("viability_monitor")["id"] == "vm_primary"
    assert models.primary_module("action_scoring")["id"] == "as_primary"
