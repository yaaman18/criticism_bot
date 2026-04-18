# TRM-Mc Harness And Test Plan

## 1. Purpose

This document defines how `TRM-Mc` should be integrated into the existing
dataset/training/production harness before implementation starts.

The goal is to ensure that `TRM-Mc` can be implemented in a way that is:

- consistent with [TRM_MC_CONTEXT_MEMORY_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_MC_CONTEXT_MEMORY_SPEC.md)
- compatible with the existing role-aware canonical-log workflow
- testable at each stage
- safe to introduce without destabilizing current `Wp/Bd/Bp/Vm/As` paths

This is a harness-and-testing design document, not the model implementation.

## 2. Scope

`TRM-Mc` is introduced first as:

- a **history-summary TRM**
- trained from canonical agentic logs
- used first as an **assistive context-bias module**
- connected to `TRM-As` before any broader memory/workspace role

The first implementation does **not** attempt:

- long-term episodic storage
- exact retrieval
- symbolic memory
- multi-episode archive lookup

## 3. Existing Harness Baseline

The current repo already supports:

- canonical dataset generation
- role-specific manifest generation
- role-aware training plans
- local training handoff
- external GPU handoff
- finalize / lineage tracking

This baseline exists for:

- `trm_wp`
- `trm_bd`
- `trm_bp`
- `trm_vm`
- `trm_as`

`TRM-Mc` should therefore follow the same pattern rather than inventing a
special-purpose side path.

## 4. Harness Design Principles For TRM-Mc

### MUST

- `TRM-Mc` MUST be derived from the canonical agentic log.
- `TRM-Mc` MUST use a role-specific view manifest: `views/trm_mc.jsonl`.
- `TRM-Mc` MUST be trainable independently from `TRM-As`.
- `TRM-Mc` MUST support local training, GPU handoff, and registry tracking.
- `TRM-Mc` MUST first be evaluated as an assistive module, not a primary
  controller.

### SHOULD

- `TRM-Mc` SHOULD be introduced without changing passive dataset flows.
- `TRM-Mc` SHOULD piggyback on the existing agentic bootstrap pipeline.
- `TRM-Mc` SHOULD be treated as a low-dimensional temporal-summary role.
- `TRM-Mc` SHOULD initially bias `TRM-As` and optionally `TRM-Bp`.

### MUST NOT

- `TRM-Mc` MUST NOT require a separate raw collection system.
- `TRM-Mc` MUST NOT bypass canonical logging.
- `TRM-Mc` MUST NOT depend on future GNW modules in phase 1.

## 5. Data Flow Design

The intended `TRM-Mc` path is:

```text
canonical agentic log
  -> role-specific temporal summary extraction
  -> views/trm_mc.jsonl
  -> train_trm_mc.py
  -> trm_mc.pt
  -> dataset_harness / production_runner registry integration
  -> runtime assistive connection to TRM-As
```

This keeps memory learning coupled to the same canonical data regime already
used by `Vm/As/Bp`.

## 6. Required Dataset Additions

The canonical log already contains most of what `TRM-Mc` needs. The new work is
to derive a temporal summary window.

### 6.1 New Role View

Add:

- `views/trm_mc.jsonl`

Each row should describe:

- `episode_npz`
- `input_view_key`
- `target_context_key`
- `target_action_bias_key`
- optional `target_boundary_bias_key`
- `window_size`
- `input_dim`
- `role`
- `quality`
- `episode_family`
- `species_context`

### 6.2 New Cached Arrays In Agentic NPZ

The agentic cache SHOULD include arrays such as:

- `mc_input_view`
- `mc_target_context_state`
- `mc_target_action_bias`
- optional `mc_target_boundary_control_bias`
- `mc_window_mask`

The exact names may change, but the pattern should mirror the existing
`bp_*/vm_*/as_*` role-specific storage.

## 7. Initial TRM-Mc Input View

The initial input view should be a short temporal window over summaries, not a
stack of raw frames.

Recommended first window contents per time step:

- `G_t`
- `B_t`
- `homeostatic_error_t`
- action one-hot
- action cost
- env contact summary:
  - energy
  - thermal
  - toxicity
  - niche
  - flow magnitude or `flow_y/flow_x`
- species contact summary
- interface summary:
  - aperture gain
  - aperture width
  - boundary/permeability summary
- uncertainty / error summary:
  - world uncertainty
  - boundary uncertainty
  - VFE summary or mean prediction-error magnitude

Recommended first shape:

- `window_size = 8` or `12`
- `feature_dim = 24..48`

This should remain compact enough for a `~7M` model with a recurrent or
temporal-conv encoder.

## 8. Initial Targets

The first implementation should avoid open-ended unsupervised memory training.

Recommended bootstrap targets:

- `mc_target_context_state`
  - analytic compressed context summary built from recent traces
- `mc_target_action_bias`
  - small residual bias over action logits for the next step
- optional `mc_target_boundary_control_bias`
  - coarse bias for `seal / intake / reconfigure`

This keeps phase 1 grounded in already available runtime signals.

## 9. Required Harness Changes

## 9.1 `prepare_trm_va_data.py`

Must add:

- `mc_*` cached arrays
- `trm_mc` role view manifest output
- `views/summary.json` update including `trm_mc`

## 9.2 `trm_input_views.py`

Must add:

- `build_trm_mc_input_view(...)`
- helper for stacking recent summary windows
- padding / masking for early timesteps

## 9.3 `train_trm_mc.py`

Must add:

- dataset loader for `views/trm_mc.jsonl`
- recurrent or temporal encoder
- outputs:
  - `context_state`
  - `sequence_bias`
  - optional `boundary_control_bias`
- metrics file and checkpoint outputs consistent with other trainers

## 9.4 `dataset_harness.py`

For agentic datasets:

- if `role_view_manifests["trm_mc"]` exists, add `train_trm_mc` to the training
  plan
- add model-eval criteria for `TRM-Mc`
- add registry/finalize handling for `train_trm_mc`
- include it in GPU handoff metadata

## 9.5 `production_runner.py`

No new logic should be needed if `dataset_harness` remains role-aware, but
`production_runner` must be validated against the expanded plan.

## 10. Initial Acceptance Metrics

`TRM-Mc` should not be accepted by reconstruction loss alone.

The first acceptance set should include:

- `val_context_state_loss`
- `val_action_bias_loss`
- `val_action_bias_alignment`
- `val_nonzero_context_fraction`
- `val_context_variance`

And runtime-facing acceptance:

- assistive `TRM-Mc + TRM-As` performs at least as well as `TRM-As` alone in
  context-sensitive families
- delayed-cost families improve:
  - `uncertain_corridor`
  - `vent_edge`
  - `fragile_boundary`
- repeated stress exploitation decreases

## 11. Runtime Integration Plan

Phase 1 runtime integration should be conservative.

### Phase 1A

- train `TRM-Mc`
- evaluate offline only

### Phase 1B

- connect `TRM-Mc` as assistive bias to `TRM-As`
- no direct action selection
- no direct boundary control

### Phase 1C

- optional assistive bias to `TRM-Bp`

Only after these steps should stronger integration be considered.

## 12. Test Strategy

Testing should be layered.

### 12.1 Pure View Tests

File:

- `tests/test_trm_input_views.py`

Add tests for:

- temporal window stacking shape
- zero-padding / mask behavior for early timesteps
- feature ordering stability
- deterministic extraction from fixed history rows

### 12.2 Cache Generation Tests

File:

- `tests/test_prepare_trm_va_data.py`

Add tests for:

- `mc_input_view` exists in saved episode cache
- `trm_mc.jsonl` is emitted into `views/`
- manifest rows carry window metadata
- summary contains `trm_mc` in `role_view_manifests`

### 12.3 Training Smoke Tests

New file:

- `tests/test_train_trm_mc.py`

Add tests for:

- small synthetic manifest can be trained for 1 epoch
- checkpoint and metrics are written
- resume works
- metrics contain the expected `TRM-Mc` keys

### 12.4 Harness Planning Tests

File:

- `tests/test_dataset_harness.py`

Add tests for:

- agentic plan includes `train_trm_mc` when `role_view_manifests["trm_mc"]`
  exists
- GPU handoff includes `train_trm_mc`
- model-eval report recognizes `TRM-Mc` metrics

### 12.5 Runtime Integration Tests

File:

- `tests/test_erie_runtime.py`

These should come later, after phase 1B.

Add tests for:

- `TRM-Mc` assistive bias changes `TRM-As` input path
- zero-context fallback remains stable
- context bias is logged in history

## 13. Failure Modes To Test Explicitly

The following failures should have explicit tests or assertions.

- window misalignment across timesteps
- leaking future information into the context window
- treating current action target as part of past context
- empty / degenerate context vectors
- silent fallback when `trm_mc` view manifest exists but is malformed
- collapse to constant action bias

## 14. Rollout Order

The implementation order should be:

1. add `TRM-Mc` role view generation
2. add `train_trm_mc.py`
3. add harness support
4. pass all dataset/training/harness tests
5. only then add runtime assistive integration

This is the safest order because it preserves current runtime behavior until the
memory role is independently validated.

## 15. Immediate Next Tasks

The next concrete tasks should therefore be:

1. implement `build_trm_mc_input_view(...)`
2. extend `prepare_trm_va_data.py` with `mc_*` arrays and `trm_mc` manifest
3. implement `train_trm_mc.py`
4. update `dataset_harness.py` planning and model-eval logic
5. add the tests listed above

Only after that should runtime coupling be attempted.
