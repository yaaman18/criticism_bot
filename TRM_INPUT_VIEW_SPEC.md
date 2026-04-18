# TRM Input View Specification

## 1. Purpose

This document defines how the canonical ERIE/Lenia world log should be sliced
into TRM-specific input views.

The main constraint is that each TRM is assumed to remain around `~7M`
parameters. Therefore, not every TRM should receive the full world state.

Instead, the canonical log is preserved richly, while each TRM consumes only a
role-appropriate view.

This document answers:

1. What is the canonical log?
2. What slice should each TRM read?
3. Which slices should be used first in implementation?

## 2. Canonical Log

The canonical log is the richest common record captured from runtime.

At minimum it includes:

- `external_state`
- `env_channels`
- `observation`
- `sensor_gate`
- `world_error`
- `boundary_error`
- `world_belief`
- `world_logvar`
- `boundary_belief`
- `boundary_logvar`
- `species_sources`
- `species_fields`
- action history
- viability history
- contact statistics
- policy diagnostics

The canonical log SHOULD be preserved even when a specific TRM does not use most
of it.

## 3. View Design Rules

### MUST

- A TRM MUST NOT receive the full canonical log by default.
- A TRM MUST receive only the smallest view compatible with its main role.
- A view MUST preserve the distinction between:
  - world state
  - interface state
  - internal state
  - module-to-module summaries

### SHOULD

- World-facing TRMs SHOULD receive field-like tensors.
- Boundary-facing TRMs SHOULD receive interface-local tensors.
- Viability and action TRMs SHOULD prefer low-dimensional summaries.
- Workspace-family TRMs SHOULD consume module-level outputs rather than raw
  pixels.

## 4. Canonical Channels

## 4.1 Observation-Compatible World View

`env_channels`

Current shape:

- base Lenia multistate: `5`
- environmental fields: `6`

Total:

- `11 channels`

The 6 environmental channels are:

- `energy_gradient`
- `thermal_stress`
- `toxicity`
- `niche_stability`
- `flow_y`
- `flow_x`

This view exists for compatibility and for observation-facing TRMs.

## 4.2 Full External World View

`external_state`

Current shape:

- base Lenia multistate: `5`
- `species_energy` multistate: `5`
- `species_toxic` multistate: `5`
- `species_niche` multistate: `5`
- environmental fields: `6`

Total:

- `26 channels`

This view is the richer world-state log and SHOULD be treated as canonical world
state.

## 4.3 Species-Specific Views

- `species_sources`: `3 channels`
  - `species_energy`
  - `species_toxic`
  - `species_niche`

- `species_fields`: `4 channels`
  - energy contribution
  - thermal contribution
  - toxicity contribution
  - niche contribution

## 5. TRM-Specific Input Views

## 5.1 TRM-Wp
World Prediction

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L72)

### Role

Predict near-future external/environmental state.

### Required input view

- `observation_t`
- `sensor_gate_t`
- `species_fields_t`
- `flow_t`

### Optional input view

- compressed `external_state_t`
- previous `world_belief_t`

### Recommended initial implementation

Use:

- `observation_t`: `11 channels`
- `sensor_gate_t`: `1 channel`
- `species_fields_t`: `4 channels`

with `flow` read from the last two channels of `observation_t` or separately as
`2 channels`.

### Rationale

`TRM-Wp` should learn from what the agent can observe, plus a compact summary of
the multispecies world dynamics.

It SHOULD NOT consume all `26 channels` by default.

## 5.2 TRM-Bd
Boundary Detection

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L115)

### Role

Estimate self/non-self boundary and candidate interface.

### Required input view

- `observation_t`
- `world_error_t`
- `sensor_gate_t`
- temporal delta of observation

### Optional input view

- `species_fields_t` as local context

### Recommended initial implementation

Use:

- `observation_t`: `11 channels`
- `world_error_t`: `11 channels`
- `sensor_gate_t`: `1 channel`
- `delta_observation_t`: `11 channels`

### Rationale

Boundary inference depends more on sensed change, error, and gating than on the
full hidden world state.

## 5.3 TRM-Bp
Boundary Permeability Control

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L135)

### Role

Control what crosses the interface.

### Required input view

- local boundary/permeability patch
- local observation patch
- local species field patch
- viability summary

### Recommended initial implementation

Use a boundary-local crop rather than the full frame.

Current bootstrap cache contract:

- local crop size: `16 x 16`
- patch channels:
  - boundary patch: `1`
  - permeability patch: `1`
  - observation patch: `11`
  - species patch: `4`
  - flow patch: `2`
  - tiled viability summary: `2`

Total:

- `21 channels`

Initial bootstrap targets:

- `bp_target_permeability_patch`
- `bp_target_interface_gain`

## 5.6 TRM-Ag
Action Gating

Reference:
- [TRM_AG_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_AG_SPEC.md)

### Role

Gate or suppress already-scored candidate actions.

### Required input view

- `TRM-As` action logits or residual logits
- viability summary
- homeostatic error summary
- viability risk
- uncertainty summary
- env/species contact summary
- optional `TRM-Mc` sequence bias

### Recommended initial implementation

Use a low-dimensional summary view rather than raw world tensors.

Initial target range:

- `18` to `23` dims

### Rationale

`TRM-Ag` is a release / suppression module, not a world model. It should
consume already-compressed policy and urgency summaries.
- `bp_target_aperture_gain`
- `bp_target_mode`
  - `0`: open-like
  - `1`: close-like
  - `2`: reconfigure-like

### Rationale

`TRM-Bp` should behave like interface control, not world modeling.

## 5.4 TRM-Vm
Viability Monitoring

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L172)

### Role

Estimate current survival-relevant internal state.

### Required input view

- current viability state
- environment contact summary
- species contact summary
- recent action cost

### Recommended initial implementation

Use:

- `G_t`, `B_t`: `2`
- env contact:
  - `energy`
  - `thermal`
  - `toxicity`
  - `niche`
- species contact:
  - `species_energy`
  - `species_thermal`
  - `species_toxicity`
  - `species_niche`
- action cost: `1`

Total:

- approximately `11 dimensions`

### Rationale

`TRM-Vm` should model internal condition, not reconstruct the image world.

## 5.5 TRM-Vr
Viability Risk Estimation

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L190)

### Role

Estimate near-future failure risk.

### Required input view

- current viability summary
- projected contact summary
- projected boundary status
- projected action outcome

### Recommended initial implementation

Use low-dimensional rollout summaries rather than raw fields.

### Rationale

`TRM-Vr` should predict margin-to-failure, not duplicate `TRM-Wp`.

## 5.6 TRM-As
Action Scoring

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L210)

### Role

Score candidate actions under risk, ambiguity, and epistemic terms.

### Required input view

- viability summary
- analytic action-score proxy
- uncertainty summary
- environment/species contact summary

### Recommended initial implementation

Use:

- viability state: `2`
- analytic action scores: `5`
- uncertainty summary: `4`
- contact summary:
  - env contact `4`
  - species contact `4`

### Rationale

`TRM-As` should operate on compact decision-state summaries and not directly on
the raw world tensor.

## 5.7 TRM-Ag
Action Gating

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L228)

### Role

Release or suppress candidate actions.

### Required input view

- action logits
- viability urgency
- salience
- conflict

### Recommended initial implementation

Use module-level summaries only.

### Rationale

`TRM-Ag` is arbitration, not world perception.

## 5.8 TRM-Mc
Memory Context

Dedicated design reference:
- [TRM_MC_CONTEXT_MEMORY_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_MC_CONTEXT_MEMORY_SPEC.md)

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L248)

### Role

Maintain and retrieve recent contextual trajectory summaries.

### Required input view

- recent viability summaries
- recent action sequence
- recent contact summaries
- recent species-contact summaries
- recent homeostatic error

### Recommended initial implementation

Use short windows of compact temporal summaries, not raw video tensors.

### Rationale

`TRM-Mc` should be history-aware and state-compact.

## 5.9 TRM-Sa
Salience Allocation

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L276)

### Role

Estimate what matters now.

### Required input view

- prediction-error summary
- uncertainty summary
- contact summary
- viability urgency

### Rationale

Salience should operate on relevance summaries, not on the full world tensor by
default.

## 5.10 TRM-Sc
Conflict Monitoring

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L292)

### Role

Detect incompatibility among internal tendencies.

### Required input view

- action scores
- viability tension
- boundary demand
- memory/context bias

### Rationale

Conflict monitoring should be module-level and low-dimensional.

## 5.11 TRM-Xc and TRM-Xi
Workspace Candidate / Ignition

Reference:
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L309)
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md#L324)

### Role

Route selected content toward future workspace-like availability.

### Required input view

- `module_state`
- `module_precision`
- `module_error`
- salience / conflict / urgency

### Rationale

Workspace-family TRMs SHOULD consume module-level summaries, not raw world
pixels.

## 6. Immediate Implementation Order

The following TRM input views should be stabilized first:

1. `TRM-Wp`
2. `TRM-Bd`
3. `TRM-Vm`
4. `TRM-As`

The following should come after:

5. `TRM-Bp`
6. `TRM-Ag`
7. `TRM-Mc`

Workspace-family TRMs are later-phase.

## 7. Practical Rule

The practical rule is:

- world-facing TRMs get world slices
- boundary-facing TRMs get interface slices
- viability/action TRMs get low-dimensional summaries
- workspace TRMs get module-level summaries

This rule SHOULD be followed unless a specific experiment justifies a broader
input view.
