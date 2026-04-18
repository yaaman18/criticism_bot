# TRM-Ag Specification

## 1. Purpose

This document defines `TRM-Ag`, the action-gating module in ERIE.

`TRM-Ag` is not the main action scorer. That role belongs to `TRM-As`.
`TRM-Ag` exists to decide whether already-scored actions should be:

- released
- suppressed
- shifted toward a safer control mode

In neuroscience terms, `TRM-Ag` is inspired by basal-ganglia-style gating and
suppression, not by a generic policy network.

For the broader library context, see
[TRM_LIBRARY_DESIGN.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_LIBRARY_DESIGN.md).
For concrete view slicing rules, see
[TRM_INPUT_VIEW_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_INPUT_VIEW_SPEC.md).

## 2. Role Definition

### Core role

`TRM-Ag` gates already-scored candidate actions.

It answers:

- which action should be inhibited right now
- whether the system should shift into a defensive / maintenance / exploratory
  control mode
- whether the currently best-scoring action is safe enough to release

### What it should do

- suppress locally attractive but globally dangerous actions
- reduce action collapse
- support fast defensive switching under high stress
- preserve useful `TRM-As` ranking where no strong inhibition is needed

### What it should not do

- replace `TRM-As` as the main scorer
- model raw world dynamics
- act like a full policy network over pixels

## 3. Brain Motif

Primary motif:

- basal ganglia

Secondary motifs:

- inhibitory control
- action release / disinhibition
- mode switching under urgency

`TRM-Ag` is therefore closer to a learned brake and release mechanism than to a
planner.

## 4. Required Inputs

`TRM-Ag` should consume low-dimensional summaries, not world tensors.

### Minimum input view

- `as_policy_logits` or `as_residual_logits`
- `viability_state`
- `homeostatic_error_vector`
- `viability_risk`
- `uncertainty_state`
- `stress_summary`
- optional `mc_sequence_bias`
- optional `bp_control_summary`

### Recommended initial input composition

- action logits: `5`
- viability state: `2`
- homeostatic error vector: `2`
- viability risk: `1`
- uncertainty summary: `4`
- env contact summary: `4`
- species contact summary: `4`
- optional context bias: `5`

Total initial range:

- `18` to `23` dims

This keeps the module plausibly small under the `~7M` parameter budget.

## 5. Outputs

`TRM-Ag` should produce gating-related outputs only.

### Primary outputs

- `module_state`: action-gating latent
- `module_precision`: confidence in current gating recommendation
- `module_aux.gated_policy_logits`
- `module_aux.inhibition_mask`
- `module_aux.control_mode_logits`

### Control mode vocabulary

Initial 3-mode recommendation:

- `exploratory`
- `maintenance`
- `defensive`

## 6. Runtime Semantics

`TRM-Ag` runs after `TRM-As`.

### Stage ordering

1. `TRM-As` proposes action logits
2. `TRM-Mc` may add contextual bias
3. `TRM-Ag` gates / suppresses / mode-shifts
4. final action policy is normalized

### Initial assistive formulation

The first implementation should be assistive, not primary-only.

Suggested formula:

```text
gated_logits
= as_logits
+ alpha_ctx * mc_sequence_bias
+ alpha_ag * ag_gating_logits
- beta_ag * inhibition_mask
```

where:

- `ag_gating_logits` are release-friendly adjustments
- `inhibition_mask` penalizes unsafe actions

### Initial mode interpretation

- `exploratory`
  - soften inhibition on `approach` / `reconfigure`
- `maintenance`
  - keep action distribution near `TRM-As`
- `defensive`
  - suppress `intake` / risky `approach`
  - favor `seal` / `withdraw` / `reconfigure`

## 7. Dataset and Targets

`TRM-Ag` should be trained from canonical agentic logs.

### Input source

- `TRM-As` score summaries
- `TRM-Vm` viability summaries
- env/species contact summaries
- uncertainty summaries
- optional `TRM-Mc` context bias

### Initial bootstrap targets

The first version may use analytic gating heuristics as weak supervision.

Recommended targets:

- `ag_target_inhibition_mask`
- `ag_target_control_mode`
- `ag_target_gated_policy`

### Heuristic bootstrap intent

Bootstrap should encode:

- inhibit `intake` when stress exceeds energy benefit
- inhibit `approach` under acute boundary danger
- inhibit over-sealing when energy deficit dominates
- switch to defensive mode when `B` risk is high
- switch to exploratory mode when uncertainty is high but viability is safe

## 8. Acceptance Criteria

`TRM-Ag` is successful only if it improves runtime behavior, not merely train
loss.

### Training-level

- `val_inhibition_accuracy` improves above trivial baseline
- `val_control_mode_accuracy` exceeds trivial majority-class baseline
- `val_gated_policy_kl` remains finite and stable

### Runtime-level

Against `TRM-As` alone, `TRM-Ag` should:

- reduce `stress_exploit_rate`
- reduce `dominant_action_fraction`
- reduce `homeostatic_error` in stress-heavy families
- not collapse action entropy to near-zero

### Family-specific expectations

- `fragile_boundary`
  - should suppress unsafe `intake`
- `vent_edge`
  - should suppress overexposure while not freezing
- `uncertain_corridor`
  - should allow exploration when safe, not only clamp down

## 9. Phased Implementation

### Phase 1

- define dataset view
- create `train_trm_ag.py`
- train with heuristic bootstrap labels
- runtime assistive integration only

### Phase 2

- compare `As` vs `As + Ag`
- add family-aware gating targets
- add `TRM-Mc` context as input

### Phase 3

- consider `module_primary`
- integrate with future salience / conflict TRMs

## 10. Non-Claims

`TRM-Ag` is not a literal model of the basal ganglia.

It is an engineering approximation of:

- action suppression
- release gating
- control-mode switching

implemented within ERIE's small-module regime.
