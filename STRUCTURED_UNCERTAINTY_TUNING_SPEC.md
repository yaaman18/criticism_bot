# Structured Uncertainty Tuning Specification

## 1. Purpose

This document defines how structured uncertainty recipes should be tuned while
running ERIE-on-Lenia experiments.

The goal is not to find a perfect recipe manually. The goal is to run a
repeatable tuning loop where recipes, runtime hooks, metrics, and dataset gates
are adjusted in small documented steps.

This specification extends:

- `STRUCTURED_UNCERTAINTY_SPEC.md`
- `LLM_STRUCTURED_UNCERTAINTY_INTEGRATION_SPEC.md`

## 2. Tuning Principle

Structured uncertainty tuning MUST follow this rule:

```text
Tune recipes and gates from executed trajectories, not from LLM plausibility.
```

The LLM may propose recipes, but only simulator-executed results decide whether
the recipe is useful.

A useful tuning change should improve at least one of:

- action branching
- history-based disambiguation
- prediction error reducibility
- delayed consequence visibility
- viability tradeoff richness
- family / recipe diversity

without collapsing survival, action diversity, or external-state variability.

## 3. Tuning Loop

Use this loop for every tuning iteration:

```text
1. Generate or select recipe candidates
2. Validate and clamp recipes
3. Run smoke episodes
4. Compute metrics
5. Classify failure modes
6. Apply bounded tuning action
7. Re-run
8. Promote accepted recipes to recipe manifest
```

No tuning change should be applied without saving:

- input recipe fingerprint
- tuning action
- reason
- before metrics
- after metrics, when available
- resulting recipe fingerprint

## 4. Tuning Stages

### 4.1 Stage A: Recipe Validity

Purpose:

Ensure LLM or hand-written recipes are syntactically valid and translatable.

Inputs:

- recipe JSON
- schema version
- validator report

Actions:

- reject invalid JSON
- reject unknown field names
- clamp numeric values
- map natural-language latent rules to fixed rule types
- reject unmapped rule types

Promotion condition:

```text
valid_recipe_count >= requested_min
and all promoted recipes have deterministic fingerprints
```

### 4.2 Stage B: Runtime Survival

Purpose:

Ensure recipes do not immediately destroy the episode.

Primary metrics:

- `non_dead_fraction`
- `mean_episode_length`
- `early_death_fraction`
- `homeostatic_error_mean`

Default initial thresholds:

```text
min_non_dead_fraction: 0.25
max_early_death_fraction: 0.65
min_mean_episode_length_fraction: 0.40
```

Tuning actions:

- reduce `toxicity.intensity`
- reduce `thermal_stress.intensity`
- reduce latent rule `strength`
- increase `delay_steps`
- increase `niche_stability.intensity`
- reduce observation distortion noise

### 4.3 Stage C: External Contact Variability

Purpose:

Ensure the agent actually encounters variable external conditions.

Primary metrics:

- `contact_energy_std`
- `contact_thermal_std`
- `contact_toxicity_std`
- `contact_niche_std`
- `external_state_contact_variability`

Default initial thresholds:

```text
min_external_contact_std: 0.005
min_distinct_contact_fields_active: 3
```

Tuning actions:

- increase `flow.strength`
- increase patch count for underactive fields
- increase `field_sigma_max` if contact is too spiky
- decrease `field_sigma_min` if contact is too uniform
- increase drift only if episode is otherwise stable

### 4.4 Stage D: Action Branching

Purpose:

Ensure the dataset is not solved by one repetitive action.

Primary metrics:

- `aggregate_policy_entropy`
- `dominant_action_fraction`
- `distinct_actions_per_episode`
- `action_rank_flip_count`
- `affordance_ambiguity_score`

Default initial thresholds:

```text
min_episode_policy_entropy: 0.45
max_dominant_action_fraction: 0.75
min_distinct_actions: 3
min_affordance_ambiguity_score: 0.05
```

Tuning actions:

- increase misleading affordance strength moderately
- reduce immediate lethality
- add delayed cost instead of immediate cost
- increase `niche_stability` contrast
- reduce `beta_pi` only in controlled smoke tests
- increase recipe diversity if a single context dominates

### 4.5 Stage E: Learnable Uncertainty

Purpose:

Ensure uncertainty is not random noise.

Primary metrics:

- `history_disambiguation_score`
- `prediction_error_reducibility`
- `prediction_error_autocorrelation`
- `cue_to_error_lead_time`
- `recipe_trigger_rate`

Default initial thresholds:

```text
min_history_disambiguation_score: 0.02
min_prediction_error_reducibility: 0.01
min_recipe_trigger_rate: 0.20
max_prediction_error_white_noise_score: 0.80
```

Tuning actions:

- reduce observation noise
- increase recoverable cue strength
- increase delay if cue and effect are simultaneous
- decrease delay if cue and effect are too far apart
- convert pure noise to conditional bias
- require at least one hidden rule with repeated-contact or flow dependency

### 4.6 Stage F: Dataset Gate

Purpose:

Promote only recipe sets that improve dataset usefulness.

Primary metrics:

- retained episode count
- recipe diversity
- action entropy
- recovery fraction
- stress defensive fraction
- stress exploit fraction
- success/failure mix
- external-state contact variability

Promotion condition:

```text
dataset passes existing agentic gates
and structured uncertainty metrics pass provisional thresholds
and no single recipe contributes more than 60% of retained episodes
```

## 5. Failure Mode Taxonomy

Every rejected recipe or tuning run SHOULD be assigned one primary failure
mode.

```text
invalid_json
schema_missing_required_key
unknown_field
unmapped_rule_type
immediate_extinction
no_contact_variability
action_collapse
pure_noise_uncertainty
untriggered_hidden_rule
unlearnable_hidden_rule
overly_easy
overly_hard
stress_exploit_dominant
insufficient_recovery
insufficient_recipe_diversity
```

The tuning system SHOULD count these failure modes and use them to propose the
next bounded adjustment.

## 6. Tuning Knobs

Initial tuning should be restricted to these knobs.

### 6.1 Recipe Knobs

```text
energy_gradient.patch_count
energy_gradient.intensity
energy_gradient.regen
thermal_stress.patch_count
thermal_stress.intensity
thermal_stress.drift
toxicity.patch_count
toxicity.intensity
toxicity.drift
niche_stability.patch_count
niche_stability.intensity
niche_stability.decay
flow.strength
flow.drift
latent_causal_rules[].delay_steps
latent_causal_rules[].strength
observation_distortion.*.bias
observation_distortion.*.noise
observation_distortion.*.delay_steps
```

### 6.2 Runtime Knobs

Runtime knobs should be tuned less often than recipe knobs.

```text
observation_noise
epistemic_scale
beta_pi
aperture_gain
aperture_width_deg
move_step
contact_w_energy
contact_w_thermal
contact_w_toxicity
contact_w_niche
```

Runtime tuning MUST be logged separately from recipe tuning.

## 7. Bounded Adjustment Rules

Use small multiplicative or additive changes.

Recommended defaults:

```text
intensity_step: +/- 10%
drift_step: +/- 15%
noise_step: +/- 10%
delay_step: +/- 2 steps
patch_count_step: +/- 1
flow_strength_step: +/- 10%
rule_strength_step: +/- 10%
```

Never tune more than three independent knob groups in one iteration.

Preferred order:

```text
1. recipe intensity / delay / cue knobs
2. observation distortion knobs
3. environment layout knobs
4. runtime policy knobs
5. dataset gate thresholds
```

Gate thresholds should be adjusted only after at least two runs show that the
metric is systematically mis-scaled rather than behaviorally bad.

## 8. Default Tuning Policy

Use the following first-pass policy.

### 8.1 If Episodes Die Too Early

Apply:

```text
toxicity.intensity *= 0.90
thermal_stress.intensity *= 0.90
latent_rule.strength *= 0.90
latent_rule.delay_steps += 2
```

Do not reduce energy difficulty in the same iteration unless starvation is the
dominant failure mode.

### 8.2 If Episodes Are Too Easy

Apply:

```text
toxicity.intensity *= 1.10
thermal_stress.drift *= 1.15
niche_stability.decay *= 1.10
flow.strength *= 1.10
```

Prefer delayed costs over immediate lethality.

### 8.3 If Action Collapse Occurs

Apply:

```text
increase delayed consequence strength
increase recoverable cue strength
reduce immediate reward clarity
increase recipe diversity
```

If collapse is caused by policy sharpness, test lower `beta_pi` in smoke runs
but do not make it the default fix.

### 8.4 If Uncertainty Is Pure Noise

Apply:

```text
observation_distortion.noise *= 0.80
observation_distortion.bias condition becomes rule-dependent
add or strengthen recoverable cue
ensure hidden rule depends on flow, repeated contact, or niche trend
```

### 8.5 If Hidden Rule Never Triggers

Apply:

```text
lower trigger threshold
increase relevant field patch count
increase flow overlap
reduce delay if episode ends before effect appears
```

### 8.6 If No Contact Variability Occurs

Apply:

```text
increase flow.strength
increase field contrast
reduce field_sigma_min
increase move_step only in smoke tests
```

## 9. Tuning Artifacts

Each tuning run MUST write:

```text
tuning_run_summary.json
tuning_actions.jsonl
candidate_recipes.raw.jsonl
candidate_recipes.clamped.jsonl
accepted_recipes.jsonl
rejected_recipes.jsonl
recipe_validation_report.json
smoke_metrics.jsonl
```

Each `tuning_actions.jsonl` row should contain:

```json
{
  "iteration": 0,
  "input_recipe_fingerprint": "...",
  "output_recipe_fingerprint": "...",
  "failure_mode": "action_collapse",
  "action": "increase delayed consequence strength",
  "knob_changes": {
    "latent_causal_rules[0].strength": {
      "old": 0.30,
      "new": 0.33
    }
  },
  "before_metrics": {},
  "after_metrics": {},
  "notes": "string"
}
```

## 10. Human Review Points

Most tuning should be automatic, but human review is required when:

- a new latent rule type is introduced
- a new canonical field is proposed
- a dataset gate threshold is relaxed
- LLM prompts are materially changed
- recipe metadata is proposed as a TRM input feature
- runtime policy knobs are changed globally

Human review is not required for:

- rejecting invalid recipes
- clamping numeric fields
- re-running smoke tests
- promoting recipes that pass established gates
- applying bounded tuning actions from this specification

## 11. Initial Defaults

Use these defaults for the first implementation:

```text
num_candidate_recipes: 16
smoke_episodes_per_recipe: 3
smoke_steps: 64
accepted_recipe_target: 4
max_tuning_iterations: 3
assignment: round_robin
default_model: deepseek-v4-flash
temperature: 0.7
max_tokens: 1800
```

Initial latent rule types:

```text
flow_exposed_toxicity
niche_decay_under_stress
toxicity_masked_until_repeated_contact
```

Initial observation distortion:

```text
external-state channels only
no Lenia 5-channel distortion
field-wise bias/noise only
delay buffers optional after first smoke pass
```

Initial TRM policy:

```text
recipe metadata is logged but not included as a TRM input feature
```

## 12. Summary

The tuning procedure is:

```text
generate recipes
validate and clamp
run smoke episodes
classify failures
apply bounded adjustments
promote only executed, gate-passing recipes
```

This makes "adjust while running" a reproducible protocol rather than an
informal manual process.
