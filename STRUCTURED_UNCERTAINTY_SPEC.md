# Structured Uncertainty Specification

## 1. Purpose

This document defines how ERIE-on-Lenia should introduce uncertainty that is
not pure noise, but remains learnable through action, history, prediction
error, and viability change.

The purpose is to support TRM pretraining and agentic bootstrap data in a way
that is compatible with the current project direction:

- Lenia is treated first as `z_lenia`, intrinsic CA dynamics.
- External state is represented by canonical fields:
  `energy_gradient`, `thermal_stress`, `toxicity`, `niche_stability`, and
  `flow`.
- Agentic significance should arise through boundary-gated contact,
  action-conditioned transition, prediction error, and self-maintenance
  pressure.
- LLM output should not replace simulator-generated trajectories.

This specification is a conceptual and implementation bridge between
enactivist framing and dataset generation.

## 2. Concept Hierarchy

The following terms MUST NOT be treated as parallel categories. They occupy
different levels.

```text
Enactive Uncertainty
  -> Structured Uncertainty
      -> Latent-Causal Uncertainty
          -> Affordance Ambiguity
              -> prediction error / viability change / boundary change
```

### 2.1 Enactive Uncertainty

`Enactive uncertainty` is the philosophical top-level frame.

It refers to uncertainty that arises for a self-maintaining agent through its
ongoing coupling with an external world. The relevant uncertainty is not merely
missing information. It is a disturbance or ambiguity that matters because it
affects boundary maintenance, viability, action, and future contact.

In ERIE terms, enactive uncertainty requires:

- a bounded agent-like process
- boundary-gated observation
- action-conditioned contact
- surprise / prediction error that affects later behavior
- viability or self-maintenance pressure
- history-dependent updating

This term SHOULD be used for theory-level discussion. It SHOULD NOT be used as
the name of a concrete generator class or data file.

### 2.2 Structured Uncertainty

`Structured uncertainty` is the dataset-design category.

It is defined as:

```text
Uncertainty where a single observation is insufficient to determine the true
external state or action value, but where history, action consequences,
contact prediction error, and viability change can partially reduce that
uncertainty.
```

Structured uncertainty is the operational form of enactive uncertainty for
ERIE-on-Lenia datasets.

It SHOULD be used in:

- dataset contracts
- curriculum descriptions
- acceptance criteria
- evaluation summaries
- prompt and recipe naming

### 2.3 Latent-Causal Uncertainty

`Latent-causal uncertainty` is the implementation mechanism.

It introduces hidden or delayed causal rules behind the external state or the
observation mapping. These rules make the agent's local observation ambiguous
without making the episode random or impossible to learn.

Examples:

- High `energy_gradient` combined with sustained high `flow` causes delayed
  `toxicity` exposure.
- `niche_stability` appears high in the short term but decays under prolonged
  `thermal_stress`.
- `toxicity` is under-observed near certain boundary states until repeated
  contact increases prediction error.
- `thermal_stress` is sensed with a delay, while boundary degradation appears
  earlier.

LLMs MAY generate latent-causal uncertainty recipes. The simulator MUST execute
the resulting dynamics and produce the actual trajectory data.

### 2.4 Affordance Ambiguity

`Affordance ambiguity` is the action-facing phenomenon that emerges when
latent-causal uncertainty affects the agent.

It means the same observed situation supports multiple plausible action
interpretations.

Examples:

- A region appears worth approaching because `energy_gradient` is high, but the
  same region may require withdrawal because hidden `toxicity` is rising.
- Sealing appears protective under stress, but persistent sealing may cause
  energy starvation.
- Intake gives short-term recovery while increasing later boundary damage.
- Reconfiguration is costly in the short term but prevents delayed collapse.

Affordance ambiguity SHOULD be evaluated from executed trajectories, not
directly declared as a label by an LLM.

## 3. Non-Goals

This project SHOULD NOT describe this mechanism as `chaos` unless the
implementation specifically depends on mathematical chaos such as deterministic
nonperiodicity or sensitivity to initial conditions.

The current target is not chaos simulation. The target is learnable,
structured, action-relevant uncertainty.

The following are explicit non-goals:

- LLM-generated full trajectories
- LLM-generated optimal actions
- LLM-generated viability labels
- pure random observation corruption
- opaque hidden rules that cannot be inferred from history
- uncertainty that does not affect action branching
- uncertainty that bypasses boundary-gated observation

## 4. System Roles

### 4.1 LLM Role

The LLM MAY be used as a `LatentCausalRecipeGenerator`.

It may generate:

- latent environmental context
- hidden causal rules
- observation distortion policies
- delayed exposure rules
- regime shift triggers
- recoverable cues
- bounded parameters for canonical external fields
- validation expectations

It MUST NOT generate:

- rollout arrays
- per-step state trajectories
- authoritative action labels
- final training targets
- unbounded parameters
- fields outside the canonical vocabulary

### 4.2 Simulator Role

The ERIE-on-Lenia simulator MUST remain the executor of actual dynamics.

It is responsible for:

- updating `z_lenia`
- updating canonical external state fields
- applying action-conditioned world updates
- producing boundary-gated observations
- recording actual contact
- recording prediction error
- recording viability and boundary changes
- determining whether the episode passes dataset gates

### 4.3 TRM Role

TRM modules learn from executed data.

Expected mappings:

- `TRM-Wp`: world prediction under latent-causal uncertainty
- `TRM-Bd`: boundary inference under distorted observation
- `TRM-Vm`: viability estimation under hidden risk and delayed consequence
- `TRM-As`: action selection under affordance ambiguity

## 5. Required Data Structure

Every structured uncertainty episode SHOULD preserve the distinction between:

```text
z_ext: true external state
o_t: boundary-gated and possibly distorted observation
a_t: action
z_ext(t+1): next external state
o_t+1: next observation
prediction_t+1: predicted next contact or state proxy
prediction_error_t+1: actual minus predicted contact/state proxy
viability_t+1: next viability state
boundary_t+1: next boundary state
```

At minimum, logs SHOULD contain:

- canonical external contact:
  - `contact_energy`
  - `contact_thermal`
  - `contact_toxicity`
  - `contact_niche`
- observation uncertainty state
- predicted next contact, once implemented
- actual next contact
- contact prediction error
- action taken
- policy score or action score
- homeostatic error
- `G`
- `B`
- death / failure status

## 6. LLM Recipe Schema

LLM output SHOULD be JSON-only and SHOULD conform to this shape.

```json
{
  "schema_version": 1,
  "scenario_id": "string",
  "difficulty": "easy | medium | medium_hard | hard",
  "latent_context": "string",
  "canonical_fields": {
    "energy_gradient": {
      "patch_count": 0,
      "intensity": 0.0,
      "regen": 0.0,
      "spatial_scale": 0.0
    },
    "thermal_stress": {
      "patch_count": 0,
      "intensity": 0.0,
      "drift": 0.0,
      "spatial_scale": 0.0
    },
    "toxicity": {
      "patch_count": 0,
      "intensity": 0.0,
      "drift": 0.0,
      "spatial_scale": 0.0
    },
    "niche_stability": {
      "patch_count": 0,
      "intensity": 0.0,
      "decay": 0.0,
      "spatial_scale": 0.0
    },
    "flow": {
      "strength": 0.0,
      "drift": 0.0,
      "directionality": 0.0
    }
  },
  "latent_causal_rules": [
    {
      "name": "string",
      "condition": "string",
      "effect": "string",
      "delay_steps": 0,
      "strength": 0.0,
      "reversibility": "reversible | partially_reversible | irreversible_within_episode",
      "recoverable_cues": ["string"]
    }
  ],
  "observation_distortion": {
    "energy_gradient": {
      "bias": 0.0,
      "noise": 0.0,
      "delay_steps": 0,
      "masking_condition": "string"
    },
    "thermal_stress": {
      "bias": 0.0,
      "noise": 0.0,
      "delay_steps": 0,
      "masking_condition": "string"
    },
    "toxicity": {
      "bias": 0.0,
      "noise": 0.0,
      "delay_steps": 0,
      "masking_condition": "string"
    },
    "niche_stability": {
      "bias": 0.0,
      "noise": 0.0,
      "delay_steps": 0,
      "masking_condition": "string"
    },
    "flow": {
      "bias": 0.0,
      "noise": 0.0,
      "delay_steps": 0,
      "masking_condition": "string"
    }
  },
  "regime_shift_triggers": [
    {
      "trigger": "string",
      "effect": "string",
      "delay_steps": 0,
      "reversibility": "reversible | partially_reversible | irreversible_within_episode"
    }
  ],
  "learnability_constraints": {
    "single_observation_ambiguous": true,
    "history_reduces_uncertainty": true,
    "action_changes_future_observation": true,
    "prediction_error_has_structure": true,
    "pure_noise": false
  },
  "validation_expectations": {
    "expected_prediction_error_pattern": "string",
    "expected_viability_tradeoff": "string",
    "expected_affordance_ambiguity": "string",
    "failure_modes": ["string"]
  }
}
```

## 7. Parameter Bounds

The exact bounds MAY evolve with runtime implementation, but the recipe
translator MUST clamp all numeric values.

Initial suggested bounds:

```text
patch_count: 0..8
intensity: 0.0..1.0
regen: 0.0..0.01
drift: 0.0..0.005
decay: 0.0..0.10
spatial_scale: 2.0..16.0
flow.strength: 0.0..1.5
flow.directionality: 0.0..1.0
delay_steps: 0..32
rule.strength: 0.0..1.0
observation.bias: -0.5..0.5
observation.noise: 0.0..0.5
```

Recipe execution MUST remain stable if the LLM emits boundary values.

## 8. Learnability Criteria

A structured uncertainty episode is useful only if it is uncertain and
learnable.

### 8.1 Must Pass

An episode MUST satisfy:

- Single-step observation is insufficient for reliable action valuation.
- History improves prediction of at least one contact or viability signal.
- Different actions produce measurably different future observations or
  viability outcomes.
- Prediction error is not white noise.
- Hidden risk or delayed cost is visible through at least one recoverable cue.
- The episode contains at least one meaningful tradeoff between energy,
  stress, toxicity, niche stability, and boundary preservation.

### 8.2 Must Reject

An episode MUST be rejected if:

- observation distortion is independent random noise
- action does not affect future state or observation
- one action dominates the whole episode without meaningful branching
- all cues are misleading
- hidden rules are impossible to infer from logged history
- the episode is all success or all failure without informative transitions

## 9. Suggested Metrics

The following metrics SHOULD be added or derived as implementation matures.

### 9.1 Uncertainty Metrics

- `observation_ambiguity_score`
- `history_disambiguation_score`
- `uncertainty_reduction_after_action`
- `prediction_error_autocorrelation`
- `prediction_error_reducibility`

### 9.2 Latent-Causal Metrics

- `hidden_rule_trigger_count`
- `delayed_effect_count`
- `mean_trigger_to_effect_delay`
- `recoverable_cue_presence`
- `cue_to_error_lead_time`

### 9.3 Affordance Metrics

- `action_score_margin`
- `action_rank_flip_count`
- `dominant_action_fraction`
- `distinct_actions_per_episode`
- `counterfactual_action_branching_score`

### 9.4 Viability Metrics

- `homeostatic_error_delta_after_action`
- `stress_exploit_fraction`
- `defensive_recovery_fraction`
- `energy_overshoot_with_stress`
- `boundary_degradation_after_short_term_gain`

## 10. Prompt Contract

The LLM system prompt SHOULD state:

```text
You are a latent-causal uncertainty recipe generator for ERIE-on-Lenia.

You do not simulate Lenia.
You do not generate trajectories.
You do not choose optimal actions.
You generate bounded, executable recipes that create uncertain but learnable
observation conditions.

The simulator will execute actual dynamics. Your output must specify only:
- canonical external fields
- hidden causal rules
- observation distortion rules
- regime shift triggers
- recoverable cues
- validation expectations

The scenario must be ambiguous from a single observation but partially
inferable from action history, contact prediction error, and viability change.

Use only these fields:
energy_gradient, thermal_stress, toxicity, niche_stability, flow.

Output JSON only.
```

The prompt SHOULD include explicit anti-goals:

```text
Do not output full state arrays.
Do not output per-step trajectories.
Do not output correct actions.
Do not output direct training labels.
Do not create pure random noise.
Do not make all signals misleading.
Do not create hidden rules with no recoverable cue.
Do not add fields outside the canonical vocabulary.
```

## 11. Implementation Plan

### Phase 0: Documentation And Manual Recipes

- Add this specification.
- Create a small set of hand-written recipe examples.
- Do not connect an LLM yet.

### Phase 1: Runtime Preconditions

Before LLM recipes become useful, runtime should implement:

- spatial memory
- contact prediction error
- logging of predicted and actual contact
- structured observation distortion hooks
- canonical `z_ext` versus `o_t` separation

### Phase 2: Recipe Translator

Implement a deterministic translator:

```text
structured_uncertainty_recipe.json
  -> validated and clamped config
  -> EnvironmentConfig / observation distortion config
  -> runtime episode
  -> dataset gate
```

The translator MUST be deterministic for a fixed recipe and seed.

### Phase 3: LLM-Assisted Curriculum

Use an LLM to propose latent-causal recipes.

The proposed recipes MUST be:

- schema-validated
- clamped
- executed in simulator
- scored by learnability and action-branching metrics
- rejected if they fail dataset gates

### Phase 4: TRM Training Use

Only simulator-executed and gate-passing trajectories may enter TRM training.

LLM-generated recipe metadata MAY be retained for analysis but SHOULD NOT be
used as direct target labels.

## 12. File And Class Naming

Recommended names:

```text
STRUCTURED_UNCERTAINTY_SPEC.md
structured_uncertainty_recipe.json
structured_uncertainty_manifest.jsonl
StructuredUncertaintyCurriculum
LatentCausalRecipeGenerator
LatentCausalRecipeValidator
LatentCausalRecipeTranslator
AffordanceAmbiguityEvaluator
StructuredUncertaintyDatasetGate
```

Avoid names based on `chaos` unless mathematical chaos is explicitly being
implemented.

## 13. Relation To Existing Documents

This specification extends:

- `EXTERNAL_STATE_SPEC.md`
- `IDEAL_DATA_CRITERIA.md`
- `TRM_INPUT_VIEW_SPEC.md`
- `TRM_REQUIREMENTS_MUST_SHOULD.md`
- `2026-05-19_引き継ぎ_進捗状況.md`

It does not replace those documents.

Concrete LLM API integration is specified separately in:

- `LLM_STRUCTURED_UNCERTAINTY_INTEGRATION_SPEC.md`

The most important dependency is the distinction between:

```text
z_ext: external state
o_t: boundary-gated observation
q(z)_t: belief state
```

Structured uncertainty is meaningful only if those layers remain distinct.

## 14. Summary

The correct role division is:

```text
LLM:
  proposes latent-causal uncertainty recipes

Simulator:
  executes actual external-state and observation dynamics

Runtime logs:
  preserve action, contact, prediction error, viability, and boundary change

Dataset gate:
  accepts only uncertain but learnable episodes

TRM:
  learns prediction, boundary inference, viability, and action selection from
  executed trajectories
```

This keeps enactive uncertainty as a theoretical frame, structured uncertainty
as the dataset condition, latent-causal uncertainty as the implementable
mechanism, and affordance ambiguity as the action-facing phenomenon.
