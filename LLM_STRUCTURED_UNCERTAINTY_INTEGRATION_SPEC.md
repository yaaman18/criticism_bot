# LLM Structured Uncertainty Integration Specification

## 1. Purpose

This document designs how to integrate a low-cost LLM API into the current
ERIE-on-Lenia program without letting the LLM replace simulator dynamics.

The LLM is used only to propose `structured_uncertainty_recipe` objects. The
runtime remains responsible for all actual state transitions, observations,
prediction errors, viability changes, and dataset acceptance.

This specification extends `STRUCTURED_UNCERTAINTY_SPEC.md`.

Runtime-driven tuning policy is specified separately in:

- `STRUCTURED_UNCERTAINTY_TUNING_SPEC.md`

## 2. Core Protocol

The integration MUST follow this protocol:

```text
LLM API
  -> raw recipe JSON
  -> schema validation
  -> numeric clamping
  -> deterministic translation
  -> ERIE-on-Lenia simulator execution
  -> structured uncertainty metrics
  -> dataset gate
  -> TRM training manifests
```

The LLM MUST NOT be called inside the per-step runtime loop.

The LLM MAY be called before collection to create recipe candidates, or after a
failed campaign to propose revised recipe candidates.

## 3. Recommended API Type

Use an OpenAI-compatible Chat Completions client.

Recommended default provider/model:

```text
provider: deepseek
base_url: https://api.deepseek.com
model: deepseek-v4-flash
response_format: {"type": "json_object"}
```

Recommended escalation model:

```text
model: deepseek-v4-pro
use only for recipe review, failed recipe diagnosis, or difficult curriculum design
```

The integration SHOULD be provider-agnostic. The code should treat DeepSeek as
one backend, not as a hard dependency of the data model.

## 4. New Modules

Add the following modules.

### 4.1 `trm_pipeline/structured_uncertainty.py`

Owns schema, validation, clamping, deterministic translation, and metrics.

Responsibilities:

- define dataclasses or typed dicts for recipe objects
- validate required keys
- reject unknown canonical fields
- clamp all numeric values
- produce deterministic `EnvironmentConfig` overlays
- produce observation distortion config
- produce latent causal rule config
- compute recipe fingerprints
- write recipe manifests

Key functions:

```python
load_structured_uncertainty_recipe(path) -> dict
validate_structured_uncertainty_recipe(recipe: dict) -> dict
clamp_structured_uncertainty_recipe(recipe: dict) -> dict
recipe_fingerprint(recipe: dict) -> str
translate_recipe_to_runtime_config(recipe: dict, base_env: EnvironmentConfig) -> StructuredUncertaintyRuntimeConfig
write_recipe_manifest(path, recipes: list[dict]) -> None
```

### 4.2 `trm_pipeline/llm_recipe_client.py`

Owns provider calls and retry logic.

Responsibilities:

- call OpenAI-compatible chat completions
- request JSON-only output
- support dry-run mode from local fixture files
- capture request metadata
- retry empty/invalid JSON responses
- never execute simulator logic

Key functions:

```python
generate_structured_uncertainty_recipe(prompt_config: dict) -> dict
generate_structured_uncertainty_batch(batch_config: dict) -> list[dict]
```

Required environment variables:

```text
DEEPSEEK_API_KEY
```

Optional environment variables:

```text
LLM_RECIPE_PROVIDER
LLM_RECIPE_MODEL
LLM_RECIPE_BASE_URL
```

### 4.3 `trm_pipeline/structured_uncertainty_campaign.py`

Owns the offline recipe-generation campaign.

Responsibilities:

- create N recipe candidates
- validate and clamp recipes
- execute smoke episodes for each recipe
- score learnability/action branching
- write accepted and rejected manifests

CLI shape:

```bash
./.venv/bin/python -m trm_pipeline.structured_uncertainty_campaign \
  --output-root artifacts/structured_uncertainty_campaign \
  --num-recipes 32 \
  --model deepseek-v4-flash \
  --seed 20260519 \
  --smoke-steps 64 \
  --accept-top-k 8
```

## 5. Runtime Extension Points

The current runtime should be extended minimally.

### 5.1 `EnvironmentConfig`

`EnvironmentConfig` already contains most field-level knobs:

- `resource_patches`
- `hazard_patches`
- `shelter_patches`
- `resource_regen`
- `hazard_drift_sigma`
- `toxicity_drift_sigma`
- `shelter_stability`
- `flow_strength`
- `flow_drift_sigma`
- `field_sigma_min`
- `field_sigma_max`

The recipe translator SHOULD map LLM field parameters onto these existing
knobs first.

Do not add many new environment fields before the translator proves useful.

### 5.2 Observation Distortion Hook

Add a small config object to runtime, not an LLM dependency:

```python
@dataclass(frozen=True)
class ObservationDistortionConfig:
    enabled: bool = False
    energy_bias: float = 0.0
    thermal_bias: float = 0.0
    toxicity_bias: float = 0.0
    niche_bias: float = 0.0
    flow_bias: float = 0.0
    energy_noise_scale: float = 0.0
    thermal_noise_scale: float = 0.0
    toxicity_noise_scale: float = 0.0
    niche_noise_scale: float = 0.0
    flow_noise_scale: float = 0.0
    energy_delay_steps: int = 0
    thermal_delay_steps: int = 0
    toxicity_delay_steps: int = 0
    niche_delay_steps: int = 0
    flow_delay_steps: int = 0
```

This config should be applied inside `_observation_mapping()` after true
`env_channels` are assembled and before `sensor_gate` mixes observation with
belief.

Important:

- true `external_state` must remain logged separately
- distorted `observation` is what the agent receives
- distortion config must be saved in episode summary

### 5.3 Latent Causal Rule Hook

Add a deterministic rule engine outside the LLM:

```python
@dataclass(frozen=True)
class LatentCausalRuleConfig:
    name: str
    condition: str
    effect: str
    delay_steps: int
    strength: float
    reversibility: str
```

Initial implementation should support a small fixed grammar, not arbitrary
natural language execution.

Allowed initial rule types:

```text
flow_exposed_toxicity
thermal_delayed_boundary_stress
niche_decay_under_stress
energy_overestimate_under_flow
toxicity_masked_until_repeated_contact
```

The LLM may output natural-language descriptions, but the validator must map
them to one of these fixed rule types or reject the recipe.

### 5.4 Recipe Metadata In Episode Logs

Each episode summary should include:

```json
{
  "structured_uncertainty": {
    "enabled": true,
    "recipe_id": "...",
    "recipe_fingerprint": "...",
    "recipe_path": "...",
    "latent_rule_types": ["flow_exposed_toxicity"],
    "observation_distortion_enabled": true
  }
}
```

Dataset manifest rows should include:

```json
{
  "structured_uncertainty_recipe_id": "...",
  "structured_uncertainty_fingerprint": "...",
  "structured_uncertainty_family": "latent_causal"
}
```

## 6. Dataset Harness Integration

### 6.1 Contract Extension

Extend agentic dataset contracts with:

```json
{
  "generator": {
    "structured_uncertainty": {
      "enabled": true,
      "recipe_manifest": "artifacts/.../accepted_recipes.jsonl",
      "assignment": "round_robin | random_seeded | top_k_weighted",
      "max_recipes": 8
    }
  },
  "acceptance": {
    "require_structured_uncertainty_gate": true,
    "min_structured_uncertainty_recipe_count": 2,
    "min_history_disambiguation_score": 0.05,
    "min_prediction_error_reducibility": 0.02,
    "min_affordance_ambiguity_score": 0.10
  }
}
```

### 6.2 Collection Flow

Current flow:

```text
dataset_harness -> prepare_trm_va_cache -> ERIERuntime -> manifest
```

New flow:

```text
dataset_harness
  -> load accepted recipe manifest
  -> assign recipe per episode or per attempt
  -> translate recipe to EnvironmentConfig + runtime structured config
  -> prepare_trm_va_cache
  -> ERIERuntime
  -> manifest with recipe metadata
```

For the first implementation, recipe assignment should happen per episode
family attempt in `prepare_trm_va_data.py`, because that file already owns
family-level variation such as `uncertain_corridor`.

## 7. Interaction With Existing Episode Families

Do not replace existing episode families.

Add recipe-backed variants:

```text
energy_starved
toxic_band
fragile_boundary
vent_edge
uncertain_corridor
structured_uncertainty
```

The first recipe-backed family should be:

```text
structured_uncertainty
```

Later, recipes may specialize existing families:

```text
uncertain_corridor + latent_causal_recipe
toxic_band + latent_causal_recipe
vent_edge + latent_causal_recipe
```

## 8. Validation And Gates

Validation must happen at three levels.

### 8.1 Recipe-Level Validation

Reject before simulation if:

- JSON is invalid
- required keys are missing
- unknown canonical fields are present
- numeric values exceed clampable bounds too severely
- no latent causal rule maps to known rule types
- `pure_noise` is true
- no recoverable cue is specified

### 8.2 Smoke Simulation Gate

Reject after short simulator runs if:

- all episodes die immediately
- no action diversity occurs
- contact variance is near zero
- prediction error is absent or pure noise
- one action dominates without branching
- hidden rule never triggers

### 8.3 Dataset Gate

Only accepted executed episodes enter TRM manifests.

Add dataset-level criteria:

- recipe diversity
- structured uncertainty episode fraction
- history disambiguation score
- affordance ambiguity score
- prediction error reducibility
- delayed consequence count

## 9. Metrics Definitions

Initial metrics can be approximate.

```text
history_disambiguation_score:
  improvement in predicting next contact using recent history over current
  observation only

prediction_error_reducibility:
  reduction of contact prediction error after repeated exposure or corrective
  actions

affordance_ambiguity_score:
  frequency of low action-score margin combined with later divergent viability
  outcomes

delayed_consequence_count:
  number of detected cases where action has low immediate cost but delayed
  boundary or viability cost

recipe_trigger_rate:
  fraction of episodes where at least one latent causal rule activates
```

## 10. Cost Control

The default campaign should be cheap:

```text
model: deepseek-v4-flash
temperature: 0.6..0.8
max_tokens: 1200..2200
batch size: small, retry invalid only
cacheable system prompt: yes
```

Use `deepseek-v4-pro` only for:

- generating a small seed set of high-quality recipes
- reviewing rejected recipe clusters
- revising the prompt contract

Do not call any model during per-step runtime.

Do not include full episode logs in LLM prompts. Summarize failures into small
diagnostic reports if recipe revision is needed.

## 11. Security And Reproducibility

The LLM integration must be reproducible and inspectable.

Every generated recipe should store:

- provider
- model
- base URL name, not secret
- prompt version
- schema version
- raw response
- parsed recipe
- clamped recipe
- validation report
- recipe fingerprint
- creation timestamp

Never store API keys in artifacts.

The simulator run must be reproducible from:

- seed catalog
- runtime config
- environment config
- structured uncertainty recipe fingerprint
- clamped recipe JSON

## 12. Implementation Order

### Step 1: Offline Recipe Support

Implement:

- `structured_uncertainty.py`
- local JSON recipe validation
- clamping
- manifest writing
- no LLM call yet

### Step 2: Runtime Observation Distortion

Implement:

- `ObservationDistortionConfig`
- delay buffers
- field-wise bias/noise
- episode summary metadata

### Step 3: Fixed Latent Rule Engine

Implement:

- small fixed rule grammar
- trigger counters
- delayed effects
- metadata logging

### Step 4: Dataset Harness Hook

Implement:

- contract `structured_uncertainty` block
- recipe manifest loading
- deterministic recipe assignment
- manifest metadata
- dataset-level gate placeholders

### Step 5: LLM Client

Implement:

- DeepSeek/OpenAI-compatible client
- JSON-only generation
- retries
- prompt versioning
- raw/clamped artifacts

### Step 6: Campaign Runner

Implement:

- batch recipe generation
- smoke simulation
- recipe ranking
- accepted/rejected recipe manifests

### Step 7: Tuning Loop

Implement the bounded tuning loop from
`STRUCTURED_UNCERTAINTY_TUNING_SPEC.md`.

The first implementation should use the default policy:

```text
num_candidate_recipes: 16
smoke_episodes_per_recipe: 3
smoke_steps: 64
accepted_recipe_target: 4
max_tuning_iterations: 3
assignment: round_robin
default_model: deepseek-v4-flash
```

## 13. Minimal First Slice

The smallest useful implementation is:

```text
1. hand-write 3 structured_uncertainty_recipe JSON files
2. validate and clamp them
3. translate only EnvironmentConfig fields
4. add observation distortion bias/noise, no delayed buffers yet
5. run agentic dataset with recipe metadata
6. evaluate existing action diversity and external contact gates
```

Only after this passes should the LLM API be connected.

## 14. Summary

The LLM protocol should be integrated as an offline curriculum generator, not a
runtime controller.

The durable architecture is:

```text
LLMStructuredRecipeClient
  -> StructuredUncertaintyRecipeValidator
  -> StructuredUncertaintyRecipeTranslator
  -> ERIERuntime structured hooks
  -> DatasetHarness gates
  -> TRM manifests
```

This keeps cost low, preserves causal validity, and allows DeepSeek V4 Flash to
serve as the default generator while retaining the option to use V4 Pro only
for harder recipe-design work.
