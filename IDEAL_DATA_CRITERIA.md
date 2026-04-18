# Ideal Data Criteria

## 1. Purpose

This document defines what "ideal data" means for ERIE-on-Lenia.

The purpose is not to maximize raw dataset size. The purpose is to define the
conditions under which a dataset is information-dense enough to teach:

- world prediction
- boundary-aware observation
- self-maintenance
- action-dependent future branching
- defensive and exploratory behavior under tradeoff

This document is stricter than a smoke-run checklist and broader than a single
dataset harness gate. It describes the target shape of data that the project
should gradually approach.

## 2. Core Principle

Ideal data is not "large and clean."

Ideal data is:

**a set of trajectories in which action choices meaningfully change future
viability, observation uncertainty, and boundary condition under nontrivial
external tradeoffs.**

In practical terms, ideal data must force ERIE to need:

- belief update
- uncertainty handling
- selective action
- boundary regulation
- defensive correction

If a dataset can be solved by repeating `intake + seal` under nearly static
conditions, it is not ideal even if it is very large.

## 3. Ideal Data Must Teach

### MUST

An ideal dataset MUST contain enough variation to teach all of the following:

- when to approach and when to withdraw
- when intake is beneficial and when it is exploitative
- when sealing helps and when it causes starvation
- when reconfiguration is necessary to avoid long-term failure
- when prediction error should trigger corrective action rather than passive
  persistence

### SHOULD

- It SHOULD contain both fast local corrections and slower structural recovery.
- It SHOULD contain episodes where short-term gain and long-term survival point
  in different directions.
- It SHOULD force ERIE to trade off energy acquisition against stress exposure.

## 4. Data Families

Ideal data in this project is split into two major families.

### 4.1 Family A

World / boundary pretraining data for:

- `TRM-Wp`
- `TRM-Bd`

This family is ideal when it improves:

- one-step prediction
- short rollout stability
- boundary inference
- uncertainty calibration

### 4.2 Family B

Agentic bootstrap data for:

- `TRM-Vm`
- `TRM-As`

This family is ideal when it improves:

- viability monitoring
- defensive action selection
- action branching quality
- homeostatic regulation

## 5. Ideal External Conditions

Ideal data MUST be produced from a world with genuine external tradeoff.

For the current ERIE-on-Lenia implementation, the minimal world components are:

- `energy_gradient`
- `thermal_stress`
- `toxicity`
- `niche_stability`
- `flow`

These do not merely decorate the scene. They define whether action differences
matter.

### MUST

- Energy-rich regions MUST not be globally safe.
- High-stability regions MUST not always be energy-rich.
- Flow MUST be able to transport at least some environmental advantage or risk.
- The external state MUST create zones where:
  - approach is useful
  - withdrawal is useful
  - sealing is useful
  - reconfiguration is useful

### SHOULD

- Vent-like band structure SHOULD exist:
  - near-core: high energy, high danger
  - habitable band: balanced
  - distal zone: safer but poorer
- Flow SHOULD create temporally nontrivial transport rather than static fields
  only.

## 6. Ideal Observation Conditions

Ideal data MUST not assume omniscience.

### MUST

- Observation MUST be boundary-gated.
- The agent MUST not receive full external state as standard training input.
- Observation noise or distortion MUST vary with stress and niche condition.

### SHOULD

- Some episodes SHOULD include higher uncertainty corridors or partially
  misleading observations.
- Observational difficulty SHOULD vary enough that epistemic behavior can
  matter.

## 7. Ideal Trajectory Shape

Ideal data is trajectory-rich, not frame-rich.

### MUST

- A retained episode MUST contain action-relevant state change.
- A retained episode MUST contain at least one meaningful viability transition:
  - recovery
  - drift toward failure
  - stress escalation
  - defensive stabilization
- A retained episode MUST be long enough for action differences to appear in
  future homeostatic error.

### SHOULD

- Episodes SHOULD contain both immediate and delayed consequences.
- Some episodes SHOULD have non-obvious best actions.
- Some episodes SHOULD punish greedy energy maximization.

## 8. Action Branching Quality

This is the central criterion for ideal data.

The dataset is ideal to the extent that different actions produce genuinely
different futures.

### MUST

- The dominant action MUST NOT explain almost the entire retained dataset.
- At dataset level, all runtime actions MUST appear.
- At episode level, the data generator SHOULD reject traces that collapse into a
  single repetitive action without meaningful state branching.

### SHOULD

- The dataset SHOULD contain episodes in which:
  - `approach` is correct in one local context and wrong in another
  - `intake` is corrective in one local context and harmful in another
  - `seal` is protective in one local context and over-defensive in another
  - `reconfigure` is necessary for medium-horizon survival

## 9. Success / Failure Mix

Ideal data MUST not be all-success and MUST not be all-failure.

### MUST

- Family B MUST include both surviving and failing trajectories.
- Failure trajectories MUST include more than one failure style.

### SHOULD

- Failure styles SHOULD include at least:
  - starvation / low energy
  - boundary degradation
  - stress exploitation
  - over-defensive stagnation

## 10. Homeostatic Quality

Ideal data teaches ERIE to remain in a preferred band, not to maximize a scalar
without limit.

### MUST

- Dataset quality MUST be evaluated relative to target-band maintenance.
- Overshoot of `G` MUST be considered harmful when it is coupled to elevated
  stress exposure.
- `B` preservation MUST not be rewarded if it causes chronic energy failure.

### SHOULD

- Data selection SHOULD explicitly reward:
  - low homeostatic error
  - defensive recovery under stress
  - correction away from overshoot

## 11. Diversity Requirements

### 11.1 Environment Diversity

Ideal data SHOULD cover multiple episode families such as:

- `energy_starved`
- `toxic_band`
- `fragile_boundary`
- `vent_edge`
- `uncertain_corridor`

No single family should dominate the retained set.

### 11.2 Regime Diversity

Family A SHOULD include both relatively stable and more chaotic Lenia regimes.

Family B SHOULD include:

- easier corrective episodes
- ambiguous episodes
- strongly defensive episodes
- exploit-prone episodes

### 11.3 Seed Diversity

### MUST

- Splits MUST be seed-disjoint.
- Evaluation MUST not rely on frame-level leakage.

## 12. Quantitative Signals of Ideality

Ideality is approximate. It cannot be reduced to one scalar.

The following signals SHOULD all move in the correct direction.

### 12.1 Dataset-Level Signals

- retained episode count
- family coverage
- policy mode coverage
- aggregate action entropy
- aggregate dominant-action fraction
- mean episode policy entropy
- mean distinct-actions per episode
- recovery fraction
- stress-defensive fraction
- stress-exploit fraction
- success/failure ratio

### 12.2 Model-Level Signals

- `TRM-Wp` rollout stability
- `TRM-Bd` boundary plausibility
- `TRM-Vm` viability MAE and risk discrimination
- `TRM-As` pairwise ranking accuracy
- `TRM-As` action collapse rate

### 12.3 Runtime-Level Signals

- `closed_loop > no_action`
- `closed_loop >= random` on mean homeostatic performance
- `module_primary` or `assistive` modes improving over analytic baseline under
  at least some seeds
- reduced stress exploitation without trivial over-sealing

## 13. Anti-Patterns

The following are explicitly not ideal:

- very large datasets with low action diversity
- datasets dominated by stable, low-event trajectories
- datasets in which `intake + seal` solves nearly everything
- datasets with near-deterministic teacher action labels
- datasets with almost no failure traces
- datasets with one episode family overrepresented
- datasets where action differences have little effect on future viability

## 14. Acceptance Tiers

### Tier 0: Smoke

Useful for:

- pipeline testing
- shape validation
- file-format validation

Not useful for:

- production training claims
- ideal-data claims

### Tier 1: Canonical

Useful for:

- local iteration
- architecture comparison
- regression testing

Still not ideal if:

- action branching remains weak
- success/failure balance remains thin

### Tier 2: Production

Useful for:

- serious training
- local and remote handoff
- module comparison

Still not ideal if:

- the data is large but behaviorally shallow

### Tier 3: Ideal-Oriented

A dataset may be treated as ideal-oriented only when:

- environment tradeoffs are strong
- action branching is robust
- success and failure are both informative
- homeostatic calibration matters more than scalar maximization
- learned modules improve runtime behavior without collapsing into repetitive
  control

## 15. Practical Guidance

When choosing between:

- more episodes
- stricter quality gates
- richer environment families

the recommended order is:

1. increase action-relevant family diversity
2. enforce moderate quality gates
3. increase episode count
4. only then tighten gates further

This reflects the current empirical result of the project:

- overly loose data leads to action collapse
- overly strict data can reduce runtime robustness
- the best datasets so far are those that preserve family diversity while
  discouraging exploit-heavy traces

## 16. Current Project Interpretation

For the current repository state, "ideal data" does **not** yet mean:

- biologically realistic data
- direct hydrothermal vent measurement ingestion
- final free-energy-complete supervision

For the current repository state, "ideal data" means:

- information-dense ERIE-on-Lenia trajectories
- external-state tradeoffs strong enough to matter
- boundary-gated observations
- action-dependent viability differences
- sufficient diversity that learned modules can beat or approach analytic
  baselines without collapsing

## 17. Relationship to Existing Documents

This document refines, but does not replace:

- [TRM_REQUIREMENTS_MUST_SHOULD.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_REQUIREMENTS_MUST_SHOULD.md)
- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md)
- [EXTERNAL_STATE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/EXTERNAL_STATE_SPEC.md)
- [HARNESS_OPERATIONS_GUIDE_2026-04-04.md](/Users/yamaguchimitsuyuki/criticism_bot/HARNESS_OPERATIONS_GUIDE_2026-04-04.md)

This document should be used when:

- revising dataset acceptance thresholds
- adding new episode families
- deciding whether a larger dataset is actually better
- deciding whether a bootstrap teacher is too behaviorally narrow
