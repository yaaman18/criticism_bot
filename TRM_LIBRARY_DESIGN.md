# TRM Library Design

## 1. Purpose

This document defines the brain-inspired TRM library strategy for ERIE.

The goal is not to build one large general model. The goal is to build a bank
of small `~7M` TRMs, each with a narrow computational responsibility, and later
compose them into larger recurrent and workspace-like assemblies.

This document therefore answers three questions:

1. What kinds of TRMs should exist?
2. What brain-function motif should each TRM approximate?
3. Which TRMs should be built first?

For the dataset-side view slicing rules that determine what each TRM should
actually receive as input, see
[TRM_INPUT_VIEW_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_INPUT_VIEW_SPEC.md).

For the dedicated contextual-memory design, see
[TRM_MC_CONTEXT_MEMORY_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_MC_CONTEXT_MEMORY_SPEC.md).
For the action-gating design, see
[TRM_AG_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_AG_SPEC.md).

## 2. Design Principle

### MUST

- TRMs MUST be inspired by brain function, not treated as literal copies of
  anatomical regions.
- Each TRM MUST keep one dominant responsibility under the `~7M params`
  constraint.
- The library MUST support many pre-trained modules with a shared contract and
  later selective composition.

### SHOULD

- A TRM SHOULD be identified by:
  - `role`
  - `brain_motif`
  - `regime_specialization`
  - `objective_variant`
- The same role SHOULD be allowed to exist in multiple variants.
- Runtime composition SHOULD select from the library rather than requiring all
  modules to be active at once.

## 3. Shared Module Contract

Every TRM in the library SHOULD map to the shared contract:

- `module_state`
- `module_precision`
- `module_error`
- `module_aux`
- `module_role`

This is required so that heterogeneous TRMs can later be connected inside:

- primary-only execution
- assistive execution
- multi-TRM recurrent loops
- future GNW-style workspace routing

## 4. Library Axes

The library is indexed along three axes.

### 4.1 Role Axis

The role axis states what the TRM computes.

- `W`: world
- `B`: boundary
- `V`: viability
- `A`: action
- `M`: memory
- `S`: salience
- `X`: workspace
- `C`: calibration

### 4.2 Brain-Motif Axis

The brain-motif axis states what systems-neuroscience function the TRM is
modeled after.

- hippocampal / entorhinal
- thalamic
- hypothalamic
- basal-ganglia
- insular
- amygdalar
- cingulate
- cerebellar
- prefrontal

### 4.3 Regime / Objective Axis

Each TRM may also specialize by:

- `stable`
- `chaotic`
- `fragile_boundary`
- `toxic_band`
- `vent_edge`
- `uncertain_corridor`

and by objective style:

- `baseline`
- `defensive`
- `exploratory`
- `calibrating`

## 5. Brain-Inspired TRM Families

## 5.1 Hypothalamic / Interoceptive Family

### TRM-Vm

**Brain motif**
- hypothalamus
- basic interoceptive monitoring

**Role**
- current viability estimation

**What it should compute**
- `G_t`
- `B_t`
- homeostatic error
- short-term survival-relevant internal summary

**Primary inputs**
- current viability state
- recent contact statistics
- boundary condition
- recent action cost

**Primary outputs**
- `module_aux.viability_state`
- `module_aux.homeostatic_error`

**Why this role exists**
- ERIE needs an explicit internal estimate of "how bad things are now"

**What it should not do**
- long-horizon planning
- raw action arbitration

### TRM-Vr

**Brain motif**
- hypothalamus + allostatic forecasting

**Role**
- future viability risk estimation

**What it should compute**
- death risk
- margin to failure
- near-horizon viability drift

**Primary outputs**
- `module_aux.death_risk`
- `module_aux.margin_to_failure`

**Why this role exists**
- separate "current condition" from "predicted risk"

## 5.2 Basal-Ganglia Family

### TRM-Ag

**Brain motif**
- basal ganglia

**Role**
- action gating and suppression

**What it should compute**
- release or suppression of action candidates
- final biasing among already-scored actions

**Primary inputs**
- action scores
- salience
- conflict
- viability urgency

**Primary outputs**
- `module_aux.gated_policy`
- `module_aux.selected_action`

**Why this role exists**
- keep `action scoring` separate from `final action release`

**What it should not do**
- act as a generic policy network

### TRM-As

**Brain motif**
- cortico-striatal action valuation

**Role**
- score candidate actions

**What it should compute**
- action values under
  - risk
  - ambiguity
  - epistemic gain

**Primary outputs**
- `module_aux.policy_logits`
- `module_aux.score_breakdown`

**Why this role exists**
- this is the main action-evaluation TRM before gating

## 5.3 Hippocampal / Entorhinal Family

### TRM-Mc

**Brain motif**
- hippocampus / entorhinal cortex

**Role**
- contextual memory

**What it should compute**
- recent trajectory context
- re-usable state summaries
- sequence-conditioned bias

**Primary inputs**
- recent beliefs
- recent actions
- local environment signature

**Primary outputs**
- `module_aux.retrieved_context`
- `module_aux.sequence_bias`

**Why this role exists**
- many action choices only make sense relative to what just happened

### TRM-Ms

**Brain motif**
- hippocampal sequence continuation

**Role**
- latent sequence stabilization

**What it should compute**
- next-context tendency
- continuity pressure across noisy or partial observation

**Why this role exists**
- separate memory retrieval from sequence continuation

## 5.4 Thalamic Family

### TRM-Xc

**Brain motif**
- thalamic relay / workspace candidate routing

**Role**
- candidate compression for workspace access

**What it should compute**
- compressed candidate state for global availability
- routing-oriented summary

**Primary outputs**
- `module_aux.workspace_candidate`
- `module_aux.priority_features`

**Why this role exists**
- GNW should not emerge from undifferentiated feature concatenation

### TRM-Xi

**Brain motif**
- thalamic gating plus ignition-like nonlinear access

**Role**
- ignition support

**What it should compute**
- whether a candidate should cross a workspace threshold

**Primary outputs**
- `module_aux.ignition_score`

**Why this role exists**
- keep workspace candidacy separate from ignition

## 5.5 Insular Family

### TRM-Si

**Brain motif**
- insula

**Role**
- interoceptive salience and embodied uncertainty

**What it should compute**
- body-relevant salience
- confidence distortion under internal stress

**Primary outputs**
- `module_aux.interoceptive_salience`
- `module_aux.embodied_uncertainty`

**Why this role exists**
- stress should shape how the world matters, not only how viable the agent is

## 5.6 Amygdalar Family

### TRM-Sa

**Brain motif**
- amygdala

**Role**
- threat / urgency salience

**What it should compute**
- hazard urgency
- fast threat weight
- action-independent stress relevance

**Primary outputs**
- `module_aux.salience_map`
- `module_aux.urgency`

**Why this role exists**
- separate generic salience from explicit threat weighting

## 5.7 Cingulate Family

### TRM-Sc

**Brain motif**
- anterior cingulate cortex

**Role**
- conflict monitoring

**What it should compute**
- score disagreement
- viability tension
- action conflict

**Primary outputs**
- `module_aux.conflict_score`
- `module_aux.control_demand`

**Why this role exists**
- ERIE needs a role that says "the current situation is contested"

## 5.8 Cerebellar Family

### TRM-Ct

**Brain motif**
- cerebellum

**Role**
- timing and corrective calibration

**What it should compute**
- fast residual correction
- smoothness and timing adjustment
- error-to-correction gain calibration

**Primary outputs**
- `module_aux.correction_delta`
- `module_aux.timing_gain`

**Why this role exists**
- not all control should go through slow deliberative selection

## 5.9 Prefrontal Family

### TRM-Pc

**Brain motif**
- prefrontal cortex

**Role**
- policy context and task-set maintenance

**What it should compute**
- which regime or behavioral mode should dominate now
- longer-horizon task-set persistence

**Primary outputs**
- `module_aux.control_context`
- `module_aux.policy_mode_bias`

**Why this role exists**
- separate short-horizon action choice from persistent control mode

## 5.10 Boundary Family

### TRM-Bd

**Brain motif**
- engineered blanket rather than one literal brain region

**Role**
- boundary detection

**What it should compute**
- self / non-self boundary estimate

**Primary outputs**
- `module_aux.boundary_map`
- `module_aux.permeability_seed`

### TRM-Bp

**Brain motif**
- thalamic gating + membrane control analogy

**Role**
- permeability control

**What it should compute**
- opening / closing / shaping of interface

**Primary outputs**
- `module_aux.permeability_map`
- `module_aux.interface_gain`

### TRM-Br

**Brain motif**
- repair-oriented homeostatic correction

**Role**
- boundary repair

**What it should compute**
- local repair pressure
- seal / reconfigure support

## 5.11 World Family

### TRM-Wp

**Brain motif**
- distributed predictive cortex

**Role**
- world prediction

### TRM-Wu

**Brain motif**
- uncertainty calibration

**Role**
- confidence auditing

## 6. Initial Library Recommendation

The first library SHOULD not try to instantiate every family.

The first buildable set should be:

- `TRM-Wp`
- `TRM-Bd`
- `TRM-Vm`
- `TRM-As`
- `TRM-Ag`
- `TRM-Mc`
- `TRM-Sa`
- `TRM-Sc`
- `TRM-Xc`

This gives:

- world prediction
- boundary estimation
- viability monitoring
- action scoring
- action gating
- memory context
- threat salience
- conflict monitoring
- workspace candidacy

without forcing full ignition or full GNW implementation.

## 7. First 12-Module Plan

The most realistic initial library is:

- `TRM-Wp::stable::baseline`
- `TRM-Wp::vent_edge::baseline`
- `TRM-Bd::baseline`
- `TRM-Bd::fragile_boundary`
- `TRM-Vm::baseline`
- `TRM-Vm::stress_heavy`
- `TRM-As::baseline`
- `TRM-As::defensive`
- `TRM-Ag::baseline`
- `TRM-Mc::baseline`
- `TRM-Sa::baseline`
- `TRM-Sc::baseline`

This is enough to begin:

- role separation
- regime specialization
- recurrent composition experiments

without exploding maintenance cost.

## 8. Why 100-200 TRMs Can Make Sense

It only makes sense if:

- they share a canonical logging substrate
- they share a module contract
- they carry metadata
- they are selected, not all run at once

The library should therefore grow by:

1. role diversity
2. regime diversity
3. objective diversity

and not by arbitrary duplication.

## 9. Metadata Requirements

Each TRM SHOULD carry:

- `module_role`
- `brain_motif`
- `regime_specialization`
- `objective_variant`
- `training_dataset_id`
- `acceptance_report`
- `known_failure_modes`

Without this metadata, a large TRM bank becomes unusable.

## 10. Composition Strategy

The long-term composition strategy is:

1. pretrain many narrow TRMs
2. evaluate each independently
3. select a small set as primaries
4. optionally run others as assistive modules
5. later connect selected modules through recurrent routing
6. only then construct GNW-like workspace behavior

This means:

- library first
- recurrent workspace later

## 11. Relationship to Existing Specifications

This document complements:

- [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md)
- [TRM_REQUIREMENTS_MUST_SHOULD.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_REQUIREMENTS_MUST_SHOULD.md)
- [IDEAL_DATA_CRITERIA.md](/Users/yamaguchimitsuyuki/criticism_bot/IDEAL_DATA_CRITERIA.md)
- [EXTERNAL_STATE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/EXTERNAL_STATE_SPEC.md)

`TRM_GRANULAR_ROLE_SPEC.md` defines the role taxonomy.

This document defines how those roles should be organized into a brain-inspired
module library and which ones should be built first.
