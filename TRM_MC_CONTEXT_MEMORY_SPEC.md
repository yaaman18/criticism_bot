# TRM-Mc Context Memory Specification

## 1. Purpose

This document defines `TRM-Mc`, the contextual-memory TRM for ERIE.

The design goal is not to reproduce human memory in full detail. The goal is to
approximate an enactivist notion of memory:

- memory is not a database
- memory is not exact replay
- memory is not detached from the body
- memory is a context-dependent residual effect of organism-environment coupling

`TRM-Mc` should therefore be implemented as a compact, decaying, action-relevant
context state that biases future inference and control.

## 2. Core Claim

For this project, contextual memory SHOULD be treated as:

> a compressed, state-dependent residue of recent bodily regulation, environmental
> contact, and action history that conditions future action selection and boundary
> control.

This means `TRM-Mc` is closer to:

- hippocampal / entorhinal contextual indexing
- short-horizon sequence conditioning
- embodied recall bias

than to:

- symbolic episodic storage
- searchable long-term database
- exact trajectory archive

## 3. Design Principles

### MUST

- `TRM-Mc` MUST NOT be modeled as exact storage-and-retrieval of full past world
  states.
- `TRM-Mc` MUST be conditioned by bodily state and recent environment contact.
- `TRM-Mc` MUST output a compact context state that can bias other TRMs.
- `TRM-Mc` MUST privilege action relevance over descriptive completeness.

### SHOULD

- `TRM-Mc` SHOULD be implemented first as a decaying recurrent context state.
- `TRM-Mc` SHOULD remember recent regularities and failures better than neutral
  background.
- `TRM-Mc` SHOULD support state-dependent recall rather than global retrieval.
- `TRM-Mc` SHOULD remain low-dimensional relative to image-like TRMs.

### MUST NOT

- It MUST NOT receive the entire canonical log as raw input by default.
- It MUST NOT be judged by reconstruction of the full past trajectory.
- It MUST NOT become a hidden generic policy network.

## 4. What Context Memory Is Approximate To

`TRM-Mc` approximates three coupled traces.

### 4.1 Body Trace

Residual trace of recent bodily regulation.

Examples:

- trajectory of `G_t`
- trajectory of `B_t`
- recent aperture gain changes
- recent boundary damage and repair tendency

### 4.2 Contact Trace

Residual trace of what the organism has been exposed to.

Examples:

- recent `energy / thermal / toxicity / niche / flow` contact
- recent multispecies contact
- recent uncertainty-heavy exposure
- recent stress accumulation

### 4.3 Procedural Bias Trace

Residual effect of recent action-outcome contingencies.

Examples:

- `withdraw -> lower stress`
- `intake -> short-term gain but boundary cost`
- `reconfigure -> later survival improvement`

This trace is not a stored rule table. It is a bias in current control space.

## 5. Computational Role

`TRM-Mc` should answer:

- what kind of situation have I been in recently?
- what action tendencies have recently helped or harmed?
- what hidden context should modulate current scoring and boundary control?

It should NOT answer:

- what exactly happened at time `t-37`?
- what was the full external state five episodes ago?

## 6. State Variables

The minimal internal variables are:

- `context_state_t`
- `body_trace_t`
- `contact_trace_t`
- `action_trace_t`
- `stress_trace_t`

Recommended interpretation:

- `context_state_t`
  - compact latent summary used by downstream TRMs
- `body_trace_t`
  - recent viability and boundary regulation history
- `contact_trace_t`
  - recent environment/species exposure history
- `action_trace_t`
  - recent action sequence residue
- `stress_trace_t`
  - recency-weighted damage / risk tendency

In the first implementation these do not need to be stored separately as public
variables; they may be implicit factors in the recurrent latent. The distinction
is conceptual and should guide the design.

## 7. Minimal Update Rule

The minimal approximation SHOULD be a gated decaying recurrent update:

```text
context_state_t =
  decay * context_state_{t-1}
  + encode(
      viability_summary_t,
      contact_summary_t,
      action_summary_t,
      boundary_summary_t,
      error_summary_t
    )
```

Recommended refinement:

```text
context_state_t =
  decay * context_state_{t-1}
  + salience_gate_t * encoded_input_t
```

where:

- `decay` expresses fading memory
- `encoded_input_t` is current embodied/environmental context
- `salience_gate_t` increases retention for high-risk or high-uncertainty events

This gives a better approximation to remembered relevance than uniform averaging.

## 8. State-Dependent Recall

Recall SHOULD NOT be modeled as exact lookup.

Instead, recall is:

```text
retrieval_bias_t = readout(current_summary_t, context_state_t)
```

This means the same latent memory may bias action differently depending on the
current bodily/environmental state.

That is closer to:

- context reinstatement
- mood/body dependent recall
- partial cue-driven reconstruction

than to direct retrieval by key.

## 9. Downstream Effects

`TRM-Mc` is useful only if it affects other modules.

In the first implementation it SHOULD feed:

- `TRM-As`
  - as `sequence_bias` or `context_bias` on action logits
- `TRM-Bp`
  - as `boundary_control_bias`
- future `TRM-Ag`
  - as gating prior
- future `TRM-Xc / TRM-Xi`
  - as candidate contextual content

The first target is therefore not memory quality in isolation, but improved
action and boundary calibration under ambiguous or delayed-cost regimes.

## 10. Input View

`TRM-Mc` SHOULD use history summaries, not raw full-frame logs.

The initial input view SHOULD be built from the last `k` steps of:

- viability summary
  - `G_t`
  - `B_t`
  - homeostatic error
- action summary
  - action one-hot
  - action cost
- environment contact summary
  - `contact_energy`
  - `contact_thermal`
  - `contact_toxicity`
  - `contact_niche`
  - `contact_flow`
- multispecies contact summary
  - species-specific contact channels
- interface summary
  - aperture gain
  - aperture width
  - boundary/permeability summary
- uncertainty / error summary
  - world uncertainty
  - boundary uncertainty
  - VFE or prediction-error summary

This should be stored as a short temporal window, not a large image stack.

## 11. Output Contract

`TRM-Mc` should emit at minimum:

- `module_state`
  - contextual latent
- `module_precision`
  - confidence in contextual bias
- `module_aux.retrieved_context`
  - low-dimensional readout of current context
- `module_aux.sequence_bias`
  - action bias vector or control bias vector
- `module_aux.context_risk_bias`
  - optional stress-weighted risk modifier

Recommended first output shapes:

- `context_state`: `[dim_c]`
- `sequence_bias`: `[num_actions]`
- optional `boundary_control_bias`: `[3]`

## 12. Training Targets

`TRM-Mc` should not be trained to reconstruct raw history.

Instead it SHOULD be trained against downstream-useful targets such as:

- next-step action improvement bias
- reduced homeostatic error under delayed-cost conditions
- improved ranking among action candidates in history-dependent regimes
- improved boundary-control choice in repeated stress contexts

Minimal bootstrap targets may include:

- whether recent action pattern led to:
  - lower stress
  - lower homeostatic error
  - better boundary preservation
- latent next-step context embedding from analytic summaries

## 13. Acceptance Criteria

`TRM-Mc` is useful only if it changes behavior under context-sensitive regimes.

### Minimum Acceptance

- improves `TRM-As` or `TRM-Bp` in at least one history-dependent family
- does not collapse to zero-context behavior
- does not simply memorize fixed action frequency

### Strong Acceptance

- improves delayed-cost handling
- reduces repeated high-stress exploitation
- improves `reconfigure / withdraw` timing in ambiguous or drifting settings
- improves multi-step homeostatic stability relative to no-memory baseline

### Rejection Signs

- behaves like a static lookup table
- no performance difference versus feedforward summary baseline
- simply copies recent action without context-sensitive modulation

## 14. Anti-Patterns

The following are design errors.

- storing full external-state trajectories as memory
- exact retrieval of old states by index
- evaluating memory by archive fidelity alone
- treating memory as a long-term database before contextual bias is validated
- allowing `TRM-Mc` to become an opaque second action scorer

## 15. Implementation Phases

### Phase 1: Recurrent Context Summary

Implement:

- fixed-length recent summary window
- recurrent or exponential-trace encoder
- output bias to `TRM-As`

This is the preferred first step.

### Phase 2: Boundary-Coupled Context

Add:

- boundary-control bias output
- interface-sensitive trace terms

### Phase 3: Retrieval-Like State Dependence

Add:

- cue-dependent recall readout
- stronger context-conditioned modulation of action/boundary control

### Phase 4: Workspace Coupling

Use `TRM-Mc` outputs as candidates for future `TRM-Xc/Xi` routing.

## 16. Recommended Initial Design

The first `TRM-Mc` implementation SHOULD therefore be:

- low-dimensional
- recurrent
- decay-based
- salience-gated
- fed by recent summary traces
- used to bias `TRM-As` first

This is the most faithful approximation to embodied contextual memory that can
be implemented under the current `~7M` TRM constraint.

## 17. Harness And Test Design

For the concrete dataset-harness and testing rollout plan, see
[TRM_MC_HARNESS_AND_TEST_PLAN.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_MC_HARNESS_AND_TEST_PLAN.md).
