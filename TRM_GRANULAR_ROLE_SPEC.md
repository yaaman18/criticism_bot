# TRM Granular Role Specification

## 1. Purpose

This document defines the fine-grained TRM role taxonomy for ERIE.

The goal is to avoid assigning multiple incompatible responsibilities to a single
TRM under the `~7M params` constraint. Each TRM should therefore be defined as
one computational responsibility, one primary output family, and one primary
training/evaluation objective.

This taxonomy is broader than the currently implemented set. It is a design
document for future extension, not a statement that every TRM below must be
implemented immediately.

## 2. Design Rules

### MUST

- A single TRM MUST have one dominant computational responsibility.
- A single TRM MUST expose one primary output family through the common module
  contract.
- A single TRM MUST be evaluable independently from the full ERIE stack.
- A single TRM MUST be small enough to remain plausible under the `~7M params`
  budget when implemented as an individual module.
- TRM names MUST not be treated as brain-region copies. They are engineering
  roles inspired by systems neuroscience.

### SHOULD

- Each TRM SHOULD have one main loss family and one main runtime purpose.
- Each TRM SHOULD be replaceable without forcing redesign of unrelated TRMs.
- Each TRM SHOULD map to the shared module contract:
  - `module_state`
  - `module_precision`
  - `module_error`
  - `module_aux`
  - `module_role`

## 3. Naming Scheme

The following convention is adopted:

- Prefix:
  - `W`: world
  - `B`: boundary
  - `V`: viability
  - `A`: action
  - `M`: memory
  - `S`: salience
  - `X`: workspace
- Suffix:
  - `p`: prediction
  - `d`: detection
  - `c`: control / candidate / context depending on role family
  - `m`: monitoring
  - `r`: risk
  - `s`: scoring
  - `g`: gating
  - `i`: ignition

The suffix is not meant to be globally semantic across all families. The full
TRM identifier is the authoritative unit.

## 4. Fine-Grained TRM Taxonomy

### 4.1 World Family

#### TRM-Wp
World Prediction

- Core responsibility:
  predict near-future external / environmental state.
- Primary inspiration:
  predictive coding, state transition modeling.
- Typical input:
  current sensed world state, optional previous world belief.
- Primary outputs:
  - `module_state`: predicted world latent / token state
  - `module_precision`: likelihood precision proxy
  - `module_aux.pred_state_t1`
  - `module_aux.pred_logvar_t1`
- Primary training mode:
  supervised or self-supervised next-state prediction.
- Runtime function:
  generate world prior for belief update.
- Notes:
  this is the fine-grained successor of current `TRM-A`.

#### TRM-Wu
World Uncertainty Calibration

- Core responsibility:
  calibrate and audit uncertainty quality for world prediction.
- Primary inspiration:
  predictive uncertainty calibration.
- Typical input:
  world belief, prediction error statistics, world logvar.
- Primary outputs:
  - `module_state`: uncertainty audit state
  - `module_precision`: confidence over confidence estimates
  - `module_aux.calibration_metrics`
- Primary training mode:
  calibration-oriented auxiliary training or runtime auditing.
- Runtime function:
  stabilize confidence estimates used by downstream weighting.
- Notes:
  not required in earliest implementation.

### 4.2 Boundary Family

#### TRM-Bd
Boundary Detection

- Core responsibility:
  estimate self / non-self boundary.
- Primary inspiration:
  engineered Markov blanket interface.
- Typical input:
  current state, temporal delta, prediction error maps.
- Primary outputs:
  - `module_state`: boundary latent
  - `module_aux.boundary_map`
  - `module_aux.permeability_seed`
- Primary training mode:
  pseudo-label boundary supervision.
- Runtime function:
  define the candidate interface between inside and outside.
- Notes:
  this is the narrowest core of current `TRM-B`.

#### TRM-Bp
Boundary Permeability Control

- Core responsibility:
  regulate what crosses the boundary.
- Primary inspiration:
  membrane permeability / sensory-action gating.
- Typical input:
  boundary state, local resource/hazard observations, viability state.
- Primary outputs:
  - `module_state`: permeability control latent
  - `module_aux.permeability_map`
  - `module_aux.interface_gain`
- Primary training mode:
  control-oriented runtime objective, possibly weak supervision.
- Runtime function:
  open / close / reshape the interface.
- Notes:
  should be separated from detection when the boundary module becomes too broad.

#### TRM-Br
Boundary Repair

- Core responsibility:
  infer and execute repair-oriented adjustments after boundary degradation.
- Typical input:
  boundary integrity signals, hazard contact, recent boundary error.
- Primary outputs:
  - `module_state`: repair latent
  - `module_aux.repair_signal`
- Runtime function:
  support `seal` and `reconfigure` under damage.
- Notes:
  postpone until boundary dynamics become richer.

### 4.3 Viability Family

#### TRM-Vm
Viability Monitoring

- Core responsibility:
  estimate current internal survival-relevant state.
- Primary inspiration:
  hypothalamic / interoceptive monitoring.
- Typical input:
  `G_t`, `B_t`, recent contact statistics, recent action cost.
- Primary outputs:
  - `module_state`: low-dimensional viability latent
  - `module_aux.viability_state`
  - `module_aux.homeostatic_error`
- Primary training mode:
  direct regression or analytic update supervision.
- Runtime function:
  maintain an explicit self-maintenance estimate.

#### TRM-Vr
Viability Risk Estimation

- Core responsibility:
  estimate near-future threat to viability.
- Primary inspiration:
  risk forecasting, allostatic error anticipation.
- Typical input:
  viability state, boundary state, projected contact, predicted action outcome.
- Primary outputs:
  - `module_state`: viability risk latent
  - `module_aux.death_risk`
  - `module_aux.margin_to_failure`
- Primary training mode:
  short-horizon failure prediction.
- Runtime function:
  feed `Risk` term of policy scoring.

### 4.4 Action Family

#### TRM-As
Action Scoring

- Core responsibility:
  score candidate actions under risk / ambiguity / epistemic terms.
- Primary inspiration:
  expected free energy approximation.
- Typical input:
  world belief, boundary belief, viability belief, projected contact.
- Primary outputs:
  - `module_state`: action evaluation latent
  - `module_aux.policy_logits`
  - `module_aux.score_breakdown`
- Primary training mode:
  runtime optimization, planning proxy, or imitation of successful closed-loop traces.
- Runtime function:
  compute candidate action values.

#### TRM-Ag
Action Gating

- Core responsibility:
  suppress, release, or bias candidate actions.
- Primary inspiration:
  basal-ganglia-like gating.
- Typical input:
  action scores, salience, conflict, viability urgency.
- Primary outputs:
  - `module_state`: action gate latent
  - `module_aux.gated_policy`
  - `module_aux.selected_action`
- Runtime function:
  final action arbitration.
- Notes:
  should remain separate from raw action scoring in later phases.

### 4.5 Memory Family

#### TRM-Mc
Memory Context

- Core responsibility:
  maintain retrievable context over longer horizons.
- Primary inspiration:
  hippocampal contextual memory.
- Typical input:
  recent belief states, recent actions, local environment signature.
- Primary outputs:
  - `module_state`: context memory latent
  - `module_aux.retrieved_context`
  - `module_aux.sequence_bias`
- Runtime function:
  bias prediction and action using history.

#### TRM-Ms
Memory Stability / Identity Support

- Core responsibility:
  stabilize longer-lived continuity markers across episodes.
- Runtime function:
  support future self-identity work.
- Notes:
  explicitly out of scope for earliest phases.

### 4.6 Salience Family

#### TRM-Sa
Salience Allocation

- Core responsibility:
  highlight what matters now.
- Primary inspiration:
  salience / attentional allocation systems.
- Typical input:
  prediction error, uncertainty, hazard/resource gradients, viability urgency.
- Primary outputs:
  - `module_state`: salience latent
  - `module_aux.salience_map`
  - `module_aux.urgency`
- Runtime function:
  bias scoring, monitoring, or future workspace entry.

#### TRM-Sc
Conflict Monitoring

- Core responsibility:
  detect competition between incompatible internal tendencies.
- Primary inspiration:
  cingulate-style conflict monitoring.
- Typical input:
  action scores, viability tension, boundary demands, memory bias.
- Primary outputs:
  - `module_state`: conflict latent
  - `module_aux.conflict_score`
- Runtime function:
  trigger gating changes or future workspace escalation.

### 4.7 Workspace Family

#### TRM-Xc
Workspace Candidate Formation

- Core responsibility:
  compress and nominate candidate content for global availability.
- Primary inspiration:
  GNW-role module, not GNW ontology.
- Typical input:
  selected module states, module precisions, salience/conflict markers.
- Primary outputs:
  - `module_state`: workspace candidate latent
  - `module_aux.workspace_candidate`
- Runtime function:
  prepare content before ignition/broadcast.

#### TRM-Xi
Ignition Support

- Core responsibility:
  estimate whether candidate content should cross the ignition threshold.
- Primary inspiration:
  non-linear workspace access.
- Typical input:
  workspace candidate, module priority, error change, precision.
- Primary outputs:
  - `module_state`: ignition latent
  - `module_aux.ignition_score`
  - `module_aux.broadcast_gate`
- Runtime function:
  help trigger broadcast in future GNW scaffolds.

## 5. Dependency Structure

### Foundational TRMs

- `TRM-Wp`
- `TRM-Bd`
- `TRM-Vm`
- `TRM-As`

These can support the earliest belief-based self-maintenance loop.

### Intermediate TRMs

- `TRM-Bp`
- `TRM-Vr`
- `TRM-Ag`
- `TRM-Mc`

These improve robustness, temporal depth, and action quality.

### Advanced TRMs

- `TRM-Wu`
- `TRM-Br`
- `TRM-Ms`
- `TRM-Sa`
- `TRM-Sc`
- `TRM-Xc`
- `TRM-Xi`

These support richer arbitration, memory continuity, and future GNW-like access.

## 6. Recommended First Four TRMs

From the full taxonomy above, the first four TRMs that should actually be
implemented are:

1. `TRM-Wp`
   - because ERIE still needs a clean world prior generator.
2. `TRM-Bd`
   - because self / non-self interface is foundational and already partially present.
3. `TRM-Vm`
   - because viability must become an explicit monitored state rather than an
     implicit side effect of runtime bookkeeping.
4. `TRM-As`
   - because action choice currently exists as a runtime score, but not yet as a
     dedicated module role.

These four are the smallest coherent set that can produce:

- world prediction
- boundary estimation
- viability awareness
- action scoring

without overloading any single TRM beyond the intended parameter budget.

## 7. Explicit Non-Recommendations for the First Step

The following SHOULD NOT be among the first four:

- `TRM-Xc`, `TRM-Xi`
  - GNW-role modules are important but should not precede the minimal
    self-maintenance loop.
- `TRM-Ms`
  - self-identity support is too early.
- `TRM-Wu`
  - useful later, but not before the base world model is stabilized.
- `TRM-Br`
  - repair should follow boundary detection and viability monitoring.

## 8. Immediate Design Consequence

The next architectural step after this document is:

- reinterpret current `TRM-A` as `TRM-Wp`
- reinterpret current `TRM-B` as the seed of `TRM-Bd`
- introduce `TRM-Vm`
- introduce `TRM-As`

Only after these are stable should ERIE move toward:

- `TRM-Bp`
- `TRM-Vr`
- `TRM-Ag`
- `TRM-Mc`
- future GNW-role TRMs

## 9. Acceptance Criteria for TRM-Vm and TRM-As

This section defines what it means for `TRM-Vm` and `TRM-As` to be considered
successful enough to justify further integration in ERIE.

The criteria are intentionally split into:

- training-level success
- runtime-level success

Low loss alone is not sufficient. Runtime utility must also be demonstrated.

### 9.1 TRM-Vm Acceptance

#### MUST

- `TRM-Vm` MUST be evaluated as a dedicated `viability_monitor` role.
- `TRM-Vm` MUST predict or refine an explicit survival-relevant state whose
  minimum initial form is `[G_t, B_t]`.
- `TRM-Vm` MUST expose at least:
  - `module_aux.viability_state`
  - `module_aux.homeostatic_error`
  - `module_aux.viability_risk`
- `TRM-Vm` MUST be judged by both state quality and runtime utility.

#### Training-Level Success for TRM-Vm

`TRM-Vm` training is successful only if all of the following hold on validation
data:

- viability-state regression is stable and clearly below an explicit analytic
  baseline.
- `homeostatic_error` ranking is directionally correct:
  states farther from target bands receive larger predicted error.
- `viability_risk` ranking is directionally correct:
  near-failure states receive larger risk than safe-band states.
- predicted risk correlates with observed short-horizon viability drop or
  failure frequency.

The following metrics SHOULD be logged:

- `val_viability_mae_G`
- `val_viability_mae_B`
- `val_homeostatic_error_mae`
- `val_viability_risk_auroc`
- `val_margin_to_failure_corr`

The initial engineering target SHOULD be:

- `val_viability_mae_G <= 0.08`
- `val_viability_mae_B <= 0.08`
- `val_homeostatic_error_mae <= 0.10`
- `val_viability_risk_auroc >= 0.75`

#### Runtime-Level Success for TRM-Vm

`TRM-Vm` runtime integration is successful only if, under otherwise matched
conditions, `assistive` is better than `analytic-only` on at least one of the
following without materially degrading the others:

- lower `mean_homeostatic_error`
- lower `final_homeostatic_error`
- fewer entries into the risk region
- fewer deaths over the same seed set
- earlier detection of viability collapse

Initial runtime acceptance for `TRM-Vm` SHOULD require:

- on a fixed seed set, `assistive` does not increase death count relative to
  `analytic-only`
- and either:
  - `mean_homeostatic_error` improves by at least `5%`
  - or viability-risk ranking / AUROC improves by at least `0.05`

#### What Does NOT Count as Success for TRM-Vm

The following MUST NOT be treated as sufficient:

- low internal loss with no runtime improvement
- smooth but uninformative viability predictions clustered near the mean
- always-high risk output
- always-low risk output
- merely matching analytic values without improving robustness or ranking

### 9.2 TRM-As Acceptance

#### MUST

- `TRM-As` MUST be evaluated as a dedicated `action_scoring` role.
- `TRM-As` MUST score the current action vocabulary and expose at least:
  - `module_aux.policy_logits`
  - `module_aux.policy_prob`
  - `module_aux.action_uncertainty` or equivalent score diagnostics
- `TRM-As` MUST be evaluated against:
  - `analytic-only`
  - `assistive`
  - later `module-primary`

#### Training-Level Success for TRM-As

`TRM-As` training is successful only if all of the following hold on validation
traces:

- it ranks actions in a way that agrees with improved short-horizon viability.
- it distinguishes clearly between obviously harmful and obviously protective
  actions.
- it does not collapse to a single action independent of state.
- action uncertainty is directionally meaningful:
  more ambiguous situations produce less peaked policy belief.

The following metrics SHOULD be logged:

- `val_top1_action_agreement`
- `val_pairwise_ranking_accuracy`
- `val_expected_homeostatic_delta`
- `val_policy_entropy_mean`
- `val_action_collapse_rate`

The initial engineering target SHOULD be:

- `val_pairwise_ranking_accuracy >= 0.65`
- `val_expected_homeostatic_delta < 0` for selected actions
- `val_action_collapse_rate <= 0.80`

`val_top1_action_agreement` may remain modest in early phases if pairwise
ranking and runtime behavior are good.

#### Runtime-Level Success for TRM-As

`TRM-As` runtime integration is successful only if, under a fixed evaluation
suite, `assistive` is better than `analytic-only` in at least one meaningful
behavioral dimension without causing degeneration elsewhere.

Preferred behavioral dimensions:

- lower `mean_homeostatic_error`
- lower `final_homeostatic_error`
- lower death count
- more environment-sensitive action switching
- greater useful action diversity without randomization collapse

Initial runtime acceptance for `TRM-As` SHOULD require:

- `assistive` does not increase deaths relative to `analytic-only`
- and at least one of:
  - `mean_homeostatic_error` improves by `>= 5%`
  - `final_homeostatic_error` improves by `>= 5%`
  - useful nontrivial actions (`approach`, `withdraw`, `reconfigure`) appear in
    seeds where `analytic-only` stayed trapped in `intake/seal`

#### What Does NOT Count as Success for TRM-As

The following MUST NOT be treated as sufficient:

- sharper policy logits without better homeostasis
- exploration that increases deaths
- always preferring one action class across environments
- merely reproducing the analytic score ordering without any runtime gain

### 9.3 Evaluation Modes

For both `TRM-Vm` and `TRM-As`, evaluation MUST distinguish at least:

- `analytic-only`
- `assistive`
- `module-primary` (future phase)

Early implementation may stop at `assistive`, but every result log SHOULD state
which mode was used.

### 9.4 Seed-Set Requirement

Claims of success for `TRM-Vm` and `TRM-As` MUST NOT rely on a single seed.

At minimum, the evaluation seed set MUST contain:

- resource-favoring cases
- hazard-dominant cases
- mixed ambiguity cases

An initial engineering target of `>= 10` seeds is acceptable for smoke
evaluation, but stronger claims SHOULD use larger sets.
