# ERIE Precision Three-Layer Memo

## Purpose

This memo isolates the `precision` concept for ERIE so it does not collapse into a single overloaded term.

## Core Separation

ERIE should treat `precision` as at least three distinct layers.

1. `Likelihood precision`
   - Meaning:
     inverse variance / inverse covariance for observation or transition noise.
   - Current ERIE status:
     approximated by `exp(-pred_logvar_t1)` from `TRM-A`.
   - What it is good for:
     weighting prediction errors under a probabilistic world model.
   - What it is not:
     not identical to Friston-style expected precision as a contextual gain controller.

2. `Inferential precision`
   - Meaning:
     a gain term that changes how strongly prediction errors update beliefs.
   - Current ERIE status:
     not yet fully implemented.
   - Minimum requirement:
     there must be an explicit update path in which precision changes error impact.
   - Failure mode:
     if removing precision does not alter update dynamics, ERIE does not yet implement precision in the active-inference sense.

3. `Policy precision`
   - Meaning:
     confidence / inverse temperature over actions or policies.
   - Current ERIE status:
     not yet implemented because the active-inference loop is not yet implemented.
   - Minimum requirement:
     policy selection sharpness must be separately parameterized from likelihood precision.

## Immediate Design Rule

- `pred_logvar_t1` should currently be called `likelihood precision proxy`, not `precision` without qualification.
- ERIE should not merge aleatoric uncertainty, epistemic uncertainty, and policy confidence into one scalar.

## Minimal Operational Tests

1. `Likelihood precision`
   - calibration check
   - NLL / proper scoring
   - residual correlation with predicted variance

2. `Inferential precision`
   - gain ablation test
   - update sensitivity comparison with precision on/off
   - module selection change under precision weighting

3. `Policy precision`
   - policy entropy change
   - action selection sharpness
   - closed-loop viability gain under policy precision control

## ERIE Implication

ERIE should only claim the following at the current stage:

- `TRM-A` provides calibrated uncertainty outputs.
- These outputs can become `likelihood precision proxies`.
- ERIE does not yet fully implement inferential precision or policy precision.

When GNW and active inference are added, `likelihood precision -> inferential precision -> policy precision` should remain explicitly separated.
