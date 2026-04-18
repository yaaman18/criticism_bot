# TRM-Ag Tuning Harness Specification

## 1. Overview

This specification maps
[TRM_AG_TUNING_REQUIREMENTS.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_AG_TUNING_REQUIREMENTS.md)
to concrete implementation in `trm_pipeline.experiment_harness`.

The tuner is implemented as a new command:

```bash
./.venv/bin/python -m trm_pipeline.experiment_harness tune --contract <path>
```

It executes iterative rounds:

```text
round contract -> run_contract -> eval report -> bounded revise -> next round
```

## 2. Artifact Layout

For a base contract under `<output_root>`, tuning artifacts are written to:

```text
<output_root>/autotune/
  tune_summary.json
  round_01/
    contract.json
    doctor_report.json
    compare/
    eval_report.json
    promotion_decision.json
    run_summary.json
  round_02/
    ...
```

`run_contract` remains the artifact producer for each round.

## 3. Primary Score

A scalar primary score is used for monotonic-improvement checks:

- source: `eval_report.family_reports[*].summary.candidate.mean_final_homeostatic_error`
- scope: required family tracks only
- aggregation: mean of finite values (lower is better)

If no finite value is available, score is treated as `inf`.

## 4. Stop Conditions

The tuning loop stops when any condition is met:

1. promotion passed (`eval_report.overall_pass == true`) -> `status=promote`
2. max rounds reached -> `status=max_rounds`
3. no safe updates available -> `status=stalled`
4. no meaningful improvement within patience -> `status=no_progress`

## 5. Revision Policy

Revision proposals are derived from failed criteria on blocked required tracks.

Initial mapping:

- `stress_exploit_rate` fail:
  - decrease `aperture_gain`
  - increase `action_gating_blend`
- `stress_defensive_rate` fail:
  - decrease `aperture_width_deg`
  - increase `action_gating_blend`
- `dead_fraction` fail:
  - decrease `move_step`
- `mean_final_homeostatic_error` / `mean_mean_homeostatic_error` fail:
  - increase `lookahead_horizon`
  - increase `lookahead_discount`

Only a small top subset is applied each round (`max_updates_per_round`).

## 6. Parameter Bounds

Each tunable parameter has explicit clamps:

- `aperture_gain`: `[0.15, 0.80]`
- `aperture_width_deg`: `[40.0, 110.0]`
- `action_gating_blend`: `[0.10, 0.90]`
- `move_step`: `[1.0, 3.0]`
- `lookahead_horizon`: `[1, 4]`
- `lookahead_discount`: `[0.70, 0.98]`

Default values are used when a track does not already define the parameter.

## 7. Track Targeting

If family tracks exist, updates are applied only to blocked required tracks via
`track.runtime_overrides`.

If no family tracks are defined (global contract), updates are applied to
`contract.runtime`.

## 8. CLI Additions

Add `tune` subcommand options:

- `--contract` (required)
- `--max-rounds` (default: `3`)
- `--min-primary-improvement` (default: `0.005`)
- `--stagnation-patience` (default: `1`)
- `--max-updates-per-round` (default: `3`)
- `--force`
- `--skip-doctor`

Outputs a final path to `tune_summary.json`.

## 9. Testing Plan

Required tests:

1. bounded clamp behavior for runtime updates
2. round loop stop-on-promotion
3. no-regression for existing `experiment_harness` tests

Tests must not require full simulation; use monkeypatching on `run_contract`.
