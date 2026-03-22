# CLAUDE TRM I/O Spec

## Purpose

This document defines the minimum implementation contract for Claude to build:

- `TRM-A`: next-state prediction for Lenia dynamics
- `TRM-B`: Markov blanket management on top of TRM-A outputs

The goal is not to implement the full ERIE engine yet.
The goal is to make `TRM-A` trainable first, then make `TRM-B` consume its outputs.

---

## Global Assumptions

- Source system: Lenia-derived 5-channel grids
- Grid size: `64 x 64`
- Channels: `5`
- Tokenization: `8 x 8` patches
- Patch count: `64`
- Patch feature size: `8 x 8 x 5 = 320`
- Embedding dimension: `D = 256`
- TRM recursion: `n = 6`
- Deep supervision: `T = 3`
- Max supervision steps: `Nsup = 16`
- ACT halt threshold initial value: `0.7`

Channel semantics are fixed:

- `ch0 = membrane`
- `ch1 = cytoplasm`
- `ch2 = nucleus`
- `ch3 = DNA` (fixed condition channel)
- `ch4 = RNA`

---

## Non-Goals

Do not implement these yet:

- GNW integration
- TRM-C
- TRM-D
- full ERIE runtime loop
- end-to-end joint training of all modules

---

## Dataset Split Rules

Split by `seed_id`, not by frame.
Do not mix frames from the same seed across train/val/test.

Recommended first split:

- train: 70%
- val: 15%
- test: 15%

---

## Lenia Recording Spec For TRM-A

### Episode generation

For each Lenia seed:

1. load one seed and its Lenia parameters
2. run warmup for `32` steps
3. record the next `256` steps
4. save each state as a `64 x 64 x 5` float tensor
5. normalize all dynamic channels to `[0, 1]`
6. keep `DNA` channel fixed for the whole episode

### Training pair construction

For each recorded episode, construct:

- input: `S_t`
- target: `S_(t+1)`

This makes TRM-A a one-step prediction model.

### Optional perturbation policy

Use mostly passive sequences first.

- 70% to 80%: no perturbation
- 20% to 30%: weak perturbation

Weak perturbation means small local or global noise only.
Do not inject action-selection signals yet.

---

## TRM-A Input / Output Contract

### Input

Model input per sample:

- `state_t`: float tensor of shape `[64, 64, 5]`

After patchification:

- `patches_t`: float tensor of shape `[64, 320]`

After embedding:

- `x`: float tensor of shape `[64, 256]`

### Output

TRM-A must output:

- `pred_state_t1`: float tensor of shape `[64, 64, 5]`
- `pred_patches_t1`: float tensor of shape `[64, 320]`
- `halt_prob`: float tensor of shape `[Nsup]` or equivalent ACT-compatible form
- `aux_metrics`: dict with at least:
  - `loss_acc`
  - `loss_complex`
  - `loss_halt`
  - `nmse`

### Core architecture constraints

- input must be patchified, not flattened to `20480`
- use a shared TRM-style recursive core
- use position embeddings for 64 patches
- keep the external module interface token-based: `[tokens, dim]`

### TRM-A loss

Use:

- `L_acc = MSE(pred_state_t1, true_state_t1)`
- `L_complex = lambda * smoothness_penalty(pred_state_t1)`
- `L_halt = BCE(halt_prob, halt_target)`

Total:

- `L_total = L_acc + L_complex + L_halt`

Initial recommendation:

- `lambda = 0.01`

For the first implementation, `smoothness_penalty` can be a spatial-temporal smoothness proxy, not full KL.

---

## TRM-A Completion Criteria

Claude should treat TRM-A as complete only if all conditions are satisfied on validation data:

- `val_nmse <= 0.02`
- improvement over persistence baseline `>= 35%`
- `8-step rollout nmse <= 0.05`
- mean recursion depth is between `2.5` and `5.5`

Persistence baseline means:

- baseline prediction: `S_(t+1) = S_t`

Failure conditions:

- good one-step loss but unstable rollout
- halt always stops at first step
- halt always uses max depth
- train improves but val stagnates

---

## TRM-B Design Decision

TRM-B must not be an exact copy of TRM-A's task.
It should reuse the same backbone style, but use different inputs, heads, and losses.

Rule:

- same backbone family
- different task head

This means:

- same patch/token interface
- same `D = 256`
- same recursive TRM core style
- different target outputs

---

## TRM-B Input / Output Contract

### Inputs

TRM-B input per sample must include:

- `state_t`: `[64, 64, 5]`
- `delta_state_t`: `[64, 64, 5]`, where `delta_state_t = S_t - S_(t-1)`
- `error_map_t`: `[64, 64, 5]` or reduced boundary-focused map derived from TRM-A

Minimum required source from TRM-A:

- `pred_state_t`
- `error_map_t = abs(pred_state_t - true_state_t)`

### Outputs

TRM-B must output:

- `boundary_map`: float tensor `[64, 64, 1]`
- `permeability_map`: float tensor `[64, 64, 1]`
- `boundary_state`: float tensor `[64]` or `[64, 256]` depending on implementation
- `halt_prob`: ACT-compatible halt output
- `aux_metrics`: dict with at least:
  - `loss_boundary`
  - `loss_temporal`
  - `loss_separation`
  - `loss_halt`

### Interpretation

- `boundary_map`: probability that a cell belongs to the self/world boundary
- `permeability_map`: degree of openness of the boundary
- `boundary_state`: compressed token-level representation for later GNW integration

---

## TRM-B Target Construction

Because Lenia has no native Markov blanket label, use derived pseudo-labels first.

Initial pseudo-label policy:

- define high-gradient membrane-like regions as candidate boundaries
- enforce temporal consistency across adjacent frames
- use inside/outside separation heuristics based on channel activity and connectivity

Claude may implement this as a deterministic preprocessing stage.
Do not depend on manual labels for the first version.

---

## TRM-B Loss

Use:

- `L_boundary`: BCE or Dice/BCE hybrid on `boundary_map`
- `L_temporal`: temporal consistency penalty between `boundary_map_t` and `boundary_map_(t+1)`
- `L_separation`: penalty when inside/outside statistics collapse
- `L_halt`: ACT halt loss

Total:

- `L_total_B = L_boundary + beta * L_temporal + gamma * L_separation + L_halt`

Suggested initial values:

- `beta = 0.1`
- `gamma = 0.1`

---

## TRM-B Completion Criteria

TRM-B is ready for later integration only if:

- boundary maps are temporally stable
- boundary occupancy does not collapse to all-zero or all-one
- inside/outside summary statistics remain separable
- TRM-B improves boundary stability over a naive gradient-threshold baseline

---

## File-Level Implementation Expectation

Claude may choose different filenames, but the repo should end up with equivalent responsibilities:

- dataset generation for Lenia rollouts
- TRM-A dataset loader
- TRM-A model
- TRM-A trainer and evaluator
- TRM-B dataset builder using TRM-A outputs
- TRM-B model
- TRM-B trainer and evaluator

---

## Minimal JSON-Like Sample Schema

### TRM-A sample

```json
{
  "seed_id": "seed_000123",
  "episode_id": "seed_000123_ep_00",
  "t": 57,
  "state_t_shape": [64, 64, 5],
  "state_t1_shape": [64, 64, 5],
  "lenia_params": {
    "R": 12,
    "T": 10,
    "b": "1,1/2,1/4",
    "m": 0.25,
    "s": 0.034,
    "kn": 2,
    "gn": 2
  }
}
```

### TRM-B sample

```json
{
  "seed_id": "seed_000123",
  "episode_id": "seed_000123_ep_00",
  "t": 57,
  "state_t_shape": [64, 64, 5],
  "delta_state_t_shape": [64, 64, 5],
  "error_map_t_shape": [64, 64, 5],
  "boundary_target_shape": [64, 64, 1]
}
```

---

## Final Instruction To Claude

Implement in this order:

1. Lenia rollout recorder
2. TRM-A dataset and trainer
3. TRM-A evaluation with rollout metrics
4. TRM-B dataset builder using TRM-A outputs
5. TRM-B trainer and evaluator

Do not implement GNW integration until TRM-A and TRM-B satisfy their completion criteria.
