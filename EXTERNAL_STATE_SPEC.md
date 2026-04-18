# External State Specification

## 1. Purpose

This document defines the explicit external-state model for ERIE-on-Lenia.

The purpose is to separate:

- the world as it exists outside the agent
- the boundary-gated observation available to the agent
- the agent's internal belief state about that world

This specification exists so that later VFE / EFE implementations can refer to
a stable world-state object instead of an implicit collection of fields.

## 2. Core Distinction

The following layers MUST be treated as distinct:

1. `z_ext`
   The external state of the world.
2. `o_t`
   The observation available to the agent through the boundary interface.
3. `q(z)_t`
   The agent's posterior belief about the world.

`z_ext` is not the same thing as observation, and it is not the same thing as
belief.

## 3. External State Definition

For the current ERIE-on-Lenia implementation, the minimal external state is:

```text
z_ext = {
  z_lenia,
  z_energy,
  z_thermal,
  z_toxic,
  z_niche,
  z_flow
}
```

Where:

- `z_lenia`
  The underlying Lenia world state.
- `z_energy`
  The energy-gradient field.
- `z_thermal`
  The thermal-stress field.
- `z_toxic`
  The toxicity field.
- `z_niche`
  The niche-stability field.
- `z_flow`
  A 2D transport field represented by `flow_y` and `flow_x`.

In the current implementation, `z_lenia` is represented by:

- `scalar_state`
- `prev_scalar_state`
- the derived multichannel Lenia representation

## 4. Observation Mapping

The observation available to ERIE MUST be modeled as:

```text
o_t = H(z_ext, boundary_interface_t, noise_t)
```

Where:

- `H`
  is the observation mapping
- `boundary_interface_t`
  is the current sensory-active boundary configuration
- `noise_t`
  is observation noise

The observation mapping MUST preserve the distinction between:

- external state
- gated observation
- posterior belief

## 5. Action-Conditioned World Update

The world update MUST be interpreted as:

```text
z_ext(t+1) = f(z_ext(t), a_t)
```

In the current implementation, this is split across:

- Lenia intrinsic update
- environmental field update
- action-conditioned interaction through the body/interface

The current system may continue to use heuristic update rules, but those rules
MUST be understood as the initial implementation of `f`.

## 6. Logging Requirements

The runtime SHOULD preserve the following separately:

- `external_state`
- `observation`
- `world_belief`
- `boundary_belief`
- `world_error`
- `boundary_error`

At minimum, `external_state` and `observation` MUST be logged distinctly so the
codebase can later support explicit VFE accounting without redefining the data
model.

## 7. Compatibility Constraint

This specification MUST be introduced without breaking existing runtime and
training pipelines.

That means:

- existing `env_channels` may remain as a compatibility output
- legacy property access such as `energy_gradient` may remain
- `ExternalState` should initially be added as an explicit internal container
  rather than by removing old APIs

## 8. Transport Extension

The current implementation includes:

```text
z_flow = {flow_y, flow_x}
```

`z_flow` represents a bounded 2D transport field used to lightly advect
`z_energy`, `z_thermal`, `z_toxic`, and `z_niche`.

This does not yet constitute a full fluid simulation. It is a transport prior
that makes the external state more vent-like and increases the temporal depth
of action-conditioned predictions.
