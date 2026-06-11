from __future__ import annotations

import numpy as np


def ag_veto_params(control_mode: int) -> tuple[float, float, float]:
    if int(control_mode) == 2:
        return 0.45, 0.60, 8.0
    if int(control_mode) == 0:
        return 0.75, 0.90, 3.0
    return 0.60, 0.80, 5.0


def apply_ag_assistive_veto(
    pre_ag_logits: np.ndarray,
    inhibition_mask: np.ndarray,
    control_mode: int,
) -> np.ndarray:
    final_logits = pre_ag_logits.astype(np.float32).copy()
    soft_threshold, hard_threshold, prune_scale = ag_veto_params(int(control_mode))
    inhibition_excess = np.clip(inhibition_mask.astype(np.float32) - soft_threshold, 0.0, 1.0).astype(np.float32)
    if float(np.max(inhibition_excess)) > 0.0:
        final_logits = (final_logits - prune_scale * inhibition_excess).astype(np.float32)
    hard_gate_mask = inhibition_mask.astype(np.float32) >= hard_threshold
    if bool(np.any(hard_gate_mask)):
        gate_floor = float(np.min(final_logits) - 7.0)
        final_logits = np.where(hard_gate_mask, gate_floor, final_logits).astype(np.float32)
    return final_logits.astype(np.float32)

