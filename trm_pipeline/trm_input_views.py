from __future__ import annotations

import numpy as np


def extract_centered_patch(
    field: np.ndarray,
    center_y: float,
    center_x: float,
    patch_size: int,
) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    if field.ndim == 2:
        field = field[..., None]
    if field.ndim != 3:
        raise ValueError(f"expected 2D or 3D field, got shape={field.shape}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    half = patch_size // 2
    y_center = int(round(float(center_y)))
    x_center = int(round(float(center_x)))
    y0 = y_center - half
    x0 = x_center - half
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - field.shape[0])
    pad_right = max(0, x1 - field.shape[1])
    if pad_top or pad_left or pad_bottom or pad_right:
        field = np.pad(
            field,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    y0 += pad_top
    y1 += pad_top
    x0 += pad_left
    x1 += pad_left
    return field[y0:y1, x0:x1, :].astype(np.float32)


def build_trm_wp_input_view(
    observation: np.ndarray,
    sensor_gate: np.ndarray,
    species_fields: np.ndarray,
    flow_channels: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            observation.astype(np.float32),
            sensor_gate.astype(np.float32),
            species_fields.astype(np.float32),
            flow_channels.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def build_trm_bd_input_view(
    observation: np.ndarray,
    world_error: np.ndarray,
    sensor_gate: np.ndarray,
    delta_observation: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            observation.astype(np.float32),
            world_error.astype(np.float32),
            sensor_gate.astype(np.float32),
            delta_observation.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def build_trm_bp_input_view(
    boundary_patch: np.ndarray,
    permeability_patch: np.ndarray,
    observation_patch: np.ndarray,
    species_patch: np.ndarray,
    flow_patch: np.ndarray,
    viability_state: np.ndarray,
) -> np.ndarray:
    patch_shape = observation_patch.shape[:2]
    viability_broadcast = np.broadcast_to(
        viability_state.astype(np.float32).reshape(1, 1, -1),
        (*patch_shape, int(viability_state.shape[-1])),
    )
    return np.concatenate(
        [
            boundary_patch.astype(np.float32),
            permeability_patch.astype(np.float32),
            observation_patch.astype(np.float32),
            species_patch.astype(np.float32),
            flow_patch.astype(np.float32),
            viability_broadcast.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def build_trm_vm_input_view(
    viability_state: np.ndarray,
    env_contact_state: np.ndarray,
    species_contact_state: np.ndarray,
    action_cost: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            viability_state.astype(np.float32),
            env_contact_state.astype(np.float32),
            species_contact_state.astype(np.float32),
            action_cost.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def build_trm_as_input_view(
    viability_state: np.ndarray,
    action_scores: np.ndarray,
    uncertainty_state: np.ndarray,
    env_contact_state: np.ndarray,
    species_contact_state: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            viability_state.astype(np.float32),
            action_scores.astype(np.float32),
            uncertainty_state.astype(np.float32),
            env_contact_state.astype(np.float32),
            species_contact_state.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def build_trm_ag_input_view(
    as_policy_logits: np.ndarray,
    viability_state: np.ndarray,
    homeostatic_error_vector: np.ndarray,
    viability_risk: np.ndarray,
    uncertainty_state: np.ndarray,
    env_contact_state: np.ndarray,
    species_contact_state: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            as_policy_logits.astype(np.float32),
            viability_state.astype(np.float32),
            homeostatic_error_vector.astype(np.float32),
            viability_risk.astype(np.float32),
            uncertainty_state.astype(np.float32),
            env_contact_state.astype(np.float32),
            species_contact_state.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def build_temporal_context_windows(
    step_features: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    step_features = np.asarray(step_features, dtype=np.float32)
    if step_features.ndim != 2:
        raise ValueError(f"expected [T, F] step_features, got shape={step_features.shape}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    time_steps, feature_dim = step_features.shape
    windows = np.zeros((time_steps, window_size, feature_dim), dtype=np.float32)
    masks = np.zeros((time_steps, window_size), dtype=np.float32)
    for t in range(time_steps):
        start = max(0, t - window_size + 1)
        segment = step_features[start : t + 1]
        seg_len = int(segment.shape[0])
        windows[t, window_size - seg_len :, :] = segment
        masks[t, window_size - seg_len :] = 1.0
    return windows.astype(np.float32), masks.astype(np.float32)


def build_trm_mc_input_view(
    step_features: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    step_features = np.asarray(step_features, dtype=np.float32)
    if step_features.ndim != 2:
        raise ValueError(f"expected [T, F] step_features, got shape={step_features.shape}")
    feature_dim = int(step_features.shape[-1])
    if feature_dim < 30:
        raise ValueError(f"expected at least 30 TRM-Mc base features, got {feature_dim}")
    # Base layout:
    # 0:2 viability, 2:4 homeostatic error, 4:8 env contact,
    # 8:12 species contact, 12:16 uncertainty, 16:18 flow,
    # 18:22 interface, 22:23 interface mass, 23:29 action one-hot, 29:30 action cost
    delta_indices = np.concatenate(
        [
            np.arange(4, 8, dtype=np.int64),
            np.arange(8, 12, dtype=np.int64),
            np.arange(12, 16, dtype=np.int64),
            np.arange(16, 18, dtype=np.int64),
        ]
    )
    delta_features = np.zeros((step_features.shape[0], delta_indices.shape[0]), dtype=np.float32)
    if step_features.shape[0] > 1:
        delta_features[1:] = step_features[1:, delta_indices] - step_features[:-1, delta_indices]
    enriched_features = np.concatenate([step_features, delta_features], axis=-1).astype(np.float32)
    return build_temporal_context_windows(enriched_features, window_size)
