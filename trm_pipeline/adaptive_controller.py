from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, dataclass, replace
from typing import Any

import numpy as np


@dataclass
class AdaptiveControllerConfig:
    enabled: bool = False
    interval: int = 4
    window_size: int = 8
    min_steps: int = 4
    learning_rate: float = 0.08
    target_homeostatic_error: float = 0.18
    min_policy_entropy: float = 0.75
    target_energy_contact: float = 0.18
    max_stress_contact: float = 0.35
    adapt_runtime: bool = True
    adapt_lenia: bool = True
    lenia_mu_center: float = 0.33
    lenia_sigma_center: float = 0.055
    lenia_mu_min: float = 0.23
    lenia_mu_max: float = 0.41
    lenia_sigma_min: float = 0.033
    lenia_sigma_max: float = 0.080


def _finite_mean(values: list[float], default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(arr.mean())


def _clip_scalar(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), float(low), float(high)))


class AdaptiveController:
    """Conservative live controller for runtime and Lenia dynamics.

    The controller does not change the external world. It watches recent
    viability/contact history and only nudges bounded organism-side parameters.
    """

    def __init__(self, config: AdaptiveControllerConfig | None = None) -> None:
        self.config = config or AdaptiveControllerConfig()
        self.events: list[dict[str, Any]] = []

    def maybe_update(self, runtime: Any, t: int) -> dict[str, Any] | None:
        cfg = self.config
        if not cfg.enabled:
            return None
        interval = max(1, int(cfg.interval))
        if (int(t) + 1) % interval != 0:
            return None
        history = list(getattr(runtime, "history", []))
        if len(history) < max(1, int(cfg.min_steps)):
            return None

        window = history[-max(1, int(cfg.window_size)) :]
        metrics = self._window_metrics(window)
        updates: dict[str, dict[str, float]] = {}
        if bool(cfg.adapt_runtime):
            updates.update(self._adapt_runtime(runtime, metrics))
        if bool(cfg.adapt_lenia):
            updates.update(self._adapt_lenia(runtime, metrics))
        if not updates:
            return None

        event = {
            "t": int(t),
            "window_start": int(window[0].get("t", max(0, int(t) - len(window) + 1))),
            "window_end": int(window[-1].get("t", int(t))),
            "controller_config": asdict(cfg),
            "metrics": metrics,
            "updates": updates,
        }
        self.events.append(event)
        return event

    def _window_metrics(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        stress_values = [
            0.5 * (float(row.get("contact_thermal", 0.0)) + float(row.get("contact_toxicity", 0.0)))
            for row in rows
        ]
        return {
            "mean_homeostatic_error": _finite_mean(
                [float(row.get("homeostatic_error", row.get("monitor_homeostatic_error", 0.0))) for row in rows]
            ),
            "mean_G": _finite_mean([float(row.get("G", row.get("monitor_G", 0.0))) for row in rows]),
            "mean_B": _finite_mean([float(row.get("B", row.get("monitor_B", 0.0))) for row in rows]),
            "mean_policy_entropy": _finite_mean([float(row.get("policy_entropy", 0.0)) for row in rows]),
            "mean_energy_contact": _finite_mean([float(row.get("contact_energy", 0.0)) for row in rows]),
            "mean_stress_contact": _finite_mean(stress_values),
            "dead_fraction": _finite_mean([1.0 if bool(row.get("dead", False)) else 0.0 for row in rows]),
        }

    def _record_update(
        self,
        updates: dict[str, dict[str, float]],
        name: str,
        old_value: float,
        new_value: float,
    ) -> None:
        old = float(old_value)
        new = float(new_value)
        if abs(new - old) <= 1e-9:
            return
        updates[name] = {"old": old, "new": new, "delta": float(new - old)}

    def _set_runtime_config_value(self, runtime: Any, name: str, value: float) -> float:
        run_cfg = runtime.cfg
        new_value = float(value)
        try:
            setattr(run_cfg, name, new_value)
        except FrozenInstanceError:
            runtime.cfg = replace(run_cfg, **{name: new_value})
        return new_value

    def _adapt_runtime(self, runtime: Any, metrics: dict[str, float]) -> dict[str, dict[str, float]]:
        cfg = self.config
        updates: dict[str, dict[str, float]] = {}
        lr = _clip_scalar(cfg.learning_rate, 0.0, 1.0)
        homeo_excess = max(0.0, float(metrics["mean_homeostatic_error"]) - float(cfg.target_homeostatic_error))
        stress_excess = max(0.0, float(metrics["mean_stress_contact"]) - float(cfg.max_stress_contact))
        energy_deficit = max(0.0, float(cfg.target_energy_contact) - float(metrics["mean_energy_contact"]))
        entropy_deficit = max(0.0, float(cfg.min_policy_entropy) - float(metrics["mean_policy_entropy"]))

        if homeo_excess > 0.0 or stress_excess > 0.0:
            old = float(runtime.cfg.contact_w_thermal)
            new = self._set_runtime_config_value(
                runtime,
                "contact_w_thermal",
                _clip_scalar(old + lr * (0.5 * homeo_excess + stress_excess), 0.05, 2.5),
            )
            self._record_update(updates, "runtime.contact_w_thermal", old, new)
            old = float(runtime.cfg.contact_w_toxicity)
            new = self._set_runtime_config_value(
                runtime,
                "contact_w_toxicity",
                _clip_scalar(old + lr * (0.5 * homeo_excess + stress_excess), 0.05, 2.8),
            )
            self._record_update(updates, "runtime.contact_w_toxicity", old, new)

        if energy_deficit > 0.0 or homeo_excess > 0.0:
            old = float(runtime.cfg.contact_w_energy)
            new = self._set_runtime_config_value(
                runtime,
                "contact_w_energy",
                _clip_scalar(old + lr * (energy_deficit + 0.3 * homeo_excess), 0.05, 2.0),
            )
            self._record_update(updates, "runtime.contact_w_energy", old, new)

        if entropy_deficit > 0.0:
            old = float(runtime.cfg.beta_pi)
            new = self._set_runtime_config_value(
                runtime,
                "beta_pi",
                _clip_scalar(old * (1.0 - 0.5 * lr * entropy_deficit), 0.5, 8.0),
            )
            self._record_update(updates, "runtime.beta_pi", old, new)
        elif homeo_excess > 0.0 and stress_excess <= 0.0:
            old = float(runtime.cfg.beta_pi)
            new = self._set_runtime_config_value(
                runtime,
                "beta_pi",
                _clip_scalar(old * (1.0 + 0.25 * lr * homeo_excess), 0.5, 8.0),
            )
            self._record_update(updates, "runtime.beta_pi", old, new)

        if stress_excess > 0.0:
            old = float(getattr(runtime.body, "aperture_gain", runtime.cfg.aperture_gain))
            runtime.body.aperture_gain = _clip_scalar(old * (1.0 - 0.25 * lr * stress_excess), 0.05, 1.2)
            self._set_runtime_config_value(runtime, "aperture_gain", float(runtime.body.aperture_gain))
            self._record_update(updates, "body.aperture_gain", old, runtime.body.aperture_gain)

        return updates

    def _adapt_lenia(self, runtime: Any, metrics: dict[str, float]) -> dict[str, dict[str, float]]:
        cfg = self.config
        env = runtime.env
        params = getattr(env, "params", {})
        if not isinstance(params, dict) or "m" not in params or "s" not in params:
            return {}

        pressure = max(0.0, float(metrics["mean_homeostatic_error"]) - float(cfg.target_homeostatic_error))
        pressure += max(0.0, float(metrics["mean_stress_contact"]) - float(cfg.max_stress_contact))
        pressure += 0.5 * max(0.0, float(cfg.target_energy_contact) - float(metrics["mean_energy_contact"]))
        pressure += float(metrics["dead_fraction"])
        if pressure <= 0.0:
            return {}

        step = _clip_scalar(float(cfg.learning_rate) * pressure, 0.0, 0.25)
        old_mu = float(params["m"])
        old_sigma = float(params["s"])
        new_mu = _clip_scalar(
            old_mu + step * (float(cfg.lenia_mu_center) - old_mu),
            cfg.lenia_mu_min,
            cfg.lenia_mu_max,
        )
        new_sigma = _clip_scalar(
            old_sigma + step * (float(cfg.lenia_sigma_center) - old_sigma),
            cfg.lenia_sigma_min,
            cfg.lenia_sigma_max,
        )
        if hasattr(env, "set_lenia_params"):
            env.set_lenia_params(m=new_mu, s=new_sigma)
        else:
            params["m"] = new_mu
            params["s"] = new_sigma

        updates: dict[str, dict[str, float]] = {}
        self._record_update(updates, "lenia.m", old_mu, new_mu)
        self._record_update(updates, "lenia.s", old_sigma, new_sigma)
        return updates
