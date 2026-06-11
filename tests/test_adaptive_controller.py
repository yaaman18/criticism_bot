from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trm_pipeline.adaptive_controller import AdaptiveController, AdaptiveControllerConfig


class _DummyEnv:
    def __init__(self) -> None:
        self.params = {"m": 0.40, "s": 0.078}
        self.updated = False

    def set_lenia_params(self, *, m=None, s=None) -> None:
        if m is not None:
            self.params["m"] = float(m)
        if s is not None:
            self.params["s"] = float(s)
        self.updated = True


def test_adaptive_controller_updates_runtime_and_lenia_under_pressure() -> None:
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(
            contact_w_thermal=0.75,
            contact_w_toxicity=0.95,
            contact_w_energy=0.35,
            beta_pi=4.0,
            aperture_gain=0.45,
        ),
        body=SimpleNamespace(aperture_gain=0.45),
        env=_DummyEnv(),
        history=[
            {
                "t": 0,
                "homeostatic_error": 0.45,
                "G": 0.20,
                "B": 0.25,
                "policy_entropy": 0.10,
                "contact_energy": 0.02,
                "contact_thermal": 0.80,
                "contact_toxicity": 0.70,
                "dead": False,
            },
            {
                "t": 1,
                "homeostatic_error": 0.42,
                "G": 0.24,
                "B": 0.27,
                "policy_entropy": 0.20,
                "contact_energy": 0.03,
                "contact_thermal": 0.76,
                "contact_toxicity": 0.74,
                "dead": False,
            },
        ],
    )
    controller = AdaptiveController(
        AdaptiveControllerConfig(
            enabled=True,
            interval=1,
            min_steps=2,
            window_size=2,
            learning_rate=0.20,
            target_homeostatic_error=0.10,
            min_policy_entropy=0.80,
            target_energy_contact=0.20,
            max_stress_contact=0.30,
        )
    )

    event = controller.maybe_update(runtime, t=1)

    assert event is not None
    assert "runtime.contact_w_thermal" in event["updates"]
    assert "runtime.contact_w_energy" in event["updates"]
    assert "runtime.beta_pi" in event["updates"]
    assert "lenia.m" in event["updates"]
    assert "lenia.s" in event["updates"]
    assert runtime.env.updated is True
    assert runtime.cfg.contact_w_thermal > 0.75
    assert runtime.cfg.contact_w_energy > 0.35
    assert runtime.cfg.beta_pi < 4.0
    assert np.isclose(runtime.env.params["m"], event["updates"]["lenia.m"]["new"])
    assert runtime.env.params["m"] < 0.40
    assert runtime.env.params["s"] < 0.078


def test_adaptive_controller_skips_when_disabled() -> None:
    runtime = SimpleNamespace(cfg=SimpleNamespace(), body=SimpleNamespace(), env=_DummyEnv(), history=[{"t": 0}])
    controller = AdaptiveController(AdaptiveControllerConfig(enabled=False))

    assert controller.maybe_update(runtime, t=0) is None
    assert controller.events == []
