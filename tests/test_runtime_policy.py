from __future__ import annotations

import numpy as np

from trm_pipeline.runtime_policy import ag_veto_params, apply_ag_assistive_veto


def test_ag_veto_params_returns_expected_thresholds() -> None:
    assert ag_veto_params(2) == (0.45, 0.60, 8.0)
    assert ag_veto_params(0) == (0.75, 0.90, 3.0)
    assert ag_veto_params(1) == (0.60, 0.80, 5.0)


def test_apply_ag_assistive_veto_prunes_and_hard_gates_logits() -> None:
    pre_logits = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    inhibition = np.array([0.1, 0.4, 0.8, 0.55, 0.2], dtype=np.float32)

    out = apply_ag_assistive_veto(pre_logits, inhibition, control_mode=2)

    # hard-threshold(0.60) を超えた index=2 は gate-floor に落ちる
    assert int(np.argmin(out)) == 2
    # hard-gate されない要素は有限のまま
    assert bool(np.all(np.isfinite(out)))
    assert out.shape == pre_logits.shape

