"""ACC-typical lead-vehicle scenarios within ISO 22179 (no AEB-class braking or cut-ins)."""

from __future__ import annotations

from typing import Callable

import torch

from acc.analysis._data import DT, iVL

LeadFn = Callable[[int, torch.Tensor], torch.Tensor]


def lead_constant_v(v_target: float, gain: float = 1.0) -> LeadFn:
    """P-controller pulling v_lead toward v_target. Comfort-clamped."""

    def fn(t: int, s: torch.Tensor) -> torch.Tensor:
        return torch.clamp(gain * (v_target - s[:, iVL]), -1.5, 1.5)

    return fn


def lead_ramp_dec(a: float, t_start_s: float, t_end_s: float) -> LeadFn:
    """Gentle decel a (must be in [-1.5, -0.1]) over [t_start, t_end] s."""
    assert -1.5 <= a < 0, f"lead_ramp_dec a={a} outside comfort band"
    t0 = int(round(t_start_s / DT))
    t1 = int(round(t_end_s / DT))

    def fn(t: int, s: torch.Tensor) -> torch.Tensor:
        if t0 <= t < t1:
            return torch.full_like(s[:, iVL], a)
        return torch.zeros_like(s[:, iVL])

    return fn


def lead_ramp_acc(a: float, t_start_s: float, t_end_s: float) -> LeadFn:
    """Gentle accel a (must be in [0.1, 1.5]) over [t_start, t_end] s."""
    assert 0 < a <= 1.5, f"lead_ramp_acc a={a} outside comfort band"
    t0 = int(round(t_start_s / DT))
    t1 = int(round(t_end_s / DT))

    def fn(t: int, s: torch.Tensor) -> torch.Tensor:
        if t0 <= t < t1:
            return torch.full_like(s[:, iVL], a)
        return torch.zeros_like(s[:, iVL])

    return fn
