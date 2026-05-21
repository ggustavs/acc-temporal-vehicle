import math

import torch

from acc import constants as C
from acc.dynamics import acc_dynamics_step


def _saturate(cmd: float) -> float:
    centre = 0.5 * (C.ACT_HI + C.ACT_LO)
    half = 0.5 * (C.ACT_HI - C.ACT_LO)
    return centre + half * math.tanh((cmd - centre) / half)


def _state(**overrides: float) -> torch.Tensor:
    s = torch.zeros(1, C.STATE_DIM)
    for name, val in overrides.items():
        idx = getattr(C, f"IDX_{name.upper()}")
        s[0, idx] = val
    return s


def test_forward_euler_one_step():
    state = _state(v_lead=30.0, v_ego=20.0, x_lead=100.0, x_ego=10.0)
    action = torch.tensor([[0.5]])
    nxt = acc_dynamics_step(state, action)
    expected_x_lead = 100.0 + C.DT * 30.0
    expected_x_ego = 10.0 + C.DT * 20.0
    expected_g_ego = 0.0 + C.DT * ((_saturate(0.5) - 0.0) / C.TAU - C.MU * 20.0**2)
    assert torch.allclose(
        nxt[0, C.IDX_X_LEAD], torch.tensor(expected_x_lead), atol=1e-5
    )
    assert torch.allclose(nxt[0, C.IDX_X_EGO], torch.tensor(expected_x_ego), atol=1e-5)
    assert torch.allclose(nxt[0, C.IDX_G_EGO], torch.tensor(expected_g_ego), atol=1e-5)


def test_positive_action_accelerates_ego():
    state = _state()
    action = torch.tensor([[1.0]])
    nxt = acc_dynamics_step(state, action)
    assert nxt[0, C.IDX_G_EGO] > 0


def test_drag_reduces_acceleration_at_high_speed():
    s_low = _state(v_ego=0.0)
    s_high = _state(v_ego=50.0)
    action = torch.tensor([[1.0]])
    g_low = acc_dynamics_step(s_low, action)[0, C.IDX_G_EGO]
    g_high = acc_dynamics_step(s_high, action)[0, C.IDX_G_EGO]
    assert g_high < g_low


def test_lead_constant_braking_decelerates_lead():
    state = _state(v_lead=30.0)
    action = torch.zeros(1, C.ACT_DIM)
    nxt = acc_dynamics_step(state, action)
    assert nxt[0, C.IDX_G_LEAD] < 0
