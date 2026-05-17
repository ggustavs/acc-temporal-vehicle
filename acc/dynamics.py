"""Forward-Euler ARCH-COMP ACC plant; jit-compiled for rollout speed."""

from jaxtyping import Float
import torch
from torch import Tensor

from acc import constants as C


_DT = C.DT
_TAU = C.TAU
_MU = C.MU
_A_LEAD_HARD = C.A_LEAD_HARD
_V_LEAD_RELEASE = C.V_LEAD_RELEASE
_K_LEAD = C.K_LEAD


@torch.jit.script
def _acc_dynamics_step_impl(
    state: Tensor,
    action: Tensor,
    dt: float,
    tau: float,
    mu: float,
    a_lead_hard: float,
    v_lead_release: float,
    k_lead: float,
) -> Tensor:
    v_lead = state[..., 1]
    g_lead = state[..., 2]
    v_ego = state[..., 4]
    g_ego = state[..., 5]
    a_ego = action[..., 0]

    a_lead = a_lead_hard * torch.sigmoid(k_lead * (v_lead - v_lead_release))

    dx_lead = v_lead
    dv_lead = g_lead
    dg_lead = (a_lead - g_lead) / tau - mu * v_lead * v_lead

    dx_ego = v_ego
    dv_ego = g_ego
    dg_ego = (a_ego - g_ego) / tau - mu * v_ego * v_ego

    deriv = torch.stack([dx_lead, dv_lead, dg_lead, dx_ego, dv_ego, dg_ego], dim=-1)
    return state + dt * deriv


def acc_dynamics_step(
    state: Float[Tensor, "B 6"],
    action: Float[Tensor, "B 1"],
) -> Float[Tensor, "B 6"]:
    return _acc_dynamics_step_impl(
        state,
        action,
        _DT,
        _TAU,
        _MU,
        _A_LEAD_HARD,
        _V_LEAD_RELEASE,
        _K_LEAD,
    )
