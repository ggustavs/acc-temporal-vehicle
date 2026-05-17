"""CasADi+IPOPT MPC oracle for behaviour-cloning data generation."""

from __future__ import annotations

import casadi as ca
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from acc import constants as C


class MPCOracle:
    """Single-shooting MPC providing controller actions for a given state."""

    def __init__(self, horizon: int = C.MPC_HORIZON) -> None:
        self.horizon = horizon
        self.fallback_count = 0
        self.call_count = 0
        self._solver, self._lbx, self._ubx, self._lbg, self._ubg = self._build()

    def _build(
        self,
    ) -> tuple[ca.Function, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = self.horizon
        u = ca.MX.sym("u", N)  # pyright: ignore[reportArgumentType]
        s = ca.MX.sym("s", N + 1)  # pyright: ignore[reportArgumentType]
        x0 = ca.MX.sym("x0", C.STATE_DIM)  # pyright: ignore[reportArgumentType]

        cost = 0
        g = []
        x_k = x0
        for k in range(N):
            v_ego_k = x_k[C.IDX_V_EGO]
            cost += C.MPC_COST_W_SPEED * (v_ego_k - C.V_SET) ** 2
            cost += C.MPC_COST_W_ACT * u[k] ** 2
            if k > 0:
                cost += C.MPC_COST_W_RATE * (u[k] - u[k - 1]) ** 2

            d_rel_k = x_k[C.IDX_X_LEAD] - x_k[C.IDX_X_EGO]
            d_safe_k = C.D_DEFAULT + C.T_GAP * v_ego_k
            g.append(d_rel_k + s[k] - d_safe_k)
            cost += C.MPC_COST_W_SLACK * s[k]

            x_k = self._step_casadi_sym(x_k, u[k])

        v_ego_N = x_k[C.IDX_V_EGO]
        cost += C.MPC_COST_W_SPEED * (v_ego_N - C.V_SET) ** 2
        d_rel_N = x_k[C.IDX_X_LEAD] - x_k[C.IDX_X_EGO]
        d_safe_N = C.D_DEFAULT + C.T_GAP * v_ego_N
        g.append(d_rel_N + s[N] - d_safe_N)
        cost += C.MPC_COST_W_SLACK * s[N]

        decision = ca.vertcat(u, s)
        constraints = ca.vertcat(*g)

        nlp = {"x": decision, "f": cost, "g": constraints, "p": x0}
        solver = ca.nlpsol(
            "mpc",
            "ipopt",
            nlp,
            {
                "print_time": 0,
                "ipopt": {
                    "print_level": 0,
                    "sb": "yes",
                    "max_iter": 200,
                    "tol": 1e-6,
                    "acceptable_tol": 1e-4,
                },
            },
        )

        u_dim = N
        s_dim = N + 1
        lbx = np.full(u_dim + s_dim, -np.inf)
        ubx = np.full(u_dim + s_dim, np.inf)
        lbx[:u_dim] = C.ACT_LO
        ubx[:u_dim] = C.ACT_HI
        lbx[u_dim:] = 0.0

        n_safety = N + 1
        lbg = np.zeros(n_safety)
        ubg = np.full(n_safety, np.inf)

        return solver, lbx, ubx, lbg, ubg

    @staticmethod
    def _step_casadi_sym(state: ca.MX, action_scalar: ca.MX) -> ca.MX:
        v_lead = state[C.IDX_V_LEAD]
        g_lead = state[C.IDX_G_LEAD]
        v_ego = state[C.IDX_V_EGO]
        g_ego = state[C.IDX_G_EGO]

        a_lead = C.A_LEAD_HARD / (1 + ca.exp(-C.K_LEAD * (v_lead - C.V_LEAD_RELEASE)))

        dx_lead = v_lead
        dv_lead = g_lead
        dg_lead = (a_lead - g_lead) / C.TAU - C.MU * v_lead * v_lead
        dx_ego = v_ego
        dv_ego = g_ego
        dg_ego = (action_scalar - g_ego) / C.TAU - C.MU * v_ego * v_ego

        deriv = ca.vertcat(dx_lead, dv_lead, dg_lead, dx_ego, dv_ego, dg_ego)
        return state + C.DT * deriv

    @classmethod
    def step_casadi(cls, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        s_sym = ca.MX.sym("s", C.STATE_DIM)  # pyright: ignore[reportArgumentType]
        a_sym = ca.MX.sym("a")  # pyright: ignore[reportArgumentType]
        f = ca.Function("f", [s_sym, a_sym], [cls._step_casadi_sym(s_sym, a_sym)])
        return np.array(f(state, action[0])).flatten()

    def __call__(
        self,
        state: Float[Tensor, "B 6"],
    ) -> Float[Tensor, "B 1"]:
        actions = []
        for s in state.detach().cpu().numpy():
            actions.append(self._solve_single(s))
        return torch.from_numpy(np.array(actions, dtype=np.float32)).reshape(-1, 1)

    def _solve_single(self, state: np.ndarray) -> float:
        self.call_count += 1
        try:
            warm_x = np.zeros(self._lbx.shape)
            sol = self._solver(
                x0=warm_x,
                lbx=self._lbx,
                ubx=self._ubx,
                lbg=self._lbg,
                ubg=self._ubg,
                p=state,
            )
            stats = self._solver.stats()
            if not stats.get("success", False):
                self.fallback_count += 1
                return C.ACT_LO
            decision = np.array(sol["x"]).flatten()  # pyright: ignore[reportOptionalSubscript, reportCallIssue, reportArgumentType]
            return float(decision[0])
        except RuntimeError:
            self.fallback_count += 1
            return C.ACT_LO
