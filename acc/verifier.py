"""CROWN/Fastlin interval-reachability verifier for the closed-loop ACC system via immrax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import equinox as eqx
import equinox.nn as enn
import immrax
import jax
import jax.numpy as jnp
import numpy as np
from immutabledict import immutabledict
from torch import nn

from acc import constants as C

# immrax has no inclusion rule for `logistic` (the lead-brake sigmoid).
# Sigmoid is monotone, so the per-bound passthrough is exact, as immrax
# does for exp_p/atan_p.
from immrax.inclusion.nif import (  # noqa: E402
    _make_inclusion_passthrough_p as _immrax_passthrough,
)
from immrax.inclusion.nif import (  # noqa: E402
    inclusion_registry as _immrax_inclusion_registry,
)

_immrax_inclusion_registry.setdefault(
    jax.lax.logistic_p, _immrax_passthrough(jax.lax.logistic_p)
)


@dataclass
class VerifierResult:
    verified_safe: bool
    note: str


# (lower, upper) -> (satisfied, reason-if-not).
InvariantPredicate: TypeAlias = Callable[[np.ndarray, np.ndarray], tuple[bool, str]]


def _safety_predicate(lower: np.ndarray, upper: np.ndarray) -> tuple[bool, str]:
    d_rel_lower = float(lower[C.IDX_X_LEAD] - upper[C.IDX_X_EGO])
    v_ego_upper_floor = max(float(upper[C.IDX_V_EGO]), 0.0)
    d_safe_upper = C.D_DEFAULT + C.T_GAP * v_ego_upper_floor
    if d_rel_lower < d_safe_upper:
        return False, (
            f"d_rel lower = {d_rel_lower:+.4f}, d_safe upper = {d_safe_upper:+.4f}"
        )
    return True, ""


def _comfort_predicate(lower: np.ndarray, upper: np.ndarray) -> tuple[bool, str]:
    g_lo = float(lower[C.IDX_G_EGO])
    g_hi = float(upper[C.IDX_G_EGO])
    if g_hi > C.COMFORT_MAX:
        return False, f"g_ego upper = {g_hi:+.4f} > {C.COMFORT_MAX:+.4f}"
    if g_lo < -C.COMFORT_MAX:
        return False, f"g_ego lower = {g_lo:+.4f} < {-C.COMFORT_MAX:+.4f}"
    return True, ""


_INVARIANT_PREDICATES: dict[str, InvariantPredicate] = {
    "safe": _safety_predicate,
    "comfortable": _comfort_predicate,
}


def _jax_weights_from_torch(net: nn.Module) -> list[tuple[jax.Array, jax.Array]]:
    pairs = []
    for layer in net.children():
        if isinstance(layer, nn.Linear):
            W = jnp.asarray(layer.weight.detach().cpu().numpy())
            b = jnp.asarray(layer.bias.detach().cpu().numpy())
            pairs.append((W, b))
    return pairs


class _ACCController(immrax.Control):
    seq: enn.Sequential
    out_len: int

    def __init__(self, weights_and_biases: list[tuple[jax.Array, jax.Array]]):
        self.out_len = int(weights_and_biases[-1][0].shape[0])
        # state->observation is linear: encode as a Linear layer so CROWN
        # bound-propagates through it. v_set/t_gap go into the bias.
        proj_W = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # v_set     (constant)
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # t_gap     (constant)
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # v_ego     = state[4]
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],  # x_lead - x_ego
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],  # v_lead - v_ego
            ]
        )
        proj_b = jnp.array([C.V_SET, C.T_GAP, 0.0, 0.0, 0.0])

        layers: list = []

        proj = enn.Linear(
            in_features=C.STATE_DIM,
            out_features=C.OBS_DIM,
            key=jax.random.PRNGKey(0),
        )
        proj = eqx.tree_at(lambda layer: layer.weight, proj, proj_W)
        proj = eqx.tree_at(lambda layer: layer.bias, proj, proj_b)
        layers.append(proj)

        # No ReLU after proj (it's the linear stateToObs) or after the
        # final layer (its output is the action).
        for i, (W, b) in enumerate(weights_and_biases):
            lin = enn.Linear(
                in_features=W.shape[1],
                out_features=W.shape[0],
                key=jax.random.PRNGKey(i + 1),
            )
            lin = eqx.tree_at(lambda layer: layer.weight, lin, W)
            lin = eqx.tree_at(lambda layer: layer.bias, lin, b)
            layers.append(lin)
            if i != len(weights_and_biases) - 1:
                layers.append(enn.Lambda(jax.nn.relu))

        self.seq = enn.Sequential(layers)

    def __call__(self, x):
        # Bare callable form: CROWN/Fastlin bound propagation calls this directly.
        return self.seq(x)

    def u(self, t, x):
        return self.seq(x)


def _acc_dynamics_step_jax(state: jax.Array, action: jax.Array) -> jax.Array:
    # JAX mirror of acc.dynamics._acc_dynamics_step_impl; equivalence is
    # enforced by tests/test_jax_matches_pytorch.py.
    v_lead = state[C.IDX_V_LEAD]
    g_lead = state[C.IDX_G_LEAD]
    v_ego = state[C.IDX_V_EGO]
    g_ego = state[C.IDX_G_EGO]

    centre = 0.5 * (C.ACT_HI + C.ACT_LO)
    half = 0.5 * (C.ACT_HI - C.ACT_LO)
    a_ego = centre + half * jnp.tanh((action[0] - centre) / half)

    a_lead = C.A_LEAD_HARD * jax.nn.sigmoid(C.K_LEAD * (v_lead - C.V_LEAD_RELEASE))

    dx_lead = v_lead
    dv_lead = g_lead
    dg_lead = (a_lead - g_lead) / C.TAU - C.MU * v_lead * v_lead

    dx_ego = v_ego
    dv_ego = g_ego
    dg_ego = (a_ego - g_ego) / C.TAU - C.MU * v_ego * v_ego

    deriv = jnp.stack([dx_lead, dv_lead, dg_lead, dx_ego, dv_ego, dg_ego])
    nxt = state + C.DT * deriv
    return (
        nxt.at[C.IDX_V_LEAD]
        .set(jnp.maximum(nxt[C.IDX_V_LEAD], 0.0))
        .at[C.IDX_V_EGO]
        .set(jnp.maximum(nxt[C.IDX_V_EGO], 0.0))
    )


class _ACCOpenLoop(immrax.OpenLoopSystem):
    def __init__(self) -> None:
        super().__init__(evolution="discrete", xlen=C.STATE_DIM)

    def f(self, t, x, u, w):  # noqa: ARG002 (w is the unused disturbance)
        return _acc_dynamics_step_jax(x, u)


def _constant_lead_dynamics_step_jax(
    state: jax.Array, action: jax.Array
) -> jax.Array:
    """JAX mirror of acc.dynamics._constant_lead_dynamics_step_impl: lead
    held at init v_lead (a_lead = 0, g_lead = 0), ego dynamics unchanged."""
    v_lead = state[C.IDX_V_LEAD]
    v_ego = state[C.IDX_V_EGO]
    g_ego = state[C.IDX_G_EGO]

    centre = 0.5 * (C.ACT_HI + C.ACT_LO)
    half = 0.5 * (C.ACT_HI - C.ACT_LO)
    a_ego = centre + half * jnp.tanh((action[0] - centre) / half)

    dx_lead = v_lead
    dv_lead = jnp.zeros_like(v_lead)
    dg_lead = jnp.zeros_like(v_lead)
    dx_ego = v_ego
    dv_ego = g_ego
    dg_ego = (a_ego - g_ego) / C.TAU - C.MU * v_ego * v_ego

    deriv = jnp.stack([dx_lead, dv_lead, dg_lead, dx_ego, dv_ego, dg_ego])
    nxt = state + C.DT * deriv
    return (
        nxt.at[C.IDX_V_LEAD]
        .set(jnp.maximum(nxt[C.IDX_V_LEAD], 0.0))
        .at[C.IDX_V_EGO]
        .set(jnp.maximum(nxt[C.IDX_V_EGO], 0.0))
    )


class _ConstantLeadOpenLoop(immrax.OpenLoopSystem):
    def __init__(self) -> None:
        super().__init__(evolution="discrete", xlen=C.STATE_DIM)

    def f(self, t, x, u, w):  # noqa: ARG002
        return _constant_lead_dynamics_step_jax(x, u)


def verify_initial_box_invariants(
    controller: nn.Module,
    initial_lo: np.ndarray = C.INITIAL_LO,
    initial_hi: np.ndarray = C.INITIAL_HI,
    n_steps: int = C.N_STEPS,
    plant: str = "default",
) -> dict[str, VerifierResult]:
    """`plant`: 'default' = sigmoid-decel lead (ARCH-COMP); 'constant_lead' =
    lead held at init v_lead (steady-follow regime)."""
    weights = _jax_weights_from_torch(controller)
    ctrl = _ACCController(weights)
    plant_obj: immrax.OpenLoopSystem
    if plant == "default":
        plant_obj = _ACCOpenLoop()
    elif plant == "constant_lead":
        plant_obj = _ConstantLeadOpenLoop()
    else:
        raise ValueError(f"unknown plant {plant!r}; pick from default | constant_lead")
    nncs = immrax.NNCSystem(plant_obj, ctrl)  # pyright: ignore[reportArgumentType]
    emb = immrax.NNCEmbeddingSystem(nncs, nn_verifier="crown")

    init_ix = immrax.interval(jnp.asarray(initial_lo), jnp.asarray(initial_hi))

    # Embedding state is upper-triangle `2 * xlen` (concat of lower+upper);
    # i2ut / ut2i are the boundary conversions.
    x0_ut = immrax.i2ut(init_ix)
    empty_w = immrax.interval(jnp.zeros(0), jnp.zeros(0))
    # corners size = 1 (time) + xlen + control.out_len + len(w).
    n_corner = 1 + C.STATE_DIM + ctrl.out_len + 0
    traj = emb.compute_trajectory(
        t0=0,
        tf=n_steps,
        x0=x0_ut,
        f_kwargs=immutabledict(
            {
                "w": empty_w,
                "permutations": immrax.standard_permutation(n_corner),
                "corners": immrax.two_corners(n_corner),
            }
        ),
        dt=1,
        solver="euler",
    )

    # traj.ys is right-padded with jnp.inf past tf; skip those rows.
    states_ut = np.asarray(traj.ys)

    results: dict[str, VerifierResult] = {}
    for step in range(min(n_steps, len(states_ut))):
        ut = states_ut[step]
        if not np.all(np.isfinite(ut)):
            break
        x_ix = immrax.ut2i(jnp.asarray(ut))
        lower, upper = immrax.i2lu(x_ix)
        for name, predicate in _INVARIANT_PREDICATES.items():
            if name in results:
                continue
            satisfied, why = predicate(np.asarray(lower), np.asarray(upper))
            if not satisfied:
                results[name] = VerifierResult(
                    verified_safe=False,
                    note=f"falsified at step {step}: {why}",
                )

    for name in _INVARIANT_PREDICATES:
        if name not in results:
            results[name] = VerifierResult(
                verified_safe=True,
                note=f"CROWN-bounded reachable set holds over {n_steps} discrete steps",
            )
    return results


def verify_initial_box_safe(
    controller: nn.Module,
    initial_lo: np.ndarray = C.INITIAL_LO,
    initial_hi: np.ndarray = C.INITIAL_HI,
    n_steps: int = C.N_STEPS,
    plant: str = "default",
) -> VerifierResult:
    """Back-compat shim for the safety-only verifier."""
    return verify_initial_box_invariants(
        controller, initial_lo, initial_hi, n_steps, plant=plant
    )["safe"]
