"""Rollout, initial-state sampling, arm loading. Pure simulation/data layer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from acc import constants as C
from acc.controller import fresh_controller, load_checkpoint, published_controller

SEED = 0
T = C.N_STEPS
DT = C.DT
VSET, TGAP, DDEF, CMAX = C.V_SET, C.T_GAP, C.D_DEFAULT, C.COMFORT_MAX
iXL, iVL, iGL, iXE, iVE, iGE = 0, 1, 2, 3, 4, 5

LeadFn = Callable[[int, torch.Tensor], torch.Tensor]


def _state_to_obs(s: torch.Tensor, v_set: float | torch.Tensor = VSET) -> torch.Tensor:
    n = s.shape[0]
    if torch.is_tensor(v_set):
        v_set_t = v_set.to(dtype=s.dtype)
    else:
        v_set_t = torch.full((n,), float(v_set), dtype=s.dtype)
    return torch.stack(
        [
            v_set_t,
            torch.full((n,), TGAP, dtype=s.dtype),
            s[:, iVE],
            s[:, iXL] - s[:, iXE],
            s[:, iVL] - s[:, iVE],
        ],
        dim=1,
    )


def _step(s: torch.Tensor, a: torch.Tensor, a_lead: torch.Tensor) -> torch.Tensor:
    vL, gL, vE, gE = s[:, iVL], s[:, iGL], s[:, iVE], s[:, iGE]
    aE = a[:, 0]
    dgL = (a_lead - gL) / C.TAU - C.MU * vL * vL
    dgE = (aE - gE) / C.TAU - C.MU * vE * vE
    deriv = torch.stack([vL, gL, dgL, vE, gE, dgE], dim=1)
    return s + DT * deriv


def _string_lead_a(mode: str, t: int, s: torch.Tensor) -> torch.Tensor:
    """Per-step a_lead for the string-named lead-vehicle profiles."""
    vL = s[:, iVL]
    if mode == "training":
        # The sigmoid-decel lead profile the controllers were trained against.
        return C.A_LEAD_HARD * torch.sigmoid(C.K_LEAD * (vL - C.V_LEAD_RELEASE))
    if mode == "delayed":
        return torch.full_like(vL, 0.0) if t < T // 2 else torch.full_like(vL, -3.0)
    if mode.startswith("a="):
        return torch.full_like(vL, float(mode.split("=")[1]))
    raise ValueError(f"unknown lead mode {mode!r}")


def rollout(
    net: nn.Module,
    x0: torch.Tensor,
    lead: str | LeadFn,
    *,
    grad: bool = False,
    v_set: float | torch.Tensor = VSET,
) -> torch.Tensor:
    """Roll out `net` from `x0` for T steps. `lead` is either a string
    profile name (training / delayed / a=<float>) or a callable
    (t, state) -> a_lead tensor [N]."""
    if isinstance(lead, str):
        mode = lead
        lead_fn: LeadFn = lambda t, s: _string_lead_a(mode, t, s)  # noqa: E731
    else:
        lead_fn = lead
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        s = x0
        traj = [s]
        for t in range(T):
            a_lead = lead_fn(t, s)
            a = net(_state_to_obs(s, v_set))
            s = _step(s, a, a_lead)
            traj.append(s)
        return torch.stack(traj, dim=1)


def sample_box(n: int, factor: float, seed: int) -> torch.Tensor:
    """Sample `n` initial states from the (centre +/- factor*half) box."""
    rng = np.random.default_rng(seed)
    centre = 0.5 * (C.INITIAL_LO + C.INITIAL_HI)
    half = 0.5 * (C.INITIAL_HI - C.INITIAL_LO)
    lo, hi = centre - factor * half, centre + factor * half
    pts = np.tile(centre, (n, 1))
    free = [iXL, iVL, iXE, iVE]
    pts[:, free] = rng.uniform(lo[free], hi[free], size=(n, 4))
    return torch.tensor(pts, dtype=torch.float32)


# ---- ACC operating-point sweep --------------------------------------
ACC_SWEEP_V_SET = (20.0, 25.0, 30.0)
ACC_SWEEP_V_REL = (-5.0, -2.0, 0.0, 2.0, 5.0)
ACC_SWEEP_SLACK = (5.0, 20.0, 40.0)


def acc_sweep_cells() -> list[tuple[float, float, float]]:
    """The 45-cell grid in fixed iteration order."""
    return [
        (v_set, v_rel, slack)
        for v_set in ACC_SWEEP_V_SET
        for v_rel in ACC_SWEEP_V_REL
        for slack in ACC_SWEEP_SLACK
    ]


def acc_sweep_sample(
    n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Return (state[N,6], v_set[N], cell_id[N]) for the 45-cell sweep."""
    rng = np.random.default_rng(seed)
    cells = acc_sweep_cells()
    per_cell = max(1, n // len(cells))
    total = per_cell * len(cells)

    states = np.zeros((total, 6), dtype=np.float32)
    v_set_per_init = np.zeros(total, dtype=np.float32)
    cell_id = np.empty(total, dtype=object)

    for i, (v_set, v_rel, slack) in enumerate(cells):
        sl = slice(i * per_cell, (i + 1) * per_cell)
        v_ego = v_set + rng.uniform(-0.5, 0.5, per_cell)
        v_lead = v_ego + v_rel + rng.uniform(-0.2, 0.2, per_cell)
        x_ego = rng.uniform(0.0, 5.0, per_cell)
        d_safe = C.D_DEFAULT + C.T_GAP * np.maximum(v_ego, 0.0)
        d_rel = d_safe + slack + rng.uniform(-2.0, 2.0, per_cell)
        x_lead = x_ego + d_rel
        states[sl, iXL] = x_lead
        states[sl, iVL] = v_lead
        states[sl, iXE] = x_ego
        states[sl, iVE] = v_ego
        v_set_per_init[sl] = v_set
        cell_id[sl] = f"v_set={v_set:g}|v_rel={v_rel:+g}|slack={slack:+g}"

    return torch.from_numpy(states), torch.from_numpy(v_set_per_init), cell_id


# ---- Arm loading ----------------------------------------------------
def default_arms() -> dict[str, str]:
    return {
        "published": "onnx",
        "stl": str(C.STL_CHECKPOINT_PATH),
        "sfo": str(C.SFO_CHECKPOINT_PATH),
    }


def load_arms(arms_to_paths: dict[str, str]) -> dict[str, nn.Module]:
    out: dict[str, nn.Module] = {}
    for tag, ck in arms_to_paths.items():
        if ck == "onnx":
            out[tag] = published_controller(C.PUBLISHED_ONNX)
        elif str(ck).endswith(".onnx"):
            out[tag] = published_controller(Path(ck))
        else:
            net = fresh_controller()
            net.load_state_dict(load_checkpoint(Path(ck)))
            out[tag] = net
    for net in out.values():
        net.eval()
    return out


def parse_checkpoint_kvs(kvs: list[str], base: dict[str, str]) -> dict[str, str]:
    """Apply repeated `--checkpoint arm=path` overrides onto a base dict."""
    arms = dict(base)
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"--checkpoint expects arm=path, got {item!r}")
        arm, path = item.split("=", 1)
        arms[arm] = path
    return arms
