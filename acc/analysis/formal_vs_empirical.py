"""CROWN reachable-set bounds vs Monte-Carlo empirical envelope, per arm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import immrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from immutabledict import immutabledict
from rich.console import Console
from torch import nn

from acc import constants as C
from acc.analysis._data import (
    DT,
    SEED,
    iVE,
    iXE,
    iXL,
    lead_mode_for_box,
    load_arms,
    rollout,
    sample_box,
)
from acc.analysis._metrics import margins
from acc.analysis._plotting import crown_vs_mc_panel, save_figure
from acc.verifier import (
    _ACCController,
    _ACCOpenLoop,
    _ConstantLeadOpenLoop,
    _jax_weights_from_torch,
)

_N_MC = 4000


def crown_bounds(
    net: nn.Module,
    *,
    init_lo: np.ndarray | None = None,
    init_hi: np.ndarray | None = None,
    plant: str = "default",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step interval reachable set [lower, upper] over N_STEPS.
    `plant`: 'default' = sigmoid-decel; 'constant_lead' = constant v_lead."""
    lo = C.INITIAL_LO if init_lo is None else init_lo
    hi = C.INITIAL_HI if init_hi is None else init_hi
    ctrl = _ACCController(_jax_weights_from_torch(net))
    plant_obj = _ACCOpenLoop() if plant == "default" else _ConstantLeadOpenLoop()
    emb = immrax.NNCEmbeddingSystem(
        immrax.NNCSystem(plant_obj, ctrl),  # pyright: ignore[reportArgumentType]
        nn_verifier="crown",
    )
    init_ix = immrax.interval(jnp.asarray(lo), jnp.asarray(hi))
    n_corner = 1 + C.STATE_DIM + ctrl.out_len
    traj = emb.compute_trajectory(
        t0=0,
        tf=C.N_STEPS,
        x0=immrax.i2ut(init_ix),
        f_kwargs=immutabledict(
            {
                "w": immrax.interval(jnp.zeros(0), jnp.zeros(0)),
                "permutations": immrax.standard_permutation(n_corner),
                "corners": immrax.two_corners(n_corner),
            }
        ),
        dt=1,
        solver="euler",
    )
    ys = np.asarray(traj.ys)
    lo, hi = [], []
    for ut in ys[: C.N_STEPS]:
        if not np.all(np.isfinite(ut)):
            break
        lo_i, hi_i = immrax.i2lu(immrax.ut2i(jnp.asarray(ut)))
        lo.append(np.asarray(lo_i))
        hi.append(np.asarray(hi_i))
    return np.array(lo), np.array(hi)


def _crown_margin_band(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower / upper bound of the safety margin from interval state bounds.
    Each component takes its monotone-worst case: lower margin uses
    upper(x_ego), upper(v_ego); upper margin uses lower(x_ego),
    lower(v_ego)."""
    crown_lo = (lo[:, iXL] - hi[:, iXE]) - (C.D_DEFAULT + C.T_GAP * hi[:, iVE])
    crown_hi = (hi[:, iXL] - lo[:, iXE]) - (C.D_DEFAULT + C.T_GAP * lo[:, iVE])
    return crown_lo, crown_hi


def formal_compute(
    arms: dict,
    *,
    init_box: str = "default",
) -> dict[str, dict[str, np.ndarray]]:
    """For each arm: CROWN safety-margin band + MC 5-95% / worst trace.
    `init_box` selects bounds + matching plant/lead for both CROWN and MC."""
    if init_box == "default":
        init_lo, init_hi = C.INITIAL_LO, C.INITIAL_HI
        plant = "default"
    elif init_box == "finetune":
        init_lo, init_hi = C.INITIAL_LO_FINETUNE, C.INITIAL_HI_FINETUNE
        plant = "constant_lead"
    else:
        raise ValueError(f"unknown init_box {init_box!r}; pick default | finetune")
    lead = lead_mode_for_box(init_box)
    x0 = sample_box(_N_MC, 1.0, SEED, box=init_box)
    out: dict[str, dict[str, np.ndarray]] = {}
    for k, net in arms.items():
        lo, hi = crown_bounds(net, init_lo=init_lo, init_hi=init_hi, plant=plant)
        crown_lo, crown_hi = _crown_margin_band(lo, hi)
        S = crown_lo.shape[0]
        mc = margins(rollout(net, x0, lead).numpy())[:, :S]
        out[k] = {
            "crown_lo": crown_lo,
            "crown_hi": crown_hi,
            "mc_lo": np.percentile(mc, 5, 0),
            "mc_hi": np.percentile(mc, 95, 0),
            "mc_worst": mc.min(0),
        }
    return out


def formal_render(data: dict[str, dict[str, np.ndarray]], fig_dir: Path) -> None:
    n_arms = len(data)
    width = 17 if n_arms == 3 else max(5.7 * n_arms, 6.0)
    fig, axes = plt.subplots(1, n_arms, figsize=(width, 4.4), sharex=True)
    if n_arms == 1:
        axes = [axes]
    for j, (arm, band) in enumerate(data.items()):
        S = band["crown_lo"].shape[0]
        t = np.arange(S) * DT
        crown_vs_mc_panel(
            axes[j],
            t,
            band["crown_lo"],
            band["crown_hi"],
            band["mc_lo"],
            band["mc_hi"],
            band["mc_worst"],
            arm=arm,
            fallback_index=j,
        )
    fig.suptitle(
        "Formal (CROWN) vs empirical (MC) safety margin, training plant. "
        "CROWN sound (lower band <0 means cannot certify); MC worst <0 "
        "means a real counterexample."
    )
    fig.tight_layout()
    save_figure(fig, fig_dir, "formal_vs_empirical.png")


def formal_summarise(
    data: dict[str, dict[str, np.ndarray]],
) -> dict:
    out: dict = {}
    for arm, band in data.items():
        crown_min = float(np.nanmin(band["crown_lo"]))
        mc_min = float(band["mc_worst"].min())
        out[arm] = {
            "crown_min_margin_lower": crown_min,
            "crown_verified_safe": bool(np.all(band["crown_lo"] >= 0)),
            "mc_min_margin": mc_min,
            "mc_violates": bool(mc_min < 0),
            "steps_bounded": int(band["crown_lo"].shape[0]),
            "conservatism_gap_at_min": float(mc_min - crown_min),
        }
    return out


def formal_vs_empirical_core(
    arms_to_paths: dict[str, str],
    out_dir: Path,
    console: Optional[Console] = None,
    init_box: str = "default",
) -> dict:
    console = console or Console()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    arms = load_arms(arms_to_paths)
    data = formal_compute(arms, init_box=init_box)
    formal_render(data, fig_dir)
    summary = formal_summarise(data)

    (out_dir / "formal_vs_empirical.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        console.print(
            "%s CROWN cert=%s (low %.2f) | MC violates=%s (worst %.2f) | gap %.2f"
            % (
                k,
                v["crown_verified_safe"],
                v["crown_min_margin_lower"],
                v["mc_violates"],
                v["mc_min_margin"],
                v["conservatism_gap_at_min"],
            )
        )
    return summary
