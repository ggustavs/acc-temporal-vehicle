"""ACC operating-point sweep: per-cell metrics over (v_set, v_rel, slack)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console

from acc.analysis._data import (
    ACC_SWEEP_SLACK,
    ACC_SWEEP_V_REL,
    ACC_SWEEP_V_SET,
    SEED,
    acc_sweep_sample,
    load_arms,
    rollout,
)
from acc.analysis._metrics import (
    accel_rms,
    jerk_rms,
    margins,
    steady_state_v_err,
    time_headway_rmse,
    time_in_saturation,
)
from acc.analysis._plotting import save_figure

_HEATMAP_METRICS = (
    "jerk_rms_median",
    "accel_rms_median",
    "time_headway_rmse_median",
    "steady_state_v_err_median",
    "median_min_margin",
    "time_in_saturation_mean",
)


def _per_cell_metrics(tr: np.ndarray, v_set_per_init: np.ndarray) -> dict[str, float]:
    return {
        "median_min_margin": float(np.median(margins(tr).min(1))),
        "jerk_rms_median": float(np.median(jerk_rms(tr))),
        "accel_rms_median": float(np.median(accel_rms(tr))),
        "time_in_saturation_mean": float(time_in_saturation(tr).mean()),
        "time_headway_rmse_median": float(np.median(time_headway_rmse(tr))),
        "steady_state_v_err_median": float(
            np.median(steady_state_v_err(tr, v_set=v_set_per_init))
        ),
    }


def _plot_heatmaps(metrics: dict, order: list[str], fig_dir: Path) -> None:
    n_arms = len(order)
    for m in _HEATMAP_METRICS:
        fig, axes = plt.subplots(
            n_arms,
            len(ACC_SWEEP_V_SET),
            figsize=(4.5 * len(ACC_SWEEP_V_SET), 3.0 * n_arms),
            sharex=True,
            sharey=True,
        )
        if n_arms == 1:
            axes = np.array([axes])
        if len(ACC_SWEEP_V_SET) == 1:
            axes = axes.reshape(-1, 1)
        grids: dict[tuple[str, float], np.ndarray] = {}
        for arm in order:
            for vs in ACC_SWEEP_V_SET:
                g = np.zeros((len(ACC_SWEEP_V_REL), len(ACC_SWEEP_SLACK)))
                for i_vr, vr in enumerate(ACC_SWEEP_V_REL):
                    for i_sl, sl in enumerate(ACC_SWEEP_SLACK):
                        cell = f"v_set={vs:g}|v_rel={vr:+g}|slack={sl:+g}"
                        g[i_vr, i_sl] = metrics["acc_sweep"][cell][arm][m]
                grids[(arm, vs)] = g
        vmin = min(g.min() for g in grids.values())
        vmax = max(g.max() for g in grids.values())
        im = None
        for i_arm, arm in enumerate(order):
            for i_vs, vs in enumerate(ACC_SWEEP_V_SET):
                ax = axes[i_arm, i_vs]
                im = ax.imshow(
                    grids[(arm, vs)],
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                    aspect="auto",
                )
                ax.set_xticks(
                    range(len(ACC_SWEEP_SLACK)),
                    [f"+{s:g}" for s in ACC_SWEEP_SLACK],
                )
                ax.set_yticks(
                    range(len(ACC_SWEEP_V_REL)),
                    [f"{v:+g}" for v in ACC_SWEEP_V_REL],
                )
                for r in range(len(ACC_SWEEP_V_REL)):
                    for c in range(len(ACC_SWEEP_SLACK)):
                        ax.text(
                            c,
                            r,
                            f"{grids[(arm, vs)][r, c]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="w",
                        )
                if i_arm == 0:
                    ax.set_title(f"v_set={vs:g}")
                if i_vs == 0:
                    ax.set_ylabel(f"{arm}\nv_rel")
                if i_arm == n_arms - 1:
                    ax.set_xlabel("slack")
        fig.suptitle(
            f"acc-sweep: {m}  (rows = arm, cols = v_set; axes = v_rel x slack)"
        )
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025)
        save_figure(fig, fig_dir, f"acc_sweep_{m}.png", bbox_inches="tight")


def deep_eval_acc_sweep(
    arms_to_paths: dict[str, str],
    out_dir: Path,
    console: Optional[Console] = None,
) -> dict:
    console = console or Console()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    nets = load_arms(arms_to_paths)
    order = list(nets.keys())

    N = 4050  # 45 cells x 90
    x0, v_set, cell_id = acc_sweep_sample(N, SEED)
    unique_cells: list[str] = []
    seen: set = set()
    for c in cell_id:
        if c not in seen:
            seen.add(c)
            unique_cells.append(c)

    metrics: dict = {
        "init_mode": "acc-sweep",
        "N": N,
        "seed": SEED,
        "cells": unique_cells,
        "acc_sweep": {},
    }

    for cell in unique_cells:
        mask = cell_id == cell
        x0_c = x0[mask]
        v_set_c = v_set[mask]
        metrics["acc_sweep"][cell] = {}
        for k, net in nets.items():
            tr = rollout(net, x0_c, "training", v_set=v_set_c).numpy()
            metrics["acc_sweep"][cell][k] = _per_cell_metrics(tr, v_set_c.numpy())

    _plot_heatmaps(metrics, order, fig_dir)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    n_png = len(list(fig_dir.glob("*.png")))
    console.print(f"wrote {metrics_path} and {n_png} figures")
    for m in ("jerk_rms_median", "median_min_margin"):
        vals = {k: [metrics["acc_sweep"][c][k][m] for c in unique_cells] for k in order}
        for k in order:
            spread = (max(vals[k]) - min(vals[k])) / (np.mean(vals[k]) + 1e-9) * 100
            console.print(
                f"  {k:10s} {m:30s} cell-spread={spread:5.1f}% "
                f"min={min(vals[k]):.3f} max={max(vals[k]):.3f}"
            )
    return metrics
