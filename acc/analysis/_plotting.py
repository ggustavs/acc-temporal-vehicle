"""Shared figure primitives: colour/label maps, plot templates, save helper."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---- Arm colour / label maps ---------------------------------------
_COLOUR = {"published": "#1f77b4", "stl": "#d62728", "sfo": "#2ca02c"}
_DEFAULT_COLOURS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_LABEL = {"published": "pub", "stl": "stl", "sfo": "sfo"}


def colour_for(arm: str, fallback_index: int = 0) -> str:
    return _COLOUR.get(arm, _DEFAULT_COLOURS[fallback_index % len(_DEFAULT_COLOURS)])


def label_for(arm: str) -> str:
    return _LABEL.get(arm, arm)


# ---- Save helper ---------------------------------------------------
def save_figure(
    fig: Figure,
    fig_dir: Path,
    name: str,
    *,
    dpi: int = 110,
    bbox_inches: str | None = None,
) -> None:
    kw: dict = {"dpi": dpi}
    if bbox_inches is not None:
        kw["bbox_inches"] = bbox_inches
    fig.savefig(fig_dir / name, **kw)
    plt.close(fig)


# ---- Plot primitives -----------------------------------------------
def time_series_band(
    ax: Axes,
    t: np.ndarray,
    traces_per_arm: dict[str, np.ndarray],
    *,
    percentile: tuple[int, int] = (5, 95),
    bands: bool = True,
    refline: float | tuple[float, ...] | None = None,
    ylabel: str | None = None,
    xlabel: str = "t [s]",
) -> None:
    """Per-arm: plot median over time, optionally with a percentile band.
    `traces_per_arm[k]` is an [N, T] array of N rollouts x T timesteps."""
    for i, (k, tr) in enumerate(traces_per_arm.items()):
        c = colour_for(k, i)
        ax.plot(t, np.median(tr, axis=0), color=c, label=label_for(k))
        if bands:
            ax.fill_between(
                t,
                np.percentile(tr, percentile[0], axis=0),
                np.percentile(tr, percentile[1], axis=0),
                color=c,
                alpha=0.15,
            )
    if refline is not None:
        for rf in refline if isinstance(refline, tuple) else (refline,):
            ax.axhline(rf, ls="--", c="k", lw=1)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid(alpha=0.3)


def per_arm_scatter(
    ax: Axes,
    xs_per_arm: dict[str, np.ndarray],
    ys_per_arm: dict[str, np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    refline_x: float | None = None,
    refline_y: float | None = None,
    s: int = 6,
    alpha: float = 0.25,
) -> None:
    """Per-arm joint scatter."""
    for i, k in enumerate(xs_per_arm):
        ax.scatter(
            xs_per_arm[k],
            ys_per_arm[k],
            s=s,
            alpha=alpha,
            color=colour_for(k, i),
            label=label_for(k),
        )
    if refline_x is not None:
        ax.axvline(refline_x, ls="--", c="k", lw=1)
    if refline_y is not None:
        ax.axhline(refline_y, ls="--", c="k", lw=1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    ax.grid(alpha=0.3)


def violin_grid(
    properties: list[str],
    data_per_property_per_arm: dict[str, dict[str, np.ndarray]],
    *,
    arms_order: list[str],
    ncols: int | None = None,
    figsize_per_panel: tuple[float, float] = (4.0, 3.5),
    suptitle: str | None = None,
    refline: float | None = 0.0,
    ylabel: str | None = None,
) -> Figure:
    """Grid of violin plots, one panel per property; each panel shows
    one violin per arm. Returns the figure (caller closes via save_figure)."""
    n = len(properties)
    if ncols is None:
        ncols = (n + 1) // 2
    nrows = (n + ncols - 1) // ncols
    fig, grid = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes = grid.flatten()
    for i, prop in enumerate(properties):
        ax = axes[i]
        data = [data_per_property_per_arm[prop][arm] for arm in arms_order]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(
            range(1, len(arms_order) + 1),
            [label_for(a) for a in arms_order],
        )
        if refline is not None:
            ax.axhline(refline, ls="--", c="k", lw=1)
        ax.set_title(prop, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if ylabel is not None:
        for r in range(nrows):
            grid[r, 0].set_ylabel(ylabel)
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def crown_vs_mc_panel(
    ax: Axes,
    t: np.ndarray,
    crown_lo: np.ndarray,
    crown_hi: np.ndarray,
    mc_lo: np.ndarray,
    mc_hi: np.ndarray,
    mc_worst: np.ndarray,
    *,
    arm: str,
    fallback_index: int = 0,
    title: str | None = None,
    ylabel: str = "D_rel - D_safe [m]",
) -> None:
    """One panel of the formal-vs-empirical figure: a CROWN reachable
    band (in arm colour), an MC 5-95% band (grey), and the MC worst
    trace (dashed black). Safety threshold at 0."""
    c = colour_for(arm, fallback_index)
    ax.fill_between(t, crown_lo, crown_hi, color=c, alpha=0.18, label="CROWN reachable")
    ax.fill_between(t, mc_lo, mc_hi, color="k", alpha=0.15, label="MC 5-95%")
    ax.plot(t, mc_worst, "k--", lw=1, label="MC worst")
    ax.axhline(0, c="r", lw=1)
    ax.set_title(title or f"{label_for(arm)}: safety margin")
    ax.set_xlabel("t [s]")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
