"""Behaviour, Pareto, factor-stress, adversarial, Vehicle-robustness, ACC-scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console

from acc import constants as C
from acc.analysis._data import (
    CMAX,
    DT,
    SEED,
    T,
    VSET,
    iGE,
    iVE,
    iVL,
    iXE,
    iXL,
    lead_mode_for_box,
    load_arms,
    rollout,
    sample_box,
)
from acc.analysis._metrics import (
    accel_rms,
    comfort_peak,
    jerk_peak,
    jerk_rms,
    margins,
    margins_t,
    time_headway_rmse,
    time_in_saturation,
    vset_recovery_time,
)
from acc.analysis._plotting import (
    colour_for,
    label_for,
    per_arm_scatter,
    save_figure,
    time_series_band,
    violin_grid,
)
from acc.analysis.scenarios import lead_constant_v, lead_ramp_acc, lead_ramp_dec

N = 4000
FACTOR_VALUES: tuple[float, ...] = (1.0, 1.5, 2.0)


def envelopes_compute(
    arms: dict, *, init_box: str = "default"
) -> dict[str, np.ndarray]:
    x0 = sample_box(N, 1.0, SEED, box=init_box)
    lead = lead_mode_for_box(init_box)
    return {k: rollout(v, x0, lead).numpy() for k, v in arms.items()}


_ENVELOPE_PANELS = [
    ("Safety margin [m]", margins, 0.0),
    ("Ego acceleration [m/s$^2$]", lambda tr: tr[..., iGE], (-CMAX, CMAX)),
    ("Ego velocity [m/s]", lambda tr: tr[..., iVE], VSET),
]


def envelopes_render(trajs: dict[str, np.ndarray], fig_dir: Path) -> None:
    t = np.arange(T + 1) * DT
    fig, axes = plt.subplots(1, len(_ENVELOPE_PANELS), figsize=(15, 4.0))
    for ax, (ylabel, metric, refline) in zip(axes, _ENVELOPE_PANELS):
        time_series_band(
            ax,
            t,
            {k: metric(tr) for k, tr in trajs.items()},
            bands=False,
            ylabel=ylabel,
            refline=refline,
        )
    fig.suptitle(f"Median closed-loop response over {N} trajectories")
    fig.tight_layout()
    save_figure(fig, fig_dir, "behaviour_envelopes.png")


def envelopes_summarise(trajs: dict[str, np.ndarray]) -> dict:
    return {
        k: {
            "median_min_margin": float(np.median(margins(tr).min(1))),
            "min_margin_p1": float(np.percentile(margins(tr).min(1), 1)),
        }
        for k, tr in trajs.items()
    }


def pareto_compute(
    arms: dict, *, init_box: str = "default"
) -> dict[str, np.ndarray]:
    # Same rollouts as envelopes; recompute here so the analysis is
    # self-contained (cheap: 4000 inits, ~1 s per arm).
    x0 = sample_box(N, 1.0, SEED, box=init_box)
    lead = lead_mode_for_box(init_box)
    return {k: rollout(v, x0, lead).numpy() for k, v in arms.items()}


def pareto_render(trajs: dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    per_arm_scatter(
        ax,
        {k: margins(tr).min(1) for k, tr in trajs.items()},
        {k: comfort_peak(tr) for k, tr in trajs.items()},
        xlabel="min safety margin [m] (>=0 safe)",
        ylabel="peak |g_ego| [m/s$^2$] (<=2 comfortable)",
        refline_x=0.0,
        refline_y=CMAX,
    )
    ax.set_title("Per-init joint safety + peak comfort by arm")
    fig.tight_layout()
    save_figure(fig, fig_dir, "pareto_safety_comfort.png")


def pareto_summarise(trajs: dict[str, np.ndarray]) -> dict:
    return {
        k: {
            "min_margin_p1": float(np.percentile(margins(tr).min(1), 1)),
            "peak_accel_p99": float(np.percentile(comfort_peak(tr), 99)),
        }
        for k, tr in trajs.items()
    }


def factor_stress_compute(
    arms: dict, *, init_box: str = "default"
) -> dict[float, dict[str, np.ndarray]]:
    lead = lead_mode_for_box(init_box)
    return {
        f: {
            k: rollout(
                v, sample_box(N, f, SEED + int(f * 10), box=init_box), lead
            ).numpy()
            for k, v in arms.items()
        }
        for f in FACTOR_VALUES
    }


def factor_stress_render(_data: dict, _fig_dir: Path) -> None:
    pass  # No figure; the multi-factor story is reported in the table.


def factor_stress_summarise(
    per_factor: dict[float, dict[str, np.ndarray]],
) -> dict:
    out: dict = {}
    for f, trajs in per_factor.items():
        out[f"f={f}"] = {}
        for k, tr in trajs.items():
            mm = margins(tr).min(1)
            cp = comfort_peak(tr)
            out[f"f={f}"][k] = {
                "median_min_margin": float(np.median(mm)),
                "min_margin_p1": float(np.percentile(mm, 1)),
                "peak_accel_p99": float(np.percentile(cp, 99)),
                "peak_jerk_p99": float(np.percentile(jerk_peak(tr), 99)),
                "jerk_rms_median": float(np.median(jerk_rms(tr))),
                "accel_rms_median": float(np.median(accel_rms(tr))),
                "time_in_saturation_mean": float(time_in_saturation(tr).mean()),
                "time_headway_rmse_median": float(np.median(time_headway_rmse(tr))),
                "vset_rmse": float(
                    np.sqrt(((tr[:, T // 2 :, iVE] - VSET) ** 2).mean())
                ),
            }
    return out


_ADV_PGD_STEPS = (0, 1, 5, 10, 20)
_ADV_N = 512
_ADV_LR = 0.10


def adversarial_compute(arms: dict, *, init_box: str = "default") -> dict:
    free = [0, 1, 3, 4]  # x_lead, v_lead, x_ego, v_ego (free state dims)
    if init_box == "finetune":
        box_lo, box_hi = C.INITIAL_LO_FINETUNE, C.INITIAL_HI_FINETUNE
    else:
        box_lo, box_hi = C.INITIAL_LO, C.INITIAL_HI
    lead = lead_mode_for_box(init_box)
    lo = torch.tensor(box_lo, dtype=torch.float32)
    hi = torch.tensor(box_hi, dtype=torch.float32)
    base = sample_box(_ADV_N, 1.0, SEED + 7, box=init_box)
    sweep: dict[str, list[float]] = {}
    worst_trajs: dict[str, np.ndarray] = {}
    for k, net in arms.items():
        curve: list[float] = []
        # Last entry of _ADV_PGD_STEPS always runs the deepest attack,
        # so worst_init is guaranteed to be set by loop end.
        worst_init = base[:1].clone()
        for steps in _ADV_PGD_STEPS:
            x = base.clone()
            x.requires_grad_(True)
            for _ in range(steps):
                mm = margins_t(rollout(net, x, lead, grad=True))
                loss = mm.min(1).values.sum()
                (g,) = torch.autograd.grad(loss, x)
                with torch.no_grad():
                    step_sz = _ADV_LR * (hi - lo)
                    x = x.clone()
                    x[:, free] = x[:, free] - step_sz[free] * torch.sign(g[:, free])
                    x[:, free] = torch.clamp(x[:, free], lo[free], hi[free])
                x.requires_grad_(True)
            with torch.no_grad():
                mmin = margins(rollout(net, x.detach(), lead).numpy()).min(1)
            curve.append(float(mmin.min()))
            if steps == _ADV_PGD_STEPS[-1]:
                wi = int(mmin.argmin())
                worst_init = x.detach()[wi : wi + 1]
        sweep[k] = curve
        worst_trajs[k] = rollout(net, worst_init, lead).numpy()[0]
    return {"sweep": sweep, "worst_trajs": worst_trajs}


def adversarial_render(data: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (k, curve) in enumerate(data["sweep"].items()):
        ax.plot(_ADV_PGD_STEPS, curve, "o-", color=colour_for(k, i), label=label_for(k))
    ax.axhline(0, ls="--", c="k", lw=1)
    ax.set(
        xlabel="PGD steps on initial state",
        ylabel="worst min margin [m]",
        title=f"Adversarial safety degradation, training plant, N={_ADV_N}",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, fig_dir, "adversarial_sweep.png")

    tt = np.arange(T + 1) * DT
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Panel 1: per-arm dRel + per-arm dSafe reference (dSafe depends
    # on each arm's v_ego, so it's a per-arm curve, not a constant).
    for i, (k, tr) in enumerate(data["worst_trajs"].items()):
        c = colour_for(k, i)
        d_rel = tr[:, iXL] - tr[:, iXE]
        d_safe = C.D_DEFAULT + C.T_GAP * tr[:, iVE]
        axes[0].plot(tt, d_rel, color=c, label=f"dRel ({label_for(k)})")
        axes[0].plot(
            tt, d_safe, color=c, lw=1.0, ls="--", label=f"dSafe ({label_for(k)})"
        )
    axes[0].set(xlabel="t [s]", ylabel="dRel, dSafe [m]")
    # Panel 2: ego accel.
    for i, (k, tr) in enumerate(data["worst_trajs"].items()):
        axes[1].plot(tt, tr[:, iGE], color=colour_for(k, i), label=label_for(k))
    for b in (-CMAX, CMAX):
        axes[1].axhline(b, ls="--", c="k", lw=0.5)
    axes[1].set(xlabel="t [s]", ylabel="Ego acceleration [m/s$^2$]")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Adversarial worst-init trajectory under training lead profile")
    fig.tight_layout()
    save_figure(fig, fig_dir, "adversarial_worst_traj.png")


def adversarial_summarise(data: dict) -> dict:
    return {
        k: {
            "worst_min_margin_by_steps": dict(
                zip([str(s) for s in _ADV_PGD_STEPS], curve)
            )
        }
        for k, curve in data["sweep"].items()
    }


_VR_N = 192


def vehicle_robustness_compute(
    arms: dict, *, init_box: str = "default"
) -> dict[str, dict[str, np.ndarray]]:
    # Import lazily; pulls in Vehicle's Python bindings + spec compile.
    from acc._vlang import load_specification
    from acc.vehicle_check import load_property_checks

    decls = load_specification(
        C.ACC_SPEC_PATH,
        logic=C.DIFFERENTIABLE_LOGIC,
        declarations=(*C.PROPERTY_NAMES, "trajectory"),
    )
    checks = load_property_checks(decls)
    sub = sample_box(_VR_N, 1.0, SEED + 3, box=init_box)
    out: dict[str, dict[str, np.ndarray]] = {}
    for prop in C.PROPERTY_NAMES:
        out[prop] = {}
        for k, net in arms.items():
            out[prop][k] = np.array(
                [checks[prop](net, sub[i]).loss for i in range(_VR_N)]
            )
    return out


def vehicle_robustness_render(
    data: dict[str, dict[str, np.ndarray]], fig_dir: Path
) -> None:
    arms_order = list(next(iter(data.values())).keys())
    fig = violin_grid(
        list(C.PROPERTY_NAMES),
        data,
        arms_order=arms_order,
        suptitle=f"Per-property quantitative robustness, N={_VR_N} box samples",
        ylabel="Vehicle loss (<=0 pass)",
        refline=0.0,
    )
    save_figure(fig, fig_dir, "vehicle_robustness_violin.png")


def vehicle_robustness_summarise(
    data: dict[str, dict[str, np.ndarray]],
) -> dict:
    return {
        prop: {
            k: {
                "median_loss": float(np.median(losses)),
                "p99_loss": float(np.percentile(losses, 99)),
            }
            for k, losses in arms_losses.items()
        }
        for prop, arms_losses in data.items()
    }


_ACC_SCENARIOS: dict[str, dict[str, Any]] = {
    "steady_follow_v22": {
        # 25 m slack; 15 m is below the comfort-bounded ACC feasibility floor.
        "desc": "v_set=30, lead settles to 22 m/s -- gap regulation under a slower lead",
        "v_set": 30.0,
        "v_ego0": 30.0,
        "v_lead0": 22.0,
        "lead_fn": lead_constant_v(22.0),
        "release_step": 0,
        "init_slack": 25.0,
    },
    "steady_follow_v25": {
        "desc": "v_set=30, lead settles to 25 m/s -- moderate-slower lead",
        "v_set": 30.0,
        "v_ego0": 30.0,
        "v_lead0": 25.0,
        "lead_fn": lead_constant_v(25.0),
        "release_step": 0,
    },
    "steady_follow_v28": {
        "desc": "v_set=30, lead settles to 28 m/s -- mild gap regulation",
        "v_set": 30.0,
        "v_ego0": 30.0,
        "v_lead0": 28.0,
        "lead_fn": lead_constant_v(28.0),
        "release_step": 0,
    },
    "lead_pullaway": {
        "desc": "v_set=30, ego following lead at 25, lead accels +1.5 m/s$^2$ -- pullaway/recovery",
        "v_set": 30.0,
        "v_ego0": 25.0,
        "v_lead0": 25.0,
        "lead_fn": lead_ramp_acc(1.5, 1.0, 5.0),
        "release_step": int(round(1.0 / DT)),
    },
    "lead_ramp_dec": {
        "desc": "lead at 30, decels -1.2 m/s$^2$ over t=1-3 s -- smooth following",
        "v_set": 30.0,
        "v_ego0": 30.0,
        "v_lead0": 30.0,
        "lead_fn": lead_ramp_dec(-1.2, 1.0, 3.0),
        "release_step": 0,
    },
}


def acc_scenarios_compute(
    arms: dict, *, init_box: str = "default"  # noqa: ARG001 -- scenes are box-agnostic
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, sc in _ACC_SCENARIOS.items():
        d_safe0 = C.D_DEFAULT + C.T_GAP * max(sc["v_ego0"], 0.0)
        x_ego0 = 0.0
        slack = float(sc.get("init_slack", 15.0))
        x_lead0 = x_ego0 + d_safe0 + slack
        x0 = torch.tensor(
            [[x_lead0, sc["v_lead0"], 0.0, x_ego0, sc["v_ego0"], 0.0]],
            dtype=torch.float32,
        )
        trajs: dict[str, np.ndarray] = {}
        for k, net in arms.items():
            tr = rollout(net, x0, sc["lead_fn"], v_set=sc["v_set"]).numpy()
            trajs[k] = tr[0]
        out[name] = {
            "desc": sc["desc"],
            "v_set": sc["v_set"],
            "release_step": sc["release_step"],
            "trajs": trajs,
        }
    return out


_SCENARIO_COL_TITLES = (
    "Lead velocity [m/s]",
    "Ego acceleration [m/s$^2$]",
    "dRel, dSafe [m]",
)


def acc_scenarios_render(data: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    scenarios = list(data.keys())
    n_scn = len(scenarios)
    tt = np.arange(T + 1) * DT

    # Gridspec layout: one thin header row + one panel row per scenario.
    # Header height is small relative to the panel rows.
    fig = plt.figure(figsize=(13, 3.6 * n_scn + 0.4))
    gs = fig.add_gridspec(
        2 * n_scn,
        3,
        height_ratios=[0.18, 1.0] * n_scn,
        hspace=0.45,
        wspace=0.28,
    )

    axes = np.empty((n_scn, 3), dtype=object)
    for r in range(n_scn):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[2 * r + 1, c])

    for r, scn in enumerate(scenarios):
        trs = data[scn]["trajs"]
        first_tr = next(iter(trs.values()))
        # Col 0: scenario context, v_lead only.
        axes[r, 0].plot(tt, first_tr[:, iVL], color="k", lw=1.8)
        # Cols 1, 2: per-arm response.
        for i, (arm, tr) in enumerate(trs.items()):
            c = colour_for(arm, i)
            axes[r, 1].plot(tt, tr[:, iGE], color=c, label=label_for(arm))
            d_rel = tr[:, iXL] - tr[:, iXE]
            d_safe = C.D_DEFAULT + C.T_GAP * tr[:, iVE]
            axes[r, 2].plot(tt, d_rel, color=c, label=f"dRel ({label_for(arm)})")
            axes[r, 2].plot(
                tt, d_safe, color=c, lw=1.0, ls="--", label=f"dSafe ({label_for(arm)})"
            )
        for b in (-CMAX, CMAX):
            axes[r, 1].axhline(b, ls="--", c="k", lw=0.5)
        for c_idx in range(3):
            axes[r, c_idx].grid(alpha=0.3)
        axes[r, 0].set_ylabel(_SCENARIO_COL_TITLES[0])
        axes[r, 1].set_ylabel(_SCENARIO_COL_TITLES[1])
        axes[r, 2].set_ylabel(_SCENARIO_COL_TITLES[2])
        axes[r, 1].legend(fontsize=7, loc="best")
        axes[r, 2].legend(fontsize=6, ncol=2, loc="best")

        # Row header: dedicated axis spanning all three columns above
        # this row's panels, no frame, just centred text.
        header_ax = fig.add_subplot(gs[2 * r, :])
        header_ax.axis("off")
        header_ax.text(
            0.5,
            0.5,
            data[scn]["desc"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            transform=header_ax.transAxes,
        )

    for c_idx, ct in enumerate(_SCENARIO_COL_TITLES):
        axes[0, c_idx].set_title(ct, fontsize=10)
    for c_idx in range(3):
        axes[-1, c_idx].set_xlabel("t [s]")

    fig.suptitle(
        "ACC scenarios -- rows: scenarios, cols: lead context / ego response",
        fontsize=11,
        y=0.995,
    )
    save_figure(fig, fig_dir, "acc_scenarios_traj.png")


def acc_scenarios_summarise(
    data: dict[str, dict[str, Any]],
) -> dict:
    out: dict = {}
    for scn, sc in data.items():
        out[scn] = {"desc": sc["desc"]}
        for arm, tr in sc["trajs"].items():
            mm = margins(tr[None])[0]
            rec = vset_recovery_time(
                tr[None],
                v_set=sc["v_set"],
                release_step=sc["release_step"],
                tol=0.5,
            )
            rec_val = float(rec[0]) if not np.isnan(rec[0]) else None
            out[scn][arm] = {
                "vset_recovery_steps": rec_val,
                "vset_recovery_seconds": (
                    rec_val * DT if rec_val is not None else None
                ),
                "min_margin": float(mm.min()),
                "final_v_ego": float(tr[-1, iVE]),
                "final_v_lead": float(tr[-1, iVL]),
                "final_dRel_dSafe": float(mm[-1]),
                "time_headway_rmse": float(np.median(time_headway_rmse(tr[None]))),
            }
    return out


ANALYSES: dict[str, tuple[Callable, Callable, Callable]] = {
    "envelopes": (envelopes_compute, envelopes_render, envelopes_summarise),
    "pareto": (pareto_compute, pareto_render, pareto_summarise),
    "factor_stress": (
        factor_stress_compute,
        factor_stress_render,
        factor_stress_summarise,
    ),
    "adversarial": (adversarial_compute, adversarial_render, adversarial_summarise),
    "vehicle_robustness": (
        vehicle_robustness_compute,
        vehicle_robustness_render,
        vehicle_robustness_summarise,
    ),
    "acc_scenarios": (
        acc_scenarios_compute,
        acc_scenarios_render,
        acc_scenarios_summarise,
    ),
}


def deep_eval_core(
    arms_to_paths: dict[str, str],
    out_dir: Path,
    console: Optional[Console] = None,
    init_mode: str = "nominal",
    init_box: str = "default",
) -> dict:
    if init_mode == "acc-sweep":
        from acc.analysis._acc_sweep import deep_eval_acc_sweep

        return deep_eval_acc_sweep(arms_to_paths, out_dir, console)
    if init_mode != "nominal":
        raise ValueError(f"unknown init_mode {init_mode!r}")

    console = console or Console()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    arms = load_arms(arms_to_paths)
    metrics: dict = {
        "N": N, "seed": SEED, "factors": list(FACTOR_VALUES),
        "init_box": init_box,
    }

    for name, (compute, render, summarise) in ANALYSES.items():
        data = compute(arms, init_box=init_box)
        render(data, fig_dir)
        metrics[name] = summarise(data)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    n_png = len(list(fig_dir.glob("*.png")))
    console.print(f"wrote {metrics_path} and {n_png} figures")
    for k, v in metrics["pareto"].items():
        console.print(
            f"{k} min_margin_p1={v['min_margin_p1']:.2f} "
            f"peak_accel_p99={v['peak_accel_p99']:.3f}"
        )
    return metrics
