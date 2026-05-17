"""CSV / JSON dump helpers for trajectories, training histories, and eval metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from torch import Tensor, nn

from acc.eval import EvalResult


def dump_trajectories_csv(
    path: Path,
    rollout_fn: Callable[[nn.Module, Tensor], Tensor],
    net: nn.Module,
    labeled_inits: Sequence[tuple[str, Tensor]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["label", "step", "x_lead", "v_lead", "g_lead", "x_ego", "v_ego", "g_ego"]
        )
        for label, x0 in labeled_inits:
            traj = rollout_fn(net, x0).cpu().numpy()
            for step, state in enumerate(traj):
                w.writerow([label, step, *state.tolist()])


def dump_bc_history_csv(path: Path, rows: Iterable[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse"])
        for row in rows:
            w.writerow(row)


def dump_stl_history_csv(
    path: Path,
    rows: Iterable[tuple[int, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "stl_loss"])
        for row in rows:
            w.writerow(row)


def dump_sfo_history_csv(
    path: Path,
    rows: Iterable[tuple[int, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "sfo_loss"])
        for row in rows:
            w.writerow(row)


def dump_eval_metrics_json(path: Path, result: EvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": result.name,
                "per_property": {
                    name: {
                        "centre_loss": p.centre_loss,
                        "corner_losses": p.corner_losses,
                        "corners_all_pass": p.corners_all_pass,
                        "pgd_worst_loss": p.pgd_worst_loss,
                        "pgd_passes": p.pgd_passes,
                    }
                    for name, p in result.per_property.items()
                },
                "verifier": {
                    name: {"verified": v.verified_safe, "note": v.note}
                    for name, v in result.verifier_per_property.items()
                },
            },
            indent=2,
        )
    )
