"""`acc compare`: aggregate per-controller `*_metrics.json` into a report."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from acc import constants as C
from acc.analysis.compare import compare_core
from acc.cli import app

_ARMS_ORDER = ("published", "stl", "sfo", "stl_finetuned", "sfo_finetuned")


@app.command(name="compare")
def compare_cmd(
    arms: Optional[str] = typer.Option(
        None,
        "--arms",
        help="Comma-separated arms to compare. Default: every arm whose "
        "*_metrics.json exists in --metrics-dir.",
    ),
    metrics_dir: Path = typer.Option(
        C.RESULTS_DIR, "--metrics-dir", help="Where the *_metrics.json files live."
    ),
    out_dir: Path = typer.Option(
        C.RESULTS_DIR, "--out-dir", help="Where to write comparison.{json,md}."
    ),
    baseline: str = typer.Option(
        "published", "--baseline", help="Arm to compute centre-loss deltas against."
    ),
) -> None:
    """Per-property pass/fail table across arms; SATISFIED = corners+PGD+verified."""
    if arms is None:
        arm_list = [
            a for a in _ARMS_ORDER if (metrics_dir / f"{a}_metrics.json").exists()
        ]
        if not arm_list:
            raise typer.BadParameter(
                f"no *_metrics.json found in {metrics_dir}; run `acc evaluate` first"
            )
    else:
        arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    compare_core(arm_list, metrics_dir, out_dir, baseline=baseline)
