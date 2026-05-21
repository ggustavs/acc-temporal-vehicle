"""`acc compare`: aggregate per-controller `*_metrics.json` into a report."""

from __future__ import annotations

from pathlib import Path

import typer

from acc import constants as C
from acc.analysis.compare import compare_core
from acc.cli import app


@app.command(name="compare")
def compare_cmd(
    arms: str = typer.Option(
        "published,stl,sfo",
        "--arms",
        help="Comma-separated controller arms to compare (matching *_metrics.json).",
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
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    compare_core(arm_list, metrics_dir, out_dir, baseline=baseline)
