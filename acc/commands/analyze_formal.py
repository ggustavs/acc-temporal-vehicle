"""`acc analyze formal`: CROWN reachable-set bounds vs Monte-Carlo envelope."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from acc import constants as C
from acc.analysis._data import default_arms, parse_checkpoint_kvs
from acc.analysis.formal_vs_empirical import formal_vs_empirical_core
from acc.cli import analyze_app


@analyze_app.command(name="formal")
def analyze_formal(
    out_dir: Path = typer.Option(C.RESULTS_DIR / "deep_eval", "--out-dir"),
    checkpoint: Optional[list[str]] = typer.Option(
        None,
        "--checkpoint",
        help="Override an arm's checkpoint: arm=path (repeatable).",
    ),
) -> None:
    """CROWN reachable-set bounds vs MC envelope; writes formal_vs_empirical.{png,json}."""
    arms = parse_checkpoint_kvs(checkpoint or [], default_arms())
    formal_vs_empirical_core(arms, out_dir)
