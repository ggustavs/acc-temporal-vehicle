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
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help="Output directory. Defaults to results/deep_eval (default box) "
        "or results/deep_eval_finetune (finetune box).",
    ),
    checkpoint: Optional[list[str]] = typer.Option(
        None,
        "--checkpoint",
        help="Override an arm's checkpoint: arm=path (repeatable).",
    ),
    init_box: str = typer.Option(
        "default",
        "--init-box",
        help="default = INITIAL_LO/HI; finetune = INITIAL_LO/HI_FINETUNE. "
        "Both CROWN init set and MC sampling box.",
    ),
) -> None:
    """CROWN reachable-set bounds vs MC envelope; writes formal_vs_empirical.{png,json}."""
    arms = parse_checkpoint_kvs(checkpoint or [], default_arms())
    if out_dir is None:
        base = "deep_eval" if init_box == "default" else f"deep_eval_{init_box}"
        out_dir = C.RESULTS_DIR / base
    formal_vs_empirical_core(arms, out_dir, init_box=init_box)
