"""`acc analyze archcomp`: ARCH-COMP 2025 ACC benchmark scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from acc import constants as C
from acc.analysis._data import default_arms, parse_checkpoint_kvs
from acc.analysis.archcomp import archcomp_core
from acc.cli import analyze_app


@analyze_app.command(name="archcomp")
def analyze_archcomp(
    out_dir: Path = typer.Option(C.RESULTS_DIR, "--out-dir"),
    checkpoint: Optional[list[str]] = typer.Option(
        None,
        "--checkpoint",
        help="Override an arm's checkpoint: arm=path (repeatable).",
    ),
) -> None:
    """Score controllers under ARCH-COMP 2025 ACC's exact benchmark definition."""
    arms = parse_checkpoint_kvs(checkpoint or [], default_arms())
    archcomp_core(arms, out_dir)
