"""`acc analyze deep`: behaviour envelopes, adversarial, robustness, ACC scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from acc import constants as C
from acc.analysis._data import default_arms, parse_checkpoint_kvs
from acc.analysis.deep_eval import deep_eval_core
from acc.cli import analyze_app


@analyze_app.command(name="deep")
def analyze_deep(
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help="Output directory. Defaults to results/deep_eval for nominal, "
        "results/deep_eval/acc_sweep for acc-sweep.",
    ),
    checkpoint: Optional[list[str]] = typer.Option(
        None,
        "--checkpoint",
        help="Override an arm's checkpoint: arm=path (repeatable).",
    ),
    init_mode: str = typer.Option(
        "nominal",
        "--init-mode",
        help="nominal = ARCH-COMP box; acc-sweep = 45-cell ACC operating-point grid.",
    ),
) -> None:
    """Behaviour envelopes + figures + per-property robustness across arms."""
    arms = parse_checkpoint_kvs(checkpoint or [], default_arms())
    if out_dir is None:
        out_dir = C.RESULTS_DIR / "deep_eval"
        if init_mode == "acc-sweep":
            out_dir = out_dir / "acc_sweep"
    deep_eval_core(arms, out_dir, init_mode=init_mode)
