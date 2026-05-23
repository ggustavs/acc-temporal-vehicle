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
        help="Output directory. Defaults to results/deep_eval (default box), "
        "results/deep_eval_finetune (finetune box), or appending /acc_sweep "
        "for the operating-point grid.",
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
    init_box: str = typer.Option(
        "default",
        "--init-box",
        help="default = INITIAL_LO/HI; finetune = INITIAL_LO/HI_FINETUNE. "
        "Affects the sampled-init analyses (envelopes, pareto, factor_stress, "
        "adversarial, vehicle_robustness); acc_scenarios is unaffected.",
    ),
) -> None:
    """Behaviour envelopes + figures + per-property robustness across arms."""
    arms = parse_checkpoint_kvs(checkpoint or [], default_arms())
    if out_dir is None:
        base = "deep_eval" if init_box == "default" else f"deep_eval_{init_box}"
        out_dir = C.RESULTS_DIR / base
        if init_mode == "acc-sweep":
            out_dir = out_dir / "acc_sweep"
    deep_eval_core(arms, out_dir, init_mode=init_mode, init_box=init_box)
