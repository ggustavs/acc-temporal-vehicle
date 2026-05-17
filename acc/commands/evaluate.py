"""Evaluate a trained or published controller; write per-property metrics JSON."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from torch import nn
from vehicle_lang.typing import Target

from acc import constants as C
from acc.cli import app
from acc.controller import fresh_controller, load_checkpoint, published_controller
from acc.eval import EvalResult, evaluate
from acc.io import dump_eval_metrics_json
from acc.presenters import render_eval_summary


_CHECKPOINTS = {
    "published": ("Published ONNX", "onnx"),
    "baseline": ("BC baseline", C.BC_CHECKPOINT_PATH),
    "stl": ("STL fine-tune", C.STL_CHECKPOINT_PATH),
    "sfo": ("SFO fine-tune", C.SFO_CHECKPOINT_PATH),
}


def _load(which: str) -> tuple[str, nn.Module]:
    label, source = _CHECKPOINTS[which]
    if source == "onnx":
        return label, published_controller(C.PUBLISHED_ONNX)
    net = fresh_controller()
    net.load_state_dict(load_checkpoint(source))
    return label, net


def evaluate_core(
    *,
    which: str,
    out_dir: Path,
    logic: Target,
    console: Optional[Console] = None,
) -> EvalResult:
    """Run the per-property evaluation suite and write `{which}_metrics.json`."""
    if which not in _CHECKPOINTS:
        raise ValueError(f"unknown which '{which}'; pick from {list(_CHECKPOINTS)}")
    console = console or Console()

    label, net = _load(which)
    result = evaluate(net, name=label, logic=logic)

    render_eval_summary(console, result)

    out_json = out_dir / f"{which}_metrics.json"
    dump_eval_metrics_json(out_json, result)
    console.log(f"Wrote {out_json}")
    return result


@app.command(name="evaluate")
def evaluate_command(
    which: str = typer.Argument(..., help="published | baseline | stl | sfo"),
    out_dir: Path = typer.Option(C.RESULTS_DIR),
    logic: str = typer.Option(
        C.DIFFERENTIABLE_LOGIC.name, help="DL: vehicle | dl2 | stl"
    ),
) -> None:
    """Run the per-property evaluation suite (centre + corners + PGD + CROWN)."""
    if which not in _CHECKPOINTS:
        raise typer.BadParameter(
            f"unknown which '{which}'; pick from {list(_CHECKPOINTS)}"
        )
    evaluate_core(which=which, out_dir=out_dir, logic=C.parse_logic(logic))
