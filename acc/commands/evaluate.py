"""`acc evaluate <arm>`: per-property evaluation; writes `{arm}_metrics.json`."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from torch import nn
from vehicle_lang.typing import Target

from acc import constants as C
from acc.cli import evaluate_app
from acc.controller import fresh_controller, load_checkpoint, published_controller
from acc.eval import EvalResult, evaluate
from acc.io import dump_eval_metrics_json
from acc.presenters import render_eval_summary

_CHECKPOINTS = {
    "published": ("Published ONNX", "onnx"),
    "stl": ("STL fine-tune", C.STL_CHECKPOINT_PATH),
    "sfo": ("SFO fine-tune", C.SFO_CHECKPOINT_PATH),
    "stl_finetuned": ("STL fine-tune (slow lead)", C.STL_FINETUNED_PATH),
    "sfo_finetuned": ("SFO fine-tune (slow lead)", C.SFO_FINETUNED_PATH),
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


def _make_command(which: str, doc: str):
    def cmd(
        out_dir: Path = typer.Option(C.RESULTS_DIR),
        logic: str = typer.Option(C.DEFAULT_LOGIC_NAME, help=C.LOGIC_OPTION_HELP),
    ) -> None:
        evaluate_core(which=which, out_dir=out_dir, logic=C.to_logic(logic))

    cmd.__doc__ = doc
    return cmd


evaluate_app.command(name="published")(
    _make_command("published", "Evaluate the published ARCH-COMP ONNX baseline.")
)
evaluate_app.command(name="stl")(
    _make_command("stl", "Evaluate the STL-trained controller checkpoint.")
)
evaluate_app.command(name="sfo")(
    _make_command("sfo", "Evaluate the SFO-trained controller checkpoint.")
)
evaluate_app.command(name="stl_finetuned")(
    _make_command("stl_finetuned", "Evaluate the slow-lead STL fine-tuned checkpoint.")
)
evaluate_app.command(name="sfo_finetuned")(
    _make_command("sfo_finetuned", "Evaluate the slow-lead SFO fine-tuned checkpoint.")
)
