"""`acc validate <arm>`: Vehicle `safe` per state + immrax CROWN cross-check."""

from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from torch import nn
from vehicle_lang.loss import pytorch as loss_pt

from acc import constants as C
from acc.cli import validate_app
from acc.controller import fresh_controller, load_checkpoint, published_controller
from acc.dynamics import acc_dynamics_step, constant_lead_dynamics_step
from acc.initial_set import centre_point, corner_points
from acc.io import dump_trajectories_csv
from acc.presenters import render_property_check_table, render_verifier_table
from acc.vehicle_check import load_property_checks, load_trajectory
from acc.verifier import verify_initial_box_invariants

_LOADERS: dict[str, Callable[[], nn.Module]] = {
    "published": lambda: published_controller(C.PUBLISHED_ONNX),
    "stl": lambda: _load_checkpoint(C.STL_CHECKPOINT_PATH),
    "sfo": lambda: _load_checkpoint(C.SFO_CHECKPOINT_PATH),
    "stl_finetuned": lambda: _load_checkpoint(C.STL_FINETUNED_PATH),
    "sfo_finetuned": lambda: _load_checkpoint(C.SFO_FINETUNED_PATH),
}


def _load_checkpoint(path: Path) -> nn.Module:
    net = fresh_controller()
    net.load_state_dict(load_checkpoint(path))
    return net


def _box_bounds(init_box: str):
    """Returns (init_lo, init_hi, dynamics_fn, plant_name)."""
    if init_box == "default":
        return C.INITIAL_LO, C.INITIAL_HI, acc_dynamics_step, "default"
    if init_box == "finetune":
        return (
            C.INITIAL_LO_FINETUNE, C.INITIAL_HI_FINETUNE,
            constant_lead_dynamics_step, "constant_lead",
        )
    raise typer.BadParameter(
        f"unknown --init-box {init_box!r}; pick from default | finetune"
    )


def validate_arm(
    arm: str,
    csv_out: Path,
    logic: str,
    init_box: str = "default",
) -> None:
    """Run Vehicle `safe` + immrax CROWN on the given arm's controller."""
    console = Console()
    if arm not in _LOADERS:
        raise typer.BadParameter(f"unknown arm '{arm}'; pick from {list(_LOADERS)}")

    net = _LOADERS[arm]()
    net.eval()

    init_lo, init_hi, dynamics_fn, plant = _box_bounds(init_box)

    console.log(f"Loading Vehicle spec (logic={logic}, init_box={init_box})...")
    declarations = loss_pt.load_specification(
        C.ACC_SPEC_PATH,
        logic=C.to_logic(logic),
        declarations=("safe", "trajectory"),
    )
    checks = load_property_checks(declarations, names=("safe",), dynamics_fn=dynamics_fn)
    rollout = load_trajectory(declarations, dynamics_fn=dynamics_fn)
    safe_check = checks["safe"]

    centre = centre_point(init_lo, init_hi).squeeze(0)
    centre_result = safe_check(net, centre)
    if not centre_result.passes:
        raise typer.Exit(code=2) from AssertionError(
            f"Vehicle reports the centre point unsafe (loss {centre_result.loss:+.3f}). "
            "Property is satisfied iff loss <= 0."
        )

    try:
        verifier_results = verify_initial_box_invariants(
            net, initial_lo=init_lo, initial_hi=init_hi, plant=plant
        )
    except NotImplementedError as exc:
        verifier_results = {}
        console.log(f"[yellow]Verifier inconclusive: {exc}[/yellow]")

    corners = corner_points(init_lo, init_hi)
    inits = [("centre", centre)] + [
        (f"corner-{i}", corners[i]) for i in range(corners.shape[0])
    ]
    rows = [(label, safe_check(net, x0)) for label, x0 in inits]

    render_property_check_table(
        console,
        rows,
        title=f"Vehicle-compiled `safe` per initial state ({arm})",
    )
    if verifier_results:
        render_verifier_table(console, verifier_results)

    dump_trajectories_csv(csv_out, rollout, net, inits)
    console.log(f"Wrote per-step CSV to {csv_out}")

    if not all(check.passes for _, check in rows):
        offenders = [label for label, check in rows if not check.passes]
        raise typer.Exit(code=1) from AssertionError(
            f"Vehicle reports unsafe at: {offenders}. "
            "Suspect dynamics sign, controller arch mismatch, or spec drift."
        )
    safe_verifier = verifier_results.get("safe")
    if safe_verifier is not None and not safe_verifier.verified_safe:
        console.log(
            "[yellow]Vehicle and immrax disagree at the set level; review the verifier note.[/yellow]"
        )


def _make_command(arm: str, doc: str):
    default_csv = C.RESULTS_DIR / f"{arm}.csv"

    def cmd(
        csv_out: Path = typer.Option(default_csv, help="Per-step CSV"),
        logic: str = typer.Option(C.DEFAULT_LOGIC_NAME, help=C.LOGIC_OPTION_HELP),
        init_box: str = typer.Option(
            "default",
            "--init-box",
            help="default = INITIAL_LO/HI; finetune = INITIAL_LO/HI_FINETUNE.",
        ),
    ) -> None:
        validate_arm(arm, csv_out, logic, init_box=init_box)

    cmd.__doc__ = doc
    return cmd


validate_app.command(name="published")(
    _make_command("published", "Validate the published ARCH-COMP ONNX baseline.")
)
validate_app.command(name="stl")(
    _make_command("stl", "Validate the STL-trained controller checkpoint.")
)
validate_app.command(name="sfo")(
    _make_command("sfo", "Validate the SFO-trained controller checkpoint.")
)
validate_app.command(name="stl_finetuned")(
    _make_command("stl_finetuned", "Validate the slow-lead STL fine-tuned checkpoint.")
)
validate_app.command(name="sfo_finetuned")(
    _make_command("sfo_finetuned", "Validate the slow-lead SFO fine-tuned checkpoint.")
)
