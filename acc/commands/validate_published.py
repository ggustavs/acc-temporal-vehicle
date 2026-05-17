"""Validate the published ARCH-COMP ONNX: Vehicle `safe` + immrax CROWN cross-check."""

from pathlib import Path

import typer
from rich.console import Console

from vehicle_lang.loss import pytorch as loss_pt

from acc import constants as C
from acc.cli import app
from acc.controller import published_controller
from acc.initial_set import centre_point, corner_points
from acc.io import dump_trajectories_csv
from acc.presenters import render_property_check_table, render_verifier_table
from acc.vehicle_check import load_property_checks, load_trajectory
from acc.verifier import verify_initial_box_invariants


@app.command(name="validate-published")
def validate_published(
    onnx_path: Path = typer.Option(
        C.PUBLISHED_ONNX, help="Path to controller_5_20.onnx"
    ),
    csv_out: Path = typer.Option(C.RESULTS_DIR / "published.csv", help="Per-step CSV"),
    logic: str = typer.Option(
        C.DIFFERENTIABLE_LOGIC.name, help="DL: vehicle | dl2 | stl"
    ),
) -> None:
    """Cross-check the published ONNX's safety: Vehicle `safe` per state + immrax box."""
    console = Console()
    logic_enum = C.parse_logic(logic)

    net = published_controller(onnx_path)
    net.eval()

    console.log(f"Loading Vehicle spec (logic={logic_enum.name})...")
    declarations = loss_pt.load_specification(
        C.ACC_SPEC_PATH,
        logic=logic_enum,
        declarations=("safe", "trajectory"),
    )
    checks = load_property_checks(declarations, names=("safe",))
    rollout = load_trajectory(declarations)
    safe_check = checks["safe"]

    centre = centre_point().squeeze(0)
    centre_result = safe_check(net, centre)
    if not centre_result.passes:
        raise typer.Exit(code=2) from AssertionError(
            f"Vehicle reports the centre point unsafe (loss {centre_result.loss:+.3f}). "
            "Property is satisfied iff loss <= 0."
        )

    try:
        verifier_results = verify_initial_box_invariants(net)
    except NotImplementedError as exc:
        verifier_results = {}
        console.log(f"[yellow]Verifier inconclusive: {exc}[/yellow]")

    corners = corner_points()
    inits = [("centre", centre)] + [
        (f"corner-{i}", corners[i]) for i in range(corners.shape[0])
    ]
    rows = [(label, safe_check(net, x0)) for label, x0 in inits]

    render_property_check_table(
        console, rows, title="Vehicle-compiled `safe` per initial state"
    )
    if verifier_results:
        render_verifier_table(console, verifier_results)

    dump_trajectories_csv(csv_out, rollout, net, inits)
    console.log(f"Wrote per-step CSV to {csv_out}")

    # Exit code gated on `safe` only.
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
