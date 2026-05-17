"""Rich-table rendering for the ACC commands."""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.table import Table

from acc.eval import EvalResult
from acc.vehicle_check import PropertyCheck
from acc.verifier import VerifierResult


def render_property_check_table(
    console: Console,
    rows: Iterable[tuple[str, PropertyCheck]],
    title: str,
) -> None:
    table = Table(title=title)
    table.add_column("init")
    table.add_column("pass", justify="center")
    table.add_column("loss", justify="right")
    for label, check in rows:
        table.add_row(label, "PASS" if check.passes else "FAIL", f"{check.loss:+.4f}")
    console.print(table)


def render_verifier_table(
    console: Console,
    results: dict[str, VerifierResult],
    title: str = "immrax interval-reachability per invariant",
) -> None:
    table = Table(title=title)
    table.add_column("property")
    table.add_column("status")
    table.add_column("note")
    for name, status in results.items():
        cell = (
            "[green]verified[/green]"
            if status.verified_safe
            else "[red]not verified[/red]"
        )
        table.add_row(name, cell, status.note)
    console.print(table)


def render_eval_summary(console: Console, result: EvalResult) -> None:
    summary = Table(title=f"{result.name} - per-property evaluation (loss)")
    summary.add_column("property")
    summary.add_column("centre", justify="right")
    summary.add_column("corners max", justify="right")
    summary.add_column("corners pass", justify="center")
    summary.add_column("PGD worst", justify="right")
    summary.add_column("PGD pass", justify="center")
    for name, p in result.per_property.items():
        summary.add_row(
            name,
            f"{p.centre_loss:+.3f}",
            f"{max(p.corner_losses):+.3f}",
            "PASS" if p.corners_all_pass else "FAIL",
            f"{p.pgd_worst_loss:+.3f}",
            "PASS" if p.pgd_passes else "FAIL",
        )
    console.print(summary)
    render_verifier_table(
        console,
        result.verifier_per_property,
        title=f"{result.name} - CROWN verifier (invariants)",
    )


def render_bc_summary(
    console: Console,
    epochs: int,
    best_val: float,
    final_train: float,
    final_val: float,
    checkpoint_path: str,
) -> None:
    table = Table(title="BC training summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("epochs", str(epochs))
    table.add_row("best val MSE", f"{best_val:.6f}")
    table.add_row("final train MSE", f"{final_train:.6f}")
    table.add_row("final val MSE", f"{final_val:.6f}")
    table.add_row("checkpoint", checkpoint_path)
    console.print(table)


def render_stl_summary(
    console: Console,
    epochs: int,
    final_stl_loss: float,
    checkpoint_path: str,
) -> None:
    table = Table(title="STL fine-tune summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("epochs", str(epochs))
    table.add_row("final STL loss (mean)", f"{final_stl_loss:.3f}")
    table.add_row("checkpoint", checkpoint_path)
    console.print(table)


def render_sfo_summary(
    console: Console,
    epochs: int,
    final_sfo_loss: float,
    checkpoint_path: str,
) -> None:
    table = Table(title="SFO fine-tune summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("epochs", str(epochs))
    table.add_row("final SFO loss (mean)", f"{final_sfo_loss:.3f}")
    table.add_row("checkpoint", checkpoint_path)
    console.print(table)
