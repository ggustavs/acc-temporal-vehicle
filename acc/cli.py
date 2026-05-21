"""Top-level dispatcher for the `acc` CLI."""

import typer

from acc import constants as C

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ACC case study commands.",
)

train_app = typer.Typer(
    no_args_is_help=True, help="Train a controller from a Vehicle spec."
)
tune_app = typer.Typer(
    no_args_is_help=True, help="Hyperparameter search via Optuna."
)
evaluate_app = typer.Typer(
    no_args_is_help=True, help="Per-property evaluation (centre + corners + PGD + CROWN)."
)
validate_app = typer.Typer(
    no_args_is_help=True,
    help="Vehicle `safe` per state + immrax CROWN cross-check.",
)
analyze_app = typer.Typer(
    no_args_is_help=True, help="Model-description analyses (figures + metrics)."
)

app.add_typer(train_app, name="train")
app.add_typer(tune_app, name="tune")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(validate_app, name="validate")
app.add_typer(analyze_app, name="analyze")


@app.callback()
def _root() -> None:
    pass


@app.command()
def status() -> None:
    """Print key paths and confirm the package is importable."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title="ACC case study status")
    table.add_column("Path")
    table.add_column("Exists")
    for label, path in [
        ("PUBLISHED_ONNX", C.PUBLISHED_ONNX),
        ("STL_CHECKPOINT_PATH", C.STL_CHECKPOINT_PATH),
        ("SFO_CHECKPOINT_PATH", C.SFO_CHECKPOINT_PATH),
        ("ACC_SPEC_PATH", C.ACC_SPEC_PATH),
        ("ACC_SFO_SPEC_PATH", C.ACC_SFO_SPEC_PATH),
    ]:
        table.add_row(f"{label} ({path})", "yes" if path.exists() else "no")
    Console().print(table)


def _register_commands() -> None:
    # Each import runs the module's @<verb>_app.command decorator.
    # Deferred to break the circular import: command modules do
    # `from acc.cli import <verb>_app`.
    from acc.commands import (  # noqa: F401
        analyze,
        compare,
        evaluate,
        train_sfo,
        train_stl,
        tune_sfo,
        tune_stl,
        validate,
    )


def main() -> None:
    _register_commands()
    app()


if __name__ == "__main__":
    main()
