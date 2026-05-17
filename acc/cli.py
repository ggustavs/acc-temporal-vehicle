"""Top-level dispatcher for the `acc` CLI."""

import typer

from acc import constants as C

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ACC case study commands.",
)


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
        ("MPC_DATA_PATH", C.MPC_DATA_PATH),
        ("BC_CHECKPOINT_PATH", C.BC_CHECKPOINT_PATH),
        ("STL_CHECKPOINT_PATH", C.STL_CHECKPOINT_PATH),
        ("SFO_CHECKPOINT_PATH", C.SFO_CHECKPOINT_PATH),
        ("ACC_SPEC_PATH", C.ACC_SPEC_PATH),
        ("ACC_SFO_SPEC_PATH", C.ACC_SFO_SPEC_PATH),
    ]:
        table.add_row(f"{label} ({path})", "yes" if path.exists() else "no")
    Console().print(table)


def _register_commands() -> None:
    pass


_register_commands()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
