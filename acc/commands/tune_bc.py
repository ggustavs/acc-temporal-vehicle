"""Optuna sweep over BC hyperparameters."""

from pathlib import Path

import optuna
import typer
from rich.console import Console
from rich.table import Table

from acc import constants as C
from acc.cli import app
from acc.commands.train_bc import train_bc_loop
from acc.tune import make_pruning_callback, run_study, suggest_bc_params


@app.command(name="tune-bc")
def tune_bc(
    n_trials: int = typer.Option(C.OPTUNA_N_TRIALS),
    storage: Path = typer.Option(C.OPTUNA_STORAGE_DIR / "bc.db"),
    data_path: Path = typer.Option(C.MPC_DATA_PATH),
    out_path: Path = typer.Option(C.BC_CHECKPOINT_PATH),
    epochs: int = typer.Option(C.BC_EPOCHS),
    val_curve_path: Path = typer.Option(C.RESULTS_DIR / "bc_val_curve.csv"),
) -> None:
    """Search BC hyperparameters with Optuna; minimise validation MSE."""
    console = Console()

    trial_dir = C.OPTUNA_STORAGE_DIR / "bc_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_bc_params(trial)
        trial_ckpt = trial_dir / f"trial_{trial.number:04d}.pt"
        trial_curve = trial_dir / f"trial_{trial.number:04d}_curve.csv"
        return train_bc_loop(
            data_path=data_path,
            out_path=trial_ckpt,
            epochs=epochs,
            val_curve_path=trial_curve,
            console=console,
            report_cb=make_pruning_callback(trial),
            **params,
        )

    study = run_study(
        study_name="bc-tune",
        storage_path=storage,
        objective=objective,
        n_trials=n_trials,
        direction="minimize",
    )

    _print_best(console, study)
    console.log(f"Re-training BC with best params to {out_path}")
    train_bc_loop(
        data_path=data_path,
        out_path=out_path,
        epochs=epochs,
        val_curve_path=val_curve_path,
        console=console,
        **study.best_params,
    )


def _print_best(console: Console, study: optuna.Study) -> None:
    table = Table(title=f"Best trial: #{study.best_trial.number}")
    table.add_column("param")
    table.add_column("value")
    for k, v in study.best_params.items():
        table.add_row(k, repr(v))
    table.add_row("[bold]best value[/bold]", f"{study.best_value:.6f}")
    console.print(table)
