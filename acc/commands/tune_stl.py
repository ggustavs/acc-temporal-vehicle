"""Optuna sweep over STL hyperparameters."""

from pathlib import Path
from typing import Optional

import optuna
import typer
from rich.console import Console
from vehicle_lang.typing import Target

from acc import constants as C
from acc.cli import app
from acc.commands.train_stl import train_stl_loop
from acc.tune import (
    make_pruning_callback,
    print_best_trial,
    require_completed_trial,
    run_study,
    suggest_stl_params,
)


def tune_stl_core(
    *,
    n_trials: int,
    storage: Path,
    spec_path: Path,
    out_path: Path,
    epochs: int,
    steps_per_epoch: int,
    history_path: Path,
    logic: Target = C.DIFFERENTIABLE_LOGIC,
    max_n_inits: Optional[int] = None,
    console: Optional[Console] = None,
) -> optuna.Study:
    """Search STL hyperparameters with Optuna; re-train at the best params.

    Per-trial artifacts go next to the study DB (`storage.parent`), so an
    isolated `storage` keeps a run from polluting the shared study.
    `max_n_inits` clamps the applied `n_inits` (the sampled value is
    still recorded); the smoke profile uses it to fit the second-order
    GradNorm graph in memory. `None` = the real search space.
    """
    console = console or Console()
    trial_dir = storage.parent / "stl_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_stl_params(trial)
        if max_n_inits is not None:
            params["n_inits"] = min(params["n_inits"], max_n_inits)
        trial_ckpt = trial_dir / f"trial_{trial.number:04d}.pt"
        trial_curve = trial_dir / f"trial_{trial.number:04d}_history.csv"
        return train_stl_loop(
            spec_path=spec_path,
            out_path=trial_ckpt,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            history_path=trial_curve,
            logic=logic,
            console=console,
            report_cb=make_pruning_callback(trial),
            **params,
        )

    study = run_study(
        study_name="stl-tune",
        storage_path=storage,
        objective=objective,
        n_trials=n_trials,
        direction="minimize",
    )

    require_completed_trial(study)
    print_best_trial(console, study)
    best_params = dict(study.best_params)
    if max_n_inits is not None:
        best_params["n_inits"] = min(best_params["n_inits"], max_n_inits)
    console.log(f"Re-training STL with best params to {out_path}")
    train_stl_loop(
        spec_path=spec_path,
        out_path=out_path,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        history_path=history_path,
        logic=logic,
        console=console,
        **best_params,
    )
    return study


@app.command(name="tune-stl")
def tune_stl(
    n_trials: int = typer.Option(C.OPTUNA_N_TRIALS),
    storage: Path = typer.Option(C.OPTUNA_STORAGE_DIR / "stl.db"),
    spec_path: Path = typer.Option(C.ACC_SPEC_PATH),
    out_path: Path = typer.Option(C.STL_CHECKPOINT_PATH),
    epochs: int = typer.Option(C.STL_EPOCHS),
    steps_per_epoch: int = typer.Option(C.STL_STEPS_PER_EPOCH),
    history_path: Path = typer.Option(C.RESULTS_DIR / "stl_history.csv"),
    logic: str = typer.Option(
        C.DIFFERENTIABLE_LOGIC.name, help="DL: vehicle | dl2 | stl | qll"
    ),
    max_n_inits: Optional[int] = typer.Option(
        None, help="Clamp applied n_inits (smoke/memory-bounded runs)"
    ),
) -> None:
    """Search STL hyperparameters with Optuna; minimise final-epoch loss."""
    tune_stl_core(
        n_trials=n_trials,
        storage=storage,
        spec_path=spec_path,
        out_path=out_path,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        history_path=history_path,
        logic=C.resolve_logic(logic),
        max_n_inits=max_n_inits,
    )
