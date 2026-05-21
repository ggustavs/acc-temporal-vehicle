"""Optuna sweep over SFO hyperparameters."""

from pathlib import Path
from typing import Optional

import optuna
import typer
from rich.console import Console
from vehicle_lang.typing import Target

from acc import constants as C
from acc.cli import tune_app
from acc.commands.train_sfo import train_sfo_loop
from acc.tune import (
    make_pruning_callback,
    print_best_trial,
    recommended_n_jobs,
    require_completed_trial,
    run_study,
    suggest_sfo_params,
)


def tune_sfo_core(
    *,
    n_trials: int,
    storage: Path,
    spec_path: Path,
    out_path: Path,
    epochs: int,
    steps_per_epoch: int,
    history_path: Path,
    logic: Target = C.DIFFERENTIABLE_LOGIC,
    max_pgd_steps: Optional[int] = None,
    console: Optional[Console] = None,
    n_jobs: int = 1,
) -> optuna.Study:
    """Search SFO hyperparameters with Optuna; re-train at the best params.

    Per-trial artifacts go next to the study DB (`storage.parent`), so an
    isolated `storage` keeps a run from polluting the shared study.
    `max_pgd_steps` clamps the applied PGD inner-loop length (the sampled
    value is still recorded); the smoke profile uses it to fit the
    second-order GradNorm graph in memory. `None` = the real search space.
    """
    console = console or Console()
    if n_jobs <= 0:
        n_jobs = recommended_n_jobs()
        console.log(f"Optuna concurrency: n_jobs={n_jobs} (memory-bounded auto)")
    trial_dir = storage.parent / "sfo_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_sfo_params(trial)
        if max_pgd_steps is not None:
            params["pgd_num_steps"] = min(params["pgd_num_steps"], max_pgd_steps)
        trial_ckpt = trial_dir / f"trial_{trial.number:04d}.pt"
        trial_curve = trial_dir / f"trial_{trial.number:04d}_history.csv"
        return train_sfo_loop(
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
        study_name="sfo-tune",
        storage_path=storage,
        objective=objective,
        n_trials=n_trials,
        direction="minimize",
        n_jobs=n_jobs,
    )

    require_completed_trial(study)
    print_best_trial(console, study)
    # Optuna records the PGD param under its registered name
    # `n_steps_pgd`; train_sfo_loop expects `pgd_num_steps`.
    bp = dict(study.best_params)
    retrain = {"lr": bp["lr"], "pgd_num_steps": bp["n_steps_pgd"]}
    if max_pgd_steps is not None:
        retrain["pgd_num_steps"] = min(retrain["pgd_num_steps"], max_pgd_steps)
    console.log(f"Re-training SFO with best params to {out_path}")
    train_sfo_loop(
        spec_path=spec_path,
        out_path=out_path,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        history_path=history_path,
        logic=logic,
        console=console,
        **retrain,
    )
    return study


@tune_app.command(name="sfo")
def tune_sfo(
    n_trials: int = typer.Option(C.OPTUNA_N_TRIALS),
    storage: Path = typer.Option(C.OPTUNA_STORAGE_DIR / "sfo.db"),
    spec_path: Path = typer.Option(C.ACC_SFO_SPEC_PATH),
    out_path: Path = typer.Option(C.SFO_CHECKPOINT_PATH),
    epochs: int = typer.Option(C.SFO_EPOCHS),
    steps_per_epoch: int = typer.Option(C.SFO_STEPS_PER_EPOCH),
    history_path: Path = typer.Option(C.RESULTS_DIR / "sfo_history.csv"),
    logic: str = typer.Option(C.DEFAULT_LOGIC_NAME, help=C.LOGIC_OPTION_HELP),
    max_pgd_steps: Optional[int] = typer.Option(
        None, help="Clamp applied PGD steps (smoke/memory-bounded runs)"
    ),
    n_jobs: int = typer.Option(
        0, help="Concurrent Optuna trials (threads); 0 = memory-bounded auto"
    ),
) -> None:
    """Search SFO hyperparameters with Optuna; minimise final-epoch loss."""
    tune_sfo_core(
        n_trials=n_trials,
        storage=storage,
        spec_path=spec_path,
        out_path=out_path,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        history_path=history_path,
        logic=C.to_logic(logic),
        max_pgd_steps=max_pgd_steps,
        n_jobs=n_jobs,
    )
