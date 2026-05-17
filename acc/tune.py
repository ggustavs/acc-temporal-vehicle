"""Optuna glue for hyperparameter tuning of the BC / STL / SFO pipelines."""

from pathlib import Path
from typing import Any, Callable

import optuna
import optunahub
from optuna.pruners import MedianPruner
from rich.console import Console
from rich.table import Table


def suggest_bc_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
    }


def suggest_stl_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "n_inits": trial.suggest_categorical("n_inits", [4, 8, 16, 32]),
    }


def suggest_sfo_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "pgd_num_steps": trial.suggest_categorical("n_steps_pgd", [5, 10, 20]),
    }


def make_pruning_callback(trial: optuna.Trial) -> Callable[[int, float], bool]:
    """Per-epoch callback: reports the metric, returns True if Optuna prunes."""

    def cb(epoch: int, value: float) -> bool:
        trial.report(value, step=epoch)
        return trial.should_prune()

    return cb


def _auto_sampler() -> optuna.samplers.BaseSampler:
    module = optunahub.load_module("samplers/auto_sampler")
    return module.AutoSampler()


def run_study(
    *,
    study_name: str,
    storage_path: Path,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    direction: str,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=_auto_sampler(),
        pruner=MedianPruner(n_warmup_steps=2),
    )

    pruned_handler = (optuna.exceptions.TrialPruned,)
    study.optimize(objective, n_trials=n_trials, catch=pruned_handler)
    return study


def require_completed_trial(study: optuna.Study) -> None:
    """Fail legibly if no trial completed; `study.best_trial` would
    otherwise raise a cryptic 'Record does not exist'."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        states = [t.state.name for t in study.trials]
        raise RuntimeError(
            f"Optuna study {study.study_name!r} has no completed trial: "
            f"{len(study.trials)} attempted, states={states}. "
            "All failed/pruned (e.g. NaN objective); training is not "
            "converging at these settings, not re-training."
        )


def print_best_trial(console: Console, study: optuna.Study) -> None:
    table = Table(title=f"Best trial: #{study.best_trial.number}")
    table.add_column("param")
    table.add_column("value")
    for k, v in study.best_params.items():
        table.add_row(k, repr(v))
    table.add_row("[bold]best value[/bold]", f"{study.best_value:.6f}")
    console.print(table)
