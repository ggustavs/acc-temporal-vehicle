"""Optuna glue for hyperparameter tuning of the STL / SFO pipelines."""

import os
from pathlib import Path
from typing import Any, Callable

import optuna
import optunahub
import torch
from optuna.pruners import MedianPruner
from rich.console import Console
from rich.table import Table


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


_PER_TRIAL_RSS_GB = 4.0
_BASE_RSS_GB = 1.0
_MEM_SAFETY = 0.75
_MAX_USEFUL_N_JOBS = 6


def _physical_cores() -> int:
    """Physical core count (not HT siblings); falls back to logical."""
    try:
        seen: set[tuple[str, str]] = set()
        phys = core = ""
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.startswith("physical id"):
                    phys = ln.split(":")[1].strip()
                elif ln.startswith("core id"):
                    core = ln.split(":")[1].strip()
                elif ln.strip() == "":
                    if phys and core:
                        seen.add((phys, core))
                    phys = core = ""
        if seen:
            return len(seen)
    except OSError:
        pass
    return os.cpu_count() or 1


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def recommended_n_jobs() -> int:
    """Min of physical cores, RAM budget, and the throughput knee."""
    mem_cap = int(
        (_mem_available_gb() * _MEM_SAFETY - _BASE_RSS_GB) / _PER_TRIAL_RSS_GB
    )
    return max(1, min(_physical_cores(), mem_cap, _MAX_USEFUL_N_JOBS))


def run_study(
    *,
    study_name: str,
    storage_path: Path,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    direction: str,
    n_jobs: int = 1,
) -> optuna.Study:
    """`n_jobs>1` runs trials in a thread pool. Compile-safe via `acc._vlang`."""
    if n_jobs > 1:
        # One torch thread per trial: parallelism is across trials.
        torch.set_num_threads(1)

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
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=pruned_handler)
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
