"""Sequential Optuna tune-stl -> tune-sfo (-> evaluate) runner.

Each Optuna stage re-trains at its best params into the main checkpoint;
the optional evaluate stage runs `acc evaluate` per arm. Resumable via
the SQLite study storage.

    nohup uv run python scripts/run_pipeline.py --profile full \
        > run.out 2>&1 &

    uv run python scripts/run_pipeline.py --profile smoke   # local smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from acc import constants as C
from acc.commands.evaluate import evaluate_core
from acc.commands.tune_sfo import tune_sfo_core
from acc.commands.tune_stl import tune_stl_core


class _Tee:
    """Minimal write/flush fan-out so a Rich Console hits stdout + a log file."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _profile_params(profile: str) -> dict[str, dict[str, int]]:
    if profile == "smoke":
        common = {"n_trials": 2, "epochs": 1, "steps_per_epoch": 2}
        return {"stl": dict(common), "sfo": dict(common)}
    if profile == "full":
        return {
            "stl": {
                "n_trials": C.OPTUNA_N_TRIALS,
                "epochs": C.STL_EPOCHS,
                "steps_per_epoch": C.STL_STEPS_PER_EPOCH,
            },
            "sfo": {
                "n_trials": C.OPTUNA_N_TRIALS,
                "epochs": C.SFO_EPOCHS,
                "steps_per_epoch": C.SFO_STEPS_PER_EPOCH,
            },
        }
    raise ValueError(f"unknown profile {profile!r}")


def _stage_plan(
    profile: str, run_dir: Path, n_jobs: int
) -> dict[str, dict[str, Any]]:
    params = _profile_params(profile)
    # smoke uses an isolated study so it never touches the shared one.
    optuna_dir = (run_dir / "optuna") if profile == "smoke" else C.OPTUNA_STORAGE_DIR
    return {
        "tune-stl": {
            "n_trials": params["stl"]["n_trials"],
            "storage": optuna_dir / "stl.db",
            "spec_path": C.ACC_SPEC_PATH,
            "out_path": C.STL_CHECKPOINT_PATH,
            "epochs": params["stl"]["epochs"],
            "steps_per_epoch": params["stl"]["steps_per_epoch"],
            "history_path": C.RESULTS_DIR / "stl_history.csv",
            # smoke clamps n_inits to fit the GradNorm graph in memory;
            # None = full {4,8,16,32}.
            "max_n_inits": 4 if profile == "smoke" else None,
            "n_jobs": n_jobs,
        },
        "tune-sfo": {
            "n_trials": params["sfo"]["n_trials"],
            "storage": optuna_dir / "sfo.db",
            "spec_path": C.ACC_SFO_SPEC_PATH,
            "out_path": C.SFO_CHECKPOINT_PATH,
            "epochs": params["sfo"]["epochs"],
            "steps_per_epoch": params["sfo"]["steps_per_epoch"],
            "history_path": C.RESULTS_DIR / "sfo_history.csv",
            # smoke clamps the PGD inner loop (the memory driver);
            # None = full {5,10,20}.
            "max_pgd_steps": 5 if profile == "smoke" else None,
            "n_jobs": n_jobs,
        },
    }


def _eval_targets(args: argparse.Namespace) -> list[str]:
    targets: list[str] = ["stl", "sfo"]
    if args.with_published:
        targets.append("published")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--with-published", action="store_true")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Concurrent Optuna trials per stage (threads; shared in-process "
        "study, memory-frugal). 0 = memory-bounded auto.",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = C.RESULTS_DIR / "run_logs" / ts
    plan = _stage_plan(args.profile, run_dir, args.n_jobs)
    eval_targets = [] if args.no_eval else _eval_targets(args)
    logic = C.DIFFERENTIABLE_LOGIC
    logic_name = C.DEFAULT_LOGIC_NAME

    if args.dry_run:
        print(f"profile={args.profile} logic={logic_name} run_dir={run_dir}")
        for name, kwargs in plan.items():
            print(f"\n{name}:")
            for k, v in kwargs.items():
                print(f"  {k} = {v}")
        print(f"\nevaluate: {eval_targets or '(skipped)'}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "profile": args.profile,
        "logic": logic_name,
        "started": ts,
        "run_dir": str(run_dir),
        "stages": [],
    }
    manifest_path = run_dir / "manifest.json"

    def _save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    def _run(name: str, fn: Any) -> Any:
        log_path = run_dir / f"{name}.log"
        entry: dict[str, Any] = {"name": name, "status": "running"}
        manifest["stages"].append(entry)
        _save()
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as fh:
            console = Console(file=_Tee(sys.stdout, fh), force_terminal=False)  # pyright: ignore[reportArgumentType]
            try:
                result = fn(console)
            except Exception as exc:
                entry.update(
                    status="failed",
                    seconds=round(time.perf_counter() - t0, 2),
                    error=f"{type(exc).__name__}: {exc}",
                    log=str(log_path),
                )
                _save()
                console.print(f"[red]stage {name} FAILED[/red]")
                traceback.print_exc(file=sys.stderr)
                raise SystemExit(1) from exc
        entry.update(
            status="ok",
            seconds=round(time.perf_counter() - t0, 2),
            log=str(log_path),
        )
        _save()
        return result

    tune_cores = {"tune-stl": tune_stl_core, "tune-sfo": tune_sfo_core}
    for name in ("tune-stl", "tune-sfo"):
        kwargs = plan[name]
        study = _run(
            name,
            lambda c, k=kwargs, fn=tune_cores[name]: fn(console=c, logic=logic, **k),
        )
        st = manifest["stages"][-1]
        st["study_name"] = study.study_name
        st["storage"] = str(kwargs["storage"])
        st["n_trials"] = kwargs["n_trials"]
        st["n_trials_completed"] = len(study.trials)
        st["best_params"] = study.best_params
        st["best_value"] = study.best_value
        st["artifacts"] = {
            "checkpoint": str(kwargs["out_path"]),
            "history": str(kwargs["history_path"]),
        }
        _save()

    for which in eval_targets:
        _run(
            f"evaluate-{which}",
            lambda c, w=which: evaluate_core(
                which=w, out_dir=C.RESULTS_DIR, logic=logic, console=c
            ),
        )
        manifest["stages"][-1]["artifacts"] = {
            "metrics": str(C.RESULTS_DIR / f"{which}_metrics.json")
        }
        _save()

    manifest["ended"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["status"] = "ok"
    _save()
    print(f"\nDone. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
