"""Train the controller against the Vehicle STL property loss (per-property GradNorm)."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from vehicle_lang.typing import Target

from acc import constants as C
from acc._vlang import load_specification
from acc.cli import train_app
from acc.controller import fresh_controller, load_checkpoint
from acc.dynamics import acc_dynamics_step
from acc.gradnorm import GuardedGradNormBalancer as GradNormBalancer
from acc.initial_set import sample_uniform, sample_uniform_box
from acc.io import dump_stl_history_csv
from acc.presenters import render_stl_summary


def train_stl_loop(
    *,
    spec_path: Path,
    out_path: Path,
    epochs: int,
    steps_per_epoch: int,
    lr: float,
    n_inits: int,
    history_path: Path,
    logic: Target = C.DIFFERENTIABLE_LOGIC,
    console: Optional[Console] = None,
    report_cb: Optional[Callable[[int, float], bool]] = None,
    warm_start_path: Optional[Path] = None,
    init_lo: Optional[np.ndarray] = None,
    init_hi: Optional[np.ndarray] = None,
) -> float:
    """`report_cb(epoch, mean_stl) -> True` aborts early."""
    console = console or Console()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(C.SEED)

    loss_dict = load_specification(spec_path, logic=logic)
    property_fns = {name: loss_dict[name] for name in C.PROPERTY_NAMES}
    console.log(
        f"Loaded {spec_path} with logic {logic} "
        f"(properties: {', '.join(C.PROPERTY_NAMES)})"
    )

    net = fresh_controller()
    if warm_start_path is not None:
        net.load_state_dict(load_checkpoint(warm_start_path))
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(C.SEED + 1)

    # Closures capture this list by reference; each step refills it.
    step_inits: list[torch.Tensor] = []

    def _sat(v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(v, beta=C.SATISFICE_BETA)

    balancer = GradNormBalancer(
        losses={
            name: (
                lambda fn=fn: torch.stack(
                    [
                        _sat(
                            fn(
                                controller=net,
                                dynamics=acc_dynamics_step,
                                initState=x0,
                            )
                        )
                        for x0 in step_inits
                    ]
                ).mean()
            )
            for name, fn in property_fns.items()
        },
        model=net,
    )

    history: list[tuple[int, float]] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("epoch {task.completed}/{task.total}"),
        TextColumn("stl={task.fields[stl]:.3f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("STL train", total=epochs, stl=0.0)
        for epoch in range(epochs):
            net.train()
            stl_total = 0.0
            for _ in range(steps_per_epoch):
                step_inits.clear()
                if init_lo is not None and init_hi is not None:
                    step_inits.extend(
                        sample_uniform_box(n_inits, init_lo, init_hi, generator=rng)
                    )
                else:
                    step_inits.extend(
                        sample_uniform(n_inits, factor=1.0, generator=rng)
                    )
                total, _per_task = balancer.step()

                optim.zero_grad()
                total.backward()
                optim.step()
                stl_total += total.item()

            mean_stl = stl_total / steps_per_epoch
            history.append((epoch, mean_stl))
            torch.save(net.state_dict(), out_path)
            progress.update(task, advance=1, stl=mean_stl)
            if report_cb is not None and report_cb(epoch, mean_stl):
                break

    dump_stl_history_csv(history_path, history)
    render_stl_summary(
        console,
        epochs=len(history),
        final_stl_loss=history[-1][1],
        checkpoint_path=str(out_path),
    )
    return history[-1][1]


@train_app.command(name="stl")
def train_stl(
    spec_path: Path = typer.Option(C.ACC_SPEC_PATH),
    out_path: Path = typer.Option(C.STL_CHECKPOINT_PATH),
    epochs: int = typer.Option(C.STL_EPOCHS),
    steps_per_epoch: int = typer.Option(C.STL_STEPS_PER_EPOCH),
    lr: float = typer.Option(C.STL_LR),
    n_inits: int = typer.Option(C.STL_BATCH_INITS),
    history_path: Path = typer.Option(C.RESULTS_DIR / "stl_history.csv"),
    logic: str = typer.Option(C.DEFAULT_LOGIC_NAME, help=C.LOGIC_OPTION_HELP),
    warm_start_path: Optional[Path] = typer.Option(
        None, help="Warm-start from this checkpoint instead of a fresh MLP."
    ),
    init_mode: str = typer.Option(
        "default",
        help="default = INITIAL_LO/HI; finetune = INITIAL_LO_FINETUNE/HI_FINETUNE.",
    ),
) -> None:
    """Train from random init using only the Vehicle STL property loss."""
    if init_mode == "finetune":
        init_lo: Optional[np.ndarray] = C.INITIAL_LO_FINETUNE
        init_hi: Optional[np.ndarray] = C.INITIAL_HI_FINETUNE
    elif init_mode == "default":
        init_lo = None
        init_hi = None
    else:
        raise typer.BadParameter(
            f"unknown --init-mode {init_mode!r}; pick from default | finetune"
        )
    train_stl_loop(
        spec_path=spec_path,
        out_path=out_path,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        n_inits=n_inits,
        history_path=history_path,
        logic=C.to_logic(logic),
        warm_start_path=warm_start_path,
        init_lo=init_lo,
        init_hi=init_hi,
    )
