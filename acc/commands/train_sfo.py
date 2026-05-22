"""Train the controller against the Vehicle SFO property loss (per-property GradNorm)."""

from pathlib import Path
from typing import Callable, Optional

import torch
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from vehicle_lang.loss import pytorch as loss_pt
from vehicle_lang.typing import Target

from acc import constants as C
from acc._vlang import load_specification
from acc.cli import train_app
from acc.controller import fresh_controller, load_checkpoint
from acc.dynamics import acc_dynamics_step
from acc.gradnorm import GuardedGradNormBalancer as GradNormBalancer
from acc.io import dump_sfo_history_csv
from acc.presenters import render_sfo_summary


def train_sfo_loop(
    *,
    spec_path: Path,
    out_path: Path,
    epochs: int,
    steps_per_epoch: int,
    lr: float,
    history_path: Path,
    pgd_num_steps: int = C.SFO_PGD_K,
    logic: Target = C.DIFFERENTIABLE_LOGIC,
    console: Optional[Console] = None,
    report_cb: Optional[Callable[[int, float], bool]] = None,
    warm_start_path: Optional[Path] = None,
) -> float:
    """`report_cb(epoch, mean_sfo) -> True` aborts early."""
    console = console or Console()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(C.SEED)

    sampler = loss_pt.DefaultPyTorchSampler(
        num_samples=C.SFO_PGD_RESTARTS, num_steps=pgd_num_steps
    )
    loss_dict = load_specification(
        spec_path,
        logic=logic,
        samplers={"x": sampler.get_loss},
    )
    property_fns = {name: loss_dict[name] for name in C.SFO_PROPERTY_NAMES}
    console.log(
        f"Loaded {spec_path} with logic {logic} "
        f"(properties: {', '.join(C.SFO_PROPERTY_NAMES)})"
    )

    net = fresh_controller()
    if warm_start_path is not None:
        net.load_state_dict(load_checkpoint(warm_start_path))
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    def _sat(v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(v, beta=C.SATISFICE_BETA)

    balancer = GradNormBalancer(
        losses={
            name: (lambda fn=fn: _sat(fn(controller=net, dynamics=acc_dynamics_step)))
            for name, fn in property_fns.items()
        },
        model=net,
    )

    history: list[tuple[int, float]] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("epoch {task.completed}/{task.total}"),
        TextColumn("sfo={task.fields[sfo]:.3f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("SFO train", total=epochs, sfo=0.0)
        for epoch in range(epochs):
            net.train()
            sfo_total = 0.0
            for _ in range(steps_per_epoch):
                total, _per_task = balancer.step()

                optim.zero_grad()
                total.backward()
                optim.step()
                sfo_total += total.item()

            mean_sfo = sfo_total / steps_per_epoch
            history.append((epoch, mean_sfo))
            torch.save(net.state_dict(), out_path)
            progress.update(task, advance=1, sfo=mean_sfo)
            if report_cb is not None and report_cb(epoch, mean_sfo):
                break

    dump_sfo_history_csv(history_path, history)
    render_sfo_summary(
        console,
        epochs=len(history),
        final_sfo_loss=history[-1][1],
        checkpoint_path=str(out_path),
    )
    return history[-1][1]


@train_app.command(name="sfo")
def train_sfo(
    spec_path: Path = typer.Option(C.ACC_SFO_SPEC_PATH),
    out_path: Path = typer.Option(C.SFO_CHECKPOINT_PATH),
    epochs: int = typer.Option(C.SFO_EPOCHS),
    steps_per_epoch: int = typer.Option(C.SFO_STEPS_PER_EPOCH),
    lr: float = typer.Option(C.SFO_LR),
    history_path: Path = typer.Option(C.RESULTS_DIR / "sfo_history.csv"),
    logic: str = typer.Option(C.DEFAULT_LOGIC_NAME, help=C.LOGIC_OPTION_HELP),
    warm_start_path: Optional[Path] = typer.Option(
        None, help="Warm-start from this checkpoint instead of a fresh MLP."
    ),
) -> None:
    """Train from random init using only the Vehicle SFO property loss."""
    train_sfo_loop(
        logic=C.to_logic(logic),
        spec_path=spec_path,
        out_path=out_path,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        history_path=history_path,
        warm_start_path=warm_start_path,
    )
