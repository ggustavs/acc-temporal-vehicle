"""Behaviour-cloning training of the fresh controller MLP on MPC data."""

from pathlib import Path
from typing import Callable, Optional

import torch
import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from torch import nn

from acc import constants as C
from acc.cli import app
from acc.controller import fresh_controller, state_to_observation
from acc.data import load_mpc_dataset
from acc.io import dump_bc_history_csv
from acc.presenters import render_bc_summary


def train_bc_loop(
    *,
    data_path: Path,
    out_path: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    val_curve_path: Path,
    console: Optional[Console] = None,
    report_cb: Optional[Callable[[int, float], bool]] = None,
) -> float:
    """`report_cb(epoch, val_loss) -> True` aborts early."""
    console = console or Console()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(C.SEED)
    train_loader, val_loader = load_mpc_dataset(
        data_path, batch_size=batch_size, seed=C.SEED
    )

    net = fresh_controller()
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history: list[tuple[int, float, float]] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("epoch {task.completed}/{task.total}"),
        TextColumn("train={task.fields[train]:.5f} val={task.fields[val]:.5f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("BC", total=epochs, train=0.0, val=0.0)
        for epoch in range(epochs):
            net.train()
            train_total = 0.0
            train_n = 0
            for states, actions in train_loader:
                pred = net(state_to_observation(states))
                loss = loss_fn(pred, actions)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_total += loss.item() * len(states)
                train_n += len(states)
            train_loss = train_total / train_n

            net.eval()
            with torch.no_grad():
                val_total = 0.0
                val_n = 0
                for states, actions in val_loader:
                    val_total += loss_fn(
                        net(state_to_observation(states)), actions
                    ).item() * len(states)
                    val_n += len(states)
                val_loss = val_total / val_n

            history.append((epoch, train_loss, val_loss))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(net.state_dict(), out_path)
            progress.update(task, advance=1, train=train_loss, val=val_loss)
            if report_cb is not None and report_cb(epoch, val_loss):
                break

    dump_bc_history_csv(val_curve_path, history)
    render_bc_summary(
        console,
        epochs=len(history),
        best_val=best_val,
        final_train=history[-1][1],
        final_val=history[-1][2],
        checkpoint_path=str(out_path),
    )
    return best_val


@app.command(name="train-bc")
def train_bc(
    data_path: Path = typer.Option(C.MPC_DATA_PATH),
    out_path: Path = typer.Option(C.BC_CHECKPOINT_PATH),
    epochs: int = typer.Option(C.BC_EPOCHS),
    lr: float = typer.Option(C.BC_LR),
    batch_size: int = typer.Option(C.BC_BATCH_SIZE),
    val_curve_path: Path = typer.Option(C.RESULTS_DIR / "bc_val_curve.csv"),
) -> None:
    """Train a fresh 5x20 ReLU controller against the MPC oracle data."""
    train_bc_loop(
        data_path=data_path,
        out_path=out_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        val_curve_path=val_curve_path,
    )
