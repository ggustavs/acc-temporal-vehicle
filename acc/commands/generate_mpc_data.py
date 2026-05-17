"""Generate behaviour-cloning data via MPC oracle rollouts."""

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from acc import constants as C
from acc.cli import app
from acc.dynamics import acc_dynamics_step
from acc.initial_set import sample_uniform
from acc.mpc_oracle import MPCOracle


@app.command(name="generate-mpc-data")
def generate_mpc_data(
    n_trajectories: int = typer.Option(C.MPC_N_TRAJECTORIES),
    n_steps: int = typer.Option(C.N_STEPS),
    enlarge_factor: float = typer.Option(C.MPC_ENLARGE_FACTOR),
    out_path: Path = typer.Option(C.MPC_DATA_PATH),
    checkpoint_every: int = typer.Option(C.MPC_CHECKPOINT_EVERY),
) -> None:
    """Roll the MPC oracle from sampled init states; save (state, action) pairs."""
    console = Console()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.npz")

    oracle = MPCOracle()
    states_all = np.zeros((n_trajectories, n_steps, C.STATE_DIM), dtype=np.float32)
    actions_all = np.zeros((n_trajectories, n_steps, C.ACT_DIM), dtype=np.float32)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("MPC rollouts", total=n_trajectories)
        for traj_id in range(n_trajectories):
            rng = np.random.default_rng(C.SEED + traj_id)
            init = sample_uniform(1, factor=enlarge_factor, generator=rng)
            state = init
            for step in range(n_steps):
                action = oracle(state)
                states_all[traj_id, step] = state[0].numpy()
                actions_all[traj_id, step] = action[0].numpy()
                state = acc_dynamics_step(state, action)
            progress.advance(task)
            if (traj_id + 1) % checkpoint_every == 0:
                np.savez(
                    partial_path,
                    states=states_all[: traj_id + 1].reshape(-1, C.STATE_DIM),
                    actions=actions_all[: traj_id + 1].reshape(-1, C.ACT_DIM),
                    traj_id=np.array(traj_id + 1),
                )

    np.savez(
        out_path,
        states=states_all.reshape(-1, C.STATE_DIM),
        actions=actions_all.reshape(-1, C.ACT_DIM),
        traj_ids=np.repeat(np.arange(n_trajectories), n_steps),
    )
    if partial_path.exists():
        partial_path.unlink()

    rate = oracle.fallback_count / max(oracle.call_count, 1)
    console.log(
        f"Wrote {out_path} ({n_trajectories * n_steps} pairs); "
        f"IPOPT fallback {oracle.fallback_count}/{oracle.call_count} = {rate:.2%}"
    )
    if rate > 0.05:
        console.log(
            "[yellow]Fallback rate exceeds 5%; consider tuning MPC weights[/yellow]"
        )
