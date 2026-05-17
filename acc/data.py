"""MPC dataset loader with trajectory-level train/val split."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from acc import constants as C


def load_mpc_dataset(
    path: Path = C.MPC_DATA_PATH,
    *,
    val_fraction: float = C.BC_VAL_FRACTION,
    batch_size: int = C.BC_BATCH_SIZE,
    seed: int = C.SEED,
) -> tuple[DataLoader, DataLoader]:
    """Return train/val DataLoaders, split at the trajectory level."""
    raw = np.load(path)
    states = raw["states"]
    actions = raw["actions"]
    traj_ids = raw["traj_ids"]

    n_traj = int(traj_ids.max()) + 1
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_traj)
    n_val = max(1, int(val_fraction * n_traj))
    val_traj = set(perm[:n_val].tolist())

    val_mask = np.array([t in val_traj for t in traj_ids])
    train_mask = ~val_mask

    train_ds = TensorDataset(
        torch.from_numpy(states[train_mask]).float(),
        torch.from_numpy(actions[train_mask]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(states[val_mask]).float(),
        torch.from_numpy(actions[val_mask]).float(),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
