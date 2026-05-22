"""Initial-state set: centre, corners, and uniform sampling."""

from itertools import product

from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor

from acc import constants as C


_FREE_INDICES = (C.IDX_X_LEAD, C.IDX_V_LEAD, C.IDX_X_EGO, C.IDX_V_EGO)


def _bounds(factor: float) -> tuple[np.ndarray, np.ndarray]:
    centre = 0.5 * (C.INITIAL_LO + C.INITIAL_HI)
    half = 0.5 * (C.INITIAL_HI - C.INITIAL_LO)
    return centre - factor * half, centre + factor * half


def centre_point() -> Float[Tensor, "1 6"]:
    centre = 0.5 * (C.INITIAL_LO + C.INITIAL_HI)
    return torch.from_numpy(centre).float().unsqueeze(0)


def corner_points() -> Float[Tensor, "16 6"]:
    free = list(_FREE_INDICES)
    masks = np.array(list(product((0, 1), repeat=len(free))), dtype=bool)
    pts = np.tile(C.INITIAL_LO, (len(masks), 1))
    pts[:, free] = np.where(masks, C.INITIAL_HI[free], C.INITIAL_LO[free])
    return torch.from_numpy(pts).float()


def sample_uniform(
    n: int,
    *,
    factor: float = 1.0,
    generator: np.random.Generator | None = None,
) -> Float[Tensor, "n 6"]:
    rng = generator if generator is not None else np.random.default_rng(C.SEED)
    lo, hi = _bounds(factor)
    raw = rng.uniform(lo, hi, size=(n, C.STATE_DIM))
    raw[:, C.IDX_G_LEAD] = 0.0
    raw[:, C.IDX_G_EGO] = 0.0
    return torch.from_numpy(raw).float()


def sample_uniform_box(
    n: int,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    generator: np.random.Generator | None = None,
) -> Float[Tensor, "n 6"]:
    """Sample n initial states uniformly from per-dim bounds [lo, hi]."""
    rng = generator if generator is not None else np.random.default_rng(C.SEED)
    raw = rng.uniform(lo, hi, size=(n, C.STATE_DIM))
    raw[:, C.IDX_G_LEAD] = 0.0
    raw[:, C.IDX_G_EGO] = 0.0
    return torch.from_numpy(raw).float()
