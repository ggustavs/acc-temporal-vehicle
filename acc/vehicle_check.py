"""Thin wrappers around Vehicle's compiled property checks and `trajectory`."""

from typing import Callable, Iterable, NamedTuple

import torch
from torch import Tensor, nn

from acc import constants as C
from acc.dynamics import acc_dynamics_step


class PropertyCheck(NamedTuple):
    """`passes` iff `loss <= 0`. `loss` is the raw Vehicle minimisation target."""

    passes: bool
    loss: float


def _make_check(decl: Callable) -> Callable[[nn.Module, Tensor], PropertyCheck]:
    def check(controller: nn.Module, x0: Tensor) -> PropertyCheck:
        with torch.no_grad():
            val = decl(
                controller=controller,
                dynamics=acc_dynamics_step,
                initState=x0,
            )
        loss = float(val.item() if isinstance(val, torch.Tensor) else val)
        return PropertyCheck(passes=loss <= 0.0, loss=loss)

    return check


def load_property_checks(
    declarations: dict[str, Callable],
    names: Iterable[str] = C.PROPERTY_NAMES,
) -> dict[str, Callable[[nn.Module, Tensor], PropertyCheck]]:
    return {name: _make_check(declarations[name]) for name in names}


def load_trajectory(
    declarations: dict[str, Callable],
) -> Callable[[nn.Module, Tensor], Tensor]:
    trajectory_fn = declarations["trajectory"]

    def rollout(controller: nn.Module, x0: Tensor) -> Tensor:
        with torch.no_grad():
            return trajectory_fn(
                controller=controller,
                dynamics=acc_dynamics_step,
                initState=x0,
            )

    return rollout
