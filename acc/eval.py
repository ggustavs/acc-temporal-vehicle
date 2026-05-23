"""Per-property evaluation: centre, corners, PGD, CROWN. Values are losses (lower is better; pass iff loss <= 0)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn
from vehicle_lang.loss import pytorch as loss_pt
from vehicle_lang.typing import Target

from typing import Any

from acc import constants as C
from acc.dynamics import acc_dynamics_step
from acc.initial_set import centre_point, corner_points
from acc.vehicle_check import load_property_checks, load_trajectory
from acc.verifier import VerifierResult, verify_initial_box_invariants


@dataclass
class PropertyEvalResult:
    name: str
    centre_loss: float
    corner_losses: list[float]
    corners_all_pass: bool
    pgd_worst_loss: float
    pgd_passes: bool


@dataclass
class EvalResult:
    name: str
    per_property: dict[str, PropertyEvalResult]
    verifier_per_property: dict[str, VerifierResult]
    centre_traj: np.ndarray


def evaluate(
    controller: nn.Module,
    name: str,
    *,
    logic: Target = C.DIFFERENTIABLE_LOGIC,
    init_lo: np.ndarray | None = None,
    init_hi: np.ndarray | None = None,
    sfo_spec_path: Path | None = None,
    dynamics_fn: Any = acc_dynamics_step,
    plant: str = "default",
) -> EvalResult:
    if hasattr(controller, "eval"):
        controller.eval()

    stl_decls = loss_pt.load_specification(
        C.ACC_SPEC_PATH,
        logic=logic,
        declarations=(*C.PROPERTY_NAMES, "trajectory"),
    )
    checks = load_property_checks(stl_decls, dynamics_fn=dynamics_fn)
    rollout = load_trajectory(stl_decls, dynamics_fn=dynamics_fn)

    sampler = loss_pt.DefaultPyTorchSampler(
        num_samples=C.EVAL_PGD_RESTARTS,
        num_steps=C.EVAL_PGD_K,
        seed=C.SEED,
    )
    sfo_decls = loss_pt.load_specification(
        sfo_spec_path if sfo_spec_path is not None else C.ACC_SFO_SPEC_PATH,
        logic=logic,
        samplers={"x": sampler.get_loss},
        declarations=C.SFO_PROPERTY_NAMES,
    )

    centre = centre_point(init_lo, init_hi).squeeze(0)
    corners = corner_points(init_lo, init_hi)

    per_property: dict[str, PropertyEvalResult] = {}
    for prop_name, sfo_name in zip(C.PROPERTY_NAMES, C.SFO_PROPERTY_NAMES):
        check = checks[prop_name]
        centre_check = check(controller, centre)
        corner_checks = [check(controller, corners[i]) for i in range(corners.shape[0])]

        sfo_fn = sfo_decls[sfo_name]
        sfo_val = sfo_fn(controller=controller, dynamics=dynamics_fn)
        if sfo_val.dim() != 0:
            sfo_val = sfo_val.flatten()[0]
        pgd_loss = float(sfo_val.item())

        per_property[prop_name] = PropertyEvalResult(
            name=prop_name,
            centre_loss=centre_check.loss,
            corner_losses=[c.loss for c in corner_checks],
            corners_all_pass=all(c.passes for c in corner_checks),
            pgd_worst_loss=pgd_loss,
            pgd_passes=pgd_loss <= 0.0,
        )

    ver_lo = C.INITIAL_LO if init_lo is None else init_lo
    ver_hi = C.INITIAL_HI if init_hi is None else init_hi
    verifier_results = verify_initial_box_invariants(
        controller, initial_lo=ver_lo, initial_hi=ver_hi, plant=plant
    )
    centre_traj = rollout(controller, centre).cpu().numpy()

    return EvalResult(
        name=name,
        per_property=per_property,
        verifier_per_property=verifier_results,
        centre_traj=centre_traj,
    )
