"""Single source of truth for the ACC case study."""

from pathlib import Path

import numpy as np
from vehicle_lang.typing import CustomLogic, DifferentiableLogic, Target

# Physical constants (ARCH-COMP ACC plant)
DT = 0.1
TAU = 0.5
MU = 0.0001
V_SET = 30.0
T_GAP = 1.4
D_DEFAULT = 10.0

# a_lead = A_LEAD_HARD * sigmoid(K_LEAD * (v_lead - V_LEAD_RELEASE)),
# state-only so the plant stays time-invariant.
A_LEAD_HARD = -4.0
V_LEAD_RELEASE = 18.0
K_LEAD = 2.0

# State indices
IDX_X_LEAD = 0
IDX_V_LEAD = 1
IDX_G_LEAD = 2
IDX_X_EGO = 3
IDX_V_EGO = 4
IDX_G_EGO = 5

# Dimensions
STATE_DIM = 6
OBS_DIM = 5
ACT_DIM = 1
N_STEPS = 50

# Initial-state set (centre point implied by midpoint of these)
INITIAL_LO = np.array([85.0, 32.0, 0.0, 10.0, 30.0, 0.0])
INITIAL_HI = np.array([89.0, 32.2, 0.0, 11.0, 30.2, 0.0])
MPC_ENLARGE_FACTOR = 1.2

# Action bounds
ACT_LO = -3.0
ACT_HI = 2.0

# MPC oracle
MPC_HORIZON = 20
MPC_N_TRAJECTORIES = 500
MPC_COST_W_SPEED = 1.0
MPC_COST_W_ACT = 0.1
MPC_COST_W_RATE = 0.1
MPC_COST_W_SLACK = 1000.0
MPC_CHECKPOINT_EVERY = 25

# BC training
BC_EPOCHS = 50
BC_BATCH_SIZE = 256
BC_LR = 1e-3
BC_VAL_FRACTION = 0.1
HIDDEN_LAYERS = 5
HIDDEN_WIDTH = 20

# STL fine-tune
STL_EPOCHS = 5
STL_STEPS_PER_EPOCH = 50
STL_LR = 1e-4
STL_BATCH_INITS = 8

# SFO fine-tune
SFO_EPOCHS = 20
SFO_STEPS_PER_EPOCH = 20
SFO_LR = 1e-4
SFO_PGD_RESTARTS = 4
SFO_PGD_K = 10
SFO_PGD_ETA = 0.5
COMFORT_MAX = 2.0

# Property catalogue
PROPERTY_NAMES: tuple[str, ...] = (
    "safe",
    "comfortable",
    "respondsToBrake",
    "stabilizes",
    "cruiseUntilFollow",
)
# CROWN box-invariant subset; the temporal properties are loss-only.
INVARIANT_PROPERTY_NAMES: tuple[str, ...] = (
    "safe",
    "comfortable",
)
SFO_PROPERTY_NAMES: tuple[str, ...] = (
    "sfoSafe",
    "sfoComfortable",
    "sfoRespondsToBrake",
    "sfoStabilizes",
    "sfoCruiseUntilFollow",
)

# Eval-time PGD: one-shot per property, so heavier than training.
EVAL_PGD_K = 50
EVAL_PGD_RESTARTS = 8

# DL2 default: STL's reduceMin/reduceMax kill gradient once any sample
# fully satisfies the property; DL2's additive aggregation doesn't.
DIFFERENTIABLE_LOGIC: DifferentiableLogic = DifferentiableLogic.DL2


def parse_logic(name: str) -> DifferentiableLogic:
    """`--logic stl/dl2/vehicle` (case-insensitive) -> enum value."""
    try:
        return DifferentiableLogic[name]
    except KeyError:
        match = {m.name.lower(): m for m in DifferentiableLogic}.get(name.lower())
        if match is None:
            choices = ", ".join(m.name for m in DifferentiableLogic)
            raise ValueError(f"unknown logic {name!r}; pick from {choices}")
        return match


# The QLL custom differentiable logic is defined in the .vcl specs as the
# `QLLLoss` record; it is selected by name rather than the builtin enum.
QLL_LOGIC_NAME = "QLLLoss"


def resolve_logic(name: str) -> Target:
    """Like `parse_logic` but also resolves `qll`/`qllloss` to the custom
    `QLLLoss` logic declared in the specs."""
    if name.lower() in ("qll", "qllloss"):
        return CustomLogic(QLL_LOGIC_NAME)
    return parse_logic(name)


def logic_label(logic: Target) -> str:
    """Display name for either a `DifferentiableLogic` enum or a
    `CustomLogic` (which has no `.name`)."""
    return getattr(logic, "name", None) or getattr(logic, "_name", None) or str(logic)


# Reproducibility
SEED = 0

# Optuna hyperparameter search
OPTUNA_N_TRIALS = 30

# Paths (relative to project root)
_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
CHECKPOINTS_DIR = _ROOT / "checkpoints"
RESULTS_DIR = _ROOT / "results"
SPECS_DIR = _ROOT / "specs"

PUBLISHED_ONNX = DATA_DIR / "controller_5_20.onnx"
PUBLISHED_ONNX_URL = (
    "https://raw.githubusercontent.com/verivital/ARCH-COMP2025"
    "/main/benchmarks/ACC/controller_5_20.onnx"
)
MPC_DATA_PATH = DATA_DIR / "mpc_dataset.npz"
BC_CHECKPOINT_PATH = CHECKPOINTS_DIR / "bc_baseline.pt"
STL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "stl_trained.pt"
SFO_CHECKPOINT_PATH = CHECKPOINTS_DIR / "sfo_trained.pt"
ACC_SPEC_PATH = SPECS_DIR / "acc_safety.vcl"
ACC_SFO_SPEC_PATH = SPECS_DIR / "acc_safety_sfo.vcl"

OPTUNA_STORAGE_DIR = RESULTS_DIR / "optuna"
