"""Constants for the ACC case study."""

from pathlib import Path

import numpy as np
from vehicle_lang.typing import CustomLogic, DifferentiableLogic, Target

# ARCH-COMP ACC plant
DT = 0.1
TAU = 0.5
MU = 0.0001
V_SET = 30.0
T_GAP = 1.4
D_DEFAULT = 10.0

# a_lead = A_LEAD_HARD * sigmoid(K_LEAD * (v_lead - V_LEAD_RELEASE));
# state-only so the plant stays time-invariant.
A_LEAD_HARD = -4.0
V_LEAD_RELEASE = 18.0
K_LEAD = 2.0

IDX_X_LEAD = 0
IDX_V_LEAD = 1
IDX_G_LEAD = 2
IDX_X_EGO = 3
IDX_V_EGO = 4
IDX_G_EGO = 5

STATE_DIM = 6
OBS_DIM = 5
ACT_DIM = 1
N_STEPS = 50

INITIAL_LO = np.array([85.0, 32.0, 0.0, 10.0, 30.0, 0.0])
INITIAL_HI = np.array([89.0, 32.2, 0.0, 11.0, 30.2, 0.0])

# Must match initialLo/Hi in acc_safety_sfo_finetune.vcl.
INITIAL_LO_FINETUNE = np.array([85.0, 22.0, 0.0, 10.0, 30.0, 0.0])
INITIAL_HI_FINETUNE = np.array([89.0, 32.2, 0.0, 11.0, 30.2, 0.0])

ACT_LO = -3.0
ACT_HI = 2.0

HIDDEN_LAYERS = 5
HIDDEN_WIDTH = 20

SATISFICE_BETA = 1.0

STL_EPOCHS = 5
STL_STEPS_PER_EPOCH = 50
STL_LR = 1e-4
STL_BATCH_INITS = 8

SFO_EPOCHS = 10
SFO_STEPS_PER_EPOCH = 20
SFO_LR = 1e-4
SFO_PGD_RESTARTS = 32
SFO_PGD_K = 10
SFO_PGD_ETA = 0.5
COMFORT_MAX = 2.0

PROPERTY_NAMES: tuple[str, ...] = (
    "safe",
    "comfortable",
    "respondsToBrake",
    "stabilizes",
    "cruiseUntilFollow",
    "tracksSetSpeed",
)
# Verifier-eligible subset (box-invariant).
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
    "sfoTracksSetSpeed",
)

# One-shot per property at eval time, so heavier than training.
EVAL_PGD_K = 50
EVAL_PGD_RESTARTS = 8

# Names outside this dict resolve to a CustomLogic defined in the .vcl.
_BUILTIN_LOGICS = {
    "VehicleLoss": DifferentiableLogic.Vehicle,
    "DL2Loss": DifferentiableLogic.DL2,
    "STLLoss": DifferentiableLogic.STL,
}


def to_logic(name: str) -> Target:
    """Builtin for VehicleLoss/DL2Loss/STLLoss; any other name -> CustomLogic."""
    return _BUILTIN_LOGICS.get(name, CustomLogic(name))


LOGIC_OPTION_HELP = (
    "Vehicle logic: VehicleLoss | DL2Loss | STLLoss, or a custom logic name"
)

DEFAULT_LOGIC_NAME = "QLLLoss"
DIFFERENTIABLE_LOGIC: Target = to_logic(DEFAULT_LOGIC_NAME)

SEED = 0

OPTUNA_N_TRIALS = 15

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
STL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "stl_trained.pt"
SFO_CHECKPOINT_PATH = CHECKPOINTS_DIR / "sfo_trained.pt"
STL_FINETUNED_PATH = CHECKPOINTS_DIR / "stl_finetuned.pt"
SFO_FINETUNED_PATH = CHECKPOINTS_DIR / "sfo_finetuned.pt"
ACC_SPEC_PATH = SPECS_DIR / "acc_safety.vcl"
ACC_SFO_SPEC_PATH = SPECS_DIR / "acc_safety_sfo.vcl"
ACC_SFO_FINETUNE_SPEC_PATH = SPECS_DIR / "acc_safety_sfo_finetune.vcl"

OPTUNA_STORAGE_DIR = RESULTS_DIR / "optuna"
