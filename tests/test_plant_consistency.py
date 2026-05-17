"""CasADi MPC dynamics must match the PyTorch plant to numerical precision."""

import numpy as np
import torch

from acc import constants as C
from acc.dynamics import acc_dynamics_step
from acc.mpc_oracle import MPCOracle


def test_casadi_matches_pytorch():
    rng = np.random.default_rng(C.SEED)
    for trial in range(10):
        state = rng.normal(size=(C.STATE_DIM,))
        action = rng.uniform(C.ACT_LO, C.ACT_HI, size=(C.ACT_DIM,))

        casadi_next = MPCOracle.step_casadi(state, action)
        torch_next = (
            acc_dynamics_step(
                torch.from_numpy(state).float().unsqueeze(0),
                torch.from_numpy(action).float().unsqueeze(0),
            )
            .squeeze(0)
            .numpy()
        )
        np.testing.assert_allclose(
            casadi_next, torch_next, atol=1e-5, err_msg=f"trial {trial}"
        )
