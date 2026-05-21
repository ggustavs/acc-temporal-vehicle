"""The JAX verifier plant must match the PyTorch training plant."""

import jax.numpy as jnp
import numpy as np
import torch

from acc import constants as C
from acc.dynamics import acc_dynamics_step
from acc.verifier import _acc_dynamics_step_jax


def _torch_step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return (
        acc_dynamics_step(
            torch.from_numpy(state).float().unsqueeze(0),
            torch.from_numpy(action).float().unsqueeze(0),
        )
        .squeeze(0)
        .numpy()
    )


def _jax_step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.asarray(
        _acc_dynamics_step_jax(jnp.asarray(state), jnp.asarray(action))
    )


def test_jax_matches_pytorch_random():
    rng = np.random.default_rng(C.SEED)
    for trial in range(10):
        state = rng.normal(size=(C.STATE_DIM,)).astype(np.float32)
        action = rng.uniform(C.ACT_LO, C.ACT_HI, size=(C.ACT_DIM,)).astype(
            np.float32
        )
        np.testing.assert_allclose(
            _jax_step(state, action),
            _torch_step(state, action),
            atol=1e-5,
            err_msg=f"trial {trial}",
        )


def test_jax_matches_pytorch_saturation():
    state = np.array([85.0, 32.0, 0.0, 10.0, 30.0, 0.0], dtype=np.float32)
    for cmd in (-100.0, -10.0, 5.0, 1e6):
        action = np.array([cmd], dtype=np.float32)
        np.testing.assert_allclose(
            _jax_step(state, action),
            _torch_step(state, action),
            atol=1e-5,
            err_msg=f"cmd={cmd}",
        )


def test_jax_matches_pytorch_negative_velocity():
    # Engineer states where the un-clamped Euler step would produce v<0,
    # so the clamp path is exercised on both sides.
    state = np.array([85.0, 0.5, -8.0, 10.0, 0.5, -8.0], dtype=np.float32)
    action = np.array([-3.0], dtype=np.float32)
    out_jax = _jax_step(state, action)
    out_torch = _torch_step(state, action)
    assert out_jax[C.IDX_V_LEAD] >= 0.0 and out_jax[C.IDX_V_EGO] >= 0.0
    np.testing.assert_allclose(out_jax, out_torch, atol=1e-5)
