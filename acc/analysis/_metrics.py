"""Per-trajectory metric functions and confidence-interval helpers."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import beta

from acc import constants as C
from acc.analysis._data import DDEF, DT, TGAP, VSET, iGE, iVE, iVL, iXE, iXL


def margins(traj: np.ndarray) -> np.ndarray:
    drel = traj[..., iXL] - traj[..., iXE]
    dsafe = DDEF + TGAP * traj[..., iVE]
    return drel - dsafe


def margins_t(traj: torch.Tensor) -> torch.Tensor:
    drel = traj[..., iXL] - traj[..., iXE]
    dsafe = DDEF + TGAP * traj[..., iVE]
    return drel - dsafe


def comfort_peak(traj: np.ndarray) -> np.ndarray:
    return np.abs(traj[..., iGE]).max(axis=1)


def jerk_peak(traj: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(traj[..., iGE], axis=1) / DT).max(axis=1)


def jerk_rms(traj: np.ndarray) -> np.ndarray:
    j = np.diff(traj[..., iGE], axis=1) / DT
    return np.sqrt((j * j).mean(axis=1))


def accel_rms(traj: np.ndarray) -> np.ndarray:
    g = traj[..., iGE]
    return np.sqrt((g * g).mean(axis=1))


def time_in_saturation(traj: np.ndarray, threshold_frac: float = 0.95) -> np.ndarray:
    centre = 0.5 * (C.ACT_HI + C.ACT_LO)
    half = 0.5 * (C.ACT_HI - C.ACT_LO)
    return (np.abs(traj[..., iGE] - centre) > threshold_frac * half).mean(axis=1)


def time_headway_rmse(traj: np.ndarray) -> np.ndarray:
    drel = traj[..., iXL] - traj[..., iXE]
    vego = np.maximum(traj[..., iVE], 1.0)
    th = drel / vego
    half = traj.shape[1] // 2
    return np.sqrt(((th[:, half:] - TGAP) ** 2).mean(axis=1))


def vset_recovery_time(
    traj: np.ndarray,
    v_set: float | np.ndarray = VSET,
    release_step: int = 0,
    tol: float = 0.5,
) -> np.ndarray:
    """Steps from `release_step` until `|v_ego - v_set| < tol` and stays
    inside the tolerance band. NaN per init that never converges."""
    v_set_arr = np.asarray(v_set)
    if v_set_arr.ndim == 0:
        v_set_arr = np.full(traj.shape[0], float(v_set_arr))
    ve = traj[..., iVE]
    err = np.abs(ve - v_set_arr[:, None])
    n_init = traj.shape[0]
    out = np.full(n_init, np.nan)
    for i in range(n_init):
        for t in range(release_step, traj.shape[1]):
            if err[i, t] < tol:
                end = min(t + 4, traj.shape[1])
                if (err[i, t:end] < tol).all():
                    out[i] = t - release_step
                    break
    return out


def steady_state_v_err(
    traj: np.ndarray, v_set: float | np.ndarray = VSET
) -> np.ndarray:
    v_set_arr = np.asarray(v_set)
    if v_set_arr.ndim == 0:
        v_set_arr = np.full(traj.shape[0], float(v_set_arr))
    vach = np.minimum(v_set_arr[:, None], traj[..., iVL])
    last = int(traj.shape[1] * 0.8)
    err = np.abs(traj[:, last:, iVE] - vach[:, last:])
    return err.mean(axis=1)


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Wilson-style binomial proportion + 95% CI (via Beta posterior)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return (k / n, float(lo), float(hi))
