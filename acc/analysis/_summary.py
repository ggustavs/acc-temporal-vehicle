"""Distributional summary helpers for building per-arm metric dicts."""

from __future__ import annotations

from typing import Callable

import numpy as np


def per_arm_quantiles(
    per_arm_values: dict[str, np.ndarray],
    *,
    metrics: dict[str, Callable[[np.ndarray], float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Build `{arm: {metric_name: value}}` for each arm by applying
    every metric function to that arm's value array. Default metrics
    are median + p1 + p99."""
    if metrics is None:
        metrics = {
            "median": lambda x: float(np.median(x)),
            "p1": lambda x: float(np.percentile(x, 1)),
            "p99": lambda x: float(np.percentile(x, 99)),
        }
    return {
        arm: {name: fn(arr) for name, fn in metrics.items()}
        for arm, arr in per_arm_values.items()
    }
