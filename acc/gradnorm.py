"""GradNorm with a task-weight floor (no task can starve the others)."""

from __future__ import annotations

import torch

from vehicle_lang.loss.pytorch import GradNormBalancer

# Floor as a fraction of the equal share T/N.
_WEIGHT_FLOOR_FRAC = 0.1


class GuardedGradNormBalancer(GradNormBalancer):
    def step(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = super().step()
        names = self._task_names
        n = len(names)
        floor = _WEIGHT_FLOOR_FRAC * (self._T / n)
        with torch.no_grad():
            for name in names:
                self._weights[name].clamp_(min=floor)
            s = sum(self._weights[name] for name in names)
            scale = self._T / s
            for name in names:
                self._weights[name].mul_(scale)
        return out
