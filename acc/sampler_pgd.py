"""PGD sampler for Vehicle's PyTorch loss backend: fixed step size + per-step projection."""

from typing import Callable, Sequence

import torch
from jaxtyping import Float
from vehicle_lang.loss.pytorch import PyTorchSampler

from acc import constants as C


class PGDSampler(PyTorchSampler):
    def __init__(
        self,
        num_restarts: int = C.SFO_PGD_RESTARTS,
        num_steps: int = C.SFO_PGD_K,
        eta: float = C.SFO_PGD_ETA,
        seed: int = C.SEED,
    ) -> None:
        self.num_restarts = num_restarts
        self.num_steps = num_steps
        self.eta = eta
        self.generator = torch.Generator().manual_seed(seed)

    def get_loss(
        self,
        dims: Sequence[int],
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        search_lambda: Callable[[torch.Tensor], torch.Tensor],
        minimise: bool,
    ) -> Float[torch.Tensor, "1 losses"]:
        results = []
        range_size = upper_bound - lower_bound

        for _ in range(self.num_restarts):
            x = (
                lower_bound
                + torch.rand(
                    lower_bound.shape,
                    generator=self.generator,
                    dtype=lower_bound.dtype,
                )
                * range_size
            )

            for _ in range(self.num_steps):
                x_var = x.detach().clone().requires_grad_(True)
                loss = search_lambda(x_var)

                if not loss.requires_grad:
                    break

                grad = torch.autograd.grad(
                    loss,
                    x_var,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                grad = torch.nan_to_num(grad)

                step_dir = -torch.sign(grad) if minimise else torch.sign(grad)
                x = torch.clamp(
                    x.detach() + self.eta * step_dir,
                    lower_bound,
                    upper_bound,
                )

            results.append(torch.as_tensor(search_lambda(x.detach())))

        return torch.stack(results)
