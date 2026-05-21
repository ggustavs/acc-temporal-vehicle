"""Controller MLPs (obs -> action)."""

from pathlib import Path

import onnx
import requests
import torch
from torch import Tensor, nn

from acc import constants as C


def _build_mlp() -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = C.OBS_DIM
    for _ in range(C.HIDDEN_LAYERS):
        layers.append(nn.Linear(in_dim, C.HIDDEN_WIDTH))
        layers.append(nn.ReLU())
        in_dim = C.HIDDEN_WIDTH
    layers.append(nn.Linear(in_dim, C.ACT_DIM))
    return nn.Sequential(*layers)


def fresh_controller() -> nn.Sequential:
    return _build_mlp()


def load_checkpoint(path: Path) -> dict[str, Tensor]:
    """Strip the legacy `_inner.*` key prefix; idempotent on new-format checkpoints."""
    sd: dict[str, Tensor] = torch.load(path, map_location="cpu")
    prefix = "_inner."
    return {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in sd.items()}


def published_controller(path: Path = C.PUBLISHED_ONNX) -> nn.Sequential:
    """Load the ARCH-COMP ONNX, fetching it from upstream if missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(C.PUBLISHED_ONNX_URL, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    model = onnx.load(str(path))
    return _onnx_to_torch(model)


def _onnx_to_torch(model: onnx.ModelProto) -> nn.Sequential:
    initialisers = {
        init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer
    }

    layers: list[nn.Module] = []
    expected_in = C.OBS_DIM
    matmul_count = 0
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            weight_name = next((n for n in node.input if n in initialisers), None)
            if weight_name is None:
                raise ValueError(
                    f"Expected weight for {node.op_type} node, found inputs {list(node.input)}"
                )
            weight = initialisers[weight_name]
            arr = weight if node.op_type == "Gemm" else weight.T
            w = torch.from_numpy(arr.copy()).float()
            bias_name = next(
                (n for n in node.input if n in initialisers and n != weight_name),
                None,
            )
            out_dim, in_dim = w.shape
            if in_dim != expected_in:
                raise ValueError(
                    f"Layer {matmul_count} expected in_dim={expected_in}, got {in_dim}. "
                    "Suspect ONNX architecture mismatch with HIDDEN_LAYERS/HIDDEN_WIDTH."
                )
            linear = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                linear.weight.copy_(w)
                if bias_name is not None:
                    linear.bias.copy_(
                        torch.from_numpy(initialisers[bias_name].copy()).float()
                    )
                else:
                    linear.bias.zero_()
            layers.append(linear)
            expected_in = out_dim
            matmul_count += 1
        elif node.op_type == "Relu":
            layers.append(nn.ReLU())

    if matmul_count != C.HIDDEN_LAYERS + 1:
        raise ValueError(
            f"Expected {C.HIDDEN_LAYERS + 1} linear layers, found {matmul_count}. "
            "ONNX architecture differs from the documented 5x20 ReLU MLP."
        )
    return nn.Sequential(*layers)
