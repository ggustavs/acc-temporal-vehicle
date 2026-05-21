"""Serialise + cache `vehicle_lang` compiles.

The Haskell FFI compiler segfaults on concurrent calls and recompiles
identical inputs per trial. Loads go through `_LOAD_LOCK`; compiled ASTs
cached by (spec, logic, declarations).
"""

import json
import threading
import time
from pathlib import Path
from typing import Any

from vehicle_lang.error import VehicleError
from vehicle_lang.loss import _ast as _loss_ast
from vehicle_lang.loss import pytorch as loss_pt

_LOAD_LOCK = threading.Lock()

_COMPILE_RETRIES = 6
_COMPILE_BACKOFF_S = 0.5

_compile_cache: dict[tuple[str, str, tuple[str, ...]], Any] = {}
_orig_ast_load = _loss_ast.load


def _cached_ast_load(
    path: "str | Path",
    *,
    declarations: Any = (),
    target: Any = None,
) -> Any:
    decls = tuple(str(d) for d in declarations)
    logic_name = getattr(target, "_vehicle_option_name", repr(target))
    key = (str(path), logic_name, decls)
    cached = _compile_cache.get(key)
    if cached is not None:
        return cached
    for attempt in range(_COMPILE_RETRIES):
        try:
            prog = _orig_ast_load(path, declarations=declarations, target=target)
            _compile_cache[key] = prog
            return prog
        except (VehicleError, json.JSONDecodeError):
            if attempt == _COMPILE_RETRIES - 1:
                raise
            time.sleep(_COMPILE_BACKOFF_S)
    raise AssertionError("unreachable")  # for type checkers


# Patch the module-level loader so every caller (incl. internal ones) hits the cache.
_loss_ast.load = _cached_ast_load


def load_specification(spec_path: Path, **kwargs: Any) -> dict[str, Any]:
    with _LOAD_LOCK:
        return loss_pt.load_specification(spec_path, **kwargs)
