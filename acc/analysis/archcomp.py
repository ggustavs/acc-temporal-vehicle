"""ARCH-COMP 2025 ACC benchmark scoring (constant lead a=-2, [90,110] init set)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import immrax
import jax.numpy as jnp
import numpy as np
from immutabledict import immutabledict
from rich.console import Console
from torch import nn

from acc import constants as C
from acc.analysis._data import load_arms
from acc.verifier import _ACCController, _INVARIANT_PREDICATES, _jax_weights_from_torch

ARCHCOMP_A_LEAD = -2.0
ARCHCOMP_LO = jnp.array([90.0, 32.0, 0.0, 10.0, 30.0, 0.0])
ARCHCOMP_HI = jnp.array([110.0, 32.2, 0.0, 11.0, 30.2, 0.0])
ARCHCOMP_STEPS = 50


def _archcomp_step(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    v_lead = state[C.IDX_V_LEAD]
    g_lead = state[C.IDX_G_LEAD]
    v_ego = state[C.IDX_V_EGO]
    g_ego = state[C.IDX_G_EGO]
    a_ego = action[0]
    a_lead = ARCHCOMP_A_LEAD
    deriv = jnp.stack([
        v_lead,
        g_lead,
        (a_lead - g_lead) / C.TAU - C.MU * v_lead * v_lead,
        v_ego,
        g_ego,
        (a_ego - g_ego) / C.TAU - C.MU * v_ego * v_ego,
    ])
    return state + C.DT * deriv


class _ArchCompPlant(immrax.OpenLoopSystem):
    def __init__(self) -> None:
        super().__init__(evolution="discrete", xlen=C.STATE_DIM)

    def f(self, t, x, u, w):  # noqa: ARG002
        return _archcomp_step(x, u)


def score_one(net: nn.Module, n_steps: int = ARCHCOMP_STEPS) -> tuple[dict, float]:
    """Verify every invariant predicate over n_steps of ARCH-COMP rollout.
    Returns ({prop: (verified, note)}, runtime_seconds)."""
    ctrl = _ACCController(_jax_weights_from_torch(net))
    nncs = immrax.NNCSystem(_ArchCompPlant(), ctrl)  # pyright: ignore[reportArgumentType]
    emb = immrax.NNCEmbeddingSystem(nncs, nn_verifier="crown")
    init_ix = immrax.interval(ARCHCOMP_LO, ARCHCOMP_HI)
    empty_w = immrax.interval(jnp.zeros(0), jnp.zeros(0))
    n_corner = 1 + C.STATE_DIM + ctrl.out_len

    t0 = time.perf_counter()
    traj = emb.compute_trajectory(
        t0=0,
        tf=n_steps,
        x0=immrax.i2ut(init_ix),
        f_kwargs=immutabledict(
            {
                "w": empty_w,
                "permutations": immrax.standard_permutation(n_corner),
                "corners": immrax.two_corners(n_corner),
            }
        ),
        dt=1,
        solver="euler",
    )
    runtime = time.perf_counter() - t0

    states_ut = np.asarray(traj.ys)
    res: dict[str, tuple[bool, str]] = {}
    for step in range(min(n_steps, len(states_ut))):
        ut = states_ut[step]
        if not np.all(np.isfinite(ut)):
            break
        lower, upper = immrax.i2lu(immrax.ut2i(jnp.asarray(ut)))
        for name, pred in _INVARIANT_PREDICATES.items():
            if name in res:
                continue
            ok, why = pred(np.asarray(lower), np.asarray(upper))
            if not ok:
                res[name] = (False, f"falsified at step {step}: {why}")
    for name in _INVARIANT_PREDICATES:
        res.setdefault(name, (True, f"holds over {n_steps} steps"))
    return res, runtime


def archcomp_compute(arms: dict[str, nn.Module]) -> dict:
    """Score each arm. Returns {arm: {prop: {verified, note}, runtime_s}}."""
    out: dict = {}
    for tag, net in arms.items():
        res, rt = score_one(net)
        out[tag] = {
            k: {"verified": v[0], "note": v[1]} for k, v in res.items()
        }
        out[tag]["runtime_s"] = round(rt, 2)
    return out


def archcomp_summarise(scores: dict, order: list[str]) -> dict:
    return {
        "benchmark": "ARCH-COMP 2025 ACC",
        "initial_set": {"lo": ARCHCOMP_LO.tolist(), "hi": ARCHCOMP_HI.tolist()},
        "a_lead": ARCHCOMP_A_LEAD,
        "steps": ARCHCOMP_STEPS,
        "dt": C.DT,
        "engine": "immrax CROWN interval reachability (sound, conservative)",
        "results": {tag: scores[tag] for tag in order},
    }


def archcomp_render(summary: dict, out_dir: Path) -> None:
    """Write the markdown summary; the JSON is written by the orchestrator."""
    order = list(summary["results"].keys())
    L = ["# ARCH-COMP 2025 ACC benchmark score\n"]
    L.append(
        "ARCH-COMP's exact ACC: x_lead in [90,110], a_lead=-2 const, "
        "x_lead-x_ego >= 10 + 1.4*v_ego, 50 steps @ dt 0.1 (5 s). "
        "Engine: immrax CROWN interval reachability. CROWN "
        "over-approximates the reachable set, so a FALSIFIED result "
        "may be engine conservatism rather than a true counterexample; "
        "ARCH-COMP submissions typically use CORA or NNV for continuous "
        "reachability.\n"
    )
    L.append("| controller | safe | comfortable | runtime |")
    L.append("|---|---|---|---|")
    for tag in order:
        r = summary["results"][tag]
        s = "VERIFIED" if r["safe"]["verified"] else "FALSIFIED"
        c = "VERIFIED" if r["comfortable"]["verified"] else "FALSIFIED"
        L.append(f"| {tag} | {s} | {c} | {r['runtime_s']}s |")
    L.append("")
    for tag in order:
        for prop in ("safe", "comfortable"):
            n = summary["results"][tag][prop]
            if not n["verified"]:
                L.append(f"- {tag}/{prop}: {n['note']}")
    (out_dir / "archcomp_score.md").write_text("\n".join(L) + "\n")


def archcomp_core(
    arms_to_paths: dict[str, str],
    out_dir: Path,
    console: Optional[Console] = None,
) -> dict:
    console = console or Console()
    out_dir.mkdir(parents=True, exist_ok=True)
    nets = load_arms(arms_to_paths)
    order = list(nets.keys())

    scores = archcomp_compute(nets)
    summary = archcomp_summarise(scores, order)
    archcomp_render(summary, out_dir)
    (out_dir / "archcomp_score.json").write_text(json.dumps(summary, indent=2))

    for tag in order:
        r = summary["results"][tag]
        verdict = {
            k: ("VERIFIED" if r[k]["verified"] else "FALSIFIED")
            for k in r if k != "runtime_s"
        }
        console.print(f"{tag} {verdict} {r['runtime_s']:.1f}s")
    console.print(f"wrote {out_dir / 'archcomp_score.md'}")
    return summary
