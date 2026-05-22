"""Aggregate per-controller eval metrics into a comparison report.

Satisfied iff: corners_all_pass AND pgd_passes AND (verifier.verified
when the property is in INVARIANT_PROPERTY_NAMES).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from acc import constants as C


def _load(metrics_dir: Path, which: str) -> dict:
    return json.loads((metrics_dir / f"{which}_metrics.json").read_text())


def _satisfied(prop: str, pp: dict, ver: dict, invariant: set[str]) -> bool:
    ok = bool(pp["corners_all_pass"]) and bool(pp["pgd_passes"])
    if prop in invariant:
        ok = ok and bool(ver.get(prop, {}).get("verified", False))
    return ok


def compare_core(
    arms: Sequence[str],
    metrics_dir: Path,
    out_dir: Path,
    baseline: str = "published",
    console: Optional[Console] = None,
) -> dict:
    console = console or Console()
    out_dir.mkdir(parents=True, exist_ok=True)
    invariant = set(C.INVARIANT_PROPERTY_NAMES)
    data = {w: _load(metrics_dir, w) for w in arms}
    props = list(C.PROPERTY_NAMES)

    rows: list[dict] = []
    satisfied = {w: 0 for w in arms}
    for prop in props:
        row: dict = {"property": prop, "invariant": prop in invariant}
        for w in arms:
            pp = data[w]["per_property"][prop]
            ver = data[w].get("verifier", {})
            sat = _satisfied(prop, pp, ver, invariant)
            satisfied[w] += int(sat)
            row[w] = {
                "centre_loss": pp["centre_loss"],
                "corners_all_pass": pp["corners_all_pass"],
                "pgd_worst_loss": pp["pgd_worst_loss"],
                "pgd_passes": pp["pgd_passes"],
                "verified": ver.get(prop, {}).get("verified")
                if prop in invariant
                else None,
                "satisfied": sat,
            }
        rows.append(row)

    delta_arms = [w for w in arms if w != baseline]
    report = {
        "controllers": list(arms),
        "properties": props,
        "invariant_properties": sorted(invariant),
        "rows": rows,
        "satisfied_count": satisfied,
        "deltas_vs_baseline": {
            w: {
                r["property"]: round(
                    r[w]["centre_loss"] - r[baseline]["centre_loss"], 4
                )
                for r in rows
            }
            for w in delta_arms
        },
        "baseline": baseline,
    }
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2))

    L: list[str] = []
    L.append(f"# Controller comparison vs {baseline}\n")
    L.append(
        "Same evaluator (QLLLoss) for all arms. Loss: lower is better, "
        "pass iff <= 0. `verified` = immrax CROWN reachability (only "
        "`safe`/`comfortable`).\n"
    )
    L.append("## Properties satisfied (of %d)\n" % len(props))
    L.append("| controller | satisfied |")
    L.append("|---|---|")
    for w in arms:
        L.append(f"| {w} | {satisfied[w]}/{len(props)} |")
    L.append("")
    for prop in props:
        r = next(x for x in rows if x["property"] == prop)
        inv = " (invariant)" if r["invariant"] else ""
        L.append(f"## {prop}{inv}\n")
        L.append(
            "| controller | centre_loss | corners_pass | pgd_worst | "
            "pgd_pass | verified | SATISFIED |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for w in arms:
            c = r[w]
            L.append(
                f"| {w} | {c['centre_loss']:+.3f} | {c['corners_all_pass']} "
                f"| {c['pgd_worst_loss']:+.3f} | {c['pgd_passes']} | "
                f"{c['verified']} | {'YES' if c['satisfied'] else 'no'} |"
            )
        L.append("")
    if delta_arms:
        L.append(f"## Centre-loss delta vs {baseline} (negative = better)\n")
        L.append(
            "| property | " + " | ".join(f"{w}-{baseline}" for w in delta_arms) + " |"
        )
        L.append("|---|" + "|".join(["---"] * len(delta_arms)) + "|")
        for prop in props:
            deltas = " | ".join(
                f"{report['deltas_vs_baseline'][w][prop]:+.3f}" for w in delta_arms
            )
            L.append(f"| {prop} | {deltas} |")
    (out_dir / "comparison.md").write_text("\n".join(L) + "\n")

    console.print(f"wrote {out_dir / 'comparison.md'} and comparison.json")
    console.print(f"satisfied: {satisfied}")
    return report
