"""Phase 1: falsifiability diagnostic on real data.

For each bar scheme (dollar, dollar_imbalance): build the axis bundle on the
SUPERVISED era with a first-60%-of-train calibration (diagnostic only; the
matrix recalibrates per CPCV split), then report degeneracy stats and the
clock-vs-info similarity. Decides the pilot's primary bar scheme.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spydt.axes import axis_similarity, build_bundle, load_daydata

CFG = yaml.safe_load(Path("configs/data.yaml").read_text())
PILOT = yaml.safe_load(Path("configs/pilot.yaml").read_text())


def main() -> None:
    days = load_daydata(
        "data/interim/spy_rth_et.parquet",
        "data/interim/day_classification.parquet",
        CFG["volume"]["unit_multiplier"],
    )
    sup_lo, sup_hi = CFG["eras"]["supervised"]
    sup = [d for d in days if sup_lo <= d.date <= sup_hi]
    print(f"supervised days: {len(sup)} ({sup[0].date} .. {sup[-1].date})")
    train_pos = np.arange(int(len(sup) * 0.6))

    K, L = PILOT["axes"]["K"], PILOT["axes"]["L"]
    results = {}
    for kind in ["dollar", "dollar_imbalance"]:
        t0 = time.time()
        b = build_bundle(sup, train_pos, bar_kind=kind, K=K, L=L,
                         floor_buckets=PILOT["axes"]["floor_buckets"],
                         cap_mult=PILOT["axes"]["cap_mult"],
                         ffd_grid_step=PILOT["axes"]["ffd_grid_step"])
        sim = axis_similarity(b)
        results[kind] = {"diag": b.diag, "similarity": sim,
                         "build_seconds": round(time.time() - t0, 1)}
        print(kind, json.dumps(results[kind], indent=1, default=str))

    # primary decision: lower degeneracy (floor+cap+pad), then lower similarity
    def degeneracy(r):
        d = r["diag"]
        return d["floor_rate"] + d["cap_rate"] + 0.5 * d["pad_rate"]

    primary = min(results, key=lambda k: (degeneracy(results[k]),
                                          results[k]["similarity"]["median_abs_corr"]))
    falsifiable = all(r["similarity"]["median_abs_corr"] < 0.95 for r in results.values())

    md = ["# Phase 1 — falsifiability diagnostic\n"]
    md.append(f"Supervised era: {sup[0].date} → {sup[-1].date} ({len(sup)} days); "
              f"K={K}, L={L}; calibration on first 60% (diagnostic only).\n")
    md.append("| scheme | d* clock | d* info | median bars/day | median morning bars | "
              "pad rate | floor rate | cap rate | median |corr| clock-info | frac>0.95 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for kind, r in results.items():
        d, s = r["diag"], r["similarity"]
        md.append(
            f"| {kind} | {d['d_clock']} | {d['d_info']} | {d['median_bars_per_day']:.0f} | "
            f"{d['median_morning_bars']:.0f} | {d['pad_rate']:.2f} | {d['floor_rate']:.3f} | "
            f"{d['cap_rate']:.3f} | {s['median_abs_corr']:.3f} | {s['frac_above_0.95']:.3f} |"
        )
    md.append(f"\n**Primary bar scheme for the pilot: `{primary}`** "
              "(lowest floor/cap/pad degeneracy, then lowest axis similarity).")
    md.append(f"\n**Falsifiability gate: {'PASS' if falsifiable else 'FAIL'}** — the info axis "
              + ("is distinguishable from the clock axis (median |corr| < 0.95); the dual-time "
                 "hypothesis is testable on this data."
                 if falsifiable else
                 "is statistically indistinguishable from the clock axis; the dual-time "
                 "comparison would be vacuous. FLAGGED."))
    Path("reports/phase1_diagnostic.md").write_text("\n".join(md) + "\n")
    Path("data/interim/phase1_decision.json").write_text(
        json.dumps({"primary_bar_scheme": primary, "falsifiable": falsifiable}, indent=1)
    )
    print(f"\nprimary={primary} falsifiable={falsifiable} -> reports/phase1_diagnostic.md")


if __name__ == "__main__":
    main()
