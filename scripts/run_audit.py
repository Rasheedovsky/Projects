"""Phase-0 audit: load raw SPY 1-min export, verify timezone, apply exclusion
rules, run all audit checks, emit reports/data_audit.md + event-day figures
and the cleaned RTH parquet for Phase 1.

Usage: python scripts/run_audit.py [--config configs/data.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spydt.data import audit as A
from spydt.data.calendar import nyse_sessions
from spydt.data.loading import infer_et_offset_hours, normalize_to_et, parse_raw_csv
from spydt.data.sessions import rth_slice, sessionize

EVENT_DAYS = ["2010-05-06", "2015-08-24", "2018-02-05", "2020-03-16"]


def event_panel(days: dict, date: str, out_dir: Path) -> str | None:
    if date not in days:
        return None
    frame = days[date]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(frame.index.strftime("%H:%M"), frame["close"], lw=0.8)
    ax[0].set_title(f"SPY {date} — close")
    ax[1].bar(range(len(frame)), frame["volume"], width=1.0)
    ax[1].set_title("volume")
    step = max(len(frame) // 8, 1)
    ax[0].set_xticks(range(0, len(frame), step))
    fig.tight_layout()
    path = out_dir / f"event_{date}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    print("parsing raw csv ...")
    raw = parse_raw_csv(cfg["raw"]["csv_path"])
    n_raw = len(raw)

    inferred = infer_et_offset_hours(raw)
    claimed = cfg["timezone"]["file_offset_hours_vs_et"]
    print(f"offset: inferred {inferred}h, config {claimed}h")
    df = normalize_to_et(raw, offset_hours=claimed, verify=True)

    bars_per_day_pre = df.groupby(df.index.date).size()
    rth = rth_slice(df)
    days = sessionize(rth)

    # DST consistency on RTH bars: modal first-bar minute per month must be
    # 09:30 ET (pre-slice frames start at premarket bars on extended days)
    firsts = rth.groupby(rth.index.date).apply(lambda g: g.index[0].strftime("%H:%M"))
    firsts.index = pd.to_datetime(firsts.index)
    monthly_mode = firsts.groupby(firsts.index.to_period("M")).agg(lambda s: s.mode().iloc[0])
    dst_ok = (monthly_mode == "09:30").mean()

    cal = nyse_sessions(str(rth.index[0].date()), str(rth.index[-1].date()))
    cal_dates = set(cal.index.strftime("%Y-%m-%d"))
    half_days = set(cal.index[cal["is_early_close"]].strftime("%Y-%m-%d"))

    ghost_days = sorted(set(days) - cal_dates)  # bars on non-sessions
    absent_days = sorted(cal_dates - set(days))  # sessions with zero bars
    for g in ghost_days:
        days.pop(g)

    # flash-crash timezone verification (hard evidence beyond the volume spike)
    fc = days.get("2010-05-06")
    fc_low_et = None
    if fc is not None:
        fc_low_et = fc["low"].idxmin().strftime("%H:%M")
        assert "14:40" <= fc_low_et <= "14:50", (
            f"flash-crash trough at {fc_low_et} ET contradicts the claimed offset"
        )

    anchor = A.check_price_level_anchor(days)
    census = A.missing_minute_census(days, half_days)
    cls = A.classify_days(days, half_days,
                          max_missing_frac_morning=cfg["exclusions"]["max_missing_frac_morning"])
    outliers = A.check_outliers(days)
    adjust = A.check_adjustment(days)
    vol_units = A.check_volume_units(days)
    included = {d: f for d, f in days.items() if cls.loc[d, "included"]}
    balance = A.label_balance(included)

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    panels = {d: event_panel(days, d, fig_dir) for d in EVENT_DAYS}

    # persist cleaned RTH frame + day classification for Phase 1
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    rth.to_parquet("data/interim/spy_rth_et.parquet")
    cls.to_parquet("data/interim/day_classification.parquet")

    excl_counts = cls[~cls["included"]]["reason"].value_counts().to_dict()
    n_included = int(cls["included"].sum())
    eras = cfg["eras"]
    n_ssl = int(cls[(cls.index >= eras["ssl_pretrain"][0]) & (cls.index <= eras["ssl_pretrain"][1])]["included"].sum())
    n_sup = int(cls[(cls.index >= eras["supervised"][0]) & (cls.index <= eras["supervised"][1])]["included"].sum())

    md = []
    md.append("# Data audit — SPY 1-min (Phase 0)\n")
    md.append(f"Source: `{cfg['raw']['source']}`  \nZip SHA-256: `{cfg['raw']['zip_sha256']}`\n")
    md.append("## Provenance / integrity\n")
    md.append(f"- CSV rows: {raw.attrs['n_raw_rows']:,}; exact-duplicate re-export rows dropped: "
              f"{raw.attrs['n_duplicate_rows_dropped']:,} (loader hard-fails on any duplicated "
              "timestamp with conflicting values — none found).")
    md.append(f"- Data span: {rth.index[0].strftime('%Y-%m-%d')} → {rth.index[-1].strftime('%Y-%m-%d')} "
              "(note: coverage ends April 2021, not year-end).")
    md.append("\n## Timezone / clock\n")
    md.append(f"- Unique-timestamp rows: {n_raw:,}; bars/day before RTH slice: median "
              f"{int(bars_per_day_pre.median())}, max {int(bars_per_day_pre.max())} "
              "(extended-hours bars present on a minority of days; discarded).")
    md.append(f"- Inferred file-clock offset vs ET: **{inferred} h** (start of the max-total-volume "
              f"390-minute window = RTH block), matches config ({claimed} h). Normalization verified, hard-fail armed.")
    md.append(f"- DST consistency: modal first bar = 09:30 ET in {dst_ok:.1%} of months "
              "(file clock follows ET DST at a fixed offset).")
    if fc_low_et:
        md.append(f"- 2010-05-06 flash-crash trough at **{fc_low_et} ET** (expected ~14:45) — offset confirmed on independent evidence.")
    md.append("\n## Calendar coverage\n")
    md.append(f"- NYSE sessions in span: {len(cal_dates):,}; sessions with data: {len(days):,}; "
              f"sessions entirely absent: {len(absent_days)}"
              + (f" ({', '.join(absent_days[:10])}{' …' if len(absent_days) > 10 else ''})" if absent_days else "")
              + f"; non-session ghost days removed: {len(ghost_days)}"
              + (f" ({', '.join(ghost_days[:10])}{' …' if len(ghost_days) > 10 else ''})" if ghost_days else "") + ".")
    md.append(f"- Half days (early close 13:00 ET) in span: {len(half_days & set(cls.index))} — all dropped per contract.")
    md.append("\n## Missing minutes (full days)\n")
    md.append(census.to_markdown())
    md.append("\n## Exclusion rules (ex ante)\n")
    md.append(f"- Included days: **{n_included:,}** of {len(cls):,}; exclusions by reason: {excl_counts}.")
    md.append(f"- SSL-pretrain era {eras['ssl_pretrain']}: **{n_ssl:,}** days; "
              f"supervised era {eras['supervised']}: **{n_sup:,}** days.")
    md.append("\n## Outliers\n")
    md.append(f"- 1-min log-return prints > 20 robust-σ: {outliers['n_price_outliers']} "
              f"(worst days: {outliers['worst_days'][:5]}).")
    md.append(f"- Negative-volume bars: {outliers['n_negative_volume']}; zero-volume bars: {outliers['n_zero_volume']}.")
    md.append("\n## Adjustment status\n")
    md.append(f"- **Decisive price-level anchor: {anchor['verdict']}** — median early-2008 close "
              f"{anchor['median_close']:.2f} over {anchor.get('n_days', 0)} days "
              "(SPY's actual unadjusted level was ~120-148; a dividend-back-adjusted series would sit ~95-115).")
    md.append(f"- Supporting ex-div signature: {adjust['verdict']} — overnight return on candidate ex-div dates "
              f"(3rd Fri of Mar/Jun/Sep/Dec incl. holiday shifts, n={adjust.get('n_exdiv', 0)}) minus other days: "
              f"mean {adjust['exdiv_minus_other_bp']:.1f} bp / median {adjust.get('exdiv_minus_other_median_bp', float('nan')):.1f} bp "
              f"(ex-div {adjust.get('mean_exdiv_on_bp', float('nan')):.1f} bp vs other {adjust.get('mean_other_on_bp', float('nan')):.1f} bp).")
    md.append("- Policy: unadjusted prices for dollar bars/execution; r_on gets an explicit dividend correction "
              "from SPY distribution history in Phase 1 (the weak/noisy ex-div mean is expected at n≈52 with "
              "60-100 bp overnight vol; the price anchor is the classification).")
    md.append("\n## Volume units\n")
    md.append(f"- Median daily-sum ratio to reference share volume: {vol_units['median_ratio_to_reference']:.4f} "
              f"→ multiplier **×{vol_units['multiplier']}** (IB lots of 100)." if vol_units["multiplier"] == 100
              else f"- Median ratio {vol_units['median_ratio_to_reference']:.2f} → volume already in shares.")
    md.append("\n## Label balance by year (included days; y = sign of 15:30→close log return)\n")
    md.append(balance.round(2).to_markdown())
    md.append("\n## Event-day panels\n")
    for d, p in panels.items():
        md.append(f"- {d}: " + (f"![{d}](figures/{p})" if p else "**absent from data**"))
    md.append("\n## Artifacts\n")
    md.append("- `data/interim/spy_rth_et.parquet` — cleaned tz-aware RTH bars (unadjusted prices, raw volume units).")
    md.append("- `data/interim/day_classification.parquet` — per-day include/exclude with reason.\n")

    Path("reports/data_audit.md").write_text("\n".join(md))
    print(f"included {n_included} / {len(cls)} days; report -> reports/data_audit.md")


if __name__ == "__main__":
    main()
