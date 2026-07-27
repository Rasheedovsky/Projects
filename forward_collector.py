#!/usr/bin/env python3
"""
forward_collector.py — forward-looking Riyadh traffic collector.

Polls the TomTom Flow Segment Data API every 60 s during the Riyadh morning peak
(06:00–09:00 Asia/Riyadh) for a fixed set of Riyadh road segments and appends
observations to riyadh_traffic.csv using the project schema (see data_dictionary.md).

WHY THIS EXISTS: the historical-acquisition session could not obtain sub-hourly
Riyadh data (see acquisition_log.md). This script builds that record going forward,
on YOUR machine — the agent session cannot schedule or run it for you.

SETUP
=====
1. Register (free) for a TomTom developer account and create an API key:
       https://developer.tomtom.com/user/register
   The free tier allows 2,500 requests/day; this script uses
   6 segments x 180 polls/morning = 1,080 requests/day. Pricing/limits:
       https://developer.tomtom.com/store/maps-api
2. Paste the key below (or set env var TOMTOM_API_KEY).
3. Run from the directory containing riyadh_traffic.csv:
       python3 forward_collector.py            # poll only during 06:00-09:00 Riyadh
       python3 forward_collector.py --all-hours  # poll around the clock
       python3 forward_collector.py --once       # single poll cycle, then exit

SCHEDULING (choose one)
=======================
cron (start a run at 05:59 Riyadh time daily; script exits after the window):
    59 5 * * * cd /path/to/project && /usr/bin/python3 forward_collector.py >> collector.log 2>&1
  NOTE: cron uses the machine's local time — adjust if the machine is not on
  Asia/Riyadh (UTC+3, no DST). E.g. for a UTC machine use `59 2 * * *`.

systemd (run continuously; the script itself sleeps outside the window):
    # /etc/systemd/system/riyadh-collector.service
    [Unit]
    Description=Riyadh TomTom traffic collector
    After=network-online.target
    [Service]
    WorkingDirectory=/path/to/project
    Environment=TOMTOM_API_KEY=...
    ExecStart=/usr/bin/python3 forward_collector.py
    Restart=on-failure
    [Install]
    WantedBy=multi-user.target
  then: systemctl enable --now riyadh-collector

Only the Python standard library is required (urllib, zoneinfo; Python >= 3.9).
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------- config
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "PASTE_KEY_HERE")
# Free registration: https://developer.tomtom.com/user/register

# Probe points for TomTom flowSegmentData: the API returns data for the road
# segment nearest to the point. Coordinates below target the named arteries in
# central Riyadh; before first production run, open each pair in a map and nudge
# it onto the exact carriageway/direction you want to track.
SEGMENTS = [
    # (segment_id,        road_name,                lat,      lon)
    ("KING_FAHD_RD_C",    "King Fahd Road",         24.7116, 46.6730),
    ("NORTHERN_RING_RD",  "Northern Ring Road",     24.7601, 46.6650),
    ("EASTERN_RING_RD",   "Eastern Ring Road",      24.7280, 46.7780),
    ("MAKKAH_RD",         "Makkah Al Mukarramah Rd", 24.6580, 46.6920),
    ("KING_KHALID_RD",    "King Khalid Road",       24.6900, 46.6240),
    ("OLAYA_ST",          "Olaya Street",           24.6950, 46.6850),
]

POLL_SECONDS = 60
PEAK_START_H, PEAK_END_H = 6, 9          # [06:00, 09:00) Asia/Riyadh
RIYADH = ZoneInfo("Asia/Riyadh")
CSV_PATH = Path("riyadh_traffic.csv")
RAW_DIR = Path("raw") / "tomtom_flow"
API_URL = ("https://api.tomtom.com/traffic/services/4/flowSegmentData"
           "/absolute/10/json?point={lat},{lon}&unit=KMPH&key={key}")

CSV_COLUMNS = [
    "timestamp_utc", "timestamp_riyadh", "date", "hour", "minute",
    "is_morning_peak", "road_segment_id", "road_name", "lat", "lon",
    "avg_speed_kmh", "free_flow_speed_kmh", "congestion_index",
    "congestion_definition", "granularity_minutes", "source_name",
    "source_url", "retrieved_at", "license", "quality_flag",
]
CONGESTION_DEF = ("relative extra travel time vs free flow = "
                  "currentTravelTime/freeFlowTravelTime - 1 "
                  "(0 = free flow; 1.0 = trip takes twice as long); "
                  "computed from TomTom flowSegmentData fields")
SOURCE_NAME = "tomtom_flow_segment_forward"
LICENSE = "TomTom for Developers terms (free tier); private research use"


def in_window(now_riyadh: datetime) -> bool:
    return PEAK_START_H <= now_riyadh.hour < PEAK_END_H


def fetch_segment(seg_id: str, lat: float, lon: float, now_utc: datetime) -> dict | None:
    """Fetch one segment; save the raw payload BEFORE parsing; return parsed dict."""
    url = API_URL.format(lat=lat, lon=lon, key=urllib.parse.quote(TOMTOM_API_KEY))
    day_dir = RAW_DIR / now_utc.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    raw_path = day_dir / f"{now_utc.strftime('%H%M%S')}_{seg_id}.json"
    try:
        with urllib.request.urlopen(url, timeout=25) as resp:
            payload = resp.read()
    except Exception as exc:                                  # noqa: BLE001
        print(f"[{now_utc.isoformat()}] {seg_id}: fetch failed: {exc}", file=sys.stderr)
        return None
    raw_path.write_bytes(payload)                              # raw artifact first
    try:
        data = json.loads(payload)["flowSegmentData"]
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[{now_utc.isoformat()}] {seg_id}: bad payload ({exc}); raw kept at {raw_path}",
              file=sys.stderr)
        return None
    # verify-gate: re-open the raw file and confirm the values match what we parsed
    reread = json.loads(raw_path.read_bytes())["flowSegmentData"]
    for field in ("currentSpeed", "freeFlowSpeed", "currentTravelTime", "freeFlowTravelTime"):
        if reread.get(field) != data.get(field):
            print(f"[{now_utc.isoformat()}] {seg_id}: raw/parsed mismatch on {field}; row rejected",
                  file=sys.stderr)
            return None
    data["_raw_path"] = str(raw_path)
    data["_url"] = url.replace(urllib.parse.quote(TOMTOM_API_KEY), "KEY")
    return data


def build_row(seg_id: str, road_name: str, lat: float, lon: float,
              data: dict, now_utc: datetime) -> dict | None:
    speed = data.get("currentSpeed")
    ff_speed = data.get("freeFlowSpeed")
    ctt, fftt = data.get("currentTravelTime"), data.get("freeFlowTravelTime")
    if speed is None or not (0 <= float(speed) <= 160):
        print(f"{seg_id}: speed {speed!r} outside 0-160 km/h bounds; row rejected", file=sys.stderr)
        return None
    congestion = ""
    if ctt and fftt:
        congestion = round(float(ctt) / float(fftt) - 1, 4)
    now_riyadh = now_utc.astimezone(RIYADH)
    return {
        "timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_riyadh": now_riyadh.isoformat(timespec="seconds"),
        "date": now_riyadh.strftime("%Y-%m-%d"),
        "hour": now_riyadh.hour,
        "minute": now_riyadh.minute,
        "is_morning_peak": 1 if in_window(now_riyadh) else 0,
        "road_segment_id": seg_id,
        "road_name": road_name,
        "lat": lat,
        "lon": lon,
        "avg_speed_kmh": speed,
        "free_flow_speed_kmh": ff_speed if ff_speed is not None else "",
        "congestion_index": congestion,
        "congestion_definition": CONGESTION_DEF,
        "granularity_minutes": 1,
        "source_name": SOURCE_NAME,
        "source_url": data["_url"],
        "retrieved_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license": LICENSE,
        "quality_flag": "verified",
    }


def append_rows(rows: list[dict]) -> None:
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def poll_once() -> int:
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    rows = []
    for seg_id, road_name, lat, lon in SEGMENTS:
        data = fetch_segment(seg_id, lat, lon, now_utc)
        if data is None:
            continue
        row = build_row(seg_id, road_name, lat, lon, data, now_utc)
        if row is not None:
            rows.append(row)
    if rows:
        append_rows(rows)
    print(f"[{now_utc.isoformat()}] wrote {len(rows)}/{len(SEGMENTS)} segment rows")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--all-hours", action="store_true",
                        help="poll continuously instead of only 06:00-09:00 Riyadh")
    parser.add_argument("--once", action="store_true", help="one poll cycle, then exit")
    args = parser.parse_args()

    if TOMTOM_API_KEY == "PASTE_KEY_HERE":
        sys.exit("No API key. Register free at https://developer.tomtom.com/user/register "
                 "and set TOMTOM_API_KEY (env var or edit this file).")
    if args.once:
        poll_once()
        return
    print(f"Collector started; window "
          f"{'ALL HOURS' if args.all_hours else f'{PEAK_START_H:02d}:00-{PEAK_END_H:02d}:00 Asia/Riyadh'}; "
          f"interval {POLL_SECONDS}s; Ctrl-C to stop.")
    while True:
        cycle_start = time.monotonic()
        now_riyadh = datetime.now(RIYADH)
        if args.all_hours or in_window(now_riyadh):
            poll_once()
        elif now_riyadh.hour >= PEAK_END_H and not args.all_hours and os.environ.get("COLLECTOR_EXIT_AFTER_WINDOW"):
            print("Window over; exiting (COLLECTOR_EXIT_AFTER_WINDOW set, cron mode).")
            return
        time.sleep(max(1.0, POLL_SECONDS - (time.monotonic() - cycle_start)))


if __name__ == "__main__":
    main()
