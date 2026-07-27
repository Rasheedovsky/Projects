#!/usr/bin/env python3
"""
forward_collector.py — Riyadh morning-peak traffic collector (FORWARD-LOOKING).

WHY THIS EXISTS
---------------
The historical objective (Jul 2021 -> Jul 2026 speed+congestion for Riyadh at
1-min..1-hour granularity) could NOT be met from this session's environment:
every fine-grained source is either commercial+keyed (TomTom/HERE/INRIX) or
behind a network egress policy that only permits GitHub. See final_report.md.

This collector lets YOU (on your own machine, with a free TomTom key) begin
building the dataset going FORWARD, writing rows in the EXACT same schema as
riyadh_traffic.csv. It fabricates nothing: it polls TomTom Flow Segment Data,
saves each raw JSON response to raw/ BEFORE parsing, and writes one row per
segment per poll from the retrieved values only.

TomTom free tier requires FREE registration (no credit card for the trial):
    Register:  https://developer.tomtom.com/  (create app -> get API key)
    Flow API:  https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data
    Free tier: ~2,500 requests/day typical (verify current limits on your dashboard).
Paste your key below.
"""

import os
import csv
import json
import time
import datetime as dt
from urllib.parse import quote
import urllib.request
import urllib.error

# ============================ CONFIG ========================================
TOMTOM_API_KEY = "PASTE_KEY_HERE"   # <-- get a free key at https://developer.tomtom.com/

OUT_CSV   = "riyadh_traffic.csv"    # same file/schema as the historical dataset
RAW_DIR   = "raw/tomtom_flow"       # raw JSON payloads saved here before parsing
POLL_SECONDS = 60                   # 1-minute granularity
PEAK_START_HOUR = 6                 # 06:00 Asia/Riyadh (inclusive)
PEAK_END_HOUR   = 9                 # up to 08:59:59; set to 24 to collect all day
UNIT = "KMPH"                       # TomTom returns speeds in km/h

# Asia/Riyadh is UTC+3 year-round (no DST).
RIYADH_TZ = dt.timezone(dt.timedelta(hours=3))

# Representative points on six major Riyadh corridors. TomTom returns the road
# SEGMENT containing each point. Coordinates are approximate — refine to the
# exact segment you care about using https://developer.tomtom.com/ map tools.
SEGMENTS = [
    # segment_id,           road_name,              lat,       lon
    ("KING_FAHD_RD",        "King Fahd Road",       24.69600,  46.68530),
    ("NORTHERN_RING_RD",    "Northern Ring Road",   24.77430,  46.65400),
    ("EASTERN_RING_RD",     "Eastern Ring Road",    24.73000,  46.79000),
    ("MAKKAH_RD",           "Makkah Al Mukarramah Road", 24.70500, 46.73000),
    ("KING_KHALID_RD",      "King Khalid Road",     24.79000,  46.63000),
    ("OLAYA_ST",            "Olaya Street",         24.69200,  46.68500),
]

# congestion_index defined FROM TomTom's own retrieved fields (documented):
CONGESTION_DEF = ("1 - currentSpeed/freeFlowSpeed (0=free-flow, ->1=gridlock); "
                  "derived from TomTom Flow Segment Data currentSpeed & freeFlowSpeed")

SCHEMA = ["timestamp_utc","timestamp_riyadh","date","hour","minute","is_morning_peak",
          "road_segment_id","road_name","lat","lon","avg_speed_kmh","free_flow_speed_kmh",
          "congestion_index","congestion_definition","granularity_minutes","source_name",
          "source_url","retrieved_at","license","quality_flag"]

FLOW_URL = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            "?point={lat}%2C{lon}&unit={unit}&openLr=false&key={key}")


def ensure_csv():
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(SCHEMA)


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def fetch_segment(lat, lon):
    """Fetch one segment; return (raw_bytes, parsed_json, url_without_key)."""
    url = FLOW_URL.format(lat=lat, lon=lon, unit=UNIT, key=quote(TOMTOM_API_KEY))
    url_public = url.replace(quote(TOMTOM_API_KEY), "***")
    req = urllib.request.Request(url, headers={"User-Agent": "riyadh-forward-collector/1.0"})
    for attempt in range(4):  # exponential backoff on transient errors
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            return raw, json.loads(raw.decode("utf-8")), url_public
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** (attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 3:
                time.sleep(2 ** (attempt + 1)); continue
            raise
    return None, None, url_public


def save_raw(seg_id, ts_utc, raw):
    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = ts_utc.strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RAW_DIR, f"{seg_id}_{stamp}.json")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def parse_row(seg_id, road_name, lat, lon, payload, ts_utc, url_public):
    """Build one schema row from RETRIEVED values only. None if unusable."""
    fsd = (payload or {}).get("flowSegmentData")
    if not fsd:
        return None
    cur = fsd.get("currentSpeed")
    ff  = fsd.get("freeFlowSpeed")
    if cur is None or ff is None:
        return None
    # bounds check (0..160 km/h)
    if not (0 <= cur <= 160) or not (0 < ff <= 160):
        return None
    congestion = round(1 - (cur / ff), 4)
    ts_ry = ts_utc.astimezone(RIYADH_TZ)
    is_peak = 1 if PEAK_START_HOUR <= ts_ry.hour < min(PEAK_END_HOUR, 9) else 0
    return [
        ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ts_ry.strftime("%Y-%m-%dT%H:%M:%S%z"),
        ts_ry.strftime("%Y-%m-%d"), ts_ry.hour, ts_ry.minute, is_peak,
        seg_id, road_name, lat, lon,
        cur, ff, congestion, CONGESTION_DEF, 1,
        "TomTom Flow Segment Data API (live)",
        url_public, ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "TomTom Traffic API (commercial; per-account terms)",
        "verified",   # each value re-read from the saved raw payload below
    ]


def verify_against_raw(row, raw_path):
    """VERIFY GATE: re-open the saved raw file and confirm the row's speeds
    match the payload exactly. Returns True/False."""
    try:
        with open(raw_path) as f:
            fsd = json.load(f)["flowSegmentData"]
        return (float(fsd["currentSpeed"]) == float(row[10]) and
                float(fsd["freeFlowSpeed"]) == float(row[11]))
    except Exception:
        return False


def in_peak_window():
    if PEAK_END_HOUR >= 24:
        return True
    h = now_utc().astimezone(RIYADH_TZ).hour
    return PEAK_START_HOUR <= h < PEAK_END_HOUR


def collect_once():
    ensure_csv()
    added = 0
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        for seg_id, road, lat, lon in SEGMENTS:
            ts = now_utc()
            try:
                raw, payload, url_pub = fetch_segment(lat, lon)
            except Exception as e:
                print(f"[{ts.isoformat()}] {seg_id}: fetch failed: {e}")
                continue
            if raw is None:
                continue
            raw_path = save_raw(seg_id, ts, raw)          # save BEFORE parse
            row = parse_row(seg_id, road, lat, lon, payload, ts, url_pub)
            if row is None:
                print(f"[{ts.isoformat()}] {seg_id}: no usable flow fields (skipped)")
                continue
            if not verify_against_raw(row, raw_path):     # VERIFY GATE
                print(f"[{ts.isoformat()}] {seg_id}: verify mismatch -> rejected")
                continue
            w.writerow(row); added += 1
            print(f"[{ts.isoformat()}] {seg_id}: {row[10]} km/h "
                  f"(free {row[11]}), congestion {row[12]}")
    return added


def main():
    if TOMTOM_API_KEY == "PASTE_KEY_HERE":
        raise SystemExit("Set TOMTOM_API_KEY first (free key: https://developer.tomtom.com/).")
    print("Riyadh forward collector started. Polling during "
          f"{PEAK_START_HOUR:02d}:00-{PEAK_END_HOUR:02d}:00 Asia/Riyadh every {POLL_SECONDS}s.")
    while True:
        if in_peak_window():
            n = collect_once()
            print(f"  wrote {n} rows -> {OUT_CSV}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

# ============================ SCHEDULING NOTES ==============================
# You must run this yourself; this environment cannot schedule anything.
#
# Option A — cron (runs the poller only during the morning peak, every minute):
#   Edit crontab:  crontab -e
#   # min hour dom mon dow  (times below are Asia/Riyadh; set CRON_TZ or convert)
#   CRON_TZ=Asia/Riyadh
#   * 6-8 * * *  cd /path/to/riyadh_traffic_dataset && /usr/bin/python3 forward_collector.py --once >> collector.log 2>&1
#   (For --once cron mode, replace main()'s while-loop with a single collect_once();
#    or keep the daemon form below and let it sleep outside the window.)
#
# Option B — systemd service (daemon form, self-gates to the peak window):
#   /etc/systemd/system/riyadh-collector.service
#     [Unit]
#     Description=Riyadh morning-peak traffic collector
#     After=network-online.target
#     [Service]
#     WorkingDirectory=/path/to/riyadh_traffic_dataset
#     Environment=TZ=Asia/Riyadh
#     ExecStart=/usr/bin/python3 /path/to/riyadh_traffic_dataset/forward_collector.py
#     Restart=always
#     [Install]
#     WantedBy=multi-user.target
#   sudo systemctl daemon-reload && sudo systemctl enable --now riyadh-collector
#
# Free-tier budget: 6 segments x 60 polls/hr x 3 hrs = 1,080 requests/morning.
# Stays under a typical 2,500/day free allowance. Widen PEAK_END_HOUR=24 to
# collect all day (then watch your quota). Deduplicate later on
# (timestamp_utc, road_segment_id, source_name) if you overlap runs.
