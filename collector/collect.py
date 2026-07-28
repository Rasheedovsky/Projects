#!/usr/bin/env python3
"""Riyadh intraday traffic collector — runs inside GitHub Actions (full network access).

Modes:
  snapshot (default) : one intraday observation cycle — keyless TomTom live index +
                       dailyStats refresh + candidate-endpoint probe + page mirrors;
                       plus per-artery segment speeds if TOMTOM_KEY secret is set.
  wayback            : one-shot historical harvest of Wayback Machine snapshots of the
                       keyless TomTom ranking API for Riyadh (real timestamped intraday
                       observations, ~2020-2024). Re-runnable; skips already-fetched.

Zero fabrication: every network response is stored verbatim under collector/data/raw/
with a SHA-256 recorded in collector/data/raw_hashes.csv before any parsing.
All parsing is fail-soft: a failed source never blocks the others.
"""
import csv, hashlib, json, os, re, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
os.makedirs(RAW, exist_ok=True)

NOW = datetime.now(timezone.utc)
STAMP = NOW.strftime("%Y%m%dT%H%M%SZ")
UA = {"User-Agent": "riyadh-traffic-research-collector/1.0 (github.com/rasheedovsky/projects; academic use; contact: repo issues)"}

CITY_LAT, CITY_LON = 24.7136, 46.6753
LIVE_URL = "https://api.midway.tomtom.com/ranking/live/SAU%2Friyadh"
DAILY_URL = "https://api.midway.tomtom.com/ranking/dailyStats/SAU_riyadh"
CANDIDATE_URLS = {
    "liveWeekly": "https://api.midway.tomtom.com/ranking/liveWeekly/SAU%2Friyadh",
    "weeklyStats": "https://api.midway.tomtom.com/ranking/weeklyStats/SAU_riyadh",
    "hourlyStats": "https://api.midway.tomtom.com/ranking/hourlyStats/SAU_riyadh",
    "liveHourly": "https://api.midway.tomtom.com/ranking/liveHourly/SAU%2Friyadh",
    "stats": "https://api.midway.tomtom.com/ranking/stats/SAU_riyadh",
}
PAGE_MIRRORS = {
    "tomtom_riyadh_page": "https://www.tomtom.com/traffic-index/city/riyadh",
    "inrix_riyadh_scorecard": "https://inrix.com/scorecard-city/?city=Riyadh&index=31",
    "numbeo_riyadh": "https://www.numbeo.com/traffic/in/Riyadh",
}
# Major Riyadh arteries for keyed flow-segment speeds (point = on-road probe location)
ARTERIES = [
    ("king_fahd_rd_center", 24.7136, 46.6753),
    ("northern_ring_rd", 24.7793, 46.6879),
    ("eastern_ring_rd", 24.7460, 46.7710),
    ("makkah_al_mukarramah_rd", 24.6870, 46.6860),
    ("king_khalid_rd", 24.6210, 46.5960),
    ("olaya_st", 24.6950, 46.6850),
]


def fetch(url, timeout=45, binary=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    return b if binary else b.decode("utf-8", errors="replace")


def save_raw(name, content):
    if isinstance(content, str):
        content = content.encode("utf-8")
    path = os.path.join(RAW, name)
    with open(path, "wb") as f:
        f.write(content)
    h = hashlib.sha256(content).hexdigest()
    hp = os.path.join(DATA, "raw_hashes.csv")
    new = not os.path.exists(hp)
    with open(hp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["fetched_utc", "file", "sha256", "bytes"])
        w.writerow([NOW.isoformat(), name, h, len(content)])
    return path


def append_rows(csv_name, header, rows):
    if not rows:
        return
    path = os.path.join(DATA, csv_name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def find_records(obj):
    """Yield every dict anywhere in obj that carries a TrafficIndex-style payload."""
    if isinstance(obj, dict):
        keys_l = {k.lower() for k in obj}
        if any(k in keys_l for k in ("trafficindexlive", "trafficindexhistoric", "congestion")):
            yield obj
        for v in obj.values():
            yield from find_records(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_records(v)


def gv(d, *names):
    low = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return ""


def collect_live():
    txt = fetch(LIVE_URL)
    save_raw(f"live_{STAMP}.json", txt)
    rows = []
    for rec in find_records(json.loads(txt)):
        live = gv(rec, "TrafficIndexLive")
        hist = gv(rec, "TrafficIndexHistoric")
        upd = gv(rec, "UpdateTime", "UpdateTimeUTC")
        if live == "" and hist == "":
            continue
        upd_iso = ""
        try:
            upd_iso = datetime.fromtimestamp(int(upd) / 1000, tz=timezone.utc).isoformat()
        except (ValueError, TypeError, OSError):
            pass
        rows.append([NOW.isoformat(), upd_iso, live, hist, f"live_{STAMP}.json"])
    append_rows("live_snapshots.csv",
                ["fetched_utc", "update_time_utc", "traffic_index_live", "traffic_index_historic", "raw_file"],
                rows)
    print(f"live: {len(rows)} record(s)")


def collect_daily():
    txt = fetch(DAILY_URL)
    save_raw(f"dailystats_{STAMP}.json", txt)
    rows = []
    for rec in find_records(json.loads(txt)):
        d = gv(rec, "date", "day", "DateTime")
        c = gv(rec, "congestion")
        if d != "" and c != "":
            rows.append([str(d), str(c)])
    if rows:
        path = os.path.join(DATA, "dailystats_riyadh.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "congestion"])
            w.writerows(sorted(set(map(tuple, rows))))
    print(f"dailyStats: {len(rows)} rows (full refresh)")


def collect_candidates():
    hits = []
    for name, url in CANDIDATE_URLS.items():
        try:
            txt = fetch(url, timeout=25)
            if txt.strip():
                save_raw(f"probe_{name}_{STAMP}.json", txt)
                hits.append(name)
        except Exception as e:
            print(f"probe {name}: {type(e).__name__}")
        time.sleep(1)
    print(f"candidate endpoints alive: {hits}")


def collect_pages():
    for name, url in PAGE_MIRRORS.items():
        try:
            txt = fetch(url, timeout=60)
            save_raw(f"page_{name}_{STAMP}.html", txt[:3_000_000])
            print(f"page {name}: {len(txt)} bytes")
        except Exception as e:
            print(f"page {name}: {type(e).__name__}")
        time.sleep(1)


def collect_flow_segments():
    key = os.environ.get("TOMTOM_KEY", "").strip()
    if not key:
        print("flow segments: TOMTOM_KEY not set — skipped (add repo secret to enable "
              "per-artery speed+location collection)")
        return
    rows = []
    for seg, lat, lon in ARTERIES:
        url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
               f"?point={lat},{lon}&unit=KMPH&key={key}")
        try:
            txt = fetch(url, timeout=30)
            save_raw(f"flow_{seg}_{STAMP}.json", txt)
            d = json.loads(txt).get("flowSegmentData", {})
            coords = d.get("coordinates", {}).get("coordinate", [])
            c0 = coords[0] if coords else {}
            rows.append([NOW.isoformat(), seg, lat, lon,
                         d.get("currentSpeed", ""), d.get("freeFlowSpeed", ""),
                         d.get("currentTravelTime", ""), d.get("freeFlowTravelTime", ""),
                         d.get("confidence", ""), d.get("frc", ""),
                         c0.get("latitude", ""), c0.get("longitude", ""),
                         f"flow_{seg}_{STAMP}.json"])
        except Exception as e:
            print(f"flow {seg}: {type(e).__name__}")
        time.sleep(1)
    append_rows("flow_segments.csv",
                ["fetched_utc", "segment", "probe_lat", "probe_lon", "current_speed_kmh",
                 "free_flow_speed_kmh", "current_travel_time_s", "free_flow_travel_time_s",
                 "confidence", "frc", "seg_lat", "seg_lon", "raw_file"],
                rows)
    print(f"flow segments: {len(rows)} rows")


def wayback_harvest(max_fetches=400, delay=1.5):
    """Pull historical snapshots of the keyless ranking API (+ Riyadh index pages)."""
    wb = os.path.join(DATA, "wayback")
    os.makedirs(wb, exist_ok=True)
    targets = [
        ("api.midway.tomtom.com/ranking/live/SAU*", "json"),
        ("api.midway.tomtom.com/ranking/dailyStats/SAU*", "json"),
        ("tomtom.com/en_gb/traffic-index/riyadh-traffic*", "html"),
        ("tomtom.com/traffic-index/riyadh-traffic*", "html"),
    ]
    captures = []
    for pat, kind in targets:
        cdx = (f"https://web.archive.org/cdx/search/cdx?url={urllib.parse.quote(pat)}"
               f"&matchType=prefix&output=json&filter=statuscode:200&collapse=timestamp:10")
        try:
            txt = fetch(cdx, timeout=90)
            save_raw(f"cdx_{re.sub('[^A-Za-z0-9]+', '_', pat)[:60]}_{STAMP}.json", txt)
            data = json.loads(txt) if txt.strip() else []
            for row in data[1:]:
                ts, orig = row[1], row[2]
                if "riyadh" not in orig.lower() and "SAU" not in orig:
                    continue
                captures.append((ts, orig, kind))
        except Exception as e:
            print(f"cdx {pat}: {type(e).__name__}: {e}")
        time.sleep(delay)
    print(f"wayback captures listed: {len(captures)}")
    fetched = 0
    for ts, orig, kind in sorted(set(captures)):
        fn = f"{ts}_{re.sub('[^A-Za-z0-9]+', '_', orig)[:80]}.{kind}"
        out = os.path.join(wb, fn)
        if os.path.exists(out):
            continue
        if fetched >= max_fetches:
            print(f"cap {max_fetches} reached — re-run wayback mode to continue")
            break
        try:
            content = fetch(f"https://web.archive.org/web/{ts}id_/{orig}", timeout=90, binary=True)
            with open(out, "wb") as f:
                f.write(content[:3_000_000])
            fetched += 1
        except Exception as e:
            print(f"wb {ts} {orig[:50]}: {type(e).__name__}")
        time.sleep(delay)
    print(f"wayback snapshots downloaded this run: {fetched}")
    # Parse every live-API JSON snapshot into a tidy observations table
    rows = []
    for fn in sorted(os.listdir(wb)):
        if "_ranking_live_" not in fn or not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(wb, fn), encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
        except Exception:
            continue
        snap_ts = fn.split("_")[0]
        for rec in find_records(obj):
            live, hist, upd = gv(rec, "TrafficIndexLive"), gv(rec, "TrafficIndexHistoric"), gv(rec, "UpdateTime")
            upd_iso = ""
            try:
                upd_iso = datetime.fromtimestamp(int(upd) / 1000, tz=timezone.utc).isoformat()
            except (ValueError, TypeError, OSError):
                pass
            if live != "" or hist != "":
                rows.append([snap_ts, upd_iso, live, hist, fn])
    if rows:
        with open(os.path.join(DATA, "wayback_live_observations.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wayback_snapshot_ts", "update_time_utc", "traffic_index_live",
                        "traffic_index_historic", "raw_file"])
            w.writerows(rows)
    print(f"wayback live observations parsed: {len(rows)}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    steps = ([collect_live, collect_daily, collect_candidates, collect_pages, collect_flow_segments]
             if mode == "snapshot" else [wayback_harvest])
    failures = 0
    for step in steps:
        try:
            step()
        except Exception as e:
            failures += 1
            print(f"STEP FAILED {step.__name__}: {type(e).__name__}: {e}")
    print(f"done mode={mode} failures={failures}")


if __name__ == "__main__":
    import urllib.parse  # used in wayback_harvest
    main()
