#!/usr/bin/env python3
"""Initialize state files for the Riyadh traffic acquisition loop.
Idempotent-ish: only writes files that do not already exist (resume-safe)."""
import csv, os

# ---- OUTPUT SCHEMA (data CSV) ----
DATA_COLS = [
    "timestamp_utc","timestamp_riyadh","date","hour","minute","is_morning_peak",
    "road_segment_id","road_name","lat","lon","avg_speed_kmh","free_flow_speed_kmh",
    "congestion_index","congestion_definition","granularity_minutes","source_name",
    "source_url","retrieved_at","license","quality_flag",
]

# ---- SOURCES MANIFEST ----
SRC_COLS = ["source_name","source_type","status","evidence_url","license","cost_note",
            "alpha","beta","value_est","effort_est","score","last_cycle"]

# value_est = granularity_weight * expected_fraction_of_0609_window_covered
# granularity weights: 1min=1.0, 5min=0.8, hourly=0.5, daily=0.1
# All Beta priors per source_type start (1,1); per-source alpha/beta start (1,1) too.
SEEDS = [
    # name, type, status, evidence_url, license, cost_note, value_est, effort_est
    ("TomTom Traffic Index (city stats web)","commercial_api","untried","","","",0.25,3),
    ("TomTom Move / Traffic Stats API","commercial_api","untried","","","",0.90,3),
    ("TomTom Flow Segment Data API (live)","commercial_api","untried","","","",0.50,2),
    ("HERE Traffic API (historical flow)","commercial_api","untried","","","",0.80,3),
    ("INRIX","commercial_api","untried","","","",0.80,3),
    ("Google BigQuery public datasets","academic_repo","untried","","","",0.30,3),
    ("Saudi Open Data portal (open.data.gov.sa)","gov_portal","untried","","","",0.35,3),
    ("Royal Commission for Riyadh City","gov_portal","untried","","","",0.30,3),
    ("Ministry of Transport & Logistic Services (KSA)","gov_portal","untried","","","",0.25,3),
    ("GASTAT (General Authority for Statistics)","gov_portal","untried","","","",0.15,3),
    ("Kaggle (Riyadh traffic search)","academic_repo","untried","","","",0.40,3),
    ("Zenodo (Riyadh traffic search)","academic_repo","untried","","","",0.40,2),
    ("IEEE DataPort (Riyadh traffic)","academic_repo","untried","","","",0.35,3),
    ("Harvard Dataverse (Riyadh traffic)","academic_repo","untried","","","",0.35,2),
    ("Academic papers w/ Riyadh speed data","academic_repo","untried","","","",0.30,4),
    ("Wayback Machine congestion dashboards","archive_snapshot","untried","","","",0.30,4),
]

def compute_score(alpha, beta, value_est, effort_est):
    p = alpha/(alpha+beta)
    return round(p*value_est/effort_est, 4)

def init_sources(path):
    if os.path.exists(path):
        print(f"EXISTS: {path} (resume) — not overwriting")
        return
    with open(path,"w",newline="") as f:
        w = csv.writer(f); w.writerow(SRC_COLS)
        for name,typ,status,url,lic,cost,ve,ee in SEEDS:
            a,b = 1,1
            w.writerow([name,typ,status,url,lic,cost,a,b,ve,ee,compute_score(a,b,ve,ee),0])
    print(f"WROTE: {path} ({len(SEEDS)} seed sources)")

def init_data(path):
    if os.path.exists(path):
        print(f"EXISTS: {path} (resume) — not overwriting"); return
    with open(path,"w",newline="") as f:
        csv.writer(f).writerow(DATA_COLS)
    print(f"WROTE: {path} (header only, 0 rows)")

def init_gaps(path):
    if os.path.exists(path):
        print(f"EXISTS: {path} (resume) — not overwriting"); return
    cols = ["gap_id","scope","date_range","hours","granularity_target","reason",
            "source_attempted","evidence_url","cycle_logged"]
    with open(path,"w",newline="") as f:
        csv.writer(f).writerow(cols)
    print(f"WROTE: {path} (header only)")

if __name__ == "__main__":
    init_sources("sources_manifest.csv")
    init_data("riyadh_traffic.csv")
    init_gaps("gaps_manifest.csv")
    print("done")
