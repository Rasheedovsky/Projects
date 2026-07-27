#!/usr/bin/env python3
"""Cycle checkpoint: write terminal source statuses, Bayesian posteriors,
and the gaps manifest reflecting this session's findings.
Rebuilds sources_manifest.csv and gaps_manifest.csv from the recorded outcomes.
riyadh_traffic.csv is intentionally left header-only (zero fabricated rows)."""
import csv

# ---------------------------------------------------------------------------
# Per-TYPE Beta priors start (1,1). Each PROBED source contributes: success->
# alpha+1, failure->beta+1. This session produced only failures (no keyless,
# reachable, fine-grained speed+congestion source materialized).
# Failures counted per type (only sources actually probed this session):
TYPE_FAILURES = {
    "commercial_api": 4,   # TomTom Index web, TomTom Move/Stats, TomTom Flow, HERE
    "gov_portal":     3,   # Saudi Open Data, GASTAT, Ministry of Transport
    "academic_repo":  5,   # Kaggle, Zenodo, Harvard Dataverse, IEEE, GitHub sweep
    "archive_snapshot": 1, # Wayback (unreachable -> cannot probe -> failure)
}
TYPE_SUCCESSES = {k: 0 for k in TYPE_FAILURES}

def type_p(t):
    a = 1 + TYPE_SUCCESSES.get(t, 0)
    b = 1 + TYPE_FAILURES.get(t, 0)
    return a / (a + b)

# ---------------------------------------------------------------------------
# Final source records: (name, type, status, evidence_url, license, cost_note,
#                        per_source_alpha, per_source_beta, value_est, effort_est)
# status in {untried, probing, acquired, exhausted, paywalled_or_keyed,
#            no_riyadh, no_history}
SRC = [
 ("TomTom Traffic Index (city stats web)","commercial_api","no_history",
  "https://www.tomtom.com/traffic-index/ranking/",
  "Proprietary (site ToS)","Egress policy blocks tomtom.com (proxy CONNECT 403); page is city-level typical-week/yearly aggregate, not timestamped per-segment observations.",
  1,2,0.25,3),
 ("TomTom Move / Traffic Stats API","commercial_api","paywalled_or_keyed",
  "https://developer.tomtom.com/move-portal/guides/traffic-stats/introduction",
  "Commercial","30-day free trial then paid; licensing per directional mile. Requires account+API key (none provided). Host also egress-blocked. Signup: https://move.tomtom.com",
  1,2,0.90,3),
 ("TomTom Flow Segment Data API (live)","commercial_api","paywalled_or_keyed",
  "https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data",
  "Commercial (free tier w/ key)","Free tier requires free registration + API key (none provided). Live only (no history). Host egress-blocked. See forward_collector.py.",
  1,2,0.50,2),
 ("HERE Traffic API (historical flow)","commercial_api","paywalled_or_keyed",
  "https://www.here.com/get-started/pricing",
  "Commercial (freemium)","Freemium 250k txns/mo; Traffic API first 5k free then $2.50/1k. Requires free account+API key (none provided). Host egress-blocked.",
  1,2,0.80,3),
 ("INRIX","commercial_api","paywalled_or_keyed",
  "https://inrix.com/products/inrix-iq-analytics/",
  "Commercial","Enterprise-only, sales-quoted; no public keyless access. Not directly probed (host egress-blocked).",
  1,1,0.80,3),
 ("Google BigQuery public datasets","academic_repo","no_riyadh",
  "https://cloud.google.com/bigquery/public-data",
  "Varies","No public BigQuery dataset with Riyadh road speed/congestion. BigQuery endpoints egress-blocked/scoped in this session.",
  1,2,0.30,3),
 ("Saudi Open Data portal (open.data.gov.sa)","gov_portal","no_history",
  "https://open.data.gov.sa/",
  "Open (Saudi Open Data License)","Host egress-blocked (proxy 403). Evidence via khaliddosari/saudi-road-safety-mlops shows underlying open data = monthly regional vehicle COUNTS and road-asset inventories; no speed/congestion time series.",
  1,2,0.35,3),
 ("Royal Commission for Riyadh City","gov_portal","no_history",
  "https://www.rcrc.gov.sa/en/",
  "Unknown","Host egress-blocked. No public keyless speed/congestion dataset identified.",
  1,1,0.30,3),
 ("Ministry of Transport & Logistic Services (KSA)","gov_portal","no_history",
  "https://mot.gov.sa/en/OpenData",
  "Open","Host egress-blocked. Per mlops-repo evidence, MoT open data = point-level traffic sensors / road specs (regional), not fine-grained speed+congestion series.",
  1,2,0.25,3),
 ("GASTAT (General Authority for Statistics)","gov_portal","no_history",
  "https://www.stats.gov.sa/en/",
  "Open","Host egress-blocked. GASTAT Land/Road Transport stats = annual/monthly regional aggregates (counts), no speed/congestion.",
  1,2,0.15,3),
 ("Kaggle (Riyadh traffic search)","academic_repo","paywalled_or_keyed",
  "https://www.kaggle.com/datasets/majedalhulayel/traffic-index-in-saudi-arabia-and-middle-east",
  "Varies (per-dataset)","Dataset 'Traffic Index in Saudi Arabia and Middle East' exists but kaggle.com egress-blocked (403); direct download needs Kaggle account/API token (none provided).",
  1,2,0.40,3),
 ("Zenodo (Riyadh traffic search)","academic_repo","no_riyadh",
  "https://zenodo.org/search?q=Riyadh%20traffic",
  "Varies (often CC)","Host egress-blocked. Search surfaced no Riyadh road speed/congestion dataset; hits were audio/other cities.",
  1,2,0.40,2),
 ("IEEE DataPort (Riyadh traffic)","academic_repo","paywalled_or_keyed",
  "https://ieee-dataport.org/",
  "Varies (mostly subscription)","Host egress-blocked; most datasets require IEEE subscription. No keyless Riyadh speed/congestion dataset identified.",
  1,1,0.35,3),
 ("Harvard Dataverse (Riyadh traffic)","academic_repo","no_riyadh",
  "https://dataverse.harvard.edu/",
  "Varies (often CC0)","Host egress-blocked. No Riyadh road speed/congestion dataset identified via search.",
  1,2,0.35,2),
 ("Academic papers w/ Riyadh speed data","academic_repo","no_history",
  "https://arxiv.org/abs/2506.01974",
  "Varies","Relevant papers exist (e.g. Dubai-vs-Riyadh AI mobility study) but publish analyses/figures, not open timestamped speed+congestion tables; hosts (arXiv/publishers) egress-blocked.",
  1,2,0.30,4),
 ("Wayback Machine congestion dashboards","archive_snapshot","no_history",
  "https://web.archive.org/",
  "Archive","web.archive.org egress-blocked (proxy 403); cannot retrieve snapshots. Live congestion dashboards are JS/API-driven and rarely archived with structured data.",
  1,2,0.30,4),
 # ---- sources discovered on GitHub this session (the only reachable data host) ----
 ("GitHub: khaliddosari/saudi-road-safety-mlops","gov_portal","exhausted",
  "https://raw.githubusercontent.com/khaliddosari/saudi-road-safety-mlops/main/data/processed/traffic_monthly.csv",
  "MIT (repo) / GASTAT+MoT (data)","REACHED & FETCHED. Contains monthly regional vehicle COUNTS (Riyadh 2024, 29 count points) — NO speed, NO congestion index. Off-target: does not meet schema's speed/congestion requirement. Saved to raw/.",
  1,1,0.10,2),
 ("GitHub: adkurylev/relocation_recsys (Numbeo scrape)","academic_repo","exhausted",
  "https://raw.githubusercontent.com/adkurylev/relocation_recsys/master/data/raw/numbeo/numbeo_traffic.csv",
  "Numbeo ToS (crowd-sourced)","REACHED. Single Riyadh row: Numbeo Traffic Index 144.67, Time Index 30.82 min. Congestion-type composite but CITY-LEVEL, NO speed, and NO timestamp in artifact -> cannot assign a verifiable observation time without fabrication. Excluded from CSV; documented as near-miss.",
  1,1,0.10,2),
 ("GitHub: Wsh7Ash/gcc-smart-traffic (SIMULATED)","academic_repo","exhausted",
  "https://github.com/Wsh7Ash/gcc-smart-traffic",
  "N/A","REACHED. Has exactly the target fields (avg_speed_kmh, congestion_index) for Riyadh BUT they are SYNTHETIC (TrafficSimulator/CongestionPredictor output). Excluded per PRIME DIRECTIVE (zero fabrication).",
  1,1,0.00,2),
 ("GitHub: human06/nexus-globe (live TomTom ingester)","commercial_api","paywalled_or_keyed",
  "https://github.com/human06/nexus-globe",
  "Repo license / TomTom data","REACHED. Code ingests TomTom live flow for cities incl. Riyadh (current/free-flow speed) but commits NO historical data and needs a TomTom API key to run.",
  1,1,0.20,2),
 ("GitHub: clemensv/real-time-sources (feed catalog)","archive_snapshot","exhausted",
  "https://github.com/clemensv/real-time-sources/tree/main/tools/candidates/road-traffic",
  "Catalog (MIT)","REACHED. Catalogs open real-time feeds worldwide. No Saudi/Riyadh road-traffic feed with speed/congestion; nearest open road-traffic feeds are other countries (Hong Kong TD 30s, UK WebTRIS 15min, Norway). Confirms scarcity of open Gulf traffic feeds.",
  1,1,0.10,3),
]

def compute_score(t, a, b, value_est, effort_est):
    # Policy: score = E[p_success]_type * value_est / effort_est
    return round(type_p(t) * value_est / effort_est, 4)

def write_sources(path="sources_manifest.csv", cycle=5):
    cols = ["source_name","source_type","status","evidence_url","license","cost_note",
            "alpha","beta","value_est","effort_est","score","last_cycle"]
    with open(path,"w",newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for name,typ,status,url,lic,cost,a,b,ve,ee in SRC:
            w.writerow([name,typ,status,url,lic,cost,a,b,ve,ee,
                        compute_score(typ,a,b,ve,ee),cycle])
    print(f"WROTE {path}: {len(SRC)} sources")
    print("Per-type E[p_success] posteriors:")
    for t in TYPE_FAILURES:
        print(f"  {t:16s} Beta(1,{1+TYPE_FAILURES[t]}) -> p={type_p(t):.3f}")

# ---------------------------------------------------------------------------
GAPS = [
 # gap_id, scope, date_range, hours, granularity_target, reason, source_attempted, evidence_url, cycle_logged
 ("G01","Riyadh all road segments","2021-07..2026-07","all","1-60 min",
  "No keyless, reachable source of fine-grained speed+congestion. Commercial APIs (TomTom/HERE/INRIX) are keyed AND their hosts are blocked by this session's egress policy (proxy allows only GitHub + package registries).",
  "TomTom Move/Stats, TomTom Flow, HERE Traffic, INRIX","https://developer.tomtom.com/move-portal/guides/traffic-stats/introduction","1-5"),
 ("G02","Riyadh (official open data)","2022..2024","all","monthly (coarsest)",
  "Saudi official open data (GASTAT / Ministry of Transport / Saudi Open Data) publishes only monthly/annual REGIONAL vehicle COUNTS and road-asset inventories — no vehicle speed and no congestion index. Also egress-blocked this session.",
  "GASTAT, Ministry of Transport, Saudi Open Data","https://raw.githubusercontent.com/khaliddosari/saudi-road-safety-mlops/main/data/processed/traffic_monthly.csv","4"),
 ("G03","Riyadh 06:00-09:00 morning peak","2021-07..2026-07","06-09","1-5 min",
  "Priority morning-peak window: zero timestamped speed/congestion observations obtainable. No open per-segment feed for Riyadh exists on the only reachable host (GitHub); simulated data excluded.",
  "GitHub exhaustive search; academic repos","https://github.com/Wsh7Ash/gcc-smart-traffic","3-5"),
 ("G04","Riyadh congestion index (city-level)","unknown (undated scrape)","n/a","snapshot",
  "A real Numbeo Traffic Index value for Riyadh was found but has NO timestamp in the artifact and no speed field; assigning an observation time would be fabrication. Excluded. User can pull dated history from numbeo.com (egress-blocked here).",
  "GitHub: adkurylev/relocation_recsys (Numbeo)","https://raw.githubusercontent.com/adkurylev/relocation_recsys/master/data/raw/numbeo/numbeo_traffic.csv","4"),
 ("G05","Archived live dashboards","2021-07..2026-07","all","varies",
  "Wayback Machine and live congestion dashboards unreachable (web.archive.org egress-blocked). Cannot mine archived snapshots.",
  "Wayback Machine","https://web.archive.org/","5"),
]

def write_gaps(path="gaps_manifest.csv"):
    cols = ["gap_id","scope","date_range","hours","granularity_target","reason",
            "source_attempted","evidence_url","cycle_logged"]
    with open(path,"w",newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for row in GAPS: w.writerow(row)
    print(f"WROTE {path}: {len(GAPS)} documented gaps")

if __name__ == "__main__":
    write_sources()
    write_gaps()
    # verify data CSV still header-only (zero fabricated rows)
    with open("riyadh_traffic.csv") as f:
        n = sum(1 for _ in f) - 1
    print(f"riyadh_traffic.csv data rows = {n} (must be 0 — zero fabrication)")
