# Acquisition Log — Riyadh Traffic Dataset

Session 1 start: 2026-07-27 (UTC). Fresh run — no prior state files found.
Working dir: repo `rasheedovsky/projects`, branch `claude/riyadh-traffic-dataset-asmfnq`.

---

## Cycle 1 — Environment sense + archive-snapshot probe

**Hypothesis:** Wayback Machine snapshots of TomTom's live ranking API
(`api.midway.tomtom.com/ranking/live/SAU%2Friyadh`) hold timestamped Riyadh congestion
values across 2021–2026 and are keyless — the highest-value keyless funnel.

**Probe & outcome:**
- Container egress: **every** external host tested returns proxy CONNECT 403
  (policy denial) — archive.org, tomtom.com, api.midway.tomtom.com, open.data.gov.sa,
  zenodo.org, data.mendeley.com, dataverse.harvard.edu, kaggle.com,
  timetravel.mementoweb.org, datasource.kapsarc.org, even example.com.
  Only package registries (pypi/npm/crates) are allowlisted. Evidence:
  `raw/environment/host_reachability_2026-07-27.txt`, `raw/environment/proxy_status_2026-07-27.json`.
- Server-side WebFetch tool: web.archive.org hard-blocked ("unable to fetch");
  all other tested hosts return HTTP 403. **WebFetch unusable for any host.**
- WebSearch tool: **working** — the only acquisition channel this session.
- Per proxy README, org-policy denials must be reported, not routed around. Complied:
  no attempt to tunnel via third-party services.

**Rows added:** 0.
**Bayesian update:** archive_snapshot failures (wayback midway API, wayback index pages):
p_success(archive_snapshot) 1/2 → 1/4. All direct-download source types are capped by the
network block; expected yield now concentrated in WebSearch-snippet aggregates.
New argmax: parallel WebSearch probe sweep across all families (workflow `wf_dc8c2a9d-a01`).

**Statuses set:** wayback_tomtom_midway_live_api → exhausted (blocked);
wayback_tomtom_index_pages → exhausted (blocked); google_bigquery_public_datasets →
paywalled_or_keyed (GCP credentials required; no key provided).

**Gap logged:** G001 — entire 2021-07→2026-07 window at ≤60-min granularity, pending
sweep results (aggregates may still land; sub-hourly obs cannot).

---
## Cycle 2 — Parallel SENSE/PROBE sweep (workflow wf_dc8c2a9d-a01)

**Hypothesis:** with direct fetching blocked, WebSearch snippets can still (a) resolve
availability/terms for every candidate family with citations, and (b) surface concrete
Riyadh values published on open pages.

**Probe:** 6 parallel agents, 122 searches total, families: TomTom index, INRIX/Numbeo,
Saudi gov portals, academic repositories, academic papers, commercial pricing.
Raw output: `raw/websearch_sweep/probe_sweep_wf_dc8c2a9d_2026-07-27.json` (+ journal).

**Outcome (statuses):**
- VERIFIED empty of Riyadh traffic data: Zenodo, Mendeley Data, IEEE DataPort,
  Harvard Dataverse (domain-restricted probes) → `no_riyadh`. Figshare `no_riyadh` (weak).
- Keyed/paywalled: TomTom Move/Traffic Stats (archive to 2008 — best paid option),
  HERE (freemium 250k tx/mo), INRIX Roadway Analytics (quote-only), Kaggle mirrors
  (free account needed; bwandowando TomTom scrape updated 2026-06-21 incl. Riyadh),
  trafficindex.org Premium (monthly Riyadh 2017–2025), xmap.ai (sales-gated).
- Disqualified: Google Maps Routes/Distance Matrix — `departure_time ... cannot be in
  the past` → no retrospective data at any price. Otonomo/Wejo defunct 2023.
- Gov portals: no speed/congestion dataset verifiable anywhere (assets/signs,
  intersection inventories, count-station volumes, accident stats only).
  Riyadh Municipality Urban Data Center is the strongest lead for a network-open re-run.
- Academic: arXiv 2304.00192 proves per-segment HERE-derived Riyadh speed data existed
  for ~Jul–Oct 2022, but it is corporate-proprietary (ELM Research), never published.
- Quotable Riyadh aggregates surfaced (TomTom 2021/2024/2025, AGBI 2024 blend, Numbeo
  current) → candidates for Cycle 3 verification.

**Rows added:** 0 (probe cycle).
**Bayesian update:** commercial_api α1→4 (3 snippet-acquirable sources), β1→9;
p 0.5 → 0.31. gov_portal Beta(1,7): p 0.5 → 0.125. academic_repo Beta(1,10):
p 0.5 → 0.09. archive_snapshot stays Beta(1,3) = 0.25.
New argmax: verify-and-integrate the snippet aggregates (only non-terminal work left).

## Cycle 3 — VERIFY GATE + INTEGRATE

**Hypothesis:** the 5 candidate value-sets survive independent re-search; each value
string must exact-match a saved raw artifact before its row is written.

**Verification (6 independent searches by main agent, saved verbatim to
`raw/websearch_verify/V1..V6`):**
- PASS TomTom 2021: congestion 23%, rank 162 (V1; ties to 2021 report PDF mirror).
- PASS TomTom 2024: 71 h rush-hour loss (V2; cross-consistent: 66 h + 5 h 06 m = 71 h 06 m → stored 71.1).
- PASS TomTom 2025: avg speed 24.8 km/h + 66 h rush-hour loss (V3).
- PASS AGBI/INRIX+TomTom 2024 blend: 34 h annual, 58 h rush-hour (V4).
- PASS Numbeo current snapshot 156.69 (V6 + second sighting in sweep artifact).
- REJECTED — logged, never written: TomTom "90.4% congestion" (internally contradicted;
  implausible vs 23% in 2021; one pass reported the field as N/A); AGBI "28% vs
  free-flow" (failed re-verification, V4); Arab News "52 h annually" (no stated
  methodology); Numbeo Time Index 33.16 (single sighting); "123 km/h Al Kharj–Riyadh"
  (unattributable); MDPI ">45 min commute" (vague, undated); simulation outputs
  (MDPI Future Internet AV study — not observations).

**Gate execution:** programmatic exact-substring match of every value against its raw
artifact + bounds checks (speed 0–160; scales per source) — all passed.
**Rows added:** 7 (all `source_aggregate`, granularity 525600 min). Total rows: 7.
**Bayesian update:** commercial_api successes already counted in Cycle 2 posterior;
no further belief shift. Queue state after integration: EVERY source terminal.

## Cycle 4 — CHECKPOINT & STOP

**Stop condition 1 (queue exhausted)** triggered: all 29 manifest sources terminal
(`acquired` 3, `paywalled_or_keyed` 8, `no_riyadh` 6, `no_history` 1, `exhausted` 11).
Cycles used: 4 of 12. Coverage of 06:00–09:00 window at ≤60-min granularity: **0.0%**
(see gaps G001–G007). Forward collector delivered (`forward_collector.py`).
Final report: `FINAL_REPORT.md`. All state flushed and pushed to
`claude/riyadh-traffic-dataset-asmfnq`.
