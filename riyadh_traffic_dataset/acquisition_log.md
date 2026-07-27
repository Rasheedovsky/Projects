# Acquisition Log — Riyadh Traffic Dataset

Autonomous data-acquisition loop. Zero fabrication. Each cycle runs 8 steps
(RELOAD, SENSE, RANK, PROBE, ACQUIRE, VERIFY GATE, INTEGRATE, UPDATE BELIEFS,
CHECKPOINT). State of record = the CSV/manifest files, not conversation memory.

**Session 1 — start 2026-07-27**
- Fresh run. Initialized `sources_manifest.csv` (16 seed sources),
  `riyadh_traffic.csv` (header only), `gaps_manifest.csv`, `data_dictionary.md`.
- Working dir: `/home/user/Projects/riyadh_traffic_dataset/`.
- No API keys provided. Commercial APIs requiring a key → `paywalled_or_keyed`.

---

## CYCLE 1 — Reconnaissance & environment probe
- **Hypothesis:** highest-value fine-grained sources are commercial APIs (TomTom/HERE); keyless historical access may exist.
- **Probes:** WebSearch on TomTom Move/Traffic Stats, Saudi Open Data, Zenodo, Kaggle.
- **Outcome:** TomTom Traffic Stats/Move → **keyed** (30-day trial + API key; per-directional-mile licensing) — VERIFIED. Saudi Open Data "speeds/congestion" claim traced to a *commercial* catalog (xmap.ai), not the gov portal.
- **Rows added:** 0. **Belief:** commercial_api p_success ↓ (keyed confirmed).

## CYCLE 2 — Egress-policy discovery (decisive constraint)
- **Hypothesis:** I can fetch data pages directly to extract structured values.
- **Probes:** WebFetch on TomTom index / trafficindex.org / Kaggle → all 403. curl domain sweep through the agent proxy.
- **Outcome:** **This session's network egress policy allows only GitHub hosts + package registries.** Every traffic-data host (tomtom.com, api.tomtom.com, kaggle.com, zenodo.org, dataverse.harvard.edu, open.data.gov.sa, od.data.gov.sa, arxiv.org, web.archive.org) returns proxy CONNECT 403 (`000` via curl). `raw.githubusercontent.com` = HTTP 200. WebFetch also routes through this proxy (Wikipedia 403). README `/root/.ccr/README.md`: "do not retry organization policy denials — report them."
- **Rows added:** 0. **Belief:** ALL non-GitHub source types → effectively unreachable this session; pivot to GitHub as sole data channel.

## CYCLE 3 — GitHub pivot: repo/code discovery
- **Hypothesis:** someone committed a Riyadh speed/congestion time series (e.g. a TomTom/HERE collector output) to a public repo.
- **Probes:** `search_repositories` ("Riyadh traffic" → 356 hits, all off-target); `search_code` for `freeFlowSpeed Riyadh`, `Riyadh currentSpeed extension:csv`, `Riyadh avg_speed extension:csv`.
- **Outcome:** Only `human06/nexus-globe` (live TomTom ingester, **no committed history**). CSV field searches → 0. Uber Movement: 13 US + 38 global cities, Riyadh unconfirmed, service shut down 2022, host blocked → no_riyadh.
- **Rows added:** 0. **Belief:** committed fine-grained Riyadh data on GitHub is scarce/absent.

## CYCLE 4 — GitHub deep sweep (near-misses found)
- **Probes:** `search_code` for `Riyadh congestion extension:json` (704, noise), `Riyadh "King Fahd" speed` (noise), `Riyadh traffic path:data extension:csv` (1008, mostly noise); fetched candidate raw files.
- **Outcome — three real datasets located and FETCHED to raw/:**
  1. `khaliddosari/saudi-road-safety-mlops` → `traffic_monthly.csv`: **monthly regional vehicle COUNTS** (Riyadh 2024, 29 count points). No speed, no congestion. README confirms sources = GASTAT + Ministry of Transport, aggregated to regional annual/monthly. → **off-target** (fails schema's speed/congestion requirement).
  2. `adkurylev/relocation_recsys` → `numbeo_traffic.csv`: one Riyadh row, Numbeo Traffic Index 144.67 / Time Index 30.82 min. Congestion-type index but **city-level, no speed, no timestamp** → cannot date without fabrication. → **near-miss, excluded.**
  3. `Wsh7Ash/gcc-smart-traffic`: has exact fields `avg_speed_kmh`+`congestion_index` for Riyadh but they are **SIMULATED** (`TrafficSimulator`/`CongestionPredictor`). → **excluded per zero-fabrication.**
- **Rows added:** 0 (no fabrication; near-misses documented in gaps_manifest G02/G04, simulated excluded).

## CYCLE 5 — Confirmation & stop
- **Probes:** `Riyadh "congestion_index" OR "speed_kmh"` → only the simulated repo; `clemensv/real-time-sources` catalog → no Saudi/Riyadh open road-traffic feed (nearest: Hong Kong 30s, UK WebTRIS 15min, Norway); pricing citations for TomTom & HERE.
- **Bayesian update (explicit):** per-type Beta posteriors after this session's failures —
  `commercial_api Beta(1,5) p=0.167`, `gov_portal Beta(1,4) p=0.200`,
  `academic_repo Beta(1,6) p=0.143`, `archive_snapshot Beta(1,2) p=0.333`.
  Every type's E[p_success] fell; no reachable+keyless+fine-grained argmax remains.
- **STOP CONDITION 1 (queue exhausted):** every source has a terminal status. → FINAL REPORT.
- **Total rows in riyadh_traffic.csv:** 0 (header only). This is the truthful result; the environment cannot deliver fabrication-free fine-grained Riyadh speed+congestion data. Forward path = `forward_collector.py` (user runs it with a free TomTom key).

---
