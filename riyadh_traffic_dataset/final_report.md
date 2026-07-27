# FINAL REPORT — Riyadh Road-Traffic Dataset Acquisition

**Session date:** 2026-07-27 · **Cycles run:** 5 · **Stop condition:** #1 (queue exhausted — every source terminal).
**Fabricated rows written:** 0. **Real observation rows meeting the objective:** 0.

> Bottom line: In this execution environment it was **not possible to obtain a
> single fabrication-free row** of fine-grained (1-min…1-hour) Riyadh vehicle
> **speed + congestion** covering Jul-2021→Jul-2026. Two hard walls made this a
> truthful negative result rather than a data pull:
> 1. **Network egress policy** — this session can only reach GitHub + package
>    registries. Every traffic-data host is blocked at the proxy (CONNECT 403).
> 2. **The data doesn't exist openly** — where it exists it is commercial+keyed
>    (TomTom/HERE/INRIX); where it is open (Saudi GASTAT/MoT) it is only monthly
>    regional *counts*, never speed/congestion.
> Per the PRIME DIRECTIVE, the dataset is left empty rather than padded with
> estimates. The deliverable is this honest map + `forward_collector.py`.

---

## 1. Coverage matrix (year × month × best granularity)

`·` = no data retrieved. Every cell is empty; best granularity achieved = **none**.

| year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 2021 | · | · | · | · | · | · | · | · | · | · | · | · |
| 2022 | · | · | · | · | · | · | · | · | · | · | · | · |
| 2023 | · | · | · | · | · | · | · | · | · | · | · | · |
| 2024 | · | · | · | · | · | · | · | · | · | · | · | · |
| 2025 | · | · | · | · | · | · | · | · | · | · | · | · |
| 2026 | · | · | · | · | · | · | · | · | · | · | · | · |

Morning-peak (06:00–09:00 Asia/Riyadh) coverage across the 5 years: **0%** (goal was ≥90%).

## 2. Row counts by source

| source | status | rows written | why |
|---|---|--:|---|
| _all 21 sources_ | terminal | **0** | see manifest; no reachable+keyless+fine-grained speed/congestion source |

Real data that *was* reached on GitHub but did **not** qualify (documented, not discarded):

| repo (fetched to `raw/`) | what it actually contains | why excluded |
|---|---|---|
| `khaliddosari/saudi-road-safety-mlops` | Monthly **vehicle counts** by region (Riyadh 2024, 29 count points) | No speed, no congestion index; monthly/regional only |
| `adkurylev/relocation_recsys` (Numbeo) | One Riyadh **Numbeo Traffic Index** = 144.67, commute-time 30.82 min | City-level, no speed, **no timestamp** → dating it = fabrication |
| `Wsh7Ash/gcc-smart-traffic` | `avg_speed_kmh` + `congestion_index` for Riyadh | **Simulated** (`TrafficSimulator`) → forbidden by zero-fabrication |

## 3. Honest limitations

1. **Egress policy is the dominant blocker.** `raw.githubusercontent.com` → HTTP 200; `tomtom.com / api.tomtom.com / kaggle.com / zenodo.org / dataverse.harvard.edu / open.data.gov.sa / od.data.gov.sa / arxiv.org / web.archive.org` → proxy CONNECT **403** (`000` via curl). `WebFetch` shares this proxy (even Wikipedia 403s). So Saudi Open Data, Kaggle, Zenodo, Dataverse, IEEE, Wayback and every commercial API could not even be *probed for content* — only learned about via `WebSearch` snippets. A different network policy (open egress) would materially change what's achievable, though wall #2 below would remain.
2. **No API keys.** TomTom/HERE/INRIX all require registration+key; none provided. Simulating their output is forbidden.
3. **Open Saudi data lacks the target variables.** GASTAT / Ministry of Transport / Saudi Open Data publish *counts* and road-asset inventories at monthly/annual regional granularity — never per-segment speed or a congestion index. Verified via the GASTAT/MoT-derived `saudi-road-safety-mlops` repo.
4. **GitHub — the one reachable data host — has no committed Riyadh speed/congestion time series.** Only live-collector code (needs keys), simulated models (excluded), one undated Numbeo snapshot, and monthly counts.
5. **Uber Movement** (a classic open speed source) never confirmed Riyadh, was shut down in 2022, and its hosts are blocked regardless.
6. **`retrieved_at`/verification integrity:** because 0 rows were written, the VERIFY GATE never fired on the historical dataset; it *is* implemented and active inside `forward_collector.py`.

## 4. Costed commercial options (with pricing-page citations)

The only realistic route to the objective is a paid/keyed commercial feed, run by the user (hosts are also egress-blocked here, so these must run on the user's network):

| Provider | Product | Cost model | Free tier | History depth | Pricing citation |
|---|---|---|---|---|---|
| **TomTom** | Traffic Stats / MOVE (historical) | Licensed **per directional mile** (quote-based) | **30-day** MOVE trial | Deep historical (speeds, travel times, density) | https://developer.tomtom.com/move-portal/guides/traffic-stats/introduction · https://www.tomtom.com/products/traffic-stats/ |
| **TomTom** | Flow Segment Data (live) | Per-transaction | Free tier w/ **free key** (~2,500 req/day typ.) | Live only (no history) | https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data |
| **HERE** | Traffic API (real-time + historical flow) | **$2.50 / 1,000** txns after first 5,000 free; freemium 250k txns/mo | **Yes** (no card, free key) | Real-time + historical in base tier | https://www.here.com/get-started/pricing |
| **INRIX** | IQ / Analytics | Enterprise, sales-quoted | No public tier | Deep historical | https://inrix.com/products/inrix-iq-analytics/ |

Recommendation: **HERE freemium** (250k txns/mo, free key) or **TomTom Flow free tier** are the cheapest way to start collecting *forward*; **TomTom Traffic Stats / MOVE trial** is the only realistic way to backfill *historical* 2021–2025 (within trial limits or a paid quote). None can backfill history for free at scale.

## 5. What to do next (forward path)

`forward_collector.py` is ready. It polls **TomTom Flow Segment Data** every 60 s during
06:00–09:00 Asia/Riyadh across six Riyadh corridors (King Fahd, Northern Ring, Eastern
Ring, Makkah, King Khalid, Olaya), saves each raw JSON to `raw/tomtom_flow/` *before*
parsing, re-verifies every value against the saved payload, and appends rows in the
**identical schema** to `riyadh_traffic.csv`. Get a free key at
https://developer.tomtom.com/ , paste it into `TOMTOM_API_KEY`, and run it (cron/systemd
notes included in the file). Budget ≈ 1,080 requests/morning — under the free daily cap.

## 6. Resume instructions

> **Re-run this prompt in `/home/user/Projects/riyadh_traffic_dataset/` to continue.**
> State persists in `sources_manifest.csv`, `gaps_manifest.csv`, `riyadh_traffic.csv`,
> `acquisition_log.md`, and `raw/`. On resume the loop reloads these and continues from
> Step 0. Note: results will not improve without either (a) an open-egress network
> policy, or (b) a commercial API key — both are outside this session's control.
