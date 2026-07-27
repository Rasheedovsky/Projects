# FINAL REPORT — Riyadh Road-Traffic Dataset Acquisition

Session 1 · 2026-07-27 (UTC) · 4 of 12 cycles used · **Stop condition 1: queue exhausted**
(every one of the 28 tracked sources reached a terminal status).

## Bottom line

**7 verified rows** were acquired — all city-level *annual/snapshot aggregates*
(`quality_flag = source_aggregate`). **Zero sub-hourly observations** were obtainable:
this session's egress policy blocks every direct fetch (all hosts 403 at the proxy;
the WebFetch tool blocked for every host tested — evidence in `raw/environment/`), and
the only working channel, WebSearch, surfaces snippets, not files. Coverage of the
06:00–09:00 Asia/Riyadh target window at ≤60-min granularity: **0.0%**.
A truthful sparse dataset beats a dense fabricated one; this is the former.

## Coverage matrix (year × month × best granularity obtained)

| Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Best granularity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | A | A | A | A | A | A | A | A | A | A | A | A | annual aggregate only (1 value, whole-year) |
| 2022 | — | — | — | — | — | — | — | — | — | — | — | — | none (gap G002) |
| 2023 | — | — | — | — | — | — | — | — | — | — | — | — | none (gap G002) |
| 2024 | A | A | A | A | A | A | A | A | A | A | A | A | annual aggregates only (3 values) |
| 2025 | A | A | A | A | A | A | A | A | A | A | A | A | annual aggregates only (2 values) |
| 2026 | — | — | — | — | — | — | S | · | · | · | · | · | one survey-index snapshot (Jul); Aug–Dec outside window |

A = the month falls inside an acquired whole-year aggregate; S = point snapshot.
**No month has daily, hourly, or sub-hourly data. No row is morning-peak-specific.**

## Row counts by source (total 7)

| source_name | rows | what |
|---|---|---|
| tomtom_traffic_index_2021_edition | 1 | 2021 congestion level 23% (rank 162) — [2021 report PDF mirror](https://nonews.co/wp-content/uploads/2022/02/TomTom2021.pdf) |
| tomtom_traffic_index_2026_edition | 3 | 2024 rush-hour loss 71.1 h; 2025 rush-hour loss 66 h; 2025 avg speed 24.8 km/h — [Riyadh city page](https://www.tomtom.com/traffic-index/city/riyadh) |
| agbi_inrix_tomtom_blend_2024 | 2 | 2024 annual delay 34 h; 2024 rush-hour delay 58 h — [AGBI, 2026-01-26](https://www.agbi.com/infrastructure/2026/01/traffic-congestion-tightens-grip-on-dubai-and-riyadh/) |
| numbeo_traffic_index | 1 | survey composite 156.69, snapshot 2026-07-27 — [Numbeo Riyadh](https://www.numbeo.com/traffic/in/Riyadh) |

Cross-source caution: TomTom's 71.1 h and the blended report's 58 h both describe 2024
rush-hour loss — different methodologies; never merge (definitions are stored per row).

## Honest limitations

1. **Provenance is search-mediated.** Direct fetching was impossible, so raw artifacts
   are verbatim-saved WebSearch outputs (`raw/websearch_sweep/`, `raw/websearch_verify/`),
   i.e. search-engine renderings of source pages, not the pages themselves. Mitigation:
   every CSV value was sighted in ≥2 independent searches, exact-substring-matched
   against its saved artifact, and bounds-checked. Values that failed re-verification
   were rejected and logged (notably a "90.4% congestion" figure, an unconfirmed "28%",
   and an unsourced "52 h"). Residual risk of search-layer transcription error is
   nonzero — re-confirm against live pages before publication-grade use.
2. **No observational rows.** Nothing at ≤60-min granularity; nothing morning-peak
   specific; no road-segment-level speeds. All 7 rows are city-wide aggregates.
3. **2022–2023 hole.** TomTom editions for those years exist, but Riyadh's row never
   surfaced in snippets (gap G002).
4. **Numbeo is a survey composite**, not a road measurement — labeled as such.
5. **The archive funnel exists but was unreachable.** Wayback snapshots of TomTom's
   live ranking API for Riyadh would give real timestamped congestion values across
   2020–2024; web.archive.org is blocked here (gap G006). This is the first thing to
   run in a network-open environment.

## Costed commercial options (pricing-page citations)

| Option | Riyadh 2021–2026 history? | Cost | Citation |
|---|---|---|---|
| **TomTom Move / Traffic Stats** — best fit | Yes: "data available back to 2008", any date range, 70+ countries; avg/median speeds, travel times; Saudi use confirmed (THTC) | 30-day free trial via MOVE portal, then Move Credits (one-off) or annual Enterprise — quote-only | [product page](https://www.tomtom.com/products/traffic-stats/) · [pricing](https://developer.tomtom.com/pricing) · [cost-model discussion](https://www.researchgate.net/post/Cost_of_TomTom_Move_Traffic_Stats_for_academic_research) |
| **HERE Traffic API (historical flow)** | Historical product exists; **Riyadh/Saudi coverage unverified** — check before buying | Freemium 250k transactions/mo free, then ~$1/1k; Pro $449/mo (third-party pricing guide — verify with HERE) | [placematic.com guide](https://placematic.com/here-location-services/here-pricing/) |
| **INRIX Roadway Analytics** | Saudi coverage confirmed country-level; archive depth unstated | Quote-only; free trial | [product](https://inrix.com/products/roadway-analytics/) · [trial](https://inrix.com/roadway-analytics-trial/) · [Saudi coverage](https://inrix.com/press-releases/roadway-analytics-eng/) |
| **trafficindex.org Premium** | Monthly Riyadh congestion 2017–2025 (lineage undocumented) | Premium subscription; price not surfaced | [Riyadh page](https://trafficindex.org/riyadh/) |
| **xmap.ai Saudi road traffic** | Historical claimed, depth unstated; segment mean/median speeds | Sales-gated; samples on request | [catalog](https://www.xmap.ai/data-catalogs/saudi-arabia-road-traffic-data) |
| Google Maps Routes/Distance Matrix | **No** — `departure_time` cannot be in the past; forward/predictive only | $5–15 per 1k elements | [billing docs](https://developers.google.com/maps/documentation/routes/usage-and-billing) |
| Otonomo / Wejo | Defunct (2023) — dropped | — | [S&P analysis](https://www.spglobal.com/mobility/en/research-analysis/the-rise-and-fall-of-otonomo-and-wejo.html) |

**Zero-cost recoverables** (blocked only by this session's network policy or a free login):
Wayback snapshots of `api.midway.tomtom.com/ranking/live/SAU%2Friyadh` (G006);
Kaggle `bwandowando` TomTom live scrape, updated 2026-06-21, Riyadh included — free
account needed (G005); TomTom 2022/2023 edition PDFs (G002); Numbeo yearly pages (G003);
INRIX scorecard Riyadh page (G007); Riyadh Municipality Urban Data Center
("traffic flow and congestion" content confirmed to exist).

## Forward collection

`forward_collector.py` (syntax-checked) polls TomTom Flow Segment Data every 60 s
during 06:00–09:00 Asia/Riyadh across King Fahd Rd, Northern Ring Rd, Eastern Ring Rd,
Makkah Rd, King Khalid Rd and Olaya St, saving raw payloads before parsing, verifying
rows against them, and appending to `riyadh_traffic.csv` in the identical schema.
Free TomTom registration required (`TOMTOM_API_KEY`); cron/systemd notes inside.
It must run on your machine — this agent cannot schedule anything for you.

## Resume instructions

Re-run this prompt in the same directory (repo root, branch
`claude/riyadh-traffic-dataset-asmfnq`) to continue: the state files
(`riyadh_traffic.csv`, `sources_manifest.csv`, `gaps_manifest.csv`,
`acquisition_log.md`, `raw/`) will be detected and the loop resumes at Cycle 5.
For real progress, resume in an environment with open network access (or provide a
Kaggle/TomTom key): work gaps G006 → G005 → G002/G003/G007 in that order.
