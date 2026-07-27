# FINAL REPORT — Riyadh Road-Traffic Dataset Acquisition

Session 1 · 2026-07-27 (UTC) · 6 acquisition cycles · updated after user directive
("get the data however possible") unlocked the GitHub public-repo channel.

## Bottom line

**601 verified rows**, zero fabricated:

- **592 daily rows** — TomTom daily congestion level for Riyadh, **2021-07-01 → 2023-02-13**
  (every day except 2021-08-27), from TomTom's own `dailyStats` API as archived by the
  user-authorized public GitHub repo `ActiveConclusion/COVID19_mobility`. Full provenance
  chain: repo commit `8c04388` → verbatim Riyadh slice + SHA-256s in
  `raw/github_activeconclusion/` → row-by-row re-verification. `quality_flag=verified`.
- **9 aggregate rows** (`source_aggregate`) — TomTom Traffic Index annuals (2021: 23%;
  2024: 71.1 h rush-hour loss; 2025: 66 h + 24.8 km/h avg speed), press-cited TomTom
  figures (2025: 43.7% avg congestion via An-Nahar; 2024: 56 h on the 9.5-km rush-hour
  metric via Sabq), the blended INRIX+TomTom 2024 report via AGBI (34 h annual / 58 h
  rush-hour), and one Numbeo survey-composite snapshot (156.69, Jul 2026).

**Hourly/sub-hourly: still zero** — see "The hourly problem" below. Morning-peak
(06:00–09:00) specific coverage: zero rows (best lead logged, unattributable so far).

## Coverage matrix (year × month × best granularity)

| Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Best granularity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | · | · | · | · | · | · | D | D* | D | D | D | D | **daily** from Jul 1 (183 d; *Aug 27 missing) + annual aggregate |
| 2022 | D | D | D | D | D | D | D | D | D | D | D | D | **daily**, complete (365 d) |
| 2023 | D | D½ | — | — | — | — | — | — | — | — | — | — | **daily** to Feb 13 (44 d); rest: none |
| 2024 | A | A | A | A | A | A | A | A | A | A | A | A | annual aggregates only (4 values, 2 methodologies) |
| 2025 | A | A | A | A | A | A | A | A | A | A | A | A | annual aggregates only (3 values) |
| 2026 | — | — | — | — | — | — | S | · | · | · | · | · | one survey snapshot (Jul); Aug–Dec outside window |

D = daily rows; A = inside an acquired whole-year aggregate; S = point snapshot; · = outside window/none.

## Row counts by source (total 601)

| source_name | rows | granularity | what |
|---|---|---|---|
| tomtom_dailystats_api_via_activeconclusion_github | 592 | daily | TomTom daily congestion %, [repo](https://github.com/ActiveConclusion/COVID19_mobility/blob/master/tomtom_reports/tomtom_trafic_index.csv), upstream `api.midway.tomtom.com/ranking/dailyStats/SAU_riyadh` |
| tomtom_traffic_index_2021_edition | 1 | annual | 2021: 23% congestion, rank 162 ([2021 PDF mirror](https://nonews.co/wp-content/uploads/2022/02/TomTom2021.pdf)) |
| tomtom_traffic_index_2026_edition | 3 | annual | 2024: 71.1 h rush-hour loss; 2025: 66 h; 2025: 24.8 km/h ([city page](https://www.tomtom.com/traffic-index/city/riyadh)) |
| annahar_citing_tomtom_2025 | 1 | annual | 2025: 43.7% avg congestion ([An-Nahar](https://www.annahar.com/economy/316355/)) |
| sabq_citing_tomtom_2024 | 1 | annual | 2024: 56 h lost, 9.5-km rush-hour metric ([Sabq](https://sabq.org/saudia/dl5qk8spdg)) |
| agbi_inrix_tomtom_blend_2024 | 2 | annual | 2024: 34 h annual / 58 h rush-hour ([AGBI](https://www.agbi.com/infrastructure/2026/01/traffic-congestion-tightens-grip-on-dubai-and-riyadh/)) |
| numbeo_traffic_index | 1 | snapshot | survey composite 156.69 ([Numbeo](https://www.numbeo.com/traffic/in/Riyadh)) |

**Comparability warning:** 2024 carries four different "hours lost"-family numbers
(71.1 / 58 / 56 / 34) from four methodologies. Each row's `congestion_definition`
states its own; never average or merge them.

## The hourly problem (user requirement — unmet, with exact unlock paths)

Every hourly-capable source is unreachable from this environment, not nonexistent:

1. **Wayback Machine snapshots** of TomTom's live API (`ranking/live/SAU%2Friyadh`) —
   real timestamped congestion 2020–2024. Blocked: web.archive.org denied by egress
   policy and WebFetch. **Unlock: run this prompt in an environment with open network
   policy** (claude.ai/code → environment settings → network access) — first thing the
   resume run should do.
2. **Kaggle `bwandowando` TomTom live scrape** (updated 2026-06-21, Riyadh included) —
   sub-daily live-index history ~2023→present. Needs a free Kaggle account **and** open
   network (kaggle.com is blocked here even with a key).
3. **TomTom Move / Traffic Stats** — hourly/segment-level speeds back to 2008, the only
   full-window hourly option: 30-day free MOVE trial then paid ([product](https://www.tomtom.com/products/traffic-stats/), [pricing](https://developer.tomtom.com/pricing)).
4. **Forward minute-level collection** — `forward_collector.py` is ready: TomTom Flow
   Segment Data every 60 s, 06:00–09:00 Riyadh, 6 arteries (King Fahd, Northern Ring,
   Eastern Ring, Makkah, King Khalid, Olaya), raw-first with verify gate; free key from
   https://developer.tomtom.com/user/register; cron/systemd notes inside. Runs on your machine.

Other paid options: HERE historical flow (freemium 250k tx/mo, ~$1/1k, Pro $449/mo per
[third-party guide](https://placematic.com/here-location-services/here-pricing/); Riyadh coverage unverified), INRIX Roadway Analytics (quote-only,
[trial](https://inrix.com/roadway-analytics-trial/), [Saudi coverage confirmed](https://inrix.com/press-releases/roadway-analytics-eng/)), trafficindex.org Premium (monthly Riyadh
2017–2025), xmap.ai (quote-only). Google Maps APIs cannot serve past dates at any price.

## Integrity notes (read before publication-grade use)

1. Daily rows have a complete artifact chain and passed full row-by-row re-verification.
2. The 9 aggregates are search-mediated (direct fetching impossible); each passed a
   ≥2-independent-sighting + exact-substring gate against saved artifacts.
3. **Rejected values** (logged in `acquisition_log.md`, never written): "90.4% congestion",
   INRIX "123 h" (proven query-echo contamination — verification queries must never
   contain the target number), AGBI "28%", Arab News "52 h", unattributable
   "26/36 km/h peak speeds" (strongest morning-peak lead; 3 clean sightings, no URL).
4. Subagent tooling broke mid-session (permission-handler bug); Cycles 5–6 ran inline.
   Cycle-2 sweep artifacts predate the bug and are valid.

## Resume instructions

Re-run this prompt in the same directory (branch `claude/riyadh-traffic-dataset-asmfnq`);
state files are detected automatically, loop resumes at Cycle 7. Priority queue for a
network-open resume: Wayback live-API snapshots (G006) → Kaggle live scrape (G005) →
TomTom 2022/2023 edition PDFs (G002) → daily series 2023-02-14→present (G009) →
alweeam/heatmap article for the 26/36 km/h morning-peak values → Numbeo yearly pages
(G003) → INRIX city page (G007).
