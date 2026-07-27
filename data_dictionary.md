# Data Dictionary — riyadh_traffic.csv

Project: Riyadh, Saudi Arabia road-traffic observations (vehicle speed + congestion level),
target window **July 2021 → July 2026**, priority hours **06:00–09:00 Asia/Riyadh (UTC+3, no DST)**.
Target granularity 1 minute, ceiling 60 minutes for coverage accounting.

## Prime directive

Every row traces to a real fetched artifact stored under `raw/` **before** transformation.
No value is ever generated, interpolated, extrapolated, averaged-in, or "reconstructed".
Missing data is documented in `gaps_manifest.csv`, never filled.

## Acquisition-channel caveat (this environment)

This session ran inside a Claude Code remote container whose **egress policy blocks all
direct internet access** (every host except package registries returns HTTP 403 at the
proxy; the server-side WebFetch tool is likewise blocked for all hosts — evidence:
`raw/environment/host_reachability_2026-07-27.txt`, `raw/environment/proxy_status_2026-07-27.json`).
The **only** working acquisition channel was the WebSearch tool. Consequently:

- Raw artifacts in `raw/websearch_*/` are **verbatim saved WebSearch tool outputs**
  (search-result text containing the quoted values), not direct HTTP payloads.
- Only values **literally visible** in saved search output, with a source URL, were
  written to the CSV. Everything else is a gap.
- All rows produced through this channel carry `quality_flag = source_aggregate`
  (they are publisher-level aggregates, e.g. annual city statistics) — **no
  sub-hourly observational rows were obtainable in this environment.**

## Columns

| column | type | definition |
|---|---|---|
| `timestamp_utc` | ISO-8601 | Observation timestamp in UTC. For aggregate rows (`granularity_minutes` > 60): the **start** of the aggregation period. |
| `timestamp_riyadh` | ISO-8601+03:00 | Same instant in Asia/Riyadh (UTC+3 year-round). |
| `date` | YYYY-MM-DD | Riyadh-local date of `timestamp_riyadh` (period start for aggregates). |
| `hour` | 0–23 | Riyadh-local hour (period start for aggregates). |
| `minute` | 0–59 | Riyadh-local minute (period start for aggregates). |
| `is_morning_peak` | 0/1 | 1 iff 06:00–08:59 Asia/Riyadh. Aggregate rows: 1 only if the aggregate is *specifically* a morning-rush-hour statistic. |
| `road_segment_id` | string | Stable segment key. City-level aggregates use `CITY_RIYADH`. Forward-collector segments use keys defined in `forward_collector.py` (e.g. `KING_FAHD_RD_C`). |
| `road_name` | string | Human-readable road name, or `Riyadh (city-wide)`. |
| `lat`, `lon` | decimal deg | Segment probe point (WGS84). City-level rows: 24.7136, 46.6753 (Riyadh centroid). |
| `avg_speed_kmh` | float | Measured average speed, km/h, bounds 0–160. Empty if the source published no speed. **Never derived from another column.** |
| `free_flow_speed_kmh` | float | Source's free-flow reference speed, km/h. Empty if not published. |
| `congestion_index` | float | The source's congestion value **on the source's own scale** (see `congestion_definition`). |
| `congestion_definition` | string | The source's own definition, quoted/paraphrased per source. Indices are NOT comparable across sources. |
| `granularity_minutes` | int | Aggregation period length in minutes. 1 = minute obs; 60 = hourly; 1440 = daily; 525600 = annual aggregate. Only rows ≤ 60 count toward coverage. |
| `source_name` | string | Matches `sources_manifest.csv`. |
| `source_url` | URL | Page/endpoint the value traces to. |
| `retrieved_at` | ISO-8601 UTC | When the artifact was captured. |
| `license` | string | Source license/terms as known. |
| `quality_flag` | enum | `verified` (row re-checked against raw artifact), `batch_sampled` (batch passed 5-row random re-check), `source_aggregate` (publisher aggregate, not a raw observation). |

## Congestion definitions used

- **TomTom Traffic Index (congestion level %)**: "the extra travel time a driver experiences
  compared to free-flow (uncongested) conditions", expressed as a percentage; e.g. 25% means
  a trip takes 25% longer than under free-flow. Annual city figure. Scale 0–100+.
- **Numbeo Traffic Index**: composite index of time in traffic, time dissatisfaction,
  CO2 and overall inefficiency, from user surveys. Dimensionless, typically 0–320.
  NOT a road-measurement; retained only as a labeled aggregate.

## Deduplication key

(`timestamp_utc`, `road_segment_id`, `source_name`)

## Timezone note

Saudi Arabia uses UTC+3 with no daylight saving; conversion is a fixed −3 h offset.
