# Data Dictionary — Riyadh Road-Traffic Observations

**Objective:** Verified, provenance-tracked observations of vehicle speed and
congestion on Riyadh (Saudi Arabia) roads, July 2021 → July 2026, finest
granularity obtainable (target 1-min, ceiling 1-hour). Priority window
06:00–09:00 Asia/Riyadh.

**Prime directive:** ZERO FABRICATION. Every row traces to a raw artifact saved
under `raw/` *before* transformation. No interpolation, extrapolation, averaging,
or synthetic rows. Missing data → `gaps_manifest.csv`, never a plausible number.
This file defines the schema; the CSV holds only retrieved observations. There
are **no example rows** anywhere.

## `riyadh_traffic.csv` columns

| column | type | definition |
|---|---|---|
| `timestamp_utc` | ISO-8601 UTC | Observation instant in UTC (e.g. `2024-03-01T03:00:00Z`). For aggregate rows, the representative instant per source docs. |
| `timestamp_riyadh` | ISO-8601 +03:00 | Same instant in Asia/Riyadh (UTC+3, no DST). |
| `date` | YYYY-MM-DD | Riyadh-local calendar date. |
| `hour` | int 0–23 | Riyadh-local hour. |
| `minute` | int 0–59 | Riyadh-local minute (0 for hourly/coarser data). |
| `is_morning_peak` | 0/1 | 1 iff Riyadh-local time ∈ [06:00, 08:59]. |
| `road_segment_id` | str | Source's segment/road identifier (verbatim). For city aggregates: `CITY_AGGREGATE`. |
| `road_name` | str | Human road name (verbatim from source). |
| `lat` | float | Segment representative latitude (WGS84), blank if source gives none. |
| `lon` | float | Segment representative longitude (WGS84), blank if none. |
| `avg_speed_kmh` | float 0–160 | Observed average vehicle speed, km/h. Blank if source provides only congestion. |
| `free_flow_speed_kmh` | float 0–160 | Source's free-flow / reference speed, km/h. Blank if none. |
| `congestion_index` | float | Source's congestion measure (verbatim value). Meaning defined by next column. |
| `congestion_definition` | str | The source's own definition of its congestion measure. Never merged across sources. |
| `granularity_minutes` | int | Temporal resolution of the observation (1, 5, 60, 1440 for daily, etc.). |
| `source_name` | str | Matches `sources_manifest.csv`. |
| `source_url` | str | Direct URL/DOI to the fetched artifact. |
| `retrieved_at` | ISO-8601 UTC | When the artifact was fetched. |
| `license` | str | License/terms of the source. |
| `quality_flag` | enum | `verified` (row value re-checked against raw file), `batch_sampled` (≥5 rows/batch re-checked, rest inferred-consistent), `source_aggregate` (source publishes an aggregate, not a per-instant observation). |

## Units & conventions
- Speed in **km/h**. Timezone stored both UTC and Asia/Riyadh (UTC+3, no DST).
- Speed bounds enforced 0–160 km/h; out-of-bounds → batch rejected.
- Congestion values kept in the source's native scale; cross-source comparison
  requires reading `congestion_definition`.

## Deduplication key
`(timestamp_utc, road_segment_id, source_name)`.

## `sources_manifest.csv`
Bandit state. `status ∈ {untried, probing, acquired, exhausted,
paywalled_or_keyed, no_riyadh, no_history}`. `source_type ∈ {commercial_api,
gov_portal, academic_repo, archive_snapshot}`. `alpha`/`beta` are the per-source
Beta counters; `score = E[p_success] × value_est / effort_est` where
`E[p_success] = α/(α+β)` uses the **per-type** Beta prior.

## `gaps_manifest.csv`
Every documented gap: `gap_id, scope, date_range, hours, granularity_target,
reason, source_attempted, evidence_url, cycle_logged`.
