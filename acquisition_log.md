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
