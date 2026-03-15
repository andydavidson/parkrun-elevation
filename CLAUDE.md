# parkrun-elevation

A Python pipeline that builds an open dataset of elevation profiles for every parkrun event worldwide (~900 UK, ~2,000 global). The full technical specification is in `SPEC.md`.

## Working style

- **Coached, collaborative, incremental.** Before writing any code, explain your proposed approach and check I'm happy with it.
- **One module at a time.** Build and test each component before moving to the next. Don't scaffold the entire project in one go.
- **Explain decisions.** When there's a non-obvious choice (algorithm, library, data structure), say why you chose it.
- **Test as you go.** Each module should be runnable and produce visible output before we move on.
- **Don't over-engineer.** We're building a data pipeline, not a production service. Prefer readable code over clever abstractions.
- **UK English** in comments, docstrings, and output messages.

## Project stack

- Python, managed with `uv`
- `httpx` for async HTTP
- `rasterio` for local SRTM elevation lookups
- `gpxpy` for GPX parsing
- `numpy` for interpolation and smoothing
- `stravalib` for Strava API (OAuth + segments)
- Output: JSON files committed to the repo

## Key constraints

- **Idempotency is essential.** If a parkrun already has computed elevation data in the output JSON, do not re-fetch or re-compute it. See the caching spec in SPEC.md.
- Strava API rate limits are a hard constraint: 200 requests/15 min, 2,000/day. The pipeline must respect these.
- The dataset will be released as open data. Track data provenance (source type, source ID, date computed) for every record.

## Project structure (to be built incrementally)

```
parkrun-elevation/
├── CLAUDE.md              ← this file
├── SPEC.md                ← full technical blueprint
├── pyproject.toml
├── data/
│   ├── events.json        ← cached copy of parkrun master event list
│   └── elevation/
│       └── uk.json        ← output dataset (append-only, idempotent)
├── srtm/                  ← local SRTM tile cache (gitignored)
├── src/
│   └── parkrun_elevation/
│       ├── __init__.py
│       ├── events.py      ← fetch and filter events.json
│       ├── routes.py      ← OSM + Strava route acquisition
│       ├── elevation.py   ← SRTM lookup and gain/loss calculation
│       ├── cache.py       ← idempotency logic
│       └── pipeline.py    ← orchestration
└── scripts/
    └── run_uk.py          ← entry point for UK batch run
```

## Build order

1. `events.py` — fetch events.json, parse, filter to UK
2. `cache.py` — load existing output JSON, expose a `is_known()` check
3. `elevation.py` — SRTM lookup, interpolation, gain/loss algorithm
4. `routes.py` — OSM Overpass route fetcher
5. `routes.py` — Strava segment matcher (added after OSM is working)
6. `pipeline.py` — orchestrate with idempotency and rate limiting
7. `scripts/run_uk.py` — CLI entry point with progress reporting
