# Technical specification: parkrun elevation dataset

A pipeline to compute and store elevation profiles (total ascent, descent, average gradient, max gradient) for every parkrun event worldwide, released as open JSON data.

---

## 1. Master event list

**Source:** `https://images.parkrun.com/events.json` — freely accessible, no authentication.

**Structure:**
```json
{
  "events": {
    "type": "FeatureCollection",
    "features": [{
      "id": 280,
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [-1.310849, 51.069286]},
      "properties": {
        "eventname": "winchester",
        "EventLongName": "Winchester parkrun",
        "countrycode": 97,
        "seriesid": 1
      }
    }]
  }
}
```

**Filters:**
- `seriesid == 1` — adult parkruns only (excludes junior parkrun)
- `countrycode == 97` — UK only for phase 1

Cache the raw file locally as `data/events.json`. Re-fetch only on explicit `--refresh-events` flag.

The `eventname` field is the URL slug: `https://www.parkrun.org.uk/{eventname}/course/`

---

## 2. Idempotency and caching

This is the most important architectural requirement. The pipeline must be safe to run repeatedly without re-fetching data for events that are already processed.

### Output file structure

The output is a single JSON file per region (e.g. `data/elevation/uk.json`):

```json
{
  "metadata": {
    "generated_at": "2026-03-15T12:00:00Z",
    "schema_version": "1.0",
    "total_events": 847,
    "events_with_elevation": 712,
    "events_missing_route": 135
  },
  "parkruns": [
    {
      "event_name": "bushy",
      ...
    }
  ]
}
```

### `cache.py` — the idempotency module

This module must be built second (after `events.py`) and used by every subsequent stage.

**Responsibilities:**
- Load the existing output JSON on startup
- Expose `is_known(event_name: str) -> bool` — returns `True` if the event already has a computed elevation record with `status == "complete"`
- Expose `get_record(event_name: str) -> dict | None`
- Expose `upsert(record: dict)` — add or replace a record, then write atomically to disk
- Write to a temp file then `os.replace()` to avoid corruption on interrupt

**Status values** for each record:
- `"complete"` — elevation data computed and stored; skip on next run
- `"no_route"` — route acquisition attempted but no source found; retry after 30 days
- `"pending"` — placeholder written but not yet computed (for crash recovery)

**Skip logic** in the pipeline:
```python
for event in events:
    if cache.is_known(event["eventname"]):
        continue  # already done, skip entirely
    # ... proceed with route acquisition and elevation computation
```

**Force-refresh flag:** `--force event_name` to recompute a specific event regardless of cache status. Useful for updating records when a better route source is found.

**Retry logic:** Events with `status == "no_route"` should be retried if `date_computed` is more than 30 days old. Check `(datetime.now() - date_computed).days > 30`.

---

## 3. Route acquisition

No official parkrun bulk route download exists. Use a cascade of sources.

### 3a. OpenStreetMap (primary source)

Query via the Overpass API. No authentication required.

**Endpoint:** `https://overpass-api.de/api/interpreter`

**Query to find all UK parkrun routes:**
```
[out:json][timeout:120];
relation["route"="running"]["name"~"parkrun",i](51.3,-10.6,60.9,1.8);
out body;
>;
out skel qt;
```

This returns complete route geometry. Match relations to events by:
1. Name similarity: strip "parkrun" and compare to `eventname` (case-insensitive, strip whitespace)
2. Proximity: centroid of route geometry must be within 500m of the event's `events.json` coordinates
3. Length: relation total length must be between 4,500m and 15,500m (some courses are multi-lap; a single lap is ~5km)

**Extracting the polyline from OSM relations:**
```python
# Relations contain ordered 'way' members; ways contain ordered 'node' members
# Build: relation -> ordered ways -> ordered nodes -> lat/lng list
```

The Overpass response includes node coordinates in the `elements` array. Build a lookup `{node_id: (lat, lon)}` then walk the relation's way members in order.

**Rate limits:** Generous. Add a 1-second delay between requests as good practice. Cache the raw Overpass response locally — don't re-query on every run.

### 3b. Strava segments (secondary source)

Use for events not covered by OSM.

**Authentication:** OAuth 2.0. Register an app at `https://www.strava.com/settings/api`. Store `client_id`, `client_secret`, and refresh token in a `.env` file (never committed). Use `stravalib` for token refresh handling.

**Segment discovery:**
```
GET https://www.strava.com/api/v3/segments/explore
  ?bounds={lat_min},{lng_min},{lat_max},{lng_max}
  &activity_type=running
```

Bounding box: ±0.005° (~550m) around the event coordinates. This keeps the box tight enough to avoid irrelevant segments in dense urban areas.

The endpoint returns up to 10 segments. Filter results:
- `name` contains "parkrun" (case-insensitive)
- `distance` between 4,500m and 5,300m (one lap)
- If multiple matches, prefer the one with the highest `athlete_count` (most popular = most likely official route)

**Fetching elevation streams:**
```
GET https://www.strava.com/api/v3/segments/{id}/streams
  ?keys=latlng,altitude,distance
```

This returns the complete elevation profile as arrays — no separate elevation lookup needed for Strava-sourced routes.

**Rate limit management:**
- 200 requests per 15 minutes, 2,000 per day
- Each unmatched event costs ~3 requests (explore + detail + streams)
- Track request count with a rolling window; sleep when approaching limits
- Log rate limit headers from each response (`X-RateLimit-Limit`, `X-RateLimit-Usage`)
- At 2,000/day, processing 600 unmatched events takes ~1 day — run overnight

**Storing Strava credentials:** `.env` file:
```
STRAVA_CLIENT_ID=12345
STRAVA_CLIENT_SECRET=abc123
STRAVA_REFRESH_TOKEN=xyz789
```

Load with `python-dotenv`. Never hardcode.

### 3c. Fallback: skip and mark `no_route`

If neither OSM nor Strava yields a route, write a `no_route` record with the event's coordinates and move on. Don't attempt page scraping in the initial implementation.

---

## 4. Elevation data sources

### Primary: local SRTM 30m tiles via `rasterio`

Download SRTM 1-arc-second (30m) tiles from AWS Open Data:
`s3://elevation-tiles-prod/skadi/{NS}{lat}/{NS}{lat}{EW}{lng}.hgt.gz`

Or via HTTP from OpenTopography: `https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/`

For UK coverage (~110 tiles), total download is ~2 GB. Store in `srtm/` (gitignored).

**Auto-download tiles on demand:**
```python
def get_elevation_for_coords(coords: list[tuple[float, float]]) -> list[float]:
    # Group coords by tile, download missing tiles, query rasterio
    tiles_needed = {get_tile_name(lat, lng) for lat, lng in coords}
    for tile in tiles_needed:
        if not tile_is_cached(tile):
            download_tile(tile)
    # Query
    results = []
    for lat, lng in coords:
        tile_path = get_tile_path(lat, lng)
        with rasterio.open(tile_path) as src:
            val = list(src.sample([(lng, lat)]))[0][0]
            results.append(float(val))
    return results
```

Note: `rasterio.sample()` takes `(lng, lat)` not `(lat, lng)`.

### Alternative: Google Elevation API

For testing or when SRTM setup is impractical. 512 locations per request (batched path), ~£8 for 1M lookups. Requires `GOOGLE_MAPS_API_KEY` in `.env`.

```
GET https://maps.googleapis.com/maps/api/elevation/json
  ?path=enc:{encoded_polyline}
  &samples=512
  &key={API_KEY}
```

### Elevation source for Strava routes

Strava segments already include `altitude` arrays — use these directly. Do not make a separate elevation lookup for Strava-sourced routes. This saves API calls and uses Strava's corrected elevation data.

---

## 5. Elevation gain algorithm

### Stage 1: Interpolate to regular 10m intervals

Raw route points are irregularly spaced. Resample to 10m steps using cumulative haversine distance:

```python
import numpy as np
from haversine import haversine

def interpolate_route(lats, lngs, elevations, interval_m=10):
    cum_dist = [0.0]
    for i in range(1, len(lats)):
        d = haversine((lats[i-1], lngs[i-1]), (lats[i], lngs[i]), unit='m')
        cum_dist.append(cum_dist[-1] + d)
    cum_dist = np.array(cum_dist)
    total_dist = cum_dist[-1]
    target_dists = np.arange(0, total_dist, interval_m)
    new_elevs = np.interp(target_dists, cum_dist, elevations)
    new_lats = np.interp(target_dists, cum_dist, lats)
    new_lngs = np.interp(target_dists, cum_dist, lngs)
    return new_lats, new_lngs, new_elevs, target_dists
```

At 10m intervals, a 5km parkrun produces ~500 points — optimal for both SRTM resolution and API batching.

### Stage 2: Smooth with a moving average

Remove high-frequency DEM noise. Use a 70m window (7 points at 10m spacing):

```python
window = 7
kernel = np.ones(window) / window
elevs_smooth = np.convolve(elevations, kernel, mode='same')
```

### Stage 3: Dead-band threshold filter

Track cumulative change from the last confirmed elevation. Register gain/loss only when change exceeds the threshold. Use **3m for SRTM-sourced elevation**, **2m for Strava altitude streams**:

```python
def calc_elevation_stats(elevations: np.ndarray, threshold: float = 3.0) -> dict:
    ascent = descent = 0.0
    last_confirmed = elevations[0]
    for elev in elevations[1:]:
        diff = elev - last_confirmed
        if diff >= threshold:
            ascent += diff
            last_confirmed = elev
        elif diff <= -threshold:
            descent += abs(diff)
            last_confirmed = elev
    return {
        "total_ascent_m": round(ascent, 1),
        "total_descent_m": round(descent, 1),
        "elevation_high_m": round(float(np.max(elevations)), 1),
        "elevation_low_m": round(float(np.min(elevations)), 1),
    }
```

### Gradient calculations

**Average absolute gradient** (more meaningful than net gradient for loops):
```python
segment_gradients = np.abs(np.diff(elevations_smooth) / interval_m * 100)
avg_abs_gradient = round(float(np.mean(segment_gradients)), 2)
```

**Max gradient over 100m window:**
```python
def calc_max_gradient(elevations, interval_m=10, window_m=100):
    window_pts = window_m // interval_m
    max_grad = 0.0
    for i in range(len(elevations) - window_pts):
        elev_diff = abs(elevations[i + window_pts] - elevations[i])
        grad = elev_diff / window_m * 100
        max_grad = max(max_grad, grad)
    return round(max_grad, 1)
```

---

## 6. Output JSON schema

One file per region: `data/elevation/uk.json`, `data/elevation/global.json`

```json
{
  "metadata": {
    "schema_version": "1.0",
    "generated_at": "2026-03-15T12:00:00Z",
    "elevation_sources": ["srtm30m", "strava"],
    "total_events": 892,
    "events_complete": 801,
    "events_no_route": 91
  },
  "parkruns": [
    {
      "event_name": "bushy",
      "event_long_name": "Bushy parkrun",
      "country": "UK",
      "country_code": 97,
      "lat": 51.4108,
      "lng": -0.3369,
      "status": "complete",
      "distance_m": 5000,
      "total_ascent_m": 12.3,
      "total_descent_m": 14.1,
      "elevation_high_m": 18.2,
      "elevation_low_m": 8.4,
      "avg_abs_gradient_pct": 1.2,
      "max_gradient_pct": 3.8,
      "max_gradient_window_m": 100,
      "smoothing_window_m": 70,
      "elevation_threshold_m": 3.0,
      "route_source_type": "osm",
      "route_source_id": "relation/12345678",
      "route_source_url": "https://www.openstreetmap.org/relation/12345678",
      "elevation_source": "srtm30m",
      "date_computed": "2026-03-15"
    },
    {
      "event_name": "shirebrook",
      "event_long_name": "Shirebrook parkrun",
      "country": "UK",
      "country_code": 97,
      "lat": 53.2061,
      "lng": -1.2156,
      "status": "no_route",
      "date_computed": "2026-03-15"
    }
  ]
}
```

`route_source_type` values: `"osm"`, `"strava"`, `"scraped"`
`elevation_source` values: `"srtm30m"`, `"strava_stream"`, `"google_elevation"`, `"ea_lidar_1m"`

---

## 7. Key gotchas

**SRTM geoid vs WGS84 ellipsoid:** SRTM elevation is relative to EGM96 geoid, GPS/GPX uses WGS84 ellipsoid — a 10–50m absolute offset by location. This does not affect gain/loss calculations (which use differences) but will cause confusion if you compare absolute elevations across sources. Document this in the output metadata.

**Multi-lap courses:** Some parkruns run a loop twice. The OSM/Strava route may only represent one lap. Detect this by checking if route length is ~2,500m and multiplying ascent/descent by 2. Flag this in the record as `"multi_lap": true, "laps": 2`.

**Strava 10-segment limit:** The explore endpoint returns at most 10 results. In dense areas this may not include the parkrun segment. If no "parkrun" name match is found, try a slightly larger bounding box (±0.01°) as a second attempt before marking `no_route`.

**Atomic file writes:** Always write the output JSON via a temp file + `os.replace()` to prevent corruption if the process is interrupted mid-write.

**Strava token expiry:** Access tokens expire every 6 hours. Use `stravalib`'s built-in refresh handling, or check expiry before each batch of requests.

---

## 8. Build order

Build and test each module in isolation before integrating:

1. **`events.py`** — fetch and cache `events.json`, return filtered list of `{event_name, lat, lng, country_code}`
2. **`cache.py`** — load/save output JSON, `is_known()`, `upsert()`, atomic writes
3. **`elevation.py`** — interpolation, smoothing, gain/loss calc, SRTM tile fetcher — test on a single known course
4. **`routes.py` (OSM)** — Overpass query, relation parsing, name/proximity matching
5. **`routes.py` (Strava)** — OAuth setup, segment explore, stream fetch, rate limit handling
6. **`pipeline.py`** — orchestrate with idempotency check at the top of the loop
7. **`scripts/run_uk.py`** — CLI with `--force`, `--limit N`, `--dry-run`, progress bar via `tqdm`

---

## 9. Validation

Validate computed values against community-verified sources:

- `parkrunmaps.com` — has elevation profiles for ~97 events; use as a test set
- `courseelevations.com` — reference dataset (cannot take data from this source directly as the maintainer is known prickly, but good for spot-checking)
- Manually verify 5–10 courses with known character (Bronte, Penrose, Yeovil = hilly; Bushy, Southampton = flat)

Acceptable tolerance: ±5m total ascent for SRTM-sourced routes, ±2m for Strava stream routes.
