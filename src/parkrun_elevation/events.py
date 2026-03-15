"""
Fetch and filter the parkrun master event list.

Source: https://images.parkrun.com/events.json (no authentication required)

The events.json file is a GeoJSON FeatureCollection. Coordinates follow the
GeoJSON convention: [longitude, latitude] — note longitude comes first.
"""

import json
import logging
from pathlib import Path

import httpx

EVENTS_URL = "https://images.parkrun.com/events.json"
EVENTS_CACHE_PATH = Path("data/events.json")

logger = logging.getLogger(__name__)


def fetch_events(refresh: bool = False) -> dict:
    """Return the raw events.json as a parsed dict.

    Uses the locally cached copy at data/events.json unless refresh=True
    or the file doesn't exist yet.
    """
    if not refresh and EVENTS_CACHE_PATH.exists():
        logger.info("Loading events from cache: %s", EVENTS_CACHE_PATH)
        return json.loads(EVENTS_CACHE_PATH.read_text())

    logger.info("Downloading events from %s", EVENTS_URL)
    response = httpx.get(EVENTS_URL, timeout=30)
    response.raise_for_status()
    raw = response.json()

    EVENTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_CACHE_PATH.write_text(json.dumps(raw, indent=2))
    logger.info("Saved %d features to %s", len(raw["events"]["features"]), EVENTS_CACHE_PATH)

    return raw


# Geographic bounding box for the British Isles (excludes overseas territories
# such as the Falkland Islands and Gibraltar, which parkrun assigns countrycode=97).
UK_LAT_MIN, UK_LAT_MAX = 49.0, 61.0
UK_LNG_MIN, UK_LNG_MAX = -8.5, 2.0


def parse_events(
    raw: dict,
    country_code: int = 97,
    series_id: int = 1,
    lat_bounds: tuple[float, float] | None = None,
    lng_bounds: tuple[float, float] | None = None,
) -> list[dict]:
    """Extract and filter events from the raw events.json dict.

    Filters to:
    - series_id == 1    (adult parkruns; excludes junior parkrun)
    - country_code      (97 = UK for phase 1)
    - lat/lng bounds    (optional; use to exclude overseas territories assigned
                         the same country_code, e.g. Falkland Islands for UK)

    Returns a list of dicts with keys:
        event_name, event_long_name, lat, lng, country_code
    """
    features = raw["events"]["features"]
    events = []

    for feature in features:
        props = feature["properties"]

        if props.get("seriesid") != series_id:
            continue
        if props.get("countrycode") != country_code:
            continue

        # GeoJSON coordinates are [longitude, latitude]
        lng, lat = feature["geometry"]["coordinates"]

        if lat_bounds is not None and not (lat_bounds[0] <= lat <= lat_bounds[1]):
            continue
        if lng_bounds is not None and not (lng_bounds[0] <= lng <= lng_bounds[1]):
            continue

        events.append({
            "event_name": props["eventname"],
            "event_long_name": props["EventLongName"],
            "lat": lat,
            "lng": lng,
            "country_code": props["countrycode"],
        })

    return events


def get_uk_events(refresh: bool = False) -> list[dict]:
    """Fetch (or load from cache) and return filtered UK adult parkrun events.

    Applies a geographic bounding box for the British Isles to exclude overseas
    territories (e.g. Falkland Islands) that share countrycode=97.
    """
    raw = fetch_events(refresh=refresh)
    events = parse_events(
        raw,
        country_code=97,
        series_id=1,
        lat_bounds=(UK_LAT_MIN, UK_LAT_MAX),
        lng_bounds=(UK_LNG_MIN, UK_LNG_MAX),
    )
    logger.info("Found %d UK events", len(events))
    return events
