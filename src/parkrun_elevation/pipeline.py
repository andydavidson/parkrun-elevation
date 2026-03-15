"""
Pipeline orchestration for the parkrun elevation dataset.

Entry point: run()
Inner loop:  process_event()

Route acquisition cascade per event:
    1. OSM (Overpass)       — match_relation() against bulk pre-fetched data
    2. Google My Maps KML   — get_kml_route() fetches course page + KML
    3. no_route             — write placeholder, retry after 30 days

Elevation build order per event:
    1. Cache check          — skip if already known (idempotent)
    2. Write "pending"      — crash recovery marker
    3. Route lookup         — OSM → KML cascade
    4. Elevation profile    — compute_elevation_profile() via SRTM
    5. Apply lap multiply   — ascent/descent × laps for multi-lap courses
    6. Write final record   — "complete" or "no_route"
"""

import datetime
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .cache import Cache
from .elevation import compute_elevation_profile
from .events import get_uk_events
from .routes import (
    build_node_lookup,
    build_way_lookup,
    extract_relations,
    fetch_overpass,
    match_relation,
)
from .routes_kml import get_kml_route

logger = logging.getLogger(__name__)

_SRTM_DIR = Path("srtm")
_CACHE_PATH = Path("data/elevation/uk.json")


def _apply_laps(elevation_data: dict, route_type: str, laps: int) -> dict:
    """Scale ascent/descent for multi-lap courses.

    - loop:         multiply total_ascent_m and total_descent_m by laps
    - out_and_back: one lap = out + back, so ascent_total = ascent + descent
                    (the return leg is the outward leg reversed)
    - unknown:      treat as loop, add laps_inferred flag
    """
    if laps == 1 or route_type not in ("loop", "out_and_back", "unknown"):
        return elevation_data

    data = dict(elevation_data)

    if route_type == "out_and_back":
        # Return leg has swapped ascent/descent — sum both directions
        total = data["total_ascent_m"] + data["total_descent_m"]
        data["total_ascent_m"] = round(total, 1)
        data["total_descent_m"] = round(total, 1)
    else:
        # loop or unknown: straight multiply
        data["total_ascent_m"] = round(data["total_ascent_m"] * laps, 1)
        data["total_descent_m"] = round(data["total_descent_m"] * laps, 1)

    data["distance_m"] = round(data["distance_m"] * laps, 1)

    return data


def process_event(
    event: dict,
    route: dict | None,
    srtm_dir: Path,
    cache: Cache,
    dry_run: bool = False,
) -> str:
    """Process a single event: compute elevation and write a cache record.

    Parameters
    ----------
    event:    event dict from get_uk_events()
    route:    resolved route dict (keys: coords, source_type, source_id,
              source_url, laps, route_type, laps_inferred) or None
    srtm_dir: path to the local SRTM tile cache
    cache:    Cache instance (written unless dry_run=True)
    dry_run:  if True, compute everything but do not write to cache

    Returns the status string: "complete" or "no_route".
    """
    today = datetime.date.today().isoformat()

    base = {
        "event_name": event["event_name"],
        "event_long_name": event["event_long_name"],
        "country_code": event["country_code"],
        "lat": event["lat"],
        "lng": event["lng"],
        "date_computed": today,
    }

    if route is None:
        record = {**base, "status": "no_route"}
        if not dry_run:
            cache.upsert(record)
        return "no_route"

    coords = route["coords"]
    threshold = 3.0

    elevation_data = compute_elevation_profile(
        lats=[c[0] for c in coords],
        lngs=[c[1] for c in coords],
        srtm_dir=srtm_dir,
        threshold=threshold,
    )

    laps = route.get("laps", 1)
    route_type = route.get("route_type", "loop")
    elevation_data = _apply_laps(elevation_data, route_type, laps)

    record = {
        **base,
        "status": "complete",
        "route_source_type": route["source_type"],
        "route_source_id": route["source_id"],
        "route_source_url": route["source_url"],
        "laps": laps,
        "route_type": route_type,
        **elevation_data,
    }

    if route.get("laps_inferred"):
        record["laps_inferred"] = True

    if not dry_run:
        cache.upsert(record)

    return "complete"


def _resolve_route(
    event: dict,
    relations: list,
    way_lookup: dict,
    node_lookup: dict,
) -> dict | None:
    """Try OSM first, fall back to Google My Maps KML.

    Returns a normalised route dict with keys:
        coords, laps, route_type, laps_inferred,
        source_type, source_id, source_url
    """
    # --- OSM ---
    osm = match_relation(event, relations, way_lookup, node_lookup)
    if osm is not None:
        rel_id = osm["relation"]["id"]
        return {
            "coords": osm["coords"],
            "laps": 1,
            "route_type": "loop",
            "laps_inferred": False,
            "source_type": "osm",
            "source_id": f"relation/{rel_id}",
            "source_url": f"https://www.openstreetmap.org/relation/{rel_id}",
        }

    # --- Google My Maps KML ---
    kml = get_kml_route(event)
    if kml is not None:
        slug = event["event_name"]
        return {
            "coords": kml["coords"],
            "laps": kml["laps"],
            "route_type": kml["route_type"],
            "laps_inferred": kml.get("laps_inferred", False),
            "source_type": "parkrun_kml",
            "source_id": f"googlemymaps/{kml['mid']}",
            "source_url": f"https://www.parkrun.org.uk/{slug}/course/",
        }

    return None


def run(
    srtm_dir: Path = _SRTM_DIR,
    cache_path: Path = _CACHE_PATH,
    refresh_events: bool = False,
    refresh_overpass: bool = False,
    force: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    show_progress: bool = False,
    retry_no_route: bool = False,
) -> None:
    """Run the UK elevation pipeline.

    Parameters
    ----------
    srtm_dir:         local SRTM tile cache directory
    cache_path:       output JSON file path
    refresh_events:   re-fetch events.json even if cached
    refresh_overpass: re-fetch Overpass data even if cached
    force:            event_name to reprocess regardless of cache status
    limit:            stop after processing this many events (for testing)
    dry_run:          compute but do not write to cache or disk
    show_progress:    display a tqdm progress bar (set False in tests)
    retry_no_route:   reprocess events whose cached status is \"no_route\"
    """
    events = get_uk_events(refresh=refresh_events)
    logger.info("Loaded %d UK events", len(events))

    cache = Cache(cache_path)

    raw = fetch_overpass(refresh=refresh_overpass)
    elements = raw["elements"]
    node_lookup = build_node_lookup(elements)
    way_lookup = build_way_lookup(elements)
    relations = extract_relations(elements)
    logger.info("Loaded %d OSM parkrun relations", len(relations))

    n_complete = n_no_route = n_skipped = 0
    processed = 0

    event_iter = tqdm(events, desc="Events", unit="event") if show_progress else events
    for event in event_iter:
        if limit is not None and processed >= limit:
            break

        name = event["event_name"]

        skip = cache.is_known(name) and force != name
        if skip and retry_no_route:
            rec = cache.get_record(name)
            if rec and rec.get("status") == "no_route":
                skip = False
        if skip:
            n_skipped += 1
            logger.debug("Skipping %s (already known)", name)
            continue

        # Write a "pending" marker before starting so a crash mid-computation
        # leaves a recoverable record rather than a silent gap
        if not dry_run:
            cache.upsert({
                "event_name": name,
                "event_long_name": event["event_long_name"],
                "country_code": event["country_code"],
                "lat": event["lat"],
                "lng": event["lng"],
                "status": "pending",
                "date_computed": datetime.date.today().isoformat(),
            })

        route = _resolve_route(event, relations, way_lookup, node_lookup)
        status = process_event(event, route, srtm_dir, cache, dry_run=dry_run)

        if status == "complete":
            n_complete += 1
        else:
            n_no_route += 1

        processed += 1
        logger.info("[%d] %s → %s", processed, name, status)

    logger.info(
        "Done. complete=%d  no_route=%d  skipped=%d",
        n_complete,
        n_no_route,
        n_skipped,
    )
