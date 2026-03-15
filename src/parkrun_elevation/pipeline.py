"""
Pipeline orchestration for the parkrun elevation dataset.

Entry point: run()
Inner loop:  process_event()

Build order per event:
    1. Cache check          — skip if already known (idempotent)
    2. Write "pending"      — crash recovery marker
    3. OSM route lookup     — match_relation() against bulk Overpass data
    4. Elevation profile    — compute_elevation_profile() via SRTM
    5. Write final record   — "complete" or "no_route"
"""

import datetime
import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)

_SRTM_DIR = Path("srtm")
_CACHE_PATH = Path("data/elevation/uk.json")


def process_event(
    event: dict,
    osm_result: dict | None,
    srtm_dir: Path,
    cache: Cache,
    dry_run: bool = False,
) -> str:
    """Process a single event: compute elevation and write a cache record.

    Parameters
    ----------
    event:      event dict from get_uk_events()
    osm_result: return value of match_relation(), or None if no route found
    srtm_dir:   path to the local SRTM tile cache
    cache:      Cache instance (written unless dry_run=True)
    dry_run:    if True, compute everything but do not write to cache

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

    if osm_result is None:
        record = {**base, "status": "no_route"}
        if not dry_run:
            cache.upsert(record)
        return "no_route"

    coords = osm_result["coords"]
    relation = osm_result["relation"]
    relation_id = relation["id"]

    elevation_data = compute_elevation_profile(
        lats=[c[0] for c in coords],
        lngs=[c[1] for c in coords],
        srtm_dir=srtm_dir,
        threshold=3.0,
    )

    record = {
        **base,
        "status": "complete",
        "route_source_type": "osm",
        "route_source_id": f"relation/{relation_id}",
        "route_source_url": f"https://www.openstreetmap.org/relation/{relation_id}",
        **elevation_data,
    }

    if not dry_run:
        cache.upsert(record)

    return "complete"


def run(
    srtm_dir: Path = _SRTM_DIR,
    cache_path: Path = _CACHE_PATH,
    refresh_events: bool = False,
    refresh_overpass: bool = False,
    force: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    show_progress: bool = False,
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

        if cache.is_known(name) and force != name:
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

        osm_result = match_relation(event, relations, way_lookup, node_lookup)
        status = process_event(event, osm_result, srtm_dir, cache, dry_run=dry_run)

        if status == "complete":
            n_complete += 1
        else:
            n_no_route += 1

        processed += 1
        logger.info(
            "[%d] %s → %s",
            processed,
            name,
            status,
        )

    logger.info(
        "Done. complete=%d  no_route=%d  skipped=%d",
        n_complete,
        n_no_route,
        n_skipped,
    )
