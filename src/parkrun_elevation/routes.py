"""
Route acquisition from OpenStreetMap via the Overpass API.

Strategy: one bulk Overpass query fetches all UK parkrun route relations.
The raw response is cached locally so subsequent runs work offline.
Relations are matched to events in memory using three filters:
    1. Length         — 4,500m to 15,500m
    2. Proximity      — centroid within 500m of event coordinates
    3. Name (soft)    — OSM name tag contains "parkrun" (sanity check only)

Name normalisation is NOT used as a hard equality gate because EventLongName
values often include location qualifiers or descriptive words that differ from
the URL slug (e.g. "Wimbledon Common parkrun" → slug "wimbledon").
"""

import json
import logging
import re
import time
from pathlib import Path

import httpx
from haversine import haversine

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_CACHE_PATH = Path("data/overpass_uk.json")

# Overpass QL: all running route relations whose name contains "parkrun",
# bounded to the UK bounding box.
_OVERPASS_QUERY = """
[out:json][timeout:120];
relation["route"="running"]["name"~"parkrun",i](49.5,-8.5,61.0,2.0);
out body;
>;
out skel qt;
""".strip()

_MIN_ROUTE_M = 4_500
_MAX_ROUTE_M = 15_500
_MAX_PROXIMITY_M = 500


# ---------------------------------------------------------------------------
# Overpass fetch and cache
# ---------------------------------------------------------------------------

def fetch_overpass(cache_path: Path = OVERPASS_CACHE_PATH, refresh: bool = False) -> dict:
    """Return the raw Overpass JSON for all UK parkrun relations.

    Uses the cached file unless refresh=True or the file doesn't exist.
    The single bulk query avoids per-event requests and is polite to the API.
    """
    if not refresh and cache_path.exists():
        logger.info("Loading Overpass data from cache: %s", cache_path)
        return json.loads(cache_path.read_text())

    logger.info("Querying Overpass API (this may take ~30 seconds)…")
    response = httpx.post(
        OVERPASS_URL,
        data={"data": _OVERPASS_QUERY},
        timeout=150,
    )
    response.raise_for_status()
    raw = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(raw, indent=2))
    n_relations = sum(1 for e in raw["elements"] if e["type"] == "relation")
    logger.info("Saved %d relations to %s", n_relations, cache_path)

    return raw


# ---------------------------------------------------------------------------
# Parsing — build in-memory lookups from the flat elements list
# ---------------------------------------------------------------------------

def build_node_lookup(elements: list) -> dict[int, tuple[float, float]]:
    """Return {node_id: (lat, lng)} for all node elements."""
    return {
        e["id"]: (e["lat"], e["lon"])
        for e in elements
        if e["type"] == "node"
    }


def build_way_lookup(elements: list) -> dict[int, list[int]]:
    """Return {way_id: [node_ids]} for all way elements."""
    return {
        e["id"]: e["nodes"]
        for e in elements
        if e["type"] == "way"
    }


def extract_relations(elements: list) -> list[dict]:
    """Return all relation elements whose name tag contains 'parkrun'."""
    return [
        e for e in elements
        if e["type"] == "relation"
        and "parkrun" in e.get("tags", {}).get("name", "").lower()
    ]


# ---------------------------------------------------------------------------
# Geometry — pure functions
# ---------------------------------------------------------------------------

def extract_polyline(
    relation: dict,
    way_lookup: dict[int, list[int]],
    node_lookup: dict[int, tuple[float, float]],
) -> list[tuple[float, float]]:
    """Build an ordered list of (lat, lng) points from a relation.

    Walks the relation's way members in order. If two adjacent ways don't
    chain end-to-start, the second way is reversed before connecting.
    Unknown way IDs (ways outside the query bbox) are skipped silently.
    """
    way_ids = [
        m["ref"] for m in relation.get("members", [])
        if m["type"] == "way"
    ]

    if not way_ids:
        return []

    # Build a list of node-id sequences for each way, skipping missing ways
    segments: list[list[int]] = []
    for way_id in way_ids:
        if way_id in way_lookup:
            segments.append(list(way_lookup[way_id]))

    if not segments:
        return []

    # Chain segments: reverse a segment if its start doesn't connect to the
    # previous segment's end
    chained = list(segments[0])
    for seg in segments[1:]:
        if seg[0] == chained[-1]:
            chained.extend(seg[1:])
        elif seg[-1] == chained[-1]:
            chained.extend(reversed(seg[:-1]))
        else:
            # Gap — append anyway to avoid silently dropping route geometry
            chained.extend(seg)

    # Convert node IDs to coordinates, skipping any unknown nodes
    coords = [node_lookup[nid] for nid in chained if nid in node_lookup]
    return coords


def calc_centroid(coords: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the mean (lat, lng) of a list of coordinate pairs."""
    lats = [c[0] for c in coords]
    lngs = [c[1] for c in coords]
    return sum(lats) / len(lats), sum(lngs) / len(lngs)


def calc_route_length_m(coords: list[tuple[float, float]]) -> float:
    """Return the total route length in metres using the haversine formula."""
    total = 0.0
    for i in range(1, len(coords)):
        total += haversine(coords[i - 1], coords[i], unit="m")
    return total


# ---------------------------------------------------------------------------
# Name normalisation — utility, not used as a hard match gate
# ---------------------------------------------------------------------------

def normalise_name(name: str) -> str:
    """Lowercase, remove 'parkrun', strip non-alphanumeric characters.

    Useful for quickly excluding clearly unrelated relations, but NOT reliable
    enough for exact slug matching due to location qualifiers in long names
    (e.g. "Wimbledon Common parkrun" → "wimbledoncommon" ≠ slug "wimbledon").
    """
    name = name.lower()
    name = re.sub(r"\bparkrun\b", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_relation(
    event: dict,
    relations: list[dict],
    way_lookup: dict[int, list[int]],
    node_lookup: dict[int, tuple[float, float]],
) -> dict | None:
    """Find the OSM relation that best matches this event.

    Filters applied in order (cheapest first):
        1. OSM name contains "parkrun"      — already guaranteed by extract_relations
        2. Route length 4,500–15,500m       — eliminates fragments and multi-lap doubles
        3. Centroid within 500m of event    — the reliable primary gate

    If multiple relations pass all filters (rare), returns the closest by centroid.
    Returns None if no match is found.
    """
    event_pos = (event["lat"], event["lng"])
    candidates = []

    for rel in relations:
        coords = extract_polyline(rel, way_lookup, node_lookup)
        if len(coords) < 2:
            continue

        length = calc_route_length_m(coords)
        if not (_MIN_ROUTE_M <= length <= _MAX_ROUTE_M):
            continue

        centroid = calc_centroid(coords)
        distance = haversine(event_pos, centroid, unit="m")
        if distance > _MAX_PROXIMITY_M:
            continue

        candidates.append((distance, rel, coords))

    if not candidates:
        return None

    # Return the closest match
    candidates.sort(key=lambda x: x[0])
    _, best_rel, best_coords = candidates[0]
    return {"relation": best_rel, "coords": best_coords}


def get_osm_route(
    event: dict,
    relations: list[dict],
    way_lookup: dict[int, list[int]],
    node_lookup: dict[int, tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """Return the matched polyline for an event, or None if no OSM route found."""
    result = match_relation(event, relations, way_lookup, node_lookup)
    if result is None:
        return None
    return result["coords"]
