"""
Route acquisition from parkrun course pages via Google My Maps KML export.

Every UK parkrun event page embeds a Google My Maps map. The map contains:
  - One LineString placemark named 'Course' — the route geometry
  - Several Point placemarks — start, finish, parking, toilets, etc.

The KML is freely exportable with no authentication:
    https://www.google.com/maps/d/kml?mid={mid}&forcekml=1

Lap handling
------------
Many courses are multi-lap. The KML LineString represents one lap; the course
page HTML describes how many laps runners complete. We parse this text to
determine:
  - "loop"         — N laps of the same circuit; multiply ascent/descent by N
  - "out_and_back" — run out and retrace; ascent_total = ascent_one_way + descent_one_way
  - "unknown"      — no explicit mention; infer laps from route length, flag as inferred

Rate limiting: two HTTP requests per event (course page + KML). Sleep 0.5s
between events to be polite to parkrun's servers.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

_KML_NS = "http://www.opengis.net/kml/2.2"
_COURSE_PAGE_URL = "https://www.parkrun.org.uk/{slug}/course/"
_KML_EXPORT_URL = "https://www.google.com/maps/d/kml?mid={mid}&forcekml=1"

_MIN_CLEAN_M = 4_500
_MAX_CLEAN_M = 5_500


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_course_page(slug: str) -> str:
    """Return the HTML of the parkrun course page for the given event slug."""
    url = _COURSE_PAGE_URL.format(slug=slug)
    response = httpx.get(url, headers=_BROWSER_HEADERS, timeout=15, follow_redirects=True)
    response.raise_for_status()
    return response.text


def fetch_kml(mid: str) -> str:
    """Return the raw KML text for a Google My Maps map ID."""
    url = _KML_EXPORT_URL.format(mid=mid)
    response = httpx.get(url, headers=_BROWSER_HEADERS, timeout=15, follow_redirects=True)
    response.raise_for_status()
    return response.text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_map_mid(html: str) -> str | None:
    """Extract the Google My Maps 'mid' parameter from an embedded iframe.

    Handles both URL formats found in the wild:
        .../maps/d/embed?mid=abc123
        .../maps/d/embed?t=h&mid=abc.def
        .../maps/d/u/0/embed?mid=abc123&ehbc=...
    """
    match = re.search(
        r"maps/d/[^?\"']*[?&](?:[^&\"']*&)*mid=([^&\"'> ]+)",
        html,
    )
    return match.group(1) if match else None


def extract_course_linestring(
    kml_text: str,
) -> list[tuple[float, float]] | None:
    """Return the course route as a list of (lat, lng) pairs.

    Prefers a Placemark named 'Course'. If no such name exists, falls back to
    the longest LineString in the document. Returns None if no LineString found.

    KML coordinates are stored as 'lng,lat,alt' — note the order.
    """
    root = ET.fromstring(kml_text.strip())

    candidates: list[tuple[int, list[tuple[float, float]], str]] = []

    for pm in root.iter(f"{{{_KML_NS}}}Placemark"):
        name_el = pm.find(f"{{{_KML_NS}}}name")
        name = (name_el.text or "").strip() if name_el is not None else ""

        line = pm.find(f".//{{{_KML_NS}}}LineString")
        if line is None:
            continue

        coords_el = line.find(f"{{{_KML_NS}}}coordinates")
        if coords_el is None or not coords_el.text:
            continue

        # KML is lng,lat,alt — swap to (lat, lng)
        pts = [
            (float(lat), float(lng))
            for lng, lat in re.findall(r"([-\d.]+),([-\d.]+)", coords_el.text)
        ]
        if pts:
            candidates.append((len(pts), pts, name))

    if not candidates:
        return None

    # Prefer placemark named 'Course'; otherwise take the longest linestring
    for _, pts, name in candidates:
        if name.lower() == "course":
            return pts

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def parse_lap_info(html: str, route_length_m: float) -> dict:
    """Determine lap count and route type from the course page HTML.

    Returns a dict with keys:
        laps        — integer number of laps (1 for a clean 5km route)
        route_type  — "loop", "out_and_back", or "unknown"
        laps_inferred — True if lap count was estimated from route length,
                        not read from the page text

    Strategy (in order):
        1. Explicit "out and back" mention → laps=2, type="out_and_back"
        2. Explicit number + "lap(s)" → laps=N, type="loop"
        3. "twice" → laps=2, "three times" → laps=3, type="loop"
        4. Route length 4,500–5,500m → laps=1, type="loop"
        5. Fallback: infer from length with round(5000/length)
    """
    # Strip HTML tags for cleaner text matching
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).lower()

    # Out-and-back check first — takes priority over lap count
    if re.search(r"out[\s-]and[\s-]back", text):
        return {"laps": 2, "route_type": "out_and_back", "laps_inferred": False}

    # Explicit lap count: "3 lap", "three lap", "5-lap", etc.
    _words = {"single": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    lap_match = re.search(
        r"(\d+|single|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"[\s-]*(?:and a bit[\s-]*)?"
        r"lap",
        text,
    )
    if lap_match:
        raw = lap_match.group(1)
        n = int(raw) if raw.isdigit() else _words.get(raw, 0)
        if n > 0:
            return {"laps": n, "route_type": "loop", "laps_inferred": False}

    # "twice" / "three times"
    if re.search(r"\btwice\b", text):
        return {"laps": 2, "route_type": "loop", "laps_inferred": False}
    if re.search(r"\bthree times\b", text):
        return {"laps": 3, "route_type": "loop", "laps_inferred": False}

    # Clean 5km route — no laps needed
    if _MIN_CLEAN_M <= route_length_m <= _MAX_CLEAN_M:
        return {"laps": 1, "route_type": "loop", "laps_inferred": False}

    # Fallback: estimate from length
    inferred = max(1, round(5_000 / route_length_m))
    return {"laps": inferred, "route_type": "unknown", "laps_inferred": True}


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def get_kml_route(event: dict, _sleep: float = 0.5) -> dict | None:
    """Fetch and parse the KML route for a single event.

    Returns a dict with keys:
        coords      — list of (lat, lng) for one lap
        laps        — number of laps
        route_type  — "loop", "out_and_back", or "unknown"
        laps_inferred — bool
        mid         — the Google My Maps map ID

    Returns None if no map mid or no Course linestring is found.

    _sleep: seconds to pause after fetching (pass 0 in tests to skip delay).
    """
    slug = event["event_name"]

    try:
        html = fetch_course_page(slug)
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch course page for %s: %s", slug, exc)
        return None

    mid = extract_map_mid(html)
    if not mid:
        logger.info("No Google My Maps mid found for %s", slug)
        return None

    try:
        kml_text = fetch_kml(mid)
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch KML for %s (mid=%s): %s", slug, mid, exc)
        return None

    coords = extract_course_linestring(kml_text)
    if not coords:
        logger.info("No Course linestring in KML for %s", slug)
        return None

    from haversine import haversine
    route_length_m = sum(
        haversine(coords[i - 1], coords[i], unit="m")
        for i in range(1, len(coords))
    )

    lap_info = parse_lap_info(html, route_length_m)

    if _sleep:
        time.sleep(_sleep)

    logger.info(
        "%s: %.0fm × %d lap(s) (%s)%s",
        slug,
        route_length_m,
        lap_info["laps"],
        lap_info["route_type"],
        " [inferred]" if lap_info["laps_inferred"] else "",
    )

    return {
        "coords": coords,
        "mid": mid,
        **lap_info,
    }
