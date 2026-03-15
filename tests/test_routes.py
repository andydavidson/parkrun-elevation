"""
Tests for routes.py (OSM half)

All tests below are pure / offline — they use synthetic Overpass-shaped
fixtures. The live test at the bottom hits the real Overpass API.

Run fast tests:  pytest tests/test_routes.py -v
Run live test:   pytest tests/test_routes.py -v -m live
"""

import json
import math

import pytest
from haversine import haversine

from parkrun_elevation.routes import (
    build_node_lookup,
    build_way_lookup,
    calc_centroid,
    calc_route_length_m,
    extract_polyline,
    extract_relations,
    fetch_overpass,
    get_osm_route,
    match_relation,
    normalise_name,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Winchester is at roughly (51.069, -1.311).
# We build a tiny synthetic square route (~5km perimeter) centred nearby.
# Each side is ~1.25km; 4 sides = ~5km total.

def _square_coords(centre_lat, centre_lng, side_m=1250):
    """Return 5 (lat, lng) pairs forming a closed square of given side length."""
    dlat = (side_m / 2) / 111_320
    dlng = (side_m / 2) / (111_320 * math.cos(math.radians(centre_lat)))
    return [
        (centre_lat - dlat, centre_lng - dlng),  # SW
        (centre_lat - dlat, centre_lng + dlng),  # SE
        (centre_lat + dlat, centre_lng + dlng),  # NE
        (centre_lat + dlat, centre_lng - dlng),  # NW
        (centre_lat - dlat, centre_lng - dlng),  # back to SW (closed)
    ]


WINCHESTER_LAT, WINCHESTER_LNG = 51.069, -1.311
ROUTE_COORDS = _square_coords(WINCHESTER_LAT, WINCHESTER_LNG, side_m=1250)

# Build synthetic Overpass elements for this route
# Nodes 1–5 form the square; nodes are at the route vertices
NODES = [
    {"type": "node", "id": i + 1, "lat": lat, "lon": lng}
    for i, (lat, lng) in enumerate(ROUTE_COORDS)
]

# Two ways: way 10 covers nodes 1-3, way 11 covers nodes 3-5
WAYS = [
    {"type": "way", "id": 10, "nodes": [1, 2, 3]},
    {"type": "way", "id": 11, "nodes": [3, 4, 5]},
]

RELATION = {
    "type": "relation",
    "id": 999,
    "members": [
        {"type": "way", "ref": 10},
        {"type": "way", "ref": 11},
    ],
    "tags": {"name": "Winchester parkrun", "route": "running"},
}

ELEMENTS = NODES + WAYS + [RELATION]

NODE_LOOKUP = build_node_lookup(ELEMENTS)
WAY_LOOKUP = build_way_lookup(ELEMENTS)
RELATIONS = extract_relations(ELEMENTS)

WINCHESTER_EVENT = {
    "event_name": "winchester",
    "event_long_name": "Winchester parkrun",
    "lat": WINCHESTER_LAT,
    "lng": WINCHESTER_LNG,
    "country_code": 97,
}


# ---------------------------------------------------------------------------
# normalise_name
# ---------------------------------------------------------------------------

def test_normalise_simple():
    assert normalise_name("Winchester parkrun") == "winchester"


def test_normalise_strips_location_qualifier():
    assert normalise_name("Albert parkrun, Middlesbrough") == "albertmiddlesbrough"


def test_normalise_capital_parkrun():
    assert normalise_name("Bushy Parkrun") == "bushy"


def test_normalise_multi_word():
    assert normalise_name("Great Notley parkrun") == "greatnotley"


def test_normalise_ampersand():
    assert normalise_name("Brighton & Hove parkrun") == "brightonhove"


# ---------------------------------------------------------------------------
# build_node_lookup
# ---------------------------------------------------------------------------

def test_node_lookup_count():
    lookup = build_node_lookup(ELEMENTS)
    assert len(lookup) == len(NODES)


def test_node_lookup_coordinates():
    lookup = build_node_lookup(ELEMENTS)
    lat, lng = lookup[1]
    assert lat == pytest.approx(ROUTE_COORDS[0][0])
    assert lng == pytest.approx(ROUTE_COORDS[0][1])


def test_node_lookup_ignores_ways_and_relations():
    lookup = build_node_lookup(ELEMENTS)
    assert 10 not in lookup   # way id
    assert 999 not in lookup  # relation id


# ---------------------------------------------------------------------------
# build_way_lookup
# ---------------------------------------------------------------------------

def test_way_lookup_count():
    lookup = build_way_lookup(ELEMENTS)
    assert len(lookup) == len(WAYS)


def test_way_lookup_node_ids():
    lookup = build_way_lookup(ELEMENTS)
    assert lookup[10] == [1, 2, 3]
    assert lookup[11] == [3, 4, 5]


# ---------------------------------------------------------------------------
# extract_relations
# ---------------------------------------------------------------------------

def test_extract_relations_finds_parkrun():
    rels = extract_relations(ELEMENTS)
    assert len(rels) == 1
    assert rels[0]["id"] == 999


def test_extract_relations_excludes_non_parkrun():
    extra = [{"type": "relation", "id": 1, "members": [], "tags": {"name": "Cycling route"}}]
    rels = extract_relations(ELEMENTS + extra)
    ids = [r["id"] for r in rels]
    assert 1 not in ids


def test_extract_relations_case_insensitive():
    extra = [{"type": "relation", "id": 2, "members": [], "tags": {"name": "SOME PARKRUN THING"}}]
    rels = extract_relations(ELEMENTS + extra)
    ids = [r["id"] for r in rels]
    assert 2 in ids


# ---------------------------------------------------------------------------
# extract_polyline
# ---------------------------------------------------------------------------

def test_extract_polyline_point_count():
    coords = extract_polyline(RELATION, WAY_LOOKUP, NODE_LOOKUP)
    # way 10 has 3 nodes, way 11 adds 2 more (node 3 shared) = 5 total
    assert len(coords) == 5


def test_extract_polyline_correct_start():
    coords = extract_polyline(RELATION, WAY_LOOKUP, NODE_LOOKUP)
    assert coords[0] == pytest.approx(ROUTE_COORDS[0], abs=1e-6)


def test_extract_polyline_reversed_way():
    """If the second way is stored in reverse, it should be flipped to chain."""
    reversed_relation = {
        "type": "relation",
        "id": 998,
        "members": [
            {"type": "way", "ref": 10},
            {"type": "way", "ref": 20},   # way 20 is nodes [5, 4, 3] — reversed
        ],
        "tags": {"name": "Test parkrun"},
    }
    reversed_way = {"type": "way", "id": 20, "nodes": [5, 4, 3]}
    extended_elements = ELEMENTS + [reversed_way]
    way_lookup = build_way_lookup(extended_elements)
    node_lookup = build_node_lookup(extended_elements)

    coords = extract_polyline(reversed_relation, way_lookup, node_lookup)
    # Should still produce a 5-point sequence without duplicating node 3
    assert len(coords) == 5


def test_extract_polyline_missing_way_skipped():
    """A way ID that doesn't exist in way_lookup should be silently skipped."""
    relation_with_ghost = {
        "type": "relation",
        "id": 997,
        "members": [
            {"type": "way", "ref": 10},
            {"type": "way", "ref": 99999},  # doesn't exist
        ],
        "tags": {"name": "Ghost parkrun"},
    }
    coords = extract_polyline(relation_with_ghost, WAY_LOOKUP, NODE_LOOKUP)
    assert len(coords) > 0


def test_extract_polyline_empty_members():
    empty_rel = {"type": "relation", "id": 0, "members": [], "tags": {}}
    coords = extract_polyline(empty_rel, WAY_LOOKUP, NODE_LOOKUP)
    assert coords == []


# ---------------------------------------------------------------------------
# calc_centroid
# ---------------------------------------------------------------------------

def test_calc_centroid_simple():
    coords = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    lat, lng = calc_centroid(coords)
    assert lat == pytest.approx(1.0)
    assert lng == pytest.approx(1.0)


def test_calc_centroid_single_point():
    lat, lng = calc_centroid([(51.5, -1.3)])
    assert lat == pytest.approx(51.5)
    assert lng == pytest.approx(-1.3)


# ---------------------------------------------------------------------------
# calc_route_length_m
# ---------------------------------------------------------------------------

def test_route_length_two_points_known_distance():
    """Two points ~1km apart should return a length within 1% of 1000m."""
    # Moving ~0.009 degrees latitude ≈ 1km
    coord_a = (51.0, -1.0)
    coord_b = (51.009, -1.0)
    length = calc_route_length_m([coord_a, coord_b])
    assert abs(length - 1000) < 10


def test_route_length_single_segment():
    length = calc_route_length_m([(51.0, -1.0), (51.0, -1.0)])
    assert length == pytest.approx(0.0, abs=1e-3)


def test_route_length_square_is_roughly_5km():
    length = calc_route_length_m(ROUTE_COORDS)
    # 4 sides × 1250m = 5000m; expect within 2% given haversine vs flat approx
    assert 4_800 < length < 5_200


# ---------------------------------------------------------------------------
# match_relation
# ---------------------------------------------------------------------------

def test_match_relation_success():
    result = match_relation(WINCHESTER_EVENT, RELATIONS, WAY_LOOKUP, NODE_LOOKUP)
    assert result is not None
    assert result["relation"]["id"] == 999


def test_match_relation_wrong_location():
    """An event far from the route centroid should not match."""
    amsterdam_event = {
        "event_name": "amsterdam",
        "lat": 52.37,
        "lng": 4.90,
        "country_code": 74,
    }
    result = match_relation(amsterdam_event, RELATIONS, WAY_LOOKUP, NODE_LOOKUP)
    assert result is None


def test_match_relation_route_too_short():
    """A route shorter than 4,500m should be filtered out."""
    tiny_coords = _square_coords(WINCHESTER_LAT, WINCHESTER_LNG, side_m=100)
    tiny_nodes = [
        {"type": "node", "id": 100 + i, "lat": lat, "lon": lng}
        for i, (lat, lng) in enumerate(tiny_coords)
    ]
    tiny_way = {"type": "way", "id": 50, "nodes": [100, 101, 102, 103, 104]}
    tiny_relation = {
        "type": "relation", "id": 50,
        "members": [{"type": "way", "ref": 50}],
        "tags": {"name": "Winchester parkrun"},
    }
    elements = tiny_nodes + [tiny_way, tiny_relation]
    result = match_relation(
        WINCHESTER_EVENT,
        extract_relations(elements),
        build_way_lookup(elements),
        build_node_lookup(elements),
    )
    assert result is None


def test_match_relation_returns_closest_when_multiple():
    """When two relations both pass all filters, the closer one wins."""
    # Second relation, further away (0.004° ≈ 440m north of event)
    offset_coords = _square_coords(WINCHESTER_LAT + 0.004, WINCHESTER_LNG, side_m=1250)
    offset_nodes = [
        {"type": "node", "id": 200 + i, "lat": lat, "lon": lng}
        for i, (lat, lng) in enumerate(offset_coords)
    ]
    offset_way = {"type": "way", "id": 60, "nodes": [200, 201, 202, 203, 204]}
    offset_relation = {
        "type": "relation", "id": 60,
        "members": [{"type": "way", "ref": 60}],
        "tags": {"name": "Winchester parkrun"},
    }
    all_elements = ELEMENTS + offset_nodes + [offset_way, offset_relation]
    result = match_relation(
        WINCHESTER_EVENT,
        extract_relations(all_elements),
        build_way_lookup(all_elements),
        build_node_lookup(all_elements),
    )
    # Should pick the original relation (id=999), which is centred on the event
    assert result["relation"]["id"] == 999


# ---------------------------------------------------------------------------
# get_osm_route
# ---------------------------------------------------------------------------

def test_get_osm_route_returns_coords():
    coords = get_osm_route(WINCHESTER_EVENT, RELATIONS, WAY_LOOKUP, NODE_LOOKUP)
    assert coords is not None
    assert len(coords) == 5


def test_get_osm_route_no_match_returns_none():
    far_event = {"event_name": "nowhere", "lat": 0.0, "lng": 0.0, "country_code": 97}
    result = get_osm_route(far_event, RELATIONS, WAY_LOOKUP, NODE_LOOKUP)
    assert result is None


# ---------------------------------------------------------------------------
# fetch_overpass — cache hit (no network)
# ---------------------------------------------------------------------------

def test_fetch_overpass_reads_cache(tmp_path, monkeypatch):
    import parkrun_elevation.routes as routes_module

    cache_file = tmp_path / "overpass_uk.json"
    fake_data = {"elements": []}
    cache_file.write_text(json.dumps(fake_data))
    monkeypatch.setattr(routes_module, "OVERPASS_CACHE_PATH", cache_file)

    result = fetch_overpass(cache_path=cache_file, refresh=False)
    assert result == fake_data


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_osm_route(tmp_path):
    """Fetch real Overpass data and match Lingwood parkrun.

    Lingwood is one of the best-matched OSM relations (centroid only 25m from
    the events.json coordinates) and makes a reliable smoke test.
    """
    raw = fetch_overpass(cache_path=tmp_path / "overpass_uk.json", refresh=True)
    elements = raw["elements"]

    node_lookup = build_node_lookup(elements)
    way_lookup = build_way_lookup(elements)
    relations = extract_relations(elements)

    assert len(relations) > 100, f"Expected >100 UK parkrun relations, got {len(relations)}"

    event = {"event_name": "lingwood", "lat": 52.619, "lng": 1.494, "country_code": 97}
    coords = get_osm_route(event, relations, way_lookup, node_lookup)

    assert coords is not None, "Lingwood parkrun not found in OSM data"
    assert len(coords) > 50, f"Suspiciously few points: {len(coords)}"

    length = calc_route_length_m(coords)
    assert 4_500 <= length <= 15_500, f"Unexpected route length: {length:.0f}m"
