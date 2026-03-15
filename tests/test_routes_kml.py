"""
Tests for routes_kml.py — all fast/offline except the live test at the bottom.
"""

import textwrap

import pytest

from parkrun_elevation.routes_kml import (
    extract_course_linestring,
    extract_map_mid,
    get_kml_route,
    parse_lap_info,
)


# ---------------------------------------------------------------------------
# KML fixture helpers
# ---------------------------------------------------------------------------

def _kml(placemarks: str) -> str:
    """Wrap placemark XML in a minimal KML document."""
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document>
            {placemarks}
          </Document>
        </kml>
    """)


def _linestring(name: str, coords: list[tuple[float, float]]) -> str:
    """Build a KML LineString placemark. coords are (lat, lng)."""
    coord_text = "\n".join(f"{lng},{lat},0" for lat, lng in coords)
    return textwrap.dedent(f"""\
        <Placemark>
          <name>{name}</name>
          <LineString>
            <coordinates>{coord_text}</coordinates>
          </LineString>
        </Placemark>
    """)


def _point(name: str, lat: float, lng: float) -> str:
    return textwrap.dedent(f"""\
        <Placemark>
          <name>{name}</name>
          <Point><coordinates>{lng},{lat},0</coordinates></Point>
        </Placemark>
    """)


# A simple 5-point square route centred near Winchester
COURSE_COORDS = [
    (51.069, -1.311),
    (51.073, -1.311),
    (51.073, -1.305),
    (51.069, -1.305),
    (51.069, -1.311),
]

FULL_KML = _kml(
    _linestring("Course", COURSE_COORDS)
    + _point("Start", 51.069, -1.311)
    + _point("Finish", 51.069, -1.311)
    + _point("Parking", 51.068, -1.312)
)


# ---------------------------------------------------------------------------
# extract_map_mid
# ---------------------------------------------------------------------------

def test_extract_map_mid_new_format():
    html = '<iframe src="https://www.google.com/maps/d/u/0/embed?mid=1fArCc2AO9y6_PZi0a9QLv6t&ehbc=2E312F">'
    assert extract_map_mid(html) == "1fArCc2AO9y6_PZi0a9QLv6t"


def test_extract_map_mid_old_format():
    html = '<iframe src="https://www.google.com/maps/d/embed?t=h&mid=zj7h2Fr7knm4.kmi-jDrZZSRc">'
    assert extract_map_mid(html) == "zj7h2Fr7knm4.kmi-jDrZZSRc"


def test_extract_map_mid_plain_format():
    html = '<iframe src="https://www.google.com/maps/d/embed?mid=abc123XYZ">'
    assert extract_map_mid(html) == "abc123XYZ"


def test_extract_map_mid_missing():
    html = '<iframe src="https://www.parkrun.com/parkrun-smart-cookie/">'
    assert extract_map_mid(html) is None


def test_extract_map_mid_no_iframe():
    assert extract_map_mid("<html><body>No map here</body></html>") is None


# ---------------------------------------------------------------------------
# extract_course_linestring
# ---------------------------------------------------------------------------

def test_extract_course_linestring_returns_course():
    coords = extract_course_linestring(FULL_KML)
    assert coords is not None
    assert len(coords) == len(COURSE_COORDS)


def test_extract_course_linestring_correct_latlon_order():
    """KML stores lng,lat — verify we unpack as (lat, lng)."""
    coords = extract_course_linestring(FULL_KML)
    # First point should be (51.069, -1.311), not (-1.311, 51.069)
    assert coords[0][0] == pytest.approx(51.069)
    assert coords[0][1] == pytest.approx(-1.311)


def test_extract_course_linestring_ignores_points():
    """Point placemarks (start, finish, parking) must not appear in result."""
    coords = extract_course_linestring(FULL_KML)
    # Result should only contain the 5 route coords, not the 3 point placemarks
    assert len(coords) == 5


def test_extract_course_linestring_prefers_course_name():
    """If multiple LineStrings exist, the one named 'Course' wins."""
    kml = _kml(
        _linestring("Other route", [(52.0, -1.0), (52.1, -1.0), (52.1, -0.9)] * 10)
        + _linestring("Course", COURSE_COORDS)
    )
    coords = extract_course_linestring(kml)
    assert len(coords) == len(COURSE_COORDS)


def test_extract_course_linestring_fallback_to_longest():
    """With no 'Course' name, return the longest LineString."""
    kml = _kml(
        _linestring("Short", [(52.0, -1.0), (52.1, -1.0)])
        + _linestring("Long", COURSE_COORDS)
    )
    coords = extract_course_linestring(kml)
    assert len(coords) == len(COURSE_COORDS)


def test_extract_course_linestring_no_linestring():
    kml = _kml(_point("Start", 51.069, -1.311) + _point("Finish", 51.069, -1.311))
    assert extract_course_linestring(kml) is None


def test_extract_course_linestring_empty_document():
    kml = _kml("")
    assert extract_course_linestring(kml) is None


# ---------------------------------------------------------------------------
# parse_lap_info
# ---------------------------------------------------------------------------

def test_parse_lap_info_explicit_number_digit():
    html = "<p>This is a 3 lap course starting at the bandstand.</p>"
    result = parse_lap_info(html, 1667.0)
    assert result["laps"] == 3
    assert result["route_type"] == "loop"
    assert result["laps_inferred"] is False


def test_parse_lap_info_explicit_number_word():
    html = "<p>Course Description This is a three lap course.</p>"
    result = parse_lap_info(html, 1667.0)
    assert result["laps"] == 3
    assert result["route_type"] == "loop"
    assert result["laps_inferred"] is False


def test_parse_lap_info_five_lap():
    html = "<p>5-lap course starts on the tarmac path.</p>"
    result = parse_lap_info(html, 1000.0)
    assert result["laps"] == 5
    assert result["route_type"] == "loop"


def test_parse_lap_info_out_and_back():
    html = "<p>This is an out and back course along the tarmac path.</p>"
    result = parse_lap_info(html, 2473.0)
    assert result["laps"] == 2
    assert result["route_type"] == "out_and_back"
    assert result["laps_inferred"] is False


def test_parse_lap_info_out_and_back_hyphenated():
    html = "<p>An out-and-back course heading north.</p>"
    result = parse_lap_info(html, 2500.0)
    assert result["route_type"] == "out_and_back"


def test_parse_lap_info_out_and_back_beats_lap_count():
    """out-and-back should take priority even if a lap count appears elsewhere."""
    html = "<p>Complete one out and back section, then one lap of the field.</p>"
    result = parse_lap_info(html, 2000.0)
    assert result["route_type"] == "out_and_back"


def test_parse_lap_info_twice():
    html = "<p>Run the loop twice to complete the 5km course.</p>"
    result = parse_lap_info(html, 2500.0)
    assert result["laps"] == 2
    assert result["route_type"] == "loop"
    assert result["laps_inferred"] is False


def test_parse_lap_info_three_times():
    html = "<p>Runners complete the circuit three times.</p>"
    result = parse_lap_info(html, 1667.0)
    assert result["laps"] == 3
    assert result["route_type"] == "loop"


def test_parse_lap_info_clean_5k():
    html = "<p>A single scenic lap around the park.</p>"
    result = parse_lap_info(html, 5000.0)
    assert result["laps"] == 1
    assert result["route_type"] == "loop"
    assert result["laps_inferred"] is False


def test_parse_lap_info_clean_5k_boundary_low():
    result = parse_lap_info("", 4500.0)
    assert result["laps"] == 1
    assert result["laps_inferred"] is False


def test_parse_lap_info_clean_5k_boundary_high():
    result = parse_lap_info("", 5500.0)
    assert result["laps"] == 1
    assert result["laps_inferred"] is False


def test_parse_lap_info_fallback_infers_laps():
    html = "<p>A lovely run through the park.</p>"
    result = parse_lap_info(html, 2500.0)
    assert result["laps"] == 2
    assert result["route_type"] == "unknown"
    assert result["laps_inferred"] is True


def test_parse_lap_info_fallback_short_course():
    result = parse_lap_info("", 1250.0)
    assert result["laps"] == 4
    assert result["laps_inferred"] is True


def test_parse_lap_info_and_a_bit():
    """'3 and a bit laps' should still register as 3."""
    html = "<p>The course is 3 and a bit laps covering the park.</p>"
    result = parse_lap_info(html, 1500.0)
    assert result["laps"] == 3
    assert result["laps_inferred"] is False


# ---------------------------------------------------------------------------
# get_kml_route — offline using monkeypatching
# ---------------------------------------------------------------------------

WINCHESTER_EVENT = {
    "event_name": "winchester",
    "event_long_name": "Winchester parkrun",
    "lat": 51.069,
    "lng": -1.311,
    "country_code": 97,
}

_WINCHESTER_HTML = '<iframe src="https://www.google.com/maps/d/embed?mid=TEST123">'
_SINGLE_LAP_HTML = _WINCHESTER_HTML + "<p>A single lap of the park.</p>"
_TWO_LAP_HTML = _WINCHESTER_HTML + "<p>Two lap course around the park.</p>"
_OAB_HTML = _WINCHESTER_HTML + "<p>Out and back course.</p>"


def test_get_kml_route_success(monkeypatch):
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_course_page", lambda slug: _SINGLE_LAP_HTML)
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_kml", lambda mid: FULL_KML)

    result = get_kml_route(WINCHESTER_EVENT, _sleep=0)

    assert result is not None
    assert len(result["coords"]) == len(COURSE_COORDS)
    assert result["mid"] == "TEST123"
    assert result["laps"] == 1
    assert result["route_type"] == "loop"


def test_get_kml_route_two_laps(monkeypatch):
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_course_page", lambda slug: _TWO_LAP_HTML)
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_kml", lambda mid: FULL_KML)

    result = get_kml_route(WINCHESTER_EVENT, _sleep=0)

    assert result["laps"] == 2
    assert result["route_type"] == "loop"


def test_get_kml_route_out_and_back(monkeypatch):
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_course_page", lambda slug: _OAB_HTML)
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_kml", lambda mid: FULL_KML)

    result = get_kml_route(WINCHESTER_EVENT, _sleep=0)

    assert result["route_type"] == "out_and_back"


def test_get_kml_route_no_mid_returns_none(monkeypatch):
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_course_page",
                        lambda slug: "<html>No map here</html>")

    assert get_kml_route(WINCHESTER_EVENT, _sleep=0) is None


def test_get_kml_route_no_linestring_returns_none(monkeypatch):
    html = '<iframe src="https://www.google.com/maps/d/embed?mid=TEST123">'
    kml = _kml(_point("Start", 51.069, -1.311))
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_course_page", lambda slug: html)
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_kml", lambda mid: kml)

    assert get_kml_route(WINCHESTER_EVENT, _sleep=0) is None


def test_get_kml_route_http_error_returns_none(monkeypatch):
    import httpx
    monkeypatch.setattr("parkrun_elevation.routes_kml.fetch_course_page",
                        lambda slug: (_ for _ in ()).throw(httpx.HTTPError("timeout")))

    assert get_kml_route(WINCHESTER_EVENT, _sleep=0) is None


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_winchester_kml():
    """Fetch Winchester's course page and KML, verify the full round-trip."""
    import time
    from haversine import haversine

    result = get_kml_route(WINCHESTER_EVENT, _sleep=0)

    assert result is not None, "No KML route found for Winchester"
    assert len(result["coords"]) > 50, f"Too few points: {len(result['coords'])}"

    coords = result["coords"]
    one_lap_m = sum(haversine(coords[i-1], coords[i], unit="m") for i in range(1, len(coords)))
    total_m = one_lap_m * result["laps"]

    assert 4_000 <= total_m <= 6_000, (
        f"Total distance {total_m:.0f}m out of range "
        f"(1 lap={one_lap_m:.0f}m × {result['laps']})"
    )
