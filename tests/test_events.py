"""
Tests for events.py

Run the fast tests (no network) with:
    python -m pytest tests/test_events.py -v

Run including the live smoke test:
    python -m pytest tests/test_events.py -v -m live
"""

import json
import pytest

from parkrun_elevation.events import parse_events, fetch_events, get_uk_events


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

RAW_FIXTURE = {
    "events": {
        "type": "FeatureCollection",
        "features": [
            {   # Should be included: UK adult
                "id": 280,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-1.310849, 51.069286]},
                "properties": {
                    "eventname": "winchester",
                    "EventLongName": "Winchester parkrun",
                    "countrycode": 97,
                    "seriesid": 1,
                },
            },
            {   # Excluded: junior parkrun (seriesid == 2)
                "id": 999,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-1.3, 51.0]},
                "properties": {
                    "eventname": "winchester-junior",
                    "EventLongName": "Winchester Junior parkrun",
                    "countrycode": 97,
                    "seriesid": 2,
                },
            },
            {   # Excluded: not UK (countrycode != 97)
                "id": 888,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [4.9, 52.3]},
                "properties": {
                    "eventname": "amsterdam",
                    "EventLongName": "Amsterdam parkrun",
                    "countrycode": 74,
                    "seriesid": 1,
                },
            },
        ],
    }
}


# ---------------------------------------------------------------------------
# parse_events — pure function, no I/O
# ---------------------------------------------------------------------------

def test_parse_events_filters_correctly():
    events = parse_events(RAW_FIXTURE)
    assert len(events) == 1
    assert events[0]["event_name"] == "winchester"


def test_parse_events_output_fields():
    events = parse_events(RAW_FIXTURE)
    e = events[0]
    assert set(e.keys()) == {"event_name", "event_long_name", "lat", "lng", "country_code"}


def test_parse_events_coordinate_order():
    """GeoJSON stores [lng, lat]; verify we unpack correctly.

    Winchester is in Hampshire: lat ~51, lng ~-1.3.
    If coordinates were swapped we'd get lat=-1.3, lng=51 — both assertions
    below would fail, catching the footgun early.
    """
    events = parse_events(RAW_FIXTURE)
    e = events[0]
    assert 49 < e["lat"] < 61, f"lat looks wrong (got {e['lat']}); coordinates may be swapped"
    assert -8 < e["lng"] < 2, f"lng looks wrong (got {e['lng']}); coordinates may be swapped"


def test_parse_events_exact_coordinates():
    events = parse_events(RAW_FIXTURE)
    assert events[0]["lat"] == pytest.approx(51.069286)
    assert events[0]["lng"] == pytest.approx(-1.310849)


def test_parse_events_excludes_junior():
    events = parse_events(RAW_FIXTURE)
    names = [e["event_name"] for e in events]
    assert "winchester-junior" not in names


def test_parse_events_excludes_non_uk():
    events = parse_events(RAW_FIXTURE)
    names = [e["event_name"] for e in events]
    assert "amsterdam" not in names


def test_parse_events_empty_features():
    raw = {"events": {"type": "FeatureCollection", "features": []}}
    assert parse_events(raw) == []


def test_parse_events_excludes_overseas_territories():
    """Falkland Islands is countrycode=97 but outside UK lat/lng bounds."""
    falklands = {
        "events": {
            "type": "FeatureCollection",
            "features": [{
                "id": 1,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-57.85, -51.68]},
                "properties": {
                    "eventname": "capepembrokelighthouse",
                    "EventLongName": "Cape Pembroke Lighthouse parkrun",
                    "countrycode": 97,
                    "seriesid": 1,
                },
            }],
        }
    }
    from parkrun_elevation.events import UK_LAT_MIN, UK_LAT_MAX, UK_LNG_MIN, UK_LNG_MAX
    assert parse_events(
        falklands,
        lat_bounds=(UK_LAT_MIN, UK_LAT_MAX),
        lng_bounds=(UK_LNG_MIN, UK_LNG_MAX),
    ) == []


def test_parse_events_country_code_passthrough():
    """Filtering by a different country code returns only matching events."""
    events = parse_events(RAW_FIXTURE, country_code=74)
    assert len(events) == 1
    assert events[0]["event_name"] == "amsterdam"


# ---------------------------------------------------------------------------
# fetch_events — cache behaviour (no network)
# ---------------------------------------------------------------------------

def test_fetch_events_reads_cache(tmp_path, monkeypatch):
    """When the cache file exists and refresh=False, no HTTP request is made."""
    import parkrun_elevation.events as ev_module

    cache_file = tmp_path / "events.json"
    cache_file.write_text(json.dumps(RAW_FIXTURE))
    monkeypatch.setattr(ev_module, "EVENTS_CACHE_PATH", cache_file)

    # If httpx.get were called it would raise (no real network in unit tests);
    # the fact that we reach the assertion proves it wasn't called.
    result = fetch_events(refresh=False)
    assert result == RAW_FIXTURE


def test_fetch_events_writes_cache(tmp_path, monkeypatch, respx_mock):
    """On a cache miss, the downloaded data is written to disk."""
    import parkrun_elevation.events as ev_module

    cache_file = tmp_path / "events.json"
    monkeypatch.setattr(ev_module, "EVENTS_CACHE_PATH", cache_file)

    import respx, httpx
    respx_mock.get(ev_module.EVENTS_URL).mock(
        return_value=httpx.Response(200, json=RAW_FIXTURE)
    )

    fetch_events(refresh=False)  # cache miss — file doesn't exist yet

    assert cache_file.exists()
    saved = json.loads(cache_file.read_text())
    assert saved == RAW_FIXTURE


# ---------------------------------------------------------------------------
# Live smoke test — hits the real network, skipped by default
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_get_uk_events(tmp_path, monkeypatch):
    """Fetch the real events.json and check basic sanity of the result.

    Run with: pytest -m live
    """
    import parkrun_elevation.events as ev_module
    monkeypatch.setattr(ev_module, "EVENTS_CACHE_PATH", tmp_path / "events.json")

    events = get_uk_events(refresh=True)

    assert len(events) > 800, f"Expected >800 UK events, got {len(events)}"

    for e in events:
        assert "event_name" in e
        assert "lat" in e and "lng" in e
        assert 49 <= e["lat"] <= 61, f"{e['event_name']}: lat {e['lat']} out of UK range"
        assert -8.5 <= e["lng"] <= 2, f"{e['event_name']}: lng {e['lng']} out of UK range"
