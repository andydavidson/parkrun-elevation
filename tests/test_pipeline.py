"""
Tests for pipeline.py

All tests are offline. compute_elevation_profile is monkeypatched throughout
so no SRTM tiles are needed.
"""

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from parkrun_elevation.cache import Cache
from parkrun_elevation.pipeline import _apply_laps, process_event, run


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_ELEVATION = {
    "distance_m": 5000.0,
    "elevation_source": "srtm30m",
    "smoothing_window_m": 70,
    "elevation_threshold_m": 3.0,
    "max_gradient_window_m": 100,
    "avg_abs_gradient_pct": 1.2,
    "max_gradient_pct": 3.8,
    "total_ascent_m": 20.0,
    "total_descent_m": 20.0,
    "elevation_high_m": 45.0,
    "elevation_low_m": 18.0,
}

LINGWOOD_EVENT = {
    "event_name": "lingwood",
    "event_long_name": "Lingwood parkrun",
    "country_code": 97,
    "lat": 52.619,
    "lng": 1.494,
}

OSM_ROUTE = {
    "coords": [(52.619, 1.494), (52.622, 1.498), (52.625, 1.502)],
    "laps": 1,
    "route_type": "loop",
    "laps_inferred": False,
    "source_type": "osm",
    "source_id": "relation/12345678",
    "source_url": "https://www.openstreetmap.org/relation/12345678",
}

KML_ROUTE_TWO_LAP = {
    "coords": [(52.619, 1.494), (52.622, 1.498), (52.625, 1.502)],
    "laps": 2,
    "route_type": "loop",
    "laps_inferred": False,
    "source_type": "parkrun_kml",
    "source_id": "googlemymaps/TEST123",
    "source_url": "https://www.parkrun.org.uk/lingwood/course/",
}

KML_ROUTE_OAB = {**KML_ROUTE_TWO_LAP, "route_type": "out_and_back"}


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "uk.json")


@pytest.fixture
def patched_elevation():
    with patch(
        "parkrun_elevation.pipeline.compute_elevation_profile",
        return_value=FAKE_ELEVATION,
    ) as mock:
        yield mock


# ---------------------------------------------------------------------------
# _apply_laps
# ---------------------------------------------------------------------------

def test_apply_laps_single_lap_unchanged():
    data = {"total_ascent_m": 20.0, "total_descent_m": 15.0, "distance_m": 5000.0}
    result = _apply_laps(data, "loop", 1)
    assert result["total_ascent_m"] == 20.0
    assert result["distance_m"] == 5000.0


def test_apply_laps_loop_multiplies():
    data = {"total_ascent_m": 20.0, "total_descent_m": 15.0, "distance_m": 2500.0}
    result = _apply_laps(data, "loop", 2)
    assert result["total_ascent_m"] == pytest.approx(40.0)
    assert result["total_descent_m"] == pytest.approx(30.0)
    assert result["distance_m"] == pytest.approx(5000.0)


def test_apply_laps_out_and_back_sums():
    """Out-and-back: return leg reverses ascent/descent, so total = ascent + descent."""
    data = {"total_ascent_m": 30.0, "total_descent_m": 10.0, "distance_m": 2500.0}
    result = _apply_laps(data, "out_and_back", 2)
    assert result["total_ascent_m"] == pytest.approx(40.0)
    assert result["total_descent_m"] == pytest.approx(40.0)


def test_apply_laps_unknown_treats_as_loop():
    data = {"total_ascent_m": 10.0, "total_descent_m": 10.0, "distance_m": 1667.0}
    result = _apply_laps(data, "unknown", 3)
    assert result["total_ascent_m"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# process_event
# ---------------------------------------------------------------------------

def test_process_event_no_route_returns_status(tmp_path, cache):
    assert process_event(LINGWOOD_EVENT, None, tmp_path, cache) == "no_route"


def test_process_event_no_route_writes_record(tmp_path, cache):
    process_event(LINGWOOD_EVENT, None, tmp_path, cache)
    assert cache.get_record("lingwood")["status"] == "no_route"


def test_process_event_osm_complete(tmp_path, cache, patched_elevation):
    status = process_event(LINGWOOD_EVENT, OSM_ROUTE, tmp_path, cache)
    assert status == "complete"
    record = cache.get_record("lingwood")
    assert record["route_source_type"] == "osm"
    assert record["route_source_id"] == "relation/12345678"
    assert record["laps"] == 1


def test_process_event_kml_two_laps_multiplies(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, KML_ROUTE_TWO_LAP, tmp_path, cache)
    record = cache.get_record("lingwood")
    assert record["route_source_type"] == "parkrun_kml"
    assert record["laps"] == 2
    # FAKE_ELEVATION has 20.0 ascent — should be doubled
    assert record["total_ascent_m"] == pytest.approx(40.0)
    assert record["distance_m"] == pytest.approx(10000.0)


def test_process_event_kml_out_and_back(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, KML_ROUTE_OAB, tmp_path, cache)
    record = cache.get_record("lingwood")
    assert record["route_type"] == "out_and_back"
    # ascent(20) + descent(20) = 40 each way
    assert record["total_ascent_m"] == pytest.approx(40.0)
    assert record["total_descent_m"] == pytest.approx(40.0)


def test_process_event_laps_inferred_flagged(tmp_path, cache, patched_elevation):
    inferred_route = {**KML_ROUTE_TWO_LAP, "laps_inferred": True, "route_type": "unknown"}
    process_event(LINGWOOD_EVENT, inferred_route, tmp_path, cache)
    assert cache.get_record("lingwood").get("laps_inferred") is True


def test_process_event_all_elevation_fields_present(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, OSM_ROUTE, tmp_path, cache)
    record = cache.get_record("lingwood")
    for field in FAKE_ELEVATION:
        assert field in record, f"Missing field: {field}"


def test_process_event_provenance_fields(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, OSM_ROUTE, tmp_path, cache)
    record = cache.get_record("lingwood")
    assert record["route_source_url"] == "https://www.openstreetmap.org/relation/12345678"
    assert record["date_computed"] == datetime.date.today().isoformat()


def test_process_event_dry_run_no_route_does_not_write(tmp_path, cache):
    process_event(LINGWOOD_EVENT, None, tmp_path, cache, dry_run=True)
    assert cache.get_record("lingwood") is None


def test_process_event_dry_run_complete_does_not_write(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, OSM_ROUTE, tmp_path, cache, dry_run=True)
    assert cache.get_record("lingwood") is None


# ---------------------------------------------------------------------------
# run — helpers
# ---------------------------------------------------------------------------

def _make_events(names):
    return [
        {
            "event_name": n,
            "event_long_name": f"{n.title()} parkrun",
            "country_code": 97,
            "lat": 52.0 + i * 0.01,
            "lng": 1.0 + i * 0.01,
        }
        for i, n in enumerate(names)
    ]


@pytest.fixture
def patched_run_deps(tmp_path, monkeypatch):
    """Patch all external I/O in run() so tests are fully offline."""
    events = _make_events(["alpha", "bravo", "charlie", "delta", "echo"])

    monkeypatch.setattr("parkrun_elevation.pipeline.get_uk_events", lambda **_: events)
    monkeypatch.setattr("parkrun_elevation.pipeline.fetch_overpass", lambda **_: {"elements": []})
    monkeypatch.setattr("parkrun_elevation.pipeline.build_node_lookup", lambda e: {})
    monkeypatch.setattr("parkrun_elevation.pipeline.build_way_lookup", lambda e: {})
    monkeypatch.setattr("parkrun_elevation.pipeline.extract_relations", lambda e: [])
    monkeypatch.setattr("parkrun_elevation.pipeline.match_relation", lambda *a, **k: None)
    monkeypatch.setattr("parkrun_elevation.pipeline.get_kml_route", lambda *a, **k: None)

    return {
        "events": events,
        "cache_path": tmp_path / "uk.json",
        "srtm_dir": tmp_path / "srtm",
    }


# ---------------------------------------------------------------------------
# run — idempotency and control flags
# ---------------------------------------------------------------------------

def test_run_processes_all_events(patched_run_deps):
    d = patched_run_deps
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"])
    cache = Cache(d["cache_path"])
    for event in d["events"]:
        assert cache.get_record(event["event_name"]) is not None


def test_run_all_no_route_when_no_sources(patched_run_deps):
    d = patched_run_deps
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"])
    cache = Cache(d["cache_path"])
    for event in d["events"]:
        assert cache.get_record(event["event_name"])["status"] == "no_route"


def test_run_skips_known_events(patched_run_deps, monkeypatch):
    d = patched_run_deps
    cache = Cache(d["cache_path"])
    cache.upsert({
        "event_name": "alpha", "event_long_name": "Alpha parkrun",
        "country_code": 97, "lat": 52.0, "lng": 1.0,
        "status": "complete", "date_computed": datetime.date.today().isoformat(),
    })

    processed = []
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, route, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, route, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"])

    assert "alpha" not in processed
    assert "bravo" in processed


def test_run_force_reprocesses_known_event(patched_run_deps, monkeypatch):
    d = patched_run_deps
    cache = Cache(d["cache_path"])
    cache.upsert({
        "event_name": "alpha", "event_long_name": "Alpha parkrun",
        "country_code": 97, "lat": 52.0, "lng": 1.0,
        "status": "complete", "date_computed": datetime.date.today().isoformat(),
    })

    processed = []
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, route, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, route, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], force="alpha")
    assert "alpha" in processed


def test_run_limit(patched_run_deps, monkeypatch):
    d = patched_run_deps
    processed = []
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, route, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, route, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], limit=3)
    assert len(processed) == 3


def test_run_dry_run_does_not_write(patched_run_deps):
    d = patched_run_deps
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], dry_run=True)
    assert not d["cache_path"].exists()


def test_run_retries_no_route_when_flag_set(patched_run_deps, monkeypatch):
    d = patched_run_deps
    cache = Cache(d["cache_path"])
    cache.upsert({
        "event_name": "alpha", "event_long_name": "Alpha parkrun",
        "country_code": 97, "lat": 52.0, "lng": 1.0,
        "status": "no_route", "date_computed": datetime.date.today().isoformat(),
    })

    processed = []
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, route, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, route, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], retry_no_route=True)

    assert "alpha" in processed


def test_run_does_not_retry_no_route_by_default(patched_run_deps, monkeypatch):
    d = patched_run_deps
    cache = Cache(d["cache_path"])
    cache.upsert({
        "event_name": "alpha", "event_long_name": "Alpha parkrun",
        "country_code": 97, "lat": 52.0, "lng": 1.0,
        "status": "no_route", "date_computed": datetime.date.today().isoformat(),
    })

    processed = []
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, route, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, route, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"])

    assert "alpha" not in processed


def test_run_pending_written_before_elevation(patched_run_deps, monkeypatch):
    d = patched_run_deps
    pending_seen = []

    def crashing_process(event, route, srtm_dir, cache, dry_run=False):
        if d["cache_path"].exists():
            data = json.loads(d["cache_path"].read_text())
            statuses = {r["event_name"]: r["status"] for r in data["parkruns"]}
            if event["event_name"] in statuses:
                pending_seen.append(statuses[event["event_name"]])
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", crashing_process)

    with pytest.raises(RuntimeError):
        run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], limit=1)

    assert "pending" in pending_seen
