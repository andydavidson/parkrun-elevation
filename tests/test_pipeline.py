"""
Tests for pipeline.py

All tests are offline. compute_elevation_profile is monkeypatched throughout
so no SRTM tiles are needed.
"""

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from parkrun_elevation.cache import Cache
from parkrun_elevation.pipeline import process_event, run


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
    "total_ascent_m": 22.5,
    "total_descent_m": 24.1,
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

OSM_RESULT = {
    "relation": {"id": 12345678, "tags": {"name": "Lingwood Parkrun"}},
    "coords": [(52.619, 1.494), (52.622, 1.498), (52.625, 1.502)],
}


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "uk.json")


@pytest.fixture
def patched_elevation():
    """Monkeypatch compute_elevation_profile to return FAKE_ELEVATION."""
    with patch(
        "parkrun_elevation.pipeline.compute_elevation_profile",
        return_value=FAKE_ELEVATION,
    ) as mock:
        yield mock


# ---------------------------------------------------------------------------
# process_event
# ---------------------------------------------------------------------------

def test_process_event_no_route_returns_status(tmp_path, cache):
    status = process_event(LINGWOOD_EVENT, None, tmp_path, cache)
    assert status == "no_route"


def test_process_event_no_route_writes_record(tmp_path, cache):
    process_event(LINGWOOD_EVENT, None, tmp_path, cache)
    record = cache.get_record("lingwood")
    assert record is not None
    assert record["status"] == "no_route"


def test_process_event_complete_returns_status(tmp_path, cache, patched_elevation):
    status = process_event(LINGWOOD_EVENT, OSM_RESULT, tmp_path, cache)
    assert status == "complete"


def test_process_event_complete_writes_record(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, OSM_RESULT, tmp_path, cache)
    record = cache.get_record("lingwood")
    assert record["status"] == "complete"
    assert record["route_source_type"] == "osm"
    assert record["route_source_id"] == "relation/12345678"
    assert "total_ascent_m" in record


def test_process_event_complete_record_has_all_elevation_fields(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, OSM_RESULT, tmp_path, cache)
    record = cache.get_record("lingwood")
    for field in FAKE_ELEVATION:
        assert field in record, f"Missing field: {field}"


def test_process_event_complete_record_has_provenance(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, OSM_RESULT, tmp_path, cache)
    record = cache.get_record("lingwood")
    assert record["route_source_url"] == "https://www.openstreetmap.org/relation/12345678"
    assert record["date_computed"] == datetime.date.today().isoformat()


def test_process_event_dry_run_no_route_does_not_write(tmp_path, cache):
    process_event(LINGWOOD_EVENT, None, tmp_path, cache, dry_run=True)
    assert cache.get_record("lingwood") is None


def test_process_event_dry_run_complete_does_not_write(tmp_path, cache, patched_elevation):
    process_event(LINGWOOD_EVENT, OSM_RESULT, tmp_path, cache, dry_run=True)
    assert cache.get_record("lingwood") is None


# ---------------------------------------------------------------------------
# run — helpers to build fake events / overpass data
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


def _empty_overpass():
    return {"elements": []}


@pytest.fixture
def patched_run_deps(tmp_path, monkeypatch):
    """Patch all external I/O in run() so tests are fully offline."""
    events = _make_events(["alpha", "bravo", "charlie", "delta", "echo"])

    monkeypatch.setattr("parkrun_elevation.pipeline.get_uk_events", lambda **_: events)
    monkeypatch.setattr("parkrun_elevation.pipeline.fetch_overpass", lambda **_: _empty_overpass())
    monkeypatch.setattr("parkrun_elevation.pipeline.build_node_lookup", lambda e: {})
    monkeypatch.setattr("parkrun_elevation.pipeline.build_way_lookup", lambda e: {})
    monkeypatch.setattr("parkrun_elevation.pipeline.extract_relations", lambda e: [])
    # No OSM match for any event (empty relations list)
    monkeypatch.setattr("parkrun_elevation.pipeline.match_relation", lambda *a, **k: None)

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


def test_run_all_no_route_when_no_osm(patched_run_deps):
    d = patched_run_deps
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"])
    cache = Cache(d["cache_path"])
    for event in d["events"]:
        assert cache.get_record(event["event_name"])["status"] == "no_route"


def test_run_skips_known_events(patched_run_deps, monkeypatch):
    d = patched_run_deps
    # Pre-populate cache with "alpha" as complete
    cache = Cache(d["cache_path"])
    cache.upsert({
        "event_name": "alpha",
        "event_long_name": "Alpha parkrun",
        "country_code": 97,
        "lat": 52.0,
        "lng": 1.0,
        "status": "complete",
        "date_computed": datetime.date.today().isoformat(),
    })

    processed = []
    # Capture the real function before patching to avoid recursion
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, osm_result, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, osm_result, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)

    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"])

    assert "alpha" not in processed
    assert "bravo" in processed


def test_run_force_reprocesses_known_event(patched_run_deps, monkeypatch):
    d = patched_run_deps
    cache = Cache(d["cache_path"])
    cache.upsert({
        "event_name": "alpha",
        "event_long_name": "Alpha parkrun",
        "country_code": 97,
        "lat": 52.0,
        "lng": 1.0,
        "status": "complete",
        "date_computed": datetime.date.today().isoformat(),
    })

    processed = []
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, osm_result, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, osm_result, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)

    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], force="alpha")

    assert "alpha" in processed


def test_run_limit(patched_run_deps, monkeypatch):
    d = patched_run_deps
    processed = []
    from parkrun_elevation.pipeline import process_event as _real_process

    def tracking_process(event, osm_result, srtm_dir, cache, dry_run=False):
        processed.append(event["event_name"])
        return _real_process(event, osm_result, srtm_dir, cache, dry_run=dry_run)

    monkeypatch.setattr("parkrun_elevation.pipeline.process_event", tracking_process)

    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], limit=3)

    assert len(processed) == 3


def test_run_dry_run_does_not_write(patched_run_deps):
    d = patched_run_deps
    run(srtm_dir=d["srtm_dir"], cache_path=d["cache_path"], dry_run=True)
    # Cache file should not exist since nothing was written
    assert not d["cache_path"].exists()


def test_run_pending_written_before_elevation(patched_run_deps, monkeypatch):
    """If elevation raises mid-flight, the 'pending' record should be on disk."""
    d = patched_run_deps

    pending_seen = []

    def crashing_process(event, osm_result, srtm_dir, cache, dry_run=False):
        # Check what's in the cache file at the moment of the crash
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
