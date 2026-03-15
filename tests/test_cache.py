"""Tests for cache.py — no network, no external dependencies."""

import datetime
import json

import pytest

from parkrun_elevation.cache import Cache


def make_record(event_name, status, days_ago=0):
    """Helper to build a minimal record dict."""
    date = datetime.date.today() - datetime.timedelta(days=days_ago)
    return {
        "event_name": event_name,
        "status": status,
        "date_computed": date.isoformat(),
    }


# ---------------------------------------------------------------------------
# is_known
# ---------------------------------------------------------------------------

def test_is_known_empty_cache(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    assert cache.is_known("winchester") is False


def test_is_known_complete(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    cache.upsert(make_record("winchester", "complete"))
    assert cache.is_known("winchester") is True


def test_is_known_pending_is_never_known(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    cache.upsert(make_record("winchester", "pending"))
    assert cache.is_known("winchester") is False


def test_is_known_no_route_within_30_days(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    cache.upsert(make_record("winchester", "no_route", days_ago=0))
    assert cache.is_known("winchester") is True


def test_is_known_no_route_exactly_30_days(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    cache.upsert(make_record("winchester", "no_route", days_ago=30))
    assert cache.is_known("winchester") is True


def test_is_known_no_route_expired(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    cache.upsert(make_record("winchester", "no_route", days_ago=31))
    assert cache.is_known("winchester") is False


# ---------------------------------------------------------------------------
# get_record
# ---------------------------------------------------------------------------

def test_get_record_missing(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    assert cache.get_record("winchester") is None


def test_get_record_returns_stored(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    record = make_record("winchester", "complete")
    cache.upsert(record)
    assert cache.get_record("winchester") == record


# ---------------------------------------------------------------------------
# upsert — replace and no duplicates
# ---------------------------------------------------------------------------

def test_upsert_replaces_existing(tmp_path):
    cache = Cache(tmp_path / "uk.json")
    cache.upsert(make_record("winchester", "pending"))
    cache.upsert(make_record("winchester", "complete"))

    assert cache.get_record("winchester")["status"] == "complete"


def test_upsert_no_duplicates(tmp_path):
    path = tmp_path / "uk.json"
    cache = Cache(path)
    cache.upsert(make_record("winchester", "pending"))
    cache.upsert(make_record("winchester", "complete"))

    data = json.loads(path.read_text())
    names = [r["event_name"] for r in data["parkruns"]]
    assert names.count("winchester") == 1


# ---------------------------------------------------------------------------
# Atomic write — disk round-trip
# ---------------------------------------------------------------------------

def test_upsert_writes_to_disk(tmp_path):
    path = tmp_path / "uk.json"
    cache = Cache(path)
    record = make_record("winchester", "complete")
    cache.upsert(record)

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["parkruns"][0]["event_name"] == "winchester"


def test_no_tmp_file_left_behind(tmp_path):
    path = tmp_path / "uk.json"
    cache = Cache(path)
    cache.upsert(make_record("winchester", "complete"))

    assert not path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Load from existing file
# ---------------------------------------------------------------------------

def test_load_from_existing_file(tmp_path):
    path = tmp_path / "uk.json"
    existing = {
        "metadata": {},
        "parkruns": [make_record("bushy", "complete")],
    }
    path.write_text(json.dumps(existing))

    cache = Cache(path)
    assert cache.is_known("bushy") is True
    assert cache.get_record("bushy")["status"] == "complete"


def test_load_preserves_all_records(tmp_path):
    path = tmp_path / "uk.json"
    existing = {
        "metadata": {},
        "parkruns": [
            make_record("bushy", "complete"),
            make_record("winchester", "no_route", days_ago=0),
            make_record("shirebrook", "pending"),
        ],
    }
    path.write_text(json.dumps(existing))

    cache = Cache(path)
    assert cache.is_known("bushy") is True
    assert cache.is_known("winchester") is True
    assert cache.is_known("shirebrook") is False


# ---------------------------------------------------------------------------
# Metadata is regenerated on write
# ---------------------------------------------------------------------------

def test_metadata_counts(tmp_path):
    path = tmp_path / "uk.json"
    cache = Cache(path)
    cache.upsert(make_record("bushy", "complete"))
    cache.upsert(make_record("winchester", "complete"))
    cache.upsert(make_record("shirebrook", "no_route"))

    data = json.loads(path.read_text())
    meta = data["metadata"]
    assert meta["total_events"] == 3
    assert meta["events_complete"] == 2
    assert meta["events_no_route"] == 1


def test_metadata_generated_at_is_present(tmp_path):
    path = tmp_path / "uk.json"
    cache = Cache(path)
    cache.upsert(make_record("bushy", "complete"))

    data = json.loads(path.read_text())
    assert "generated_at" in data["metadata"]
    assert "schema_version" in data["metadata"]
