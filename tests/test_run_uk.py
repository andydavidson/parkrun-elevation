"""Tests for scripts/run_uk.py — argument parsing and run() wiring."""

from pathlib import Path
from unittest.mock import call, patch

import pytest

from scripts.run_uk import main, parse_args


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_defaults():
    args = parse_args([])
    assert args.force is None
    assert args.limit is None
    assert args.dry_run is False
    assert args.refresh_events is False
    assert args.refresh_overpass is False
    assert args.retry_no_route is False
    assert args.srtm_dir == Path("srtm")
    assert args.cache_path == Path("data/elevation/uk.json")
    assert args.verbose is False


def test_retry_no_route():
    args = parse_args(["--retry-no-route"])
    assert args.retry_no_route is True


def test_force():
    args = parse_args(["--force", "winchester"])
    assert args.force == "winchester"


def test_limit():
    args = parse_args(["--limit", "5"])
    assert args.limit == 5


def test_dry_run():
    args = parse_args(["--dry-run"])
    assert args.dry_run is True


def test_refresh_flags():
    args = parse_args(["--refresh-events", "--refresh-overpass"])
    assert args.refresh_events is True
    assert args.refresh_overpass is True


def test_srtm_dir():
    args = parse_args(["--srtm-dir", "/data/srtm"])
    assert args.srtm_dir == Path("/data/srtm")


def test_cache_path():
    args = parse_args(["--cache-path", "/tmp/uk.json"])
    assert args.cache_path == Path("/tmp/uk.json")


def test_verbose_short():
    args = parse_args(["-v"])
    assert args.verbose is True


def test_verbose_long():
    args = parse_args(["--verbose"])
    assert args.verbose is True


# ---------------------------------------------------------------------------
# main() wires args → pipeline.run() correctly
# ---------------------------------------------------------------------------

@patch("scripts.run_uk.pipeline.run")
def test_main_default_call(mock_run):
    main([])
    mock_run.assert_called_once_with(
        srtm_dir=Path("srtm"),
        cache_path=Path("data/elevation/uk.json"),
        refresh_events=False,
        refresh_overpass=False,
        force=None,
        limit=None,
        dry_run=False,
        show_progress=True,
        retry_no_route=False,
    )


@patch("scripts.run_uk.pipeline.run")
def test_main_passes_all_flags(mock_run):
    main([
        "--force", "lingwood",
        "--limit", "3",
        "--dry-run",
        "--refresh-events",
        "--refresh-overpass",
        "--srtm-dir", "/tmp/srtm",
        "--cache-path", "/tmp/uk.json",
    ])
    mock_run.assert_called_once_with(
        srtm_dir=Path("/tmp/srtm"),
        cache_path=Path("/tmp/uk.json"),
        refresh_events=True,
        refresh_overpass=True,
        force="lingwood",
        limit=3,
        dry_run=True,
        show_progress=True,
        retry_no_route=False,
    )
