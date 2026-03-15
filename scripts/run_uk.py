#!/usr/bin/env python
"""
Run the UK parkrun elevation pipeline.

Examples
--------
# Normal run (uses cached events.json and Overpass data):
    python scripts/run_uk.py

# First run, or force a re-fetch of source data:
    python scripts/run_uk.py --refresh-events --refresh-overpass

# Re-compute a single event regardless of cache:
    python scripts/run_uk.py --force winchester

# Test run — process 10 events, write nothing to disk:
    python scripts/run_uk.py --limit 10 --dry-run

# Verbose logging:
    python scripts/run_uk.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parkrun_elevation import pipeline


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the UK parkrun elevation dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        metavar="EVENT_NAME",
        help="Recompute this event even if it already has a complete record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Stop after processing N events (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute elevation but do not write anything to disk.",
    )
    parser.add_argument(
        "--refresh-events",
        action="store_true",
        help="Re-fetch events.json even if a cached copy exists.",
    )
    parser.add_argument(
        "--refresh-overpass",
        action="store_true",
        help="Re-fetch Overpass OSM data even if a cached copy exists.",
    )
    parser.add_argument(
        "--srtm-dir",
        type=Path,
        default=Path("srtm"),
        metavar="PATH",
        help="Local SRTM tile cache directory (default: srtm/).",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/elevation/uk.json"),
        metavar="PATH",
        help="Output JSON file path (default: data/elevation/uk.json).",
    )
    parser.add_argument(
        "--retry-no-route",
        action="store_true",
        help="Reprocess events whose cached status is 'no_route'.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline.run(
        srtm_dir=args.srtm_dir,
        cache_path=args.cache_path,
        refresh_events=args.refresh_events,
        refresh_overpass=args.refresh_overpass,
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
        show_progress=True,
        retry_no_route=args.retry_no_route,
    )


if __name__ == "__main__":
    main()
