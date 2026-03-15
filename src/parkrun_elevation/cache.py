"""
Idempotency cache for the elevation pipeline.

Loads the existing output JSON on startup and tracks which events have already
been processed. Every upsert is written atomically to disk via a temp file +
os.replace(), so an interrupted run cannot corrupt the output file.

Status values:
    "complete"  — elevation computed; skip on next run
    "no_route"  — no route found; retry after 30 days
    "pending"   — placeholder written but not computed (crash recovery); always retry
"""

import datetime
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"


class Cache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[str, dict] = {}

        if path.exists():
            data = json.loads(path.read_text())
            for record in data.get("parkruns", []):
                self._records[record["event_name"]] = record
            logger.info(
                "Loaded %d records from %s", len(self._records), path
            )
        else:
            logger.info("No existing cache at %s — starting fresh", path)

    def is_known(self, event_name: str) -> bool:
        """Return True if this event should be skipped on the current run.

        - "complete"  → always skip
        - "no_route"  → skip if computed within the last 30 days
        - "pending"   → never skip (treat as crashed, retry)
        - missing     → never skip
        """
        record = self._records.get(event_name)
        if record is None:
            return False
        if record["status"] == "complete":
            return True
        if record["status"] == "no_route":
            date_computed = datetime.date.fromisoformat(record["date_computed"])
            return (datetime.date.today() - date_computed).days <= 30
        # "pending" — always retry
        return False

    def get_record(self, event_name: str) -> dict | None:
        """Return the stored record for event_name, or None if not present."""
        return self._records.get(event_name)

    def upsert(self, record: dict) -> None:
        """Add or replace a record, then write the output file atomically."""
        event_name = record["event_name"]
        self._records[event_name] = record
        self._write()

    def _write(self) -> None:
        """Serialise the full output JSON to disk via temp file + os.replace()."""
        records = list(self._records.values())

        n_complete = sum(1 for r in records if r.get("status") == "complete")
        n_no_route = sum(1 for r in records if r.get("status") == "no_route")

        output = {
            "metadata": {
                "schema_version": _SCHEMA_VERSION,
                "generated_at": datetime.datetime.now(datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_events": len(records),
                "events_complete": n_complete,
                "events_no_route": n_no_route,
            },
            "parkruns": records,
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(output, indent=2))
        os.replace(tmp_path, self._path)
        logger.debug("Wrote %d records to %s", len(records), self._path)
