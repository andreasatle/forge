"""Read-only repository for telemetry run data."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class TraceEvent:
    """Parsed telemetry event plus source line number."""

    line_number: int
    data: dict[str, Any]


@dataclass(frozen=True)
class RunTrace:
    """Telemetry run metadata and parsed events."""

    run_dir: Path
    run_json: dict[str, Any]
    events: list[TraceEvent]
    malformed_event_count: int

    @property
    def run_id(self) -> str:
        """Return the run id from metadata, falling back to the directory name."""
        value = self.run_json.get("run_id")
        return value if isinstance(value, str) else self.run_dir.name

    @property
    def created_at(self) -> str:
        """Return the run creation timestamp, falling back to directory mtime."""
        value = self.run_json.get("created_at")
        return value if isinstance(value, str) else _mtime_iso(self.run_dir)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return run metadata as a dictionary."""
        value = self.run_json.get("metadata")
        return cast(dict[str, Any], value) if isinstance(value, dict) else {}

    @property
    def workspace(self) -> str:
        """Return the workspace recorded for this run."""
        value = self.metadata.get("workspace")
        return value if isinstance(value, str) else "unknown"

    @property
    def northstar(self) -> str:
        """Return the run northstar goal."""
        value = self.metadata.get("northstar")
        return " ".join(value.split()) if isinstance(value, str) and value else "unknown"


class TraceViewerError(ValueError):
    """User-facing trace viewer error."""


class TraceRepository:
    """Read-only repository for telemetry run data."""

    def __init__(self, workspace: Path) -> None:
        self._runs_dir = workspace / "telemetry" / "runs"

    def list_runs(self) -> list[RunTrace]:
        """Return all runs sorted newest first."""
        if not self._runs_dir.is_dir():
            raise TraceViewerError(f"telemetry directory not found: {self._runs_dir}")
        runs = [self.load_run(path) for path in self._runs_dir.iterdir() if path.is_dir()]
        return sorted(runs, key=_run_sort_key, reverse=True)

    def latest_run(self) -> RunTrace | None:
        """Return the newest run, or None if the runs directory is empty."""
        runs = self.list_runs()
        return runs[0] if runs else None

    @staticmethod
    def load_run(run_dir: Path) -> RunTrace:
        """Load a run from its directory, tolerating missing or malformed files."""
        run_path = run_dir / "run.json"
        run_json: dict[str, Any] = {}
        if run_path.is_file():
            try:
                value = json.loads(run_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    run_json = cast(dict[str, Any], value)
            except json.JSONDecodeError:
                run_json = {}

        events_path = run_dir / "events.jsonl"
        events: list[TraceEvent] = []
        malformed = 0
        if events_path.is_file():
            for line_number, line in enumerate(
                events_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(parsed, dict):
                    events.append(
                        TraceEvent(line_number=line_number, data=cast(dict[str, Any], parsed))
                    )
                else:
                    malformed += 1
        return RunTrace(
            run_dir=run_dir, run_json=run_json, events=events, malformed_event_count=malformed
        )

    def resolve_run_dir(self, run_id_prefix: str) -> Path:
        """Resolve a run id or unambiguous prefix to a run directory."""
        if not self._runs_dir.is_dir():
            raise TraceViewerError(f"telemetry directory not found: {self._runs_dir}")
        matches = sorted(
            path
            for path in self._runs_dir.iterdir()
            if path.is_dir() and path.name.startswith(run_id_prefix)
        )
        if not matches:
            raise TraceViewerError(f"no telemetry run matches: {run_id_prefix}")
        if len(matches) > 1:
            short = ", ".join(path.name[:8] for path in matches[:8])
            raise TraceViewerError(f"ambiguous run id prefix {run_id_prefix!r}: {short}")
        return matches[0]


def _run_sort_key(run: RunTrace) -> tuple[datetime, float]:
    return (_parse_datetime(run.created_at), run.run_dir.stat().st_mtime)


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
