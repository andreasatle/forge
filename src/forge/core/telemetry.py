"""Append-only run telemetry for framework-owned audit events."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _empty_data() -> dict[str, Any]:
    return {}


class TelemetryEvent(BaseModel, frozen=True):
    """One immutable framework telemetry event."""

    schema_version: Literal[1] = 1
    event_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node_id: UUID | None = None
    request_id: UUID | None = None
    agent_type: str | None = None
    attempt_number: int | None = None
    role: str
    phase: str
    event_type: str
    status: str | None = None
    summary: str | None = None
    data: dict[str, Any] = Field(default_factory=_empty_data)


class TelemetrySink(Protocol):
    """Append-only sink for framework telemetry events."""

    def append(self, event: TelemetryEvent) -> None:
        """Persist one telemetry event."""
        ...


class JsonlTelemetrySink:
    """Append telemetry events as one JSON object per line."""

    def __init__(self, root: Path, run_id: UUID, *, metadata: dict[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.run_dir = root / "runs" / str(run_id)
        self.events_path = self.run_dir / "events.jsonl"
        self.run_path = self.run_dir / "run.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        run_data = {
            "schema_version": 1,
            "run_id": str(run_id),
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        self.run_path.write_text(json.dumps(run_data, indent=2) + "\n")
        self.events_path.touch(exist_ok=True)

    def append(self, event: TelemetryEvent) -> None:
        """Append a JSONL event without touching scheduler state."""
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")


def safe_append_telemetry(sink: TelemetrySink | None, event: TelemetryEvent) -> None:
    """Append telemetry best-effort; telemetry failure must not affect runtime behavior."""
    if sink is None:
        return
    try:
        sink.append(event)
    except Exception:
        logger.exception("telemetry append failed")
