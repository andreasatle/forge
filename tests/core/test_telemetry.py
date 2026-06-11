"""Tests for immutable run telemetry persistence and isolation."""

import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from forge.core.models import SchedulerState
from forge.core.persistence import save_run
from forge.core.state_service import StateService
from forge.core.telemetry import JsonlTelemetrySink, TelemetryEvent
from forge.core.workspace import Workspace
from forge.tools.builtin import build_read_registry
from forge.tools.schemas import ListFilesRequest, ListFilesResponse


def test_jsonl_telemetry_sink_writes_run_metadata_and_valid_events(tmp_path: Path) -> None:
    """JsonlTelemetrySink creates the run layout and appends one JSON object per line."""
    run_id = uuid4()
    sink = JsonlTelemetrySink(tmp_path / "telemetry", run_id, metadata={"northstar": "test"})

    event = TelemetryEvent(
        run_id=run_id,
        role="producer",
        phase="pwc",
        event_type="pwc.attempt.started",
        status="started",
        data={"attempt": 1},
    )
    sink.append(event)

    run_dir = tmp_path / "telemetry" / "runs" / str(run_id)
    assert (run_dir / "run.json").exists()
    assert (run_dir / "events.jsonl").exists()
    run_data = json.loads((run_dir / "run.json").read_text())
    assert run_data["run_id"] == str(run_id)
    assert run_data["metadata"]["northstar"] == "test"
    lines = (run_dir / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "pwc.attempt.started"
    assert parsed["run_id"] == str(run_id)


def test_state_json_does_not_embed_telemetry_history(tmp_path: Path) -> None:
    """Scheduler state remains separate from telemetry event history."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    run_id = uuid4()
    sink = JsonlTelemetrySink(ws.telemetry_dir(), run_id)
    sink.append(
        TelemetryEvent(run_id=run_id, role="scheduler", phase="scheduler", event_type="node.failed")
    )

    save_run(SchedulerState(northstar="test goal"), ws)

    state_data = json.loads(ws.state_path().read_text())
    assert "telemetry" not in state_data
    assert "events" not in state_data
    assert (ws.telemetry_dir() / "runs" / str(run_id) / "events.jsonl").exists()


def test_telemetry_is_not_exposed_through_state_view(tmp_path: Path) -> None:
    """StateView is built from artifact files only, not workspace telemetry."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    (ws.artifact_dir("codebase") / "main.py").write_text("x = 1")
    run_id = uuid4()
    sink = JsonlTelemetrySink(ws.telemetry_dir(), run_id)
    sink.append(
        TelemetryEvent(run_id=run_id, role="scheduler", phase="scheduler", event_type="node.failed")
    )

    view = StateService(ws, "codebase").build_state_view()

    assert [file.path for file in view.files] == ["main.py"]
    assert "telemetry" not in StateService(ws, "codebase").build_state_view().model_dump()


async def test_telemetry_is_not_exposed_through_read_tools(tmp_path: Path) -> None:
    """Read tools are scoped to the artifact directory and cannot list telemetry."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    (ws.artifact_dir("codebase") / "main.py").write_text("x = 1")
    JsonlTelemetrySink(ws.telemetry_dir(), uuid4())
    registry = build_read_registry(ws, "codebase")
    list_files = registry.get("list_files")

    response = cast(ListFilesResponse, await list_files.fn(ListFilesRequest(directory="")))

    assert response.paths == ["main.py"]
