"""Tests for read-only telemetry trace rendering."""

import json
from pathlib import Path

import pytest

from forge.core.trace_viewer import (
    TraceViewerError,
    render_latest_trace,
    render_run_trace,
    render_trace_list,
    resolve_run_dir,
)

RUN_1 = "11111111-1111-4111-8111-111111111111"
RUN_2 = "22222222-2222-4222-8222-222222222222"
NODE_1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
NODE_2 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def test_trace_list_renders_multiple_runs_newest_first(tmp_path: Path) -> None:
    """trace list shows runs newest first and marks the latest."""
    workspace = tmp_path / "ws"
    _write_run(
        workspace,
        RUN_1,
        created_at="2026-01-01T00:00:00+00:00",
        northstar="older goal",
        events=[_event(RUN_1, NODE_1, "work", "producer.response.parsed", status="completed")],
    )
    _write_run(
        workspace,
        RUN_2,
        created_at="2026-01-02T00:00:00+00:00",
        northstar="newer goal",
        events=[
            _event(RUN_2, NODE_2, "plan", "producer.response.parsed", status="completed"),
            _event(RUN_2, NODE_2, "plan", "referee.decision.parsed", status="accept"),
        ],
    )

    text = render_trace_list(workspace)

    newer = text.index(RUN_2[:8])
    older = text.index(RUN_1[:8])
    assert newer < older
    assert "latest 2026-01-02T00:00:00+00:00" in text
    assert "      2" in text
    assert "newer goal" in text


def test_latest_resolves_newest_run(tmp_path: Path) -> None:
    """trace latest renders the newest run by created_at."""
    workspace = tmp_path / "ws"
    _write_run(workspace, RUN_1, created_at="2026-01-01T00:00:00+00:00", northstar="old")
    _write_run(workspace, RUN_2, created_at="2026-01-03T00:00:00+00:00", northstar="latest")

    text = render_latest_trace(workspace)

    assert f"run_id: {RUN_2}" in text
    assert "northstar: latest" in text


def test_run_summary_groups_events_by_node(tmp_path: Path) -> None:
    """Run summary groups PWC events by node."""
    workspace = tmp_path / "ws"
    run_dir = _write_run(
        workspace,
        RUN_1,
        events=[
            _event(
                RUN_1, NODE_1, "plan", "producer.response.parsed", attempt=1, status="completed"
            ),
            _event(RUN_1, NODE_1, "plan", "referee.decision.parsed", attempt=1, status="accept"),
            _event(
                RUN_1, NODE_2, "work", "producer.response.parsed", attempt=1, status="completed"
            ),
            _event(RUN_1, NODE_2, "work", "pwc.exhausted", attempt=1, status="failed"),
        ],
    )

    text = render_run_trace(run_dir)

    assert "nodes:" in text
    assert "aaaaaaaa  agent=plan  status=accept  attempts=1" in text
    assert "bbbbbbbb  agent=work  status=failed  attempts=1" in text
    assert "node aaaaaaaa:" in text
    assert "node bbbbbbbb:" in text


def test_node_prefix_resolution_works_and_renders_fuller_details(tmp_path: Path) -> None:
    """A unique node prefix narrows output and includes revision items."""
    workspace = tmp_path / "ws"
    run_dir = _write_run(
        workspace,
        RUN_1,
        events=[
            _event(
                RUN_1, NODE_1, "work", "producer.response.parsed", attempt=1, status="completed"
            ),
            _revision_event(RUN_1, NODE_1, attempt=1),
            _event(
                RUN_1, NODE_2, "plan", "producer.response.parsed", attempt=1, status="completed"
            ),
        ],
    )

    text = render_run_trace(run_dir, node_prefix="aaaa")

    assert "aaaaaaaa  agent=work" in text
    assert "bbbbbbbb" not in text
    assert "revision appended: items=1" in text
    assert "1. AC1: Add tests." in text


def test_node_prefix_errors_on_ambiguity(tmp_path: Path) -> None:
    """Ambiguous node prefixes raise a clear error."""
    workspace = tmp_path / "ws"
    run_dir = _write_run(
        workspace,
        RUN_1,
        events=[
            _event(
                RUN_1, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "work", "producer.response.parsed"
            ),
            _event(
                RUN_1, "aaaaaaaa-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "work", "producer.response.parsed"
            ),
        ],
    )

    with pytest.raises(TraceViewerError, match="ambiguous node prefix"):
        render_run_trace(run_dir, node_prefix="aaaaaaaa")


def test_missing_telemetry_directory_gives_clear_message(tmp_path: Path) -> None:
    """Missing telemetry directory is reported as a user-facing error."""
    with pytest.raises(TraceViewerError, match="telemetry directory not found"):
        resolve_run_dir(tmp_path / "missing", "abc")


def test_malformed_events_are_reported_without_crashing(tmp_path: Path) -> None:
    """Malformed JSONL events are skipped and counted."""
    workspace = tmp_path / "ws"
    run_dir = _write_run(
        workspace,
        RUN_1,
        events=[_event(RUN_1, NODE_1, "work", "producer.response.parsed", status="completed")],
    )
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json}\n")

    text = render_run_trace(run_dir)

    assert "event_count: 1" in text
    assert "malformed_events_skipped: 1" in text


def _write_run(
    workspace: Path,
    run_id: str,
    *,
    created_at: str = "2026-01-01T00:00:00+00:00",
    northstar: str = "test goal",
    events: list[dict[str, object]] | None = None,
) -> Path:
    run_dir = workspace / "telemetry" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": created_at,
                "metadata": {"workspace": str(workspace), "northstar": northstar},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for event in events or []:
            f.write(json.dumps(event) + "\n")
    return run_dir


def _event(
    run_id: str,
    node_id: str,
    agent_type: str,
    event_type: str,
    *,
    attempt: int | None = None,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "event_id": f"event-{node_id[:4]}-{event_type}",
        "run_id": run_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "node_id": node_id,
        "request_id": node_id,
        "agent_type": agent_type,
        "attempt_number": attempt,
        "role": "producer",
        "phase": "producer",
        "event_type": event_type,
        "status": status,
        "summary": f"{event_type} summary",
        "data": _event_data(event_type, status),
    }


def _event_data(event_type: str, status: str) -> dict[str, object]:
    if event_type == "producer.response.parsed":
        return {
            "status": status,
            "output_type": "DeltaState",
            "delta": {"new_file_paths": ["a.py"]},
        }
    if event_type == "referee.decision.parsed":
        return {"referee_decision": {"disposition": status, "rationale": "looks good"}}
    return {"error": "failed"}


def _revision_event(run_id: str, node_id: str, *, attempt: int) -> dict[str, object]:
    event = _event(
        run_id, node_id, "work", "pwc.revision.appended", attempt=attempt, status="revise"
    )
    event["data"] = {
        "revision_request": {
            "rationale": "missing coverage",
            "prior_attempts": attempt,
            "items": [
                {
                    "criterion_id": "AC1",
                    "required_change": "Add tests.",
                    "rationale": "Coverage required.",
                }
            ],
        }
    }
    return event
