"""Direct tests for TextTraceRenderer — text presentation of RunTrace objects."""

from pathlib import Path

import pytest

from forge.core.text_trace_renderer import TextTraceRenderer
from forge.core.trace_repository import RunTrace, TraceEvent, TraceViewerError

RUN_1 = "11111111-1111-4111-8111-111111111111"
RUN_2 = "22222222-2222-4222-8222-222222222222"
NODE_1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
NODE_2 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = RUN_1,
    *,
    created_at: str = "2026-01-01T00:00:00+00:00",
    northstar: str = "test goal",
    workspace: str = "/workspace",
    events: list[dict[str, object]] | None = None,
    malformed: int = 0,
) -> RunTrace:
    run_json = {
        "run_id": run_id,
        "created_at": created_at,
        "metadata": {"workspace": workspace, "northstar": northstar},
    }
    parsed_events = [
        TraceEvent(line_number=i + 1, data=e)  # type: ignore[arg-type]
        for i, e in enumerate(events or [])
    ]
    return RunTrace(
        run_dir=Path(f"/fake/telemetry/runs/{run_id}"),
        run_json=run_json,
        events=parsed_events,
        malformed_event_count=malformed,
    )


def _event(
    node_id: str,
    agent_type: str,
    event_type: str,
    *,
    attempt: int | None = None,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "run_id": RUN_1,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "node_id": node_id,
        "agent_type": agent_type,
        "attempt_number": attempt,
        "event_type": event_type,
        "status": status,
        "summary": f"{event_type} summary",
        "data": _event_data(event_type, status),
    }


def _event_data(event_type: str, status: str) -> dict[str, object]:
    if event_type == "producer.response.parsed":
        return {
            "status": status,
            "output_type": "WorkOutput",
            "work_output": {"file_paths": ["a.py"]},
        }
    if event_type == "referee.decision.parsed":
        return {"referee_decision": {"disposition": status, "rationale": "looks good"}}
    return {"error": "some error"}


def _revision_event(node_id: str, *, attempt: int) -> dict[str, object]:
    event = _event(node_id, "work", "pwc.revision.appended", attempt=attempt, status="revise")
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


# ---------------------------------------------------------------------------
# render_list
# ---------------------------------------------------------------------------


def test_render_list_shows_header_and_rows() -> None:
    """render_list produces a header line followed by one row per run."""
    run = _make_run(RUN_1, created_at="2026-03-01T10:00:00+00:00", northstar="do the thing")
    text = TextTraceRenderer().render_list([run])
    assert "created_at" in text
    assert "run_id" in text
    assert RUN_1[:8] in text
    assert "do the thing" in text


def test_render_list_marks_first_run_as_latest() -> None:
    """render_list marks the first run (newest) with 'latest'."""
    run1 = _make_run(RUN_1, created_at="2026-01-02T00:00:00+00:00")
    run2 = _make_run(RUN_2, created_at="2026-01-01T00:00:00+00:00")
    text = TextTraceRenderer().render_list([run1, run2])
    assert "latest" in text
    newer_pos = text.index(RUN_1[:8])
    older_pos = text.index(RUN_2[:8])
    assert newer_pos < older_pos
    assert text.count("latest") == 1


def test_render_list_warns_on_malformed_events() -> None:
    """render_list shows a warning line when a run has malformed events."""
    run = _make_run(RUN_1, malformed=3)
    text = TextTraceRenderer().render_list([run])
    assert "warning: skipped 3 malformed events" in text


def test_render_list_no_warning_when_no_malformed() -> None:
    """render_list omits the warning line when there are no malformed events."""
    run = _make_run(RUN_1, malformed=0)
    text = TextTraceRenderer().render_list([run])
    assert "warning" not in text


# ---------------------------------------------------------------------------
# render_run — run header
# ---------------------------------------------------------------------------


def test_render_run_includes_run_header_fields() -> None:
    """render_run includes run_id, created_at, workspace, northstar, event_count."""
    run = _make_run(
        RUN_1,
        created_at="2026-05-10T08:30:00+00:00",
        northstar="deliver value",
        workspace="/proj",
        events=[_event(NODE_1, "work", "producer.response.parsed", attempt=1)],
    )
    text = TextTraceRenderer().render_run(run)
    assert f"run_id: {RUN_1}" in text
    assert "created_at: 2026-05-10T08:30:00+00:00" in text
    assert "workspace: /proj" in text
    assert "northstar: deliver value" in text
    assert "event_count: 1" in text


def test_render_run_reports_malformed_events_in_header() -> None:
    """render_run includes malformed_events_skipped when count > 0."""
    run = _make_run(RUN_1, malformed=2)
    text = TextTraceRenderer().render_run(run)
    assert "malformed_events_skipped: 2" in text


def test_render_run_no_malformed_line_when_zero() -> None:
    """render_run omits malformed_events_skipped when count is 0."""
    run = _make_run(RUN_1, malformed=0)
    text = TextTraceRenderer().render_run(run)
    assert "malformed_events_skipped" not in text


# ---------------------------------------------------------------------------
# render_run — node summary
# ---------------------------------------------------------------------------


def test_render_run_groups_events_into_node_summary() -> None:
    """render_run includes a nodes: section with one line per node."""
    run = _make_run(
        RUN_1,
        events=[
            _event(NODE_1, "plan", "producer.response.parsed", attempt=1, status="completed"),
            _event(NODE_1, "plan", "referee.decision.parsed", attempt=1, status="accept"),
            _event(NODE_2, "work", "producer.response.parsed", attempt=1, status="completed"),
            _event(NODE_2, "work", "pwc.exhausted", attempt=1, status="failed"),
        ],
    )
    text = TextTraceRenderer().render_run(run)
    assert "nodes:" in text
    assert "aaaaaaaa  agent=plan  status=accept  attempts=1" in text
    assert "bbbbbbbb  agent=work  status=failed  attempts=1" in text


def test_render_run_no_events_shows_empty_message() -> None:
    """render_run shows a message when there are no node telemetry events."""
    run = _make_run(RUN_1, events=[])
    text = TextTraceRenderer().render_run(run)
    assert "no node telemetry events" in text


# ---------------------------------------------------------------------------
# render_run — attempt timeline
# ---------------------------------------------------------------------------


def test_render_run_shows_attempt_timeline() -> None:
    """render_run includes a timeline: section with per-attempt groupings."""
    run = _make_run(
        RUN_1,
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1, status="completed"),
            _event(NODE_1, "work", "referee.decision.parsed", attempt=1, status="accept"),
        ],
    )
    text = TextTraceRenderer().render_run(run)
    assert "timeline:" in text
    assert "node aaaaaaaa:" in text
    assert "attempt 1:" in text
    assert "producer parsed: status=completed" in text


def test_render_run_groups_multiple_attempts() -> None:
    """render_run shows separate attempt blocks when a node has multiple attempts."""
    run = _make_run(
        RUN_1,
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1, status="completed"),
            _event(NODE_1, "work", "producer.response.parsed", attempt=2, status="completed"),
        ],
    )
    text = TextTraceRenderer().render_run(run)
    assert "attempt 1:" in text
    assert "attempt 2:" in text


# ---------------------------------------------------------------------------
# render_run — revision items
# ---------------------------------------------------------------------------


def test_render_run_shows_revision_summary() -> None:
    """render_run includes revision appended line in the timeline."""
    run = _make_run(
        RUN_1,
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1, status="completed"),
            _revision_event(NODE_1, attempt=1),
        ],
    )
    text = TextTraceRenderer().render_run(run)
    assert "revision appended: items=1" in text


def test_render_run_single_node_view_shows_revision_items() -> None:
    """Full node detail view includes individual revision item lines."""
    run = _make_run(
        RUN_1,
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1, status="completed"),
            _revision_event(NODE_1, attempt=1),
        ],
    )
    text = TextTraceRenderer().render_run(run, node_prefix=NODE_1[:4])
    assert "revision appended: items=1" in text
    assert "1. AC1: Add tests." in text


# ---------------------------------------------------------------------------
# render_run — single-node view
# ---------------------------------------------------------------------------


def test_render_run_node_prefix_narrows_to_one_node() -> None:
    """node_prefix filters output to the matching node only."""
    run = _make_run(
        RUN_1,
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1, status="completed"),
            _event(NODE_2, "plan", "producer.response.parsed", attempt=1, status="completed"),
        ],
    )
    text = TextTraceRenderer().render_run(run, node_prefix="aaaa")
    assert "aaaaaaaa  agent=work" in text
    assert "bbbbbbbb" not in text


def test_render_run_node_prefix_raises_on_no_match() -> None:
    """node_prefix raises TraceViewerError when no node matches."""
    run = _make_run(
        RUN_1,
        events=[_event(NODE_1, "work", "producer.response.parsed", attempt=1)],
    )
    with pytest.raises(TraceViewerError, match="no node matches"):
        TextTraceRenderer().render_run(run, node_prefix="zzzz")


def test_render_run_node_prefix_raises_on_ambiguity() -> None:
    """node_prefix raises TraceViewerError when multiple nodes match."""
    run = _make_run(
        RUN_1,
        events=[
            _event("aaaaaaaa-1111-4111-8111-111111111111", "work", "producer.response.parsed"),
            _event("aaaaaaaa-2222-4222-8222-222222222222", "work", "producer.response.parsed"),
        ],
    )
    with pytest.raises(TraceViewerError, match="ambiguous node prefix"):
        TextTraceRenderer().render_run(run, node_prefix="aaaaaaaa")


def test_render_run_node_prefix_full_timeline_excludes_other_nodes() -> None:
    """Single-node view includes full timeline for that node and excludes others."""
    run = _make_run(
        RUN_1,
        events=[
            _event(NODE_1, "work", "pwc.exhausted", attempt=1, status="failed"),
            _event(NODE_2, "plan", "producer.response.parsed", attempt=1, status="completed"),
        ],
    )
    text = TextTraceRenderer().render_run(run, node_prefix=NODE_1[:4])
    assert "exhausted:" in text
    assert NODE_2[:8] not in text
