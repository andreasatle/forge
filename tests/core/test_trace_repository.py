"""Tests for TraceRepository — read-only telemetry run data access."""

import json
from pathlib import Path

import pytest

from forge.core.trace_repository import TraceRepository, TraceViewerError

RUN_1 = "11111111-1111-4111-8111-111111111111"
RUN_2 = "22222222-2222-4222-8222-222222222222"
RUN_3 = "33333333-3333-4333-8333-333333333333"
NODE_1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _event(node_id: str, event_type: str, *, attempt: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "event_id": f"event-{node_id[:4]}-{event_type}",
        "run_id": RUN_1,
        "timestamp": "2026-01-01T00:01:00+00:00",
        "node_id": node_id,
        "agent_type": "work",
        "attempt_number": attempt,
        "event_type": event_type,
        "status": "completed",
        "summary": f"{event_type} summary",
        "data": {},
    }


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_returns_newest_first(tmp_path: Path) -> None:
    """list_runs orders runs newest-first by created_at."""
    workspace = tmp_path / "ws"
    _write_run(workspace, RUN_1, created_at="2026-01-01T00:00:00+00:00")
    _write_run(workspace, RUN_2, created_at="2026-01-02T00:00:00+00:00")
    _write_run(workspace, RUN_3, created_at="2026-01-03T00:00:00+00:00")

    runs = TraceRepository(workspace).list_runs()

    assert [r.run_id for r in runs] == [RUN_3, RUN_2, RUN_1]


def test_list_runs_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    """list_runs returns an empty list when the runs directory exists but is empty."""
    workspace = tmp_path / "ws"
    (workspace / "telemetry" / "runs").mkdir(parents=True)

    runs = TraceRepository(workspace).list_runs()

    assert runs == []


def test_list_runs_missing_directory_raises(tmp_path: Path) -> None:
    """list_runs raises TraceViewerError when the telemetry directory does not exist."""
    with pytest.raises(TraceViewerError, match="telemetry directory not found"):
        TraceRepository(tmp_path / "missing").list_runs()


# ---------------------------------------------------------------------------
# latest_run
# ---------------------------------------------------------------------------


def test_latest_run_returns_newest(tmp_path: Path) -> None:
    """latest_run returns the run with the most recent created_at."""
    workspace = tmp_path / "ws"
    _write_run(workspace, RUN_1, created_at="2026-01-01T00:00:00+00:00")
    _write_run(workspace, RUN_2, created_at="2026-01-05T00:00:00+00:00")

    run = TraceRepository(workspace).latest_run()

    assert run is not None
    assert run.run_id == RUN_2


def test_latest_run_empty_directory_returns_none(tmp_path: Path) -> None:
    """latest_run returns None when the runs directory exists but has no runs."""
    workspace = tmp_path / "ws"
    (workspace / "telemetry" / "runs").mkdir(parents=True)

    run = TraceRepository(workspace).latest_run()

    assert run is None


def test_latest_run_missing_directory_raises(tmp_path: Path) -> None:
    """latest_run propagates TraceViewerError when telemetry directory is absent."""
    with pytest.raises(TraceViewerError, match="telemetry directory not found"):
        TraceRepository(tmp_path / "missing").latest_run()


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------


def test_load_run_parses_run_json_and_events(tmp_path: Path) -> None:
    """load_run reads run.json metadata and all valid events.jsonl lines."""
    workspace = tmp_path / "ws"
    run_dir = _write_run(
        workspace,
        RUN_1,
        northstar="verify loading",
        events=[
            _event(NODE_1, "producer.response.parsed"),
            _event(NODE_1, "referee.decision.parsed"),
        ],
    )

    trace = TraceRepository.load_run(run_dir)

    assert trace.run_id == RUN_1
    assert trace.northstar == "verify loading"
    assert len(trace.events) == 2
    assert trace.malformed_event_count == 0


def test_load_run_missing_run_json_falls_back_gracefully(tmp_path: Path) -> None:
    """load_run tolerates a missing run.json and falls back to directory name."""
    run_dir = tmp_path / "telemetry" / "runs" / RUN_1
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    trace = TraceRepository.load_run(run_dir)

    assert trace.run_id == RUN_1
    assert trace.metadata == {}


def test_load_run_malformed_run_json_falls_back_gracefully(tmp_path: Path) -> None:
    """load_run tolerates a malformed run.json and falls back to empty metadata."""
    run_dir = tmp_path / "telemetry" / "runs" / RUN_1
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{not json}", encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    trace = TraceRepository.load_run(run_dir)

    assert trace.run_json == {}


def test_load_run_counts_malformed_event_lines(tmp_path: Path) -> None:
    """load_run skips and counts lines that are not valid JSON objects."""
    workspace = tmp_path / "ws"
    run_dir = _write_run(
        workspace,
        RUN_1,
        events=[_event(NODE_1, "producer.response.parsed")],
    )
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write("[1, 2, 3]\n")

    trace = TraceRepository.load_run(run_dir)

    assert len(trace.events) == 1
    assert trace.malformed_event_count == 2


def test_load_run_preserves_line_numbers(tmp_path: Path) -> None:
    """load_run records the 1-based source line number for each parsed event."""
    workspace = tmp_path / "ws"
    run_dir = _write_run(
        workspace,
        RUN_1,
        events=[
            _event(NODE_1, "producer.response.parsed"),
            _event(NODE_1, "referee.decision.parsed"),
        ],
    )

    trace = TraceRepository.load_run(run_dir)

    assert trace.events[0].line_number == 1
    assert trace.events[1].line_number == 2


def test_load_run_skips_blank_lines(tmp_path: Path) -> None:
    """load_run skips blank lines without counting them as malformed."""
    run_dir = tmp_path / "telemetry" / "runs" / RUN_1
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": RUN_1}), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        "\n" + json.dumps(_event(NODE_1, "producer.response.parsed")) + "\n\n",
        encoding="utf-8",
    )

    trace = TraceRepository.load_run(run_dir)

    assert len(trace.events) == 1
    assert trace.malformed_event_count == 0


# ---------------------------------------------------------------------------
# resolve_run_dir
# ---------------------------------------------------------------------------


def test_resolve_run_dir_exact_match(tmp_path: Path) -> None:
    """resolve_run_dir returns the directory for an exact run id."""
    workspace = tmp_path / "ws"
    _write_run(workspace, RUN_1)

    result = TraceRepository(workspace).resolve_run_dir(RUN_1)

    assert result.name == RUN_1


def test_resolve_run_dir_prefix_match(tmp_path: Path) -> None:
    """resolve_run_dir resolves an unambiguous run id prefix."""
    workspace = tmp_path / "ws"
    _write_run(workspace, RUN_1)
    _write_run(workspace, RUN_2)

    result = TraceRepository(workspace).resolve_run_dir(RUN_1[:8])

    assert result.name == RUN_1


def test_resolve_run_dir_ambiguous_prefix_raises(tmp_path: Path) -> None:
    """resolve_run_dir raises on an ambiguous prefix."""
    workspace = tmp_path / "ws"
    _write_run(workspace, "aaaaaaaa-1111-4111-8111-111111111111")
    _write_run(workspace, "aaaaaaaa-2222-4222-8222-222222222222")

    with pytest.raises(TraceViewerError, match="ambiguous run id prefix"):
        TraceRepository(workspace).resolve_run_dir("aaaaaaaa")


def test_resolve_run_dir_no_match_raises(tmp_path: Path) -> None:
    """resolve_run_dir raises when no run matches the prefix."""
    workspace = tmp_path / "ws"
    _write_run(workspace, RUN_1)

    with pytest.raises(TraceViewerError, match="no telemetry run matches"):
        TraceRepository(workspace).resolve_run_dir("ffffffff")


def test_resolve_run_dir_missing_directory_raises(tmp_path: Path) -> None:
    """resolve_run_dir raises when the telemetry directory does not exist."""
    with pytest.raises(TraceViewerError, match="telemetry directory not found"):
        TraceRepository(tmp_path / "missing").resolve_run_dir("abc")
