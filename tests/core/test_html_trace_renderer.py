"""Direct tests for HtmlTraceRenderer — HTML presentation of RunTrace objects."""

from pathlib import Path

from forge.core.html_trace_renderer import HtmlTraceRenderer
from forge.core.trace_repository import RunTrace, TraceEvent

RUN_ID = "11111111-1111-4111-8111-111111111111"
NODE_1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
NODE_2 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _run(
    *,
    run_id: str = RUN_ID,
    created_at: str = "2026-01-01T00:00:00+00:00",
    northstar: str = "test goal",
    events: list[TraceEvent] | None = None,
    run_dir: Path | None = None,
    malformed: int = 0,
    tmp_path: Path | None = None,
) -> RunTrace:
    if run_dir is None:
        assert tmp_path is not None, "provide run_dir or tmp_path"
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    return RunTrace(
        run_dir=run_dir,
        run_json={
            "run_id": run_id,
            "created_at": created_at,
            "metadata": {"workspace": "/ws", "northstar": northstar},
        },
        events=events or [],
        malformed_event_count=malformed,
    )


def _event(
    node_id: str,
    agent_type: str,
    event_type: str,
    *,
    attempt: int | None = 1,
    status: str = "completed",
    data: dict[str, object] | None = None,
) -> TraceEvent:
    return TraceEvent(
        line_number=1,
        data={
            "schema_version": 1,
            "event_id": f"event-{node_id[:4]}-{event_type}",
            "run_id": RUN_ID,
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
            "data": data
            or {
                "status": status,
                "output_type": "WorkOutput",
                "work_output": {"file_paths": ["a.py"]},
            },
        },
    )


def _revision_event(node_id: str, *, attempt: int = 1) -> TraceEvent:
    return TraceEvent(
        line_number=2,
        data={
            "schema_version": 1,
            "event_id": f"event-{node_id[:4]}-revision",
            "run_id": RUN_ID,
            "timestamp": "2026-01-01T00:00:01+00:00",
            "node_id": node_id,
            "request_id": node_id,
            "agent_type": "work",
            "attempt_number": attempt,
            "role": "critic",
            "phase": "revision",
            "event_type": "pwc.revision.appended",
            "status": "revise",
            "summary": "revision summary",
            "data": {
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
            },
        },
    )


def _failure_event(node_id: str, *, attempt: int = 1) -> TraceEvent:
    return TraceEvent(
        line_number=3,
        data={
            "schema_version": 1,
            "event_id": f"event-{node_id[:4]}-failed",
            "run_id": RUN_ID,
            "timestamp": "2026-01-01T00:00:02+00:00",
            "node_id": node_id,
            "request_id": node_id,
            "agent_type": "work",
            "attempt_number": attempt,
            "role": "scheduler",
            "phase": "terminal",
            "event_type": "node.failed",
            "status": "failed",
            "summary": "node failed after exhaustion",
            "data": {"error": "max attempts reached"},
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_renders_run_header(tmp_path: Path) -> None:
    """render_run includes run_id, created_at, northstar, and event count."""
    run = _run(
        run_id=RUN_ID,
        created_at="2026-03-15T12:00:00+00:00",
        northstar="build a widget",
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert RUN_ID in result
    assert "2026-03-15T12:00:00+00:00" in result
    assert "build a widget" in result
    assert "event count" in result
    assert "Forge Telemetry Trace" in result


def test_renders_node_overview_cards(tmp_path: Path) -> None:
    """render_run includes one overview card per node with agent type and attempt count."""
    run = _run(
        events=[
            _event(NODE_1, "plan", "producer.response.parsed", attempt=1, status="completed"),
            _event(NODE_2, "work", "producer.response.parsed", attempt=1, status="completed"),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert NODE_1[:8] in result
    assert NODE_2[:8] in result
    assert "plan" in result
    assert "work" in result
    assert "Scheduler / Node Overview" in result


def test_renders_node_detail_anchors(tmp_path: Path) -> None:
    """render_run emits id and href anchors so overview cards link to detail sections."""
    run = _run(
        events=[_event(NODE_1, "work", "producer.response.parsed", attempt=1)],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert f'href="#node-{NODE_1}"' in result
    assert f'id="node-{NODE_1}"' in result


def test_renders_attempt_cards(tmp_path: Path) -> None:
    """render_run groups events into attempt cards with per-attempt headings."""
    run = _run(
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1),
            _event(NODE_1, "work", "producer.response.parsed", attempt=2),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "Attempt 1" in result
    assert "Attempt 2" in result


def test_renders_revision_items(tmp_path: Path) -> None:
    """render_run renders revision criterion id, required change, and collapsible details."""
    run = _run(
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1),
            _revision_event(NODE_1, attempt=1),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "Revision appended with 1 item(s)." in result
    assert "AC1" in result
    assert "Add tests." in result
    assert "Revision item details" in result


def test_renders_node_failure_events(tmp_path: Path) -> None:
    """render_run includes failure summary text from node.failed events."""
    run = _run(
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1),
            _failure_event(NODE_1, attempt=1),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "node.failed" in result
    assert "failed" in result


def test_escapes_unsafe_model_text(tmp_path: Path) -> None:
    """render_run HTML-escapes all user and model-supplied text."""
    run = _run(
        northstar="<script>alert('xss')</script>",
        events=[
            TraceEvent(
                line_number=1,
                data={
                    "schema_version": 1,
                    "event_id": "ev1",
                    "run_id": RUN_ID,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "node_id": NODE_1,
                    "request_id": NODE_1,
                    "agent_type": "work",
                    "attempt_number": 1,
                    "role": "critic",
                    "phase": "revision",
                    "event_type": "pwc.revision.appended",
                    "status": "revise",
                    "summary": "<img src=x onerror=alert(1)>",
                    "data": {
                        "revision_request": {
                            "rationale": "<script>bad</script>",
                            "prior_attempts": 1,
                            "items": [
                                {
                                    "criterion_id": "AC<b>",
                                    "required_change": "<b>inject</b>",
                                    "rationale": "<i>why</i>",
                                }
                            ],
                        }
                    },
                },
            )
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "<script>alert" not in result
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in result
    assert "<b>inject</b>" not in result
    assert "&lt;b&gt;inject&lt;/b&gt;" in result


def test_writes_index_html_to_run_directory(tmp_path: Path) -> None:
    """write_run writes index.html inside run_dir by default."""
    run = _run(
        events=[_event(NODE_1, "work", "producer.response.parsed", attempt=1)],
        tmp_path=tmp_path,
    )

    output_path = HtmlTraceRenderer().write_run(run)

    assert output_path == run.run_dir / "index.html"
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert RUN_ID in content


def test_write_run_accepts_custom_output_path(tmp_path: Path) -> None:
    """write_run writes to the provided output_path when given."""
    run = _run(tmp_path=tmp_path)
    custom_path = tmp_path / "report.html"

    output_path = HtmlTraceRenderer().write_run(run, output_path=custom_path)

    assert output_path == custom_path
    assert custom_path.exists()


def test_empty_events_renders_placeholder_messages(tmp_path: Path) -> None:
    """render_run shows useful messages when there are no node events."""
    run = _run(events=[], tmp_path=tmp_path)

    result = HtmlTraceRenderer().render_run(run)

    assert "No node telemetry events found." in result
    assert "No event details available." in result


def test_malformed_events_shows_warning(tmp_path: Path) -> None:
    """render_run shows a warning when malformed events were skipped."""
    run = _run(events=[], malformed=3, tmp_path=tmp_path)

    result = HtmlTraceRenderer().render_run(run)

    assert "Skipped 3 malformed events." in result


def _dispatched_event(node_id: str) -> TraceEvent:
    return TraceEvent(
        line_number=1,
        data={
            "schema_version": 1,
            "event_id": f"event-{node_id[:4]}-dispatched",
            "run_id": RUN_ID,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "node_id": node_id,
            "request_id": node_id,
            "agent_type": "work",
            "attempt_number": None,
            "role": "scheduler",
            "phase": "scheduler",
            "event_type": "node.dispatched",
            "status": "dispatched",
            "summary": "Implement parser",
            "data": {
                "contract": {
                    "objective": "Implement parser",
                    "success_condition": "tests pass",
                    "artifact": "codebase",
                    "adapter": "coding",
                    "acceptance_criteria": [
                        {"id": "AC1", "text": "parse tags"},
                        {"id": "AC2", "text": "parse classes"},
                    ],
                }
            },
        },
    )


def test_renders_node_contract_section(tmp_path: Path) -> None:
    """render_run includes a collapsible Contract section for work nodes."""
    run = _run(
        events=[
            _dispatched_event(NODE_1),
            _event(NODE_1, "work", "producer.response.parsed"),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "contract-section" in result
    assert "Contract" in result
    assert "Implement parser" in result
    assert "codebase" in result
    assert "coding" in result
    assert "parse tags" in result
    assert "parse classes" in result


def test_renders_no_contract_section_without_dispatched_event(tmp_path: Path) -> None:
    """render_run omits the contract section when no node.dispatched event is present."""
    run = _run(
        events=[_event(NODE_1, "work", "producer.response.parsed")],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert '<details class="contract-section">' not in result


# ---------------------------------------------------------------------------
# New fixtures
# ---------------------------------------------------------------------------


def _plan_producer_event(node_id: str, *, attempt: int = 1) -> TraceEvent:
    return TraceEvent(
        line_number=4,
        data={
            "schema_version": 1,
            "event_id": f"event-{node_id[:4]}-plan",
            "run_id": RUN_ID,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "node_id": node_id,
            "request_id": node_id,
            "agent_type": "plan",
            "attempt_number": attempt,
            "role": "producer",
            "phase": "producer",
            "event_type": "producer.response.parsed",
            "status": "completed",
            "summary": "producer returned completed",
            "data": {
                "status": "completed",
                "output_type": "PlanResponse",
                "plan": {
                    "task_count": 2,
                    "tasks": [
                        {
                            "objective": "Implement the parser",
                            "adapter": "coding",
                            "artifact": "codebase",
                            "language": "python",
                            "depends_on": [],
                        },
                        {
                            "objective": "Add unit tests",
                            "adapter": "coding",
                            "artifact": "codebase",
                            "language": "python",
                            "depends_on": [1],
                        },
                    ],
                },
            },
        },
    )


def _max_iter_producer_event(node_id: str, *, attempt: int = 1) -> TraceEvent:
    return TraceEvent(
        line_number=5,
        data={
            "schema_version": 1,
            "event_id": f"event-{node_id[:4]}-maxiter",
            "run_id": RUN_ID,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "node_id": node_id,
            "request_id": node_id,
            "agent_type": "work",
            "attempt_number": attempt,
            "role": "producer",
            "phase": "producer",
            "event_type": "producer.response.parsed",
            "status": "failed",
            "summary": "producer returned failed",
            "data": {
                "status": "failed",
                "output_type": None,
                "failure_kind": "max_iterations",
                "diagnostics": [
                    {
                        "kind": "max_iterations",
                        "message": (
                            "last_tool_calls=[write_file, run_tests] "
                            "ran_tests_and_passed=False "
                            "final_response_only=True "
                            "has_run_tests=True "
                            "mutating_tool_succeeded=False"
                        ),
                        "raw_response_excerpt": "here is my partial response...",
                    }
                ],
            },
        },
    )


# ---------------------------------------------------------------------------
# New tests for rendering improvements
# ---------------------------------------------------------------------------


def test_contract_section_open_by_default(tmp_path: Path) -> None:
    """Contract details section has the open attribute so it shows expanded by default."""
    run = _run(
        events=[
            _dispatched_event(NODE_1),
            _event(NODE_1, "work", "producer.response.parsed"),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert '<details class="contract-section" open>' in result


def test_attempt_cards_are_collapsible_details(tmp_path: Path) -> None:
    """Attempt cards are rendered as <details> elements with a summary."""
    run = _run(
        events=[_event(NODE_1, "work", "producer.response.parsed", attempt=1)],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert 'class="attempt-card"' in result
    assert "<summary>Attempt 1</summary>" in result


def test_final_attempt_expanded_earlier_attempts_collapsed(tmp_path: Path) -> None:
    """Final attempt card has open attribute; earlier attempt cards do not."""
    run = _run(
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1),
            _event(NODE_1, "work", "producer.response.parsed", attempt=2),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert '<details class="attempt-card" open>' in result
    assert result.count('<details class="attempt-card" open>') == 1
    assert '<details class="attempt-card">' in result


def test_revision_block_appears_between_attempts(tmp_path: Path) -> None:
    """A revision-block div appears between attempts when a revision was appended."""
    run = _run(
        events=[
            _event(NODE_1, "work", "producer.response.parsed", attempt=1),
            _revision_event(NODE_1, attempt=1),
            _event(NODE_1, "work", "producer.response.parsed", attempt=2),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert 'class="revision-block"' in result
    assert "Revision request after attempt 1" in result
    assert "missing coverage" in result
    assert "AC1" in result


def test_plan_task_list_rendered(tmp_path: Path) -> None:
    """Plan tasks are rendered as a readable list for producer.response.parsed with plan data."""
    run = _run(
        events=[_plan_producer_event(NODE_1)],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "Plan tasks (2)" in result
    assert "Implement the parser" in result
    assert "Add unit tests" in result
    assert "plan-task" in result
    assert "depends on: 1" in result


def test_max_iterations_diagnostics_block(tmp_path: Path) -> None:
    """Max-iteration failures render a loop-diag block with key diagnostic fields."""
    run = _run(
        events=[_max_iter_producer_event(NODE_1)],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "loop-diag" in result
    assert "Loop diagnostics (max_iterations)" in result
    assert "last_tool_calls=" in result
    assert "ran_tests_and_passed=False" in result
    assert "here is my partial response..." in result


# ---------------------------------------------------------------------------
# Node overview objective tests
# ---------------------------------------------------------------------------


def _dispatched_event_with_objective(node_id: str, agent_type: str, objective: str) -> TraceEvent:
    return TraceEvent(
        line_number=10,
        data={
            "schema_version": 1,
            "event_id": f"event-{node_id[:4]}-dispatched",
            "run_id": RUN_ID,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "node_id": node_id,
            "request_id": node_id,
            "agent_type": agent_type,
            "attempt_number": None,
            "role": "scheduler",
            "phase": "scheduler",
            "event_type": "node.dispatched",
            "status": "dispatched",
            "summary": objective,
            "data": {
                "contract": {
                    "objective": objective,
                    "success_condition": "done",
                    "artifact": "codebase",
                    "adapter": "coding",
                    "acceptance_criteria": [],
                }
            },
        },
    )


def test_work_node_overview_card_includes_objective(tmp_path: Path) -> None:
    """Work node overview card shows the contract objective prominently."""
    run = _run(
        events=[
            _dispatched_event_with_objective(NODE_1, "work", "Set up Python web scraper"),
            _event(NODE_1, "work", "producer.response.parsed", attempt=1),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "node-objective" in result
    assert "Set up Python web scraper" in result


def test_plan_node_overview_card_includes_objective(tmp_path: Path) -> None:
    """Plan node overview card shows the contract objective when available."""
    run = _run(
        events=[
            _dispatched_event_with_objective(NODE_1, "plan", "Plan the project structure"),
            _plan_producer_event(NODE_1),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "node-objective" in result
    assert "Plan the project structure" in result


def test_overview_card_renders_without_objective(tmp_path: Path) -> None:
    """Overview card renders correctly when no node.dispatched event is present."""
    run = _run(
        events=[_event(NODE_1, "work", "producer.response.parsed", attempt=1)],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert NODE_1[:8] in result
    assert '<p class="node-objective">' not in result


def test_long_objective_truncated_in_overview(tmp_path: Path) -> None:
    """Objectives longer than 120 chars are truncated in the overview card (117 chars + '...')."""
    long_objective = "A" * 130
    run = _run(
        events=[
            _dispatched_event_with_objective(NODE_1, "work", long_objective),
            _event(NODE_1, "work", "producer.response.parsed", attempt=1),
        ],
        tmp_path=tmp_path,
    )

    result = HtmlTraceRenderer().render_run(run)

    assert "node-objective" in result
    assert "A" * 117 + "..." in result
