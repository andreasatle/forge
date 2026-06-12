"""Read-only telemetry trace rendering for human CLI inspection."""

import json
from collections import defaultdict
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
        return _one_line(value) if isinstance(value, str) and value else "unknown"


class TraceViewerError(ValueError):
    """User-facing trace viewer error."""


def render_trace_list(workspace: Path) -> str:
    """Render all telemetry runs newest first."""
    runs = _load_runs(workspace)
    if not runs:
        return f"no telemetry runs found in {workspace / 'telemetry' / 'runs'}"

    latest = runs[0].run_id
    lines = ["created_at                         run_id    events  status    northstar"]
    for run in runs:
        marker = "latest " if run.run_id == latest else "       "
        lines.append(
            f"{marker}{_fit(run.created_at, 32)}  "
            f"{run.run_id[:8]:<8}  "
            f"{len(run.events):>6}  "
            f"{_fit(_final_status(run.events), 8)}  "
            f"{_fit(run.northstar, 80)}"
        )
        if run.malformed_event_count:
            lines.append(f"        warning: skipped {run.malformed_event_count} malformed events")
    return "\n".join(lines)


def render_latest_trace(workspace: Path) -> str:
    """Render the newest telemetry run."""
    runs = _load_runs(workspace)
    if not runs:
        return f"no telemetry runs found in {workspace / 'telemetry' / 'runs'}"
    return render_run_trace(runs[0].run_dir)


def render_run_trace(run_dir: Path, *, node_prefix: str | None = None) -> str:
    """Render a run summary, optionally narrowed to one node."""
    trace = _load_run(run_dir)
    if node_prefix is not None:
        node_id = _resolve_node_prefix(trace.events, node_prefix)
        return _render_node_detail(trace, node_id)
    return _render_run_summary(trace)


def resolve_run_dir(workspace: Path, run_id_prefix: str) -> Path:
    """Resolve a run id or unambiguous run id prefix to a telemetry run directory."""
    runs_dir = _runs_dir(workspace)
    if not runs_dir.is_dir():
        raise TraceViewerError(f"telemetry directory not found: {runs_dir}")

    matches = sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and path.name.startswith(run_id_prefix)
    )
    if not matches:
        raise TraceViewerError(f"no telemetry run matches: {run_id_prefix}")
    if len(matches) > 1:
        short = ", ".join(path.name[:8] for path in matches[:8])
        raise TraceViewerError(f"ambiguous run id prefix {run_id_prefix!r}: {short}")
    return matches[0]


def _render_run_summary(trace: RunTrace) -> str:
    lines = _run_header(trace)
    grouped = _events_by_node(trace.events)
    if not grouped:
        lines.append("")
        lines.append("no node telemetry events")
        return "\n".join(lines)

    lines.append("")
    lines.append("nodes:")
    for node_id, events in sorted(grouped.items(), key=lambda item: _event_sort_key(item[1])):
        lines.append(_node_summary_line(node_id, events))

    lines.append("")
    lines.append("timeline:")
    for node_id, events in sorted(grouped.items(), key=lambda item: _event_sort_key(item[1])):
        lines.append(f"node {node_id[:8]}:")
        lines.extend(_compact_timeline(events, indent="  "))
    return "\n".join(lines)


def _render_node_detail(trace: RunTrace, node_id: str) -> str:
    grouped = _events_by_node(trace.events)
    events = grouped[node_id]
    lines = _run_header(trace)
    lines.append("")
    lines.append(_node_summary_line(node_id, events))
    lines.append("")
    lines.append("timeline:")
    lines.extend(_compact_timeline(events, indent="  ", full=True))
    return "\n".join(lines)


def _run_header(trace: RunTrace) -> list[str]:
    lines = [
        f"run_id: {trace.run_id}",
        f"created_at: {trace.created_at}",
        f"workspace: {trace.workspace}",
        f"northstar: {trace.northstar}",
        f"event_count: {len(trace.events)}",
    ]
    if trace.malformed_event_count:
        lines.append(f"malformed_events_skipped: {trace.malformed_event_count}")
    return lines


def _node_summary_line(node_id: str, events: list[TraceEvent]) -> str:
    agent_type = _last_str(events, "agent_type") or "unknown"
    attempts = sorted(
        {
            attempt
            for event in events
            if isinstance((attempt := event.data.get("attempt_number")), int)
        }
    )
    final = _final_status(events)
    last_failure = _last_event(events, {"node.failed", "pwc.exhausted"})
    last_revision = _last_event(events, {"pwc.revision.appended"})
    tail = _event_summary(last_failure or last_revision)
    if tail:
        tail = f"  last: {tail}"
    return f"  {node_id[:8]}  agent={agent_type}  status={final}  attempts={len(attempts)}{tail}"


def _compact_timeline(events: list[TraceEvent], *, indent: str, full: bool = False) -> list[str]:
    lines: list[str] = []
    by_attempt: dict[int, list[TraceEvent]] = defaultdict(list)
    no_attempt: list[TraceEvent] = []
    for event in events:
        attempt = event.data.get("attempt_number")
        if isinstance(attempt, int):
            by_attempt[attempt].append(event)
        else:
            no_attempt.append(event)

    for attempt in sorted(by_attempt):
        lines.append(f"{indent}attempt {attempt}:")
        for event in _interesting_events(by_attempt[attempt]):
            lines.extend(_render_event(event, indent=f"{indent}  ", full=full))

    for event in _interesting_events(no_attempt):
        lines.extend(_render_event(event, indent=indent, full=full))
    return lines or [f"{indent}no timeline events"]


def _render_event(event: TraceEvent, *, indent: str, full: bool) -> list[str]:
    event_type = _str_value(event.data.get("event_type")) or "unknown"
    if event_type == "producer.response.parsed":
        data = _dict_value(event.data.get("data"))
        status = _str_value(data.get("status")) or _str_value(event.data.get("status")) or "unknown"
        output_type = _str_value(data.get("output_type")) or "none"
        parts = [f"{indent}producer parsed: status={status} output_type={output_type}"]
        plan = _dict_value(data.get("plan"))
        delta = _dict_value(data.get("delta"))
        if plan:
            parts[0] += f" plan_tasks={plan.get('task_count', 0)}"
        if delta:
            parts[0] += f" delta={_delta_text(delta)}"
        error = _str_value(data.get("error"))
        if error:
            parts[0] += f" error={error}"
        return parts

    if event_type == "critic.finding.parsed":
        return [f"{indent}critic: {_disposition_line(event, 'critic_finding')}"]
    if event_type == "referee.decision.parsed":
        return [f"{indent}referee: {_disposition_line(event, 'referee_decision')}"]
    if event_type == "pwc.revision.appended":
        return _render_revision(event, indent=indent, full=full)
    if event_type == "pwc.exhausted":
        return [f"{indent}exhausted: {_event_summary(event)}"]
    if event_type == "node.failed":
        return [f"{indent}node.failed: {_event_summary(event)}"]
    if event_type == "pwc.decompose.requested":
        return [f"{indent}decompose requested: {_event_summary(event)}"]
    if event_type == "node.decomposed":
        return [f"{indent}node.decomposed: {_event_summary(event)}"]
    if full:
        return [f"{indent}{event_type}: {_event_summary(event)}"]
    return []


def _render_revision(event: TraceEvent, *, indent: str, full: bool) -> list[str]:
    data = _dict_value(event.data.get("data"))
    revision = _dict_value(data.get("revision_request"))
    rationale = _str_value(revision.get("rationale")) or _str_value(event.data.get("summary")) or ""
    items = revision.get("items")
    revision_items = _list_value(items)
    item_count = len(revision_items)
    lines = [
        f"{indent}revision appended: items={item_count} {_fit(_one_line(rationale), 120)}".rstrip()
    ]
    if full:
        for index, item in enumerate(revision_items, start=1):
            item_data = _dict_value(item)
            if not item_data:
                continue
            change = (
                _str_value(item_data.get("required_change"))
                or _str_value(item_data.get("rationale"))
                or ""
            )
            criterion = _str_value(item_data.get("criterion_id"))
            prefix = f"{index}."
            if criterion:
                prefix = f"{index}. {criterion}:"
            lines.append(f"{indent}  {prefix} {_one_line(change)}")
    return lines


def _disposition_line(event: TraceEvent, model_key: str) -> str:
    data = _dict_value(event.data.get("data"))
    model = _dict_value(data.get(model_key))
    disposition = (
        _str_value(model.get("disposition")) or _str_value(event.data.get("status")) or "unknown"
    )
    rationale = _str_value(model.get("rationale")) or _str_value(event.data.get("summary")) or ""
    return f"{disposition} {_fit(_one_line(rationale), 120)}".rstrip()


def _event_summary(event: TraceEvent | None) -> str:
    if event is None:
        return ""
    status = _str_value(event.data.get("status"))
    summary = _str_value(event.data.get("summary"))
    data = _dict_value(event.data.get("data"))
    error = _str_value(data.get("error"))
    bits = [bit for bit in [status, summary, error] if bit]
    return _fit(_one_line(" | ".join(bits)), 140)


def _interesting_events(events: list[TraceEvent]) -> list[TraceEvent]:
    keep = {
        "producer.response.parsed",
        "critic.finding.parsed",
        "referee.decision.parsed",
        "pwc.revision.appended",
        "pwc.exhausted",
        "node.failed",
        "pwc.decompose.requested",
        "node.decomposed",
    }
    return [event for event in events if event.data.get("event_type") in keep]


def _load_runs(workspace: Path) -> list[RunTrace]:
    runs_dir = _runs_dir(workspace)
    if not runs_dir.is_dir():
        raise TraceViewerError(f"telemetry directory not found: {runs_dir}")
    runs = [_load_run(path) for path in runs_dir.iterdir() if path.is_dir()]
    return sorted(runs, key=_run_sort_key, reverse=True)


def _load_run(run_dir: Path) -> RunTrace:
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
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                events.append(TraceEvent(line_number=line_number, data=cast(dict[str, Any], value)))
            else:
                malformed += 1
    return RunTrace(
        run_dir=run_dir, run_json=run_json, events=events, malformed_event_count=malformed
    )


def _resolve_node_prefix(events: list[TraceEvent], prefix: str) -> str:
    node_ids = sorted(
        {
            node_id
            for event in events
            if isinstance((node_id := event.data.get("node_id")), str) and node_id
        }
    )
    matches = [node_id for node_id in node_ids if node_id.startswith(prefix)]
    if not matches:
        raise TraceViewerError(f"no node matches: {prefix}")
    if len(matches) > 1:
        short = ", ".join(node_id[:8] for node_id in matches[:8])
        raise TraceViewerError(f"ambiguous node prefix {prefix!r}: {short}")
    return matches[0]


def _events_by_node(events: list[TraceEvent]) -> dict[str, list[TraceEvent]]:
    grouped: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in events:
        node_id = event.data.get("node_id")
        if isinstance(node_id, str) and node_id:
            grouped[node_id].append(event)
    return dict(grouped)


def _final_status(events: list[TraceEvent]) -> str:
    failed = _last_event(events, {"node.failed", "pwc.exhausted"})
    if failed is not None:
        return _str_value(failed.data.get("status")) or "failed"
    referee = _last_event(events, {"referee.decision.parsed"})
    if referee is not None:
        return _str_value(referee.data.get("status")) or "unknown"
    critic = _last_event(events, {"critic.finding.parsed"})
    if critic is not None:
        return _str_value(critic.data.get("status")) or "unknown"
    producer = _last_event(events, {"producer.response.parsed"})
    if producer is not None:
        return _str_value(producer.data.get("status")) or "unknown"
    return "unknown"


def _last_event(events: list[TraceEvent], event_types: set[str]) -> TraceEvent | None:
    for event in reversed(events):
        if event.data.get("event_type") in event_types:
            return event
    return None


def _last_str(events: list[TraceEvent], key: str) -> str | None:
    for event in reversed(events):
        value = event.data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _event_sort_key(events: list[TraceEvent]) -> tuple[str, int]:
    first = events[0] if events else None
    timestamp = _str_value(first.data.get("timestamp")) if first is not None else ""
    line = first.line_number if first is not None else 0
    return (timestamp or "", line)


def _run_sort_key(run: RunTrace) -> tuple[datetime, float]:
    return (_parse_datetime(run.created_at), run.run_dir.stat().st_mtime)


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0)


def _runs_dir(workspace: Path) -> Path:
    return workspace / "telemetry" / "runs"


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat()


def _delta_text(delta: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key, label in [
        ("new_file_paths", "new"),
        ("edit_paths", "edits"),
        ("dependencies", "deps"),
    ]:
        value = delta.get(key)
        values = _list_value(value)
        if values:
            pieces.append(f"{label}={len(values)}")
    base = _str_value(delta.get("base_version"))
    if base:
        pieces.append(f"base={base}")
    return ",".join(pieces) if pieces else "none"


def _dict_value(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _list_value(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def _str_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _fit(value: str, width: int) -> str:
    text = _one_line(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."
