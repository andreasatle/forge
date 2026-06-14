"""Text presentation of telemetry trace objects."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast

from forge.core.trace_repository import RunTrace, TraceEvent, TraceViewerError

__all__ = ["TextTraceRenderer"]


class TextTraceRenderer:
    """Renders loaded RunTrace objects as human-readable text."""

    def render_list(self, runs: Sequence[RunTrace]) -> str:
        """Render a list of runs newest-first with a latest marker."""
        latest = runs[0].run_id
        lines = ["created_at                         run_id    events  status    northstar"]
        for run in runs:
            marker = "latest " if run.run_id == latest else "       "
            lines.append(
                f"{marker}{fit(run.created_at, 32)}  "
                f"{run.run_id[:8]:<8}  "
                f"{len(run.events):>6}  "
                f"{fit(final_status(run.events), 8)}  "
                f"{fit(run.northstar, 80)}"
            )
            if run.malformed_event_count:
                lines.append(
                    f"        warning: skipped {run.malformed_event_count} malformed events"
                )
        return "\n".join(lines)

    def render_run(self, trace: RunTrace, node_prefix: str | None = None) -> str:
        """Render a run summary, optionally narrowed to one node."""
        if node_prefix is not None:
            node_id = self._resolve_node_prefix(trace.events, node_prefix)
            return self._render_node_detail(trace, node_id)
        return self._render_run_summary(trace)

    def _render_run_summary(self, trace: RunTrace) -> str:
        lines = self._run_header(trace)
        grouped = events_by_node(trace.events)
        if not grouped:
            lines.append("")
            lines.append("no node telemetry events")
            return "\n".join(lines)

        lines.append("")
        lines.append("nodes:")
        for node_id, evts in sorted(grouped.items(), key=lambda item: event_sort_key(item[1])):
            lines.append(_node_summary_line(node_id, evts))

        lines.append("")
        lines.append("timeline:")
        for node_id, evts in sorted(grouped.items(), key=lambda item: event_sort_key(item[1])):
            lines.append(f"node {node_id[:8]}:")
            lines.extend(_compact_timeline(evts, indent="  "))
        return "\n".join(lines)

    def _render_node_detail(self, trace: RunTrace, node_id: str) -> str:
        grouped = events_by_node(trace.events)
        evts = grouped[node_id]
        lines = self._run_header(trace)
        lines.append("")
        lines.append(_node_summary_line(node_id, evts))
        lines.append("")
        lines.append("timeline:")
        lines.extend(_compact_timeline(evts, indent="  ", full=True))
        return "\n".join(lines)

    def _run_header(self, trace: RunTrace) -> list[str]:
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

    def _resolve_node_prefix(self, events: list[TraceEvent], prefix: str) -> str:
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


# ---------------------------------------------------------------------------
# Text-specific formatting helpers (private to this module)
# ---------------------------------------------------------------------------


def _node_summary_line(node_id: str, evts: list[TraceEvent]) -> str:
    agent_type = last_str(evts, "agent_type") or "unknown"
    attempts = attempt_numbers(evts)
    status = final_status(evts)
    last_failure = last_event(evts, {"node.failed", "pwc.exhausted"})
    last_revision = last_event(evts, {"pwc.revision.appended"})
    tail = event_summary(last_failure or last_revision)
    if tail:
        tail = f"  last: {tail}"
    return f"  {node_id[:8]}  agent={agent_type}  status={status}  attempts={len(attempts)}{tail}"


def _compact_timeline(evts: list[TraceEvent], *, indent: str, full: bool = False) -> list[str]:
    lines: list[str] = []
    no_attempt: list[TraceEvent] = []
    for event in evts:
        if not isinstance(event.data.get("attempt_number"), int):
            no_attempt.append(event)

    for attempt, attempt_evts in attempt_groups(evts):
        lines.append(f"{indent}attempt {attempt}:")
        for event in interesting_events(attempt_evts):
            lines.extend(_render_event(event, indent=f"{indent}  ", full=full))

    for event in interesting_events(no_attempt):
        lines.extend(_render_event(event, indent=indent, full=full))
    return lines or [f"{indent}no timeline events"]


def _render_event(event: TraceEvent, *, indent: str, full: bool) -> list[str]:
    event_type = str_value(event.data.get("event_type")) or "unknown"
    if event_type == "producer.response.parsed":
        data = dict_value(event.data.get("data"))
        status = str_value(data.get("status")) or str_value(event.data.get("status")) or "unknown"
        output_type = str_value(data.get("output_type")) or "none"
        parts = [f"{indent}producer parsed: status={status} output_type={output_type}"]
        plan = dict_value(data.get("plan"))
        work_out = dict_value(data.get("work_output"))
        if plan:
            parts[0] += f" plan_tasks={plan.get('task_count', 0)}"
        if work_out:
            parts[0] += f" work_output={work_output_text(work_out)}"
        error = str_value(data.get("error"))
        if error:
            parts[0] += f" error={error}"
        for diag in list_value(data.get("diagnostics")):
            diag_dict = dict_value(diag)
            excerpt = str_value(diag_dict.get("raw_response_excerpt"))
            if excerpt:
                parts.append(f"{indent}  raw_response: {fit(excerpt, 300)}")
            if str_value(diag_dict.get("kind")) == "max_iterations":
                msg = str_value(diag_dict.get("message"))
                if msg:
                    parts.append(f"{indent}  iterations: {fit(msg, 200)}")
        return parts

    if event_type == "critic.finding.parsed":
        return [f"{indent}critic: {_disposition_line(event, 'critic_finding')}"]
    if event_type == "referee.decision.parsed":
        return [f"{indent}referee: {_disposition_line(event, 'referee_decision')}"]
    if event_type == "pwc.revision.appended":
        return _render_revision(event, indent=indent, full=full)
    if event_type == "pwc.exhausted":
        return [f"{indent}exhausted: {event_summary(event)}"]
    if event_type == "node.failed":
        return [f"{indent}node.failed: {event_summary(event)}"]
    if event_type == "pwc.decompose.requested":
        return [f"{indent}decompose requested: {event_summary(event)}"]
    if event_type == "node.decomposed":
        return [f"{indent}node.decomposed: {event_summary(event)}"]
    if full:
        return [f"{indent}{event_type}: {event_summary(event)}"]
    return []


def _render_revision(event: TraceEvent, *, indent: str, full: bool) -> list[str]:
    data = dict_value(event.data.get("data"))
    revision = dict_value(data.get("revision_request"))
    rationale = str_value(revision.get("rationale")) or str_value(event.data.get("summary")) or ""
    revision_items = list_value(revision.get("items"))
    item_count = len(revision_items)
    lines = [
        f"{indent}revision appended: items={item_count} {fit(one_line(rationale), 120)}".rstrip()
    ]
    if full:
        for index, item in enumerate(revision_items, start=1):
            item_data = dict_value(item)
            if not item_data:
                continue
            change = (
                str_value(item_data.get("required_change"))
                or str_value(item_data.get("rationale"))
                or ""
            )
            criterion = str_value(item_data.get("criterion_id"))
            prefix = f"{index}."
            if criterion:
                prefix = f"{index}. {criterion}:"
            lines.append(f"{indent}  {prefix} {one_line(change)}")
    return lines


def _disposition_line(event: TraceEvent, model_key: str) -> str:
    data = dict_value(event.data.get("data"))
    model = dict_value(data.get(model_key))
    disposition = (
        str_value(model.get("disposition")) or str_value(event.data.get("status")) or "unknown"
    )
    rationale = str_value(model.get("rationale")) or str_value(event.data.get("summary")) or ""
    return f"{disposition} {fit(one_line(rationale), 120)}".rstrip()


# ---------------------------------------------------------------------------
# Shared domain helpers (imported by trace_viewer for HTML rendering too)
# ---------------------------------------------------------------------------


def events_by_node(evts: list[TraceEvent]) -> dict[str, list[TraceEvent]]:
    """Group events by node_id, preserving per-node order."""
    grouped: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in evts:
        node_id = event.data.get("node_id")
        if isinstance(node_id, str) and node_id:
            grouped[node_id].append(event)
    return dict(grouped)


def final_status(evts: list[TraceEvent]) -> str:
    """Return the terminal status string for a node's event list."""
    failed = last_event(evts, {"node.failed", "pwc.exhausted"})
    if failed is not None:
        return str_value(failed.data.get("status")) or "failed"
    referee = last_event(evts, {"referee.decision.parsed"})
    if referee is not None:
        return str_value(referee.data.get("status")) or "unknown"
    critic = last_event(evts, {"critic.finding.parsed"})
    if critic is not None:
        return str_value(critic.data.get("status")) or "unknown"
    producer = last_event(evts, {"producer.response.parsed"})
    if producer is not None:
        return str_value(producer.data.get("status")) or "unknown"
    return "unknown"


def last_event(evts: list[TraceEvent], event_types: set[str]) -> TraceEvent | None:
    """Return the last event whose event_type is in event_types, or None."""
    for event in reversed(evts):
        if event.data.get("event_type") in event_types:
            return event
    return None


def last_str(evts: list[TraceEvent], key: str) -> str | None:
    """Return the last non-empty string value for key across events, or None."""
    for event in reversed(evts):
        value = event.data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def event_sort_key(evts: list[TraceEvent]) -> tuple[str, int]:
    """Return (timestamp, line_number) of the first event for stable ordering."""
    first = evts[0] if evts else None
    timestamp = str_value(first.data.get("timestamp")) if first is not None else ""
    line = first.line_number if first is not None else 0
    return (timestamp or "", line)


def event_summary(event: TraceEvent | None) -> str:
    """Return a short one-line summary of status, summary, and error fields."""
    if event is None:
        return ""
    status = str_value(event.data.get("status"))
    summary = str_value(event.data.get("summary"))
    data = dict_value(event.data.get("data"))
    error = str_value(data.get("error"))
    bits = [bit for bit in [status, summary, error] if bit]
    return fit(one_line(" | ".join(bits)), 140)


def interesting_events(evts: list[TraceEvent]) -> list[TraceEvent]:
    """Return only the events worth rendering in timelines and HTML cards."""
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
    return [event for event in evts if event.data.get("event_type") in keep]


def attempt_numbers(evts: list[TraceEvent]) -> list[int]:
    """Return the sorted distinct attempt numbers present in the event list."""
    return sorted(
        {
            attempt
            for event in evts
            if isinstance((attempt := event.data.get("attempt_number")), int)
        }
    )


def attempt_groups(evts: list[TraceEvent]) -> list[tuple[int, list[TraceEvent]]]:
    """Return events partitioned by attempt number in ascending attempt order."""
    by_attempt: dict[int, list[TraceEvent]] = defaultdict(list)
    for event in evts:
        attempt = event.data.get("attempt_number")
        if isinstance(attempt, int):
            by_attempt[attempt].append(event)
    return [(attempt, by_attempt[attempt]) for attempt in sorted(by_attempt)]


def work_output_text(output: dict[str, Any]) -> str:
    """Return a compact summary of a WorkOutput dict (files, deps, base version)."""
    pieces: list[str] = []
    for key, label in [
        ("file_paths", "files"),
        ("dependencies", "deps"),
    ]:
        value = output.get(key)
        values = list_value(value)
        if values:
            pieces.append(f"{label}={len(values)}")
    base = str_value(output.get("base_version"))
    if base:
        pieces.append(f"base={base}")
    return ",".join(pieces) if pieces else "none"


# ---------------------------------------------------------------------------
# Pure data utilities (shared)
# ---------------------------------------------------------------------------


def dict_value(value: Any) -> dict[str, Any]:
    """Return value as a dict if it is one, otherwise an empty dict."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    """Return value as a list if it is one, otherwise an empty list."""
    return cast(list[Any], value) if isinstance(value, list) else []


def str_value(value: Any) -> str | None:
    """Return value if it is a non-empty string, otherwise None."""
    return value if isinstance(value, str) and value else None


def one_line(value: str) -> str:
    """Collapse all whitespace in value to single spaces."""
    return " ".join(value.split())


def fit(value: str, width: int) -> str:
    """Truncate value to width characters, appending '...' if needed."""
    text = one_line(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."
