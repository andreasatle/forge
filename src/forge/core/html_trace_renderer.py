"""HTML presentation of telemetry traces."""

import html
import json
from pathlib import Path
from typing import Any

from forge.core.text_trace_renderer import (
    attempt_groups,
    attempt_numbers,
    dict_value,
    event_sort_key,
    event_summary,
    events_by_node,
    final_status,
    fit,
    interesting_events,
    last_event,
    last_str,
    list_value,
    str_value,
    work_output_text,
)
from forge.core.trace_repository import RunTrace, TraceEvent


class HtmlTraceRenderer:
    """Renders loaded RunTrace objects as standalone static HTML reports."""

    def render_run(self, run: RunTrace, node_prefix: str | None = None) -> str:
        """Render a RunTrace as standalone static HTML."""
        grouped = events_by_node(run.events)
        ordered_nodes = sorted(grouped.items(), key=lambda item: event_sort_key(item[1]))
        node_overview = "\n".join(self._node_overview(nid, evts) for nid, evts in ordered_nodes)
        if not node_overview:
            node_overview = '<p class="empty">No node telemetry events found.</p>'
        node_details = "\n".join(self._node_detail(nid, evts) for nid, evts in ordered_nodes)
        if not node_details:
            node_details = '<section class="node-detail"><p class="empty">No event details available.</p></section>'

        malformed = ""
        if run.malformed_event_count:
            malformed = f'<p class="warning">Skipped {self._e(str(run.malformed_event_count))} malformed events.</p>'

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forge Trace {run.run_id[:8]}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1c2430;
      --muted: #647184;
      --line: #d8dee8;
      --accent: #0f766e;
      --bad: #b42318;
      --warn: #9a6700;
      --code: #eef2f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
    header, section {{ margin-bottom: 22px; }}
    h1, h2, h3, h4 {{ margin: 0; line-height: 1.2; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; margin-bottom: 12px; }}
    h3 {{ font-size: 17px; }}
    h4 {{ font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .panel, .node-card, .attempt-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .panel {{ padding: 18px; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .meta-item span, .event-label {{ display: block; color: var(--muted); font-size: 12px; }}
    .meta-item strong {{ display: block; overflow-wrap: anywhere; }}
    .overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .node-card {{ display: block; padding: 14px; color: inherit; }}
    .node-card:hover {{ border-color: var(--accent); text-decoration: none; }}
    .node-top {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; background: var(--code); color: var(--muted); }}
    .pill.failed {{ color: var(--bad); background: #fff1f0; }}
    .pill.accept, .pill.completed {{ color: var(--accent); background: #e7f7f4; }}
    .summary {{ color: var(--muted); margin: 10px 0 0; overflow-wrap: anywhere; }}
    .node-detail {{ scroll-margin-top: 20px; }}
    .node-heading {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; }}
    .attempts {{ display: grid; gap: 12px; }}
    .attempt-card {{ padding: 14px; }}
    .event {{ border-top: 1px solid var(--line); padding-top: 10px; margin-top: 10px; }}
    .event:first-of-type {{ border-top: 0; padding-top: 0; margin-top: 0; }}
    .event-title {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 6px; }}
    .event-body {{ margin: 0; overflow-wrap: anywhere; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--accent); }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: var(--code);
      border-radius: 6px;
      padding: 10px;
      margin: 8px 0 0;
      font-size: 12px;
    }}
    ul {{ margin: 8px 0 0; padding-left: 20px; }}
    li {{ margin-bottom: 8px; }}
    .empty {{ color: var(--muted); margin: 0; }}
    .warning {{ color: var(--warn); }}
  </style>
</head>
<body>
<main id="top">
  <header class="panel">
    <h1>Forge Telemetry Trace</h1>
    <div class="meta-grid">
      <div class="meta-item"><span>run_id</span><strong>{self._e(run.run_id)}</strong></div>
      <div class="meta-item"><span>created_at</span><strong>{self._e(run.created_at)}</strong></div>
      <div class="meta-item"><span>event count</span><strong>{len(run.events)}</strong></div>
      <div class="meta-item"><span>northstar</span><strong>{self._e(run.northstar)}</strong></div>
    </div>
    {malformed}
  </header>
  <section>
    <h2>Scheduler / Node Overview</h2>
    <div class="overview-grid">
      {node_overview}
    </div>
  </section>
  <section>
    <h2>Node Details</h2>
    {node_details}
  </section>
</main>
</body>
</html>
"""

    def write_run(
        self,
        run: RunTrace,
        output_path: Path | None = None,
        node_prefix: str | None = None,
    ) -> Path:
        """Write the HTML report; defaults to run_dir/index.html."""
        if output_path is None:
            output_path = run.run_dir / "index.html"
        output_path.write_text(self.render_run(run, node_prefix=node_prefix), encoding="utf-8")
        return output_path

    def _node_overview(self, node_id: str, evts: list[TraceEvent]) -> str:
        agent_type = last_str(evts, "agent_type") or "unknown"
        attempts = attempt_numbers(evts)
        status = final_status(evts)
        last_failure = last_event(evts, {"node.failed", "pwc.exhausted"})
        last_revision = last_event(evts, {"pwc.revision.appended"})
        summary = event_summary(last_failure or last_revision) or "No failure or revision summary."
        return f"""<a class="node-card" href="#node-{self._e(self._anchor_id(node_id))}">
  <div class="node-top">
    <h3>{self._e(node_id[:8])}</h3>
    <span class="pill {self._e(self._status_class(status))}">{self._e(status)}</span>
  </div>
  <div class="meta-grid">
    <div class="meta-item"><span>agent</span><strong>{self._e(agent_type)}</strong></div>
    <div class="meta-item"><span>attempts</span><strong>{len(attempts)}</strong></div>
  </div>
  <p class="summary">{self._e(summary)}</p>
</a>"""

    def _node_detail(self, node_id: str, evts: list[TraceEvent]) -> str:
        attempt_cards = "\n".join(
            self._attempt_card(number, grouped) for number, grouped in attempt_groups(evts)
        )
        no_attempt_events = interesting_events(
            [event for event in evts if not isinstance(event.data.get("attempt_number"), int)]
        )
        if no_attempt_events:
            attempt_cards += "\n" + self._attempt_card(None, no_attempt_events)
        if not attempt_cards:
            attempt_cards = '<p class="empty">No PWC timeline events for this node.</p>'

        return f"""<section class="node-detail panel" id="node-{self._e(self._anchor_id(node_id))}">
  <div class="node-heading">
    <h3>{self._e(node_id[:8])}</h3>
    <span class="pill">{self._e(last_str(evts, "agent_type") or "unknown")}</span>
    <a href="#top">top</a>
  </div>
  <div class="attempts">
    {attempt_cards}
  </div>
</section>"""

    def _attempt_card(self, attempt_number: int | None, evts: list[TraceEvent]) -> str:
        title = f"Attempt {attempt_number}" if attempt_number is not None else "Node Events"
        rendered_events = "\n".join(self._event(event) for event in interesting_events(evts))
        if not rendered_events:
            rendered_events = '<p class="empty">No rendered events.</p>'
        return f"""<article class="attempt-card">
  <h4>{self._e(title)}</h4>
  {rendered_events}
</article>"""

    def _event(self, event: TraceEvent) -> str:
        event_type = str_value(event.data.get("event_type")) or "unknown"
        status = str_value(event.data.get("status")) or "no status"
        body = self._event_body(event)
        return f"""<div class="event">
  <div class="event-title">
    <strong>{self._e(event_type)}</strong>
    <span class="pill {self._e(self._status_class(status))}">{self._e(status)}</span>
  </div>
  {body}
  <details><summary>Event JSON</summary><pre>{self._e(self._json(event.data))}</pre></details>
</div>"""

    def _event_body(self, event: TraceEvent) -> str:
        event_type = str_value(event.data.get("event_type")) or "unknown"
        data = dict_value(event.data.get("data"))
        if event_type == "producer.response.parsed":
            status = (
                str_value(data.get("status")) or str_value(event.data.get("status")) or "unknown"
            )
            output_type = str_value(data.get("output_type")) or "none"
            summary = f"status={status} output_type={output_type}"
            plan = dict_value(data.get("plan"))
            work_out = dict_value(data.get("work_output"))
            if plan:
                summary += f" plan_tasks={plan.get('task_count', 0)}"
            if work_out:
                summary += f" work_output={work_output_text(work_out)}"
            parts = [f'<p class="event-body">{self._e(summary)}</p>']
            for diag in list_value(data.get("diagnostics")):
                excerpt = str_value(dict_value(diag).get("raw_response_excerpt"))
                if excerpt:
                    parts.append(
                        f"<details><summary>raw_response_excerpt</summary>"
                        f"<pre>{self._e(excerpt)}</pre></details>"
                    )
            return "".join(parts)
        if event_type == "critic.finding.parsed":
            return self._disposition(event, "critic_finding")
        if event_type == "referee.decision.parsed":
            return self._disposition(event, "referee_decision")
        if event_type == "pwc.revision.appended":
            return self._revision(event)
        return f'<p class="event-body">{self._e(event_summary(event))}</p>'

    def _disposition(self, event: TraceEvent, model_key: str) -> str:
        data = dict_value(event.data.get("data"))
        model = dict_value(data.get(model_key))
        disposition = (
            str_value(model.get("disposition")) or str_value(event.data.get("status")) or "unknown"
        )
        rationale = str_value(model.get("rationale")) or str_value(event.data.get("summary")) or ""
        return (
            f'<p class="event-body"><strong>{self._e(disposition)}</strong> '
            f"{self._e(fit(rationale, 160))}</p>"
            f"<details><summary>Longer rationale</summary><p>{self._e(rationale)}</p></details>"
        )

    def _revision(self, event: TraceEvent) -> str:
        data = dict_value(event.data.get("data"))
        revision = dict_value(data.get("revision_request"))
        rationale = (
            str_value(revision.get("rationale")) or str_value(event.data.get("summary")) or ""
        )
        items = list_value(revision.get("items"))
        item_list = "\n".join(
            self._revision_item(index, item) for index, item in enumerate(items, start=1)
        )
        if not item_list:
            item_list = "<li>No revision items recorded.</li>"
        return f"""<p class="event-body">Revision appended with {len(items)} item(s).</p>
<details open><summary>Revision items</summary><ul>{item_list}</ul></details>
<details><summary>Longer rationale</summary><p>{self._e(rationale)}</p></details>"""

    def _revision_item(self, index: int, item: Any) -> str:
        item_data = dict_value(item)
        criterion = str_value(item_data.get("criterion_id"))
        required = str_value(item_data.get("required_change")) or ""
        rationale = str_value(item_data.get("rationale")) or ""
        label = f"{index}. {criterion}: " if criterion else f"{index}. "
        return (
            f"<li><strong>{self._e(label)}</strong>{self._e(required)}"
            f"<details><summary>Revision item details</summary><pre>{self._e(self._json(item_data))}</pre>"
            f"<p>{self._e(rationale)}</p></details></li>"
        )

    @staticmethod
    def _e(value: object) -> str:
        return html.escape(str(value), quote=True)

    @staticmethod
    def _anchor_id(value: str) -> str:
        return "".join(char if char.isalnum() else "-" for char in value.lower())

    @staticmethod
    def _status_class(status: str) -> str:
        normalized = status.lower()
        if normalized in {"failed", "reject", "rejected"}:
            return "failed"
        if normalized in {"accept", "completed", "already_done"}:
            return "accept"
        return ""

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, indent=2, sort_keys=True)
