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
    .attempt-card > summary {{
      cursor: pointer; list-style: none;
      font-size: 14px; font-weight: 700; color: var(--muted); text-transform: uppercase;
      padding-bottom: 4px;
    }}
    .attempt-card > summary::-webkit-details-marker {{ display: none; }}
    .attempt-card[open] > summary {{ padding-bottom: 10px; border-bottom: 1px solid var(--line); margin-bottom: 10px; }}
    .revision-block {{
      border-left: 3px solid var(--warn); background: #fffbeb;
      border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 2px 0;
    }}
    .revision-block-header {{ font-weight: 700; color: var(--warn); font-size: 13px; margin-bottom: 4px; }}
    .revision-summary {{ margin: 4px 0; font-size: 13px; }}
    .revision-criteria {{ margin: 4px 0; color: var(--muted); font-size: 12px; }}
    .loop-diag {{
      border-left: 3px solid var(--bad); background: #fff1f0;
      border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 8px 0;
    }}
    .loop-diag-header {{ font-weight: 700; color: var(--bad); font-size: 13px; display: block; margin-bottom: 6px; }}
    .loop-diag pre {{ margin: 0; }}
    .plan-tasks {{ margin: 6px 0 0; padding-left: 0; list-style: none; }}
    .plan-task {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; }}
    .task-meta {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .task-deps {{ margin-left: 8px; }}
    .event {{ border-top: 1px solid var(--line); padding-top: 10px; margin-top: 10px; }}
    .event:first-of-type {{ border-top: 0; padding-top: 0; margin-top: 0; }}
    .event-title {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 6px; }}
    .event-body {{ margin: 0; overflow-wrap: anywhere; }}
    .contract-section {{ margin-bottom: 14px; border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px; background: var(--bg); }}
    .contract-section > summary {{ cursor: pointer; color: var(--accent); font-weight: 600; }}
    .contract-body dl {{ margin: 8px 0 0; display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; }}
    .contract-body dt {{ color: var(--muted); font-size: 12px; }}
    .contract-body dd {{ margin: 0; overflow-wrap: anywhere; }}
    .contract-body ul {{ margin: 0; padding-left: 18px; }}
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
    .node-objective {{ font-size: 14px; font-weight: 600; color: var(--text); margin: 8px 0 0; overflow-wrap: anywhere; }}
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
        objective = self._card_objective(evts)
        objective_html = (
            f'<p class="node-objective">{self._e(fit(objective, 120))}</p>' if objective else ""
        )
        return f"""<a class="node-card" href="#node-{self._e(self._anchor_id(node_id))}">
  <div class="node-top">
    <h3>{self._e(node_id[:8])}</h3>
    <span class="pill {self._e(self._status_class(status))}">{self._e(status)}</span>
  </div>
  <div class="meta-grid">
    <div class="meta-item"><span>agent</span><strong>{self._e(agent_type)}</strong></div>
    <div class="meta-item"><span>attempts</span><strong>{len(attempts)}</strong></div>
  </div>
  {objective_html}<p class="summary">{self._e(summary)}</p>
</a>"""

    def _node_detail(self, node_id: str, evts: list[TraceEvent]) -> str:
        groups = list(attempt_groups(evts))
        final_attempt_number = groups[-1][0] if groups else None
        attempt_parts: list[str] = []
        for number, grouped in groups:
            is_final = number == final_attempt_number
            attempt_parts.append(self._attempt_card(number, grouped, is_final=is_final))
            if not is_final:
                revision_evt = last_event(grouped, {"pwc.revision.appended"})
                if revision_evt:
                    attempt_parts.append(self._revision_block(revision_evt, source_attempt=number))

        no_attempt_events = interesting_events(
            [event for event in evts if not isinstance(event.data.get("attempt_number"), int)]
        )
        if no_attempt_events:
            attempt_parts.append(self._attempt_card(None, no_attempt_events))

        attempt_cards = (
            "\n".join(attempt_parts)
            if attempt_parts
            else '<p class="empty">No PWC timeline events for this node.</p>'
        )

        contract_html = self._node_contract(evts)
        return f"""<section class="node-detail panel" id="node-{self._e(self._anchor_id(node_id))}">
  <div class="node-heading">
    <h3>{self._e(node_id[:8])}</h3>
    <span class="pill">{self._e(last_str(evts, "agent_type") or "unknown")}</span>
    <a href="#top">top</a>
  </div>
  {contract_html}
  <div class="attempts">
    {attempt_cards}
  </div>
</section>"""

    def _card_objective(self, evts: list[TraceEvent]) -> str | None:
        for event in evts:
            if event.data.get("event_type") == "node.dispatched":
                data = dict_value(event.data.get("data"))
                contract = dict_value(data.get("contract"))
                objective = str_value(contract.get("objective"))
                if objective:
                    return objective
        return None

    def _node_contract(self, evts: list[TraceEvent]) -> str:
        for event in evts:
            if event.data.get("event_type") == "node.dispatched":
                data = dict_value(event.data.get("data"))
                contract = dict_value(data.get("contract"))
                if contract:
                    return self._render_contract(contract)
        return ""

    def _render_contract(self, contract: dict[str, Any]) -> str:
        objective = str_value(contract.get("objective")) or ""
        success = str_value(contract.get("success_condition")) or ""
        artifact = str_value(contract.get("artifact")) or "—"
        adapter = str_value(contract.get("adapter")) or "—"
        criteria = list_value(contract.get("acceptance_criteria"))
        criteria_html = (
            "".join(
                f"<li>{self._e(str_value(dict_value(c).get('text')) or '')}</li>"
                for c in criteria
                if str_value(dict_value(c).get("text"))
            )
            or "<li>None</li>"
        )
        return f"""<details class="contract-section" open>
  <summary>Contract</summary>
  <div class="contract-body">
    <dl>
      <dt>Objective</dt><dd>{self._e(objective)}</dd>
      <dt>Success condition</dt><dd>{self._e(success)}</dd>
      <dt>Artifact</dt><dd>{self._e(artifact)}</dd>
      <dt>Adapter</dt><dd>{self._e(adapter)}</dd>
      <dt>Acceptance criteria</dt><dd><ul>{criteria_html}</ul></dd>
    </dl>
  </div>
</details>"""

    def _attempt_card(
        self, attempt_number: int | None, evts: list[TraceEvent], *, is_final: bool = True
    ) -> str:
        title = f"Attempt {attempt_number}" if attempt_number is not None else "Node Events"
        open_attr = " open" if is_final else ""
        rendered_events = "\n".join(self._event(event) for event in interesting_events(evts))
        if not rendered_events:
            rendered_events = '<p class="empty">No rendered events.</p>'
        return f"""<details class="attempt-card"{open_attr}>
  <summary>{self._e(title)}</summary>
  {rendered_events}
</details>"""

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
            if plan:
                parts.append(self._plan_task_list(plan))
            for diag in list_value(data.get("diagnostics")):
                diag_dict = dict_value(diag)
                if str_value(diag_dict.get("kind")) == "max_iterations":
                    parts.append(self._max_iterations_block(diag_dict))
                else:
                    excerpt = str_value(diag_dict.get("raw_response_excerpt"))
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

    def _revision_block(self, event: TraceEvent, source_attempt: int) -> str:
        """Render a between-attempt summary of a revision request."""
        data = dict_value(event.data.get("data"))
        revision = dict_value(data.get("revision_request"))
        rationale = (
            str_value(revision.get("rationale")) or str_value(event.data.get("summary")) or ""
        )
        items = list_value(revision.get("items"))
        criterion_ids = [
            cid for item in items if (cid := str_value(dict_value(item).get("criterion_id")))
        ]
        cid_html = (
            f'<p class="revision-criteria">Criteria: {self._e(", ".join(criterion_ids))}</p>'
            if criterion_ids
            else ""
        )
        return (
            f'<div class="revision-block">'
            f'<div class="revision-block-header">Revision request after attempt {source_attempt}</div>'
            f'<p class="revision-summary">{self._e(fit(rationale, 200))}</p>'
            f"{cid_html}"
            f"</div>"
        )

    def _plan_task_list(self, plan: dict[str, Any]) -> str:
        """Render planner output as a numbered task list."""
        tasks = list_value(plan.get("tasks"))
        if not tasks:
            return ""
        rows: list[str] = []
        for i, task in enumerate(tasks, start=1):
            task_data = dict_value(task)
            objective = str_value(task_data.get("objective")) or "—"
            artifact = str_value(task_data.get("artifact")) or "—"
            adapter = str_value(task_data.get("adapter")) or "—"
            depends_on = list_value(task_data.get("depends_on"))
            deps_html = (
                f'<span class="task-deps">depends on: {self._e(", ".join(str(d) for d in depends_on))}</span>'
                if depends_on
                else ""
            )
            rows.append(
                f'<li class="plan-task">'
                f"<strong>Task {i}:</strong> {self._e(objective)}"
                f'<div class="task-meta">'
                f'<span class="task-artifact">{self._e(artifact)}</span>'
                f' · <span class="task-adapter">{self._e(adapter)}</span>'
                f"{deps_html}"
                f"</div></li>"
            )
        task_count = len(tasks)
        return (
            f"<details open><summary>Plan tasks ({task_count})</summary>"
            f'<ol class="plan-tasks">{"".join(rows)}</ol>'
            f"</details>"
        )

    def _max_iterations_block(self, diag: dict[str, Any]) -> str:
        """Render a compact loop-diagnostics block for max_iterations failures."""
        message = str_value(diag.get("message")) or ""
        excerpt = str_value(diag.get("raw_response_excerpt")) or ""
        formatted = message
        for key in (
            "ran_tests_and_passed",
            "final_response_only",
            "has_run_tests",
            "mutating_tool_succeeded",
        ):
            formatted = formatted.replace(f" {key}=", f"\n{key}=")
        msg_html = f"<pre>{self._e(formatted)}</pre>" if formatted else ""
        excerpt_html = (
            f"<details><summary>Raw response excerpt</summary><pre>{self._e(excerpt)}</pre></details>"
            if excerpt
            else ""
        )
        return (
            f'<div class="loop-diag">'
            f'<strong class="loop-diag-header">Loop diagnostics (max_iterations)</strong>'
            f"{msg_html}{excerpt_html}"
            f"</div>"
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
