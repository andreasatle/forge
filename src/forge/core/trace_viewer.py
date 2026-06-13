"""Read-only telemetry trace rendering for human CLI and HTML inspection."""

from pathlib import Path

from forge.core.html_trace_renderer import HtmlTraceRenderer
from forge.core.text_trace_renderer import TextTraceRenderer
from forge.core.trace_repository import RunTrace, TraceEvent, TraceRepository, TraceViewerError

__all__ = [
    "RunTrace",
    "TraceEvent",
    "TraceRepository",
    "TraceViewerError",
    "render_latest_trace",
    "render_run_trace",
    "render_run_trace_html",
    "render_trace_list",
    "resolve_run_dir",
    "write_latest_trace_html",
    "write_run_trace_html",
]


def render_trace_list(workspace: Path) -> str:
    """Render all telemetry runs newest first."""
    runs = TraceRepository(workspace).list_runs()
    if not runs:
        return f"no telemetry runs found in {workspace / 'telemetry' / 'runs'}"
    return TextTraceRenderer().render_list(runs)


def render_latest_trace(workspace: Path) -> str:
    """Render the newest telemetry run."""
    run = TraceRepository(workspace).latest_run()
    if run is None:
        return f"no telemetry runs found in {workspace / 'telemetry' / 'runs'}"
    return TextTraceRenderer().render_run(run)


def render_run_trace(run_dir: Path, *, node_prefix: str | None = None) -> str:
    """Render a run summary, optionally narrowed to one node."""
    trace = TraceRepository.load_run(run_dir)
    return TextTraceRenderer().render_run(trace, node_prefix=node_prefix)


def write_latest_trace_html(workspace: Path) -> Path:
    """Write an HTML report for the newest telemetry run."""
    run = TraceRepository(workspace).latest_run()
    if run is None:
        raise TraceViewerError(f"no telemetry runs found in {workspace / 'telemetry' / 'runs'}")
    return HtmlTraceRenderer().write_run(run)


def write_run_trace_html(run_dir: Path) -> Path:
    """Write an HTML report for one telemetry run and return the output path."""
    trace = TraceRepository.load_run(run_dir)
    return HtmlTraceRenderer().write_run(trace)


def render_run_trace_html(run_dir: Path) -> str:
    """Render one telemetry run as standalone static HTML."""
    return HtmlTraceRenderer().render_run(TraceRepository.load_run(run_dir))


def resolve_run_dir(workspace: Path, run_id_prefix: str) -> Path:
    """Resolve a run id or unambiguous run id prefix to a telemetry run directory."""
    return TraceRepository(workspace).resolve_run_dir(run_id_prefix)
