"""CLI entry point for the forge command."""

import argparse
import asyncio
from pathlib import Path

from forge.core.config import ForgeConfig
from forge.core.runtime import ForgeRuntime
from forge.core.runtime import format_failed_node as _format_failed_node
from forge.core.trace_viewer import (
    TraceViewerError,
    render_latest_trace,
    render_run_trace,
    render_trace_list,
    resolve_run_dir,
    write_latest_trace_html,
    write_run_trace_html,
)
from forge.core.workspace import Workspace

__all__ = ["_format_failed_node"]


def main() -> None:
    """Parse CLI arguments and dispatch to commands."""
    parser = argparse.ArgumentParser(prog="forge")
    parser.add_argument("command", choices=["start", "reset", "trace"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--node")
    args = parser.parse_args()

    if args.command == "trace":
        _trace(args.args, workspace=args.workspace, node=args.node)
        return

    if not args.args:
        parser.error(f"{args.command} requires a config path")
    if len(args.args) > 1:
        parser.error(f"{args.command} accepts exactly one config path")
    if args.workspace is not None or args.node is not None:
        parser.error("--workspace and --node are only valid for trace")

    config = ForgeConfig.load(Path(args.args[0]))

    if args.command == "reset":
        _reset(config)
    else:
        asyncio.run(_start(config, verbose=args.verbose))


def _trace(args: list[str], *, workspace: Path | None, node: str | None) -> None:
    if not args:
        raise SystemExit("trace requires one of: list, latest, <run_id>")

    trace_workspace = workspace or _default_trace_workspace()
    try:
        target = args[0]
        if target == "html":
            if node is not None:
                raise TraceViewerError("--node is not valid with trace html")
            if len(args) != 2:
                raise TraceViewerError("trace html requires one of: latest, <run_id>")
            html_target = args[1]
            if html_target == "latest":
                output_path = write_latest_trace_html(trace_workspace)
            else:
                output_path = write_run_trace_html(resolve_run_dir(trace_workspace, html_target))
            print(output_path)
            return
        if len(args) != 1:
            raise TraceViewerError("trace requires one of: list, latest, <run_id>")
        if target == "list":
            if node is not None:
                raise TraceViewerError("--node is only valid with a run id")
            output = render_trace_list(trace_workspace)
        elif target == "latest":
            if node is not None:
                raise TraceViewerError("--node is only valid with a run id")
            output = render_latest_trace(trace_workspace)
        else:
            output = render_run_trace(
                resolve_run_dir(trace_workspace, target),
                node_prefix=node,
            )
    except TraceViewerError as e:
        raise SystemExit(str(e)) from e
    print(output)


def _default_trace_workspace() -> Path:
    config_path = Path("forge.yaml")
    if config_path.is_file():
        try:
            return ForgeConfig.load(config_path).workspace
        except Exception:
            pass
    return Path("workspaces")


def _reset(config: ForgeConfig) -> None:
    workspace = Workspace(config.workspace)
    workspace.init()
    workspace.reset([a.name for a in config.artifacts])
    print(f"workspace reset: {config.workspace}")


async def _start(config: ForgeConfig, *, verbose: bool = False) -> None:
    await ForgeRuntime(config, verbose=verbose).start()
