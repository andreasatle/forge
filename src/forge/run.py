"""CLI entry point for the forge command."""

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from forge.adapters.registry import AdapterRegistry
from forge.core.config import ForgeConfig
from forge.core.models import (
    AgentRequest,
    AgentType,
    DAGNode,
    PlanSpec,
    RequestSource,
    SchedulerState,
)
from forge.core.persistence import load_run, save_run
from forge.core.runner import (
    Runner,
    make_plan_handler,
    make_work_handler,
)
from forge.core.scheduler import Scheduler, SchedulerCallbacks
from forge.core.state_service import StateService
from forge.core.telemetry import JsonlTelemetrySink, TelemetrySink
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
from forge.languages.registry import LanguageRegistry
from forge.llm.factory import make_provider

_ADAPTERS_DIR = Path(__file__).parent.parent.parent / "adapters"
_LANGUAGES_DIR = Path(__file__).parent.parent.parent / "languages"


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


def _format_failed_node(node: DAGNode) -> str:
    lines = [f"✗ failed: {node.request.agent_type.value} ({node.request.id})"]
    response = node.response
    if response is None:
        return "\n".join(lines)

    lines.append(f"  status: {response.status.value}")
    if response.failure_kind is not None:
        lines.append(f"  failure_kind: {response.failure_kind.value}")
    if response.error:
        lines.append(f"  error: {response.error}")
    return "\n".join(lines)


async def _start(config: ForgeConfig, *, verbose: bool = False) -> None:
    artifact_names = [a.name for a in config.artifacts]
    artifact_languages = {a.name: a.language for a in config.artifacts if a.language}
    artifact_types = {a.name: a.type for a in config.artifacts}
    artifact_descriptions = {a.name: a.description for a in config.artifacts if a.description}

    language_registry = LanguageRegistry()
    language_registry.load(_LANGUAGES_DIR)
    print(f"languages: {language_registry.names()}")
    artifact_language_guidance = {
        artifact.name: language_registry.get(artifact.language).prompt_supplement
        for artifact in config.artifacts
        if artifact.language
    }

    workspace = Workspace(config.workspace)
    workspace.init()

    state_services: dict[str, StateService] = {}
    for artifact in config.artifacts:
        plugin = language_registry.get(artifact.language) if artifact.language else None
        workspace.init_artifact(artifact.name, plugin)
        state_services[artifact.name] = StateService(workspace, artifact.name, plugin)

    if workspace.state_path().exists():
        print(f"resuming: {workspace.path}")
        initial_state: SchedulerState | None = load_run(workspace)
        northstar = initial_state.northstar
    else:
        initial_state = None
        northstar = config.northstar

    run_id = uuid4()
    telemetry_sink: TelemetrySink | None
    try:
        telemetry_sink = JsonlTelemetrySink(
            workspace.telemetry_dir(),
            run_id,
            metadata={"workspace": str(workspace.path), "northstar": northstar},
        )
    except Exception as e:
        print(f"telemetry disabled: {type(e).__name__}: {e}")
        telemetry_sink = None

    registry = AdapterRegistry()
    registry.load(_ADAPTERS_DIR)
    print(f"adapters: {registry.names()}")

    planner_provider = make_provider(config.models.planner.producer, config.max_tokens)
    planner_critic_provider = (
        make_provider(config.models.planner.critic, config.max_tokens)
        if config.models.planner.critic
        else None
    )
    planner_referee_provider = (
        make_provider(config.models.planner.referee, config.max_tokens)
        if config.models.planner.referee
        else None
    )
    worker_provider = make_provider(config.models.worker.producer, config.max_tokens)
    worker_critic_provider = (
        make_provider(config.models.worker.critic, config.max_tokens)
        if config.models.worker.critic
        else None
    )
    worker_referee_provider = (
        make_provider(config.models.worker.referee, config.max_tokens)
        if config.models.worker.referee
        else None
    )

    runner = Runner()
    runner.register(
        AgentType.PLAN,
        make_plan_handler(
            registry,
            artifact_names,
            artifact_languages,
            planner_provider,
            config.max_retries,
            critic_provider=planner_critic_provider,
            referee_provider=planner_referee_provider,
            artifact_types=artifact_types,
            artifact_descriptions=artifact_descriptions,
            artifact_language_guidance=artifact_language_guidance,
            telemetry_sink=telemetry_sink,
            max_attempts=config.models.planner.max_attempts,
        ),
    )
    runner.register(
        AgentType.WORK,
        make_work_handler(
            registry,
            workspace,
            language_registry,
            worker_provider,
            state_services=state_services,
            max_retries=config.max_retries,
            max_tool_iterations=config.max_tool_iterations,
            critic_provider=worker_critic_provider,
            referee_provider=worker_referee_provider,
            telemetry_sink=telemetry_sink,
            max_attempts=config.models.worker.max_attempts,
        ),
    )

    state = initial_state or SchedulerState(northstar=northstar, max_concurrency=config.concurrency)

    global_planner = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar=northstar),
    )

    callbacks = SchedulerCallbacks(
        on_node_dispatched=lambda node: print(
            f"→ dispatched: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_node_completed=lambda node: print(
            f"✓ integrated: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_node_failed=lambda node: print(_format_failed_node(node)),
        on_idle=lambda state: print(f"~ idle: {len(state.dag)} nodes in DAG"),
    )

    final = await Scheduler(
        runner=runner,
        state_services=state_services,
        callbacks=callbacks,
        telemetry_sink=telemetry_sink,
        run_id=run_id,
    ).run(state, global_planner)

    path = save_run(final, workspace)
    print(f"run saved: {path}")

    completed = sum(1 for n in final.dag.values() if n.node_state.value == "integrated")
    failed = sum(1 for n in final.dag.values() if n.node_state.value == "failed")
    cancelled = sum(1 for n in final.dag.values() if n.node_state.value == "cancelled")
    print(f"\nDone — integrated: {completed}, failed: {failed}, cancelled: {cancelled}")

    if verbose:
        print("\nDAG summary:")
        for node in sorted(final.dag.values(), key=lambda n: len(n.request.dependencies)):
            short_id = str(node.request.id)[:8]
            agent_type = node.request.agent_type.value
            node_state = node.node_state.value
            n_deps = len(node.request.dependencies)
            output = node.response.output if node.response else None
            print(
                f"  {short_id}  {agent_type:<10}  {node_state:<10}  deps={n_deps}  output={output}"
            )
