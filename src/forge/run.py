"""CLI entry point for the forge command — start and reset subcommands."""

import argparse
import asyncio
from pathlib import Path

from forge.adapters.registry import AdapterRegistry
from forge.core.config import ForgeConfig
from forge.core.models import (
    AgentRequest,
    AgentType,
    PlanSpec,
    Priority,
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
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.factory import make_provider

_ADAPTERS_DIR = Path(__file__).parent.parent.parent / "adapters"
_LANGUAGES_DIR = Path(__file__).parent.parent.parent / "languages"


def main() -> None:
    """Parse CLI arguments and dispatch to start or reset."""
    parser = argparse.ArgumentParser(prog="forge")
    parser.add_argument("command", choices=["start", "reset"])
    parser.add_argument("config", type=Path)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    config = ForgeConfig.load(args.config)

    if args.command == "reset":
        _reset(config)
    else:
        asyncio.run(_start(config, verbose=args.verbose))


def _reset(config: ForgeConfig) -> None:
    workspace = Workspace(config.workspace)
    workspace.init()
    workspace.reset([a.name for a in config.artifacts])
    print(f"workspace reset: {config.workspace}")


async def _start(config: ForgeConfig, *, verbose: bool = False) -> None:
    artifact_names = [a.name for a in config.artifacts]
    artifact_languages = {a.name: a.language for a in config.artifacts if a.language}

    language_registry = LanguageRegistry()
    language_registry.load(_LANGUAGES_DIR)
    print(f"languages: {language_registry.names()}")

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

    registry = AdapterRegistry()
    registry.load(_ADAPTERS_DIR)
    print(f"adapters: {registry.names()}")

    planner_provider = make_provider(config.models.planner, config.max_tokens)
    worker_provider = make_provider(config.models.worker, config.max_tokens)

    runner = Runner()
    runner.register(AgentType.PLAN, make_plan_handler(registry, artifact_names, artifact_languages, planner_provider, config.max_retries))
    runner.register(AgentType.WORK, make_work_handler(registry, workspace, language_registry, worker_provider, max_tool_iterations=config.max_tool_iterations))

    state = initial_state or SchedulerState(northstar=northstar, max_concurrency=config.concurrency)

    global_planner = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar=northstar),
        priority=Priority.HIGH,
    )

    callbacks = SchedulerCallbacks(
        on_node_dispatched=lambda node: print(
            f"→ dispatched: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_node_completed=lambda node: print(
            f"✓ completed: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_node_failed=lambda node: print(
            f"✗ failed: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_idle=lambda state: print(f"~ idle: {len(state.dag)} nodes in DAG"),
    )

    final = await Scheduler(runner=runner, state_services=state_services, callbacks=callbacks).run(state, global_planner)

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
            delta = node.response.delta if node.response else None
            print(f"  {short_id}  {agent_type:<10}  {node_state:<10}  deps={n_deps}  delta={delta}")
