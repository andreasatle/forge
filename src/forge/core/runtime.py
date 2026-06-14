"""ForgeRuntime — composition root for a Forge run."""

from __future__ import annotations

from dataclasses import dataclass
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
from forge.core.runner import Runner, make_plan_handler, make_work_handler
from forge.core.scheduler import Scheduler, SchedulerCallbacks
from forge.core.state_service import StateService
from forge.core.telemetry import JsonlTelemetrySink, TelemetrySink
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.factory import make_provider

_ADAPTERS_DIR = Path(__file__).parent.parent.parent.parent / "adapters"
_LANGUAGES_DIR = Path(__file__).parent.parent.parent.parent / "languages"


@dataclass
class StartResult:
    """Outcome of a completed Forge run."""

    final_state: SchedulerState
    save_path: Path


def format_failed_node(node: DAGNode) -> str:
    """Format a failed DAGNode for display, including status and failure details."""
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


class ForgeRuntime:
    """Assembles and executes a Forge run from typed config."""

    def __init__(self, config: ForgeConfig, *, verbose: bool = False) -> None:
        self._config = config
        self._verbose = verbose

    async def start(self) -> StartResult:
        """Assemble all components and run the scheduler to completion."""
        config = self._config

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

        state = initial_state or SchedulerState(
            northstar=northstar, max_concurrency=config.concurrency
        )

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
            on_node_failed=lambda node: print(format_failed_node(node)),
            on_idle=lambda s: print(f"~ idle: {len(s.dag)} nodes in DAG"),
        )

        final = await Scheduler(
            runner=runner,
            callbacks=callbacks,
            telemetry_sink=telemetry_sink,
            run_id=run_id,
        ).run(state, global_planner)

        save_path = save_run(final, workspace)
        print(f"run saved: {save_path}")

        completed = sum(1 for n in final.dag.values() if n.node_state.value == "integrated")
        failed = sum(1 for n in final.dag.values() if n.node_state.value == "failed")
        cancelled = sum(1 for n in final.dag.values() if n.node_state.value == "cancelled")
        print(f"\nDone — integrated: {completed}, failed: {failed}, cancelled: {cancelled}")

        if self._verbose:
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

        return StartResult(final_state=final, save_path=save_path)
