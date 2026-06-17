"""ForgeRuntime — composition root for a Forge run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from forge.adapters.registry import AdapterRegistry
from forge.core.config import ForgeConfig
from forge.core.models import (
    AgentRequest,
    AgentType,
    DAGNode,
    NodeState,
    PlanSpec,
    RequestSource,
    SchedulerState,
)
from forge.core.persistence import load_run, save_run
from forge.core.profile_assignment import (
    ComplexityProfileAssigner,
    DefaultProfileAssigner,
    ProfileAssigner,
)
from forge.core.runner import (
    Handler,
    Runner,
    make_plan_handler,
    make_profile_dispatch_handler,
    make_work_handler,
)
from forge.core.scheduler import Scheduler, SchedulerCallbacks
from forge.core.state_service import StateService
from forge.core.task_complexity import LLMTaskComplexityClassifier
from forge.core.telemetry import JsonlTelemetrySink, TelemetrySink
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.factory import make_provider

logger = logging.getLogger(__name__)

_ADAPTERS_DIR = Path(__file__).parent.parent.parent.parent / "adapters"
_ROLES_DIR = Path(__file__).parent.parent.parent.parent / "roles"
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
        logger.info("languages: %s", language_registry.names())
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
            logger.info("resuming: %s", workspace.path)
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
            logger.warning("telemetry disabled: %s: %s", type(e).__name__, e)
            telemetry_sink = None

        registry = AdapterRegistry()
        registry.load(_ADAPTERS_DIR)
        registry.load(_ROLES_DIR)
        logger.info("adapters: %s", registry.names())

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
        all_worker_profiles = {
            "default": config.models.worker,
            **config.models.worker_profiles,
        }
        profile_handlers: dict[str, Handler] = {}
        for _profile_name, pwc in all_worker_profiles.items():
            p_producer = make_provider(pwc.producer, config.max_tokens)
            p_critic = make_provider(pwc.critic, config.max_tokens) if pwc.critic else None
            p_referee = make_provider(pwc.referee, config.max_tokens) if pwc.referee else None
            profile_handlers[_profile_name] = make_work_handler(
                registry,
                workspace,
                language_registry,
                p_producer,
                state_services=state_services,
                max_retries=config.max_retries,
                max_tool_iterations=config.max_tool_iterations,
                critic_provider=p_critic,
                referee_provider=p_referee,
                telemetry_sink=telemetry_sink,
                max_attempts=pwc.max_attempts,
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
            make_profile_dispatch_handler(profile_handlers),
        )
        profile_assigner = _make_profile_assigner(config)

        state = initial_state or SchedulerState(
            northstar=northstar, max_concurrency=config.concurrency
        )

        if not state.dag:
            root_node = AgentRequest(
                agent_type=AgentType.PLAN,
                source=RequestSource.USER,
                spec=PlanSpec(northstar=northstar),
            )
            state = state.add_nodes([DAGNode(request=root_node)])

        callbacks = SchedulerCallbacks(
            on_node_dispatched=lambda node: logger.info(
                "→ dispatched: %s (%s)", node.request.agent_type.value, node.request.id
            ),
            on_node_completed=lambda node: logger.info(
                "✓ integrated: %s (%s)", node.request.agent_type.value, node.request.id
            ),
            on_node_failed=lambda node: logger.info("%s", format_failed_node(node)),
            on_idle=lambda s: logger.info("~ idle: %d nodes in DAG", len(s.dag)),
        )

        final = await Scheduler(
            runner=runner,
            callbacks=callbacks,
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            state_services=state_services,
            profile_assigner=profile_assigner,
        ).run(state)

        save_path = save_run(final, workspace)
        logger.info("run saved: %s", save_path)

        completed = sum(1 for n in final.dag.values() if n.node_state == NodeState.INTEGRATED)
        failed = sum(1 for n in final.dag.values() if n.node_state == NodeState.FAILED)
        cancelled = sum(1 for n in final.dag.values() if n.node_state == NodeState.CANCELLED)
        logger.info(
            "Done — integrated: %d, failed: %d, cancelled: %d", completed, failed, cancelled
        )

        if self._verbose:
            logger.info("DAG summary:")
            for node in sorted(final.dag.values(), key=lambda n: len(n.request.dependencies)):
                short_id = str(node.request.id)[:8]
                agent_type = node.request.agent_type.value
                node_state = node.node_state.value
                n_deps = len(node.request.dependencies)
                output = node.response.output if node.response else None
                logger.info(
                    "  %s  %-10s  %-10s  deps=%d  output=%s",
                    short_id,
                    agent_type,
                    node_state,
                    n_deps,
                    output,
                )

        return StartResult(final_state=final, save_path=save_path)


def _make_profile_assigner(config: ForgeConfig) -> ProfileAssigner:
    """Build the scheduler-owned profile assigner from runtime config."""
    classifier_config = config.models.complexity_classifier
    if classifier_config is None:
        return DefaultProfileAssigner()

    classifier_provider = make_provider(classifier_config.model, classifier_config.max_tokens)
    classifier = LLMTaskComplexityClassifier(classifier_provider)
    return ComplexityProfileAssigner(
        classifier=classifier,
        complexity_to_profile=classifier_config.complexity_to_profile,
    )
