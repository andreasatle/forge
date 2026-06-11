"""Runner that routes agent requests to registered runtime handlers."""

from collections.abc import Awaitable, Callable

from forge.adapters.registry import AdapterRegistry
from forge.agents.planner import plan_agent
from forge.agents.worker import work_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    RequestSource,
    ResponseStatus,
    StateView,
    WorkSpec,
)
from forge.core.state_service import StateService
from forge.core.telemetry import TelemetrySink
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.providers import LLMProvider

Handler = Callable[[AgentRequest], Awaitable[AgentResponse]]


class Runner:
    """Dispatcher that routes each AgentRequest to its registered async handler."""

    def __init__(self) -> None:
        self._handlers: dict[AgentType, Handler] = {}

    def register(self, agent_type: AgentType, handler: Handler) -> None:
        """Associate handler with agent_type, replacing any previous registration."""
        self._handlers[agent_type] = handler

    async def __call__(self, request: AgentRequest) -> AgentResponse:
        handler = self._handlers.get(request.agent_type)
        if handler is None:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                error=f"no handler registered for: {request.agent_type.value}",
            )
        return await handler(request)


def make_work_handler(
    registry: AdapterRegistry,
    workspace: Workspace,
    language_registry: LanguageRegistry,
    provider: LLMProvider,
    state_services: dict[str, StateService] | None = None,
    max_retries: int = 3,
    max_tool_iterations: int = 25,
    critic_provider: LLMProvider | None = None,
    referee_provider: LLMProvider | None = None,
    telemetry_sink: TelemetrySink | None = None,
    max_attempts: int = 3,
) -> Handler:
    """Return a handler that delegates work requests to work_agent."""

    async def work_handler(request: AgentRequest) -> AgentResponse:
        spec = request.spec
        if isinstance(spec, WorkSpec):
            ss = (state_services or {}).get(spec.artifact)
            if ss is not None:
                state_view = ss.build_state_view()
            else:
                plugin = language_registry.get(spec.language) if spec.language else None
                state_view = StateService(workspace, spec.artifact, plugin).build_state_view()
        else:
            state_view = StateView(artifact_name="", language=None, files=[], dependencies=[])
        return await work_agent(
            request,
            registry,
            workspace,
            language_registry,
            provider,
            state_view,
            max_retries=max_retries,
            max_tool_iterations=max_tool_iterations,
            critic_provider=critic_provider,
            referee_provider=referee_provider,
            telemetry_sink=telemetry_sink,
            max_attempts=max_attempts,
            integration_revision=request.integration_revision,
        )

    return work_handler


def make_plan_handler(
    registry: AdapterRegistry,
    artifact_names: list[str],
    artifact_languages: dict[str, str],
    provider: LLMProvider,
    max_retries: int = 3,
    critic_provider: LLMProvider | None = None,
    referee_provider: LLMProvider | None = None,
    artifact_types: dict[str, str] | None = None,
    artifact_descriptions: dict[str, str] | None = None,
    artifact_language_guidance: dict[str, str] | None = None,
    telemetry_sink: TelemetrySink | None = None,
    max_attempts: int = 3,
) -> Handler:
    """Return a handler that delegates user-source plan requests to plan_agent."""

    async def plan_handler(request: AgentRequest) -> AgentResponse:
        if request.source == RequestSource.PLANNER:
            return AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[]
            )
        return await plan_agent(
            request,
            artifact_names,
            artifact_languages,
            provider,
            max_retries,
            critic_provider=critic_provider,
            referee_provider=referee_provider,
            registry=registry,
            artifact_types=artifact_types,
            artifact_descriptions=artifact_descriptions,
            artifact_language_guidance=artifact_language_guidance,
            telemetry_sink=telemetry_sink,
            max_attempts=max_attempts,
        )

    return plan_handler
