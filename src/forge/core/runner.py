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
)
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
    max_tool_iterations: int = 25,
) -> Handler:
    """Return a handler that delegates work requests to work_agent."""

    async def work_handler(request: AgentRequest) -> AgentResponse:
        return await work_agent(request, registry, workspace, language_registry, provider, max_tool_iterations=max_tool_iterations)

    return work_handler


def make_plan_handler(
    registry: AdapterRegistry,
    artifact_names: list[str],
    artifact_languages: dict[str, str],
    provider: LLMProvider,
    max_retries: int = 3,
) -> Handler:
    """Return a handler that delegates user-source plan requests to plan_agent."""

    async def plan_handler(request: AgentRequest) -> AgentResponse:
        if request.source == RequestSource.PLANNER:
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])
        return await plan_agent(request, artifact_names, artifact_languages, provider, max_retries)

    return plan_handler
