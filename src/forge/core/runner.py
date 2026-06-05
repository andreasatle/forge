"""Runner that routes agent requests to registered handlers, plus built-in handlers."""

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
    WorkSpec,
)
from forge.core.workspace import Workspace

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


async def stub_plan_handler(request: AgentRequest) -> AgentResponse:
    """Return a completed response with a placeholder plan delta, for testing."""
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta={"plan": "stub plan"},
    )


def make_work_handler(registry: AdapterRegistry, workspace: Workspace) -> Handler:
    """Return a handler that delegates work requests to work_agent."""
    async def work_handler(request: AgentRequest) -> AgentResponse:
        return await work_agent(request, registry, workspace)

    return work_handler


async def stub_integrate_handler(request: AgentRequest) -> AgentResponse:
    """Return a completed response with a placeholder integration delta, for testing."""
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta={"integrated": True},
    )


async def scripted_plan_handler(request: AgentRequest) -> AgentResponse:
    """Return a hardcoded A→B→C dependency chain for use in integration tests."""
    if request.source == RequestSource.PLANNER:
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])

    a = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task A",
            success_condition="A done",
            adapter="coding",
            artifact="codebase",
        ),
    )
    b = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task B",
            success_condition="B done",
            adapter="coding",
            artifact="codebase",
        ),
        dependencies=frozenset({a.id}),
    )
    c = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task C",
            success_condition="C done",
            adapter="coding",
            artifact="codebase",
        ),
        dependencies=frozenset({b.id}),
    )
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        follow_up=[c, b, a],
    )


def make_plan_handler(registry: AdapterRegistry, artifact_names: list[str]) -> Handler:
    """Return a handler that delegates user-source plan requests to plan_agent."""
    async def plan_handler(request: AgentRequest) -> AgentResponse:
        if request.source == RequestSource.PLANNER:
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])
        return await plan_agent(request, registry, artifact_names)

    return plan_handler
