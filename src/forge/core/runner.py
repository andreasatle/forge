from collections.abc import Awaitable, Callable

from forge.adapters.registry import AdapterRegistry
from forge.agents.planner import plan_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    RequestSource,
    ResponseStatus,
    WorkSpec,
)

Handler = Callable[[AgentRequest], Awaitable[AgentResponse]]


class Runner:
    def __init__(self) -> None:
        self._handlers: dict[AgentType, Handler] = {}

    def register(self, agent_type: AgentType, handler: Handler) -> None:
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
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta={"plan": "stub plan"},
    )


def make_work_handler(registry: AdapterRegistry) -> Handler:
    async def work_handler(request: AgentRequest) -> AgentResponse:
        adapter = registry.get(request.spec.adapter) if isinstance(request.spec, WorkSpec) else registry.get("coding")  # type: ignore[union-attr]
        print(f"  adapter: {adapter.name} — {adapter.description}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            delta={"result": f"stub result for adapter: {adapter.name}"},
        )

    return work_handler


async def stub_integrate_handler(request: AgentRequest) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta={"integrated": True},
    )


async def scripted_plan_handler(request: AgentRequest) -> AgentResponse:
    if request.source == RequestSource.PLANNER:
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])

    a = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task A",
            success_condition="A done",
            adapter="coding",
        ),
    )
    b = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task B",
            success_condition="B done",
            adapter="coding",
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
        ),
        dependencies=frozenset({b.id}),
    )
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        follow_up=[c, b, a],
    )


def make_plan_handler(registry: AdapterRegistry) -> Handler:
    async def plan_handler(request: AgentRequest) -> AgentResponse:
        if request.source == RequestSource.PLANNER:
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])
        return await plan_agent(request, registry)

    return plan_handler
