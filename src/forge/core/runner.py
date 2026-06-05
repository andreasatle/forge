from collections.abc import Awaitable, Callable

from forge.core.models import (
    AdapterType,
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


async def stub_work_handler(request: AgentRequest) -> AgentResponse:
    adapter = request.spec.adapter_type.value if isinstance(request.spec, WorkSpec) else "unknown"
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta={"result": f"stub result for adapter: {adapter}"},
    )


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
            adapter_type=AdapterType.CODING,
        ),
    )
    b = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task B",
            success_condition="B done",
            adapter_type=AdapterType.CODING,
        ),
        dependencies=frozenset({a.id}),
    )
    c = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task C",
            success_condition="C done",
            adapter_type=AdapterType.CODING,
        ),
        dependencies=frozenset({b.id}),
    )
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        follow_up=[c, b, a],
    )
