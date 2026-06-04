from collections.abc import Awaitable, Callable

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
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
