"""Base agent runner with shared error handling for all agent types."""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel

from forge.core.models import AgentRequest, AgentResponse, ResponseStatus

S = TypeVar("S", bound=BaseModel)


async def run_agent(
    request: AgentRequest,
    spec_type: type[S],
    build_response: Callable[[S], Awaitable[AgentResponse]],
) -> AgentResponse:
    """Validate request spec type, call build_response, and wrap any exception as FAILED."""
    try:
        if not isinstance(request.spec, spec_type):
            raise TypeError(f"expected {spec_type.__name__}, got {type(request.spec).__name__}")
        return await build_response(request.spec)
    except Exception as e:
        print(f"agent error: {type(e).__name__}: {e}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
        )
