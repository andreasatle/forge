"""Integrator agent — applies a worker DeltaState to disk and runs tests."""

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    DeltaState,
    IntegrationError,
    ResponseStatus,
)
from forge.core.state_service import StateService


async def integrate_agent(
    request: AgentRequest,
    state_service: StateService,
    delta: DeltaState,
) -> AgentResponse:
    """Apply delta via StateService, run tests, return AgentResponse."""
    errors: list[IntegrationError] = []

    try:
        state_service.apply_delta(delta)
    except Exception as e:
        errors.append(IntegrationError(kind="apply_failed", description=str(e)))
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            delta=delta.model_copy(update={"errors": errors}),
        )

    test_result = state_service.run_tests()
    if not test_result.passed:
        lines = [test_result.summary, *test_result.failures]
        description = "\n".join(line for line in lines if line)
        errors.append(IntegrationError(kind="test_failed", description=description))

    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta=delta.model_copy(update={"errors": errors}),
    )
