"""Integration infrastructure — applies a worker DeltaState to disk and runs tests."""

from uuid import UUID

from forge.core.models import (
    AgentResponse,
    DeltaState,
    FailureKind,
    IntegrationError,
    ResponseStatus,
)
from forge.core.state_service import StateService


async def integrate(
    request_id: UUID,
    state_service: StateService,
    delta: DeltaState,
) -> AgentResponse:
    """Apply delta via StateService, run tests, return AgentResponse."""
    errors: list[IntegrationError] = []

    if delta.base_version != state_service.current_version:
        return AgentResponse(
            request_id=request_id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.STALE_DELTA,
            error=(
                f"Stale delta: based on version {delta.base_version}, "
                f"current state is version {state_service.current_version}"
            ),
        )

    try:
        state_service.apply_delta(delta)
    except Exception as e:
        errors.append(IntegrationError(kind="apply_failed", description=str(e)))
        return AgentResponse(
            request_id=request_id,
            status=ResponseStatus.FAILED,
            delta=delta.model_copy(update={"errors": errors}),
        )

    test_result = state_service.run_tests()
    if not test_result.passed:
        lines = [test_result.summary, *test_result.failures]
        description = "\n".join(line for line in lines if line)
        errors.append(IntegrationError(kind="test_failed", description=description))
        return AgentResponse(
            request_id=request_id,
            status=ResponseStatus.FAILED,
            delta=delta.model_copy(update={"errors": errors}),
        )

    return AgentResponse(
        request_id=request_id,
        status=ResponseStatus.COMPLETED,
        delta=delta.model_copy(update={"errors": errors}),
    )
