"""Tests for the integrator agent — apply, test failure reporting, and response shape."""

from unittest.mock import MagicMock

from forge.agents.integrator import integrate_agent
from forge.core.models import (
    AgentRequest,
    AgentType,
    DeltaState,
    FileWrite,
    RequestSource,
    ResponseStatus,
    RunResult,
    WorkSpec,
)
from forge.core.state_service import StateService


def _integrate_request(artifact: str = "codebase") -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.WORKER,
        spec=WorkSpec(
            objective="integrate workers",
            success_condition="integrated",
            adapter="coding",
            artifact=artifact,
        ),
    )


def _mock_ss(
    passed: bool = True, failures: list[str] | None = None, summary: str = ""
) -> MagicMock:
    ss = MagicMock(spec=StateService)
    ss.run_tests.return_value = RunResult(passed=passed, failures=failures or [], summary=summary)
    return ss


# --- apply ---


async def test_apply_delta_called_with_provided_delta():
    """integrate_agent calls state_service.apply_delta with the exact delta passed in."""
    ss = _mock_ss()
    delta = DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")])
    await integrate_agent(request=_integrate_request(), state_service=ss, delta=delta)
    ss.apply_delta.assert_called_once_with(delta)


async def test_run_tests_called_after_successful_apply():
    """integrate_agent calls state_service.run_tests after apply_delta succeeds."""
    ss = _mock_ss()
    await integrate_agent(request=_integrate_request(), state_service=ss, delta=DeltaState())
    ss.run_tests.assert_called_once()


async def test_delta_files_reflected_in_response():
    """Response delta carries the same new_files as the input delta."""
    ss = _mock_ss()
    delta = DeltaState(new_files=[FileWrite(path="x.py", content="x = 1")])
    response = await integrate_agent(request=_integrate_request(), state_service=ss, delta=delta)
    assert response.delta is not None
    assert response.delta.new_files == delta.new_files


async def test_delta_dependencies_reflected_in_response():
    """Response delta carries the same dependencies as the input delta."""
    ss = _mock_ss()
    delta = DeltaState(dependencies=["requests"])
    response = await integrate_agent(request=_integrate_request(), state_service=ss, delta=delta)
    assert response.delta is not None
    assert response.delta.dependencies == ["requests"]


# --- apply failure ---


async def test_apply_failure_adds_apply_failed_error():
    """When apply_delta raises, the response contains an apply_failed IntegrationError."""
    ss = _mock_ss()
    ss.apply_delta.side_effect = ValueError("cannot apply")
    response = await integrate_agent(
        request=_integrate_request(), state_service=ss, delta=DeltaState()
    )
    assert response.delta is not None
    errors = [e for e in response.delta.errors if e.kind == "apply_failed"]
    assert len(errors) == 1
    assert "cannot apply" in errors[0].description


async def test_run_tests_not_called_when_apply_fails():
    """When apply_delta raises, run_tests is not called."""
    ss = _mock_ss()
    ss.apply_delta.side_effect = ValueError("failed")
    await integrate_agent(request=_integrate_request(), state_service=ss, delta=DeltaState())
    ss.run_tests.assert_not_called()


async def test_returns_failed_when_apply_raises():
    """integrate_agent returns FAILED status when apply_delta raises."""
    ss = _mock_ss()
    ss.apply_delta.side_effect = RuntimeError("boom")
    response = await integrate_agent(
        request=_integrate_request(), state_service=ss, delta=DeltaState()
    )
    assert response.status == ResponseStatus.FAILED


# --- test failure ---


async def test_adds_test_failed_error_when_tests_fail():
    """A failing test run adds IntegrationError(kind='test_failed') to delta.errors."""
    ss = _mock_ss(passed=False, failures=["FAILED tests/test_foo.py::test_x"], summary="1 failed")
    response = await integrate_agent(
        request=_integrate_request(), state_service=ss, delta=DeltaState()
    )
    assert response.delta is not None
    test_errors = [e for e in response.delta.errors if e.kind == "test_failed"]
    assert len(test_errors) == 1
    assert "1 failed" in test_errors[0].description


async def test_returns_failed_when_tests_fail():
    """integrate_agent returns FAILED status when run_tests returns passed=False."""
    ss = _mock_ss(passed=False, failures=["FAILED tests/test_foo.py::test_x"], summary="1 failed")
    response = await integrate_agent(
        request=_integrate_request(), state_service=ss, delta=DeltaState()
    )
    assert response.status == ResponseStatus.FAILED


# --- clean integration ---


async def test_returns_empty_errors_on_clean_integration():
    """No apply errors, tests pass → delta.errors is empty."""
    ss = _mock_ss(passed=True)
    response = await integrate_agent(
        request=_integrate_request(),
        state_service=ss,
        delta=DeltaState(new_files=[FileWrite(path="x.py", content="x = 1")]),
    )
    assert response.status == ResponseStatus.COMPLETED
    assert response.delta is not None
    assert response.delta.errors == []
